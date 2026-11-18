"""Bilingual streaming ASR, bound to loopback.

Each connection owns its streaming state. Model calls are serialized, not
microphone sessions; silent listeners do not occupy the model. PCM/text stay
in memory. CPU Silero VAD gates speech before any ASR generation.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import time

from aiohttp import web, WSMsgType
import numpy as np

from speech_gate import SileroVad, SpeechGate

RATE = 16000
MAX_SECONDS = 30
MAX_CLIENTS = 2
SILENCE_SECONDS = 1.5
LOG = logging.getLogger('alive-asr')


def create_app(root, model=None):
    metadata = json.loads((root / 'models.json').read_text())['qwen']
    languages = {'auto': None, 'zh': 'Chinese', 'en': 'English'}
    if model is None:
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.LLM(model=metadata['path'], dtype='bfloat16',
                                  gpu_memory_utilization=0.2, max_model_len=4096,
                                  max_num_seqs=1, enforce_eager=True, max_new_tokens=256,
                                  max_inference_batch_size=1, disable_log_stats=True)
        
    app = web.Application()
    app['active'] = 0
    app['lock'] = asyncio.Lock()
    vad = SileroVad()

    async def call(function, *args, **kwargs):
        async with app['lock']:
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task  # Never release the lock while a GPU call still runs.
                raise

    async def health(request):
        return web.json_response({'ok': True, 'engine': 'Qwen3-ASR-1.7B',
                                  'revision': metadata.get('revision'), 'languages': ['auto', 'zh', 'en'], 'auto_language_detection': True,
                                  'active': app['active'], 'max_clients': MAX_CLIENTS,
                                  'streaming_partials': True, 'chunk_sec': 1.0,
                                  'endpoint_detection': True, 'endpoint_silence_sec': SILENCE_SECONDS,
                                  'speech_gate': 'silero-vad', 'speech_gate_start_ms': 192,
                                  'max_recording_sec': MAX_SECONDS, 'max_new_tokens': 256})

    async def websocket(request):
        lang = request.query.get('lang', 'auto')
        if lang not in languages:
            raise web.HTTPBadRequest(text='lang must be auto, zh or en')
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=65536)
        # Reserve capacity before the first await to prevent admission races.
        if app['active'] >= MAX_CLIENTS:
            await ws.prepare(request)
            await ws.send_json({'type': 'error', 'message': 'Speech recognition is busy. Please retry.'})
            await ws.close()
            return ws
        app['active'] += 1
        endpoint = None
        total = 0
        try:
            await ws.prepare(request)
            await ws.send_json({'type': 'status', 'message': 'connected', 'sample_rate': RATE,
                                'max_recording_sec': MAX_SECONDS, 'engine': 'Qwen3-ASR-1.7B',
                                'speech_detection': 'silero-vad'})
            state = None
            continuous = request.query.get('continuous') == '1'
            endpoint = SpeechGate(vad.new_stream(), continuous=continuous)
            last = ''
            deadline = time.monotonic() + MAX_SECONDS + 20

            async def final():
                nonlocal state
                started = time.perf_counter()
                tail = await asyncio.to_thread(endpoint.finish)
                if tail.size:
                    if state is None:
                        await ws.send_json({'type': 'speech_activity', 'active': True})
                        state = await call(model.init_streaming_state, language=languages[lang], chunk_size_sec=1.0)
                    await call(model.streaming_transcribe, tail, state)
                text = ''
                if state is not None and endpoint.heard_voice:
                    await call(model.finish_streaming_transcribe, state)
                    text = state.text.strip()
                await ws.send_json({'type': 'result', 'event': 'IS_FINAL', 'final': True,
                                    'text': text, 'language': getattr(state, 'language', ''),
                                    'speech_gate': endpoint.metrics(),
                                    'flush_ms': (time.perf_counter() - started) * 1000})

            while not ws.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError('Recording timed out. Please retry.')
                msg = await ws.receive(timeout=min(15, remaining))
                if msg.type == WSMsgType.BINARY:
                    if len(msg.data) % 2:
                        raise ValueError('Invalid PCM audio format.')
                    if not msg.data:
                        continue
                    pcm = np.frombuffer(msg.data, dtype='<i2').astype(np.float32) / 32768.0
                    total += len(pcm)
                    if total > MAX_SECONDS * RATE:
                        raise ValueError('A single utterance can be at most 30 seconds.')
                    gated = await asyncio.to_thread(endpoint.feed, pcm)
                    if gated.size == 0:
                        if endpoint.ended and not continuous:
                            await final()
                            break
                        continue
                    if state is None:
                        await ws.send_json({'type': 'speech_activity', 'active': True})
                        state = await call(model.init_streaming_state, language=languages[lang], chunk_size_sec=1.0)
                    await call(model.streaming_transcribe, gated, state)
                    if state.text != last:
                        await ws.send_json({'type': 'result', 'event': 'IS_PARTIAL', 'final': False, 'text': state.text, 'language': state.language})
                        last = state.text
                    if endpoint.ended and not continuous:
                        await final()
                        break
                elif msg.type == WSMsgType.TEXT:
                    command = json.loads(msg.data)
                    if not isinstance(command, dict):
                        raise ValueError('Invalid control message.')
                    if command.get('type') in ('end', 'segment_end'):
                        await final()
                        break
                    if command.get('type') == 'ping':
                        await ws.send_json({'type': 'pong'})
                else:
                    break
        except ConnectionResetError:
            pass  # Client ended the utterance while a model call was in flight.
        except (ValueError, asyncio.TimeoutError) as exc:
            if not ws.closed:
                await ws.send_json({'type': 'error', 'message': str(exc) or 'Speech connection timed out.'})
        except Exception:
            LOG.exception('ASR request failed')
            if ws.prepared and not ws.closed:
                await ws.send_json({'type': 'error', 'message': 'Speech recognition is unavailable. Please retry.'})
        finally:
            if endpoint is not None:
                LOG.info('speech_gate %s', json.dumps(dict(endpoint.metrics(), audio_seconds=round(total / RATE, 3))))
            app['active'] -= 1
            await ws.close()
        return ws

    app.router.add_get('/health', health)
    app.router.add_get('/ws/asr', websocket)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18108)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(args.root), host='127.0.0.1', port=args.port, access_log=None)
