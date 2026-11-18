"""Loopback Qwen TTS adapter matching the current demo's Piper WebSocket API."""
import argparse
import asyncio
import json
import re
import time
from contextlib import suppress
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

ENGINE = 'Qwen3-TTS-12Hz-1.7B-CustomVoice'
RATE = 24000
VOICES = {'zh': 'serena', 'en': 'serena'}
AVAILABLE_VOICES = ('serena', 'vivian', 'uncle_fu', 'ryan', 'aiden', 'ono_anna', 'sohee', 'eric', 'dylan')
MAX_CHARS = 500
MAX_CLIENTS = 4


def chunks(text):
    # Preserve all input characters, with short phrase boundaries for cancellation.
    pieces = re.findall(r'.{1,100}?(?:[。！？!?；;\n]|(?<=[.!?])\s|$)|.{1,100}', text, re.S)
    return [piece.strip() for piece in pieces if piece.strip()]


def validate(data):
    text = data.get('text', '')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Text must not be empty.')
    if len(text) > MAX_CHARS:
        raise ValueError(f'Maximum {MAX_CHARS} characters per request.')
    lang = data.get('lang', 'auto')
    if lang not in ('zh', 'en', 'auto'):
        raise ValueError('lang must be zh, en or auto.')
    inferred = 'zh' if re.search(r'[\u3400-\u9fff]', text) else 'en'
    voice = data.get('voice', VOICES[inferred if lang == 'auto' else lang])
    if not isinstance(voice, str) or voice not in AVAILABLE_VOICES:
        raise ValueError('Unsupported voice.')
    return text.strip(), voice, {'zh': 'Chinese', 'en': 'English', 'auto': 'Auto'}[lang]


def create_app(upstream, model, manifest=None):
    app = web.Application(client_max_size=8192)
    app['lock'] = asyncio.Lock()
    app['clients'] = 0
    app['active'] = 0
    app['waiting'] = 0
    app['manifest'] = manifest or {}

    async def engine_health():
        try:
            async with app['http'].get(upstream + '/health', timeout=ClientTimeout(total=3)) as r:
                return r.status == 200
        except (OSError, asyncio.TimeoutError):
            return False
        except Exception:
            return False

    async def health(request):
        ok = await engine_health()
        return web.json_response({'ok': ok, 'engine': ENGINE, 'clients': app['clients'],
                                  'active': app['active'], 'waiting': app['waiting'],
                                  'max_clients': MAX_CLIENTS, 'sample_rate': RATE,
                                  'streaming': True,
                                  'revision': app['manifest'].get('revision')}, status=200 if ok else 503)

    async def status(request):
        return web.json_response({'tts': {'available': await engine_health(),
                                          'engine': ENGINE, 'languages': ['zh', 'en']},
                                  'asr': {'available': False}})

    async def config(request):
        return web.json_response({'available': await engine_health(), 'engine': ENGINE,
                                  'languages': ['zh', 'en'], 'sample_rate': RATE,
                                  'voices': VOICES, 'available_voices': list(AVAILABLE_VOICES),
                                  'default_voice': 'serena', 'max_text_chars': MAX_CHARS})

    async def speak(ws, data):
        reqid = str(data.get('request_id') or data.get('reqid') or 'speech')
        try:
            text, voice, lang = validate(data)
            app['waiting'] += 1
            try:
                await asyncio.wait_for(app['lock'].acquire(), timeout=45)
            finally:
                app['waiting'] -= 1
            app['active'] += 1
            try:
                await ws.send_json({'type': 'start', 'request_id': reqid, 'sample_rate': RATE,
                                    'format': 'pcm16', 'channels': 1, 'voice': voice})
                total = 0
                begin = time.perf_counter()
                first_audio_ms = None
                for piece in chunks(text):
                    payload = {'model': model, 'input': piece, 'voice': voice, 'language': lang,
                               'stream': True, 'stream_format': 'audio', 'response_format': 'pcm'}
                    # Closing this request on stop/disconnect aborts the upstream stream.
                    async with app['http'].post(upstream + '/v1/audio/speech', json=payload) as r:
                        if r.status != 200:
                            raise RuntimeError(f'TTS engine returned HTTP {r.status}')
                        carry = b''
                        async for packet in r.content.iter_any():
                            carry += packet
                            aligned = len(carry) // 2 * 2
                            if not aligned:
                                continue
                            if first_audio_ms is None:
                                first_audio_ms = (time.perf_counter() - begin) * 1000
                            for offset in range(0, aligned, 16384):
                                await ws.send_bytes(carry[offset:min(offset + 16384, aligned)])
                            total += aligned
                            carry = carry[aligned:]
                        if carry:
                            raise RuntimeError('Incomplete PCM sample from engine')
                if total == 0:
                    raise RuntimeError('TTS engine returned no audio')
                await ws.send_json({'type': 'done', 'request_id': reqid, 'audio_bytes': total,
                                    'first_audio_ms': first_audio_ms,
                                    'elapsed_ms': (time.perf_counter() - begin) * 1000})
            finally:
                app['active'] -= 1
                app['lock'].release()
        except asyncio.CancelledError:
            if not ws.closed:
                with suppress(Exception):
                    await ws.send_json({'type': 'stopped', 'request_id': reqid})
            raise
        except Exception as exc:
            if not ws.closed:
                message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else 'Speech synthesis unavailable.'
                with suppress(Exception):
                    await ws.send_json({'type': 'error', 'request_id': reqid, 'error': message})

    async def cancel(task):
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def websocket(request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8192)
        await ws.prepare(request)
        if app['clients'] >= MAX_CLIENTS:
            await ws.send_json({'type': 'error', 'error': 'TTS busy; please retry.'})
            await ws.close()
            return ws
        app['clients'] += 1
        task = None
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                try:
                    data = json.loads(msg.data)
                    if not isinstance(data, dict):
                        raise ValueError('Expected an object')
                except ValueError:
                    await ws.send_json({'type': 'error', 'error': 'Invalid JSON request.'})
                    continue
                kind = data.get('type', 'speak')
                if kind in ('speak', 'stop'):
                    await cancel(task)
                    task = asyncio.create_task(speak(ws, data)) if kind == 'speak' else None
                elif kind == 'ping':
                    await ws.send_json({'type': 'pong', 'id': data.get('id')})
                else:
                    await ws.send_json({'type': 'error', 'error': 'Unknown command.'})
        finally:
            await cancel(task)
            app['clients'] -= 1
            await ws.close()
        return ws

    async def lifetime(app):
        async with ClientSession(timeout=ClientTimeout(total=180, sock_connect=5, sock_read=90)) as client:
            app['http'] = client
            yield

    app.cleanup_ctx.append(lifetime)
    app.router.add_get('/health', health)
    app.router.add_get('/api/voice/status', status)
    app.router.add_get('/api/tts/config', config)
    app.router.add_get('/api/tts', websocket)
    return app


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', default='http://127.0.0.1:18109')
    p.add_argument('--model', required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--port', type=int, default=18110)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
    web.run_app(create_app(args.upstream.rstrip('/'), args.model, manifest),
                host='127.0.0.1', port=args.port, access_log=None)
