"""Local same-origin ALIVE gateway for video and speech."""
import argparse
import asyncio
import json
import re
import ssl
import time
from contextlib import suppress

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate',
               'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'}


def create_app(web_base, asr_base, tts_base, lease_seconds=45):
    app = web.Application(client_max_size=16 * 1024 * 1024)
    slot = {'owner': None, 'deadline': 0.0, 'releasing': False}
    admission_lock = asyncio.Lock()
    active_sockets = {}

    def session_id(value):
        value = str(value or '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{8,160}', value):
            raise web.HTTPBadRequest(text='Invalid session identifier')
        return value

    def owns(sid):
        return bool(sid and slot['owner'] == sid and not slot['releasing']
                    and time.monotonic() < slot['deadline'])

    def touch(sid):
        if owns(sid):
            slot['deadline'] = time.monotonic() + lease_seconds

    def public_status(sid):
        return {'busy': slot['owner'] is not None, 'mine': owns(sid),
                'releasing': slot['releasing'], 'lease_seconds': lease_seconds}

    async def release(sid):
        # Hold the mutex until all old streams and inference state are cleaned:
        # a new owner must never overlap with an old owner's in-flight request.
        async with admission_lock:
            if slot['owner'] != sid:
                return False
            slot['releasing'] = True
            sockets = list(active_sockets.get(sid, ()))
            await asyncio.gather(*(ws.close() for ws in sockets), return_exceptions=True)
            try:
                async with app['client'].post(web_base + '/api/session/cleanup',
                        json={'session_id': sid, 'reset_adapter': True}, ssl=False,
                        timeout=ClientTimeout(total=15)) as response:
                    if response.status != 200:
                        raise RuntimeError('Session cleanup failed')
                    await response.read()
            except Exception:
                # Fail closed; the expiry loop retries cleanup instead of
                # admitting another user while old model work may still run.
                slot['deadline'] = time.monotonic()
                return False
            slot.update(owner=None, deadline=0.0, releasing=False)
            return True

    async def expiry_loop():
        while True:
            await asyncio.sleep(min(5, lease_seconds / 2))
            if slot['owner'] and time.monotonic() >= slot['deadline']:
                await release(slot['owner'])

    async def admission_status(request):
        return web.json_response(public_status(request.query.get('session_id')), headers={'Cache-Control':'no-store'})

    async def admission_acquire(request):
        data = await request.json()
        sid = session_id(data.get('session_id'))
        async with admission_lock:
            if slot['owner'] and slot['owner'] != sid or slot['releasing']:
                return web.json_response({'error':'ALIVE_BUSY', **public_status(sid)}, status=409)
            slot.update(owner=sid, deadline=time.monotonic()+lease_seconds, releasing=False)
            return web.json_response(public_status(sid))

    async def admission_heartbeat(request):
        sid = session_id((await request.json()).get('session_id'))
        if not owns(sid):
            return web.json_response({'error':'ALIVE_NOT_OWNER', **public_status(sid)}, status=409)
        touch(sid)
        return web.json_response(public_status(sid))

    async def admission_release(request):
        sid = session_id((await request.json()).get('session_id'))
        return web.json_response({'released':await release(sid)})

    async def lifetime(app):
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=10),
                                 trust_env=False, auto_decompress=False) as client:
            app['client'] = client
            expiry_task = asyncio.create_task(expiry_loop())
            try:
                yield
            finally:
                expiry_task.cancel()
                await asyncio.gather(expiry_task, return_exceptions=True)

    async def check_health(base):
        try:
            async with app['client'].get(base + '/health', timeout=ClientTimeout(total=4), ssl=False) as response:
                data = await response.json()
                return {'available': response.status == 200 and data.get('ok', False),
                        'engine': data.get('engine'), 'languages': data.get('languages', ['zh', 'en'])}
        except Exception:
            return {'available': False}

    async def voice_status(request):
        asr, tts = await asyncio.gather(check_health(asr_base), check_health(tts_base))
        return web.json_response({'asr': asr, 'tts': tts})

    async def relay(source, target, *, sid=None, path=None, inbound=False):
        async for message in source:
            if inbound:
                if path == '/ws' and not owns(sid):
                    # Waiting visitors may view the UI and receive config, but
                    # cannot submit work or change global inference settings.
                    continue
                if path != '/ws' and not owns(sid):
                    break
                touch(sid)
            if message.type == WSMsgType.TEXT:
                await target.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await target.send_bytes(message.data)
            elif message.type == WSMsgType.ERROR:
                break

    async def proxy(request):
        sid = request.query.get('session_id', '')
        guarded = request.path in ('/ws/camera', '/ws/asr', '/api/tts', '/offer',
                                   '/api/rtsp/start', '/api/rtsp/stop', '/api/rtsp/status')
        if guarded and request.method == 'POST':
            try:
                sid = (await request.json()).get('session_id', '')
            except (ValueError, AttributeError):
                raise web.HTTPBadRequest(text='Expected a session identifier')
        if guarded and not owns(sid):
            return web.json_response({'error':'ALIVE_BUSY', **public_status(sid)}, status=409)
        if request.path == '/ws/asr':
            base = asr_base
        elif request.path in ('/api/tts', '/api/tts/config'):
            base = tts_base
        else:
            base = web_base
        url = base + str(request.rel_url)
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            try:
                upstream = await app['client'].ws_connect(url, ssl=False, heartbeat=20,
                                                          max_msg_size=4 * 1024 * 1024)
            except Exception:
                raise web.HTTPServiceUnavailable(text='Service is starting or unavailable. Please retry.')
            browser = web.WebSocketResponse(heartbeat=20, max_msg_size=4 * 1024 * 1024)
            tasks = []
            try:
                await browser.prepare(request)
                if request.path != '/ws':
                    active_sockets.setdefault(sid, set()).update((browser, upstream))
                tasks = [asyncio.create_task(relay(browser, upstream, sid=sid,
                                                  path=request.path, inbound=True)),
                         asyncio.create_task(relay(upstream, browser))]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            except ConnectionResetError:
                pass
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await upstream.close()
                if browser.prepared:
                    await browser.close()
                if request.path != '/ws':
                    active_sockets.get(sid, set()).difference_update((browser, upstream))
                if request.path == '/ws/camera' and owns(sid):
                    # Avoid awaiting cleanup inside the stream being closed.
                    asyncio.create_task(release(sid))
            return browser
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in HOP_HEADERS | {'host', 'content-length'}}
        try:
            async with app['client'].request(request.method, url, headers=headers,
                                             data=await request.read(), ssl=False,
                                             allow_redirects=False) as upstream:
                result = web.StreamResponse(status=upstream.status,
                    headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS})
                await result.prepare(request)
                async for packet in upstream.content.iter_chunked(65536):
                    await result.write(packet)
                await result.write_eof()
                return result
        except (OSError, asyncio.TimeoutError):
            raise web.HTTPBadGateway(text='ALIVE service is unavailable. Please retry.')

    app.cleanup_ctx.append(lifetime)
    app.router.add_get('/api/voice/status', voice_status)
    app.router.add_get('/api/use/status', admission_status)
    app.router.add_post('/api/use/acquire', admission_acquire)
    app.router.add_post('/api/use/heartbeat', admission_heartbeat)
    app.router.add_post('/api/use/release', admission_release)
    app.router.add_post('/_bridge/release', admission_release)
    app.router.add_route('*', '/{path:.*}', proxy)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--web', required=True)
    parser.add_argument('--asr', default='http://127.0.0.1:19308')
    parser.add_argument('--tts', default='http://127.0.0.1:19310')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8295)
    parser.add_argument('--cert', required=True)
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.cert, args.key)
    web.run_app(create_app(args.web.rstrip('/'), args.asr.rstrip('/'), args.tts.rstrip('/')),
                host=args.host, port=args.port, ssl_context=context, access_log=None)
