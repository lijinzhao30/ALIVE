"""CPU integration checks using synthetic frames and a mock model endpoint."""
import asyncio
import base64
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'webinfer'))
from adapter_config import AdapterConfig
from adapter_request import _resize_image_if_needed
from live_adapter import create_app

spec = importlib.util.spec_from_file_location('gateway', ROOT / 'voice/gateway.py')
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)

class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.requests = []
        async def completion(request):
            payload = await request.json()
            self.requests.append(payload)
            return web.json_response({'id':'test', 'object':'chat.completion', 'created':0, 'model':'ALIVE',
                'choices':[{'index':0, 'message':{'role':'assistant','content':'</response> A green object.'}, 'finish_reason':'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
        model = web.Application()
        model.router.add_post('/v1/chat/completions', completion)
        self.model = TestServer(model)
        await self.model.start_server()
        config = AdapterConfig(main_api_base=str(self.model.make_url('/v1')), main_model='ALIVE',
            enable_summarizer=False, per_session_dirs=False, frame_save_dir=self.tmp.name,
            chunk=30, retain_frames=10, input_format='no_time')
        self.client = TestClient(TestServer(create_app(config)))
        await self.client.start_server()
        self.adapter = self.client.server.app['adapter']
        image = io.BytesIO()
        Image.new('RGB', (320, 240), 'green').save(image, format='JPEG')
        self.image = 'data:image/jpeg;base64,' + base64.b64encode(image.getvalue()).decode()

    async def asyncTearDown(self):
        if self.adapter._cleanup_task:
            self.adapter._cleanup_task.cancel()
            await asyncio.gather(self.adapter._cleanup_task, return_exceptions=True)
        await self.adapter.main_client.close()
        await self.client.close()
        await self.model.close()
        self.tmp.cleanup()

    async def frame(self, session, question=''):
        content = [{'type':'image_url','image_url':{'url':self.image}}]
        if question:
            content.append({'type':'text','text':question})
        response = await self.client.post('/v1/chat/completions', headers={'x-streaming-session':session},
            json={'model':'streaming-infer-adapter','messages':[{'role':'user','content':content}]})
        payload = await response.json()
        self.assertEqual(response.status, 200, payload)
        return payload

    async def test_window_history_and_generation(self):
        for i in range(32):
            await self.frame('session-one', 'What is in view?' if i == 0 else '')
        payload = self.requests[-1]
        self.assertEqual(payload['temperature'], 0.8)
        self.assertEqual(payload['top_p'], 0.9)
        self.assertEqual(payload['top_k'], 40)
        self.assertEqual(payload['repetition_penalty'], 1.1)
        self.assertEqual(payload['presence_penalty'], 1.5)
        self.assertEqual(payload['max_tokens'], 256)
        roles = [m['role'] for m in payload['messages']]
        self.assertEqual(roles[0], 'system')
        self.assertTrue(all(a != b for a,b in zip(roles[1:], roles[2:])))
        self.assertIn('What is in view?', str(payload['messages']))
        state = self.adapter.sessions['session-one']
        self.assertLess(len(state.current_chunk['image_paths']), 30)
        self.assertGreaterEqual(len(state.current_chunk['image_paths']), 10)
        self.assertEqual(state.turn_count, 32)
        self.assertIsNone(self.adapter.summarizer)
        await self.frame('session-two', 'Describe the color.')
        self.assertNotIn('What is in view?', str(self.requests[-1]['messages']))
        response = await self.client.post('/v1/streaming/reset', headers={'x-streaming-session':'session-one'}, json={})
        self.assertTrue((await response.json())['removed'])
        self.assertNotIn('session-one', self.adapter.sessions)
        self.assertIn('session-two', self.adapter.sessions)

    def test_downscale_only(self):
        for size, expected in [((320,240),(320,240)),((1280,720),(853,480)),((720,1280),(480,853))]:
            image = Image.new('RGB',size)
            result = _resize_image_if_needed(image, max_pixels=786432, resize_short_edge=480)
            self.assertEqual((result or image).size, expected)

class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_lock_cleanup_and_voice_routing(self):
        cleaned=[]
        async def health(request):
            return web.json_response({'ok':True,'engine':'test'})
        async def cleanup(request):
            cleaned.append((await request.json())['session_id'])
            return web.json_response({'ok':True})
        async def ws(request):
            stream=web.WebSocketResponse()
            await stream.prepare(request)
            await stream.send_json({'type':'start'})
            async for msg in stream:
                if isinstance(msg.data,str):await stream.send_str(msg.data)
            return stream
        upstream=web.Application()
        upstream.router.add_get('/health',health)
        upstream.router.add_post('/api/session/cleanup',cleanup)
        upstream.router.add_get('/ws/asr',ws)
        upstream.router.add_get('/api/tts',ws)
        async with TestServer(upstream) as server:
            url=str(server.make_url('')).rstrip('/')
            async with TestClient(TestServer(gateway.create_app(url,url,url))) as client:
                a='owner-session';b='waiting-session'
                self.assertEqual((await client.post('/api/use/acquire',json={'session_id':a})).status,200)
                self.assertEqual((await client.post('/api/use/acquire',json={'session_id':b})).status,409)
                for path in ('/ws/asr','/api/tts'):
                    async with client.ws_connect(path+'?session_id='+a) as sock:
                        self.assertEqual((await sock.receive_json())['type'],'start')
                status=await (await client.get('/api/voice/status')).json()
                self.assertTrue(status['asr']['available'] and status['tts']['available'])
                self.assertTrue((await (await client.post('/api/use/release',json={'session_id':a})).json())['released'])
                self.assertEqual(cleaned,[a])
                self.assertEqual((await client.post('/api/use/acquire',json={'session_id':b})).status,200)

if __name__ == '__main__':
    unittest.main()
