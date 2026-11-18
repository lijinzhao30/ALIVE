"""Run one ALIVE instance on two GPUs. All paths resolve from this directory."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import ssl
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / '.runtime'
PORTS = {'main': 7060, 'adapter': 8070, 'web': 8091, 'asr': 19411, 'tts_engine': 19412, 'tts': 19413}
PROCESSES = []
LOGS = []

def environment(name, gpu=None):
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'SYSTEM_PROMPT', 'EVAL_RECORD_DIR'):
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CAMERA_TRANSPORT='websocket',
               LIVE_SAVE_OUTPUTS='false', LIVE_SAVE_DEBUG_INPUTS='false', SAVE_MODEL_INPUTS='false',
               SUMMARIZER_DEBUG='false', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    env['VLLM_PLUGINS'] = 'vllm_omni_register_models' if name == 'tts' else ''
    for key, folder in [('VLLM_CACHE_ROOT', 'vllm'), ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        env[key] = str(RUNTIME / 'cache' / name / folder)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env

def interpreter(name):
    return os.environ.get('ALIVE_' + name.upper() + '_PYTHON', str(ROOT / '.venvs' / name / 'bin/python'))

def spawn(name, command, env):
    log = (RUNTIME / 'logs' / (name + '.log')).open('w')
    LOGS.append(log)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PROCESSES.append((name, process))
    print('Starting ' + name, flush=True)
    return process

def wait_http(name, url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for label, process in PROCESSES:
            if process.poll() is not None:
                raise RuntimeError(label + ' exited; check .runtime/logs/' + label + '.log')
        try:
            with opener.open(url, timeout=3) as response:
                if response.status == 200:
                    print('Ready: ' + name, flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(name + ' did not become ready')

def cleanup():
    for _, process in reversed(PROCESSES):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    end = time.monotonic() + 20
    for _, process in PROCESSES:
        try:
            process.wait(timeout=max(0.1, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for log in LOGS:
        log.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--check', action='store_true', help='Validate files and configuration without starting services')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if str(config['main_gpu']) == str(config['voice_gpu']):
        parser.error('main_gpu and voice_gpu must select different devices')
    models = {k: (ROOT / config[k]).resolve() for k in ('model_path', 'asr_model_path', 'tts_model_path')}
    for key, path in models.items():
        if not (path / 'config.json').is_file():
            parser.error('Missing checkpoint config for ' + key + '; download it or edit config.json')
    for name in ('main', 'asr', 'tts'):
        if not Path(interpreter(name)).is_file():
            parser.error('Missing ' + name + ' environment; run scripts/setup.py')
    if args.check:
        print('Configuration and checkpoint directories are valid.')
        return
    # Fail before starting any worker if another instance owns a required port.
    for port in list(PORTS.values()) + [int(config['port'])]:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    (RUNTIME / 'logs').mkdir(parents=True, exist_ok=True)
    tls = RUNTIME / 'tls'
    tls.mkdir(exist_ok=True)
    if not (tls / 'cert.pem').exists() or not (tls / 'key.pem').exists():
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:3072', '-nodes', '-days', '365',
            '-out', str(tls / 'cert.pem'), '-keyout', str(tls / 'key.pem'), '-subj', '/CN=localhost',
            '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (tls / 'key.pem').chmod(0o600)
    py = interpreter('main')
    main_env = environment('main', config['main_gpu'])
    normalized = subprocess.check_output([py, str(ROOT / 'webinfer/fix_tokenizer_config.py'),
        str(models['model_path']), str(RUNTIME / 'model_links')], env=main_env, text=True).strip()
    timeout = int(config.get('startup_timeout', 2400))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        spawn('model', [py, '-m', 'vllm.entrypoints.openai.api_server', '--model', normalized,
            '--served-model-name', 'ALIVE', '--host', '127.0.0.1', '--port', str(PORTS['main']),
            '--gpu-memory-utilization', '0.9', '--max-model-len', '131072', '--tensor-parallel-size', '1',
            '--enable-prefix-caching', '--enable-chunked-prefill'], main_env)
        tts_env = environment('tts', config['voice_gpu'])
        tts_python = interpreter('tts')
        spawn('tts-engine', [tts_python, '-m', 'vllm.entrypoints.cli.main', 'serve', str(models['tts_model_path']),
            '--served-model-name', 'qwen-tts', '--deploy-config', 'voice/tts/deploy.yaml', '--omni',
            '--host', '127.0.0.1', '--port', str(PORTS['tts_engine'])], tts_env)
        wait_http('tts-engine', 'http://127.0.0.1:19412/health', timeout)
        spawn('tts', [tts_python, 'voice/tts/service.py', '--model', 'qwen-tts',
            '--upstream', 'http://127.0.0.1:19412', '--port', str(PORTS['tts'])], tts_env)
        asr_config = RUNTIME / 'asr'
        asr_config.mkdir(exist_ok=True)
        (asr_config / 'models.json').write_text(json.dumps({'qwen': {'path': str(models['asr_model_path'])}}))
        # Initialize ASR after TTS to avoid concurrent GPU memory profiling.
        spawn('asr', [interpreter('asr'), 'voice/asr/service.py', '--root', str(asr_config),
            '--port', str(PORTS['asr'])], environment('asr', config['voice_gpu']))
        wait_http('main', 'http://127.0.0.1:7060/v1/models', timeout)
        spawn('adapter', [py, 'webinfer/live_adapter.py', '--host', '127.0.0.1', '--port', str(PORTS['adapter']),
            '--adapter-model', 'streaming-infer-adapter', '--main-api-base', 'http://127.0.0.1:7060/v1',
            '--main-model', 'ALIVE', '--disable-summarizer', '--input-format', 'no_time',
            '--chunk', '30', '--retain-frames', '10', '--frame-seconds', '1.0',
            '--resize-short-edge', '480', '--max-pixels', '786432', '--main-max-tokens', '256',
            '--main-temperature', '0.8', '--main-top-p', '0.9', '--main-top-k', '40',
            '--main-repetition-penalty', '1.1', '--main-presence-penalty', '1.5',
            '--frame-save-dir', str(RUNTIME / 'frames'), '--allowed-local-image-roots', str(RUNTIME / 'frames'),
            '--save-root', str(RUNTIME / 'state'), '--no-live-save', '--no-debug-inputs',
            '--no-save-model-inputs', '--no-summarizer-debug'], main_env)
        wait_http('adapter', 'http://127.0.0.1:8070/health', timeout)
        web_env = environment('web')
        web_env['PYTHONPATH'] = str(ROOT / 'webui/src')
        spawn('web', [py, '-m', 'alive_webui.server', '--host', '127.0.0.1', '--port', str(PORTS['web']),
            '--model', 'streaming-infer-adapter', '--api-base', 'http://127.0.0.1:8070/v1',
            '--ssl-cert', str(tls / 'cert.pem'), '--ssl-key', str(tls / 'key.pem')], web_env)
        for name, url in [('web', 'https://127.0.0.1:8091/'), ('asr', 'http://127.0.0.1:19411/health'), ('tts', 'http://127.0.0.1:19413/health')]:
            wait_http(name, url, timeout)
        spawn('gateway', [py, 'voice/gateway.py', '--web', 'https://127.0.0.1:8091',
            '--asr', 'http://127.0.0.1:19411', '--tts', 'http://127.0.0.1:19413',
            '--host', '127.0.0.1', '--port', str(config['port']), '--cert', str(tls / 'cert.pem'), '--key', str(tls / 'key.pem')], web_env)
        url = 'https://localhost:' + str(config['port'])
        wait_http('ALIVE', url + '/api/use/status', timeout)
        print('\nOpen ' + url + ' in a browser on this machine. Press Ctrl+C to stop all services.', flush=True)
        while True:
            for label, process in PROCESSES:
                if process.poll() is not None:
                    raise RuntimeError(label + ' exited; inspect .runtime/logs')
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nStopping ALIVE.', flush=True)
    finally:
        cleanup()

if __name__ == '__main__':
    main()
