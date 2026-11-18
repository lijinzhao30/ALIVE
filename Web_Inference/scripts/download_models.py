"""Download public voice checkpoints. HF_ENDPOINT can select a mirror."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'asr_model_path': 'Qwen/Qwen3-ASR-1.7B',
    'tts_model_path': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()
    path = ROOT / args.config
    if not path.exists():
        path = ROOT / 'config.example.json'
    config = json.loads(path.read_text())
    for key, repo in MODELS.items():
        snapshot_download(repo_id=repo, local_dir=str(ROOT / config[key]))

if __name__ == '__main__':
    main()
