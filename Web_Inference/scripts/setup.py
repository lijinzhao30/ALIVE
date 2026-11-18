"""Create isolated dependency environments, then download the voice checkpoints."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Use Python 3.12 to create the validated environment layout.')
    for name in ('main', 'asr', 'tts'):
        target = ROOT / '.venvs' / name
        subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True)
        python = str(target / 'bin/python')
        subprocess.run([python, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([python, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements' / (name + '.txt'))], check=True)
    if not args.skip_download:
        subprocess.run([str(ROOT / '.venvs/main/bin/python'), str(ROOT / 'scripts/download_models.py')], check=True)

if __name__ == '__main__':
    main()
