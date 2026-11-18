# ALIVE Web Inference

A local streaming camera assistant with text input, streaming speech recognition, streaming speech synthesis, Chinese/English UI labels, and nine selectable voices. The visual layout and inference defaults match the ALIVE demo configuration.

## Requirements

- Linux, Python 3.12, `venv`, `pip`, and OpenSSL.
- Two CUDA-capable GPUs with sufficient memory for the configured models and context. A compatible driver is required for the selected PyTorch/vLLM wheels. No specific GPU model is required by the launcher.
- A camera, microphone, and a browser on the same machine. Use Chrome or another browser supporting the required media APIs.
- Internet access for initial package/model downloads and the UI's icon/Markdown/math dependencies.

Three Python environments isolate the different inference-engine dependencies. They do not imply three GPUs: the ASR and TTS processes explicitly share the second GPU. TTS initializes before ASR to avoid overlapping memory profiling.

## 1. Install

From the repository root:

```bash
cd Web_Inference
python3.12 scripts/setup.py
cp config.example.json config.json
```

This creates `.venvs/main`, `.venvs/asr`, and `.venvs/tts`, and downloads the two voice checkpoints into `models/`. The dependency versions are based on the running demo environments. Installation must succeed with the target machine's CUDA/driver combination; do not merge all three requirements files into one environment.

To download through a Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3.12 scripts/setup.py
```

Or install without downloads, then download separately:

```bash
python3.12 scripts/setup.py --skip-download
.venvs/main/bin/python scripts/download_models.py
```

| Component | Checkpoint |
| --- | --- |
| Main model | |
| Speech recognition | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| Speech synthesis | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) |

The main-model URL is intentionally blank. Put the complete ALIVE checkpoint, including its tokenizer and image processor files, in `models/ALIVE`, or set `model_path` in `config.json`. Model locations resolve relative to `Web_Inference`; they are not bundled with the code.

## 2. Configure and start

Default `config.json` settings:

```json
{
  "model_path": "models/ALIVE",
  "model_url": "",
  "asr_model_path": "models/Qwen3-ASR-1.7B",
  "tts_model_path": "models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "main_gpu": "0",
  "voice_gpu": "1",
  "port": 8090,
  "startup_timeout": 2400
}
```

Device identifiers refer to local physical GPU indices or GPU UUIDs. Choose two distinct devices. The launcher explicitly assigns visibility separately for main and voice workers; the two TTS stages both use device `0` within the voice process's visible device set.

```bash
python3.12 start.py --check
python3.12 start.py
```

`--check` verifies checkpoint configuration files and environment executables; it does not load weights or establish GPU compatibility. The normal launcher waits for all services to become healthy before printing the browser URL.

Open **https://localhost:8090**. The launcher creates a local self-signed certificate; accept it for this local address, allow camera/microphone access, and start the camera. Enable **Sound** for model speech. The language selector changes interface labels only; it does not translate the conversation. Voice mode and experimental full duplex are available; full duplex is disabled by default.

Press **Ctrl+C** in the launcher terminal to stop all of its workers and release both GPUs. A worker failure also shuts down the other workers. Service logs are in `.runtime/logs/`. The launcher binds services to loopback and does not create public links.

## Inference configuration

| Setting | Default |
| --- | --- |
| Public model identity | ALIVE |
| Temperature | 0.8 |
| Top-p | 0.9 |
| Top-k | 40 |
| Repetition penalty | 1.1 |
| Presence penalty | 1.5 |
| Maximum generated tokens | 256 |
| Maximum model context | 131072 |
| Main model GPU memory utilization | 0.9 |
| Prefix caching / chunked prefill | Enabled |
| Streaming input format | `no_time` |
| Visual window | 30 frames; retain the most recent 10 when the window rolls |
| Frame interval / batch | 1 second / 1 frame |
| Summary model | Disabled |
| Image processor pixel limit | 786432, unchanged |

The browser requests a camera stream around 1280×720. It proportionally shrinks uploaded frames only when their short edge exceeds 480 pixels; smaller frames are not enlarged. For example, 1280×720 becomes about 853×480. The adapter also enforces downscale-only behavior. Model-internal image preprocessing is left to the checkpoint's processor.

Inference admits at most one in-flight camera request; frames arriving while it is busy are skipped. Consequently, one frame per second is a target, not a guaranteed throughput. User and assistant messages retain their alternating structure. Older visual frames leave the window while sparse question/answer history remains; no separate summary-memory model runs.

The streaming system prompt is in `webinfer/adapter_common.py`. It requests `</silence>` when no answer is warranted and `</response>` followed by a reply when the evidence is sufficient. It retains the ALIVE identity section. `webinfer/adapter_identity.py` provides the same deterministic identity responses for explicit identity questions. Normal model reply text is displayed after action-token removal and is passed to TTS; the displayed answer is not transcribed from synthesized speech.

The loader creates a runtime-only tokenizer mirror if `extra_special_tokens` requires a format conversion. It never edits the original checkpoint.

## Speech

ASR uses streaming recognition with CPU voice-activity detection, automatic/Chinese/English language selection, a 30-second utterance cap, and approximately 1.5 seconds of silence for endpoint detection. Its GPU memory utilization is 0.2. TTS uses two streaming stages, each with GPU memory utilization 0.2, on the same voice GPU. See `voice/tts/deploy.yaml` for the unchanged synthesis settings.

Default voice: `serena`. Available voices: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`. The browser's sound toggle controls model-audio playback. Microphone audio is handled for live recognition, not stored as a demo recording.

## Session data and validation

This distribution excludes screen recording, downloadable demo recordings, and evaluation-session frame/question logging. The adapter uses temporary image files for its active context under `.runtime/frames`; these are inference working files, not a benchmark export. Runtime caches, logs, certificates, checkpoints, and local configuration are ignored by Git.

Run the CPU-only integration checks with the main environment:

```bash
.venvs/main/bin/python -m unittest discover -s tests -v
```

These checks exercise request formatting, image resizing, window rollover, session isolation/reset, and voice gateway routing with synthetic inputs. They do not substitute for a full GPU warmup and real microphone/camera test on a new machine.

Upstream copyright/license headers are retained; see the repository-level `THIRD_PARTY_NOTICES.md`.
