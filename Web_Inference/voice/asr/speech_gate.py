"""CPU Silero VAD before ASR; recurrent state belongs to one utterance.

No transcript, conversation history, or user audio is persisted here.
"""
from collections import deque
import importlib.util
from pathlib import Path

import numpy as np

RATE = 16000
FRAME = 512
START_THRESHOLD = 0.60
CONTINUE_THRESHOLD = 0.35
MIN_START_FRAMES = 6  # 192 ms of consecutive likely speech, not cumulative noise.
PREROLL_FRAMES = 16  # 512 ms includes the confirmation interval.
END_FRAMES = 47  # 1504 ms of non-speech, independent of background volume.
TAIL_FRAMES = 6  # At most 192 ms of non-speech is passed to ASR at the end.


class SileroVad:
    """Share the immutable ONNX session, never the recurrent h/c state."""
    def __init__(self, path=None):
        import onnxruntime as ort
        if path is None:
            spec = importlib.util.find_spec('faster_whisper')
            if spec is None:
                raise RuntimeError('The existing faster-whisper Silero VAD asset is required')
            path = Path(spec.origin).parent / 'assets/silero_vad_v6.onnx'
        self.path = Path(path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(self.path), sess_options=opts,
                                           providers=['CPUExecutionProvider'])

    def new_stream(self):
        return VadStream(self.session)


class VadStream:
    def __init__(self, session):
        self.session = session
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)

    def probability(self, frame):
        x = np.concatenate((self.context, frame)).reshape(1, -1)
        result, self.h, self.c = self.session.run(None, {'input': x, 'h': self.h, 'c': self.c})
        self.context = frame[-64:].copy()
        return float(np.asarray(result).reshape(-1)[0])


class SpeechGate:
    def __init__(self, scorer, continuous=False):
        self.scorer = scorer
        self.continuous = continuous
        self.buffer = np.empty(0, dtype=np.float32)
        self.preroll = deque(maxlen=PREROLL_FRAMES)
        self.tail = deque()
        self.onset_frames = self.silent_frames = 0
        self.heard_voice = False
        self.ended = False
        self.frames = self.speech_frames = 0
        self.max_probability = 0.0

    def feed(self, pcm):
        if self.ended:
            return np.empty(0, dtype=np.float32)
        self.buffer = np.concatenate((self.buffer, pcm))
        output = []
        while len(self.buffer) >= FRAME and not self.ended:
            frame, self.buffer = self.buffer[:FRAME], self.buffer[FRAME:]
            output.extend(self._frame(frame))
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    def _frame(self, frame, valid=None):
        valid = FRAME if valid is None else valid
        probability = self.scorer.probability(frame)
        self.frames += 1
        self.max_probability = max(probability, self.max_probability)
        if probability >= START_THRESHOLD:
            self.speech_frames += 1
        actual = frame[:valid].copy()
        if not self.heard_voice:
            self.preroll.append(actual)
            self.onset_frames = self.onset_frames + 1 if probability >= START_THRESHOLD else 0
            if self.onset_frames < MIN_START_FRAMES:
                return []
            self.heard_voice = True
            output = list(self.preroll)
            self.preroll.clear()
            return output
        if probability >= CONTINUE_THRESHOLD:
            self.silent_frames = 0
            output = list(self.tail) + [actual]
            self.tail.clear()
            return output
        self.silent_frames += 1
        self.tail.append(actual)
        if self.continuous and len(self.tail) > END_FRAMES:
            # Bound long press-to-talk pauses; keep recent pre-roll for resumption.
            self.tail.popleft()
        if not self.continuous and self.silent_frames >= END_FRAMES:
            self.ended = True
        return []

    def finish(self):
        output = []
        if len(self.buffer) and not self.ended:
            size = len(self.buffer)
            output.extend(self._frame(np.pad(self.buffer, (0, FRAME - size)), valid=size))
        self.buffer = np.empty(0, dtype=np.float32)
        if self.heard_voice:
            output.extend(list(self.tail)[:TAIL_FRAMES])
        self.tail.clear()
        self.preroll.clear()
        self.ended = True
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    def metrics(self):
        return {'speech_detected': self.heard_voice, 'speech_frames': self.speech_frames,
                'frames': self.frames, 'max_speech_probability': round(self.max_probability, 4)}
