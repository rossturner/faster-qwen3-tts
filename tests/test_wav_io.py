import io, wave
import numpy as np
from faster_qwen3_tts.wav_io import to_wav_bytes

def test_wav_roundtrip_header_and_frames():
    sr = 24000
    pcm = np.sin(np.linspace(0, 3.14, sr)).astype(np.float32)  # 1.0s
    data = to_wav_bytes(pcm, sr)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == sr
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == sr  # 1 second

def test_wav_clips_out_of_range():
    data = to_wav_bytes(np.array([2.0, -2.0], dtype=np.float32), 24000)
    with wave.open(io.BytesIO(data)) as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert frames[0] == 32767 and frames[1] == -32768
