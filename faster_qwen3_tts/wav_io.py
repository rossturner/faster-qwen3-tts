import struct
import numpy as np


def _to_pcm16(pcm: np.ndarray) -> bytes:
    return np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()


def to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    raw = _to_pcm16(np.asarray(pcm, dtype=np.float32).flatten())
    n_channels, bits = 1, 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                                    byte_rate, block_align, bits)
    header += b"data" + struct.pack("<I", len(raw))
    return header + raw
