"""A streaming-safe implementation of Billy's chorus.

The approved filter -- two copies at +-26 cents, delayed 8 and 16 ms, 55% wet -- was
built with librosa's phase vocoder, which needs the whole utterance. The streaming
endpoint emits every ~333 ms, so a per-chunk phase vocoder would seam three times a
second.

This gets the same detune from delay-line pitch shifting instead: read the delay line at
rate r and the pitch shifts by r, at the cost of the delay drifting by (1-r) per sample.
The drift is bounded by wrapping every grain, with a second tap half a grain out of phase
so the crossfade lands where each tap's gain is zero and the wrap is inaudible.

The reason this is a good fit here rather than a compromise: the detune is tiny. At 26
cents, r = 1.0151, so the delay drifts 0.0151 samples per sample and an 80 ms grain lasts
    0.080 / 0.0151 = 5.3 seconds
between wraps. A pitch shifter that only splices every five seconds is essentially
artifact-free, which is not true of the large shifts these are usually criticised for.

Gains are sin(pi*u) and |cos(pi*u)|, so gA^2 + gB^2 = 1 and the crossfade is
power-preserving rather than dipping in the middle.

State carries across process() calls, so chunked and whole-file output are sample
identical -- verified in main() rather than asserted.

Usage:
    .venv/bin/python spikes/billy_filter/streaming_chorus.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
OUT = HERE / "out" / "streaming"
SR = 48000


class StreamingChorus:
    """Billy's doubling, as a stateful block processor."""

    def __init__(self, sr=SR, cents=(+26.0, -26.0), delays_ms=(8.0, 16.0),
                 amount=0.55, grain_ms=80.0):
        self.sr = sr
        self.amount = float(amount)
        self.ratios = [2 ** (c / 1200.0) for c in cents]
        self.base = [d * sr / 1000.0 for d in delays_ms]
        self.grain = grain_ms * sr / 1000.0
        self.frac = [0.0 for _ in cents]
        self.tail_len = int(max(self.base) + self.grain) + 4
        self.tail = np.zeros(self.tail_len, np.float32)

    def reset(self):
        self.frac = [0.0 for _ in self.ratios]
        self.tail = np.zeros(self.tail_len, np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, np.float32)
        n = len(block)
        if n == 0:
            return block
        buf = np.concatenate([self.tail, block])
        grid = np.arange(len(buf), dtype=np.float64)
        idx = np.arange(n, dtype=np.float64) + len(self.tail)
        wet = np.zeros(n, np.float64)
        for v, (r, base) in enumerate(zip(self.ratios, self.base)):
            step = 1.0 - r
            frac = self.frac[v] + step * np.arange(n, dtype=np.float64)
            a = np.mod(frac, self.grain)
            b = np.mod(frac + self.grain / 2.0, self.grain)
            u = a / self.grain
            ga, gb = np.sin(np.pi * u), np.abs(np.cos(np.pi * u))
            ra = np.interp(idx - (base + a), grid, buf)
            rb = np.interp(idx - (base + b), grid, buf)
            wet += (ga * ra + gb * rb) / len(self.ratios)
            self.frac[v] = float(np.mod(frac[-1] + step, self.grain))
        self.tail = buf[-self.tail_len:].astype(np.float32)
        return ((1.0 - self.amount) * block + self.amount * wet).astype(np.float32)


def _norm(x):
    p = float(np.abs(x).max())
    return x * (0.89 / p) if p > 0 else x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = sorted((HERE / "out" / "refs").glob("*_A_plain.wav")) + \
        sorted((HERE / "out" / "tts_settled").glob("*_A_plain.wav"))

    print("chunked vs whole-file (must be identical for streaming to be safe):")
    worst = 0.0
    for src in srcs:
        x, sr = sf.read(src, dtype="float32")
        whole = StreamingChorus(sr).process(x)
        ch = StreamingChorus(sr)
        step = int(sr * 0.333)                      # lyrebird's ~333 ms chunk
        chunked = np.concatenate([ch.process(x[i:i + step])
                                  for i in range(0, len(x), step)])
        d = float(np.abs(whole - chunked).max())
        worst = max(worst, d)
        print(f"  {src.stem.replace('_A_plain',''):16s} max sample diff {d:.3e}")
        sf.write(OUT / f"{src.stem.replace('_A_plain','')}_STREAM.wav", _norm(whole), sr)
        sf.write(OUT / f"{src.stem.replace('_A_plain','')}_00_dry.wav", _norm(x), sr)
    print(f"\n  worst across all files: {worst:.3e}")

    # how close is the streaming detune to the phase-vocoder version it replaces?
    from families import load
    from grid import double
    print("\nstreaming vs approved offline filter:")
    for src in srcs:
        x = load(src)
        off = np.asarray(double(x, src.stem, offsets=[+26, -26], delays=[8, 16],
                                amount=0.55), np.float32)
        st = StreamingChorus(SR).process(x)
        m = min(len(off), len(st))
        a, b = off[:m], st[:m]
        a = a / (np.abs(a).max() or 1); b = b / (np.abs(b).max() or 1)
        diff = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
        sf.write(OUT / f"{src.stem.replace('_A_plain','')}_OFFLINE.wav", _norm(off), SR)
        print(f"  {src.stem.replace('_A_plain',''):16s} difference {diff:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
