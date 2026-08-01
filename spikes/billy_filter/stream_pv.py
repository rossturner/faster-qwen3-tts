"""Streaming Billy chorus, take 2: a stateful phase vocoder.

Take 1 used delay-line pitch shifting and was rejected -- "too many copies of the voice".
That was a real defect, not a tuning problem. Two taps half a grain apart means two copies
40 ms apart, and crossfading them across the WHOLE grain leaves both audible essentially
all the time: four copies plus the dry, where the approved filter has two plus the dry.

Shortening the crossfade does not rescue it. The duty cycle of the overlap is X/D, and
suppressing it needs X << D, i.e. a large drift range -- but the drift range IS the delay,
and the whole filter is built on the delay sitting at 8 and 16 ms. Delay-line shifting
cannot hold a fixed delay and a fixed detune at once. Wrong tool.

A phase vocoder holds both, because it shifts pitch in the frequency domain and leaves the
delay to a separate fixed line. It is inherently block-based, so it streams as long as the
analysis phase, synthesis phase and overlap-add tail persist between calls -- which is what
this does. The cost is one window of algorithmic latency:

    n_fft 1024 at 24 kHz = 42.7 ms

against a measured TTFA of 260-480 ms. Under 10% of the budget, and constant.

Usage:
    .venv/bin/python spikes/billy_filter/stream_pv.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
OUT = HERE / "out" / "stream_pv"


def _princarg(p):
    return np.mod(p + np.pi, 2 * np.pi) - np.pi


class PhaseVocoderShifter:
    """Stateful pitch shift by a fixed ratio. Latency is one window."""

    def __init__(self, ratio, n_fft=1024, hop=None):
        self.ratio = float(ratio)
        self.n_fft = int(n_fft)
        self.hop = int(hop or n_fft // 4)
        self.win = np.hanning(self.n_fft + 1)[:self.n_fft]
        nbins = self.n_fft // 2 + 1
        self.expected = 2 * np.pi * self.hop / self.n_fft * np.arange(nbins)
        self.last_phase = np.zeros(nbins)
        self.sum_phase = np.zeros(nbins)
        self.inbuf = np.zeros(self.n_fft)
        self.outbuf = np.zeros(self.n_fft)
        self.pending = np.zeros(0, np.float32)
        # hann^2 overlap-add at 75% sums to 1.5
        self.norm = 1.5 * (self.n_fft / self.hop) / 4.0
        self.idx = np.round(np.arange(nbins) * self.ratio).astype(int)
        self.valid = self.idx < nbins

    def _frame(self):
        X = np.fft.rfft(self.win * self.inbuf)
        mag, phase = np.abs(X), np.angle(X)
        dphi = _princarg(phase - self.last_phase - self.expected)
        true_freq = self.expected + dphi
        self.last_phase = phase
        nbins = len(mag)
        new_mag = np.zeros(nbins)
        new_freq = np.zeros(nbins)
        np.add.at(new_mag, self.idx[self.valid], mag[self.valid])
        new_freq[self.idx[self.valid]] = true_freq[self.valid] * self.ratio
        self.sum_phase = _princarg(self.sum_phase + new_freq)
        frame = np.fft.irfft(new_mag * np.exp(1j * self.sum_phase), self.n_fft)
        self.outbuf += self.win * frame
        out = self.outbuf[:self.hop] / self.norm
        self.outbuf = np.concatenate([self.outbuf[self.hop:], np.zeros(self.hop)])
        return out

    def process(self, block):
        self.pending = np.concatenate([self.pending, np.asarray(block, np.float32)])
        out = []
        while len(self.pending) >= self.hop:
            self.inbuf = np.concatenate([self.inbuf[self.hop:], self.pending[:self.hop]])
            self.pending = self.pending[self.hop:]
            out.append(self._frame())
        return (np.concatenate(out) if out else np.zeros(0, np.float32)).astype(np.float32)


class StreamingChorus:
    """Billy's doubling: two phase-vocoder copies through fixed delay lines."""

    def __init__(self, sr, cents=(+26.0, -26.0), delays_ms=(8.0, 16.0),
                 amount=0.55, n_fft=1024):
        self.sr = sr
        self.amount = float(amount)
        self.shifters = [PhaseVocoderShifter(2 ** (c / 1200.0), n_fft=n_fft) for c in cents]
        self.delays = [int(round(d * sr / 1000.0)) for d in delays_ms]
        self.lines = [np.zeros(d, np.float32) for d in self.delays]
        self.latency = n_fft                       # samples of algorithmic delay
        self.dry_line = np.zeros(self.latency, np.float32)

    def process(self, block):
        block = np.asarray(block, np.float32)
        wet_parts = []
        for i, sh in enumerate(self.shifters):
            y = sh.process(block)
            if self.delays[i]:
                y = np.concatenate([self.lines[i], y])
                self.lines[i] = y[-self.delays[i]:].astype(np.float32)
                y = y[:-self.delays[i]] if self.delays[i] else y
            wet_parts.append(y)
        n = min(len(p) for p in wet_parts) if wet_parts else 0
        wet = sum(p[:n] for p in wet_parts) / len(wet_parts) if n else np.zeros(0)
        # delay the dry by the vocoder latency so the copies stay aligned to it
        d = np.concatenate([self.dry_line, block])
        dry = d[:n] if n else np.zeros(0, np.float32)
        self.dry_line = d[n:].astype(np.float32) if n else d.astype(np.float32)
        if n == 0:
            return np.zeros(0, np.float32)
        return ((1.0 - self.amount) * dry + self.amount * wet).astype(np.float32)


def _norm(x):
    p = float(np.abs(x).max())
    return x * (0.89 / p) if p > 0 else x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = sorted((HERE / "out" / "refs").glob("*_A_plain.wav")) + \
        sorted((HERE / "out" / "tts_settled").glob("*_A_plain.wav"))
    print("chunk invariance (333 ms vs 83 ms vs whole file):")
    for src in srcs:
        x, sr = sf.read(src, dtype="float32")
        outs = {}
        for label, step in (("whole", len(x)), ("333ms", int(sr * .333)), ("83ms", int(sr / 12))):
            c = StreamingChorus(sr)
            outs[label] = np.concatenate(
                [c.process(x[i:i + step]) for i in range(0, len(x), step)])
        n = min(len(v) for v in outs.values())
        d1 = float(np.abs(outs["whole"][:n] - outs["333ms"][:n]).max())
        d2 = float(np.abs(outs["whole"][:n] - outs["83ms"][:n]).max())
        stem = src.stem.replace("_A_plain", "")
        print(f"  {stem:16s} 333ms {d1:.2e}   83ms {d2:.2e}")
        sf.write(OUT / f"{stem}_STREAMPV.wav", _norm(outs["whole"]), sr)

    from families import load
    from grid import double
    print("\nstreaming vs approved offline filter:")
    for src in srcs:
        x = load(src)
        off = np.asarray(double(x, src.stem, offsets=[+26, -26], delays=[8, 16],
                                amount=0.55), np.float32)
        st = StreamingChorus(48000).process(x)
        m = min(len(off), len(st))
        a = off[:m] / (np.abs(off[:m]).max() or 1)
        b = st[:m] / (np.abs(st[:m]).max() or 1)
        d = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
        stem = src.stem.replace("_A_plain", "")
        sf.write(OUT / f"{stem}_OFFLINE.wav", _norm(off), 48000)
        print(f"  {stem:16s} difference {d:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
