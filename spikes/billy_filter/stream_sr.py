"""Streaming Billy chorus, take 3: time-stretch then resample.

Take 2 (bin-shifting phase vocoder) came back "flatter / less effect", and it had two
real defects behind that:

  1. At 26 cents the shift is ~3 Hz against 47 Hz bins, so round(k*r) == k for every bin
     below about 33. The entire low end was not being shifted at all.
  2. Bin shifting moves pitch while leaving the spectral envelope roughly in place. The
     approved filter used librosa's pitch_shift, which is time-stretch THEN resample --
     the resample scales the whole spectrum, formants included, so each copy sounds like
     a slightly different voice. That difference is where the richness comes from, and
     approximately preserving formants throws it away.

So this reproduces the actual algorithm, in two streaming stages:

    stretch by Hs/Ha   phase vocoder, integer hops, longer output
    resample by Hs/Ha  cubic, fractional position, back to the original length

Net: pitch multiplied by Hs/Ha, duration unchanged, formants scaled -- the same operation
librosa performs, arranged so both stages carry state between calls.

Integer hops make the ratio slightly inexact: 260/256 is 26.8 cents rather than 26.0, and
252/256 is -27.3. Under 1.5 cents of error on a setting that was indistinguishable across
12 to 80 cents, which is well inside the noise of how it was chosen.

Usage:
    .venv/bin/python spikes/billy_filter/stream_sr.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
OUT = HERE / "out" / "stream_sr"


def _princarg(p):
    return np.mod(p + np.pi, 2 * np.pi) - np.pi


class PVStretch:
    """Phase-vocoder time-stretch by hs/ha, streaming."""

    def __init__(self, n_fft=1024, ha=256, hs=260):
        self.n, self.ha, self.hs = int(n_fft), int(ha), int(hs)
        self.win = np.hanning(self.n + 1)[:self.n]
        nb = self.n // 2 + 1
        self.expected = 2 * np.pi * self.ha / self.n * np.arange(nb)
        self.last_phase = np.zeros(nb)
        self.sum_phase = np.zeros(nb)
        self.inbuf = np.zeros(self.n)
        self.outbuf = np.zeros(self.n + self.hs)
        self.pending = np.zeros(0, np.float32)
        # exact overlap-add gain of win^2 at hop hs, in steady state
        acc = np.zeros(self.n * 4)
        for m in range(0, self.n * 3, self.hs):
            acc[m:m + self.n] += self.win ** 2
        self.norm = float(np.mean(acc[self.n:self.n * 2]))

    def process(self, block):
        self.pending = np.concatenate([self.pending, np.asarray(block, np.float64)])
        out = []
        while len(self.pending) >= self.ha:
            self.inbuf = np.concatenate([self.inbuf[self.ha:], self.pending[:self.ha]])
            self.pending = self.pending[self.ha:]
            X = np.fft.rfft(self.win * self.inbuf)
            mag, phase = np.abs(X), np.angle(X)
            true_freq = self.expected + _princarg(phase - self.last_phase - self.expected)
            self.last_phase = phase
            self.sum_phase = _princarg(self.sum_phase + true_freq * (self.hs / self.ha))
            frame = np.fft.irfft(mag * np.exp(1j * self.sum_phase), self.n)
            self.outbuf[:self.n] += self.win * frame
            out.append(self.outbuf[:self.hs] / self.norm)
            self.outbuf = np.concatenate([self.outbuf[self.hs:], np.zeros(self.hs)])
        return np.concatenate(out) if out else np.zeros(0)


class Resampler:
    """Fractional resampler: consumes `rate` input samples per output sample.

    'cubic' is Catmull-Rom over 4 points; 'sinc' is a 32-tap Blackman-windowed sinc,
    which is what librosa's resample does at high quality. At a 1.5% rate change cubic
    is already gentle, so this exists to rule the resampler out rather than because it
    is expected to matter.
    """

    def __init__(self, rate, quality="cubic"):
        self.rate = float(rate)
        self.quality = quality
        self.taps = 16 if quality == "sinc" else 2
        self.pos = 0.0
        self.hist = np.zeros(2 * self.taps)

    def _sinc(self, buf, t):
        i = np.floor(t).astype(int)
        f = t - i
        k = np.arange(-self.taps + 1, self.taps + 1)
        # offsets relative to each centre; window over the same span
        d = f[:, None] - k[None, :]
        w = np.sinc(d) * (0.42 - 0.5 * np.cos(2 * np.pi * (k[None, :] + self.taps) /
                                              (2 * self.taps)) +
                          0.08 * np.cos(4 * np.pi * (k[None, :] + self.taps) /
                                        (2 * self.taps)))
        idx = i[:, None] + k[None, :]
        return np.sum(buf[idx] * w, axis=1) / np.sum(w, axis=1)

    def process(self, block):
        buf = np.concatenate([self.hist, np.asarray(block, np.float64)])
        # a centre at t reads buf[i-taps+1 .. i+taps], so both ends need taps of margin
        n_avail = len(buf) - 2 * self.taps
        idx = []
        p = self.pos
        while p < n_avail - 1:
            idx.append(p)
            p += self.rate
        keep_n = 2 * self.taps
        if not idx:
            self.hist = buf[-keep_n:] if len(buf) >= keep_n else buf
            self.pos = p - max(len(buf) - keep_n, 0)
            return np.zeros(0)
        t = np.asarray(idx) + self.taps
        if self.quality == "sinc":
            y = self._sinc(buf, t)
        else:
            i = np.floor(t).astype(int)
            f = t - i
            p0, p1, p2, p3 = buf[i - 1], buf[i], buf[i + 1], buf[i + 2]
            a = p1
            b = 0.5 * (p2 - p0)
            c = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
            d = 0.5 * (p3 - p0) + 1.5 * (p1 - p2)
            y = ((d * f + c) * f + b) * f + a
        keep = len(buf) - keep_n
        self.hist = buf[keep:]
        self.pos = p - keep
        return y


class Shifter:
    """Pitch shift by hs/ha with duration preserved."""

    def __init__(self, ha, hs, n_fft=1024, quality="cubic"):
        self.pv = PVStretch(n_fft=n_fft, ha=ha, hs=hs)
        self.rs = Resampler(hs / ha, quality=quality)

    def process(self, block):
        return self.rs.process(self.pv.process(block))


def _hops(cents, ha=256):
    return ha, int(round(ha * 2 ** (cents / 1200.0)))


class StreamingChorus:
    """Billy's doubling: two pitch-shifted copies through fixed delay lines."""

    def __init__(self, sr, cents=(+26.0, -26.0), delays_ms=(8.0, 16.0),
                 amount=0.55, n_fft=1024, quality="cubic"):
        self.amount = float(amount)
        ha = n_fft // 4
        self.shifters = [Shifter(*_hops(c, ha), n_fft=n_fft, quality=quality)
                         for c in cents]
        self.delays = [int(round(d * sr / 1000.0)) for d in delays_ms]
        self.lines = [np.zeros(d) for d in self.delays]
        self.dry_line = np.zeros(n_fft)
        self.actual_cents = [1200 * np.log2(hs / a) for a, hs in
                             (_hops(c, ha) for c in cents)]

    def process(self, block):
        block = np.asarray(block, np.float64)
        parts = []
        for i, sh in enumerate(self.shifters):
            y = sh.process(block)
            if self.delays[i]:
                y = np.concatenate([self.lines[i], y])
                self.lines[i] = y[len(y) - self.delays[i]:]
                y = y[:len(y) - self.delays[i]]
            parts.append(y)
        n = min(len(p) for p in parts)
        if n == 0:
            self.dry_line = np.concatenate([self.dry_line, block])
            return np.zeros(0, np.float32)
        wet = sum(p[:n] for p in parts) / len(parts)
        for i, p in enumerate(parts):
            if len(p) > n:
                self.lines[i] = np.concatenate([p[n:], self.lines[i]])
        d = np.concatenate([self.dry_line, block])
        dry, self.dry_line = d[:n], d[n:]
        return ((1 - self.amount) * dry + self.amount * wet).astype(np.float32)


def _norm(x):
    p = float(np.abs(x).max())
    return x * (0.89 / p) if p > 0 else x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = StreamingChorus(48000)
    print(f"actual detune: {c.actual_cents[0]:+.1f} / {c.actual_cents[1]:+.1f} cents "
          f"(target +-26)\n")
    srcs = sorted((HERE / "out" / "refs").glob("*_A_plain.wav")) + \
        sorted((HERE / "out" / "tts_settled").glob("*_A_plain.wav"))

    print("chunk invariance:")
    for src in srcs:
        x, sr = sf.read(src, dtype="float32")
        outs = {}
        for label, step in (("whole", len(x)), ("333ms", int(sr * .333)), ("83ms", int(sr / 12))):
            ch = StreamingChorus(sr)
            outs[label] = np.concatenate(
                [ch.process(x[i:i + step]) for i in range(0, len(x), step)])
        n = min(len(v) for v in outs.values())
        d1 = float(np.abs(outs["whole"][:n] - outs["333ms"][:n]).max())
        d2 = float(np.abs(outs["whole"][:n] - outs["83ms"][:n]).max())
        stem = src.stem.replace("_A_plain", "")
        print(f"  {stem:16s} 333ms {d1:.2e}   83ms {d2:.2e}")
        sf.write(OUT / f"{stem}_STREAMSR.wav", _norm(outs["whole"]), sr)

    from families import load
    from grid import double
    print("\nstreaming vs approved offline (aligned, level-matched):")
    for src in srcs:
        x = load(src)
        off = np.asarray(double(x, src.stem, offsets=[+26, -26], delays=[8, 16],
                                amount=0.55), np.float32)
        st = StreamingChorus(48000).process(x)
        m = min(len(off), len(st))
        a, b = off[:m].astype(float), st[:m].astype(float)
        cc = np.correlate(a - a.mean(), b - b.mean(), "full")
        lag = int(np.argmax(np.abs(cc))) - (m - 1)
        aa, bb = (a[lag:], b[:m - lag]) if lag > 0 else (a[:m + lag], b[-lag:]) if lag < 0 else (a, b)
        k = min(len(aa), len(bb))
        aa, bb = aa[:k], bb[:k]
        s = np.dot(aa, bb) / max(np.dot(bb, bb), 1e-12)
        res = 20 * np.log10(np.sqrt(((aa - s * bb) ** 2).mean()) / np.sqrt((aa ** 2).mean()) + 1e-12)
        stem = src.stem.replace("_A_plain", "")
        sf.write(OUT / f"{stem}_OFFLINE.wav", _norm(off), 48000)
        print(f"  {stem:16s} lag {lag:6d}  corr {np.corrcoef(aa, bb)[0,1]:+.3f}  "
              f"residual {res:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
