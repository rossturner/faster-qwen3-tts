"""Post-synthesis audio filters, declared per character.

Some characters are processed in their source material and do not sound right cloned
plain. Billy Kid is a cyborg whose in-game voice carries a doubling effect; his reference
recordings are the unprocessed studio takes, so the effect has to be reapplied here.

Reapplying it downstream is deliberate, not a shortcut. Baking it into the reference
clips instead would ask the model to reproduce a chorus as *timbre*, which it renders
inconsistently take to take -- the same mistake as cloning from already-processed audio.
Applied here it is deterministic and identical on every request.

The chorus is two pitch-shifted copies mixed against the dry through fixed delay lines,
with a makeup gain because that mix is not loudness-neutral: the copies are decorrelated,
so they sum as power rather than amplitude.
Pitch shifting is phase-vocoder time-stretch followed by fractional resampling, which
scales the whole spectrum including formants, so each copy reads as a slightly different
voice. Shifting bins instead approximately preserves formants and measurably sounds
flatter, so the two-stage form is load-bearing rather than incidental.

Both stages carry state between calls, so chunked and whole-file output are sample
identical and the streaming endpoint is free to chunk however it likes. The cost is one
window of algorithmic latency -- 42.7 ms -- which is constant, so it consumes TTFA
headroom without adding jitter, and is mostly hidden because a decoded chunk is normally
longer than the window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

CHORUS_KEYS = {"type", "cents", "delays_ms", "amount", "window_ms", "makeup_db"}


SOFT_CLIP_THRESHOLD = 0.9


def _princarg(p):
    return np.mod(p + np.pi, 2 * np.pi) - np.pi


def _soft_clip(x, threshold: float = SOFT_CLIP_THRESHOLD):
    """Memoryless soft knee, so restoring the level cannot clip on transients.

    Makeup gain pushes a few samples past full scale -- measured at 0.001-0.003% of them,
    peaking at 1.16 -- because the chorus raises crest factor as well as lowering RMS.
    Hard clipping those would be audible; this bends them instead. Memoryless matters:
    a look-ahead limiter would add latency and break chunk invariance.
    """
    over = np.abs(x) > threshold
    if not over.any():
        return x
    out = np.array(x, copy=True)
    excess = (np.abs(x[over]) - threshold) / (1.0 - threshold)
    out[over] = np.sign(x[over]) * (threshold + (1.0 - threshold) * np.tanh(excess))
    return out


@dataclass(frozen=True)
class ChorusSpec:
    """Declarative chorus settings. Defaults are Billy's, settled by ear."""
    cents: Tuple[float, ...] = (26.0, -26.0)
    delays_ms: Tuple[float, ...] = (8.0, 16.0)
    amount: float = 0.55
    # In milliseconds, NOT samples. The window's audible effect -- how much the vocoder
    # smears in time -- is a duration, so a fixed sample count means a different filter
    # at every sample rate. 2048 samples is 42.7 ms at 48 kHz but 85.3 ms at 24 kHz, and
    # those were heard as two different settings: the first was chosen, the second was
    # rejected as "too far apart".
    window_ms: float = 42.7
    # The copies are pitch-shifted, so they are decorrelated from the dry and from each
    # other: the mix sums as power, not amplitude, and loses level even though nothing is
    # attenuated. Measured at -5.66 dB RMS over seven real clones (tight, -5.44 to -5.93),
    # against -4.51 dB predicted by decorrelation alone -- the delay lines decorrelate
    # further. Without this Billy is audibly quieter than the unfiltered characters.
    #
    # Content-dependent (-5.1 dB on a harmonic probe, -6.0 on noise), so it is declared
    # rather than auto-calibrated. Re-measure if `amount` or the copy count changes:
    # spikes/billy_filter/ has the script.
    makeup_db: float = 5.7

    @classmethod
    def from_config(cls, data: Dict[str, Any], where: str) -> "ChorusSpec":
        unknown = set(data) - CHORUS_KEYS
        if unknown:
            raise ValueError(
                f"{where}: unknown filter key(s) {sorted(unknown)}; "
                f"allowed: {sorted(CHORUS_KEYS)}")
        kind = data.get("type", "chorus")
        if kind != "chorus":
            raise ValueError(f"{where}: unknown filter type {kind!r}; only 'chorus'")
        cents = tuple(float(c) for c in data.get("cents", cls.cents))
        delays = tuple(float(d) for d in data.get("delays_ms", cls.delays_ms))
        if not cents:
            raise ValueError(f"{where}: filter needs at least one entry in 'cents'")
        if len(cents) != len(delays):
            raise ValueError(
                f"{where}: 'cents' has {len(cents)} entries but 'delays_ms' has "
                f"{len(delays)}; there is one delay per copy")
        if any(d < 0 for d in delays):
            raise ValueError(f"{where}: 'delays_ms' must not be negative")
        amount = float(data.get("amount", cls.amount))
        if not 0.0 <= amount <= 1.0:
            raise ValueError(f"{where}: 'amount' must be between 0 and 1, got {amount}")
        window_ms = float(data.get("window_ms", cls.window_ms))
        if not 5.0 <= window_ms <= 500.0:
            raise ValueError(
                f"{where}: 'window_ms' must be between 5 and 500, got {window_ms}")
        makeup_db = float(data.get("makeup_db", cls.makeup_db))
        if not -24.0 <= makeup_db <= 24.0:
            raise ValueError(
                f"{where}: 'makeup_db' must be between -24 and 24, got {makeup_db}")
        return cls(cents, delays, amount, window_ms, makeup_db)

    def n_fft(self, sample_rate: int) -> int:
        """Nearest power of two to the requested window at this sample rate."""
        target = self.window_ms * sample_rate / 1000.0
        return max(256, int(2 ** round(np.log2(target))))

    def latency_ms(self, sample_rate: int) -> float:
        return self.n_fft(sample_rate) / sample_rate * 1000.0

    def build(self, sample_rate: int) -> "StreamingChorus":
        return StreamingChorus(sample_rate, self)


class _Stretch:
    """Phase-vocoder time-stretch by hs/ha, streaming."""

    def __init__(self, n_fft: int, ha: int, hs: int):
        self.n, self.ha, self.hs = n_fft, ha, hs
        self.win = np.hanning(n_fft + 1)[:n_fft]
        nb = n_fft // 2 + 1
        self.expected = 2 * np.pi * ha / n_fft * np.arange(nb)
        self.last_phase = np.zeros(nb)
        self.sum_phase = np.zeros(nb)
        self.inbuf = np.zeros(n_fft)
        self.outbuf = np.zeros(n_fft + hs)
        self.pending = np.zeros(0)
        acc = np.zeros(n_fft * 4)
        for m in range(0, n_fft * 3, hs):
            acc[m:m + n_fft] += self.win ** 2
        self.norm = float(np.mean(acc[n_fft:n_fft * 2]))

    def process(self, block):
        self.pending = np.concatenate([self.pending, block])
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


class _Resampler:
    """Catmull-Rom fractional resampler consuming `rate` input samples per output sample.

    Cubic rather than windowed sinc because at a ~1.5% rate change the two are
    indistinguishable by ear and cubic is 3.4x cheaper.
    """

    TAPS = 2

    def __init__(self, rate: float):
        self.rate = rate
        self.pos = 0.0
        self.hist = np.zeros(2 * self.TAPS)

    def process(self, block):
        buf = np.concatenate([self.hist, block])
        # a centre at t reads buf[i-1 .. i+2], so both ends need margin
        limit = len(buf) - 2 * self.TAPS
        positions = []
        p = self.pos
        while p < limit - 1:
            positions.append(p)
            p += self.rate
        keep_n = 2 * self.TAPS
        if not positions:
            self.hist = buf[-keep_n:] if len(buf) >= keep_n else buf
            self.pos = p - max(len(buf) - keep_n, 0)
            return np.zeros(0)
        t = np.asarray(positions) + self.TAPS
        i = np.floor(t).astype(int)
        f = t - i
        p0, p1, p2, p3 = buf[i - 1], buf[i], buf[i + 1], buf[i + 2]
        a = p1
        b = 0.5 * (p2 - p0)
        c = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
        d = 0.5 * (p3 - p0) + 1.5 * (p1 - p2)
        keep = len(buf) - keep_n
        self.hist = buf[keep:]
        self.pos = p - keep
        return ((d * f + c) * f + b) * f + a


class _Shifter:
    """Pitch shift by hs/ha with duration preserved."""

    def __init__(self, n_fft: int, ha: int, hs: int):
        self.stretch = _Stretch(n_fft, ha, hs)
        self.resample = _Resampler(hs / ha)

    def process(self, block):
        return self.resample.process(self.stretch.process(block))


class StreamingChorus:
    """Stateful chorus. Feed it blocks of any size; flush() releases the tail."""

    def __init__(self, sample_rate: int, spec: ChorusSpec = ChorusSpec()):
        self.spec = spec
        self.sample_rate = sample_rate
        self.n_fft = spec.n_fft(sample_rate)
        ha = self.n_fft // 4
        # Integer hops quantise the ratio slightly -- at ha=512, 26 cents becomes 26.8.
        # Far inside the resolution at which the value was chosen by ear.
        self.shifters = [_Shifter(self.n_fft, ha, max(1, round(ha * 2 ** (c / 1200.0))))
                         for c in spec.cents]
        self.delays = [int(round(d * sample_rate / 1000.0)) for d in spec.delays_ms]
        self.lines = [np.zeros(d) for d in self.delays]
        self.dry_line = np.zeros(self.n_fft)
        self.makeup = 10.0 ** (spec.makeup_db / 20.0)
        # The first window is the ramp-in, and is near silence on both paths. Dropping it
        # keeps output aligned with input: without this every line would gain n_fft
        # samples of leading silence on top of the wall-clock latency, paying for the
        # window twice.
        self._skip = self.n_fft

    def process(self, block) -> np.ndarray:
        block = np.asarray(block, dtype=np.float64)
        parts = []
        for i, shifter in enumerate(self.shifters):
            y = shifter.process(block)
            if self.delays[i]:
                y = np.concatenate([self.lines[i], y])
                self.lines[i] = y[len(y) - self.delays[i]:]
                y = y[:len(y) - self.delays[i]]
            parts.append(y)
        n = min(len(p) for p in parts)
        wet = sum(p[:n] for p in parts) / len(parts) if n else None
        # Re-queue whatever this call could not emit, before anything already in the
        # line. The voices run at different hop sizes, so one routinely produces samples
        # the other cannot yet be paired with -- dropping them loses audio.
        for i, p in enumerate(parts):
            if len(p) > n:
                self.lines[i] = np.concatenate([p[n:], self.lines[i]])
        held = np.concatenate([self.dry_line, block])
        if n == 0:
            self.dry_line = held
            return np.zeros(0, dtype=np.float32)
        dry, self.dry_line = held[:n], held[n:]
        out = ((1.0 - self.spec.amount) * dry + self.spec.amount * wet) * self.makeup
        out = _soft_clip(out)
        if self._skip:
            drop = min(self._skip, len(out))
            self._skip -= drop
            out = out[drop:]
        return out.astype(np.float32)

    def flush(self) -> np.ndarray:
        """Release the audio still inside the window. Call once, at end of stream."""
        return self.process(np.zeros(self.n_fft * 2, dtype=np.float32))

    def apply(self, audio) -> np.ndarray:
        """Whole-file convenience: filter and return the same number of samples."""
        audio = np.asarray(audio, dtype=np.float32)
        out = np.concatenate([self.process(audio), self.flush()])
        if len(out) < len(audio):
            out = np.pad(out, (0, len(audio) - len(out)))
        return out[:len(audio)].astype(np.float32)
