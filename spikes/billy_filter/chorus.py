"""Billy's filter, take 5: work the chorus.

G7 (18 cents, 70% wet, copies delayed 8/16 ms) was the closest so far -- closer than the
pure detunes, which says the delay is contributing and not just the pitch offset.

This sweeps one dimension at a time around G7 so each axis can be judged independently,
and adds structural variants. Two of those are diagnostic rather than candidates:

  D5 delay with NO detune   } between them these say which of the two ingredients is
  D6 detune with NO delay   } actually carrying the effect. If D5 sounds right, this is
                              a doubler; if D6 does, the delay was incidental.

Filenames carry their parameters: d08 = 8 ms base delay, c18 = 18 cents, m70 = 70% wet.
A3, B3 and C3 are the same settings as G7 -- the anchor each sweep passes through.

Usage:
    .venv/bin/python spikes/billy_filter/chorus.py
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from families import load, mix

OUT = Path(__file__).parent / "out" / "chorus"
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
SR = 48000
BASES = ["GalGame_TheGoldenMechaGodBattle_Billy_401_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_404_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_405_006"]

_shift_cache: dict[tuple[int, float], np.ndarray] = {}


def shifted(x, cents, key):
    if cents == 0.0:
        return x
    ck = (key, cents)
    if ck not in _shift_cache:
        _shift_cache[ck] = librosa.effects.pitch_shift(x, sr=SR, n_steps=cents / 100.0)
    return _shift_cache[ck]


def mod_delay(x, base_ms, depth_ms=0.0, rate_hz=0.0, phase=0.0):
    """Fractional delay line, optionally swept by an LFO (a real chorus, not a doubler)."""
    n = len(x)
    d = np.full(n, base_ms, float)
    if depth_ms and rate_hz:
        t = np.arange(n) / SR
        d = d + depth_ms * np.sin(2 * np.pi * rate_hz * t + phase)
    pos = np.arange(n) - d * SR / 1000.0
    return np.interp(pos, np.arange(n), x, left=0.0, right=0.0).astype(np.float32)


def chorus(x, key, cents=18.0, delay_ms=8.0, amount=0.70, voices=2,
           depth_ms=0.0, rate_hz=0.0, feedback=0.0):
    offsets = [+cents, -cents] if voices == 2 else [+cents, -cents, 0.0]
    out = np.zeros(len(x), np.float32)
    for k, c in enumerate(offsets):
        y = shifted(x, c, key)[:len(x)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        if delay_ms or depth_ms:
            y = mod_delay(y, delay_ms * (k + 1), depth_ms, rate_hz,
                          phase=2 * np.pi * k / len(offsets))
        out[:len(y)] += y / len(offsets)
    if feedback:
        fb = int(SR * max(delay_ms, 1.0) / 1000)
        for i in range(fb, len(out)):
            out[i] += feedback * out[i - fb]
        out /= max(np.abs(out).max(), 1e-9) / max(np.abs(x).max(), 1e-9)
    return mix(x, out, amount)


VARIANTS = {
    # delay sweep, holding 18 cents / 70% wet
    "A1_d03_c18_m70": dict(delay_ms=3),
    "A2_d05_c18_m70": dict(delay_ms=5),
    "A3_d08_c18_m70": dict(delay_ms=8),                 # = G7, the anchor
    "A4_d12_c18_m70": dict(delay_ms=12),
    "A5_d18_c18_m70": dict(delay_ms=18),
    "A6_d28_c18_m70": dict(delay_ms=28),
    # detune sweep, holding 8 ms / 70% wet
    "B1_d08_c08_m70": dict(cents=8),
    "B2_d08_c12_m70": dict(cents=12),
    "B4_d08_c25_m70": dict(cents=25),
    "B5_d08_c35_m70": dict(cents=35),
    # wet/dry sweep, holding 8 ms / 18 cents
    "C1_d08_c18_m50": dict(amount=0.50),
    "C2_d08_c18_m60": dict(amount=0.60),
    "C4_d08_c18_m85": dict(amount=0.85),
    "C5_d08_c18_m100": dict(amount=1.00),
    # structural
    "D1_triple_c22": dict(cents=22, voices=3, amount=0.75),
    "D2_lfo_slow": dict(depth_ms=3.0, rate_hz=0.35),
    "D3_lfo_deep": dict(depth_ms=6.0, rate_hz=0.8, amount=0.8),
    "D4_feedback": dict(feedback=0.45, amount=0.8),
    "D5_delay_only": dict(cents=0.0, delay_ms=10, amount=0.7),
    "D6_detune_only": dict(delay_ms=0.0),
}


def main():
    for stem in BASES:
        x = load(ZZZ / f"{stem}.wav")
        tag = stem.rsplit("_Billy_", 1)[1]
        d = OUT / f"base_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n=== base_{tag} ({len(x)/SR:.2f}s)")
        sf.write(d / "00_dry.wav", x * (0.89 / (np.abs(x).max() or 1)), SR)
        for name, kw in VARIANTS.items():
            y = np.asarray(chorus(x, tag, **kw), np.float32)
            p = float(np.abs(y).max())
            if p > 0:
                y = y * (0.89 / p)
            sf.write(d / f"{name}.wav", y, SR)
            m = min(len(x), len(y))
            a = x[:m] / (np.abs(x).max() or 1)
            b = y[:m] / (np.abs(y).max() or 1)
            ch = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
            print(f"  {name:18s} change {ch:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
