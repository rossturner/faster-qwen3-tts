"""Billy's filter, take 6: optimise delay and detune together.

D5 (delay, no detune) and D6 (detune, no delay) both lost to A3 (8 ms + 18 cents), so the
effect needs both ingredients and the two axes cannot be tuned separately -- hence a grid
rather than more 1D sweeps. A3 sits at the centre of it.

Wet/dry is held at 70% throughout. It is the one axis measured to move magnitude rather
than character (the C sweep in chorus.py spans -7.0 to +1.6 dB while delay and detune
both stay flat at about -2.4 dB), so it is the last thing to set, not the first.

Three structural questions the 1D sweeps could not ask:

  X1  both copies at the SAME delay, instead of staggered 8/16 ms
  X2  asymmetric detune (+9/-27 rather than +-18) -- pulls the centre of mass down
  X3  a SINGLE detuned copy rather than a pair

Usage:
    .venv/bin/python spikes/billy_filter/grid.py
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from families import load, mix

OUT = Path(__file__).parent / "out" / "grid"
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
SR = 48000
BASES = ["GalGame_TheGoldenMechaGodBattle_Billy_401_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_404_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_405_006"]

DELAYS = [5, 8, 12]
CENTS = [12, 18, 26]
AMOUNT = 0.70

_cache: dict[tuple[str, float], np.ndarray] = {}


def shifted(x, cents, key):
    if cents == 0.0:
        return x
    ck = (key, round(cents, 3))
    if ck not in _cache:
        _cache[ck] = librosa.effects.pitch_shift(x, sr=SR, n_steps=cents / 100.0)
    return _cache[ck]


def delayed(y, ms):
    d = int(SR * ms / 1000)
    return np.concatenate([np.zeros(d, np.float32), y])[:len(y)] if d else y


def double(x, key, offsets, delays, amount=AMOUNT):
    """offsets in cents, delays in ms, one per copy."""
    out = np.zeros(len(x), np.float32)
    for c, ms in zip(offsets, delays):
        y = shifted(x, c, key)[:len(x)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        out += delayed(y, ms) / len(offsets)
    return mix(x, out, amount)


def build():
    v = {}
    for ms in DELAYS:
        for c in CENTS:
            tag = f"G_d{ms:02d}_c{c:02d}"
            if ms == 8 and c == 18:
                tag += "_ANCHOR"
            v[tag] = dict(offsets=[+c, -c], delays=[ms, 2 * ms])
    v["X1_same_delay"] = dict(offsets=[+18, -18], delays=[8, 8])
    v["X2_asym_detune"] = dict(offsets=[+9, -27], delays=[8, 16])
    v["X3_single_copy"] = dict(offsets=[-18], delays=[8])
    return v


def main():
    variants = build()
    for stem in BASES:
        x = load(ZZZ / f"{stem}.wav")
        tag = stem.rsplit("_Billy_", 1)[1]
        d = OUT / f"base_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n=== base_{tag} ({len(x)/SR:.2f}s)")
        sf.write(d / "00_dry.wav", x * (0.89 / (np.abs(x).max() or 1)), SR)
        for name, kw in variants.items():
            y = np.asarray(double(x, tag, **kw), np.float32)
            p = float(np.abs(y).max())
            if p > 0:
                y = y * (0.89 / p)
            sf.write(d / f"{name}.wav", y, SR)
            m = min(len(x), len(y))
            a = x[:m] / (np.abs(x).max() or 1)
            b = y[:m] / (np.abs(y).max() or 1)
            ch = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
            print(f"  {name:22s} change {ch:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
