"""Billy's filter, take 4: work the leads.

From the family round on A0_real_dry.wav:
  F5 detune/doubling  -- very promising, the lead
  F2 robotize, F3 comb -- right direction but too subtle at the strengths tried
  F1 ring mod         -- too robotic (Dalek)
  F4 comms            -- a DIFFERENT effect, sometimes layered on top of Billy, not the base
  F6 formant, F7 thin, F8 crush -- wrong

So: a proper sweep of the doubling family, stronger F2 and F3, and a few blends.

Note on the earlier evidence -- my partial-placement test reported no detuned layer, but
its resolution (IQR ~0.017 in frequency ratio) is wider than a subtle detune (18 cents =
0.010), so it could never have detected this. It was not sensitive enough to rule out the
one thing that now sounds right.

Usage:
    .venv/bin/python spikes/billy_filter/variations.py
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from families import f2_robotize, f3_comb, load, mix

OUT = Path(__file__).parent / "out"
SR = 48000


def detune(x, cents, amount, delay_ms=0.0, voices=2):
    """Detuned copies mixed against the dry. delay_ms > 0 makes it a chorus rather than
    a pure detune; voices=3 adds an undelayed centre copy."""
    out = np.zeros(len(x), np.float32)
    offsets = [+cents, -cents] if voices == 2 else [+cents, -cents, 0.0]
    for k, c in enumerate(offsets):
        y = librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100.0)
        if delay_ms:
            d = int(SR * delay_ms * (k + 1) / 1000)
            y = np.concatenate([np.zeros(d, np.float32), y])[:len(x)]
        n = min(len(y), len(out))
        out[:n] += y[:n] / len(offsets)
    return mix(x, out, amount)


VARIANTS = {
    # the doubling sweep -- weaker to stronger
    "G1_DETUNE_06c_40":  lambda x: detune(x, 6, 0.40),
    "G2_DETUNE_12c_50":  lambda x: detune(x, 12, 0.50),
    "G3_DETUNE_18c_60":  lambda x: detune(x, 18, 0.60),   # = the original F5
    "G4_DETUNE_28c_70":  lambda x: detune(x, 28, 0.70),
    "G5_DETUNE_40c_85":  lambda x: detune(x, 40, 0.85),
    "G6_DETUNE_60c_100": lambda x: detune(x, 60, 1.00),
    # structural variants on the same idea
    "G7_CHORUS_18c_d8":  lambda x: detune(x, 18, 0.70, delay_ms=8.0),
    "G8_TRIPLE_22c":     lambda x: detune(x, 22, 0.75, voices=3),
    # the two that needed more level
    "H1_ROBOTIZE_100":   lambda x: f2_robotize(x, amount=1.0),
    "H2_ROBOTIZE_x2":    lambda x: f2_robotize(f2_robotize(x, amount=1.0), amount=0.8),
    "I1_COMB_strong":    lambda x: f3_comb(x, delay_ms=1.6, fb=0.85, amount=0.95),
    "I2_COMB_tight":     lambda x: f3_comb(x, delay_ms=0.9, fb=0.80, amount=1.0),
    # blends -- the answer may not be one family
    "J1_DETUNE_ROBOT":   lambda x: f2_robotize(detune(x, 28, 0.70), amount=0.35),
    "J2_DETUNE_COMB":    lambda x: f3_comb(detune(x, 28, 0.70), delay_ms=1.6, fb=0.7, amount=0.4),
}


ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
BASES = ["GalGame_TheGoldenMechaGodBattle_Billy_401_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_404_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_405_006"]


def main():
    for stem in BASES:
        src = ZZZ / f"{stem}.wav"
        x = load(src)
        tag = stem.rsplit("_Billy_", 1)[1]
        d = OUT / f"base_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        line = (ZZZ / f"{stem}.txt").read_text().strip()
        print(f"\n=== base_{tag}  ({len(x)/SR:.2f}s)  \"{line[:72]}...\"")
        sf.write(d / "00_dry.wav", x * (0.89 / (np.abs(x).max() or 1)), SR)
        for name, fn in VARIANTS.items():
            y = np.asarray(fn(x), np.float32)
            p = float(np.abs(y).max())
            if p > 0:
                y = y * (0.89 / p)
            sf.write(d / f"{name}.wav", y, SR)
            m = min(len(x), len(y))
            a = x[:m] / (np.abs(x).max() or 1)
            b = y[:m] / (np.abs(y).max() or 1)
            ch = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
            print(f"  {name:20s} change {ch:+6.1f} dB")


if __name__ == "__main__":
    main()
