"""Billy's filter, take 7: bracket the detune depth instead of refining it.

c12 vs c18 vs c26 at 8 ms came back as "hard to tell, leaning c26". A weak preference
across a 14-cent span means the steps are below the resolution of the comparison, so
narrowing further just produces coin-flips.

Bracketing instead: push the detune well past anything plausible until it is obviously
wrong, then bisect back toward the anchor. An audible upper bound is worth more than
another inaudible step, because every later judgement can be made against it.

Delay stays at 8 ms and wet/dry at 70% -- unchanged from the anchor, so detune is still
the only variable.

Usage:
    .venv/bin/python spikes/billy_filter/ext.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from families import load
from grid import double

OUT = Path(__file__).parent / "out" / "grid_ext"
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
SR = 48000
BASES = ["GalGame_TheGoldenMechaGodBattle_Billy_401_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_404_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_405_006"]

CENTS = [34, 45, 60, 80]


def main():
    for stem in BASES:
        x = load(ZZZ / f"{stem}.wav")
        tag = stem.rsplit("_Billy_", 1)[1]
        d = OUT / f"base_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n=== base_{tag} ({len(x)/SR:.2f}s)")
        sf.write(d / "00_dry.wav", x * (0.89 / (np.abs(x).max() or 1)), SR)
        for c in CENTS:
            y = np.asarray(double(x, tag, offsets=[+c, -c], delays=[8, 16]), np.float32)
            p = float(np.abs(y).max())
            if p > 0:
                y = y * (0.89 / p)
            sf.write(d / f"E_d08_c{c:02d}.wav", y, SR)
            m = min(len(x), len(y))
            a = x[:m] / (np.abs(x).max() or 1)
            b = y[:m] / (np.abs(y).max() or 1)
            ch = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
            print(f"  E_d08_c{c:02d}  change {ch:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
