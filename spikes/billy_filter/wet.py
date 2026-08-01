"""Billy's filter, take 9: the wet/dry level, with the shape settled.

Settled by ear over rounds 1-5, on the dry GoldenMechaGod bases:

    two detuned copies      (a single copy lost to the pair)
    +-26 cents              (indistinguishable from 12 to 80; set at the mild lean)
    8 and 16 ms, staggered  (d22 too much, d60 way too much, d02 acceptable)

Level is the last axis and the only one that has ever moved the change metric: delay and
detune both sat at -2.0..-2.7 dB across every setting tried, while wet/dry spans roughly
-7 to +2 dB. If anything about this effect is going to be audibly tunable, it is this.

Usage:
    .venv/bin/python spikes/billy_filter/wet.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from families import load
from grid import double

OUT = Path(__file__).parent / "out" / "wet_c26_d08"
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
SR = 48000
CENTS, DELAYS = 26, [8, 16]
BASES = ["GalGame_TheGoldenMechaGodBattle_Billy_401_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_404_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_405_006"]

AMOUNTS = [40, 55, 70, 85, 100]


def main():
    for stem in BASES:
        x = load(ZZZ / f"{stem}.wav")
        tag = stem.rsplit("_Billy_", 1)[1]
        d = OUT / f"base_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n=== base_{tag} ({len(x)/SR:.2f}s)")
        sf.write(d / "00_dry.wav", x * (0.89 / (np.abs(x).max() or 1)), SR)
        for a in AMOUNTS:
            y = np.asarray(double(x, tag, offsets=[+CENTS, -CENTS], delays=DELAYS,
                                  amount=a / 100.0), np.float32)
            p = float(np.abs(y).max())
            if p > 0:
                y = y * (0.89 / p)
            sf.write(d / f"W_c26_d08_m{a:03d}.wav", y, SR)
            m = min(len(x), len(y))
            aa = x[:m] / (np.abs(x).max() or 1)
            bb = y[:m] / (np.abs(y).max() or 1)
            ch = 20 * np.log10(np.sqrt(((bb - aa) ** 2).mean()) / np.sqrt((aa ** 2).mean()) + 1e-12)
            print(f"  W_c26_d08_m{a:03d}  change {ch:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
