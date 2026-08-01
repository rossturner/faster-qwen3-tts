"""Billy's filter, take 8: sweep delay with detune locked at 26 cents.

Detune is settled at c26 -- not because it won convincingly, but because it could not be
told apart across 12..80 cents, so it is fixed at the mild lean and taken off the table.

Delay is swept wide (2..60 ms) rather than nudged, for the same reason c80 was tried:
narrow steps have been below the resolution of the comparison every round. At the long
end this stops being a chorus and becomes an audible slapback, which is the point -- an
obvious upper bound makes every later judgement cheaper.

Copies remain staggered at (ms, 2*ms), the anchor's arrangement. Wet/dry stays at 70%.

Usage:
    .venv/bin/python spikes/billy_filter/delay.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from families import load
from grid import double

OUT = Path(__file__).parent / "out" / "delay_c26"
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
SR = 48000
CENTS = 26
BASES = ["GalGame_TheGoldenMechaGodBattle_Billy_401_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_404_004",
         "GalGame_TheGoldenMechaGodBattle_Billy_405_006"]

DELAYS = [2, 5, 8, 14, 22, 35, 60]


def main():
    for stem in BASES:
        x = load(ZZZ / f"{stem}.wav")
        tag = stem.rsplit("_Billy_", 1)[1]
        d = OUT / f"base_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n=== base_{tag} ({len(x)/SR:.2f}s)")
        sf.write(d / "00_dry.wav", x * (0.89 / (np.abs(x).max() or 1)), SR)
        for ms in DELAYS:
            y = np.asarray(
                double(x, tag, offsets=[+CENTS, -CENTS], delays=[ms, 2 * ms]), np.float32)
            p = float(np.abs(y).max())
            if p > 0:
                y = y * (0.89 / p)
            sf.write(d / f"D_c26_d{ms:02d}.wav", y, SR)
            m = min(len(x), len(y))
            a = x[:m] / (np.abs(x).max() or 1)
            b = y[:m] / (np.abs(y).max() or 1)
            ch = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
            print(f"  D_c26_d{ms:02d}  change {ch:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
