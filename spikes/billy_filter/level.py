"""Back off the duplicate voice.

"A bit too much of the duplicate voice" is the wet/dry mix, not the detune or the delay --
those change the character of the copies, this changes how loud they sit against the dry.
Current setting is 70%. m040 was called too dry on game audio, so the answer is between.

Re-filters the ALREADY GENERATED clones rather than resynthesising, so the underlying
audio is byte-identical across levels and the only thing that varies is the filter. A
fresh generation would resample the model's sampling noise into the comparison.

Usage:
    .venv/bin/python spikes/billy_filter/level.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from families import load
from grid import double

HERE = Path(__file__).parent
OUT = HERE / "out" / "level"
SR = 48000
CENTS, DELAYS = 26, [8, 16]
LEVELS = [55, 62]

SOURCES = [
    *sorted((HERE / "out" / "refs").glob("*_A_plain.wav")),
    *sorted((HERE / "out" / "tts_settled").glob("*_A_plain.wav")),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        stem = src.stem.replace("_A_plain", "")
        x = load(src)
        p = float(np.abs(x).max()) or 1.0
        sf.write(OUT / f"{stem}_m000_dry.wav", x * (0.89 / p), SR)
        for lv in LEVELS:
            y = np.asarray(double(x, stem, offsets=[+CENTS, -CENTS], delays=DELAYS,
                                  amount=lv / 100.0), np.float32)
            q = float(np.abs(y).max()) or 1.0
            sf.write(OUT / f"{stem}_m{lv:03d}.wav", y * (0.89 / q), SR)
        m = len(x)
        print(f"  {stem:16s} {m/SR:5.2f}s  ->  " +
              "  ".join(f"m{lv:03d}" for lv in LEVELS) + "   (m070 is in refs/tts_settled)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
