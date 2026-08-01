"""Close the last of the gap to the offline filter.

Take 3 is "better but still a bit flatter". Spectral balance and envelope variation are
already within 0.1-0.9 dB of offline, so the remaining difference is the quality of the
pitch shift itself, not tone or level.

Prime suspect: window size. librosa's pitch_shift defaults to n_fft=2048; take 3 used
1024. At 48 kHz that is 23 Hz bins against 47 Hz -- half the frequency resolution, which
smears partials and is exactly the kind of loss that reads as "flatter".

Control: resampler quality. At a 1.5% rate change cubic interpolation should be
transparent, so if sinc changes nothing that suspect is eliminated rather than assumed.

Latency is the cost of a longer window, and it is charged against the measured 260-480 ms
TTFA budget at the SERVER's 24 kHz, not the 48 kHz these files use:

    n_fft 1024 -> 42.7 ms      2048 -> 85.3 ms      4096 -> 170.7 ms

Usage:
    .venv/bin/python spikes/billy_filter/stream_variants.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from families import load
from grid import double
from stream_sr import StreamingChorus

HERE = Path(__file__).parent
OUT = HERE / "out" / "stream_variants"
SR = 48000

VARIANTS = {
    "V1_n1024_cubic": dict(n_fft=1024, quality="cubic"),   # = take 3
    "V2_n2048_cubic": dict(n_fft=2048, quality="cubic"),   # librosa's resolution
    "V3_n2048_sinc": dict(n_fft=2048, quality="sinc"),
    "V4_n4096_sinc": dict(n_fft=4096, quality="sinc"),
}


def _norm(x):
    p = float(np.abs(x).max())
    return x * (0.89 / p) if p > 0 else x


def align_residual(a, b):
    m = min(len(a), len(b))
    a, b = a[:m].astype(float), b[:m].astype(float)
    c = np.correlate(a - a.mean(), b - b.mean(), "full")
    lag = int(np.argmax(np.abs(c))) - (m - 1)
    aa, bb = (a[lag:], b[:m - lag]) if lag > 0 else \
             (a[:m + lag], b[-lag:]) if lag < 0 else (a, b)
    k = min(len(aa), len(bb))
    aa, bb = aa[:k], bb[:k]
    s = np.dot(aa, bb) / max(np.dot(bb, bb), 1e-12)
    res = 20 * np.log10(np.sqrt(((aa - s * bb) ** 2).mean()) /
                        np.sqrt((aa ** 2).mean()) + 1e-12)
    return np.corrcoef(aa, bb)[0, 1], res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = sorted((HERE / "out" / "refs").glob("*_A_plain.wav")) + \
        sorted((HERE / "out" / "tts_settled").glob("*_A_plain.wav"))
    print(f"{'variant':16s} {'latency@24k':>12s}  {'corr vs offline':>16s}  "
          f"{'residual':>9s}  chunk-invariant")
    for name, kw in VARIANTS.items():
        corrs, ress, inv = [], [], True
        for src in srcs:
            x = load(src)
            off = np.asarray(double(x, src.stem, offsets=[+26, -26], delays=[8, 16],
                                    amount=0.55), np.float32)
            st = StreamingChorus(SR, **kw).process(x)
            c, r = align_residual(off, st)
            corrs.append(c)
            ress.append(r)
            ch = StreamingChorus(SR, **kw)
            step = int(SR * 0.333)
            chunked = np.concatenate([ch.process(x[i:i + step])
                                      for i in range(0, len(x), step)])
            n = min(len(st), len(chunked))
            if float(np.abs(st[:n] - chunked[:n]).max()) > 1e-6:
                inv = False
            stem = src.stem.replace("_A_plain", "")
            sf.write(OUT / f"{stem}__{name}.wav", _norm(st), SR)
            if name == "V1_n1024_cubic":
                sf.write(OUT / f"{stem}__OFFLINE.wav", _norm(off), SR)
        lat = kw["n_fft"] / 24000 * 1000
        print(f"  {name:14s} {lat:9.1f} ms  {np.mean(corrs):+15.3f}  "
              f"{np.mean(ress):+8.1f} dB  {'yes' if inv else 'NO'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
