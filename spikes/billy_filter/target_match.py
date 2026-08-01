"""Target-matched Billy filter: shape the added layer so the OUTPUT band balance lands
on the measured processed-Billy median, instead of trusting an open-loop boost curve.

The open-loop version (billy_filter_spike.py, B3) overshot the top octave by ~13 dB
because it applied a measured *boost* to a signal that had almost nothing there to
boost. This closes the loop: measure the target, measure the input, add the difference.

Target is the median band balance of 60 processed GalGame_ dialogue lines; 90-93% of
them sit above the dry median in every band over 4 kHz, so the separation is a real
property of the effect and not an averaging artifact.

Usage:
    .venv/bin/python spikes/billy_filter/target_match.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, firwin2, lfilter

OUT = Path(__file__).parent / "out"
SR = 48000
BANDS = [(200, 1000), (1000, 4000), (4000, 8000), (8000, 12000), (12000, 18000)]

# median band balance, dB relative to the 200 Hz-18 kHz total
TARGET_WET = np.array([-1.8, -6.2, -13.0, -16.5, -22.6])
TARGET_DRY = np.array([-2.0, -4.6, -17.6, -23.1, -29.5])


def band_energies(x):
    out = []
    for lo, hi in BANDS:
        sos = butter(6, [lo / (SR / 2), hi / (SR / 2)], "bandpass", output="sos")
        out.append(float(np.sum(sosfilt(sos, x) ** 2)))
    return np.array(out)


def balance_db(x):
    e = band_energies(x)
    return 10 * np.log10(e / e.sum() + 1e-20)


def envelope(x, attack_ms=1.5, release_ms=12.0):
    sos = butter(4, [300 / (SR / 2), 4000 / (SR / 2)], "bandpass", output="sos")
    rect = np.abs(sosfilt(sos, x))
    a_a = np.exp(-1.0 / (SR * attack_ms / 1000))
    a_r = np.exp(-1.0 / (SR * release_ms / 1000))
    env = np.empty_like(rect)
    prev = 0.0
    for i, v in enumerate(rect):
        coef = a_a if v > prev else a_r
        prev = coef * prev + (1 - coef) * v
        env[i] = prev
    return env


def band_noise(env, lo, hi, seed):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(env)).astype(np.float32) * env
    sos = butter(8, [lo / (SR / 2), min(hi, SR / 2 - 1) / (SR / 2)], "bandpass", output="sos")
    return sosfilt(sos, n).astype(np.float32)


def low_cut(x):
    taps = firwin2(513, [0, 50, 150, 250, 400, SR / 2],
                   10 ** (np.array([-7.4, -7.4, -4.5, -2.6, 0.0, 0.0]) / 20), fs=SR)
    y = lfilter(taps, [1.0], np.concatenate([x, np.zeros(len(taps) // 2)]))
    return y[len(taps) // 2:].astype(np.float32)


def match(x, target_db=TARGET_WET, amount=1.0):
    """Add envelope-gated inharmonic noise per band so the output balance hits target."""
    x = low_cut(x)
    e = band_energies(x)
    r = 10 ** (target_db / 10)
    r = r / r.sum()
    scale = np.max(e / r)                       # no band may need a cut
    added = np.maximum(scale * r - e, 0.0) * amount
    y = x.copy()
    for i, ((lo, hi), a) in enumerate(zip(BANDS, added)):
        if a <= 0 or lo < 4000:                 # only synthesise above 4 kHz
            continue
        layer = band_noise(envelope(x), lo, hi, seed=11 + i)
        got = float(np.sum(layer ** 2))
        if got > 0:
            y += layer * np.sqrt(a / got)
    return y


def save(name, x, note):
    x = np.asarray(x, np.float32)
    p = float(np.abs(x).max())
    if p > 0:
        x = x * (0.89 / p)
    sf.write(OUT / name, x, SR)
    print(f"  {name:38s} {note}")
    print(f"     balance " + "  ".join(f"{v:+6.1f}" for v in balance_db(x)))


def main():
    print("     bands   " + "  ".join(f"{lo//1000}-{hi//1000}k".rjust(6) for lo, hi in BANDS))
    print("     TARGET (real processed Billy, n=60)")
    print("     target  " + "  ".join(f"{v:+6.1f}" for v in TARGET_WET))
    print("     real dry population median (n=12)")
    print("     dry     " + "  ".join(f"{v:+6.1f}" for v in TARGET_DRY))

    b1, sr = sf.read(OUT / "B1_clone_from_dry.wav", dtype="float32")
    assert sr == SR
    print("\nC-series -- target-matched, applied to the dry-cloned TTS output")
    save("C0_tts_dry_unfiltered.wav", b1, "no filter (same as B1, for level-matched A/B)")
    save("C1_tts_matched.wav", match(b1), "matched to the processed-Billy median")
    save("C2_tts_matched_subtle.wav", match(b1, amount=0.5), "half the added layer")
    save("C3_tts_matched_hot.wav", match(b1, amount=1.8), "1.8x the added layer")

    a0, sr = sf.read(OUT / "A0_real_dry.wav", dtype="float32")
    print("\nD-series -- the same filter on the REAL dry recording, to check it by ear")
    save("D1_real_dry_matched.wav", match(a0), "real dry recording, filter applied")
    print(f"\ncompare D1 against A3_real_processed.wav (genuine in-game effect)")


if __name__ == "__main__":
    main()
