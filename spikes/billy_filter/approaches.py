"""Billy's filter, take 2: six different mechanisms, all matched to the same measured
spectrum, so what you are judging is character rather than loudness.

Take 1 used broadband noise gated by a single wideband envelope. That is hiss laid over
the voice -- it does not track the voice in frequency, only in time, which is why it read
as "too heavily applied" on some phonemes. Every approach here follows the voice more
closely, or generates its high end from the voice itself.

The target is now a full-resolution LTAS ratio measured over 12 dry vs 60 processed
GalGame_ dialogue lines, rather than five bands.

  E1 EQ         linear filter only. No added inharmonicity -- the control that says how
                much of the effect is simply tone.
  E2 VOCODER    noise excited through the voice's OWN time-varying spectral envelope,
                high-passed and mixed. Follows the voice in time and frequency.
  E3 PHASE      magnitude kept exactly, phase randomised above the crossover. Destroys
                harmonic phase coherence without moving any partial -- which is what the
                measurements actually showed.
  E4 SHIFT      the 2-6 kHz band translated up into 7-18 kHz. Inharmonic by construction
                and perfectly voice-following.
  E5 EXCITE     waveshaped high-passed band -- an aural exciter. Generates its high end
                as harmonics of the voice, so it fuses rather than hisses.
  E6 CRUSH      sample-and-hold decimation residue. The classic digital-robot artifact.

Usage:
    .venv/bin/python spikes/billy_filter/approaches.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, firwin2, lfilter

HERE = Path(__file__).parent
OUT = HERE / "out"
CACHE = HERE / "ltas.npz"
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
SR = 48000
NFFT = 2048
HOP = 256
XOVER = 3500.0


def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x


def population_ltas():
    """Average spectrum of dry vs processed, each normalised to equal 200 Hz-18 kHz energy."""
    if CACHE.exists():
        d = np.load(CACHE)
        return d["freqs"], d["dry"], d["wet"]
    allw = sorted(glob.glob(str(ZZZ / "*.wav")))
    sets = {
        "dry": [p for p in allw if "GoldenMechaGod" in p],
        "wet": [p for p in allw if "GoldenMechaGod" not in p and "GalGame_" in Path(p).name][:60],
    }
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    band = (freqs >= 200) & (freqs <= 18000)
    out = {}
    for name, paths in sets.items():
        acc = np.zeros(len(freqs))
        for p in paths:
            S = np.abs(librosa.stft(load(p), n_fft=NFFT, hop_length=HOP))
            e = S.sum(0)
            keep = S[:, e > np.percentile(e, 65)]
            if not keep.size:
                continue
            P = (keep ** 2).mean(1)
            acc += P / P[band].sum()
        out[name] = acc / len(paths)
        print(f"  {name}: {len(paths)} files")
    np.savez(CACHE, freqs=freqs, dry=out["dry"], wet=out["wet"])
    return freqs, out["dry"], out["wet"]


def smooth_log(freqs, p, frac=0.12):
    """Smooth a spectrum on a log-frequency scale so the FIR follows the trend, not
    this speaker's particular formants."""
    out = np.copy(p)
    for i, f in enumerate(freqs):
        if f < 40:
            continue
        m = (freqs > f * (1 - frac)) & (freqs < f * (1 + frac))
        if m.sum() > 1:
            out[i] = p[m].mean()
    return out


def fir_from_curve(freqs, gain_db, ntaps=1025):
    hz, db = list(freqs[::8]), list(gain_db[::8])
    if hz[-1] < SR / 2:
        hz.append(SR / 2)
        db.append(db[-1])
    return firwin2(ntaps, hz, 10 ** (np.asarray(db) / 20), fs=SR)


def apply_fir(x, taps):
    y = lfilter(taps, [1.0], np.concatenate([x, np.zeros(len(taps) // 2)]))
    return y[len(taps) // 2:].astype(np.float32)


def hp(x, f):
    return sosfilt(butter(6, f / (SR / 2), "highpass", output="sos"), x).astype(np.float32)


def bp(x, lo, hi):
    sos = butter(6, [lo / (SR / 2), min(hi, SR / 2 - 1) / (SR / 2)], "bandpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


# ---------------------------------------------------------------- generators
def gen_vocoder(x, rng):
    S = librosa.stft(x, n_fft=NFFT, hop_length=HOP)
    mag = np.abs(S)
    env = np.copy(mag)
    for i in range(env.shape[1]):                    # smooth across frequency = envelope
        env[:, i] = np.convolve(mag[:, i], np.ones(9) / 9, mode="same")
    phase = np.exp(2j * np.pi * rng.random(env.shape))
    y = librosa.istft(env * phase, hop_length=HOP, n_fft=NFFT, length=len(x))
    return hp(y, XOVER)


def gen_phase(x, rng):
    S = librosa.stft(x, n_fft=NFFT, hop_length=HOP)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    S2 = np.copy(S)
    hi = freqs >= XOVER
    S2[hi] = np.abs(S[hi]) * np.exp(2j * np.pi * rng.random(S[hi].shape))
    y = librosa.istft(S2, hop_length=HOP, n_fft=NFFT, length=len(x))
    return hp(y, XOVER)


def gen_shift(x, rng):
    """Translate the 2-6 kHz band up into 7-18 kHz: inharmonic, perfectly voice-following."""
    S = librosa.stft(x, n_fft=NFFT, hop_length=HOP)
    mag = np.abs(S)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    src = np.where((freqs >= 2000) & (freqs <= 6000))[0]
    out = np.zeros_like(mag)
    for mult, gain in ((2.6, 1.0), (3.9, 0.6)):
        dst = np.clip((src * mult).astype(int), 0, len(freqs) - 1)
        np.add.at(out, dst, mag[src] * gain)
    phase = np.exp(2j * np.pi * rng.random(out.shape))
    y = librosa.istft(out * phase, hop_length=HOP, n_fft=NFFT, length=len(x))
    return hp(y, XOVER)


def gen_excite(x, rng):
    band = bp(x, 2000, 7000)
    peak = np.abs(band).max() or 1.0
    shaped = np.tanh(6.0 * band / peak) * peak
    return hp(shaped - band, XOVER)


def gen_crush(x, rng):
    hold = 3                                          # 48 kHz / 3 = 16 kHz sample-and-hold
    y = np.repeat(x[::hold], hold)[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return hp(y - x, XOVER)


GENERATORS = {
    "E2_VOCODER": gen_vocoder,
    "E3_PHASE": gen_phase,
    "E4_SHIFT": gen_shift,
    "E5_EXCITE": gen_excite,
    "E6_CRUSH": gen_crush,
}


def main():
    freqs, dry_p, wet_p = population_ltas()
    ratio_db = 10 * np.log10(smooth_log(freqs, wet_p) / (smooth_log(freqs, dry_p) + 1e-20) + 1e-20)
    ratio_db = np.clip(ratio_db, -12, 14)
    ratio_db[freqs > 18500] = 0.0
    eq_taps = fir_from_curve(freqs, ratio_db)

    print("\nmeasured processed-minus-dry response (12 dry vs 60 processed lines):")
    for f in (60, 120, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 13000, 16000, 18000):
        print(f"   {f:6d} Hz  {ratio_db[np.argmin(np.abs(freqs - f))]:+6.2f} dB")

    # the added-only component, as a spectrum to shape each generator's layer with
    add_p = np.maximum(smooth_log(freqs, wet_p) - smooth_log(freqs, dry_p), 0.0)
    add_db = 10 * np.log10(add_p / (add_p.max() + 1e-20) + 1e-12)
    add_db = np.clip(add_db, -60, 0)
    add_db[freqs < XOVER] = -60
    add_db[freqs > 18500] = -60
    add_taps = fir_from_curve(freqs, add_db)
    want_ratio = add_p[(freqs >= XOVER) & (freqs <= 18500)].sum() / \
        smooth_log(freqs, dry_p)[(freqs >= 200) & (freqs <= 18500)].sum()

    low = fir_from_curve(freqs, np.where(freqs < 400, ratio_db, 0.0))
    x = load(OUT / "A0_real_dry.wav")
    rng = np.random.default_rng(3)

    def energy(sig, lo, hi):
        return float(np.sum(bp(sig, lo, hi) ** 2))

    def save(name, y, note):
        y = np.asarray(y, np.float32)
        p = float(np.abs(y).max())
        if p > 0:
            y = y * (0.89 / p)
        sf.write(OUT / f"{name}.wav", y, SR)
        tot = energy(y, 200, 18000) + 1e-12
        bal = "  ".join(f"{lo//1000}-{hi//1000}k:{10*np.log10(energy(y,lo,hi)/tot+1e-12):+6.1f}"
                        for lo, hi in [(200,1000),(1000,4000),(4000,8000),(8000,12000),(12000,18000)])
        print(f"  {name:22s} {note}")
        print(f"     {bal}")

    print("\nE-series, all applied to A0_real_dry.wav")
    save("E1_EQ", apply_fir(x, eq_taps), "linear EQ only (no added inharmonicity)")

    base = apply_fir(x, low)
    ref = energy(base, 200, 18500)
    for name, gen in GENERATORS.items():
        layer = apply_fir(gen(x, rng), add_taps)
        got = energy(layer, XOVER, 18500)
        if got > 0:
            layer = layer * np.sqrt(want_ratio * ref / got)
        save(name, base + layer, gen.__doc__ or name)
    print(f"\ncompare against A3_real_processed.wav")


if __name__ == "__main__":
    main()
