"""Billy's robot filter: reconstruct it, and decide where to clone him from.

Billy's shipped references are all *processed*. The GoldenMechaGodBattle assets are the
same performer with the effect not applied, which makes a dry/wet comparison possible.
Differencing them (12 dry files vs 24 processed, identical 48 kHz PCM_16 containers with
a shared ~18.5 kHz codec cliff, so the pipeline is not the confound) says the effect:

  - cuts the low end          -7.4 dB below 100 Hz, -4.5 dB at 100-200 Hz
  - leaves 300-3000 Hz alone  within +-1 dB
  - adds a broad HF shelf     +5 dB at 4-5 kHz rising to a +8..+12 dB plateau over
                              6-12 kHz, still +11 dB at 14-16 kHz
  - the added energy is inharmonic and voice-gated, and halves pitch trackability
    (voiced frames 43.9% -> 23.9%, pyin confidence 0.086 -> 0.041)

and, measured and ruled out: no pitch or frequency shift (partials sit at identical
places, median ratio 0.9992 in both sets over ~40k measurements), no ring modulation,
vocoder or tremolo (no envelope-modulation carrier), no comb filtering, no clipping
(odd/even harmonic balance unchanged).

So two candidate reconstructions, which differ in mechanism and therefore in what
survives a 24 kHz codec:

  EQ        the measured magnitude difference as a linear-phase FIR. Boosts what is
            already there -- and the model's output has nothing above 12 kHz to boost.
  EXCITE    low-end cut, plus a synthesized inharmonic noise layer gated to the speech
            envelope and shaped to the measured *added* spectrum (Aw - Ad). Creates the
            band rather than amplifying it, so it works on 24 kHz model output.

Outputs, all 48 kHz. The A-series validates the filter model against ground truth; the
B-series is the actual pipeline decision.

Usage:
    .venv/bin/python spikes/billy_filter/billy_filter_spike.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import firwin2, lfilter, sosfilt, butter

from faster_qwen3_tts import FasterQwen3TTS

REPO = Path(__file__).resolve().parents[2]
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
DRY = ZZZ / "GalGame_TheGoldenMechaGodBattle_Billy_401_004.wav"
CHARS = REPO / "faster_qwen3_tts" / "server_voices" / "characters" / "billy"
OUT = Path(__file__).parent / "out"
SR = 48000

TEXT = ("Whoa, hold on a second! You're telling me this thing has been sitting here the "
        "whole time and nobody noticed? That is exactly like episode three seventy-two!")

# Measured processed-minus-dry response, at band centres. Above 18 kHz both sets hit the
# shared codec cliff, so the measurement there is meaningless -- taper it out instead.
RESPONSE_HZ = [0, 50, 150, 250, 400, 600, 850, 1250, 1750, 2500, 3500, 4500,
               5500, 7000, 9000, 11000, 13000, 15000, 17000, 18500, 21000, SR / 2]
RESPONSE_DB = [-7.4, -7.4, -4.5, -2.6, 0.8, 0.8, -1.2, -0.3, -0.9, -0.6, 1.7, 5.0,
               9.0, 8.1, 7.1, 11.9, 11.6, 10.7, 11.0, 8.0, 0.0, 0.0]


def design_fir(hz, db, ntaps=1025):
    freq = np.asarray(hz, float) / (SR / 2)
    freq[-1] = 1.0
    return firwin2(ntaps, freq, 10 ** (np.asarray(db, float) / 20))


def apply_fir(x, taps):
    y = lfilter(taps, [1.0], np.concatenate([x, np.zeros(len(taps) // 2)]))
    return y[len(taps) // 2:]


def to_48k(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    import librosa
    return librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR)


def band_energy(x, lo, hi):
    sos = butter(6, [lo / (SR / 2), min(hi, SR / 2 - 1) / (SR / 2)], "bandpass", output="sos")
    return float(np.sum(sosfilt(sos, x) ** 2))


def envelope(x, attack_ms=1.5, release_ms=12.0):
    """Speech envelope from the 300-4000 Hz band -- what the added layer is gated by."""
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


def eq_only(x):
    return apply_fir(x, design_fir(RESPONSE_HZ, RESPONSE_DB))


def excite(x, drive=1.0):
    """Low-end cut, plus inharmonic noise gated to the envelope and shaped to the
    measured ADDED spectrum, scaled so its 4-18 kHz energy matches the measurement."""
    low = apply_fir(x, design_fir(
        RESPONSE_HZ, [d if h < 3000 else 0.0 for h, d in zip(RESPONSE_HZ, RESPONSE_DB)]))
    # target added energy per band = dry_energy * (10^(diff/10) - 1)
    add_db = []
    for h, d in zip(RESPONSE_HZ, RESPONSE_DB):
        if h < 3500 or h > 18500:
            add_db.append(-90.0)
        else:
            add_db.append(10 * np.log10(max(10 ** (d / 10) - 1, 1e-9)))
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    layer = apply_fir(noise * envelope(x), design_fir(RESPONSE_HZ, add_db))
    ref = band_energy(x, 4000, 18000)
    got = band_energy(layer, 4000, 18000)
    want = sum(ref * (10 ** (d / 10) - 1) for h, d in zip(RESPONSE_HZ, RESPONSE_DB)
               if 3500 <= h <= 18500) / 7
    if got > 0:
        layer *= drive * np.sqrt(want / got)
    return low + layer


def save(name, x, note):
    x = np.asarray(x, np.float32)
    peak = float(np.abs(x).max())
    if peak > 0:
        x = x * (0.89 / peak)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / name, x, SR)
    print(f"  {name:42s} {len(x)/SR:5.2f}s   {note}")


def report(label, x):
    tot = band_energy(x, 200, 18000) + 1e-12
    print(f"     {label:26s} " + "  ".join(
        f"{lo//1000}-{hi//1000}k:{10*np.log10(band_energy(x,lo,hi)/tot+1e-12):+6.1f}"
        for lo, hi in [(200, 1000), (1000, 4000), (4000, 8000), (8000, 12000), (12000, 18000)]))


def main():
    dry, dsr = sf.read(DRY, dtype="float32")
    if dry.ndim > 1:
        dry = dry.mean(1)
    dry = to_48k(dry, dsr)
    ref_text = DRY.with_suffix(".txt").read_text().strip()
    # A processed line, from the source tree -- the shipped billy library is now
    # entirely dry (all 11 references are GoldenMechaGod), so it holds no wet example.
    wet_example = next(p for p in sorted(ZZZ.glob("*.wav"))
                       if "GoldenMechaGod" not in p.name and p.stat().st_size > 700_000)
    wet, wsr = sf.read(wet_example, dtype="float32")
    if wet.ndim > 1:
        wet = wet.mean(1)
    wet = to_48k(wet, wsr)

    print(f"base   : {DRY.name}")
    print(f"ref_text: {ref_text[:90]}...")
    print(f"\nA-series -- does the reconstruction match ground truth?")
    save("A0_real_dry.wav", dry, "the base recording, untouched (effect OFF)")
    save("A1_real_dry_EQ.wav", eq_only(dry), "base + measured EQ")
    save("A2_real_dry_EXCITE.wav", excite(dry), "base + low cut + synthesized HF layer")
    save("A3_real_processed.wav", wet, f"real game line WITH the effect ({wet_example.name})")
    print("   band balance, dB relative to 200 Hz-18 kHz total:")
    for lbl, sig in [("A0 real dry", dry), ("A1 dry+EQ", eq_only(dry)),
                     ("A2 dry+EXCITE", excite(dry)), ("A3 real processed", wet)]:
        report(lbl, sig)

    print(f"\nloading model...")
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)

    def clone(ref_audio, rtext):
        prompt = model.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio), ref_text=rtext, x_vector_only_mode=False)
        wavs, sr = model.generate_voice_clone(
            text=TEXT, language="English", voice_clone_prompt=prompt, temperature=0.7)
        return to_48k(np.asarray(wavs[0], np.float32), sr)

    print(f"\nB-series -- which pipeline should ship?")
    wet_text = wet_example.with_suffix(".txt").read_text().strip()
    b0 = clone(wet_example, wet_text)
    save("B0_clone_from_processed.wav", b0, "today's behaviour: cloned from a processed clip")
    b1 = clone(DRY, ref_text)
    save("B1_clone_from_dry.wav", b1, "cloned from the dry base, no filter")
    save("B2_clone_from_dry_EQ.wav", eq_only(b1), "cloned from dry + measured EQ")
    save("B3_clone_from_dry_EXCITE.wav", excite(b1), "cloned from dry + synthesized HF layer")
    save("B4_clone_from_dry_EXCITE_hot.wav", excite(b1, drive=1.8),
         "as B3 with the layer pushed +5 dB")
    print("   band balance, dB relative to 200 Hz-18 kHz total:")
    for lbl, sig in [("B0 from processed", b0), ("B1 from dry", b1),
                     ("B2 dry+EQ", eq_only(b1)), ("B3 dry+EXCITE", excite(b1))]:
        report(lbl, sig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
