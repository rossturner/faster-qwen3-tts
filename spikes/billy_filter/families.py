"""Billy's filter, take 3: stop being precise about the wrong thing.

Takes 1 and 2 were measurement-led. Six mechanisms matched to the measured spectrum were
all reported inaudible, which means the dry/processed spectral difference I measured is
not what makes Billy sound robotic. Measuring harder will not fix that -- the gap is
between "modest spectral difference" and "categorically different voice".

So this casts a wide net over effect FAMILIES instead, each at obvious strength. The
question is no longer "how much" but "which kind". Once a family is identified, it can be
dialled back to taste and matched to the measurements properly.

Nothing here is derived from the dry/processed comparison. These are the standard robot
palette, applied to the dry base at strengths chosen to be unmistakable.

Usage:
    .venv/bin/python spikes/billy_filter/families.py
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, firwin2, lfilter

OUT = Path(__file__).parent / "out"
SR = 48000
NFFT, HOP = 1024, 256


def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x


def mix(dry, wet, amount):
    n = min(len(dry), len(wet))
    return (1 - amount) * dry[:n] + amount * wet[:n]


def f1_ringmod(x, fc=110.0, amount=0.55):
    """Ring modulation -- the canonical Dalek/robot effect."""
    t = np.arange(len(x)) / SR
    return mix(x, x * np.cos(2 * np.pi * fc * t).astype(np.float32), amount)


def f2_robotize(x, amount=0.8):
    """Zeroed STFT phase: the classic 'robotize'. Forces a constant buzz pitch."""
    S = librosa.stft(x, n_fft=NFFT, hop_length=HOP)
    y = librosa.istft(np.abs(S).astype(complex), hop_length=HOP, n_fft=NFFT, length=len(x))
    return mix(x, y.astype(np.float32), amount)


def f3_comb(x, delay_ms=1.6, fb=0.72, amount=0.7):
    """Short fixed comb -- metallic / helmet resonance."""
    d = int(SR * delay_ms / 1000)
    y = np.copy(x)
    for i in range(d, len(x)):
        y[i] += fb * y[i - d]
    y /= max(np.abs(y).max(), 1e-9)
    return mix(x, y.astype(np.float32) * np.abs(x).max(), amount)


def f4_comms(x, amount=0.85):
    """Radio/intercom: narrow band plus soft clipping."""
    sos = butter(6, [400 / (SR / 2), 3600 / (SR / 2)], "bandpass", output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    p = np.abs(y).max() or 1.0
    y = np.tanh(3.5 * y / p) * p
    return mix(x, y, amount)


def f5_detune(x, cents=18.0, amount=0.6):
    """Two detuned copies -- chorus/doubling, the 'not quite one voice' cue."""
    out = np.zeros(len(x), np.float32)
    for c in (+cents, -cents):
        y = librosa.effects.pitch_shift(x, sr=SR, n_steps=c / 100.0)
        n = min(len(y), len(out))
        out[:n] += y[:n] * 0.5
    return mix(x, out, amount)


def f6_formant(x, shift=1.18, amount=0.9):
    """Formant shift with pitch untouched -- changes apparent vocal-tract size."""
    S = librosa.stft(x, n_fft=NFFT, hop_length=HOP)
    mag, ph = np.abs(S), np.angle(S)
    idx = np.clip((np.arange(mag.shape[0]) / shift).astype(int), 0, mag.shape[0] - 1)
    y = librosa.istft(mag[idx] * np.exp(1j * ph), hop_length=HOP, n_fft=NFFT, length=len(x))
    return mix(x, y.astype(np.float32), amount)


def f7_thin(x, amount=1.0):
    """Heavy low cut plus presence lift -- an exaggerated version of the measured EQ."""
    taps = firwin2(1025, [0, 100, 200, 300, 500, 900, 2000, 4000, 8000, 14000, SR / 2],
                   10 ** (np.array([-24, -22, -16, -10, -4, 0, 3, 7, 8, 6, 0]) / 20), fs=SR)
    y = lfilter(taps, [1.0], np.concatenate([x, np.zeros(512)]))[512:]
    return mix(x, y.astype(np.float32), amount)


def f8_bitcrush(x, bits=6, hold=6, amount=0.7):
    """Quantisation plus sample-and-hold -- overt digital degradation."""
    y = np.repeat(x[::hold], hold)[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    q = 2 ** (bits - 1)
    y = np.round(y * q) / q
    return mix(x, y.astype(np.float32), amount)


FAMILIES = {
    "F1_RINGMOD": f1_ringmod,
    "F2_ROBOTIZE": f2_robotize,
    "F3_COMB": f3_comb,
    "F4_COMMS": f4_comms,
    "F5_DETUNE": f5_detune,
    "F6_FORMANT": f6_formant,
    "F7_THIN": f7_thin,
    "F8_BITCRUSH": f8_bitcrush,
}


def main():
    x = load(OUT / "A0_real_dry.wav")
    print("F-series -- effect families at obvious strength, on A0_real_dry.wav\n")
    for name, fn in FAMILIES.items():
        y = np.asarray(fn(x), np.float32)
        p = float(np.abs(y).max())
        if p > 0:
            y = y * (0.89 / p)
        sf.write(OUT / f"{name}.wav", y, SR)
        m = min(len(x), len(y))
        a = x[:m] / (np.abs(x).max() or 1)
        b = y[:m] / (np.abs(y).max() or 1)
        d = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
        print(f"  {name:14s} change {d:+6.1f} dB   {fn.__doc__.splitlines()[0]}")
    print(f"\nreference: A0_real_dry.wav (base) and A3_real_processed.wav (the real effect)")


if __name__ == "__main__":
    main()
