"""Billy's filter, take 10: the settled doubling on TTS output.

Settled by ear on the dry GoldenMechaGod recordings:

    two detuned copies, +-26 cents, at 8 and 16 ms, 70% wet

That was tuned on game audio. This is the case that has to work: the model's output has
less high end, a different noise floor and its own artifacts, so the doubling is not
guaranteed to behave the same way on it. Each line is written twice, unfiltered and
filtered, so the filter is judged rather than the clone.

The clone reference is the dry 401_004 recording -- Billy's shipped library is now
entirely dry, so this is the real production path, not a spike-only shortcut.

Usage:
    .venv/bin/python spikes/billy_filter/tts_apply.py
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS
from grid import double

REPO = Path(__file__).resolve().parents[2]
ZZZ = Path("/mnt/d/workspace/zzz_audio_v3/characters_long/Billy")
REF = ZZZ / "GalGame_TheGoldenMechaGodBattle_Billy_401_004.wav"
OUT = Path(__file__).parent / "out" / "tts_settled"
SR = 48000
CENTS, DELAYS, AMOUNT = 26, [8, 16], 0.70

LINES = {
    "L1_calm": "Boss, I've been thinking about that mech we saw yesterday, and I'm "
               "almost certain I've seen it somewhere before.",
    "L2_excited": "Whoa, hold on! That's the exact move from episode three seventy-two! "
                  "I can't believe you actually pulled it off!",
    "L3_flat": "Understood. I'll wait here and keep an eye on the entrance until you "
               "get back.",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref_text = REF.with_suffix(".txt").read_text().strip()
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)
    prompt = model.model.create_voice_clone_prompt(
        ref_audio=str(REF), ref_text=ref_text, x_vector_only_mode=False)

    def save(name, x):
        p = float(np.abs(x).max())
        sf.write(OUT / f"{name}.wav", x * (0.89 / p) if p > 0 else x, SR)

    for name, text in LINES.items():
        wavs, sr = model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=prompt, temperature=0.7)
        x = np.asarray(wavs[0], np.float32)
        x = librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x
        y = np.asarray(double(x, name, offsets=[+CENTS, -CENTS], delays=DELAYS,
                              amount=AMOUNT), np.float32)
        save(f"{name}_A_plain", x)
        save(f"{name}_B_billy", y)
        m = min(len(x), len(y))
        a, b = x[:m] / np.abs(x).max(), y[:m] / np.abs(y).max()
        ch = 20 * np.log10(np.sqrt(((b - a) ** 2).mean()) / np.sqrt((a ** 2).mean()) + 1e-12)
        print(f"  {name:12s} {len(x)/SR:5.2f}s   filter change {ch:+6.1f} dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
