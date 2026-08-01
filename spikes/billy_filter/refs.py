"""Which dry reference clones Billy best?

401_004 was chosen as the base for tuning the FILTER, because it was the clip named for
that job. Nothing has ever tested whether it is a good clone SOURCE -- a different
question, and the likely reason the plain clone does not sound like Billy.

The server picks at random among these four, so if they clone differently the voice is
inconsistent per request regardless of the filter.

One line, cloned from each of Billy's four dry neutral references, plain and filtered.

Usage:
    .venv/bin/python spikes/billy_filter/refs.py
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
LIB = REPO / "faster_qwen3_tts" / "server_voices" / "characters" / "billy" / "neutral"
OUT = Path(__file__).parent / "out" / "refs"
SR = 48000
CENTS, DELAYS, AMOUNT = 26, [8, 16], 0.70

TEXT = ("Boss, I've been thinking about that mech we saw yesterday, and I'm almost "
        "certain I've seen it somewhere before.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = sorted(LIB.glob("*.wav"))
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)

    def save(name, x):
        p = float(np.abs(x).max())
        sf.write(OUT / f"{name}.wav", x * (0.89 / p) if p > 0 else x, SR)

    for ref in refs:
        tag = ref.stem.rsplit("_Billy_", 1)[1]
        rtext = ref.with_suffix(".txt").read_text().strip()
        dur = sf.info(ref).duration
        prompt = model.model.create_voice_clone_prompt(
            ref_audio=str(ref), ref_text=rtext, x_vector_only_mode=False)
        wavs, sr = model.generate_voice_clone(
            text=TEXT, language="English", voice_clone_prompt=prompt, temperature=0.7)
        x = np.asarray(wavs[0], np.float32)
        x = librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x
        save(f"ref{tag}_A_plain", x)
        save(f"ref{tag}_B_billy",
             np.asarray(double(x, tag, offsets=[+CENTS, -CENTS], delays=DELAYS,
                               amount=AMOUNT), np.float32))
        # the reference itself, so the clone can be judged against its source
        r = librosa.load(str(ref), sr=SR, mono=True)[0]
        save(f"ref{tag}_0_SOURCE", r)
        print(f"  {tag}  ref {dur:5.2f}s -> clone {len(x)/SR:5.2f}s   \"{rtext[:56]}...\"")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
