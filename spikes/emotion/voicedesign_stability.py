"""Does VoiceDesign hold one identity across calls with a fixed persona description?

VoiceDesign is the only model that combines free-text instruction control with an
arbitrary designed voice. It was ruled out for serving on the assumption that its
unseeded sampling makes the speaker drift call-to-call -- an assumption never measured.
If the drift is small, it gives a unique voice AND open-ended emotional direction, which
neither the clone path (pace-only instruct) nor CustomVoice (nine fixed timbres) can.

Measured as pairwise cosine similarity between speaker embeddings, on a scale set by two
baselines generated the same way:

  upper bound   repeated calls to one CustomVoice speaker token -- as stable as this
                engine gets, since the identity is a fixed embedding
  lower bound   two different CustomVoice speakers -- unmistakably different people

VoiceDesign is run twice: with the persona description alone, and with an emotional
direction appended, to separate baseline drift from drift the direction causes.

Usage:
    .venv/bin/python spikes/emotion/voicedesign_stability.py
"""

from __future__ import annotations

import json
import wave
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from faster_qwen3_tts import FasterQwen3TTS

OUT = Path(__file__).parent / "voicedesign_stability"
LANGUAGE = "English"
TEMPERATURE = 0.7
RUNS = 8

PERSONA = ("A bright, friendly young woman's voice in a natural mid-range, clear and "
           "expressive, with an easy conversational warmth and a lively, engaged delivery.")

DIRECTIONS = [
    "She is delighted by what she has just read out.",
    "She is thinking it over carefully, half to herself.",
    "She is teasing, holding back a laugh.",
    "She is tired and a little fed up with the whole thing.",
    "She is quietly moved by it.",
    "She is explaining it patiently for the second time.",
    "She is genuinely surprised and slightly thrown.",
    "She is winding up to the punchline and enjoying it.",
]

TEXT = ("Someone in chat just asked me whether I had ever tried any of this before, and "
        "the honest answer is no, not once.")


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def generate() -> None:
    print("Loading VoiceDesign...", flush=True)
    vd = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)

    for label, instructs in [
        ("vd_persona_only", [PERSONA] * RUNS),
        ("vd_with_direction", [f"{PERSONA} {d}" for d in DIRECTIONS]),
    ]:
        print(f"  {label}...", flush=True)
        for i, instruct in enumerate(instructs):
            wavs, sr = vd.generate_voice_design(
                text=TEXT, instruct=instruct, language=LANGUAGE, temperature=TEMPERATURE)
            write_wav(OUT / label / f"{i}.wav", np.asarray(wavs[0], dtype=np.float32), sr)

    del vd
    torch.cuda.empty_cache()

    print("Loading CustomVoice for the baselines...", flush=True)
    cv = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)
    for label, speaker in [("cv_ono_anna", "ono_anna"), ("cv_vivian", "vivian")]:
        print(f"  {label}...", flush=True)
        for i in range(RUNS):
            wavs, sr = cv.generate_custom_voice(
                text=TEXT, speaker=speaker, language=LANGUAGE, temperature=TEMPERATURE)
            write_wav(OUT / label / f"{i}.wav", np.asarray(wavs[0], dtype=np.float32), sr)
    del cv
    torch.cuda.empty_cache()


def embeddings() -> dict:
    print("Loading Base to extract speaker embeddings...", flush=True)
    base = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)
    out = {}
    for group in sorted(p for p in OUT.iterdir() if p.is_dir()):
        vecs = []
        for f in sorted(group.glob("*.wav")):
            items = base.model.create_voice_clone_prompt(
                ref_audio=str(f), ref_text="", x_vector_only_mode=True)
            vecs.append(items[0].ref_spk_embedding.detach().float().flatten().cpu())
        out[group.name] = torch.stack(vecs)
    del base
    torch.cuda.empty_cache()
    return out


def cosines(a: torch.Tensor, b: torch.Tensor | None = None) -> np.ndarray:
    if b is None:
        return np.array([torch.nn.functional.cosine_similarity(a[i], a[j], dim=0).item()
                         for i, j in combinations(range(len(a)), 2)])
    return np.array([torch.nn.functional.cosine_similarity(x, y, dim=0).item()
                     for x in a for y in b])


def main() -> None:
    if not OUT.exists():
        generate()
    else:
        print(f"reusing clips in {OUT}/")
    emb = embeddings()

    rows = [
        ("same CustomVoice speaker (upper bound)", cosines(emb["cv_ono_anna"])),
        ("VoiceDesign, persona only", cosines(emb["vd_persona_only"])),
        ("VoiceDesign, persona + direction", cosines(emb["vd_with_direction"])),
        ("two CustomVoice speakers (lower bound)",
         cosines(emb["cv_ono_anna"], emb["cv_vivian"])),
    ]

    print(f"\n  {'group':<40} {'mean':>7} {'min':>7} {'max':>7}")
    for label, c in rows:
        print(f"  {label:<40} {c.mean():>7.3f} {c.min():>7.3f} {c.max():>7.3f}")

    upper = rows[0][1].mean()
    lower = rows[3][1].mean()
    print(f"\n  scale: {lower:.3f} (different people) .. {upper:.3f} (same speaker token)")
    for label, c in rows[1:3]:
        pos = (c.mean() - lower) / (upper - lower)
        print(f"  {label:<40} sits {pos:5.1%} of the way to a fixed identity")

    (OUT / "results.json").write_text(json.dumps(
        {"persona": PERSONA, "directions": DIRECTIONS, "text": TEXT, "runs": RUNS,
         "cosines": {label: c.tolist() for label, c in rows}}, indent=2))
    print(f"\n  clips in {OUT}/<group>/ -- listen to vd_persona_only/*.wav for whether "
          f"they are one person")


if __name__ == "__main__":
    main()
