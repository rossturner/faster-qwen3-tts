"""Does VoiceDesign obey a plain emotional direction appended to a fixed persona?

Replaces the narrative directions in voicedesign_stability.py ("holding back a laugh"),
which were too subtle to judge. Four unambiguous emotions, three takes each, so a real
effect can be told apart from VoiceDesign's draw-to-draw variation -- if all three
excited takes are excited and all three sad takes are sad, the direction landed
regardless of how much the identity wobbles between them.

Persona string and text are identical to voicedesign_stability.py so the two runs are
directly comparable.

Usage:
    .venv/bin/python spikes/emotion/voicedesign_directions.py
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from instruct_inertness import measure  # noqa: E402

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

OUT = Path(__file__).parent / "voicedesign_directions_v2"
LANGUAGE = "English"
TEMPERATURE = 0.7
TAKES = 3
GAP_SECONDS = 0.6

PERSONA = ("A bright, friendly young woman's voice in a natural mid-range, clear and "
           "expressive.")

EMOTIONS = {
    "neutral": "",
    "excited": " She sounds excited: bright, energetic and animated.",
    "sad": " She sounds sad: quiet, slow and downcast.",
    "angry": " She sounds angry: sharp, hard and forceful.",
}

TEXT = ("Someone in chat just asked me whether I had ever tried any of this before, and "
        "the honest answer is no, not once.")


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    print("Loading VoiceDesign...", flush=True)
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)

    clips, manifest, sr = {}, {}, None
    for emotion, suffix in EMOTIONS.items():
        manifest[emotion] = []
        for take in range(TAKES):
            wavs, sr = model.generate_voice_design(
                text=TEXT, instruct=PERSONA + suffix, language=LANGUAGE,
                temperature=TEMPERATURE)
            audio = np.asarray(wavs[0], dtype=np.float32)
            write_wav(OUT / emotion / f"take{take}.wav", audio, sr)
            clips[(emotion, take)] = audio
            manifest[emotion].append(measure(audio, sr, len(TEXT)))
        m = manifest[emotion]
        print(f"  {emotion:<9} dur {np.mean([r['duration_s'] for r in m]):5.2f}s   "
              f"pitch {np.nanmean([r['f0_median'] for r in m]):6.1f}Hz   "
              f"loudness {np.mean([r['rms'] for r in m]):.4f}", flush=True)

    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
    for take in range(TAKES):
        tour = [c for e in EMOTIONS for c in (clips[(e, take)], gap)]
        write_wav(OUT / f"tour_take{take}.wav", np.concatenate(tour), sr)

    (OUT / "results.json").write_text(json.dumps(
        {"persona": PERSONA, "emotions": EMOTIONS, "text": TEXT, "takes": TAKES,
         "tour_order": list(EMOTIONS), "measurements": manifest}, indent=2))

    print(f"\n  {len(clips)} clips in {OUT}/")
    print("    tour_take<N>.wav   neutral, excited, sad, angry back to back")
    print("    <emotion>/take<N>.wav   the three takes of one emotion")


if __name__ == "__main__":
    main()
