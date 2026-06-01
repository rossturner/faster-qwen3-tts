#!/usr/bin/env python3
"""Design-once -> pin -> clone: stable narrator for gap voices (EN-female, JP-male)."""
import os
import gc
import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_samples")
os.makedirs(OUT_DIR, exist_ok=True)

# Per gap voice: persona instruct (tone baked in), a dedicated reference line, and
# the 3 illustration-course sentences to audition.
VOICES = {
    "en_female": {
        "language": "English",
        "instruct": "A calm, clear, professional adult woman's voice with a warm, natural mid-range tone, suitable for an instructional video.",
        "ref_text": "Welcome to the course. Let's get started with today's lesson.",
        "sentences": [
            "In this lesson, we'll start by sketching the basic shapes that form the foundation of your character.",
            "Select the brush tool, lower the opacity to about thirty percent, and build up your shading in gentle layers.",
            "Notice how the light source on the left defines where the shadows fall across the object.",
        ],
    },
    "jp_male": {
        "language": "Japanese",
        "instruct": "A calm, clear, professional adult man's voice with a natural mid-range tone, suitable for an instructional video.",
        "ref_text": "このコースへようこそ。今日のレッスンを始めましょう。",
        "sentences": [
            "このレッスンでは、まずキャラクターの土台となる基本的な形を描いていきます。",
            "ブラシツールを選び、不透明度を約三十パーセントに下げて、少しずつ重ねるように陰影をつけていきます。",
            "左側の光源によって、物体に落ちる影の位置がどう決まるかに注目してください。",
        ],
    },
}

# --- Step 1: design one reference clip per voice (VoiceDesign) ---
print("Loading VoiceDesign 1.7B...")
design = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
design.generate_voice_design(text="Test.", instruct=VOICES["en_female"]["instruct"], language="English", max_new_tokens=20)

ref_paths = {}
for key, v in VOICES.items():
    wavs, sr = design.generate_voice_design(text=v["ref_text"], instruct=v["instruct"], language=v["language"], temperature=0.7)
    p = os.path.join(OUT_DIR, f"design_ref_{key}.wav")
    sf.write(p, wavs[0], sr)
    ref_paths[key] = p
    print(f"  reference: {key:10s} {len(wavs[0])/sr:4.1f}s -> {os.path.basename(p)}")

del design
gc.collect()
torch.cuda.empty_cache()

# --- Step 2: clone each pinned reference for all 3 sentences (Base, ICL mode) ---
print("\nLoading Base 1.7B...")
base = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
base.generate_voice_clone(text="Test.", language="English",
                          ref_audio=ref_paths["en_female"], ref_text=VOICES["en_female"]["ref_text"], max_new_tokens=20)

for key, v in VOICES.items():
    for i, text in enumerate(v["sentences"], 1):
        wavs, sr = base.generate_voice_clone(
            text=text, language=v["language"],
            ref_audio=ref_paths[key], ref_text=v["ref_text"],
            xvec_only=False, temperature=0.7,
        )
        p = os.path.join(OUT_DIR, f"clone_{key}_{i}.wav")
        sf.write(p, wavs[0], sr)
        print(f"  clone: {key:10s} #{i} {len(wavs[0])/sr:4.1f}s -> {os.path.basename(p)}")

print("\nDone. design_ref_* = the pinned voice; clone_*_{1,2,3} should all be the SAME person.")
