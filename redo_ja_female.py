#!/usr/bin/env python3
"""Re-roll Japanese-female audition voices: v1/v3 fresh takes, v2 redesigned
(warm mid-pitch + relaxed pace instead of the too-high/too-fast 'young-bright')."""
import os
import gc
import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_design_audition")

LANG = "Japanese"
REF_TEXT = "このコースへようこそ。今日のレッスンを始めましょう。"
SENTENCES = [
    "このレッスンでは、まずキャラクターの土台となる基本的な形を描いていきます。",
    "ブラシツールを選び、不透明度を約三十パーセントに下げて、少しずつ重ねるように陰影をつけていきます。",
    "左側の光源によって、物体に落ちる影の位置がどう決まるかに注目してください。",
]

# (id, instruct)  -- v1/v3 same prompts (fresh takes); v2 new approach.
VOICES = [
    ("ja_f_v1", "A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course."),
    ("ja_f_v2", "A friendly, conversational younger woman's voice with a warm mid-pitch and a relaxed, unhurried pace, clear and natural, suitable for an instructional video course."),
    ("ja_f_v3", "A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video."),
]

print("Loading VoiceDesign 1.7B...")
design = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
design.generate_voice_design(text="テスト。", instruct=VOICES[0][1], language=LANG, max_new_tokens=20)

for key, instruct in VOICES:
    wavs, sr = design.generate_voice_design(text=REF_TEXT, instruct=instruct, language=LANG, temperature=0.7)
    sf.write(os.path.join(OUT_DIR, f"{key}_ref.wav"), wavs[0], sr)
    print(f"  ref  {key}: {len(wavs[0])/sr:4.1f}s")

del design
gc.collect()
torch.cuda.empty_cache()

print("\nLoading Base 1.7B...")
base = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
base.generate_voice_clone(text="テスト。", language=LANG,
                          ref_audio=os.path.join(OUT_DIR, "ja_f_v1_ref.wav"), ref_text=REF_TEXT, max_new_tokens=20)

for key, _ in VOICES:
    ref = os.path.join(OUT_DIR, f"{key}_ref.wav")
    for i, text in enumerate(SENTENCES, 1):
        wavs, sr = base.generate_voice_clone(
            text=text, language=LANG, ref_audio=ref, ref_text=REF_TEXT,
            xvec_only=False, temperature=0.7,
        )
        sf.write(os.path.join(OUT_DIR, f"{key}_{i}.wav"), wavs[0], sr)
    print(f"  clone {key}: 3 sentences")

print("\nDone: ja_f_v1/v2/v3 regenerated.")
