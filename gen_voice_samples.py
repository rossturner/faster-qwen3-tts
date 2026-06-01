#!/usr/bin/env python3
"""Generate 3 samples per CustomVoice preset in its native language."""
import os
import time
import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_samples")
os.makedirs(OUT_DIR, exist_ok=True)

# instruct must be English or Chinese only (Alibaba VoiceDesign docs); it is
# independent of the output language, so one English instruct covers all languages.
_INSTRUCT_EN = "Speak in a calm, clear, professional tone suitable for an instructional video."
INSTRUCT = {lang: _INSTRUCT_EN for lang in ("English", "Chinese", "Japanese", "Korean")}

SENTENCES = {
    "English": [
        "In this lesson, we'll start by sketching the basic shapes that form the foundation of your character.",
        "Select the brush tool, lower the opacity to about thirty percent, and build up your shading in gentle layers.",
        "Notice how the light source on the left defines where the shadows fall across the object.",
    ],
    "Chinese": [
        "在这节课中，我们先画出构成角色的基本形状。",
        "选择画笔工具，把不透明度降到大约百分之三十，然后一层一层地轻轻叠加阴影。",
        "注意左侧的光源是如何决定阴影在物体上的分布的。",
    ],
    "Japanese": [
        "このレッスンでは、まずキャラクターの土台となる基本的な形を描いていきます。",
        "ブラシツールを選び、不透明度を約三十パーセントに下げて、少しずつ重ねるように陰影をつけていきます。",
        "左側の光源によって、物体に落ちる影の位置がどう決まるかに注目してください。",
    ],
    "Korean": [
        "이번 강의에서는 먼저 캐릭터의 기초가 되는 기본 도형을 스케치해 보겠습니다.",
        "브러시 도구를 선택하고 불투명도를 약 삼십 퍼센트로 낮춘 다음, 한 겹씩 부드럽게 음영을 쌓아 올리세요.",
        "왼쪽의 광원이 물체에 드리워지는 그림자의 위치를 어떻게 결정하는지 살펴보세요.",
    ],
}

# (speaker, language, note)
VOICES = [
    ("aiden", "English", ""),
    ("ryan", "English", ""),
    ("uncle_fu", "Chinese", ""),
    ("serena", "Chinese", ""),
    ("vivian", "Chinese", ""),
    ("dylan", "Chinese", "Beijing dialect"),
    ("eric", "Chinese", "Sichuan dialect"),
    ("ono_anna", "Japanese", ""),
    ("sohee", "Korean", ""),
]

print("Loading CustomVoice 1.7B...")
model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    max_seq_len=2048,
)

print("Warmup (CUDA graph capture)...")
model.generate_custom_voice(text="Hello.", speaker="aiden", language="English", max_new_tokens=20)

t0 = time.perf_counter()
n = 0
for speaker, language, note in VOICES:
    tag = f"{speaker} ({language}{', ' + note if note else ''})"
    for i, text in enumerate(SENTENCES[language], 1):
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker, language=language, instruct=INSTRUCT[language], temperature=0.7)
        path = os.path.join(OUT_DIR, f"{speaker}_{language.lower()}_{i}.wav")
        sf.write(path, wavs[0], sr)
        dur = len(wavs[0]) / sr
        n += 1
        print(f"  [{n:2d}/27] {tag:32s} #{i}  {dur:4.1f}s  -> {os.path.basename(path)}")

print(f"\nDone: {n} samples in {time.perf_counter() - t0:.1f}s -> {OUT_DIR}")
