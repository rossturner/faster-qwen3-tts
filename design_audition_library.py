#!/usr/bin/env python3
"""Audition library: design a range of instructional voices per language+gender,
then clone 3 illustration sentences off each so they can be heard consistently."""
import os
import gc
import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_design_audition")
os.makedirs(OUT_DIR, exist_ok=True)

# 3 instructional-appropriate personas per gender (pitch/age spread, all calm+professional).
PERSONAS = {
    "m": [
        ("v1", "mid-warm",     "A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course."),
        ("v2", "young-bright", "A bright, approachable younger man's voice in a higher tenor range, with clear articulation and a relaxed, friendly delivery, suitable for an online tutorial course."),
        ("v3", "mature-deep",  "A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video."),
    ],
    "f": [
        ("v1", "mid-warm",      "A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course."),
        ("v2", "young-bright",  "A bright, friendly younger woman's voice with a light, articulate and approachable delivery, suitable for an online tutorial course."),
        ("v3", "mature-smooth", "A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video."),
    ],
}

LANGS = [("en", "English"), ("es", "Spanish"), ("fr", "French"),
         ("zh", "Chinese"), ("ja", "Japanese"), ("ko", "Korean")]

REF_SENTENCE = {
    "English":  "Welcome to the course. Let's get started with today's lesson.",
    "Spanish":  "Bienvenido al curso. Vamos a empezar con la lección de hoy.",
    "French":   "Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui.",
    "Chinese":  "欢迎来到本课程。让我们开始今天的课程吧。",
    "Japanese": "このコースへようこそ。今日のレッスンを始めましょう。",
    "Korean":   "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다.",
}

SENTENCES = {
    "English": [
        "In this lesson, we'll start by sketching the basic shapes that form the foundation of your character.",
        "Select the brush tool, lower the opacity to about thirty percent, and build up your shading in gentle layers.",
        "Notice how the light source on the left defines where the shadows fall across the object.",
    ],
    "Spanish": [
        "En esta lección, comenzaremos esbozando las formas básicas que componen la base de tu personaje.",
        "Selecciona la herramienta de pincel, baja la opacidad a un treinta por ciento y construye el sombreado en capas suaves.",
        "Observa cómo la fuente de luz a la izquierda define dónde caen las sombras sobre el objeto.",
    ],
    "French": [
        "Dans cette leçon, nous commencerons par esquisser les formes de base qui constituent le fondement de votre personnage.",
        "Sélectionnez l'outil pinceau, réduisez l'opacité à environ trente pour cent, et construisez votre ombrage en couches légères.",
        "Remarquez comment la source de lumière à gauche définit l'endroit où les ombres tombent sur l'objet.",
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

# Build the full voice list.
voices = []
for code, lang in LANGS:
    for gender in ("m", "f"):
        for vid, label, instruct in PERSONAS[gender]:
            voices.append({
                "id": f"{code}_{gender}_{vid}", "lang": lang, "gender": gender,
                "label": label, "instruct": instruct, "ref_text": REF_SENTENCE[lang],
                "sentences": SENTENCES[lang],
            })
print(f"Total voices to design: {len(voices)}")

# --- Step 1: design one reference clip per voice (VoiceDesign) ---
print("\nLoading VoiceDesign 1.7B...")
design = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
design.generate_voice_design(text="Test.", instruct=voices[0]["instruct"], language="English", max_new_tokens=20)

for n, v in enumerate(voices, 1):
    wavs, sr = design.generate_voice_design(text=v["ref_text"], instruct=v["instruct"], language=v["lang"], temperature=0.7)
    p = os.path.join(OUT_DIR, f"{v['id']}_ref.wav")
    sf.write(p, wavs[0], sr)
    v["ref_path"] = p
    print(f"  [design {n:2d}/{len(voices)}] {v['id']:10s} ({v['label']:13s}) {len(wavs[0])/sr:4.1f}s")

del design
gc.collect()
torch.cuda.empty_cache()

# --- Step 2: clone each pinned reference for all 3 illustration sentences (Base, ICL) ---
print("\nLoading Base 1.7B...")
base = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
base.generate_voice_clone(text="Test.", language="English",
                          ref_audio=voices[0]["ref_path"], ref_text=voices[0]["ref_text"], max_new_tokens=20)

done = 0
for v in voices:
    for i, text in enumerate(v["sentences"], 1):
        wavs, sr = base.generate_voice_clone(
            text=text, language=v["lang"],
            ref_audio=v["ref_path"], ref_text=v["ref_text"],
            xvec_only=False, temperature=0.7,
        )
        sf.write(os.path.join(OUT_DIR, f"{v['id']}_{i}.wav"), wavs[0], sr)
        done += 1
    print(f"  [clone {v['id']:10s}] 3 sentences done  ({done}/{len(voices)*3})")

print(f"\nDone: {len(voices)} voices, {len(voices)} refs + {len(voices)*3} clones -> {OUT_DIR}")
