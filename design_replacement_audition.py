#!/usr/bin/env python3
"""Replacement-audition for the two CustomVoice slots (en_m=aiden, ko_f=sohee).

Designs a spread of VoiceDesign personas for English-male and Korean-female, then
clones the reference line + 3 illustration sentences off each designed _ref.wav so
each candidate can be judged as a stable narrator (design-once -> pin -> clone).
"""
import os
import gc
import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_design_replacements")
os.makedirs(OUT_DIR, exist_ok=True)

REF_SENTENCE = {
    "English": "Welcome to the course. Let's get started with today's lesson.",
    "Korean":  "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다.",
}

SENTENCES = {
    "English": [
        "In this lesson, we'll start by sketching the basic shapes that form the foundation of your character.",
        "Select the brush tool, lower the opacity to about thirty percent, and build up your shading in gentle layers.",
        "Notice how the light source on the left defines where the shadows fall across the object.",
    ],
    "Korean": [
        "이번 강의에서는 먼저 캐릭터의 기초가 되는 기본 도형을 스케치해 보겠습니다.",
        "브러시 도구를 선택하고 불투명도를 약 삼십 퍼센트로 낮춘 다음, 한 겹씩 부드럽게 음영을 쌓아 올리세요.",
        "왼쪽의 광원이 물체에 드리워지는 그림자의 위치를 어떻게 결정하는지 살펴보세요.",
    ],
}

# A spread of distinct instructional-narrator personas for each custom slot.
# instruct is English-only (model accepts English/Chinese only) and independent of output language.
CANDIDATES = {
    "en_m": {
        "language": "English",
        "personas": [
            ("warm_baritone",   "A warm, resonant adult man's voice with a deep baritone tone, calm and authoritative, with an unhurried, reassuring delivery, suitable for narrating a professional instructional video."),
            ("bright_tenor",    "A bright, energetic younger man's voice in a clear higher tenor range, friendly and upbeat, with crisp articulation, suitable for an engaging online tutorial."),
            ("neutral_broadcast","A neutral, polished adult man's voice in a balanced mid-range with even pacing and clean diction, in a professional broadcast-narrator style, suitable for an instructional course."),
            ("mellow_low",      "A soft, mellow man's voice with a low, smooth tone and a gentle, calm, soothing delivery, suitable for a relaxed instructional video."),
            ("crisp_professional","A clear, articulate adult man's voice in a confident mid-range, precise and professional, with measured, deliberate pacing, suitable for a corporate training video."),
            ("friendly_casual", "A relaxed, approachable man's voice in a natural mid-range with a warm, conversational tone, easy-going and personable, suitable for a friendly tutorial."),
            ("mature_gravel",   "A mature, seasoned man's voice with a warm lower-mid tone and a slight gravelly texture, steady and trustworthy, suitable for a documentary-style instructional narration."),
            ("upbeat_dynamic",  "A lively, dynamic man's voice in a bright mid-range with strong rhythmic energy and an enthusiastic, motivating delivery, suitable for an upbeat lesson."),
        ],
    },
    "ko_f": {
        "language": "Korean",
        "personas": [
            ("warm_midrange",   "A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course."),
            ("bright_young",    "A bright, friendly younger woman's voice with a light, articulate and approachable delivery, cheerful and energetic, suitable for an online tutorial."),
            ("mature_smooth",   "A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video."),
            ("soft_gentle",     "A soft, gentle woman's voice with a calm, soothing tone and slow, careful pacing, warm and intimate, suitable for a relaxed instructional narration."),
            ("crisp_clear",     "A crisp, articulate adult woman's voice in a clear mid-range, precise and confident, with clean diction and even pacing, suitable for a corporate training video."),
            ("lively_upbeat",   "A lively, upbeat woman's voice in a bright tone with warm energy and an engaging, expressive delivery, suitable for a friendly online course."),
            ("calm_low",        "A calm, composed woman's voice in a lower, settled register with a measured, grounded delivery, mature and reassuring, suitable for a professional instructional video."),
            ("friendly_casual", "A relaxed, approachable woman's voice in a natural mid-range with a warm, conversational tone, easy-going and personable, suitable for a friendly tutorial."),
        ],
    },
}

voices = []
for slot, cfg in CANDIDATES.items():
    lang = cfg["language"]
    for persona, instruct in cfg["personas"]:
        voices.append({
            "id": f"{slot}_{persona}", "slot": slot, "lang": lang,
            "persona": persona, "instruct": instruct,
            "ref_text": REF_SENTENCE[lang], "sentences": SENTENCES[lang],
        })
print(f"Total candidate voices to design: {len(voices)}")

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
    print(f"  [design {n:2d}/{len(voices)}] {v['id']:24s} {len(wavs[0])/sr:4.1f}s")

del design
gc.collect()
torch.cuda.empty_cache()

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
    print(f"  [clone {v['id']:24s}] 3 sentences done  ({done}/{len(voices)*3})")

print(f"\nDone: {len(voices)} candidates, {len(voices)} refs + {len(voices)*3} clones -> {OUT_DIR}")
