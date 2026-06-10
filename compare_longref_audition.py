#!/usr/bin/env python3
"""A/B audition: current (short-ref) production voices vs the same personas
re-designed from a LONGER language-matched reference line.

For each of the 11 non-en_m voices it renders two versions of the same sample
lines so they can be compared by ear:
  - current : clone off the existing production ref clip (server_voices/refs/)
  - longref : design a fresh ref clip from the same persona instruct but a longer
              reference line, then clone off that
Generation is unseeded, so 'longref' is a NEW take of the persona (this is exactly
the trade-off being judged). Outputs compare_longref.html.
"""
import os
import gc
import html
import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

ROOT = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(ROOT, "faster_qwen3_tts", "server_voices", "refs")
OUT_DIR = os.path.join(ROOT, "voice_design_longref_compare")
os.makedirs(OUT_DIR, exist_ok=True)

# Longer, language-matched reference line per language (translation of the en line
# used for en_m_confident_mid). Used only to DESIGN the 'longref' ref clip.
# Non-English lines tightened to ~one calm sentence to land near the English ~7s length
# (the first pass ran 10-12s, perturbing the non-English voices more than English).
LONG_REF = {
    "English":  "In today's lesson we'll work through each step slowly and carefully, so take your time and follow along at your own pace as we go.",
    "Spanish":  "En la lección de hoy avanzaremos paso a paso, con calma, así que tómate tu tiempo y sigue tu propio ritmo.",
    "French":   "Dans la leçon d'aujourd'hui, nous avancerons calmement, étape par étape, alors prenez votre temps et suivez votre propre rythme.",
    "Chinese":  "在今天的课程里，我们会一步一步、慢慢地讲解，请放轻松，按自己的节奏跟着做。",
    "Japanese": "今日のレッスンでは、各ステップをゆっくり丁寧に進めますので、焦らず自分のペースで進めましょう。",
    "Korean":   "오늘 수업에서는 각 단계를 천천히 살펴볼 거예요. 서두르지 말고 자신의 속도에 맞춰 따라오세요.",
}

# Two sample lines per language (from the existing audition set) to judge both versions on.
SAMPLES = {
    "English":  ["In this lesson, we'll start by sketching the basic shapes that form the foundation of your character.",
                 "Select the brush tool, lower the opacity to about thirty percent, and build up your shading in gentle layers."],
    "Spanish":  ["En esta lección, comenzaremos esbozando las formas básicas que componen la base de tu personaje.",
                 "Selecciona la herramienta de pincel, baja la opacidad a un treinta por ciento y construye el sombreado en capas suaves."],
    "French":   ["Dans cette leçon, nous commencerons par esquisser les formes de base qui constituent le fondement de votre personnage.",
                 "Sélectionnez l'outil pinceau, réduisez l'opacité à environ trente pour cent, et construisez votre ombrage en couches légères."],
    "Chinese":  ["在这节课中，我们先画出构成角色的基本形状。",
                 "选择画笔工具，把不透明度降到大约百分之三十，然后一层一层地轻轻叠加阴影。"],
    "Japanese": ["このレッスンでは、まずキャラクターの土台となる基本的な形を描いていきます。",
                 "ブラシツールを選び、不透明度を約三十パーセントに下げて、少しずつ重ねるように陰影をつけていきます。"],
    "Korean":   ["이번 강의에서는 먼저 캐릭터의 기초가 되는 기본 도형을 스케치해 보겠습니다.",
                 "브러시 도구를 선택하고 불투명도를 약 삼십 퍼센트로 낮춘 다음, 한 겹씩 부드럽게 음영을 쌓아 올리세요."],
}

# 11 non-en_m production voices: existing ref clip + ref_text + persona instruct.
SHORT_EN = "Welcome to the course. Let's get started with today's lesson."
SHORT_ES = "Bienvenido al curso. Vamos a empezar con la lección de hoy."
SHORT_FR = "Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui."
SHORT_ZH = "欢迎来到本课程。让我们开始今天的课程吧。"
SHORT_JA = "このコースへようこそ。今日のレッスンを始めましょう。"
SHORT_KO = "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다."

MID_WARM_M = "A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course."
MID_WARM_F = "A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course."
MATURE_SMOOTH_F = "A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video."
MATURE_DEEP_M = "A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video."

# en_f already promoted to production (en_f_longref); en_m switched to confident_mid.
# This pass re-auditions only the non-English voices with the tightened ref lines above.
VOICES = [
    ("es_m", "Spanish",  "es_m_v2_ref.wav",              SHORT_ES, "A bright, approachable younger man's voice in a higher tenor range, with clear articulation and a relaxed, friendly delivery, suitable for an online tutorial course."),
    ("es_f", "Spanish",  "es_f_v3_ref.wav",              SHORT_ES, MATURE_SMOOTH_F),
    ("fr_m", "French",   "fr_m_v1_ref.wav",              SHORT_FR, MID_WARM_M),
    ("fr_f", "French",   "fr_f_v3_ref.wav",              SHORT_FR, MATURE_SMOOTH_F),
    ("zh_m", "Chinese",  "zh_m_v3_ref.wav",              SHORT_ZH, MATURE_DEEP_M),
    ("zh_f", "Chinese",  "zh_f_v3_ref.wav",              SHORT_ZH, MATURE_SMOOTH_F),
    ("ja_m", "Japanese", "ja_m_v1_ref.wav",              SHORT_JA, MID_WARM_M),
    ("ja_f", "Japanese", "ja_f_v3_ref.wav",              SHORT_JA, MATURE_SMOOTH_F),
    ("ko_m", "Korean",   "ko_m_v3_ref.wav",              SHORT_KO, MATURE_DEEP_M),
    ("ko_f", "Korean",   "ko_f_friendly_casual_ref.wav", SHORT_KO, "A relaxed, approachable woman's voice in a natural mid-range with a warm, conversational tone, easy-going and personable, suitable for a friendly tutorial."),
]

print(f"Voices: {len(VOICES)} | 2 samples each x (current + longref)")

# --- Phase 1: design longref ref clips (VoiceDesign) ---
print("\nLoading VoiceDesign 1.7B...")
design = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup..."); design.generate_voice_design(text="Test.", instruct=VOICES[0][4], language="English", max_new_tokens=20)

longref_paths = {}
for n, (vid, lang, _short_ref, _short_text, instruct) in enumerate(VOICES, 1):
    wavs, sr = design.generate_voice_design(text=LONG_REF[lang], instruct=instruct, language=lang, temperature=0.7)
    p = os.path.join(OUT_DIR, f"{vid}_longref_ref.wav")
    sf.write(p, wavs[0], sr)
    longref_paths[vid] = p
    print(f"  [design {n:2d}/{len(VOICES)}] {vid:5s} {lang:9s} {len(wavs[0])/sr:4.1f}s")

del design; gc.collect(); torch.cuda.empty_cache()

# --- Phase 2: clone both versions (Base) ---
print("\nLoading Base 1.7B...")
base = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup..."); base.generate_voice_clone(text="Test.", language="English",
    ref_audio=os.path.join(REFS, VOICES[0][2]), ref_text=VOICES[0][3], max_new_tokens=20)

def clone(ref_audio, ref_text, lang, text):
    wavs, sr = base.generate_voice_clone(text=text, language=lang, ref_audio=ref_audio,
                                         ref_text=ref_text, xvec_only=False, temperature=0.7)
    return wavs[0], sr

for vid, lang, short_ref, short_text, instruct in VOICES:
    cur_ref = os.path.join(REFS, short_ref)
    for i, text in enumerate(SAMPLES[lang]):
        a, sr = clone(cur_ref, short_text, lang, text)
        sf.write(os.path.join(OUT_DIR, f"{vid}_current_{i}.wav"), a, sr)
        a, sr = clone(longref_paths[vid], LONG_REF[lang], lang, text)
        sf.write(os.path.join(OUT_DIR, f"{vid}_longref_{i}.wav"), a, sr)
    print(f"  [clone {vid:5s}] current + longref done")

# --- compare HTML ---
LANG_OF = {v[0]: v[1] for v in VOICES}
rows = []
for vid, lang, short_ref, short_text, instruct in VOICES:
    for i, text in enumerate(SAMPLES[lang]):
        first = (i == 0)
        head = (f'<td rowspan=2 class=v><b>{vid}</b><div class=l>{lang}</div>'
                f'<div class=ins>{html.escape(instruct)}</div></td>') if first else ''
        rows.append(
            f"<tr>{head}<td class=ln>{html.escape(text)}</td>"
            f'<td><audio controls src="{vid}_current_{i}.wav"></audio></td>'
            f'<td><audio controls src="{vid}_longref_{i}.wav"></audio></td></tr>')

page = f"""<!doctype html><meta charset=utf-8><title>longref A/B</title>
<style>
body{{font:14px/1.4 system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:18px}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #333;padding:8px;vertical-align:top}} th{{background:#1c1c1c}}
.v{{width:260px;background:#181818}} .l{{color:#8ab;font-size:12px}} .ins{{color:#9bb;font-size:11px;margin-top:4px}}
.ln{{width:340px;color:#cbd}} audio{{width:220px;height:34px}}
</style>
<h1>Reference-length A/B — current (short ref) vs longref</h1>
<p>Left audio = current production voice (existing short ref clip). Right audio = same persona
re-designed from a longer language-matched reference line, tightened to ~7s to match the English
length. Unseeded, temperature=0.7. en_m / en_f excluded (already in production).</p>
<table><thead><tr><th>voice</th><th>sample line</th><th>current (short ref)</th><th>longref</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""
with open(os.path.join(OUT_DIR, "compare_longref.html"), "w") as f:
    f.write(page)

print(f"\nDone -> {OUT_DIR}")
print(f"Compare page: {os.path.join(OUT_DIR, 'compare_longref.html')}")
