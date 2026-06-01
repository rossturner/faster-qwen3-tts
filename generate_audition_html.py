#!/usr/bin/env python3
"""Build a standalone HTML page to audition all generated TTS clips."""
import os
import html

ROOT = os.path.dirname(os.path.abspath(__file__))
CV_DIR = "voice_samples"
AUD_DIR = "voice_design_audition"

LANG_NAME = {"en": "English", "es": "Spanish", "fr": "French",
             "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}
LANG_ORDER = ["en", "es", "fr", "zh", "ja", "ko"]

PERSONA_LABEL = {
    ("m", "v1"): "mid-warm", ("m", "v2"): "young-bright", ("m", "v3"): "mature-deep",
    ("f", "v1"): "mid-warm", ("f", "v2"): "young-bright", ("f", "v3"): "mature-smooth",
}
PERSONA_INSTRUCT = {
    ("m", "v1"): "A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course.",
    ("m", "v2"): "A bright, approachable younger man's voice in a higher tenor range, with clear articulation and a relaxed, friendly delivery, suitable for an online tutorial course.",
    ("m", "v3"): "A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video.",
    ("f", "v1"): "A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course.",
    ("f", "v2"): "A bright, friendly younger woman's voice with a light, articulate and approachable delivery, suitable for an online tutorial course.",
    ("f", "v3"): "A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video.",
}

# Per-language persona overrides (code, gender, vid) -> (label, instruct).
# ja_f_v2 redesigned away from the too-high/too-fast 'young-bright'.
PERSONA_OVERRIDE = {
    ("ja", "f", "v2"): ("youthful-warm", "A friendly, conversational younger woman's voice with a warm mid-pitch and a relaxed, unhurried pace, clear and natural, suitable for an instructional video course."),
}

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

# CustomVoice presets: speaker -> (language, gender, description)
SPEAKERS = [
    ("aiden",    "English",  "male",   "Sunny American male, clear midrange"),
    ("ryan",     "English",  "male",   "Dynamic male, strong rhythmic drive"),
    ("uncle_fu", "Chinese",  "male",   "Seasoned male, low mellow timbre"),
    ("serena",   "Chinese",  "female", "Warm, gentle young female"),
    ("vivian",   "Chinese",  "female", "Bright, slightly edgy young female"),
    ("dylan",    "Chinese",  "male",   "Youthful Beijing male — Beijing dialect"),
    ("eric",     "Chinese",  "male",   "Lively Chengdu male — Sichuan dialect"),
    ("ono_anna", "Japanese", "female", "Playful Japanese female, light nimble"),
    ("sohee",    "Korean",   "female", "Warm Korean female, rich emotion"),
]


def exists(rel):
    return os.path.isfile(os.path.join(ROOT, rel))


def esc(s):
    return html.escape(s, quote=True)


def player(rel, label, sentence):
    if not exists(rel):
        return ""
    s = f'<div class="sent">{esc(sentence)}</div>' if sentence else ""
    return (f'<div class="clip"><div class="clab">{esc(label)}</div>{s}'
            f'<audio controls preload="none" src="{esc(rel)}"></audio></div>')


parts = []
parts.append("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-TTS voice audition</title>
<link rel="icon" href="data:,">

<style>
:root{color-scheme:light dark}
body{font:15px/1.5 system-ui,sans-serif;margin:0;padding:0 4vw 6rem;max-width:1100px}
h1{margin:1.2rem 0 .2rem}h2{margin:2.2rem 0 .4rem;padding-top:.4rem;border-top:2px solid #8884}
h3{margin:1.4rem 0 .3rem}
.note{color:#888;font-size:13px;margin:.2rem 0 1rem}
nav{position:sticky;top:0;background:Canvas;padding:.6rem 0;border-bottom:1px solid #8884;margin-bottom:1rem;font-size:13px}
nav a{margin-right:1rem;white-space:nowrap}
.voice{border:1px solid #8884;border-radius:8px;padding:.7rem .9rem;margin:.7rem 0;background:#8881}
.vhead{font-weight:600;font-size:15px}
.vmeta{color:#888;font-size:13px;margin:.1rem 0 .5rem}
.prompt{font-size:13px;background:#8882;border-left:3px solid #69f;padding:.4rem .6rem;border-radius:4px;margin:.2rem 0 .6rem;font-style:italic}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:.7rem}
.clip{padding:.4rem 0}
.clab{font-size:12px;color:#888}
.sent{font-size:13px;margin:.1rem 0 .25rem}
audio{width:100%;height:34px}
.pill{display:inline-block;font-size:11px;padding:.05rem .4rem;border-radius:10px;background:#69f3;margin-left:.4rem}
.src{display:inline-block;font-size:10px;font-weight:700;letter-spacing:.03em;padding:.05rem .4rem;border-radius:4px;margin-left:.4rem;text-transform:uppercase}
.src-builtin{background:#3a3;color:#fff}.src-designed{background:#69f;color:#fff}
.gender-m{border-left:3px solid #59f}.gender-f{border-left:3px solid #e7a}
.ghead{margin:.8rem 0 .2rem;font-size:14px;color:#aaa;text-transform:uppercase;letter-spacing:.05em}
</style></head><body>
<h1>Qwen3-TTS voice audition</h1>
<p class="note">Standalone page — keep it in the repo root next to <code>voice_samples/</code> and
<code>voice_design_audition/</code>. All clips are 24&nbsp;kHz WAV, generated at temperature&nbsp;0.7.
Players are lazy-loaded; click to play.</p>
<nav><b>Jump:</b>
""" + " ".join(f'<a href="#lang-{c}">{LANG_NAME[c]}</a>' for c in LANG_ORDER) +
'<a href="#demo">Demos</a>\n</nav>\n')

# --- Main: grouped by language -> gender, mixing built-in presets + designed voices ---
parts.append('<p class="note">Each language section lists both <span class="src src-builtin">built-in</span> '
             'CustomVoice presets and <span class="src src-designed">designed</span> VoiceDesign voices, '
             'split by gender. Built-ins use the corrected English instruct; designed voices show their prompt '
             'and include a <b>reference</b> clip (the clone source) plus 3 sentences.</p>')

# Lookup: (language, gender_full) -> list of built-in speakers
BUILTIN = {}
for sp, lang, gender, desc in SPEAKERS:
    BUILTIN.setdefault((lang, gender), []).append((sp, desc))


def builtin_card(sp, lang, gender, desc):
    rows = "".join(
        player(f"{CV_DIR}/{sp}_{lang.lower()}_{i}.wav", f"sentence {i}", SENTENCES[lang][i - 1])
        for i in (1, 2, 3))
    if not rows:
        return ""
    return (f'<div class="voice gender-{gender[0]}"><div class="vhead">{esc(sp)}'
            f'<span class="src src-builtin">built-in</span></div>'
            f'<div class="vmeta">{esc(desc)}</div><div class="grid">{rows}</div></div>')


def designed_card(code, lang, gender, vid):
    if (code, gender, vid) in PERSONA_OVERRIDE:
        label, instruct = PERSONA_OVERRIDE[(code, gender, vid)]
    else:
        label = PERSONA_LABEL[(gender, vid)]
        instruct = PERSONA_INSTRUCT[(gender, vid)]
    vidkey = f"{code}_{gender}_{vid}"
    ref = player(f"{AUD_DIR}/{vidkey}_ref.wav", "reference (clone source)", REF_SENTENCE[lang])
    clones = "".join(
        player(f"{AUD_DIR}/{vidkey}_{i}.wav", f"sentence {i}", SENTENCES[lang][i - 1])
        for i in (1, 2, 3))
    if not (ref or clones):
        return ""
    return (f'<div class="voice gender-{gender}"><div class="vhead">{esc(vidkey)}'
            f'<span class="src src-designed">designed</span>'
            f'<span class="pill">{esc(label)}</span></div>'
            f'<div class="prompt">{esc(instruct)}</div>'
            f'<div class="grid">{ref}{clones}</div></div>')


for code in LANG_ORDER:
    lang = LANG_NAME[code]
    parts.append(f'<h2 id="lang-{code}">{esc(lang)}</h2>')
    for gender, gfull in (("m", "male"), ("f", "female")):
        cards = []
        for sp, desc in BUILTIN.get((lang, gfull), []):
            cards.append(builtin_card(sp, lang, gfull, desc))
        for vid in ("v1", "v2", "v3"):
            cards.append(designed_card(code, lang, gender, vid))
        cards = [c for c in cards if c]
        if not cards:
            continue
        parts.append(f'<div class="ghead">{esc(gfull)}</div>')
        parts.extend(cards)

# --- Appendix: design->clone demos ---
parts.append('<h2 id="demo">Demos — why we design-then-clone</h2>')
parts.append('<p class="note">Proof that a <em>pinned</em> designed voice stays consistent across sentences, '
             'vs raw VoiceDesign drifting to a different person each call.</p>')

# EN female + JP male design->clone
demo_defs = [
    ("en_female", "English",  "EN female — designed once, then cloned for each line (same person)"),
    ("jp_male",   "Japanese", "JP male — designed once, then cloned for each line (same person)"),
]
for key, lang, title in demo_defs:
    ref = player(f"{CV_DIR}/design_ref_{key}.wav", "reference (the clip we clone from)", REF_SENTENCE[lang])
    clones = "".join(
        player(f"{CV_DIR}/clone_{key}_{i}.wav", f"clone · sentence {i}", SENTENCES[lang][i - 1])
        for i in (1, 2, 3))
    if not (ref or clones):
        continue
    parts.append(f'<div class="voice"><div class="vhead">{esc(title)}</div>'
                 f'<div class="grid">{ref}{clones}</div></div>')

# JP male drift demo
drift = "".join(
    player(f"{CV_DIR}/voicedesign_jp_male_take{i}.wav", f"take {i} (identical prompt)", SENTENCES["Japanese"][0])
    for i in (1, 2, 3))
if drift:
    parts.append('<div class="voice"><div class="vhead">JP male — raw VoiceDesign, same prompt ×3 '
                 '<span class="pill">drift: 3 different people</span></div>'
                 f'<div class="grid">{drift}</div></div>')

parts.append("</body></html>")

out = os.path.join(ROOT, "audition.html")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))

# quick count of referenced-and-present clips
import glob
n = len(glob.glob(os.path.join(ROOT, CV_DIR, "*.wav"))) + len(glob.glob(os.path.join(ROOT, AUD_DIR, "*.wav")))
print(f"Wrote {out}")
print(f"Audio files available: {n}")
