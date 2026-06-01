# server_voices — production voice registry

This directory is the source-controlled data layer for the OpenAI-compatible TTS
server. It holds the voice registry (`voices.yaml`) and the reference audio clips
(`refs/*.wav`) those voices clone from.

## The `.wav` binaries are canonical

Generation is stochastic with **no fixed seed**, so re-running the generation
scripts will NOT reproduce the exact clips committed here — it reproduces the
*method*, not the waveform. The committed `refs/*.wav` files are the authoritative
artifacts; the scripts below document how they were made and let you regenerate
the family if needed (then re-audition by ear). All clips were generated at
`temperature=0.7`.

## Voice sources

The 14 voices in `voices.yaml` break down as:

- **12 production voices** — 6 languages (English, Spanish, French, Chinese,
  Japanese, Korean) × male/female. Two of these (`en_m` = aiden, `ko_f` = sohee)
  are CustomVoice built-in presets, called directly with `type: custom`. The other
  ten were created by **VoiceDesign → Base clone**: a persona was rendered once
  with `generate_voice_design(...)` into a reference clip, then that clip is used
  as the clone source (`type: clone`) for every production line so the narrator
  stays consistent.
- **2 clone-comparison variants** — `en_m_clone` / `ko_f_clone`. These clone from
  reference clips rendered from the `aiden` / `sohee` CustomVoice presets (via
  `scripts/make_builtin_clone_refs.py`), so the built-in voices can also be served
  through the Base clone path and compared against the direct CustomVoice route.

The canonical 12-voice selection, with the full persona `instruct` text per voice,
lives in [`../../voice-mapping.md`](../../voice-mapping.md).

## Generation scripts

- [`../../design_audition_library.py`](../../design_audition_library.py) — repo
  root. Rendered the designed VoiceDesign reference families (the `*_v{1,2,3}_ref.wav`
  audition takes); the chosen take per voice was copied into `refs/`.
- [`../../redo_ja_female.py`](../../redo_ja_female.py) — repo root. Re-rendered the
  Japanese-female family; `ja_f_v3_ref.wav` came from this run.
- [`../../scripts/make_builtin_clone_refs.py`](../../scripts/make_builtin_clone_refs.py)
  — rendered `aiden_ref.wav` / `sohee_ref.wav` from the CustomVoice presets.
- [`../../gen_voice_samples.py`](../../gen_voice_samples.py),
  [`../../generate_audition_html.py`](../../generate_audition_html.py) — repo root.
  Supporting sample-generation and audition-page tooling.

## Per-clip provenance

`instruct` is the VoiceDesign / CustomVoice persona prompt (English only — the
model accepts only English/Chinese instructions). `ref line` is the spoken
text and the `ref_text` used when cloning.

| Ref clip | Voice id(s) | Generating script | Source method | Persona `instruct` | Reference line |
| --- | --- | --- | --- | --- | --- |
| `en_f_v1_ref.wav` | `en_f` | `design_audition_library.py` | VoiceDesign (English) | A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. | "Welcome to the course. Let's get started with today's lesson." |
| `es_m_v2_ref.wav` | `es_m` | `design_audition_library.py` | VoiceDesign (Spanish) | A bright, approachable younger man's voice in a higher tenor range, with clear articulation and a relaxed, friendly delivery, suitable for an online tutorial course. | "Bienvenido al curso. Vamos a empezar con la lección de hoy." |
| `es_f_v3_ref.wav` | `es_f` | `design_audition_library.py` | VoiceDesign (Spanish) | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | "Bienvenido al curso. Vamos a empezar con la lección de hoy." |
| `fr_m_v1_ref.wav` | `fr_m` | `design_audition_library.py` | VoiceDesign (French) | A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. | "Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui." |
| `fr_f_v3_ref.wav` | `fr_f` | `design_audition_library.py` | VoiceDesign (French) | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | "Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui." |
| `zh_m_v3_ref.wav` | `zh_m` | `design_audition_library.py` | VoiceDesign (Chinese) | A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video. | "欢迎来到本课程。让我们开始今天的课程吧。" |
| `zh_f_v3_ref.wav` | `zh_f` | `design_audition_library.py` | VoiceDesign (Chinese) | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | "欢迎来到本课程。让我们开始今天的课程吧。" |
| `ja_m_v1_ref.wav` | `ja_m` | `design_audition_library.py` | VoiceDesign (Japanese) | A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. | "このコースへようこそ。今日のレッスンを始めましょう。" |
| `ja_f_v3_ref.wav` | `ja_f` | `redo_ja_female.py` | VoiceDesign (Japanese) | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | "このコースへようこそ。今日のレッスンを始めましょう。" |
| `ko_m_v3_ref.wav` | `ko_m` | `design_audition_library.py` | VoiceDesign (Korean) | A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video. | "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다." |
| `aiden_ref.wav` | `en_m_clone` | `scripts/make_builtin_clone_refs.py` | CustomVoice preset `aiden` | Speak in a calm, clear, professional tone suitable for an instructional video. | "Welcome to the course. Let's get started with today's lesson." |
| `sohee_ref.wav` | `ko_f_clone` | `scripts/make_builtin_clone_refs.py` | CustomVoice preset `sohee` | Speak in a calm, clear, professional tone suitable for an instructional video. | "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다." |

The `en_m` (aiden) and `ko_f` (sohee) production voices use no reference clip —
they call the CustomVoice preset directly. Their clone-comparison counterparts
(`en_m_clone`, `ko_f_clone`) clone from `aiden_ref.wav` / `sohee_ref.wav` above.
