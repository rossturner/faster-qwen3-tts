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

The 12 voices in `voices.yaml` are 6 languages (English, Spanish, French, Chinese,
Japanese, Korean) × male/female, **all `type: clone`**. Each was created by
**VoiceDesign → Base clone**: a persona was rendered once with
`generate_voice_design(...)` into a reference clip, then that clip is the clone
source for every production line so the narrator stays consistent.

The registry is therefore Base-only at serve time — no CustomVoice model is loaded.
(The server still loads CustomVoice automatically if a `type: custom` voice is ever
added back to `voices.yaml`; today none are.)

The canonical 12-voice selection, with the full persona `instruct` text per voice,
lives in [`../../voice-mapping.md`](../../voice-mapping.md).

## Generation scripts

- [`../../design_audition_library.py`](../../design_audition_library.py) — repo
  root. Rendered the designed VoiceDesign reference families (the `*_v{1,2,3}_ref.wav`
  audition takes); the chosen take per voice was copied into `refs/`.
- [`../../redo_ja_female.py`](../../redo_ja_female.py) — repo root. Re-rendered the
  Japanese-female family; `ja_f_v3_ref.wav` came from this run.
- [`../../design_replacement_audition.py`](../../design_replacement_audition.py) —
  repo root. Designed the `friendly_casual` candidates that replaced the former
  CustomVoice slots; `ko_f_friendly_casual_ref.wav` came from this run (the original
  `en_m_friendly_casual_ref.wav` from this run was later superseded — see below).
- [`../../design_en_m_calm_audition.py`](../../design_en_m_calm_audition.py) — repo
  root. Calm/clear English-male replacement audition (designs each persona, clones
  the first 4 chunks of a real dub off each ref); `en_m_confident_mid_ref.wav` came
  from this run, replacing the former `friendly_casual` en_m.
- [`../../gen_voice_samples.py`](../../gen_voice_samples.py),
  [`../../generate_audition_html.py`](../../generate_audition_html.py) — repo root.
  Supporting sample-generation and audition-page tooling.

## Per-clip provenance

`instruct` is the VoiceDesign / CustomVoice persona prompt (English only — the
model accepts only English/Chinese instructions). `ref line` is the spoken
text and the `ref_text` used when cloning.

| Ref clip | Voice id(s) | Generating script | Source method | Persona `instruct` | Reference line |
| --- | --- | --- | --- | --- | --- |
| `en_m_confident_mid_ref.wav` | `en_m` | `design_en_m_calm_audition.py` | VoiceDesign (English) | A confident, steady adult man's voice in a clear mid-range, assured and grounded, with a natural, brisk-but-composed pace, suitable for an instructional video. | "In today's lesson we'll work through each step slowly and carefully, so take your time and follow along at your own pace as we go." |
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
| `ko_f_friendly_casual_ref.wav` | `ko_f` | `design_replacement_audition.py` | VoiceDesign (Korean) | A relaxed, approachable woman's voice in a natural mid-range with a warm, conversational tone, easy-going and personable, suitable for a friendly tutorial. | "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다." |

`en_m` and `ko_f` were originally the CustomVoice presets `aiden` / `sohee`; they
were replaced by designed voices, so every shipped voice now clones from a reference
clip and the server runs Base-only. (`en_m` was briefly the `friendly_casual` designed
voice before being re-auditioned to the calmer, more present `confident_mid` above.)
