# media-worker TTS voice mapping — FINAL selection (2026-06-01)

The 12 chosen voices for the `media-worker` dubbing pipeline: 6 languages × male/female.
Two come from **CustomVoice built-in speakers**; the other ten are **VoiceDesign-designed then
pinned via Base voice-clone** (one reference clip per voice, reused for every line so the narrator
is consistent). All generated at `temperature=0.7`. Audition everything in `audition.html`.

## The 12 slots

| Language | Gender | Method | Voice / clip |
| --- | --- | --- | --- |
| English  | male   | CustomVoice built-in | `aiden` |
| English  | female | VoiceDesign → clone  | `en_f_v1` (`voice_design_audition/en_f_v1_ref.wav`) |
| Spanish  | male   | VoiceDesign → clone  | `es_m_v2` (`voice_design_audition/es_m_v2_ref.wav`) |
| Spanish  | female | VoiceDesign → clone  | `es_f_v3` (`voice_design_audition/es_f_v3_ref.wav`) |
| French   | male   | VoiceDesign → clone  | `fr_m_v1` (`voice_design_audition/fr_m_v1_ref.wav`) |
| French   | female | VoiceDesign → clone  | `fr_f_v3` (`voice_design_audition/fr_f_v3_ref.wav`) |
| Chinese  | male   | VoiceDesign → clone  | `zh_m_v3` (`voice_design_audition/zh_m_v3_ref.wav`) |
| Chinese  | female | VoiceDesign → clone  | `zh_f_v3` (`voice_design_audition/zh_f_v3_ref.wav`) |
| Japanese | male   | VoiceDesign → clone  | `ja_m_v1` (`voice_design_audition/ja_m_v1_ref.wav`) |
| Japanese | female | VoiceDesign → clone  | `ja_f_v3` (`voice_design_audition/ja_f_v3_ref.wav`) |
| Korean   | male   | VoiceDesign → clone  | `ko_m_v3` (`voice_design_audition/ko_m_v3_ref.wav`) |
| Korean   | female | CustomVoice built-in | `sohee` |

## Built-in voices (CustomVoice)

Call `generate_custom_voice(text, speaker, language, instruct=...)`. Tone instruct (English only —
the model only accepts English/Chinese instructions):
`"Speak in a calm, clear, professional tone suitable for an instructional video."`

| Voice | Language | Notes |
| --- | --- | --- |
| `aiden` | English | Sunny American male, clear midrange |
| `sohee` | Korean  | Warm Korean female, rich emotion |

## Designed voices (VoiceDesign → Base clone)

Each was created by designing a reference clip with `generate_voice_design(text=<ref line>,
instruct=<persona>, language, temperature=0.7)`, then cloned for production via
`generate_voice_clone(text, language, ref_audio=<ref clip>, ref_text=<ref line>, xvec_only=False)`.
The persona `instruct` is English-only and independent of output language.

Reference line per language (the `ref_text` for cloning):
- English: "Welcome to the course. Let's get started with today's lesson."
- Spanish: "Bienvenido al curso. Vamos a empezar con la lección de hoy."
- French:  "Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui."
- Chinese: "欢迎来到本课程。让我们开始今天的课程吧。"
- Japanese:"このコースへようこそ。今日のレッスンを始めましょう。"
- Korean:  "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다."

| Voice | Persona | `instruct` |
| --- | --- | --- |
| `en_f_v1` | mid-warm | A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. |
| `es_m_v2` | young-bright | A bright, approachable younger man's voice in a higher tenor range, with clear articulation and a relaxed, friendly delivery, suitable for an online tutorial course. |
| `es_f_v3` | mature-smooth | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. |
| `fr_m_v1` | mid-warm | A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. |
| `fr_f_v3` | mature-smooth | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. |
| `zh_m_v3` | mature-deep | A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video. |
| `zh_f_v3` | mature-smooth | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. |
| `ja_m_v1` | mid-warm | A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. |
| `ja_f_v3` | mature-smooth | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. |
| `ko_m_v3` | mature-deep | A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video. |

## Notes / open items

- **`es_m_v2`** uses the `young-bright` persona ("higher tenor range") — confirmed as-is by ear
  (the high/fast issue seen elsewhere wasn't a problem for this Spanish-male take).
- The reference clips (`*_ref.wav`) are single un-cherry-picked VoiceDesign takes. For production,
  build a clone prompt from each chosen `_ref.wav` once and reuse it (design-once → pin → reuse).
- Two models needed at serve time: **CustomVoice** (for `aiden`, `sohee`) + **Base** (for the 10 clones).
  Alternatively pre-bake clone prompts to disk so only Base is resident alongside CustomVoice.
