# Voice design — canonical reference

The single source of truth for **how the media-worker TTS voices were designed and
what is shipped**. Machine-readable config lives in
[`faster_qwen3_tts/server_voices/voices.yaml`](../faster_qwen3_tts/server_voices/voices.yaml);
per-clip provenance lives in
[`faster_qwen3_tts/server_voices/README.md`](../faster_qwen3_tts/server_voices/README.md).
This doc is the narrative + final selection that ties them together.

## Method: VoiceDesign → Base clone (design-once → pin → reuse)

Every shipped voice is produced the same way:

1. **Design** a reference clip once with the **VoiceDesign** 1.7B model
   (`generate_voice_design(text=<ref line>, instruct=<persona>, language, temperature=0.7)`).
2. **Pin** that clip as the canonical reference (`server_voices/refs/*.wav`).
3. **Clone** every production line from it with the **Base** 1.7B model
   (`generate_voice_clone(text, language, ref_audio=<ref clip>, ref_text=<ref line>, xvec_only=False)`),
   so the narrator stays identical call-to-call.

Consequences:
- **Serving is Base-only.** No CustomVoice model is loaded (the registry still
  supports `type: custom` if one is ever re-added; today none are).
- All voices are `type: clone`. The `.wav` ref clips are the **canonical artifacts** —
  generation is unseeded, so re-running a design script reproduces the *method*, not
  the exact waveform.
- All clips generated at `temperature=0.7`, 24 kHz output.

## The `instruct` rule (read before designing)

`instruct` is the **persona description**, and per Alibaba's VoiceDesign docs it must be
written in **English or Chinese only**. It is **independent of the output language**: a
Spanish/French/Japanese/Korean voice still takes an *English* instruct. Instruct written
in the output language (e.g. Korean) is effectively ignored. This is the most common
mistake — the output language is set by the `language` argument and the reference text,
**not** by the instruct.

## Reference line

The reference line is the text the VoiceDesign clip speaks; it also becomes the
`ref_text` used for cloning. A **longer reference clip gives the Base clone a richer
speaker basis**.

- **Original lines** were short "Welcome to the course…" style sentences (~3.5–5.5 s clips).
- **English (en_m, en_f) now use a longer line** (~7 s clip):
  > "In today's lesson we'll work through each step slowly and carefully, so take your time and follow along at your own pace as we go."
- **Other languages still use the short line** (longer-ref A/B in review — see Open items).

Caveat learned the hard way: when translating a longer reference line, keep it to ~7 s.
The first non-English longref pass rendered 10–12 s clips (translations were too verbose),
which perturbed those voices more than a length-matched line would.

## Shipped voices (12 = 6 languages × male/female)

All `type: clone`, Base-only at serve time. `instruct` is English (the persona); the
reference line is in the target language.

| Voice | Lang | Persona | Ref clip (`server_voices/refs/`) | `instruct` (English) | Reference line |
| --- | --- | --- | --- | --- | --- |
| `en_m` | English | confident-mid | `en_m_confident_mid_ref.wav` | A confident, steady adult man's voice in a clear mid-range, assured and grounded, with a natural, brisk-but-composed pace, suitable for an instructional video. | (English longer line above) |
| `en_f` | English | mid-warm | `en_f_longref_ref.wav` | A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. | (English longer line above) |
| `es_m` | Spanish | young-bright | `es_m_v2_ref.wav` | A bright, approachable younger man's voice in a higher tenor range, with clear articulation and a relaxed, friendly delivery, suitable for an online tutorial course. | Bienvenido al curso. Vamos a empezar con la lección de hoy. |
| `es_f` | Spanish | mature-smooth | `es_f_v3_ref.wav` | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | Bienvenido al curso. Vamos a empezar con la lección de hoy. |
| `fr_m` | French | mid-warm | `fr_m_v1_ref.wav` | A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. | Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui. |
| `fr_f` | French | mature-smooth | `fr_f_v3_ref.wav` | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui. |
| `zh_m` | Chinese | mature-deep | `zh_m_v3_ref.wav` | A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video. | 欢迎来到本课程。让我们开始今天的课程吧。 |
| `zh_f` | Chinese | mature-smooth | `zh_f_v3_ref.wav` | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | 欢迎来到本课程。让我们开始今天的课程吧。 |
| `ja_m` | Japanese | mid-warm | `ja_m_v1_ref.wav` | A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. | このコースへようこそ。今日のレッスンを始めましょう。 |
| `ja_f` | Japanese | mature-smooth | `ja_f_v3_ref.wav` | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. | このコースへようこそ。今日のレッスンを始めましょう。 |
| `ko_m` | Korean | mature-deep | `ko_m_v3_ref.wav` | A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video. | 이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다. |
| `ko_f` | Korean | friendly-casual | `ko_f_friendly_casual_ref.wav` | A relaxed, approachable woman's voice in a natural mid-range with a warm, conversational tone, easy-going and personable, suitable for a friendly tutorial. | 이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다. |

## Generation / audition scripts (repo root)

- `design_audition_library.py` — designed the original per-language families
  (`*_v{1,2,3}` takes); the chosen take per voice was copied into `server_voices/refs/`.
- `redo_ja_female.py` — re-rendered the Japanese-female family (`ja_f_v3`).
- `design_replacement_audition.py` — designed the `friendly_casual` candidates that
  replaced the former CustomVoice presets; `ko_f_friendly_casual_ref.wav` came from here.
- `design_en_m_calm_audition.py` — calm/clear English-male re-audition; produced
  `en_m_confident_mid_ref.wav`. Designs each persona, then clones the first 4 chunks of a
  real dub off each ref.
- `compare_longref_audition.py` — A/B of current (short-ref) voices vs the same persona
  re-designed from a longer reference line; produced `en_f_longref_ref.wav` and the
  non-English longref comparison set.

## History

- **2026-06-01** — Original 12-voice selection (`design_audition_library.py`), all
  VoiceDesign → Base clone, short reference lines. CustomVoice dropped; serving Base-only.
- **2026-06-02** — `en_m` / `ko_f` (formerly CustomVoice `aiden` / `sohee`) replaced by
  designed `friendly_casual` voices (`design_replacement_audition.py`).
- **2026-06-10** — `en_m` re-auditioned `friendly_casual` → `confident_mid` (the previous
  voice read as too energetic). Both English voices moved to a **longer reference line**;
  `en_f` switched to `en_f_longref`. Non-English longref A/B run for review.

## Open items

- **Non-English longer-reference A/B** is under review (`voice_design_longref_compare/`,
  rendered by `compare_longref_audition.py`). Each non-English voice was re-designed from a
  ~7 s reference line for comparison against the current short-ref production voice; not yet
  promoted. The non-English longref takes are unseeded fresh rolls, so the comparison judges
  both "longer ref" and "this particular re-roll's character" at once.
