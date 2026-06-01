# Voice-design audition library

A selection set of **designed instructional-narrator voices** for every target language × gender.
Pick the voice(s) you like here; each voice's `*_ref.wav` is the clip we'll clone from in production
(via the Base model), exactly like the EN-female / JP-male demo.

- **Languages:** English, Spanish, French, Chinese, Japanese, Korean
- **Genders:** male (`m`), female (`f`)
- **Personas per gender:** 3 (a pitch/age spread, all calm + professional)
- **Total:** 6 × 2 × 3 = **36 voices**
- Generated with the **VoiceDesign 1.7B** model at `temperature=0.7`, then cloned with **Base 1.7B** (ICL).

## File naming

```
{lang}_{gender}_{persona}_ref.wav     # the designed reference clip = the clone source
{lang}_{gender}_{persona}_1.wav       # illustration sentence 1, cloned from the reference
{lang}_{gender}_{persona}_2.wav       # illustration sentence 2  (same person)
{lang}_{gender}_{persona}_3.wav       # illustration sentence 3  (same person)
```

Lang codes: `en es fr zh ja ko`. Example: `ja_m_v2_1.wav` = Japanese, male, persona v2, sentence 1.

> The `_1/2/3` clones are the **same person** across all three sentences (that's the point — a stable
> narrator). The `_ref` clip is what gets cloned in production.

## Personas (the prompt behind each voice)

The same 3 persona prompts per gender are applied to all 6 languages (`language` is set separately;
the instruct only describes the voice). `v2` is the higher/younger option, added because the first
JP-male came out too deep.

### Male
| Persona | Label | `instruct` |
|---|---|---|
| `m_v1` | mid-warm | A clear, warm adult man's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. |
| `m_v2` | young-bright | A bright, approachable younger man's voice in a higher tenor range, with clear articulation and a relaxed, friendly delivery, suitable for an online tutorial course. |
| `m_v3` | mature-deep | A mature, confident man's voice with a warm lower-mid tone and a steady, measured delivery, suitable for a professional instructional video. |

### Female
| Persona | Label | `instruct` |
|---|---|---|
| `f_v1` | mid-warm | A clear, warm adult woman's voice in a natural mid-range, calm and professional, with a relaxed, friendly delivery, suitable for narrating an instructional video course. |
| `f_v2` | young-bright | A bright, friendly younger woman's voice with a light, articulate and approachable delivery, suitable for an online tutorial course. |
| `f_v3` | mature-smooth | A warm, mature woman's voice with a smooth lower-mid tone and a steady, reassuring delivery, suitable for a professional instructional video. |

## Sentences

**Reference line** (used to design the `_ref.wav`, per language):
- EN: Welcome to the course. Let's get started with today's lesson.
- ES: Bienvenido al curso. Vamos a empezar con la lección de hoy.
- FR: Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui.
- ZH: 欢迎来到本课程。让我们开始今天的课程吧。
- JA: このコースへようこそ。今日のレッスンを始めましょう。
- KO: 이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다.

**Illustration sentences** (`_1/_2/_3`, an illustration-course theme): sketching basic shapes →
brush opacity + layered shading → light source defining shadows. Translated per language; see
`design_audition_library.py` for the exact strings.

## Notes
- Each `_ref` is a single VoiceDesign take (temp 0.7, no cherry-picking). If a persona's timbre is
  close but not quite right, re-rolling the design gives a different take within the same brief.
- Once you select voices, the production path is: keep the chosen `_ref.wav` (or a re-rolled better
  take), build a clone prompt from it once, and reuse it for every line.
