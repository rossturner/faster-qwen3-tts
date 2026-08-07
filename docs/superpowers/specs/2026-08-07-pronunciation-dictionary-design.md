# Pronunciation dictionary — design

2026-08-07

## Problem

Qwen3-TTS offers no pronunciation control. There is no G2P frontend, no lexicon, no
phoneme or IPA input and no SSML: the upstream `qwen_tts` package contains no
phonemiser, and the tokenizer's 33 added tokens are all plumbing (`<|audio_start|>`,
`<tts_pad>`, …). Request text goes verbatim into the chat template and through the
ordinary Qwen tokenizer.

The only lever that has any purchase is therefore the spelling of the input itself.
`spikes/emotion/text_markup.py` measured typography — ellipses, em-dash, ALL CAPS,
`sooo` — as inert, while lexical content drove the model reliably. So a respelling has
to change *letters*, not punctuation or case, to do anything.

The immediate need is the name **Anby** (and her surname **Demara**) from Zenless Zone
Zero. `anby` is one of the three characters in the shipped library, and the name appears
in Nicole's and Billy's lines as well as her own, so the fix cannot live with one
character.

### Target pronunciation

**AN-bee** `/ˈænbi/` — two syllables, stress on the first, second syllable "bee" not
"by". **Demara** is *de-MAH-ra*. Evidenced by the official localised spellings, which are
phonetic scripts rather than Latin orthography and so carry no ambiguity:

- Japanese: アンビー・デマラ — アン (*an*) + ビー (*bī*, long *ee*)
- Chinese (original): 安比・德玛拉, pinyin *Ānbǐ Démǎlā*
- HoYoDex renders the English as "Anbii"

The failure mode to guard against is the model reading `-by` as /baɪ/.

## Approach

A global substitution dictionary, loaded from its own YAML file and applied to request
text on both synthesis endpoints immediately before the text reaches the model.

Global rather than per-character because the name is spoken by three characters today
and would drift as characters are added; its own file rather than a block inside
`voices.yaml` because `--characters` alone never reads `voices.yaml`, so entries placed
there would be invisible to exactly the deployment that needs them.

## Components

### `faster_qwen3_tts/pronunciations.py` (new, ~60 lines)

Shaped like `audio_filter.py`: a frozen dataclass carrying validated config plus its
compiled form.

- `Pronouncer` — holds the entries and one compiled pattern. `apply(text) -> str`.
- `load_pronunciations(path) -> Pronouncer`
- `EMPTY` — a `Pronouncer` with no entries whose `apply` is the identity, so no consumer
  has to test for `None`.

### `faster_qwen3_tts/server_voices/pronunciations.yaml` (new)

```yaml
pronunciations:
  Anby: "Anbee"
  Demara: "Demarra"
```

Already covered by `pyproject.toml`'s `server_voices/*.yaml` package-data glob;
packaging needs no change.

## Matching

One compiled alternation, applied in a single pass:

```
(?<![A-Za-z])(?:Demara|Anby)(?![A-Za-z])
```

with `re.IGNORECASE`, keys sorted longest-first, and a callback that looks the matched
text up by its lowercased form.

**Single pass, not per-entry `re.sub`.** A loop of substitutions cascades: with
`Demara → Demarra` and a later `Demarra → …` entry, the output of one rule becomes the
input of the next. That is a silent correctness bug that only appears once the dictionary
has a few entries. One pass makes it impossible by construction.

**Longest-first** so overlapping keys resolve deterministically, and so multi-word keys
("Zenless Zone Zero") beat their constituents.

**Boundary is `(?<![A-Za-z])…(?![A-Za-z])`, not `\b`.** With `\b`, `Anbyさん` would not
match, because `さ` is a word character — the rule would silently stop working in exactly
the Japanese and Korean lines this server also serves. The ASCII-letter lookaround matches
there while still leaving `Anbys` and `Banby` alone. `Anby's → Anbee's` falls out for free,
since the apostrophe is not an ASCII letter.

**Replacement case is verbatim.** Matching ignores case, but the output is the configured
string exactly as written, so `ANBY → Anbee`. Preserving the input's capitalisation would
be work spent on something measured to be inert.

## Data flow

Sanitise first, then pronounce, on both endpoints:

- `/v1/audio/speech` — `pronouncer.apply(req.input.strip())`
- `/v1/audio/stream` — `pronouncer.apply(strip_stage_directions(req.input))`

That order means the dictionary sees the final speech text and never rewrites inside a
stage direction that is about to be deleted.

The existing stage-direction asymmetry is deliberately preserved: `strip_stage_directions`
stays stream-only. media-worker sends authored dubbing scripts rather than LLM-written
dialogue, so changing what it does with bracketed text is a behaviour change nobody asked
for. The pronunciation pass, by contrast, runs on both — a name is a name on either
endpoint, and a global dictionary that silently did nothing on one of them would be a trap.

Wiring:

- `build_app(manager, registry, serve_page=..., pronouncer=EMPTY)` — defaulting to `EMPTY`
  keeps every existing test untouched.
- `create_app(..., pronunciations_path=None)` — `None` means the bundled file.
- `serve-http --pronunciations PATH`.

## Error handling

Every malformed input is a fatal `ValueError` at load: non-string or empty key or value,
leading or trailing whitespace in a key, two keys colliding case-insensitively (the
lowercased lookup would be ambiguous), unknown top-level keys, and a missing file.

This follows `voices.yaml`'s policy rather than the character library's skip-and-warn, for
the reason already stated in `characters.py`: filesystem-discovered content gets the
lenient treatment, deliberate config does not, and a typo in a hand-written file should
not be shrugged off.

## Testing

`tests/test_pronunciations.py`, no GPU:

- case-insensitive matching: `Anby`, `anby`, `ANBY`, `AnBy` all → `Anbee`
- possessive: `Anby's` → `Anbee's`
- negative boundaries: `Anbys` and `Banby` unchanged
- CJK adjacency: `Anbyさん` → `Anbeeさん`
- no cascade: with `a → b` and `b → c`, `a` yields `b`
- longest-first on overlapping keys
- replacement emitted verbatim: `ANBY` → `Anbee`
- `EMPTY.apply` is the identity
- each validation failure raises `ValueError`

Integration, through the existing `TestClient` harness in `tests/test_server.py`:
`FakeManager.calls` already records the text handed to the model, so one assertion per
endpoint confirms the substitution actually reached synthesis rather than just the
request object.

## Out of scope

**Reference transcripts stay untouched.** Two contain the name —
`characters/anby/excited/Galgame_Chapter0_Anbi_05.txt` and
`characters/nicole/annoyed/GalGame_Chapter030_Nicole_020_014.txt`. A transcript's job is
to be an accurate transcription of its audio for ICL; rewriting it to `Anbee` would change
measured-good conditioning data to fix a problem on a different input entirely.

**`MAX_INPUT_CHARS` stays a check on the raw input.** Substitution can lengthen text, so a
request may end up a few characters over the bound. That bound exists to protect prefill
against `max_seq_len=2048`, and a handful of characters is immaterial to it.

## Known risk

The shipped values are a first-principles guess. Nobody has heard the model say either
name, so if it already pronounces `Anby` correctly then `Anbee` is a regression — and no
test can catch that, only ears. This was an explicit call: the values are one config file,
so correcting them is a one-line edit and a restart.

The follow-up that would close it is an audition in the manner of the Billy filter: the
raw spellings against three or four candidate respellings, across the anby/nicole/billy
voices, several takes each since sampling is unseeded.
