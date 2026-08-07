# Pronunciation dictionary — design

2026-08-07

Revised after adversarial review; the review's fixes are folded in and noted where they
changed a decision.

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

### `faster_qwen3_tts/pronunciations.py` (new, ~80 lines)

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

Already covered by `pyproject.toml`'s `server_voices/*.yaml` package-data glob, which
`tests/test_packaging.py` already asserts matches real files; packaging needs no change.

### `CLAUDE.md` (updated)

A new subsection under the serving documentation. CLAUDE.md documents
`strip_stage_directions` in detail because a silent rewrite of caller text is a surprise;
a second such rewrite, plus a new config file and CLI flag, belongs in the same place.

## Matching

One compiled alternation, applied in a single pass:

```
(?<![LETTER])(?:Demara|Anby)(?![LETTER])
```

with `re.IGNORECASE`, keys `re.escape`d, sorted longest-first, and a callback that looks
the matched text up by its lowercased form.

**Keys are `re.escape`d before joining.** Without it a key containing `.`, `(`, `+` or
`*` either matches the wrong thing — a key `Mr.` rewrites `Mrs Smith`, because `.`
matches the `s` and the space satisfies the trailing lookaround — or raises `re.error` at
compile time, which is not the `ValueError` the error-handling section promises. Pattern
compilation is additionally wrapped so any residual `re.error` surfaces as `ValueError`.

**The lookup callback must be total:** `mapping.get(matched.lower(), matched)`, never
`mapping[...]`. Under `re.IGNORECASE`, `İ` (U+0130) matches the pattern letter `i` but
`'İ'.lower()` is two codepoints, and `ſ` (U+017F) matches `s` but lowercases to itself.
Both were verified against this repo's venv. A bare subscript would raise `KeyError`
inside `re.sub` and surface as an uncaught 500 mid-request.

**Single pass, not per-entry `re.sub`.** A loop of substitutions cascades: with
`Demara → Demarra` and a later `Demarra → …` entry, the output of one rule becomes the
input of the next. That is a silent correctness bug that only appears once the dictionary
has a few entries. One pass makes it impossible by construction.

**Longest-first** so that alternatives *starting at the same offset* resolve to the
longest. This is narrower than "multi-word keys beat their constituents": `re` is
leftmost-first across positions, so with keys `Zenless Zone` and `Zone Zero` the input
`Zenless Zone Zero` matches the former and the latter never gets a chance. Deterministic,
but not longest-overall.

### Boundary

`LETTER` is Latin script generally, not `[A-Za-z]`:

```
A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F
```

That is ASCII letters, Latin-1 Supplement, Latin Extended-A and -B, Latin Extended
Additional, and the combining diacritical marks.

Two failure modes are being avoided at once, and the first draft of this spec avoided
only one of them:

- `\b` would not match `Anbyさん`, because `さ` is a word character — the rule would
  silently stop working in the Japanese and Korean lines this server also serves.
- A bare `[A-Za-z]` lookaround matches `Anbyé`, and a key `Ana` would rewrite inside
  `Anaïs` — in the French, Spanish, German, Portuguese and Italian lines this server
  *also* serves. Extending through Latin-1 Supplement, Latin Extended-A/B, Latin
  Extended Additional and the combining diacritical marks closes it.

`Anbys` and `Banby` remain untouched. `Anby's → Anbee's` falls out for free, since the
apostrophe is not a letter. Digits are deliberately not boundaries: `Anby2 → Anbee2`.

**Replacement case is verbatim.** Matching ignores case, but the output is the configured
string exactly as written, so `ANBY → Anbee`. Preserving the input's capitalisation would
be work spent on something measured to be inert.

### Key charset

Keys are validated at load to Latin letters (the class above), spaces, apostrophes and
hyphens; anything else is a fatal `ValueError`. The boundary rule is built out of letter
lookarounds, so a CJK key would degenerate to a bare substring match and fire inside any
longer kana or hanzi run. A rule that quietly misbehaves is worse than one that refuses to
load, so non-Latin keys are rejected rather than half-supported.

Replacement *values* carry no charset restriction beyond being non-empty strings — the
value is what gets spoken, and a kana respelling of a Latin key is a coherent thing to
want.

## Data flow

Sanitise first, then pronounce, on both endpoints. The insertion point is pinned relative
to the existing validation, because getting it wrong silently changes what
`MAX_INPUT_CHARS` bounds:

- `/v1/audio/speech` — after the empty check and *after* the `MAX_INPUT_CHARS` check
  (`server.py:317-321`), as its own statement: `text = pronouncer.apply(text)`.
- `/v1/audio/stream` — `pronouncer.apply(strip_stage_directions(req.input))`. The length
  check here already runs against `req.input` at `server.py:353`, before any
  transformation, so no reordering is needed.

Both endpoints therefore bound the *raw* caller input, and substitution happens after.

On the stream endpoint, sanitise-then-pronounce also means the dictionary sees the final
speech text and never rewrites inside a stage direction that is about to be deleted. That
rationale does not extend to `/v1/audio/speech`, which does no stripping at all — there,
a name inside `[Anby enters]` *is* rewritten. That follows from the deliberate decision
below to leave the stripping asymmetry alone.

The existing stage-direction asymmetry is preserved: `strip_stage_directions` stays
stream-only. media-worker sends authored dubbing scripts rather than LLM-written dialogue,
so changing what it does with bracketed text is a behaviour change nobody asked for. The
pronunciation pass, by contrast, runs on both — a name is a name on either endpoint, and a
dictionary that silently did nothing on one of them would be a trap.

The `clone` / `custom` / `design` / `serve` CLI subcommands bypass the dictionary
entirely. It is a property of the HTTP service, not of the library.

## Loading

Opt-in, mirroring `--characters` exactly:

```
sp.add_argument("--pronunciations", nargs="?", const="BUNDLED", default=None, ...)
```

resolved in `cmd_serve_http` the same way `--characters` is (`cli.py:311-312`). Bare flag
loads the bundled file; a PATH loads that file; absent loads nothing and the server runs
with `EMPTY`.

This was reconsidered during review. An always-on bundled default would put unauditioned
ZZZ name respellings on media-worker's dubbing server with no way to switch them off, and
would make `None` mean the opposite of what it means for `voices_path` immediately beside
it in the same signature. Opt-in matches the convention CLAUDE.md already states for the
registry — "the bundled registry is a default, not a floor".

Wiring:

- `build_app(manager, registry, serve_page=..., pronouncer=EMPTY)` — the `EMPTY` default
  keeps every existing call site working.
- `create_app(..., pronunciations_path=None)` — `None` means no dictionary.

## Error handling

Every malformed input is a fatal `ValueError` at load: non-string or empty key or value,
leading or trailing whitespace in a key, a key outside the permitted charset, two keys
colliding case-insensitively (the lowercased lookup would otherwise be ambiguous), unknown
top-level keys, a file that does not exist, and any pattern that fails to compile.

This follows `voices.yaml`'s policy rather than the character library's skip-and-warn, for
the reason already stated in `characters.py`: filesystem-discovered content gets the
lenient treatment, deliberate config does not, and a typo in a hand-written file should
not be shrugged off.

**An absent or empty `pronunciations:` mapping is not an error** — it yields a
`Pronouncer` equivalent to `EMPTY`. The loader must special-case zero entries and not
compile an alternation, because `(?<![LETTER])(?:)(?![LETTER])` matches the empty string
at every non-letter position. An empty dictionary is a legitimate state: it is how a
deployment that passed the flag turns the feature off without removing the flag.

## Testing

`tests/test_pronunciations.py`, no GPU:

- case-insensitive matching: `Anby`, `anby`, `ANBY`, `AnBy` all → `Anbee`
- possessive: `Anby's` → `Anbee's`
- negative boundaries: `Anbys`, `Banby` unchanged
- accented-Latin adjacency: `Anbyé` and `Anaïs` (with a key `Ana`) unchanged
- CJK adjacency: `Anbyさん` → `Anbeeさん`
- digits are not boundaries: `Anby2` → `Anbee2`
- no cascade: with `a → b` and `b → c`, `a` yields `b`
- longest-first at a shared offset
- replacement emitted verbatim: `ANBY` → `Anbee`
- a key containing regex metacharacters matches literally
- a key containing `i` or `s` does not raise on `İ` / `ſ` input
- zero entries: identity, and no empty-alternation match
- `EMPTY.apply` is the identity
- each validation failure raises `ValueError`

Integration. The two endpoints use **different** harnesses, and both need a change before
any assertion is possible:

- `/v1/audio/speech` — `FakeManager` + `client()` in `tests/test_server.py:18-28`, whose
  `calls` records `(cfg.id, text)`.
- `/v1/audio/stream` — `FakeStreamManager` + `_stream_client()` in
  `tests/test_server_stream.py:164-208`, whose `calls` records
  `(cfg.key, text, temperature, chunk_size)`. `FakeManager` has no `synthesize_stream`
  and cannot serve this endpoint.

Both helpers currently call `build_app(mgr, registry)` with no pronouncer, so each needs
an optional pronouncer parameter threaded through. One assertion per endpoint then
confirms the substitution reached synthesis rather than only the request object.

## Out of scope

**Reference transcripts stay untouched.** Two contain the name —
`characters/anby/excited/Galgame_Chapter0_Anbi_05.txt` and
`characters/nicole/annoyed/GalGame_Chapter030_Nicole_020_014.txt`. A transcript's job is
to be an accurate transcription of its audio for ICL; rewriting it to `Anbee` would change
measured-good conditioning data to fix a problem on a different input entirely.

A consequence worth naming: on requests that happen to draw one of those two recordings,
the ICL prompt says `Anby` (paired with audio of an actor saying it correctly) while the
target text says `Anbee`. Because `pick_reference` is random, whether that mismatch occurs
varies per request. It is accepted, but it is the reason the audition below must vary the
reference and not only the voice.

**`MAX_INPUT_CHARS` stays a check on the raw input**, per the pinned insertion points
above. Substitution can lengthen text, so a request may end up a few characters over the
bound. That bound exists to protect prefill against `max_seq_len=2048`, and a handful of
characters is immaterial to it.

**Nothing reports that a substitution occurred.** No `X-TTS-*` header and no field in the
`0x01` header frame. This was offered and declined; it sits against the codebase's stated
principle that an invisible substitution is a trap, and is the obvious first addition if
debugging ever needs it.

## Known risk

The shipped values are a first-principles guess. Nobody has heard the model say either
name, so if it already pronounces `Anby` correctly then `Anbee` is a regression — and no
test can catch that, only ears. This was an explicit call: the values are one config file,
so correcting them is a one-line edit and a restart.

The follow-up that would close it is an audition in the manner of the Billy filter: the
raw spellings against three or four candidate respellings, across the anby/nicole/billy
voices and — per the note above — across different reference draws within a voice, several
takes each since sampling is unseeded.
