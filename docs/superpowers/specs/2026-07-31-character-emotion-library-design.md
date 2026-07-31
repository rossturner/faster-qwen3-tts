# Character emotion reference library — design

## Goal

Serve named characters whose emotion comes from the reference recording, not from an
instruct string. A character is a directory; an emotion is a subdirectory; the recordings
inside it are interchangeable takes of that emotion. A request names a character and an
emotion, the server picks one recording at random, and clones from it.

Add a browser page that auditions this and shows streaming timing, so a character's
emotion set can be judged by ear and by latency.

## Why this shape

The spike (`docs/lyrebird-tts-spike-findings.md`, Experiment 3) established that on the
ICL clone path `instruct` moves speaking rate and nothing else — pitch and energy come
from the reference recording, which outvotes it. Emotion on a cloned voice therefore has
to be carried by the clip.

That spike's Recommendation 5 nevertheless rejected emotion-from-clips and chose the
`ono_anna` built-in voice directed by `instruct`. Its stated reason was specific:
*"VoiceDesign cannot hold an identity across the calls needed to build a clip set"* — the
clip set would have had to be **synthesised**, and the generator drifted between calls.

This work removes that premise rather than contradicting the finding. The clips are real
recordings of three characters, so identity is fixed by the recording rather than by a
generator that cannot hold one. The measured constraint that emotion must live in the clip
is unchanged and is exactly what this design implements. It does supersede Recommendation
5's choice of route; `voices_lyrebird.yaml` stays in the tree as the record of that route,
and nothing is deleted.

## User decisions (already made)

- Three characters to start: `nicole`, `anby`, `billy`.
- Ten emotions, hardcoded in source: neutral, amused, smug, excited, impressed, earnest,
  deadpan, annoyed, panicked, confused.
- Characters are discovered from directory names, never hardcoded. Emotions are hardcoded.
- Recordings live in-repo and are committed, in **plain git, not LFS** — matching how the
  existing twelve refs are stored.
- An emotion with no recordings falls back to `neutral` rather than erroring.
- Per-character settings come from an optional `character.yaml`; a character without one
  gets defaults.
- Expect 5-10 recordings per emotion per character.
- The test page uses the framed streaming endpoint and displays timing, not just audio.
- `/v1/audio/speech` gains an `emotion` field too.
- The page is exercised with Playwright MCP.
- Registry approach A: characters compile into the existing registry rather than living
  beside it as a parallel system.
- The Playwright and GPU end-to-end checks are **gated on real recordings arriving**;
  no placeholder or synthetic character library is committed to satisfy them.

## 1. On-disk layout

Root: `faster_qwen3_tts/server_voices/characters/`. The tree for the three characters
exists and is tracked (thirty emotion directories, each holding a `.gitkeep`), with a
README recording the convention.

```
characters/
  nicole/
    character.yaml          # optional
    neutral/
      calm_intro.wav
      calm_intro.txt
      take2.wav
      take2.txt
    amused/
      ...
  anby/                     # no character.yaml -- defaults apply
  billy/
```

- A **character** is any subdirectory of the root whose name does not start with `.` or
  `_`. Its id is the directory name.
- An **emotion directory** is a subdirectory whose name is one of the ten. Directories
  with any other name are skipped with a warning — a misspelling such as `anoyed` must be
  visible in the startup log, not silently invisible.
- A **reference** is a `.wav` with a sibling `.txt` of the same stem. Its id is the stem.
- Files are read in sorted order so ids and test expectations are deterministic.
- Transcripts are read as UTF-8 and stripped.

Skipped with a warning, never fatal: a `.wav` with no matching `.txt`, a `.txt` with no
matching `.wav`, an empty transcript, an unreadable or zero-length audio file, and any
other file — including the `.gitkeep` placeholders, which the scanner must ignore silently
since they are structural.

**Duration is checked but only warned about.** A reference outside 2-30 seconds is logged
and still used. It is worth stating why this is not an error: the reference contributes
roughly 12 tokens per second to the prefill, so a 15 s clip is ~180 tokens against
`max_seq_len=2048`. Prefill overflow (`talker_graph.py`) is driven by input text length,
which `MAX_INPUT_CHARS` already bounds — not by clip length. The duration check is a
recording-quality guard, and claiming otherwise would be inventing a risk.

### Version control

`.gitignore` carries a global `*.wav`, and the existing refs are tracked only because of
an explicit negation. The same negation now exists for `characters/**/*.wav` — without it
a dropped-in library is silently untracked, `git status` shows nothing, and the recordings
look committed when they are not. This is already fixed in the tree; it is recorded here
because it is invisible and easy to undo.

### character.yaml

Optional. Recognised keys and defaults:

| Key | Default | Meaning |
| --- | --- | --- |
| `language` | `English` | Passed to the model as the generation language |
| `temperature` | server default | Sampling temperature for this character |

An unrecognised key is a **startup error**. A silently ignored typo in a config file is
the same failure mode as a misspelled emotion directory, and it deserves the same
loudness — here it can be fatal because the file is explicit, deliberate content.

The default temperature has one source: the existing `DEFAULT_TEMPERATURE` used by the
YAML loader. `load_characters` takes it as an argument rather than redeclaring `0.7`.

## 2. Emotions, fallback and errors

The ten emotion names are a module-level constant in `voice_registry.py`:

```python
EMOTIONS = ("neutral", "amused", "smug", "excited", "impressed",
            "earnest", "deadpan", "annoyed", "panicked", "confused")
```

**An emotion with zero valid references produces no registry entry at all.** This is
load-bearing: if an empty `nicole:smug` existed, `resolve()` would return it, selection
would face an empty tuple, and the request would crash instead of falling back. Absence
is what makes the fallback reachable.

Three outcomes, deliberately distinct:

| Situation | Outcome |
| --- | --- |
| Emotion not in `EMOTIONS` | 400 — the client asked for something that does not exist |
| Emotion in `EMOTIONS`, no entry for it | Use the character's `neutral` entry; report the substitution |
| Character has references but none under `neutral` | Character skipped at startup with a warning; absent from discovery |

The third case is the invariant that makes the fallback safe: `neutral` is the one
emotion a character must have. A character that cannot satisfy the fallback would fail
unpredictably per-request depending on which emotion was asked for, so it is better
absent and obvious than present and unreliable. A character directory with **no**
references at all is skipped quietly — that is an unpopulated character, not a broken one.

If the merged registry ends up with zero voices, startup fails.

Because the fallback is silent to the audio, the response must state it. See §6.

## 3. Registry changes

`voice_registry.py` today pins one `ref_audio`/`ref_text` per `VoiceConfig`. That widens
to a list:

```python
@dataclass(frozen=True)
class Reference:
    id: str
    audio: Path
    text: str

@dataclass(frozen=True)
class VoiceConfig:
    id: str
    type: str
    language: str
    temperature: float
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    references: tuple[Reference, ...] = ()
    emotion: Optional[str] = None
```

`ref_audio` and `ref_text` are **removed, not aliased**. Two names for one thing invites
drift, and production code stops using them entirely once selection is per-reference.

### The regression pin, stated honestly

An earlier draft of this spec claimed the existing tests stay green unmodified. That is
false and the claim is withdrawn. Concretely:

| File | What changes | Why it is still a pin |
| --- | --- | --- |
| `tests/test_voice_registry.py:13,16` | reads `en_m.ref_audio` → `en_m.references[0].audio` | asserts the same thing (all twelve real voices load with absolute, existing paths) through the new shape |
| `tests/test_server_stream.py:19,191,193` | three `VoiceConfig(...)` constructions | these are test doubles, not behaviour assertions |
| `tests/test_server.py`, `tests/test_server_stream.py` fakes | `synthesize`/`synthesize_stream` signatures (§5) | doubles must match the real interface |

Everything else in those files — every behavioural assertion about the dubbing voices,
emotion resolution, streaming frames and cancellation — stays byte-identical. *That* is
the pin. Rewriting a field access is not a regression; changing what the tests assert
would be, and does not happen here.

### Fallback in `resolve()`

`Registry` gains `emotion_fallback_ids: frozenset[str]`, populated **only by the character
loader**. The algorithm:

1. If `emotion is None`: return `voices[voice_id]` if present, else look up the voice's
   default emotion (unchanged from today).
2. If `<id>:<emotion>` is present → return it. **Exact-key lookup comes first**, before
   any `EMOTIONS` check.
3. Else if `voice_id in emotion_fallback_ids` and `emotion in EMOTIONS` and the voice's
   default-emotion key exists → return that entry.
4. Else raise `KeyError` → 400.

Two things this ordering protects, both of which an earlier draft broke:

- **Emotive YAML voices keep their contract.** Their emotion names are persona-defined and
  are deliberately *not* constrained to the ten (see `voices_lyrebird.yaml`'s documented
  form and `tests/test_voice_registry.py`). Exact-key-first means a persona handle like
  `resigned` still resolves; validate-the-name-first would have broken it.
- **They do not silently start falling back.** A registry-wide step 3 would mean an
  emotive YAML voice asked for `smug` returns its default emotion instead of a 400. On a
  custom voice that is *wrong delivery with no error* — precisely the failure this design
  exists to avoid elsewhere. Gating on `emotion_fallback_ids` confines the new behaviour
  to characters.

### Merging sources

`merge(yaml_registry, character_registry) -> Registry` — two arguments, not variadic;
there are exactly two sources and there is no third in prospect.

- **Collision is detected on voice id, not dict key.** Keys are flattened, so a flat YAML
  voice `nicole` and a character's `nicole:neutral` do not collide as keys — yet
  `resolve("nicole")` would silently prefer the flat entry by existing precedence,
  shadowing the character. Compare the set of `cfg.id` values plus `defaults` keys.
  Any overlap is a startup error.
- **`sample_rate` must match.** `Registry.sample_rate` feeds `ModelManager.sample_rate`
  and every stream header frame. The character registry declares 24000 (the model's
  output rate); a mismatch with the YAML registry is a startup error rather than one
  silently winning.
- `defaults` maps merge; `emotion_fallback_ids` unions.

## 4. Character loader

New module `faster_qwen3_tts/characters.py`. One public function:

```python
def load_characters(root: Path, default_temperature: float,
                    sample_rate: int = 24000) -> Registry
```

It returns exactly what `load_registry` returns — `VoiceConfig` entries keyed
`nicole:amused`, `defaults["nicole"] = "neutral"`, and `emotion_fallback_ids` containing
every discovered character id — so everything downstream is source-agnostic. All warnings
described in §1 and §2 are emitted through the module logger during this call.

Keeping discovery in its own module, returning the established type, is what makes
approach A cheap: no consumer of the registry needs to know a character from a YAML voice.

## 5. Reference selection and the manager interface

```python
def pick_reference(cfg: VoiceConfig, rng: random.Random | None = None) -> Reference
```

Uniform random choice over `cfg.references`. The rng is injectable so tests are
deterministic; production passes nothing and uses the module default.

**Selection happens in the route, not the model manager.** The route needs the chosen
reference in order to report it, and the manager should not be the thing that decides what
a request means. The manager receives the reference and looks up its pre-baked prompt.

No attempt is made to avoid repeating the previous choice. With 5-10 references an
occasional repeat is unremarkable, and tracking per-character history would add state to a
stateless path for no measured benefit.

### Signatures (pinned — two test doubles must match them)

```python
def synthesize(self, cfg: VoiceConfig, reference: Optional[Reference],
               text: str, temperature: float, max_new_tokens=None): ...

def synthesize_stream(self, cfg: VoiceConfig, reference: Optional[Reference],
                      text: str, temperature: float, chunk_size: int,
                      max_new_tokens=None, cancel=None, instruct=None): ...
```

`reference` is `None` for custom voices, which have no reference recording.

`_clone_prompts` is keyed `(cfg.key, reference.id)`.

Call sites in `server.py` that must change: prompt baking at warmup (~line 100), the
warmup generation's `ref_text` (~111), `_synthesize_blocking`'s `ref_text` (~128), and the
streaming branch's prompt lookup and `ref_text` (~167-168). In every case the `ref_text`
passed to the model must come from the **chosen reference**, not from the config.

## 6. Endpoints

### `GET /v1/voices` (new)

Not gated on warmup — it reads the registry and needs no GPU, so the page can populate its
dropdowns during the ~50 s startup while `/health` still returns 503.

```json
[
  {"id": "nicole", "type": "clone", "language": "English",
   "default_emotion": "neutral", "emotion_fallback": true,
   "emotions": {"neutral": 7, "amused": 5, "smug": 0, ...}},
  {"id": "ono_anna", "type": "custom", "language": "English",
   "default_emotion": "neutral", "emotion_fallback": false,
   "emotions": {"neutral": 1, "amused": 1}},
  {"id": "en_m", "type": "clone", "language": "English",
   "default_emotion": null, "emotion_fallback": false, "emotions": null}
]
```

Three shapes, because there are three kinds of entry. A **character** enumerates all ten
`EMOTIONS` with counts, reporting `0` for those absent from the registry. An **emotive YAML
voice** reports only its own declared emotion names. A **flat voice** reports `null`.
`emotion_fallback` tells the page whether a `0` means "falls back to neutral" or "cannot
be requested" — without it the page would mislabel one as the other.

### `POST /v1/audio/stream`

No request change — it already accepts `voice` and `emotion`. The header frame gains four
fields:

```json
{"sample_rate": 24000, "channels": 1, "format": "s16le",
 "voice": "nicole", "emotion": "neutral", "requested_emotion": "smug",
 "reference": "calm_intro"}
```

`requested_emotion` is `null` when the caller omitted `emotion`; `reference` is `null` for
custom voices. Additive JSON keys, so existing clients ignore them.

### `POST /v1/audio/speech`

Gains `emotion: Optional[str]`, resolved identically. The body is raw WAV bytes with no
envelope, so the same reporting goes in response headers:

```
X-TTS-Voice: nicole
X-TTS-Emotion: neutral
X-TTS-Requested-Emotion: smug
X-TTS-Reference: calm_intro
```

Headers are omitted rather than sent empty where the value would be null. Omitting
`emotion` resolves through `default_emotion`, so a character works on this endpoint
without the caller knowing it has emotions at all.

### `GET /`

Serves the test page from `faster_qwen3_tts/server_static/index.html`, **mounted only when
`--characters` was passed**. "Mounted if the file exists" would be no guard at all, since
the file is committed and therefore always exists — media-worker's dubbing deployment
would end up serving an unauthenticated development page. Tying it to `--characters` keeps
it on the box that wants it, consistent with §8.

## 7. Warmup and startup

Warmup bakes one clone prompt per reference, cached as `(voice_key, reference_id)`.

Measured on this machine (RTX 4090, 1.7B Base, bf16): **~101 ms per prompt** steady-state
across the twelve existing refs, 55-139 ms depending on clip length. The first call is
~3.2 s including one-off lazy initialisation.

At 150-300 references that is **15-30 s** added to startup, on top of the ~20 s model load
and graph capture. Progress is logged rather than stalling silently. The cost grows
linearly with library size — 1000 references would be ~100 s.

Pre-baking everything is chosen over lazy baking because selection is random: with lazy
baking, early requests would each pay ~101 ms on top of a ~260 ms TTFA budget, and would
keep doing so until the cache happened to fill. Baked prompts are codec tokens plus a
speaker embedding, so holding all of them costs little memory.

**A bake failure is a startup error**, not a silent skip. Discovery (§1) already rejects
the things that plausibly go wrong with a hand-assembled library — missing pairs, empty
transcripts, unreadable audio. A reference that passes discovery and then fails to bake
indicates a real defect, and stopping is better than pruning the registry after it has
been published to the routes.

## 8. CLI and packaging

`serve-http` gains `--characters [PATH]` (`nargs="?"`, `const=<bundled>`, `default=None`):

- absent → no character library
- passed bare → the bundled `server_voices/characters`
- passed with a path → that directory

**The bundled `voices.yaml` loads only when it is implied or asked for**: when neither flag
is given (today's behaviour, unchanged, so media-worker is untouched), or when `--voices`
is given explicitly. `--characters` alone means characters only — the lyrebird box should
not carry twelve dubbing voices and bake twelve prompts it never serves. Both flags
together merge the two sources (§3).

`create_app` gains a `characters_path=None` keyword, preserving existing callers such as
`tests/test_server_gpu.py`.

### Packaging is a deliverable, not an assumption

None of the new files ship under the current configuration, which would make §12's claim
that the library "ships with the package" false:

- **`pyproject.toml`** — `package-data` globs do **not** recurse, so explicit depths are
  required: `server_voices/characters/*.md`, `server_voices/characters/*/*.yaml`,
  `server_voices/characters/*/*/*.wav`, `server_voices/characters/*/*/*.txt`, and
  `server_static/*.html`.
- **`MANIFEST.in`** — currently `recursive-include faster_qwen3_tts *.py` only, so the
  sdist carries no audio. Needs matching includes.
- **`.dockerignore`** — excludes `*.html`, which would silently drop the test page from an
  image while §6's existence check made its absence look deliberate. Needs a negation for
  `faster_qwen3_tts/server_static/`.

## 9. Test page

One self-contained `index.html` — no CDN, no build step, inline CSS and JS. A page that
makes no external requests is also the only kind that works offline on the box.

Controls: character dropdown, emotion dropdown (from `/v1/voices`, marking emotions whose
count is `0` on a fallback-enabled voice as falling back), text area, chunk size (default
4), generate, stop.

**During warmup** the page polls `/health`. Dropdowns populate immediately from
`/v1/voices`; Generate is disabled with a visible "warming" state until `/health` returns
200, rather than letting the user click into a bare 503.

**Playback** is WebAudio. The `AudioContext` is constructed on the first Generate click,
not at page load — browsers block an ungestured context and log a warning, which would
also trip the clean-console check in §10. Each audio frame's s16le payload converts to
Float32 and is scheduled against a running clock. Buffers are created at 24 kHz and the
browser resamples to the device rate. Nothing is written to disk; audio exists only as
scheduled buffers, and closing the tab is the cleanup. Stop cancels the fetch, which
aborts the decode server-side.

Scheduling clamps: `nextStart = max(nextStart, currentTime)`. Without the clamp a late
chunk is scheduled in the past, `AudioBufferSourceNode.start()` plays it immediately, and
the result is overlapping garbled audio — which would *hide* the underrun the timing panel
exists to reveal. When the clamp fires, the panel marks that chunk as an underrun.

**Timing panel** — the reason the page uses the streaming endpoint:

- TTFA: request sent → first audio frame
- Per chunk: index, `decode_ms`, `prefill_ms`, gap since the previous chunk, audio
  delivered so far, and **playback margin** (delivered audio minus elapsed time since
  playback began). Margin is the quantity that decides whether playback starves; showing
  it live is the browser-side equivalent of `spikes/streaming/chunk_margin.py`.
  `prefill_ms` is present only on the first chunk and must render as blank, not
  `undefined`, thereafter.
- Totals from the end frame.
- The resolved reference id, and a visible note when the emotion fell back.

**TTFA varies by reference**, because ICL prefill scales with the chosen clip's duration
and selection is random. The panel notes this next to the figure so run-to-run variation
is not misread as jitter; the reported reference id is what makes it diagnosable.

**Failure handling** follows the wire contract: an error frame is shown as a failure, and
a stream that ends *without* an end frame is reported as a failed generation, not a short
one. Pre-stream failures arrive as normal 400/503 and are shown inline.

## 10. Testing

**Unit — discovery** (`tests/test_characters.py`), fixture trees built in a tmpdir with
the `wave` module, no GPU:

- pairs discovered; ids are file stems; ordering deterministic
- `.wav` without `.txt`, `.txt` without `.wav`, empty transcript, zero-length audio →
  skipped, warned
- misspelled emotion directory → skipped, warned
- `.gitkeep` and other stray files ignored silently
- an emotion with no valid references produces **no** registry entry
- character with references but no `neutral` → skipped, warned, absent from the registry
- entirely empty character directory → skipped quietly
- `character.yaml` honoured; absent file → defaults; unknown key → startup error
- hidden and underscored directories ignored
- out-of-range duration warns but still yields a usable reference

**Unit — registry** (extends `tests/test_voice_registry.py`):

- YAML single clip becomes a one-element `references` tuple
- `resolve` returns the exact entry when present, including a persona-defined emotion name
  outside `EMOTIONS`
- `resolve` falls back for a character (in `emotion_fallback_ids`) asked for a known
  emotion with no entry
- `resolve` does **not** fall back for an emotive YAML voice — same request still raises
- `resolve` still raises for an emotion outside `EMOTIONS`
- merge succeeds; colliding voice ids raise; mismatched `sample_rate` raises

**Unit — selection:** seeded rng gives a deterministic choice; every returned reference
belongs to the requested config; a one-element list always yields that element.

**Server** (extends `tests/test_server_stream.py`, `tests/test_server.py`, fake manager):

- `/v1/voices` for all three entry shapes, and that it answers before warmup completes
- header frame carries `voice`, `emotion`, `requested_emotion`, `reference`, with nulls
  where specified
- `/v1/audio/speech` accepts `emotion` and sets the four `X-TTS-*` headers
- unknown emotion → 400 on both endpoints
- fallback observable via the header frame and the response header
- `GET /` is absent when `--characters` was not passed

**Gated on real recordings — not part of implementation completion.** These cannot run
against an empty library, and no placeholder character is committed to make them runnable:

- **Browser (Playwright MCP)** against a locally running server: page loads, dropdowns
  populate, an emotion with zero references is marked as falling back, generate produces a
  TTFA figure and per-chunk rows and an end-frame total, the resolved reference id is
  displayed, a rejected request surfaces an inline error, console is clean.
- **GPU** (extends `tests/test_server_gpu.py`): one end-to-end test that a real character
  streams and produces audio of plausible length.

What neither can prove is whether the audio *sounds* right. That stays a human check,
which is the page's actual purpose; the automation covers the plumbing around the ear.

## 11. Out of scope

- Curating or recording the reference audio itself.
- Avoiding repeated reference selection.
- Emotion on the custom-voice (`instruct`) path — unchanged, and §3 takes deliberate care
  not to disturb it.
- Changing media-worker's twelve dubbing voices or their registry file.
- Removing `voices_lyrebird.yaml`.
- Authentication on the test page; it is a local development tool, and §6 keeps it off the
  production deployment instead.
- `/v1/audio/speech` does not call `strip_stage_directions` while `/v1/audio/stream` does.
  Pre-existing asymmetry, untouched here, noted because the page will make it visible.

## 12. Risks

**Implementation completes without end-to-end verification.** By decision, the browser and
GPU checks wait for real recordings. Every unit and server-level behaviour is covered, but
the first real audition is also the first time the whole path runs. Expect to find
plumbing issues then rather than during the work.

**Shared type change reaches production voices.** `VoiceConfig` is what media-worker's
twelve voices flow through. Mitigated by the widening being degenerate (one clip = a
one-element tuple) and by the behavioural assertions in the existing tests being left
untouched — see §3 for what actually changes and why it is still a pin.

**Startup time grows with the library.** 15-30 s at the expected size, linear thereafter.
Acceptable for a resident service, and `--characters` defaults to off so only the
deployment that wants it pays. If the library grows past a few hundred references this
should be revisited — a disk cache of baked prompts is the obvious next step, deliberately
not built now.

**Repository size, permanently.** 150-300 recordings at the size of the existing refs
(~217 KB) is ~40-75 MB; at the 5-15 s the README asks for, it is more like 70-215 MB.
Plain git was chosen deliberately over LFS, so these blobs are in history forever and
every clone pays for them — and re-recording a character adds the new version rather than
replacing it. Accepted; recorded so the growth is not a surprise later.

**Random selection makes output non-reproducible.** Two identical requests can differ in
both timbre and TTFA. That is the intent, but it means a bad-sounding result needs the
reported reference id to diagnose — which is why the reference is reported on both
endpoints and shown on the page.
