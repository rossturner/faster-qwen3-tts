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
to be carried by the clip. One directory of clips per emotion is the direct expression of
that finding.

This supersedes the `ono_anna` custom-voice route for lyrebird, which took the opposite
approach (a fixed preset directed per-request by `instruct`) because no cloned character
voices existed yet. `voices_lyrebird.yaml` stays in the tree as the record of that route;
nothing is deleted as part of this work.

## User decisions (already made)

- Three characters to start: `nicole`, `anby`, `billy`.
- Ten emotions, hardcoded in source: neutral, amused, smug, excited, impressed, earnest,
  deadpan, annoyed, panicked, confused.
- Characters are discovered from directory names, never hardcoded. Emotions are hardcoded.
- Recordings live in-repo and are committed, following the existing `refs/` pattern.
- An emotion with no recordings falls back to `neutral` rather than erroring.
- Per-character settings come from an optional `character.yaml`; a character without one
  gets defaults.
- Expect 5-10 recordings per emotion per character.
- The test page uses the framed streaming endpoint and displays timing, not just audio.
- `/v1/audio/speech` gains an `emotion` field too.
- The page is exercised with Playwright MCP.
- Registry approach A: characters compile into the existing registry rather than living
  beside it as a parallel system.

## 1. On-disk layout

Root: `faster_qwen3_tts/server_voices/characters/` (created, with a README recording the
convention).

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
matching `.wav`, an empty transcript, an unreadable file, a non-`.wav`/`.txt` file
(including the `.gitkeep` placeholders).

### character.yaml

Optional. Recognised keys and defaults:

| Key | Default | Meaning |
| --- | --- | --- |
| `language` | `English` | Passed to the model as the generation language |
| `temperature` | server default (0.7) | Sampling temperature for this character |

An unrecognised key is a **startup error**. A silently ignored typo in a config file is
the same failure mode as a misspelled emotion directory, and it deserves the same
loudness — here it can be fatal because the file is explicit, deliberate content.

## 2. Emotions, fallback and errors

The ten emotion names are a module-level constant in `voice_registry.py`:

```python
EMOTIONS = ("neutral", "amused", "smug", "excited", "impressed",
            "earnest", "deadpan", "annoyed", "panicked", "confused")
```

Three outcomes, deliberately distinct:

| Situation | Outcome |
| --- | --- |
| Emotion not in `EMOTIONS` | 400 — the client asked for something that does not exist |
| Emotion in `EMOTIONS`, no references for it | Use a `neutral` reference; report the substitution |
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

The YAML loader wraps its single clip in a one-element tuple, so the twelve dubbing
voices and the emotive YAML form are unchanged in behaviour and only widened in type.
This is the one place the change reaches media-worker's production path; the existing
registry tests pin that behaviour and must stay green unmodified.

### Fallback in `resolve()`

`Registry.resolve(voice_id, emotion)` gains step 2:

1. `<id>:<emotion>` present → return it.
2. Else if `emotion in EMOTIONS` and the voice has a default emotion whose key is
   present → return that, and the caller can detect the substitution by comparing
   `cfg.emotion` to what it asked for.
3. Else raise `KeyError` → 400.

The existing rejection of an invented emotion (`furious`) survives, because `furious` is
not in `EMOTIONS`.

### Merging sources

A new `merge(*registries)` (or equivalent in the loader) combines the YAML registry and
the character registry into one `Registry`. A voice id present in both is a **startup
error** — silently letting one win would make deployments depend on load order.

## 4. Character loader

New module `faster_qwen3_tts/characters.py`. One public function:

```python
def load_characters(root: Path, default_temperature: float = 0.7) -> Registry
```

It returns exactly what `load_registry` returns — `VoiceConfig` entries keyed
`nicole:amused`, and `defaults["nicole"] = "neutral"` — so everything downstream is
source-agnostic. All warnings described in §1 and §2 are emitted through the module
logger during this call.

Keeping discovery in its own module, returning the established type, is what makes
approach A cheap: no consumer of the registry needs to know a character from a YAML voice.

## 5. Reference selection

```python
def pick_reference(cfg: VoiceConfig, rng: random.Random | None = None) -> Reference
```

Uniform random choice over `cfg.references`. The rng is injectable so tests are
deterministic; production passes nothing and uses the module default.

**Selection happens in the route, not the model manager.** The route needs the chosen
reference in order to report it, and the manager should not be the thing that decides
what a request means. The manager receives the reference as an argument and looks up its
pre-baked prompt.

No attempt is made to avoid repeating the previous choice. With 5-10 references an
occasional repeat is unremarkable, and tracking per-character history would add state to
a stateless path for no measured benefit.

## 6. Endpoints

### `GET /v1/voices` (new)

Fills the page's dropdowns and gives clients a way to discover ids without guessing.

```json
[
  {"id": "nicole", "type": "clone", "language": "English",
   "default_emotion": "neutral",
   "emotions": {"neutral": 7, "amused": 5, "smug": 0, ...}},
  {"id": "en_m", "type": "clone", "language": "English",
   "default_emotion": null, "emotions": null}
]
```

Counts let the page mark which emotions will fall back before the user generates.
Flat voices report `null` for both emotion fields.

### `POST /v1/audio/stream`

No request change — it already accepts `voice` and `emotion`. The header frame gains four
fields:

```json
{"sample_rate": 24000, "channels": 1, "format": "s16le",
 "voice": "nicole", "emotion": "neutral", "requested_emotion": "smug",
 "reference": "calm_intro"}
```

Additive JSON keys; existing clients ignore them.

### `POST /v1/audio/speech`

Gains `emotion: Optional[str]`, resolved identically. The body is raw WAV bytes with no
envelope, so the same reporting goes in response headers:

```
X-TTS-Voice: nicole
X-TTS-Emotion: neutral
X-TTS-Requested-Emotion: smug
X-TTS-Reference: calm_intro
```

Omitting `emotion` resolves through `default_emotion`, so a character works on this
endpoint without the caller knowing it has emotions at all.

### `GET /`

Serves the test page from `faster_qwen3_tts/server_static/index.html`, mounted only when
that file exists so a stripped deployment is unaffected.

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

## 8. CLI

`serve-http` gains `--characters [PATH]`:

- absent → no character library loaded
- passed bare → the bundled `server_voices/characters`
- passed with a path → that directory

`--voices` keeps its meaning. If **neither** flag is given, the bundled `voices.yaml`
loads exactly as today. This matters: media-worker's deployment must not silently start
paying 15-30 s of startup and loading a library it never serves.

Both flags together merge the two sources (§3).

## 9. Test page

One self-contained `index.html` — no CDN, no build step, inline CSS and JS. A strict
no-external-requests page is also the only kind that works offline on the box.

Controls: character dropdown, emotion dropdown (from `/v1/voices`, marking emotions with
zero references as falling back), text area, chunk size (default 4), generate, stop.

**Playback** is WebAudio. Each audio frame's s16le payload converts to Float32 and is
scheduled against a running clock, back to back. Buffers are created at 24 kHz and the
browser resamples to the device rate. Nothing is written to disk; audio exists only as
scheduled buffers and closing the tab is the cleanup. Stop cancels the fetch, which
aborts the decode server-side.

**Timing panel** — the reason the page uses the streaming endpoint:

- TTFA: request sent → first audio frame
- Per chunk: index, `decode_ms`, `prefill_ms`, gap since the previous chunk, audio
  delivered so far, and **playback margin** (delivered audio minus elapsed time since
  playback began). Margin is the quantity that decides whether playback starves; showing
  it live is the browser-side equivalent of `spikes/streaming/chunk_margin.py`.
- Totals from the end frame.
- The resolved reference id, and a visible note when the emotion fell back.

**Failure handling** follows the wire contract: an error frame is shown as a failure, and
a stream that ends *without* an end frame is reported as a failed generation, not a short
one. Pre-stream failures arrive as normal 400/503 and are shown inline.

## 10. Testing

**Unit — discovery** (`tests/test_characters.py`), fixture trees built in a tmpdir with
the `wave` module, no GPU, no committed fixtures:

- pairs discovered; ids are file stems; ordering deterministic
- `.wav` without `.txt`, `.txt` without `.wav`, empty transcript → skipped, warned
- misspelled emotion directory → skipped, warned
- `.gitkeep` and other stray files ignored
- character with references but no `neutral` → skipped, warned, absent from the registry
- entirely empty character directory → skipped quietly
- `character.yaml` honoured; absent file → defaults; unknown key → startup error
- hidden/underscored directories ignored

**Unit — registry** (extends `tests/test_voice_registry.py`):

- YAML single clip becomes a one-element `references` tuple
- `resolve` returns the exact entry when present
- `resolve` falls back to the default emotion for a known emotion with no entry
- `resolve` still raises for an emotion outside `EMOTIONS`
- merge succeeds; colliding ids raise
- **existing tests stay green unmodified** — the check that widening did not disturb the
  dubbing path

**Unit — selection:** seeded rng gives a deterministic choice; every returned reference
belongs to the requested config; a one-element list always yields that element.

**Server** (extends `tests/test_server_stream.py`, `tests/test_server.py`, fake manager):

- `/v1/voices` shape for both character and flat voices
- header frame carries `voice`, `emotion`, `requested_emotion`, `reference`
- `/v1/audio/speech` accepts `emotion` and sets the four `X-TTS-*` headers
- unknown emotion → 400 on both endpoints
- fallback observable via the header frame and the response header

**Browser (Playwright MCP)**, against a locally running server with the real library:

- page loads, dropdowns populate from `/v1/voices`
- an emotion with zero references is marked as falling back
- generate produces a TTFA figure, per-chunk rows, and an end-frame total
- the resolved reference id is displayed; fallback is visibly noted
- a rejected request surfaces an inline error rather than silence
- browser console is clean

What this cannot prove is whether the audio *sounds* right — that stays a human check,
which is the page's actual purpose. Playwright covers the plumbing around the ear.

**GPU** (extends `tests/test_server_gpu.py`, existing marker): one end-to-end test that a
real character streams and produces audio of plausible length.

## 11. Out of scope

- Curating or recording the reference audio itself.
- Avoiding repeated reference selection.
- Emotion on the custom-voice (`instruct`) path — unchanged.
- Changing media-worker's twelve dubbing voices or their registry file.
- Removing `voices_lyrebird.yaml`.
- Authentication on the test page; it is a local development tool.

## 12. Risks

**Shared type change reaches production voices.** `VoiceConfig` is what media-worker's
twelve voices flow through. Mitigated by the widening being degenerate (one clip = a
one-element list) and by leaving the existing tests unmodified as the pin.

**Startup time grows with the library.** 15-30 s at the expected size, linear thereafter.
Acceptable for a resident service, and `--characters` defaults to off so only the
deployment that wants it pays. If the library grows past a few hundred references this
should be revisited — a disk cache of baked prompts is the obvious next step, deliberately
not built now.

**Repository size.** 150-300 committed recordings at the size of the existing refs is
roughly 40-75 MB. Accepted deliberately; it buys versioned, reproducible voices that ship
with the package.

**Random selection makes output non-reproducible.** Two identical requests can differ.
That is the intent, but it means a bad-sounding result needs the reported reference id to
diagnose — which is precisely why the reference is reported on both endpoints.
