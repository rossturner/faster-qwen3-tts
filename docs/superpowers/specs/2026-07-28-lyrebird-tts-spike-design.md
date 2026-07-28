# Lyrebird TTS spike — design

Transient design artifact. Scopes a three-experiment spike answering whether
`faster-qwen3-tts` can serve as the TTS engine behind lyrebird's `Voice` port, and
what the WSL→Windows→OBS audio path costs. **This spike builds no product code.**

Authority on what lyrebird is remains `~/workspace/lyrebird/docs/00`–`08`.

## Why

lyrebird (`~/workspace/lyrebird`) is an autonomous AI VTuber: a Java 21 Spring Boot
process in WSL driving sidecars. Its TTS engine is explicitly undecided — `00-goals.md`
records "Qwen3-TTS vs Voxtral under parallel investigation". This spike is the
Qwen3-TTS half of that investigation.

Three constraints from lyrebird's docs shape everything:

1. **First audio within ~1 s of a beat's start** (`00-goals.md` goal 3). That budget
   covers the fast-profile LLM's wording call *and* synthesis *and* playback start —
   TTS gets a fraction of it, not the whole thing.
2. **Playback happens on Windows**, because that is where OBS captures
   (`01-architecture.md`: "Playback runs in a Windows-side audio player sidecar so
   sound is produced where OBS captures it").
3. **The engine is behind a contract.** Cognition never names a vendor. A persona
   declares a voice *handle*; adapter config maps it to an engine voice id and
   parameters (`08-persona.md`). Whatever emotion mechanism wins must express itself
   as adapter config, not as something cognition knows about.

## Scope

**In:** three experiments, below. Each produces throwaway code under `spikes/` and a
findings section appended to this repo's `docs/lyrebird-tts-spike-findings.md`.

**Out, by explicit decision:**

- **Mid-utterance cancellation.** Cancellation happens at sentence boundaries only, so
  the aired portion is known at sentence granularity and no playback-position feedback
  is needed. This relaxes `05-performer.md`, which currently says a preempting focus
  "cancels the in-flight sentence stream" and that "a beat cancelled mid-sentence is
  never recorded as fully said". **That doc needs amending — outside this spike.**
- **GPU co-tenancy.** No measurement of a local fast LLM sharing the 4090.
- **VB-CABLE.** The right long-term answer for isolating the VTuber's voice from system
  sound, deferred. Experiments target the Windows default device.
- **Any change to the shipped server.** `faster_qwen3_tts/server.py` and its registry are
  load-bearing for media-worker's dubbing pipeline, which now routes *all* languages
  through them. The spike runs separate processes on separate ports and touches neither.

## Experiment 1 — WSL → Windows → OBS

**Question:** can audio generated in WSL reach a Windows device OBS captures, with
bounded latency and no dropouts, and is the zero-code path good enough to defer
building a player sidecar?

Two paths, both confirmed feasible with no new installs:

| Path | Mechanism | Why it might win |
|---|---|---|
| **A — WSLg PulseAudio** | `/mnt/wslg/PulseServer` is live; `ffmpeg`/`ffplay` have `pulse` output. Audio surfaces on the Windows default device. | Zero new code. If latency is acceptable, the player sidecar can be deferred entirely. |
| **B — Windows player sidecar** | Windows-side Python 3.13.5 process accepting raw PCM over TCP from WSL, playing to an explicitly chosen device. | Matches `01-architecture.md`. Selects its device (VB-CABLE later). Can report playback completion, which is how the aired portion becomes knowable. |

**Measured:** wall-clock from first byte handed to the player to the player reporting
playback started; dropouts across a continuous 60 s chunked stream; whether OBS
registers the audio.

Dropouts are detected differently per path, because path A gives no introspection:
for **path B**, the player counts buffer starvations directly; for **path A**, by
comparing elapsed wall time against the audio duration submitted — a stream that takes
materially longer than its own duration to play out has stalled. Both paths also feed a
continuous tone, where a dropout is trivially audible on the one required human check.

**Honest limit:** true acoustic mouth-to-ear latency is not measurable without a
loopback capture rig. This experiment measures *pipeline* timestamps and requires one
human confirmation that OBS's meter moves. Any number reported is pipeline latency and
will be labelled as such — it is a lower bound on what a listener experiences.

**Unblocks:** whether the Voice adapter needs a player sidecar in v1 or can start with
WSLg, and what fixed latency to subtract from the ~1 s budget.

## Experiment 2 — is a streaming endpoint necessary?

**Question:** does the existing non-streaming contract fit lyrebird's budget, or must a
streaming endpoint be built?

The arithmetic that motivates it: non-streaming RTF is ~3.76, so synthesis takes
`duration ÷ 3.76`. A 4-second sentence costs ~1.06 s before the first byte exists —
already over budget, before the LLM's share, and worse for longer sentences. Streaming
decouples first-audio from sentence length (README reports ~159 ms TTFA, ICL,
chunk_size=8).

**Method:** load Base with an existing pinned clone prompt (`en_f` — no new voice design
needed for this experiment) and, across beat-realistic texts of 1/2/3 sentences:

- non-streaming: total wall time to first byte, per text length;
- streaming at `chunk_size` 2/4/8: TTFA, per-chunk inter-arrival times, and cumulative
  audio produced vs wall time.

That last metric is the one that decides playback quality: if generation ever falls
behind realtime mid-sentence, the player underruns and the voice stutters. A margin
that shrinks as the sentence goes on is a failure even when TTFA looks good.

If streaming is needed, a minimal chunked-PCM HTTP wrapper is built to measure the
transport overhead on top of the library numbers. If not, that work is skipped.

**Unblocks:** whether to build a streaming endpoint at all, the chunk size to use, and
the TTS share of the ~1 s budget.

## Experiment 3 — emotion control

**Question:** how does a *specific, pinned* voice express emotion?

### What Qwen actually offers

| Mechanism | Emotion control | Preserves a custom pinned identity? |
|---|---|---|
| CustomVoice + `instruct` | Officially supported by Alibaba | **No** — only the 9 preset speakers |
| VoiceDesign + `instruct` | Persona-level, offline | **No** — identity re-rolls each generation (unseeded) |
| Base clone + `instruct` | **Unproven** | Yes — pinned reference |
| Emotional reference clip | Certain — ICL copies the reference's prosody | Yes, if derived from the base voice |

The state of the `instruct`-on-clone evidence, since it is the crux: the fork added it
deliberately (PR #40) with tests, but `tests/test_e2e_parity.py:1025` only asserts the
instruct tokens are *prepended to the talker input* — sequence length grows by exactly
the instruct length. It is mechanically wired and perceptually unproven. The warning at
`model.py:439` about unreliable instruction-following fires only for `xvec_only=True`;
the server uses ICL mode, which the README calls "much more predictable". No record
exists in this repo of instruct having been disproven on the ICL clone path.

The last row is the mechanism that cannot really fail, because it relies on no
instruction-following at all — ICL clones the reference's prosody, so an excited
reference yields excited speech. It is what the old system's `EmotionCache` exploited
(`lyrebird-old`: emotion name → list of reference clips, chosen at random so repeats
don't sound identical — an idea worth keeping regardless of mechanism).

### Stage A — is `instruct` inert on our path?

A cheap, objective gate before any voice-design work. Fixed text, fixed pinned clone
prompt, fixed temperature. **Four conditions** spanning contradictory prosody — no
instruct, "slow and sad", "fast and excited", "whispered and quiet" — and because
generation is unseeded, **5 runs per condition** so the instruct effect can be separated
from sampling noise.

Metrics: audio duration, speaking rate, F0 mean/spread, RMS energy.

**Positive control, mandatory:** the same instructs run through the CustomVoice model,
where Alibaba documents instruct as supported. Without it, a null result is ambiguous —
it could mean instruct is inert, or merely that the metrics are too insensitive to
detect it. The control tells those apart.

**Decision rule, fixed in advance** (stated numerically so the result cannot be argued
after the fact): for each metric, compare the spread of condition means against the
mean within-condition spread across the 5 runs. Instruct is judged **effective** on a
metric when the between-condition spread exceeds **2×** the within-condition spread —
i.e. the instructs move the output further than resampling the same instruct does.

- Clone path effective on ≥1 metric → instruct works; Stage B can use it.
- Clone path inert **and control effective** → instruct is inert on the clone path;
  Stage B uses reference clips.
- Control also inert → the measurement is at fault, not the engine; reported as
  inconclusive rather than as a finding about Qwen.

### Stage B — build the emotion set

Requires a persona base voice, which does not exist yet: one is designed with the same
VoiceDesign → pin → clone method as the 12 dubbing voices (`docs/voice-design.md`).

Then the emotion set is built by whichever mechanism survived Stage A. If it is
reference clips, the preferred construction is **derived** refs — clone the pinned base
voice into each emotion, then pin those outputs — so emotional variants inherit the base
identity instead of being separate people, which independently-designed clips would be.

**Identity drift is measured, not just heard:** cosine similarity of the model's own
speaker embedding between each emotion clip and the base reference. Deliverable is an
audition page in the style of the existing voice work, plus the drift numbers.

**Unblocks:** the emotion contract between lyrebird's Voice adapter and the engine —
which, per `08-persona.md`, must be a named handle (`happy`, `sad`) resolved by adapter
config, exactly as the old system's `{text, emotion}` tool call did.

## Deliverables

- `spikes/` — throwaway scripts, one subdirectory per experiment.
- `docs/lyrebird-tts-spike-findings.md` — findings per experiment, each stating what was
  measured, what was assumed, and what remains unknown.
- A recommendation on whether Qwen3-TTS should be lyrebird's engine, with the numbers
  behind it.

## Risks

- **The ~1 s budget may not survive** once real LLM time is added, even with streaming.
  The spike measures the TTS share honestly rather than assuming the remainder fits.
- **Stage A may be inconclusive** if the positive control also shows no effect — that
  would indicate a measurement problem, not an engine finding, and is reported as such.
- **WSLg latency may be unbounded or drifty** under sustained streaming; that is the
  specific failure Experiment 1 is looking for, and it would force path B into v1.
