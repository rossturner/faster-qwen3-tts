# Lyrebird TTS spike — findings

Results for the experiments specified in
`docs/superpowers/specs/2026-07-28-lyrebird-tts-spike-design.md`. Measured on the
RTX 4090 / WSL2 box, 1.7B models, bf16, sdpa.

---

## Experiment 2 — is a streaming endpoint necessary? **Yes.**

`spikes/streaming/ttfa_probe.py`, voice `en_f` (existing pinned clone prompt), 3 runs per
cell, warmed up before measuring.

### Time to first audio

| Text | Non-streaming | Streaming cs=2 | cs=4 | cs=8 |
|---|---|---|---|---|
| 1 sentence (63 ch) | **1.41–1.64 s** | 0.203–0.334 s | 0.253–0.264 s | 0.336–0.340 s |
| 2 sentences (126 ch) | **2.23–2.29 s** | 0.197–0.202 s | 0.242–0.249 s | 0.334–0.345 s |
| 3 sentences (182 ch) | **3.34–3.49 s** | 0.199–0.202 s | 0.245–0.248 s | 0.335–0.341 s |

**The existing non-streaming endpoint cannot meet lyrebird's budget.** A single sentence
costs 1.41 s before the first byte exists — already over the ~1 s first-audio budget
with nothing left for the LLM, and it degrades linearly with sentence length (2.2 s,
3.3 s). This is the endpoint media-worker uses, and it is fine there, where whole-file
latency is what matters. It is the wrong shape for a live loop.

Streaming TTFA is **flat regardless of text length** — ~200 ms at chunk_size 2, ~250 ms
at 4, ~335 ms at 8 — because first audio depends on the chunk, not the utterance.

### Does generation stay ahead of realtime?

Yes, on every configuration — the realtime margin never went negative, and its minimum
was always the first chunk's own duration, meaning generation only ever pulled *further*
ahead as the sentence proceeded.

But the safety margin against a single hiccup differs sharply:

| chunk_size | Chunk duration | Worst observed gap | Headroom |
|---|---|---|---|
| 2 | 167 ms | 142 ms | **25 ms** |
| 4 | 333 ms | 302 ms | **31 ms** |
| 8 | 667 ms | 260 ms | **407 ms** |

At cs=2 and cs=4 a single slow chunk very nearly starves playback. **cs=8 is the
defensible choice**: 335 ms TTFA and 407 ms of headroom, versus cs=4 saving 85 ms of
TTFA while running within 31 ms of a stutter.

Streaming RTF (2.0–2.75) is below non-streaming (3.0–3.37), consistent with the
documented behaviour that the codec decoder dominates streaming.

### Budget implication

At cs=8, TTS consumes ~335 ms of the ~1 s beat budget, leaving ~665 ms for the fast
LLM's first sentence plus transport plus playback start. Playback start is not yet
known — see Experiment 1. This fits only if the fast-profile LLM is local and quick;
it does not fit a cloud model with typical round-trip latency.

---

## Experiment 1 — WSL → Windows → OBS. **Path A is dead on this machine.**

`spikes/audio_path/pulse_probe.py`, `spikes/audio_path/buffer_sweep.py`.

**WSLg PulseAudio does not deliver audio to the Windows host here.** Confirmed by the
operator: nothing was audible through the speakers during any run.

The failure is silent and then progressive, which is worth recording because it is
exactly the shape that would waste a day:

1. **First runs appeared to succeed.** `ffmpeg -f pulse` returned 0 and timings looked
   plausible — a 5 s tone "played" in 4.0 s. Writing into the sink succeeds whether or
   not anything reaches Windows.
2. `/mnt/wslg/pulseaudio.log` shows the truth: `[rdp-sink] data_send: send failed sent
   -1 bytes 7000`, then an endless connect/fail/reconnect loop. The RDP sink's link to
   the Windows host is broken.
3. **Later, every write blocked indefinitely** — including the exact configuration that
   had "worked" minutes earlier — and a killed client left a wedged `ffmpeg` holding the
   sink.

Two lessons regardless of which path is chosen:

- **A zero-exit-code from the player proves nothing.** Any health check for the audio
  path must be based on something observed downstream, not on the writer's return code.
- **The apparent 1 s of "in-flight audio"** measured before the operator check was an
  artifact of writing into a sink that discards, not a real device buffer. It should not
  be carried forward as a latency figure.

Whether this is a repairable WSLg fault (a `wsl --shutdown` may restore it) or a
standing condition is not yet established. It does not change the recommendation:
path B — the Windows-side player sidecar — is what `01-architecture.md` specifies, does
not depend on WSLg, and is the only path that can report playback completion. **Path A
should not be adopted even if a restart revives it**, on the evidence above that it
fails silently.

### Path B — the Windows player sidecar works

`spikes/audio_path/win_player.py` (Windows) + `spikes/audio_path/feed_player.py` (WSL).
The player **connects out to a listener in WSL** rather than listening itself: outbound
connections need no Windows Firewall rule, inbound ones prompt.

Three constraints emerged from building it, all of which belong in the real adapter:

**1. Use WASAPI, not the default host API.** PortAudio's default on Windows is MME.

| Host API | Reported output latency | Send → device consumed |
|---|---|---|
| MME (default) | 100 ms | 50.6 ms |
| **WASAPI** | **40 ms** | **21–22 ms** |
| DirectSound | 240 ms (device-reported) | not measured |
| WDM-KS | 40 ms (device-reported) | not measured |

WASAPI is **~62 ms to audible** versus MME's ~151 ms.

**2. The player must resample.** WASAPI shared mode accepts only the device mix rate and
rejects the TTS's 24 kHz with `Invalid sample rate [PaErrorCode -9997]`. This is not an
optimisation — a player that assumes 24 kHz passes through simply fails to open the
stream. The spike resamples 24 kHz → 48 kHz with interpolation that carries state across
chunk boundaries; a naive per-chunk resampler clicks at every seam.

**3. A jitter buffer is required, and the obvious starvation metric does not work.**
The first 60 s soak reported **0 underruns while inserting ~4 s of silence**: the feeder
sends at realtime and the device consumes at realtime, so with no slack the queue runs
dry constantly. PortAudio's `output_underflow` flag never fired, because the callback
*did* return on time — just with zeros.

> Recorded because it would mislead anyone instrumenting this later: **starvation must
> be counted as zero-fill inside the callback.** Trusting the host API's underflow flag
> reports a perfectly healthy stream that is audibly stuttering. The tell was wall time
> (64.12 s) exceeding the audio duration (60 s), not any counter.

With starvation counted properly, a small prefill fixes it outright (20 s runs, WASAPI,
160 ms feed chunks):

| Prefill | Starvation | Start latency | To audible |
|---|---|---|---|
| 0 ms | 40 ms in 2 gaps | 10 ms | ~50 ms |
| **100 ms** | **none** | **21 ms** | **~61 ms** |
| 250 ms | none | 177 ms | ~217 ms |

**100 ms of prefill is effectively free** because it is smaller than the 160 ms feed
chunk — the first chunk alone primes it, so nothing is delayed. 250 ms needs a second
chunk and costs ~156 ms. The prefill should be set below the chunk size for this reason.

**Path B budget: ~61 ms to audible, zero starvation.**

### Still outstanding

- **OBS capture is unconfirmed.** Requires the operator to watch the meter; not yet done.
- Whether the WSLg fault is repairable is unestablished, and deliberately not pursued —
  path A fails *silently*, so it should not be adopted even if a restart revives it.

---

## Experiment 3, Stage A — `instruct` controls pace only, not emotion

`spikes/emotion/instruct_inertness.py` (generation) + `spikes/emotion/reanalyse.py`
(analysis). 4 instruct conditions × 5 runs, voice `en_f` ICL clone, plus the same
conditions on CustomVoice/`aiden` as a positive control.

### The pre-registered rule was wrong, and the control is what caught it

The spec's rule compared the spread of condition *means* against the spread of
*individual runs*. But a mean of 5 runs varies by `std/√5`, so that rule demanded an
effect ~2.2× larger than it should have — and it duly declared the **positive control**
inert, when the control's raw numbers are unmistakable (`whispered` RMS 0.073 vs 0.11,
`fast_excited` F0 170 Hz vs 133 Hz).

The statistic was replaced with a one-way ANOVA (F, α = 0.05, df = 3,16 → F_crit 3.24).
**The threshold was not tuned to produce a result** — it is the conventional one, and
the control was re-checked under it first. This is still a post-hoc change to a
pre-registered rule and is flagged as such; the protection retained is that a
measurement which cannot detect the control is treated as invalid, not as a finding.

### Result

| Metric | Control F | Clone F | |
|---|---|---|---|
| duration_s | 7.70 ✓ | **7.26 ✓** | pace responds |
| chars_per_s | 8.27 ✓ | **7.96 ✓** | pace responds |
| rms | 4.59 ✓ | 0.53 ✗ | loudness does not |
| f0_median | 5.50 ✓ | 1.47 ✗ | pitch does not |
| voiced_frac | 20.27 ✓ | 1.22 ✗ | phonation does not |

The control responds on **every** metric, in semantically correct directions —
`whispered` quieter and slower, `fast_excited` faster and higher-pitched. So the
measurement works.

On the clone path, **only pace responds**. `fast_excited` shortens the utterance from
5.20 s to 4.78 s, but pitch moves 207 → 216 Hz (inside the noise) and loudness does not
move at all. `whispered` is indistinguishable from no instruct on every metric except a
statistically insignificant pace change.

### Interpretation

This is consistent with the mechanism: in ICL mode the reference audio pins timbre,
pitch and energy, and `instruct` can only modulate what the reference does not fix —
rate. It also broadly vindicates the recollection that prompted this experiment:
`instruct` is *not* inert on the clone path, but it is close enough to inert for
**emotion** that it cannot carry emotion on its own.

**Consequence for Stage B: emotion must come from reference clips**, per the old
`EmotionCache` model. `instruct` remains usable as a secondary pace control.

**Caveat:** this is an objective-metrics result. Pitch and energy are the measurable
carriers of emotion, but not the only ones — the clips in `spikes/emotion/out/` should
be listened to before Stage B commits, in case instruct is shifting something the
metrics do not capture.

## Experiment 3, Stage B — the persona voice is `ono_anna`, directed by free-text `instruct`

Stage B was specified as: design a persona voice, build an emotion set from derived
reference clips, measure identity drift. It did not end there. **Every route to a
*designed* voice that can also act was tried and failed**, and the voice that survived is
a built-in one — accepting a shared timbre in exchange for delivery that works.

Scripts, in the order they were run: `spikes/emotion/female_voice_audition.py`,
`female_voice_range.py`, `voicedesign_stability.py`, `voicedesign_directions.py`,
`xvec_splice.py`, `text_markup.py`, `text_tags_neutral.py`. Every verdict below marked
*by ear* is the operator's; objective metrics use the same estimator as Stage A.

### The built-in voices are nine, and there is no tenth

The nine CustomVoice speakers are token ids in the talker's 3072-row embedding table,
scattered between 2861 and 3066 — a spread that looks like nine names exposed out of a
larger trained block. They are not. Comparing each row against the same row in the Base
checkpoint:

| rows | vector length | changed vs Base |
|---|---|---|
| the 9 named speakers | 13.9 – 15.6 | 13.9 – 15.6 |
| the 202 unnamed ids in the block | 0.023 | **0.000** |

The nine started at zero in Base and were trained into full-sized vectors. Every other id
is byte-identical to Base and still at initialisation scale — never trained. The 0.6B
CustomVoice ships the identical nine-entry map. **The roster is fixed and complete.**

This matters because **there is no English female preset.** Both English voices (Ryan,
Aiden) are male; all four female voices are Chinese (Vivian, Serena), Japanese
(Ono_Anna) or Korean (Sohee). An English-speaking female persona has to come from a
non-native preset reading English, or be designed.

Licensing is clean — the repo LICENSE is plain Apache 2.0, no acceptable-use addendum,
no likeness clause. Provenance is not published: no voice actor credits, no dataset
documentation. These nine are also *not* the hosted Qwen3-TTS-Flash roster (17 voices,
Cherry/Ethan/Jennifer/…); only Ryan, Dylan and Eric appear in both.

### Instruct works on the built-in voices, and barely at all on clones

This is Stage A's result seen from the other side. Same instructions, same estimator:

| | clone (`en_f`, ICL) | built-in (`aiden`) |
|---|---|---|
| pitch spread across conditions | 9 Hz — inside its own noise | **37 Hz** |
| loudness, plain → whispered | 0.108 → 0.111 (nothing) | 0.110 → **0.073** |

In ICL mode the reference recording pins timbre, pitch and energy, and `instruct` can
only modulate what the recording leaves free — rate. A built-in speaker has no recording
competing with the instruction, so the instruction lands. **That asymmetry is the whole
reason this experiment ended where it did.**

### Ono_Anna over the other three

Two auditions of the four female presets, all forced to `language: English`:

- **Short sentences** (`female_voice_audition.py`, 3 sentences × 3 emotions × 4 voices,
  36 clips). Every voice moved, and moved hard — Vivian widest at 84 Hz of pitch swing.
- **Long passages** (`female_voice_range.py`, 9 emotions × 3 passages of 30–40 words ×
  4 voices, 108 clips). Emotions weighted toward the everyday registers a persona
  actually inhabits, with `neutral` passing no instruct at all as the honest baseline.

Ono_Anna was chosen by ear. The measured ranges over the long passages support it —
41.3 Hz pitch spread against Vivian's 35.7, Sohee's 26.2 and Serena's 25.5 — but the
choice was a listening call, not a metric.

**One unproven observation, recorded because it points somewhere useful.** Comparing the
first 3 s of each long passage against the remainder, the pitch spread across emotions
narrows: Ono_Anna 56.8 → 46.8 Hz (82% retained), Vivian 64.1 → 35.0, Serena 38.0 → 24.6,
Sohee 55.3 → 26.8. That is consistent with the instruction gripping at the start and the
voice drifting back to its habitual register — which would argue for short per-request
text, something lyrebird already does for latency. **But this is one sample per cell with
no repeats, against Stage A's ~9 Hz of run-to-run noise on pitch, and nobody has ever
listened for it.** It is indicative, not established, and Ono_Anna is the voice it
applies to least.

Two other observations from the same run, both untested:

- `neutral` — no instruct at all — gave the *highest*-pitched opening for Vivian
  (274 Hz) and Serena (271 Hz), above their own `excited`. These voices default to
  animated, and instructions may work more by calming than by exciting.
- Sohee's `amused` sat below her `sad` on pitch, which is the wrong way round and was
  never chased down.

### VoiceDesign cannot hold an identity across calls — **this is the load-bearing finding**

VoiceDesign was the one model that could have given both a unique voice and open-ended
free-text direction. It cannot, because it does not stay the same person.

Eight takes of one line from a **byte-identical** persona description
(`voicedesign_stability.py`) were judged by ear as too different to be one speaker. A
second run with the delivery language stripped out of the persona
(`voicedesign_directions.py`, 4 emotions × 3 takes) was judged the same way, on the
`neutral` takes where the instruct is identical across all three.

**An objective attempt to measure this failed and should not be repeated in that form.**
Pairwise speaker-embedding cosine put two obviously different speakers at 0.967 and the
same speaker token at 0.990 — a usable range 0.023 wide, on which VoiceDesign's 0.986
means nothing. The metric cannot resolve what the ear resolves easily. **Identity is a
perceptual judgement here; measure it by listening.**

Whether VoiceDesign obeys an emotional direction was left unresolved. Four plain
emotions (neutral/excited/sad/angry) produced a 1.1 s pace spread once the persona's own
delivery language was removed — but sad came out *higher*-pitched than excited in both
runs, and the clips were not conclusive by ear.

**This does not touch the design-once-and-pin route.** Drift exists only because clips
are re-rolled; one pinned clip cloned forever has none, which is why the 12 dubbing
voices are stable. What died is VoiceDesign as a *source of emotion*, live or as a
multi-clip factory — every clip in such a set would be an independent draw with exactly
this problem.

### The x-vector splice does not work

A clone prompt carries `ref_spk_embedding` (identity) and `ref_code` (the reference
recording) as separate fields, so a prompt can be built from an emotional Ono_Anna clip
with a designed voice's identity vector swapped in — prosody from one source, identity
from another. Six emotions, two takes each (`xvec_splice.py`).

The expected failure was that the recording would win, per Stage A. It did not: in five
of six emotions the spliced output landed *above both* controls on pitch (sad: persona
289 Hz, splice 331 Hz, Ono_Anna 224 Hz). Judged by ear as worse than either source —
consistent with two conditioning signals disagreeing rather than one winning. **Dead.**

### What the text itself can carry

No markup support exists. The tokenizer's 33 added tokens are all plumbing
(`<|audio_start|>`, `<tts_pad>`, …), and text goes verbatim into the chat template with
no preprocessing. The 86 inline tags (`[laughing]`, `[gasp]`) belong to
**Qwen-Audio-3.0-TTS**, a hosted API-only model, not this one. Open feature requests
asking for tags exist on GitHub and HF with no maintainer reply.

Twelve devices tested against matched controls — same instruct, same sentence, device
removed (`text_markup.py`, 36 clips), then a neutral-instruct follow-up
(`text_tags_neutral.py`, 11 clips):

| device class | verdict |
|---|---|
| **Lexical vocalisations** — `Haha,` `Ugh,` `Hmm.` `Oh!` | **Work.** Every one produced the intended sound. |
| **Typography** — ellipses, em-dash, ALL CAPS, `sooo` | **No effect.** Indistinguishable from controls; the instruct was doing the work. |
| **Bracketed / asterisked tags** — `[laughs]` `(laughs)` `*laughs*` `<laugh>` | **Unusable — and actively dangerous.** |

The tag result needs stating carefully, because the first run got it wrong. Under an
instruct that already said "laughing", no tag was ever read aloud and all four looked
safe — but the controls laughed too, so the instruct was producing the laugh and masking
the tags entirely. Re-run under a neutral instruct:

- `*laughs*` at the start — **spoken aloud.**
- `[laughs]` mid-sentence — laughed on take 1, **spoke the word on take 2.** Same input.
- `[laughs]`, `[gasps]` at the start — no effect at all, except `[gasps]` causing a
  stumble into the first word ("Tha… That is genuinely…").

So tags are ignored where they are safe, and spoken where they do anything,
non-deterministically. **A confounded control can manufacture a safety property that
does not exist** — the first run's "never read aloud" was an artifact, not a finding.

**Requirement that falls out of this: bracketed and asterisked stage directions must
never reach the TTS.** An LLM writing in-character dialogue emits `*laughs*` unprompted,
so both the prompt and the adapter should exclude them.

Also recorded: **the model injects paralinguistic content uninvited.** The `[sighs]`
control sighed without being asked, matching the upstream bug report about unwanted
laughter in output. A "neutral" read is not guaranteed clean.

### Instruct form

`instruct` must be written in **English or Chinese only**, independent of the output
language — Japanese instruct for a Japanese voice is silently ignored, which is the worst
available failure mode. Clip-caption form works and matches the house style already
proven on the 12 dubbing voices (`docs/voice-design.md`): delivery attributes only, no
speaker description, since the speaker is fixed by the token. For example
`"Tired and annoyed, slow and heavy with low pitch and low energy."`

The three plausible forms — caption, imperative (`"Read this in…"`), bare label — were
never compared head to head. Caption form was adopted on the strength of the existing
voice-design evidence, not a measurement.

### What this costs

**Ono_Anna is one of nine voices available to everyone using this model.** The character
will share a timbre with other projects. That was accepted deliberately, against the
alternative of a unique designed voice that only varies in pace. The mitigating argument
— untested — is that a Japanese voice reading English is already an unusual combination,
and that the writing carries more of the character than the timbre.

### Consequences for the server

The streaming endpoint does not support this today:

- `server.py:137` (`synthesize_stream`) and `server.py:228` (the route) both reject
  anything that is not `type: clone`; the manager only ever calls
  `generate_voice_clone_streaming`. `generate_custom_voice_streaming` exists at
  `model.py:1125` and is what a custom voice needs.
- `StreamRequest` has no free-text `instruct` field. Its `emotion` field resolves a named
  handle to a reference clip.
- Warmup loads CustomVoice automatically once a `type: custom` entry exists, but both
  models are then resident (~4.8 GB each — fine on the 4090 alone, less so under the GPU
  co-tenancy risk below).

**The emotive registry work — `voice:emotion` flattening and per-emotion clip baking — is
unused by this design.** It still works and media-worker is unaffected; it was built for
the clip-based emotion model that the evidence has since ruled out.

Open design question: whether `emotion` is replaced by free-text `instruct`, or both are
kept — a persona-declared handle resolving to an instruct string, with free text as an
override. The second preserves `08-persona.md`'s persona-declares-handles rule while
still allowing LLM-written direction per line.

---

# Recommendation

**Qwen3-TTS can serve lyrebird's `Voice` port, conditional on building two things that
do not exist yet: a streaming endpoint, and the Windows player sidecar.** Neither is
speculative — both were prototyped here and measured.

## The latency budget

| Stage | Measured | Note |
|---|---|---|
| TTS time-to-first-audio | **335 ms** | streaming, chunk_size=8 |
| Playback to audible | **61 ms** | WASAPI, 100 ms prefill |
| **TTS + playback subtotal** | **~400 ms** | |
| Remaining for the fast LLM | **~600 ms** | of the ~1 s beat budget |

~600 ms is enough for a local fast-profile model's first sentence. It is not enough for
a cloud model at typical round-trip latency. This corroborates `01-architecture.md`'s own
expectation that "a budget this tight likely points the fast profile at a local model" —
now with a number attached rather than an assumption.

Two caveats on that subtotal. It excludes the HTTP transport, because no streaming
endpoint exists to measure — the 335 ms is library-level. And it excludes GPU
contention, which was explicitly out of scope; the LLM sharing the 4090 will move these
numbers.

**Superseded by measurement.** The endpoint now exists and was measured end to end
against the chosen `ono_anna` custom voice, warm, 5 runs
(`spikes/streaming/stream_client.py`): **410–459 ms at `chunk_size` 8**, 325 ms at 4,
290 ms at 2. The model accounts for 304 ms of the cs=8 figure (prefill ~85 ms, decode
~222 ms) and **~130 ms is server and transport**, roughly constant across chunk sizes —
exactly the cost this subtotal excluded. The custom path is slightly cheaper at the model
level than the clone path's 335 ms, as expected from having no reference clip in the
prefill, but that saving is smaller than the transport cost it was hiding.

Revised budget at cs=8: **~410 ms TTS + 61 ms playback ≈ 471 ms**, leaving **~530 ms**
for the fast LLM rather than ~600 ms. cs=4 would return ~90 ms of that, but the headroom
argument for cs=8 (407 ms against a slow chunk, vs 31 ms at cs=4) was derived on the clone
path and has not been re-derived here.

The design that follows from this is in
`docs/superpowers/specs/2026-07-28-lyrebird-streaming-api-design.md`. Note one scope
change decided after the spike: **playback moves to lyrebird**, which needs the samples
for lip-sync anyway. This repo becomes synthesis only; the player sidecar built here
survives as reference for lyrebird's implementation.

## What the design must do, on the evidence

1. **Build a streaming endpoint.** The existing one-shot `/v1/audio/speech` cannot meet
   the budget — 1.41 s to first byte for one sentence. Leave it alone: media-worker's
   dubbing pipeline depends on it and is well served by it.
2. **Use chunk_size=8.** 335 ms TTFA with 407 ms of headroom, versus cs=4 saving 85 ms
   while running within 31 ms of a stutter.
3. **Player sidecar, not WSLg.** WSLg delivered no audio at all here, and failed
   *silently* — the writer exited 0 throughout. Do not adopt it even if a restart
   revives it.
4. **WASAPI, resampling, and a prefill under the chunk size** are all mandatory in the
   player, for the reasons in Experiment 1.
5. **Use the `ono_anna` built-in voice, directed by free-text `instruct`** (Stage B).
   Emotion cannot come from reference clips on a *designed* voice, because `instruct` is
   pace-only on the clone path and VoiceDesign cannot hold an identity across the calls
   needed to build a clip set. A built-in speaker has no reference recording competing
   with the instruction, so the instruction lands. The cost is a shared timbre, accepted
   deliberately. Lexical vocalisations (`Haha,` `Ugh,` `Hmm.`) work in the text and are
   worth exposing; bracketed and asterisked stage directions **must be stripped before
   the request** — they are ignored where safe and spoken aloud where not.
6. **Never health-check the audio path on a return code.** Both audio failures in this
   spike reported success. Health must be observed downstream.

## What remains unproven

- **OBS capture is unverified.** The final link in the chain; needs an operator check.
- **Stage A's conclusion is objective-only.** Pitch and energy are the measurable
  carriers of emotion, not the only ones. The 40 clips in `spikes/emotion/out/` were
  never listened to. Stage B's outcome does not depend on it — the clone path was ruled
  out on the strength of the built-in voices working, not on Stage A alone.
- **Stage B's decay observation is unproven.** One sample per cell, no repeats, never
  listened for. It argues for short per-request text; nothing has been built on it.
- **The instruct form was not compared.** Caption form was adopted from the existing
  voice-design evidence; imperative and bare-label forms were never tested against it.
- **Whether VoiceDesign obeys emotional direction is unresolved.** It was abandoned for
  identity drift before that question was settled, so the answer is unknown rather than
  negative.
- **The shared-timbre mitigation is untested.** That a Japanese voice reading English is
  distinctive enough, and that writing carries more character than timbre, are arguments,
  not findings.
- **Mid-utterance cancellation was excluded by decision.** `05-performer.md` still
  specifies it and needs amending to sentence-granularity cancellation.
- **GPU co-tenancy was excluded by decision** and is the largest unmeasured risk to the
  budget above.
