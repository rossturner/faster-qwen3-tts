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

## Experiment 3, Stage B — not started.
