# Billy Kid — the voice and its filter

The single source of truth for **why Billy needs post-synthesis processing, how the
settings were arrived at, and what was tried and rejected**. Machine-readable config lives
in
[`faster_qwen3_tts/server_voices/characters/billy/character.yaml`](../faster_qwen3_tts/server_voices/characters/billy/character.yaml);
the implementation is
[`faster_qwen3_tts/audio_filter.py`](../faster_qwen3_tts/audio_filter.py); the exploratory
scripts and every audio artefact are under `spikes/billy_filter/`.

Read this before changing any number in that YAML. Several of the values are weakly
determined, several plausible-looking approaches were tried and failed, and the most
expensive mistakes in this work came from measurements that looked authoritative and
weren't.

## The problem

Billy Kid is a cyborg, and his Zenless Zone Zero dialogue carries a processing effect that
makes him sound synthetic. His reference recordings here are the
`GalGame_TheGoldenMechaGodBattle_*` assets, which are the same performer with the effect
**not applied** — an oversight in the game's own audio pipeline, and the only clean source
of his unprocessed voice.

That is the right material to clone from: the speaker encoder gets a natural voice instead
of one carrying inharmonic processing it will reproduce inconsistently. But it means the
clone sounds like an ordinary man, so the effect has to be reapplied downstream.

**Reapplying it after synthesis is deliberate.** Baking the effect into the reference clips
instead would ask the model to reproduce a chorus as *timbre* — the same failure mode as
cloning from already-processed audio, where the result varies take to take. Applied
downstream it is deterministic and identical on every request.

## What was measured, and why it was wrong

The first three attempts derived the filter by differencing the dry `GoldenMechaGodBattle`
recordings against 60 processed dialogue lines. That produced a confident-looking result:

- a low-end cut, −7 dB below 100 Hz
- an untouched midrange, ±1 dB across 300–3000 Hz
- a broad high-frequency shelf, +5 dB at 4 kHz rising to a +8–12 dB plateau over 6–12 kHz
- 90–93% of individual processed files above the dry median in every band over 4 kHz

Both sets were 48 kHz PCM_16 with an identical ~18.5 kHz codec cliff, so the asset pipeline
was ruled out as the cause. Six different mechanisms were built to reproduce that shelf —
linear EQ, envelope-gated noise, a channel vocoder, phase randomisation, spectral
translation, waveshaping, sample-and-hold. **Every one was reported inaudible.**

The reason: *the dry set is one scene and the processed set is sixty lines from
everywhere else*. Different takes, different vocal range, different delivery. The spectral
difference was mostly the actor, not the effect. A population comparison can show two sets
differ consistently without showing what *causes* the difference, and no amount of
per-file verification fixes that — the check confirmed the sets differ, which was never in
doubt.

**Discard the HF shelf finding.** It is recorded here only so nobody re-derives it.

What does survive is the narrower set of negative results, because those compared each
frame against its own fundamental rather than comparing sets:

- partials sit at identical positions in both (median ratio 0.9992 over ~40k measurements)
- no envelope-modulation carrier
- no difference in comb spacing
- unchanged odd/even harmonic balance

Those argue against ring modulation, frequency shifting, fixed comb filtering and
clipping. They do **not** rule out a subtle detune: that test's resolution was ~0.017 in
frequency ratio, and an 18-cent detune is 0.010 — it could never have seen the thing that
turned out to be right.

## What worked: by ear, family first

Measurement having failed, the effect was found by casting a wide net over effect
*families* at deliberately excessive strength, then narrowing. The question at each round
was never "does this sound good" but "is this the right *kind* of wrong".

| Family | Verdict |
|---|---|
| Ring modulation | too robotic, Dalek-like |
| Zeroed-phase "robotize" | too subtle even at full wet |
| Comb / metallic resonance | too subtle |
| Radio/intercom bandpass | **a different effect**, sometimes layered over Billy in game, not the base |
| **Detune / doubling** | **the lead** |
| Formant shift | wrong, reads as pitched up |
| Heavy low cut | thin, not the effect |
| Bitcrush | wrong kind of digital |

Doubling then narrowed over five rounds of A/B, two files at a time. Diagnostics mattered
more than sweeps: a single detuned copy lost to a pair, delay-without-detune lost, and
detune-without-delay lost — so both ingredients are required and neither is decorative.

### Settled values, and how firmly

```yaml
filter:
  type: chorus
  cents: [26, -26]
  delays_ms: [8, 16]
  amount: 0.50
  window_ms: 42.7
  makeup_db: 5.09
```

| Parameter | Confidence |
|---|---|
| Two copies, not one | **firm** — a single copy lost cleanly |
| Both detune and delay present | **firm** — each alone lost |
| `amount: 0.50` | **firm** — the only axis that ever moved audibly; 0.40 too dry, 0.70 too much, toned from 0.55 on a later listen |
| `cents: [26, -26]` | **weak** — indistinguishable anywhere from 12 to 80 cents |
| `delays_ms: [8, 16]` | **weak** — 22 ms was clearly too far, 2 ms acceptable, nothing in between separable |
| `window_ms: 42.7` | **firm** — 21 ms flatter, 85 ms "too far apart" |
| `makeup_db: 5.09` | **firm** — measured for `amount: 0.50`; restores level to within 0.01 dB |

The weakly-determined values are a real finding, not a gap. Across a 4× change in detune
the residual-difference metric moved by less than 0.5 dB, which matches the difficulty of
hearing it. Do not treat 26 and 8/16 as tuned constants; do not "improve" them without a
listening test that can actually resolve them.

**Both other axes are exhausted.** A later round swept detune and delay again, level-
matched, across four emotions. Changing the detune was reported as doing the same thing as
changing the amount — the two are perceptually redundant, which fits detune being
indistinguishable from 12 to 80 cents — and delays from 4 to 8 ms could not be told apart
at all. `amount` is the only knob worth reaching for; the other two have now failed to
separate twice.

**A note on metrics.** Correlation against the reference implementation scored +0.73–0.75
for versions that sounded clearly wrong *and* versions that sounded right. It never once
predicted a listening result. It is not evidence here.

## The streaming implementation

The filter has to work on `/v1/audio/stream`, which emits every ~333 ms, so an offline
algorithm that needs the whole utterance would seam several times a second. Three attempts:

**1. Delay-line pitch shifting — structurally wrong.** Reading a delay line at rate *r*
shifts pitch by *r* while the delay drifts, so the drift is wrapped every grain with a
crossfaded second tap. But the two taps sit half a grain apart, so crossfading them across
the grain leaves *four* copies plus the dry audible nearly always. Reported immediately as
"too many copies of the voice". Shortening the crossfade cannot rescue it: suppressing the
overlap requires a large drift range, and the drift range *is* the delay — the parameter
the whole filter is built on. Delay-line shifting cannot hold a fixed delay and a fixed
detune simultaneously.

**2. Bin-shifting phase vocoder — two defects.** At 26 cents the shift is ~3 Hz against
47 Hz bins, so `round(k·r) == k` for every bin below ~33 and the low end was not shifted at
all. And bin shifting moves pitch while approximately preserving the spectral envelope,
where the approved effect scales formants too. Reported as "flatter".

**3. Time-stretch then resample — correct.** Phase-vocoder stretch by *Hs/Ha* followed by
fractional resampling, which is what `librosa.pitch_shift` does: the resample scales the
whole spectrum including formants, so each copy reads as a slightly different voice, and
that difference is where the richness comes from. Both stages carry state between calls.

Chunked and whole-file output are **sample identical** — asserted in
`tests/test_audio_filter.py` at 333 ms, 83 ms, 512-sample and 1-sample chunks. `flush()`
at end of stream releases the held window; without it the last window of every line is
silently lost.

Cubic interpolation rather than windowed sinc in the resampler: at a ~1.5% rate change the
two were indistinguishable by ear, and cubic is 3.4× cheaper.

### Loudness — the filter is not level-neutral by construction

The copies are pitch-shifted, so they are decorrelated from the dry and from each other.
The mix therefore sums as **power, not amplitude**, and loses level even though nothing is
attenuated. Measured at `amount: 0.50` over four clones: **−5.09 dB** RMS, tight (−4.86 to −5.21).
Decorrelation alone predicts less; the delay lines decorrelate further. It scales steeply
with `amount` — 4.01 dB at 0.40, 5.61 dB at 0.55 — which is why the two must move together.

Reported from listening as Billy being quieter than the other characters, which is exactly
what it is: he sits next to unfiltered voices, so any level change shows.

`makeup_db: 5.09` restores it, measured back to within **0.01 dB** mean.
Two properties worth knowing:

- **It is declared, not derived.** The loss is content-dependent (−5.1 dB on a harmonic
  probe, −6.0 dB on noise, −5.7 dB on real speech), so auto-calibrating from a probe would
  be ~0.5 dB off. Changing `amount` or the number of copies means re-measuring `makeup_db`
  — they are edited together, and a test pins that they do not track each other.
- **A soft knee follows it.** The chorus raises crest factor as well as lowering RMS, so
  restoring the level overshoots on transients: 0.001–0.003% of samples exceed full scale,
  peaking at 1.16. A memoryless `tanh` knee above 0.9 bends them. Memoryless is required —
  a look-ahead limiter would add latency and break chunk invariance.

### The sample-rate bug — read this before touching `window_ms`

The window was originally specified as `n_fft` in **samples**. Its audible effect is how
much the vocoder smears in **time**, so a fixed sample count is a different filter at every
sample rate:

- 2048 samples at 48 kHz = 42.7 ms — auditioned and **chosen**
- 2048 samples at 24 kHz = 85.3 ms — auditioned and **rejected**, "goes too far"

All tuning was done on 48 kHz files; the server runs at 24 kHz. It therefore shipped the
rejected variant, reported as "too much echoing/doubling/too far apart". `window_ms` is now
a duration and `n_fft` is derived per sample rate. `test_the_window_is_the_same_duration_at_every_sample_rate`
pins it.

## Costs

Measured on an RTX 4090, 1.7B Base, at the server's 24 kHz.

| | |
|---|---|
| Algorithmic latency | 42.7 ms, constant |
| TTFA penalty, `chunk_size` 8 | +18.7 ms |
| CPU | ~1 ms per chunk, under 2% of realtime |

The latency is mostly *not* charged to TTFA, because a decoded chunk is normally longer
than the window, so the filter still emits on the first chunk. At the earlier 85.3 ms
window `chunk_size=1` was a cliff (+363 ms) — an 83.3 ms chunk was just under the window,
so the first chunk emitted nothing. At 42.7 ms every chunk size clears it.

**Benchmark on an idle GPU.** Work is serialised on one worker thread, so a second client
generating concurrently inflates TTFA into the seconds and looks exactly like a
regression — this cost a round of false diagnosis during development.

## Reference library

Billy's references are the 11 dry `GoldenMechaGodBattle` recordings, across 7 emotions.
`earnest`, `deadpan` and `annoyed` have none and fall back to neutral.

`405_008` was moved out of `neutral` into `panicked` during this work: its transcript is
"Manager! I think I just screwed up the negotiation!", and since the server picks a
reference at random per request, a panicked take in `neutral` made Billy's plain voice
change unpredictably between requests. **The same reasoning applies to any future
addition** — a reference's delivery is cloned along with its timbre, so `neutral` must
hold only neutral takes.

## Open

- **Nothing is committed** as of writing: the reference library is untracked and the code
  changes unstaged.
- The three surviving `neutral` references have not been compared for clone quality
  against each other. Rendered at `spikes/billy_filter/out/refs/` and `out/level/`.
- The filter has only been auditioned on conversational lines. Emotion comes from the
  reference clip on the ICL clone path, so an `excited` request uses a different reference
  entirely and has not been checked with the filter applied.
