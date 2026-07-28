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

**Status: path B not yet built.** It needs `sounddevice` installed into the Windows
Python 3.13.5 (pip 25.1.1 present) — a change to the Windows host, pending operator
approval.

---

## Experiment 3 — emotion control. Not started.
