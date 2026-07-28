# Lyrebird streaming TTS API — design

Transient design artifact. Defines what gets built **in this repo** to serve lyrebird's
`Voice` port, and what deliberately does not. Follows from the measurements in
`docs/lyrebird-tts-spike-findings.md`; authority on lyrebird itself remains
`~/workspace/lyrebird/docs/00`–`08`.

## The split

**This repo becomes a streaming synthesis service and nothing else.** Playback,
buffering, lip-sync and avatar control all live in lyrebird, which needs the samples
anyway to drive mouth shapes.

| | Owner |
|---|---|
| Streaming synthesis endpoint, voice/emotion registry | **faster-qwen3-tts** |
| Sentence splitting, batching policy, cancellation decisions | **lyrebird** |
| Windows playback, jitter buffer, device selection | **lyrebird** |
| Lip-sync, VTube Studio, OBS | **lyrebird** (boundary deferred) |

The spike's `spikes/audio_path/win_player.py` moves to lyrebird as reference, carrying
its four hard-won constraints: WASAPI (not MME), resample from 24 kHz, prefill smaller
than the feed chunk, and count zero-fill rather than trusting underflow flags.

## Batching: one knob, owned by lyrebird

TTFA is flat with input length — 0.336 s for one sentence, 0.334 s for two, 0.335 s for
three. **Batching costs nothing at the TTS.** The reason to send an early first sentence
is that the LLM has not produced the rest yet, not that longer input is slower.

The server cannot report sentence boundaries: `fast_generate_streaming` yields only
chunk timings, and while text hiddens are fed one per decode step
(`trailing_text_hiddens[:, gen_step]`), the audio for a given text token lags its feed,
so any derived boundary is approximate. Exact boundaries would need a forced aligner.

Therefore:

> **Whatever lyrebird batches into one request is its cancellation unit.**
> Send `(s1)(s2+s3)` and you may stop after `s1` or after the batch, never between
> `s2` and `s3`.

This is a feature, not a limitation to work around: segmentation becomes a single knob
trading **prosodic flow** (larger batches — the model sees the whole text and picks its
own inter-sentence pauses) against **interrupt granularity** (smaller batches). The beat
planner can vary it by context. The server stays stateless and knows nothing about
sentences, which matters because it also serves media-worker.

**Assumption, accepted without measurement by decision:** that multi-sentence generation
flows more naturally than concatenated single-sentence generations. Plausible but
untested. If seams later sound wrong, this is the first thing to check —
`spikes/` is the place for that A/B (F0 discontinuity and pause duration at seams).

## The endpoint

New, additive. **The existing `POST /v1/audio/speech` is not touched** — media-worker's
dubbing pipeline depends on it and whole-file responses suit it.

### `POST /v1/audio/stream`

```json
{
  "input": "Okay chat, that was closer than I want to admit.",
  "voice": "nicole",
  "emotion": "amused",
  "temperature": null,
  "chunk_size": 8
}
```

`voice` and `emotion` are **handles**, resolved to reference clips by registry config —
per `08-persona.md`, persona content never names an engine artifact. `emotion` is
optional and falls back to the voice's default. `chunk_size` defaults to **8**: 335 ms
TTFA with 407 ms of headroom against a slow chunk, where cs=4 saves 85 ms of TTFA while
running within 31 ms of a stutter.

### Response framing

`200`, `Content-Type: application/vnd.lyrebird.tts-stream`, `Transfer-Encoding: chunked`.
A stream of length-prefixed frames:

```
  1 byte   frame type
  4 bytes  payload length, big-endian
  N bytes  payload
```

| Type | Payload | Meaning |
|---|---|---|
| `0x01` header | JSON `{sample_rate, channels, format}` | first frame; makes the stream self-describing |
| `0x02` audio | raw s16le PCM | one decoded chunk |
| `0x03` mark | JSON `{chunk_index, decode_ms, prefill_ms, audio_ms_so_far}` | feeds lyrebird's marks ring and the Observatory timeline |
| `0x04` error | JSON `{message}` | failure **after** the response began |
| `0x05` end | JSON `{total_audio_ms, total_decode_ms}` | clean completion |

Framing exists for one reason worth stating: **once the first byte is sent, the HTTP
status is already committed**, so a mid-stream failure cannot be reported as a 5xx. The
error frame is how a truncated generation is distinguished from a completed one — and
lyrebird must treat *stream ended without an end frame* as a failed beat, not a short
one, or the ledger will record speech that never happened.

Errors before the first frame use normal HTTP status codes, matching the existing
endpoint: `400` bad voice/emotion/empty input, `503` warming, `500` generation failure.

### Cancellation

Lyrebird closes the connection. **The server must abort the decode loop on disconnect,
not merely stop writing.** GPU work is serialised on one worker thread, so a cancelled
three-sentence batch that keeps decoding to completion becomes a head-of-line blocker
that delays the *next* beat — the opposite of what cancelling was for. This needs an
explicit disconnect check in the decode loop; it does not come for free from the
framework.

## Registry: emotion as a second axis

`voices.yaml` grows from `voice → clip` to `voice → emotion → clip`, with clone prompts
pre-baked per pair at warmup exactly as they are per voice today:

```yaml
voices:
  nicole:
    language: English
    default_emotion: neutral
    emotions:
      neutral: {ref_audio: refs/nicole_neutral.wav, ref_text: "..."}
      amused:  {ref_audio: refs/nicole_amused.wav,  ref_text: "..."}
```

**The existing 12 flat entries stay valid and unchanged** — media-worker must not be
disturbed by this. The loader accepts both shapes.

Emotion comes from reference clips because Stage A showed `instruct` moves only pace on
the ICL clone path (F 7.26 for duration, 1.47 for F0, 0.53 for RMS, against a control
that responded on all five metrics). `instruct` may still be exposed as a secondary
pace control, which is the one thing it demonstrably does.

Per the old system's `EmotionCache`, an emotion may map to *several* clips chosen at
random, so repeated emotions do not sound identical. Worth keeping; not required for v1.

## Deployment

A second instance, separate from media-worker's: own port, own registry, own process.
Chosen so a long dubbing generate cannot head-of-line block a beat. ~4.8 GB more VRAM
and its own warmup; both fit the 4090.

## Sequence

1. **Stage B — persona voice + emotion clips.** Blocks anything sounding like the
   persona. Independent of the API work, so the two can proceed in parallel.
2. **Streaming endpoint + registry extension**, developed against the existing `en_f`
   voice; contract tests over the frame protocol, mocked as `tests/test_server.py` does.
3. **Lyrebird side** — `adapter:tts` implementing `Voice`, plus the player sidecar.
4. **Lip-sync** — deferred; see open items.

## Open items

- **Lip-sync boundary undecided.** `01-architecture.md` puts playback under Voice and
  lip-sync under Body & Stage, but only the player holds the samples. Qwen3-TTS cannot
  supply phoneme or viseme timings — it emits codec tokens with no aligned text — so
  mouth shapes must be derived from audio amplitude at playback time. Delivery chunk
  size does not constrain this: lyrebird holds the PCM and can window it at 30–50 ms
  regardless of the 667 ms delivery chunks.
- **`05-performer.md` still specifies mid-utterance cancellation** and needs amending to
  batch granularity. Lyrebird's repo, not this one.
- **OBS capture unverified** (spike open item).
- **GPU co-tenancy unmeasured** — excluded by decision, and the largest unquantified
  risk to the ~600 ms left for the LLM.
