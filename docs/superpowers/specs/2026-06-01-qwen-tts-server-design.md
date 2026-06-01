# Qwen3-TTS HTTP server for media-worker — design spec

- **Date:** 2026-06-01
- **Status:** Approved (brainstorming) — ready for implementation planning
- **Repo:** `faster-qwen3-tts` (this repo). Consumer: `media-worker` (separate repo, Java/Spring).

## 1. Context & motivation

`media-worker` dubs video and currently synthesizes Korean speech with **MeloTTS**, which has no
distinct male Korean voice. We are replacing it with **Qwen3-TTS** (via this CUDA-graph-optimized
fork) — Korean first, then the other target languages. The deliverable is a **GPU Docker image**
running an HTTP TTS server that media-worker calls as a sidecar, exactly like its existing
`melotts`/`kokoro` services.

The 12 production voices (6 languages × male/female) were selected and recorded in
[`voice-mapping.md`](../../../voice-mapping.md): 2 are CustomVoice built-in presets (`aiden`,
`sohee`); the other 10 are VoiceDesign-designed then reproduced by Base voice-clone from a pinned
reference clip.

## 2. Goals / non-goals

**Goals**
- A packaged, tested FastAPI server exposing OpenAI-compatible `POST /v1/audio/speech` → complete WAV.
- Serve all 14 voices (12 primary + 2 clone-of-builtin) from a versioned registry.
- Ship as a GPU Docker image consuming a mounted Hugging Face cache volume.
- Warmup-gated `/health` for orchestration.
- Versioned clone-source clips + provenance doc + reproducible generation scripts.

**Non-goals (this spec)**
- The media-worker Java integration is **documented as a handoff** (§12), not implemented here.
- No streaming endpoint (media-worker reads full bytes; a complete WAV is required for its header
  parse). No multi-GPU / horizontal scaling (single GPU, serialized calls).
- VoiceDesign is **not** loaded at runtime — it was only used to create the reference clips.

## 3. The 14 voices

Voice ids are `<lang>_<gender>` so they map 1:1 onto media-worker's `(Language, Gender)` key.
`zh-Hans` and `zh-Hant` both map to `zh_*` (same spoken Mandarin).

| id | method | model | source |
| --- | --- | --- | --- |
| `en_m` | custom | CustomVoice | speaker `aiden` |
| `en_f` | clone | Base | `refs/en_f_v1_ref.wav` |
| `es_m` | clone | Base | `refs/es_m_v2_ref.wav` |
| `es_f` | clone | Base | `refs/es_f_v3_ref.wav` |
| `fr_m` | clone | Base | `refs/fr_m_v1_ref.wav` |
| `fr_f` | clone | Base | `refs/fr_f_v3_ref.wav` |
| `zh_m` | clone | Base | `refs/zh_m_v3_ref.wav` |
| `zh_f` | clone | Base | `refs/zh_f_v3_ref.wav` |
| `ja_m` | clone | Base | `refs/ja_m_v1_ref.wav` |
| `ja_f` | clone | Base | `refs/ja_f_v3_ref.wav` |
| `ko_m` | clone | Base | `refs/ko_m_v3_ref.wav` |
| `ko_f` | custom | CustomVoice | speaker `sohee` |
| `en_m_clone` | clone | Base | `refs/aiden_ref.wav` (clone of the aiden preset) |
| `ko_f_clone` | clone | Base | `refs/sohee_ref.wav` (clone of the sohee preset) |

The two `*_clone` voices let media-worker A/B preset vs clone, and keep a Base-only deployment
option open without re-plumbing.

## 4. HTTP contract

### `POST /v1/audio/speech`
Request JSON:
```json
{ "input": "<text>", "voice": "<id>", "response_format": "wav",
  "model": "<ignored>", "speed": 1.0, "temperature": null }
```
- `input` (required), `voice` (required, one of the 14 ids).
- `response_format`: `wav` (default). `pcm` optional. `model`/`speed` accepted for OpenAI
  compatibility but ignored (`speed` may be applied later). `temperature` optional override;
  default is the per-voice value (0.7).

Response: `200`, `Content-Type: audio/wav`, body = **one complete WAV file** — 24 kHz mono 16-bit
PCM with a valid RIFF/`fmt `/`data` header and correct data length (media-worker parses the header
for duration). Non-streaming.

Errors:
- `400` — empty/whitespace `input`, or unknown `voice` id.
- `503` — model not loaded / warmup not complete.
- `500` — generation failure.

media-worker treats any non-2xx as failure and retries on 5xx/429/timeout, so transient/internal
failures must surface as `5xx` (not `400`).

### `GET /health`
- `200 {"status":"ok"}` only after both models are loaded, clone prompts pre-baked, and CUDA graphs
  warmed.
- `503 {"status":"warming"}` before that. This gates the Docker healthcheck and media-worker's
  `QwenHealthIndicator`.

## 5. Voice registry & source-controlled clone sources

Co-located under the package and **tracked in git**, bundled into the image as package data:
```
faster_qwen3_tts/server_voices/
  refs/*.wav      # the 12 clone-source clips (force-tracked)
  voices.yaml     # the 14-voice registry
  README.md       # provenance: how each clip was created
```

- **`refs/`** — exactly the clips the server clones from: the 10 designed `*_ref.wav` (promoted out
  of `voice_design_audition/`), plus `aiden_ref.wav` and `sohee_ref.wav` (CustomVoice renders of the
  two presets speaking the reference line, used by the `*_clone` voices). `*.wav` is globally
  gitignored, so a negation is added (precedent: `!ref_audio.wav`).
- **`voices.yaml`** — the registry. Each entry:
  - `type: custom` → `{ speaker, language, instruct }`
  - `type: clone`  → `{ ref_audio (refs/…wav), ref_text, language }`
  - common: `temperature: 0.7`.
  YAML also avoids the existing `*.json` ignore rule.
- **`README.md`** — provenance per clip: source model (VoiceDesign vs CustomVoice), the exact
  `instruct` prompt or speaker id, the reference-line text, `temperature=0.7`, the generating script,
  and date; plus the design→clone explanation. Cross-links `voice-mapping.md`.

The reference line per language (the `ref_text` used for cloning) and the persona prompts are as
recorded in `voice-mapping.md`.

**Reproducibility:** the generation scripts are committed alongside —
`design_audition_library.py` (the 10 designed refs), `redo_ja_female.py` (the `ja_f` re-roll),
`gen_voice_samples.py` (the aiden/sohee CustomVoice renders), and a small new script that produces
`aiden_ref.wav` / `sohee_ref.wav` for the builtin-clone voices.

## 6. Model loading & warmup

On startup the server:
1. Loads **CustomVoice** and **Base** (both 1.7B, bf16, `attn_implementation="sdpa"`).
2. Pre-bakes the clone prompts for all `clone` voices from `refs/` (skips per-request extraction),
   using the fork's voice-clone-prompt API (see `tests/test_voice_clone_prompt_api.py`).
3. Warms up both models (triggers CUDA-graph capture) with a short dummy generate each.
4. Flips `/health` to ready.

**De-risking spike (do first):** confirm two graph-captured `FasterQwen3TTS` instances
(CustomVoice + Base) coexist on one 24 GB GPU. Prior usage only ever loaded them sequentially. If
coexistence fails, fall back to **Base-only** (clone `aiden`/`sohee` too — the `*_clone` voices
already prove this path) and drop CustomVoice. Peak VRAM target: well under 24 GB (~10 GB expected).

## 7. Server internals

- FastAPI + uvicorn, **single process / single worker** (one GPU).
- A global lock serializes GPU inference; generation runs in a threadpool executor so the event loop
  is not blocked, but only one inference proceeds at a time (matches media-worker's per-engine
  resource lock).
- Routing by voice `type`:
  - `custom` → `generate_custom_voice(text, speaker, language, instruct, temperature)`
  - `clone`  → `generate_voice_clone(text, language, voice_clone_prompt=<prebaked>, ref_text,
    xvec_only=False, temperature)`
- Build the complete WAV in memory (reuse the reference server's WAV helper) and return it as the
  response body.

## 8. Packaging & CLI

- New module `faster_qwen3_tts/server.py` (FastAPI app) + the `server_voices/` registry package data.
- New CLI subcommand **`faster-qwen3-tts serve-http`** (`--host`, `--port`, `--voices`, `--device`),
  reusing the existing `[demo]` extra (FastAPI/uvicorn already declared in `pyproject.toml`).
- The existing stdin-loop `serve` command is left unchanged.

## 9. Docker image (shipped deliverable)

- CUDA-capable Python base; install torch (cu130 wheels) + `faster-qwen3-tts[demo]`; bundle the
  package incl. `server_voices/` (code + registry + ref clips).
- **Models via mounted HF cache volume** → `HF_HOME` (e.g. `/hf-cache`), pre-populated once from the
  host's existing `~/.cache/huggingface`. Documented populate step.
- GPU via nvidia-container-toolkit (compose `deploy.resources.reservations.devices` / `--gpus all`).
- Expose a port (default **8092**, next after melotts `8091`).
- **Healthcheck:** `curl -f http://localhost:8092/health`, generous `start_period` (~120 s, covering
  cold graph capture + clone prebake + both models), `interval` ~30 s. Container is "healthy" only
  once warmup completes.
- Entrypoint runs `serve-http` via uvicorn, single worker.

## 10. Testing

- **pytest with the model mocked** (monkeypatch `FasterQwen3TTS`), runs on CI without a GPU:
  - registry loads and resolves all 14 voice ids;
  - request validation: empty `input` → 400, unknown `voice` → 400;
  - WAV-header correctness (RIFF/`fmt `/`data`, sample rate, data length) and a sane duration;
  - health gating: `503` before ready, `200` after.
- **Optional GPU smoke test** (pytest marker, opt-in): start the server, generate one clip per voice
  id, assert non-empty valid WAV. Not run on CI.

## 11. Risks & open items

- **Two-model coexistence** (see §6 spike) — primary technical risk; Base-only fallback defined.
- **Cold-start latency** — first graph capture ~47 s; mitigated by warmup-gated health + Docker
  `start_period`. Server must never be marked healthy mid-warmup.
- **`aiden`/`sohee` clone quality** — the `*_clone` voices are for comparison; if they match the
  presets closely, a future simplification to Base-only becomes viable.
- **Read timeout** — media-worker's MeloTTS client has no read timeout; the handoff notes adding one
  for the Qwen client so a hung generate cannot block a worker thread indefinitely.

## 12. Handoff appendix — media-worker Java integration (not built here)

The server matches media-worker's existing OpenAI-compatible engine pattern. To integrate (in the
media-worker repo, its own spec):

1. **`QwenTTSService implements TTSService`** + **`QwenClient`** (Spring `RestClient`,
   `POST {base-url}/v1/audio/speech`, JSON body `{ "input", "voice", "response_format":"wav" }`,
   read `byte[]`), modeled on `KokoroClient`.
2. **`QwenProperties`** `@ConfigurationProperties(prefix="tts.qwen")` with `base-url`, `timeout`, and
   an explicit **read timeout** (MeloTTS lacks one). `application.yml` `tts.qwen.base-url:
   http://qwen-tts:8092`.
3. Enums: add `QWEN` to `TtsProvider`, `QWEN_TTS` to `Resource`. `CompositeTTSService` gets a `QWEN`
   case that acquires the `QWEN_TTS` lock (single-stream).
4. **`PresetVoiceMapper`** — repoint the Korean rows first:
   `(KOREAN, MALE) → (QWEN, "ko_m")`, `(KOREAN, FEMALE) → (QWEN, "ko_f")`. Extend other languages
   later (the server already serves all 12). Voice ids are `<lang>_<gender>`.
5. **`QwenHealthIndicator`** (`@Component implements HealthIndicator`) probing `GET /health` with
   ~5 s timeouts, mirroring `MeloTTSHealthIndicator`.
6. **docker-compose** service `qwen-tts`: the image from §9, GPU reservation, port `8092`, HF cache
   volume, healthcheck. media-worker waits on the healthcheck before routing.

Audio handling needs no change: the worker base64-encodes the returned WAV internally and resamples
to 48 kHz downstream via ffmpeg; Qwen's 24 kHz mono WAV is accepted as-is (only a valid header is
required).
