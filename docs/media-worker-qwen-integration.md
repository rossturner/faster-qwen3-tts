# media-worker ↔ Qwen3-TTS integration handoff

**Audience:** the `media-worker` team (Java/Spring).
**Status:** the server described here is fully built and tested in the `faster-qwen3-tts` repo. This
document is the implementation spec for the **Java side** — no Python changes are required.

The Qwen3-TTS server runs as a **GPU sidecar container**, exactly like the existing `melotts` /
`kokoro` services, and exposes the same **OpenAI-compatible `POST /v1/audio/speech`** contract. The
motivation is good **male Japanese/Korean** voices; **Korean is the first language to cut over**.

This document is self-contained — you should not need to read the server design spec to implement.

---

## 1. Wire contract

### `POST /v1/audio/speech`

Request body (JSON):

```json
{
  "input": "이 강좌에 오신 것을 환영합니다.",
  "voice": "ko_m",
  "response_format": "wav",
  "temperature": null
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `input` | yes | Text to synthesize. Must be non-empty (after trim) and ≤ **2000 characters**. |
| `voice` | yes | One of the 12 voice ids in §2. |
| `response_format` | no (default `"wav"`) | **`wav` only.** Any other value → `400`. |
| `temperature` | no (default `null`) | Optional override. When omitted, the server uses the per-voice default (`0.7`). |
| `model` | no | Accepted for OpenAI compatibility, **ignored**. |
| `speed` | no | Accepted for OpenAI compatibility, **currently a no-op** — speed is not applied. Do not rely on it. |

### Response

- `200 OK`, `Content-Type: audio/wav`.
- Body is **one complete WAV file**: **24 kHz, mono, 16-bit PCM**, with a valid RIFF/`fmt `/`data`
  header and correct `data` length. Non-streaming — the full file arrives in one response.

### Errors

| Status | When |
| --- | --- |
| `400` | empty/whitespace `input`; `input` > 2000 chars; `response_format` != `wav`; unknown `voice` id. |
| `503` | model not ready (warming up). |
| `500` | generation failure. |

Treat any non-2xx as a synthesis failure. As with the existing engines, retry on `5xx` / timeout;
`400` is a client error (bad request) and should not be retried.

> **Note vs. the design spec:** the implemented server bounds input by a **2000-character** limit
> (returns `400` above it), not a token-count check. Generation is additionally capped internally by
> `max_new_tokens` so a runaway decode cannot tie up the GPU. media-worker already chunks text
> upstream, so the 2000-char limit is a guard, not a normal path.

### `GET /health`

| Status | Body | Meaning |
| --- | --- | --- |
| `503` | `{"status":"warming"}` | Models still loading / CUDA graphs still capturing. |
| `200` | `{"status":"ok"}` | Both models loaded, clone prompts pre-baked, graphs warmed — ready to serve. |

The server accepts connections immediately and answers `/health` with `503` **throughout warmup**
(it does not refuse connections). Warmup takes on the order of **~5 minutes** cold (two 1.7B models
load + clone-prompt pre-bake + CUDA-graph capture); the Docker healthcheck below uses a `300s`
`start_period`. **Never route traffic until `/health` returns `200`.**

### curl example

```bash
curl -sS -X POST http://qwen-tts:8092/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"이 강좌에 오신 것을 환영합니다.","voice":"ko_m","response_format":"wav"}' \
  -o out.wav

# health
curl -f http://qwen-tts:8092/health
```

---

## 2. The 12 voice ids

Voice ids are `<lang>_<gender>`, so they map 1:1 onto media-worker's `(Language, Gender)` key. The
server resolves each id to a Base voice-clone internally — the caller only sends the id. All 12
voices clone from a pinned reference clip, so the server loads **only the Base model**.

**12 primary voices** (6 languages × male/female):

| Voice id | Language | Gender | Backend | `language` string sent to model |
| --- | --- | --- | --- | --- |
| `en_m` | English | male | Base clone | English |
| `en_f` | English | female | Base clone | English |
| `es_m` | Spanish | male | Base clone | Spanish |
| `es_f` | Spanish | female | Base clone | Spanish |
| `fr_m` | French | male | Base clone | French |
| `fr_f` | French | female | Base clone | French |
| `zh_m` | Chinese | male | Base clone | Chinese |
| `zh_f` | Chinese | female | Base clone | Chinese |
| `ja_m` | Japanese | male | Base clone | Japanese |
| `ja_f` | Japanese | female | Base clone | Japanese |
| `ko_m` | Korean | male | Base clone | Korean |
| `ko_f` | Korean | female | Base clone | Korean |

These 12 ids are the canonical registry (`faster_qwen3_tts/server_voices/voices.yaml`).

---

## 3. `(Language, Gender) → voice id` mapping

media-worker should map its `(Language, Gender)` enum pair to a Qwen voice id as follows. Note
**`zh-Hans` and `zh-Hant` both map to the `zh_*` voices** (same spoken Mandarin).

| Language | Gender | Voice id |
| --- | --- | --- |
| English | Male | `en_m` |
| English | Female | `en_f` |
| Spanish | Male | `es_m` |
| Spanish | Female | `es_f` |
| French | Male | `fr_m` |
| French | Female | `fr_f` |
| Chinese (`zh-Hans` / `zh-Hant`) | Male | `zh_m` |
| Chinese (`zh-Hans` / `zh-Hant`) | Female | `zh_f` |
| Japanese | Male | `ja_m` |
| Japanese | Female | `ja_f` |
| **Korean** | **Male** | **`ko_m`** |
| **Korean** | **Female** | **`ko_f`** |

The two Korean rows are the **first to switch** (see §6).

---

## 4. Audio handling downstream — no change

**Nothing changes downstream of the TTS call.** media-worker already:

- base64-encodes the returned audio bytes internally, and
- resamples the audio to 48 kHz via ffmpeg.

Qwen's **24 kHz mono 16-bit WAV** is accepted as-is — only a valid WAV header is required (which the
server always produces). There is no resampling or format change to add on the Java side.

---

## 5. Java change list (media-worker repo)

The server matches media-worker's existing OpenAI-compatible engine pattern (modeled on the Kokoro
client), so integration mirrors the MeloTTS/Kokoro wiring. Required changes:

1. **`QwenTTSService implements TTSService`** + **`QwenClient`** — a Spring `RestClient` caller
   that does `POST {base-url}/v1/audio/speech` with JSON body
   `{ "input", "voice", "response_format": "wav" }` and reads the response as `byte[]`. Model it on
   the existing `KokoroClient`.

2. **`QwenProperties`** — `@ConfigurationProperties(prefix = "tts.qwen")` exposing `base-url`,
   `timeout`, and an explicit **read timeout** sized for synthesis latency. (The MeloTTS client has
   **no** read timeout; add one here so a hung generate cannot block a worker thread indefinitely.)

3. **Enums:**
   - add **`QWEN`** to the `TtsProvider` enum;
   - add **`QWEN_TTS`** to the `Resource` enum (the per-engine resource/serialization lock).

4. **`CompositeTTSService`** — add a routing case for provider `QWEN` that acquires the `QWEN_TTS`
   lock before calling `QwenTTSService` (the server serializes all GPU work on a single thread, so
   single-stream/serialized access matches it).

5. **`PresetVoiceMapper`** — add the routing rows, **Korean first**:
   - `(KOREAN, MALE)   → (QWEN, "ko_m")`
   - `(KOREAN, FEMALE) → (QWEN, "ko_f")`

   Extend the other languages later (the server already serves all 12 primary ids). Voice ids are
   `<lang>_<gender>` (see §3).

6. **`QwenHealthIndicator`** — `@Component implements HealthIndicator` probing `GET /health` with
   ~5 s timeouts, mirroring `MeloTTSHealthIndicator`. Healthy only when `/health` returns `200`.

7. **`application.yml`** — `tts.qwen.base-url: http://qwen-tts:8092`.

8. **docker-compose** — add the `qwen-tts` service (GPU reservation, port `8092`, HF cache volume,
   healthcheck with a generous `start_period`). media-worker waits on the healthcheck before
   routing. A ready-to-adapt snippet ships in this repo at **`docker-compose.example.yml`**:

   ```yaml
   services:
     qwen-tts:
       build: .
       ports: ["8092:8092"]
       volumes:
         - ${HOME}/.cache/huggingface:/hf-cache
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8092/health"]
         interval: 30s
         timeout: 5s
         start_period: 300s
         retries: 3
   ```

Port **8092** is the convention (next after melotts `8091`).

---

## 6. Rollout — repoint Korean first

1. **Cut over Korean only.** Switch the two Korean `PresetVoiceMapper` rows to the Qwen provider:
   `(KOREAN, MALE) → (QWEN, "ko_m")` and `(KOREAN, FEMALE) → (QWEN, "ko_f")`.
   Leave every other `(Language, Gender)` on its current engine (MeloTTS/Kokoro).
2. **Validate in the dubbing pipeline** — run real Korean dubbing jobs end to end, confirm audio
   quality, latency, and the resource-lock serialization behave as expected. `ko_m` (male Korean) is
   the headline new capability MeloTTS could not provide.
3. **Extend incrementally** once Korean is proven: Japanese, then Chinese, English, Spanish, French
   — by adding the corresponding `PresetVoiceMapper` rows from §3. The server already serves all 12
   primary voices, so each extension is a mapper change only, no server work.
