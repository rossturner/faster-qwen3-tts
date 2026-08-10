# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is / why we use it here

`faster-qwen3-tts` (by andimarafioti) is a CUDA-graph-optimized fork of Qwen3-TTS inference. It wraps the upstream `qwen-tts` model's own forward pass in `torch.cuda.CUDAGraph` + transformers `StaticCache`. Same algorithm as upstream, different (faster) kernel path. **Measured here (2026-06-01, RTX 4090, 1.7B, bf16, no flash-attn): RTF ~3.76 non-streaming / ~3.2 streaming, vs ~0.36x realtime for the stock package — a ~10x speedup.** (The fork's README claims 4.22 on a 4090; the small gap is our torch 2.12/cu130 + eager-attn stack. See `## Verified state` for the full numbers.)

We are trialing this to **replace MeloTTS as the TTS engine for the `media-worker` dubbing pipeline** (Java/Spring worker at `/home/ross/workspace/prentora/media-worker`), starting with **Korean** and extending to Japanese/Chinese/English. Speed is the gating factor: stock 1.7B without flash-attn runs ~0.36x realtime on our RTX 4090 and the worker serializes TTS calls, so the CUDA-graph speedup is what makes Qwen3-TTS viable in production.

Scope notes for this trial:
- **We only care about the 1.7B model.** The fork's published benchmarks are 0.6B (and 0.6B is faster), but 0.6B is a quality downgrade and out of scope.
- The primary motivation is **good male Japanese/Korean voices**. Qwen3-TTS CustomVoice has no male JP/KO preset (the only JP/KO presets are female: Ono_Anna, Sohee), so male JP/KO must come from the **VoiceDesign** model (natural-language instruct) or the **Base** model's voice-clone.
- Output is 24 kHz. media-worker is sample-rate-agnostic and normalizes to 48 kHz at output, so no resampling is needed.
- Quality must not regress. CUDA graphs + StaticCache are algorithmically equivalent to upstream but not bit-identical (BF16/TF32 kernel/reduction differences); a parity mode exists to validate.

Sibling repo `../Qwen3-TTS` is the official Qwen3-TTS package (its own CLAUDE.md). Keep its conda env `qwen3-tts` untouched; this fork uses its own isolated venv.

## Verified state (2026-06-01)

This repo is **already installed and benchmarked** on this machine — don't re-run `setup.sh` (it hardcodes a 0.6B download we don't want).

- **Env:** isolated `uv` venv at `./.venv` (Python 3.12.12, `torch 2.12.0+cu130` CUDA-enabled, `transformers 4.57.3`), `faster-qwen3-tts` installed **editable** (edits under `faster_qwen3_tts/` are live). The sibling `qwen3-tts` conda env was untouched. Invoke directly:
  ```bash
  cd /home/ross/workspace/prentora/tts/faster-qwen3-tts
  .venv/bin/python <script.py>        # or: source .venv/bin/activate
  ```
- **Models cached** (shared `~/.cache/huggingface/hub`, ~4.3 GB each, all three 1.7B variants ready — no downloads needed): `Qwen3-TTS-12Hz-1.7B-{Base,CustomVoice,VoiceDesign}`.
- **How it was installed** (manual, to skip the 0.6B pull `setup.sh` forces):
  ```bash
  uv venv .venv --python 3.12
  uv pip install -e . --python .venv/bin/python
  # then snapshot_download of ONLY the 1.7B repos (Base/CustomVoice/VoiceDesign)
  ```
- **Measured 1.7B numbers (RTX 4090, bf16, no flash-attn):** non-streaming RTF ~3.76; streaming RTF ~3.2 with TTFA 264 ms (chunk=4) / 338 ms (chunk=8) / 405 ms (chunk=12); parity/dynamic-cache slow path 0.24 RTF. Peak VRAM ~4.8 GB. Cold-start graph capture ~47 s first run / ~10 s warm, then replayed (per-process, once). Streaming RTF < non-streaming because the codec decoder (~225 ms/12-frame chunk), not the LLM, dominates streaming.
- **Sample outputs** to listen to: `bench_out/sample_{nonstreaming,streaming,custom_voice}_1.7B.wav`.
- **Housekeeping:** the benchmark run left untracked `.idea/` and `faster_qwen3_tts.egg-info/` — safe to add to `.gitignore` or delete. `.venv/` is already gitignored. `CLAUDE.md` is untracked (keep it).

## Setup (only if recreating from scratch)

Requires Python 3.10+, **PyTorch 2.5.1+** with CUDA (capture is unreliable on `torch<=2.5.0`), NVIDIA GPU. 1.7B is ~4 GB VRAM in bf16 (fits the RTX 4090's 24 GB). Prefer the manual install above over `./setup.sh`, which pulls the unwanted 0.6B model and attempts an optional flash-attn install the fast path does not need. Do not disturb the sibling `qwen3-tts` conda env.

## Benchmark

Prefer the scripts directly with `MODEL_SIZE=1.7B` — the `./benchmark.sh` wrapper defaults to running **both** sizes (pulls 0.6B) and only invokes `throughput.py`:

```bash
MODEL_SIZE=1.7B .venv/bin/python benchmarks/streaming.py      # streaming + non-streaming RTF, per-chunk
MODEL_SIZE=1.7B .venv/bin/python benchmarks/custom_voice.py   # CustomVoice path
MODEL_SIZE=1.7B .venv/bin/python benchmarks/compare_modes.py  # mode comparison
MODEL_SIZE=1.7B .venv/bin/python benchmarks/chunk_sweep.py    # TTFA vs chunk_size
MODEL_SIZE=1.7B .venv/bin/python benchmarks/throughput.py     # streaming TTFA + fast-vs-parity RTF
```

Other benchmarks: `baseline.py`, `parakeet_coexistence.py`, `generate_{non_streaming,parity,parity_icl}_samples.py`. Measure on a free GPU — a background process during one run depressed the throughput figure.

## Basic generation

CLI subcommands (`faster-qwen3-tts <cmd>`): `clone`, `custom`, `design`, `serve`. For our use case:

```bash
# Voice-clone (Base model) — male JP/KO from a reference clip
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "..." --language Korean \
  --ref-audio ref.wav --ref-text "<transcript of ref.wav>" \
  --output out.wav

# VoiceDesign (1.7B-VoiceDesign) — male JP/KO via instruct
faster-qwen3-tts design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "A calm middle-aged Korean man with a low, warm voice" \
  --text "..." --language Korean --output out.wav
```

Add `--streaming` to stream chunks. `custom --list-speakers` lists CustomVoice presets (note: no male JP/KO there).

## How it works

Per decode step Qwen3-TTS runs two autoregressive transformers — the **Talker** (28 layers, generates the first codebook token) and the **Code Predictor** (5 layers, generates 15 more codebook tokens). That is ~500 tiny CUDA kernel launches per step; the GPU spends most of its time waiting on Python/launch overhead. The fork eliminates that by:

1. **StaticCache** — fixed-size pre-allocated KV buffers updated in-place, with an explicit attention mask handling variable-length KV within the fixed buffer.
2. **CUDA graph capture** of both the predictor and talker decode steps, replayed as a single GPU op per step (`predictor_graph.py`, `talker_graph.py`).
3. **Vectorized repetition penalty** (`sampling.py`) — removes a per-token CPU↔GPU sync that was a hotspot on fast GPUs. Does not change behavior.
4. **ICL silence-padding fix** — appends 0.5 s of silence to the reference audio before encoding so the first generated token is not conditioned on a trailing phoneme (`append_silence=True` default; set False to match upstream exactly).

Explicitly **no Flash Attention, no vLLM, no Triton, no torch.compile** — the author measured all attention backends as identical RTF (attention is not the bottleneck) and found torch.compile gave zero speedup once CUDA graphs remove launch overhead. The model's own forward pass is used unchanged.

Prefill still uses the normal HF forward (variable length); only the single-token decode steps are graph-replayed.

### Modules under `faster_qwen3_tts/`
- `model.py` — `FasterQwen3TTS` wrapper: loading, warmup/capture, all public `generate_*` methods, prompt construction.
- `generate.py` — `fast_generate`: the non-streaming CUDA-graph decode loop, plus the `parity_mode` branch that calls upstream `talker.generate(...)` (dynamic cache, no graphs) for validation.
- `streaming.py` — streaming decode generator (yields codec-ID chunks every `chunk_size` steps, decoded with a 25-frame left-context sliding window).
- `predictor_graph.py`, `talker_graph.py` — CUDA graph capture/replay for the two transformers.
- `sampling.py` — vectorized repetition penalty + logits sampling.
- `cli.py` — `clone`/`custom`/`design`/`serve` commands.

### Fast path vs parity mode
The fast path (StaticCache + CUDA graphs) is what runs in production. Parity mode is a dynamic-cache, no-graph path used **only in tests** (`tests/test_e2e_parity.py`) to prove token-level equality with upstream; it is intentionally slow (~0.77 s TTFA vs ~0.16 s fast). Streaming and non-streaming share the same decode core.

## Public API (load-once / generate-many)

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",   # or -CustomVoice / -VoiceDesign
    device="cuda", dtype=torch.bfloat16,
    attn_implementation="sdpa", max_seq_len=2048,
)
# model.sample_rate is 24000
```

Generation methods return `([audio_np], sample_rate)`; the `*_streaming` variants are generators yielding `(audio_chunk, sr, timing)`:

- `generate_voice_clone(text, language, ref_audio=None, ref_text="", xvec_only=False, append_silence=True, instruct=None, voice_clone_prompt=None, non_streaming_mode=None, ...)` and `generate_voice_clone_streaming(..., chunk_size=12)`
- `generate_custom_voice(text, speaker, language, instruct=None, ...)` and `_streaming`
- `generate_voice_design(text, instruct, language, ...)` and `_streaming`

`from_pretrained` initializes the graphs; the first generation triggers capture/warmup (`_warmup`). After that the model stays resident and replays. This load-once/keep-hot behavior is exactly what we want for the HTTP service.

### Selecting language + a male voice
- `language` is a string. **10 languages supported** (all variants): Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian — plus `"Auto"` (auto-detect; also the value to omit). Query at runtime with `model.model.get_supported_languages()`.
- **Male JP/KO** — two options:
  - **VoiceDesign:** `generate_voice_design(text, instruct, language)` with an instruct describing a male voice. VoiceDesign is 1.7B-only. Prompting guidance (from upstream): keep `instruct` ~15–40 words, use concrete physical descriptors ("deep", "resonant", "husky"), and **stack dimensions** (gender + age + pace + tone). Set `language` to the target language. **The `instruct` text must be written in English or Chinese only** — per Alibaba's official VoiceDesign docs ("Description text supports Chinese and English only"), the instruct language is independent of the output language: it does NOT need to match it, and for JP/KO/ES/FR/etc. *cannot* be in the output language. Instruct written in an unsupported language (e.g. Japanese/Korean) is effectively ignored. This applies to CustomVoice `instruct` too, not just VoiceDesign.
  - **Base voice-clone:** `generate_voice_clone(text, language, ref_audio, ref_text)` with a clean 10–15 s language-matched male reference clip and its transcript. Default is ICL mode (`xvec_only=False`), which needs accurate `ref_text`; `xvec_only=True` uses only the speaker embedding (shorter prefill, cleaner language switching, no `ref_text`).
- **Voice consistency across many lines:** synthesize one short clip with VoiceDesign, then feed it into the Base model's voice-clone prompt and reuse it via `generate_voice_clone(voice_clone_prompt=...)`. This pins a single designed voice so it doesn't drift call-to-call.
- Precomputed speaker embeddings: extract once (`examples/extract_speaker.py`) and pass via `voice_clone_prompt` to skip per-call prompt extraction.

### CustomVoice built-in speakers (9 presets)
CustomVoice ships **9 fixed speakers** (config IDs are lowercase; pass via `generate_custom_voice(speaker=...)`, validated case-insensitively against `get_supported_speakers()`). Any speaker can speak any of the 10 languages, but each sounds best in its native language. **No built-in male JP/KO voice** — the only JP/KO presets (`ono_anna`, `sohee`) are female, which is the whole reason male JP/KO must go through VoiceDesign or Base clone (above).

| Speaker | Voice description | Native language |
| --- | --- | --- |
| Vivian | Bright, slightly edgy young female voice | Chinese |
| Serena | Warm, gentle young female voice | Chinese |
| Uncle_Fu | Seasoned male voice, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural timbre | Chinese (Beijing dialect) |
| Eric | Lively Chengdu male, slightly husky brightness | Chinese (Sichuan dialect) |
| Ryan | Dynamic male voice, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Ono_Anna | Playful Japanese female, light nimble timbre | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

`eric`/`dylan` carry Chinese dialect flags handled specially in `model.py` (`spk_is_dialect`).

## Serving as an HTTP API

`faster_qwen3_tts/server.py` loads the 1.7B model once, warms the CUDA graphs, and keeps
it resident. Start it with `faster-qwen3-tts serve-http --voices <voices.yaml>` (defaults:
`--host 0.0.0.0 --port 8092`, bundled registry). `GET /health` returns 503 while warming,
`{"status":"failed","error":...}` at 503 if warmup itself threw, and 200 once ready —
never route traffic before 200. Two endpoints, two consumers:

### `POST /v1/audio/speech` — whole-file, for media-worker

`{input, voice, emotion?, response_format:"wav", temperature?}` → one complete 24 kHz
mono 16-bit WAV. The resolved voice, emotion, requested emotion and reference come back
as `X-TTS-Voice`, `X-TTS-Emotion`, `X-TTS-Requested-Emotion` and `X-TTS-Reference`
headers (each omitted, not sent empty, when there is nothing to report) — the body has no
envelope to carry them, and without them an emotion fallback is invisible. Non-streaming:
nothing is emitted until generation finishes, which costs ~1.4 s for a single sentence and
scales with length. Right for dubbing, wrong for a live loop.

### `POST /v1/audio/stream` — framed streaming, for live use

`{input, voice, emotion?, instruct?, temperature?, chunk_size?}` → a stream of
length-prefixed frames, `Content-Type: application/vnd.lyrebird.tts-stream`. First audio
arrives at a fixed cost regardless of input length. Measured end to end over HTTP against
the `ono_anna` custom voice, warm, 5 runs (`spikes/streaming/stream_client.py`):

| `chunk_size` | TTFA | of which model | server + transport |
|---|---|---|---|
| 2 | 290 ms | 144 ms | ~146 ms |
| **4 (default)** | **260–325 ms** | 190 ms | ~135 ms |
| 8 | 410–459 ms | 304 ms | ~127 ms |

`chunk_size` is in codec steps (12 Hz), bounded 1..48, default 4. **The ~335 ms figure in
the spike findings is library-level and excludes ~130 ms of server and transport cost**,
which is roughly constant across chunk sizes; budget against the table above.

The chunk duration doubles as the buffer against a slow chunk. At the default that buffer
is 320 ms, against a worst observed inter-arrival gap of 159 ms — and the margin only
grows from there, ~180 ms per chunk, so the opening chunk is the tightest moment of a
stream. Measured in `spikes/streaming/chunk_margin.py`.

`instruct` is free text describing the delivery, and overrides whatever instruct the
resolved voice/emotion declares. It only does anything on `custom` voices: on the clone
path it moves speaking rate and nothing else, so it is not plumbed there.

**`input` is sanitised before synthesis.** `*action*`, `[action]` and `<action>` are
stripped, because Qwen3-TTS has no markup support and — measured — ignores them where
they are harmless and speaks them aloud where they are not, varying between takes of the
same input. Parentheses are left alone, since an aside is usually speech the caller meant
to keep. An input that is nothing but stage directions is a 400.

Frame layout is `1 byte type | 4 byte big-endian length | payload`:

| Type | Payload |
|---|---|
| `0x01` header | JSON `{sample_rate, channels, format, voice, emotion, requested_emotion, reference}` |
| `0x02` audio | raw s16le PCM |
| `0x03` mark | JSON `{chunk_index, decode_ms, prefill_ms, audio_ms_so_far}` |
| `0x04` error | JSON `{message}` — failure after the response began |
| `0x05` end | JSON `{total_audio_ms, total_decode_ms}` |

`emotion` is what was actually used and `requested_emotion` what was asked for; they
differ when a character's emotion had no recordings and neutral stood in. `reference`
names the recording chosen for this request, and is null for custom voices.

Framing exists because the HTTP status is committed once the first byte is sent, so a
mid-stream failure cannot be a 5xx. **A stream that ends without an end frame is a
failed generation, not a short one** — clients must treat it that way. Errors detected
before streaming begins still use normal status codes (400/503).

Clients cancel by closing the connection, which aborts the decode loop. That matters:
GPU work is serialised on one worker thread, so a cancelled request that kept decoding
would delay the next one. Both `clone` and `custom` voices stream.

### Voice registry

`voices.yaml` accepts two entry shapes. Flat, which the dubbing voices use:

```yaml
  en_m: {type: clone, language: English, ref_audio: refs/en_m.wav, ref_text: "..."}
```

and emotive, where an emotion selects a reference clip:

```yaml
  nicole:
    type: clone
    language: English
    default_emotion: neutral
    emotions:
      neutral: {ref_audio: refs/nicole_neutral.wav, ref_text: "..."}
      amused:  {ref_audio: refs/nicole_amused.wav,  ref_text: "..."}
```

Emotive entries are flattened at load into `"<voice>:<emotion>"` keys, and a clone
prompt is pre-baked per entry at warmup. Each emotion must declare its own `ref_audio`
and `ref_text` — they are not inherited from the voice level, because the clip *is* the
emotion. On a clone voice emotion has to come from the clip, since `instruct` measurably
moves only speaking rate on the ICL clone path, not pitch or energy.

On a `custom` voice it is the other way round: `speaker` inherits from the voice level
(the speaker *is* the voice) and each emotion supplies its own `instruct`.

**Two registries ship.** `voices.yaml` holds media-worker's 12 clone voices and loads
Base only. `voices_lyrebird.yaml` holds the `ono_anna` custom voice and loads CustomVoice
— kept separate so the dubbing deployment does not carry a ~4.8 GB model it never serves.
Select with `serve-http --voices`.

Lyrebird takes the custom route because `instruct` moves pitch and energy there, with no
reference recording competing with it — see `docs/lyrebird-tts-spike-findings.md`
(Experiment 3, Stage B) for the evidence, and for why the designed-voice-with-emotion-clips
route was abandoned. **No shipped voice uses the emotive form**; it works, and is what a
persona would use to declare named handles.

### Character reference library

A third voice source, alongside the two registries: a directory tree where a character is
a directory and an emotion a subdirectory holding interchangeable `.wav`/`.txt` pairs.
Each request picks one pair at random and clones from it. Load it with
`serve-http --characters [PATH]`; bare flag uses the bundled
`faster_qwen3_tts/server_voices/characters`.

```
characters/nicole/neutral/calm_intro.wav
characters/nicole/neutral/calm_intro.txt
characters/nicole/character.yaml          # optional: language, temperature
```

Characters are discovered by directory name — adding one needs no code or config change.
The ten emotion names are fixed in `voice_registry.EMOTIONS`: neutral, amused, smug,
excited, impressed, earnest, deadpan, annoyed, panicked, confused.

**Emotion has to come from the clip** on this path: `instruct` moves only speaking rate on
an ICL clone, so a reference recording is the only thing that moves pitch and energy. That
is measured, not assumed — see `docs/lyrebird-tts-spike-findings.md`.

Three request outcomes, deliberately distinct:

- an emotion outside the ten is a **400**
- one of the ten with no recordings **falls back to neutral**, reported in the response
  and stream header frame so the fallback is never silent
- a character with recordings but none under `neutral` is **skipped at startup** — neutral
  is what everything else falls back to, so without it the character would fail
  unpredictably per request

Malformed content is logged and skipped, never silently dropped: orphan `.wav` or `.txt`,
empty transcript, unreadable audio, a directory whose name is not one of the ten, an
unreadable character directory (e.g. permissions), or a character/emotion/recording name
that cannot be encoded into an HTTP header — `cfg.id` and `reference.id` are echoed into
`X-TTS-*` headers, and an unencodable value would raise `UnicodeEncodeError` while building
the response, surfacing as a bare 500 instead of the real error. The same check applies to
`voices.yaml`, with the opposite policy: a voice id or emotion name there that fails it is
**fatal** at load (`ValueError`), because that file is deliberate config and a typo in it
should not be shrugged off — only filesystem-discovered content gets the skip-and-warn
treatment.

A character's own `character.yaml` carries the same fatal policy for the same reason —
it is deliberate content, not filesystem discovery — and an unknown key there takes down
the **entire** `load_characters()` call, not just that one character, since nothing catches
it before it leaves the loop.

**`--characters` alone does not load `voices.yaml`.** The bundled registry is a default,
not a floor — it loads when neither flag is given, or when `--voices` is given. This keeps
media-worker's deployment from baking clone prompts for a library it never serves, and
keeps the lyrebird deployment from carrying twelve dubbing voices.

**Warmup bakes one clone prompt per recording**, ~101 ms each (measured, RTX 4090, 1.7B
Base, bf16). The shipped library holds 149 recordings, so budget ~15 s on top of the model
load. Pre-baking is deliberate: selection is random, so lazy baking would make early
requests pay ~101 ms on a ~260 ms TTFA budget until the cache happened to fill. A warmup
failure is reported by `/health` as `{"status": "failed", "error": ...}`.

### Per-character audio filters

Full narrative — what was tried, what failed, and how firmly each value is determined —
in [`docs/billy-voice-filter.md`](docs/billy-voice-filter.md). **Read it before changing
any of these numbers**; several are weakly determined and several plausible approaches
were tried and rejected.

A character whose source material is processed needs that processing reapplied, or the
clone sounds like an ordinary person. Billy Kid is a cyborg with a doubling effect on his
in-game voice; his references here are the `GoldenMechaGodBattle` assets, which are the
same performer with the effect *not* applied. He declares the filter in `character.yaml`:

```yaml
filter:
  type: chorus
  cents: [26, -26]      # one entry per copy
  delays_ms: [8, 16]    # one delay per copy, same length as cents
  amount: 0.50          # wet/dry
  window_ms: 42.7       # vocoder window; sets the latency
  makeup_db: 5.09       # restores the level the mix loses
```

**`makeup_db` is not optional polish.** The copies are pitch-shifted, so they are
decorrelated from the dry and from each other: the mix sums as power rather than
amplitude and loses a measured **5.09 dB** even though nothing is attenuated. Without it
Billy is audibly quieter than the unfiltered characters beside him. It is a declared
constant for a *particular* `amount`, not derived from it — **change one and re-measure
the other**. A memoryless soft knee above 0.9 catches the few samples the makeup pushes
past full scale (0.001–0.003%, peaking at 1.16), since the chorus raises crest factor as
well as lowering RMS.

**`window_ms` is a duration, not a sample count, and that distinction is load-bearing.**
The window's audible effect is how much the vocoder smears in time, so a fixed `n_fft`
is a different filter at every sample rate: 2048 samples is 42.7 ms at 48 kHz but 85.3 ms
at 24 kHz. Both were auditioned and heard as different settings -- 42.7 ms was chosen and
85.3 ms rejected as "too far apart" -- and the server shipped the rejected one until this
was expressed in milliseconds.

Discovered like everything else about characters, so adding another processed character
needs no code change. A malformed block is **fatal** at load, like every other
`character.yaml` error -- it is deliberate config, not filesystem discovery.

**Applied after synthesis, never baked into the references.** Baking it in would ask the
model to reproduce a chorus as *timbre*, which it renders inconsistently take to take --
the same failure as cloning from already-processed audio. Downstream it is deterministic
and identical every request.

The chorus is two pitch-shifted copies mixed against the dry through fixed delay lines.
Pitch shifting is phase-vocoder time-stretch then fractional resample, which scales the
whole spectrum including formants so each copy reads as a slightly different voice.
Shifting FFT bins instead approximately preserves formants and audibly sounds flatter, so
the two-stage form is load-bearing. Both stages carry state, so **chunked and whole-file
output are sample identical** and the filter is applied on both endpoints. `flush()` at
end of stream releases the held window; without it the last 85 ms of every line is lost.

**Latency is one window -- 42.7 ms -- but it is mostly not charged to TTFA**,
because a decoded chunk is normally longer than the window, so the filter still emits on
the first chunk. Measured against unfiltered `nicole`, same line, over HTTP:

| `chunk_size` | chunk audio | TTFA penalty |
|---|---|---|
| 1 | 83.3 ms | see below |
| 2 | 166.7 ms | +3 ms |
| **4 (default)** | 333.3 ms | **+10 ms** |
| 8 | 666.7 ms | +18 ms |

Those were measured at the earlier 85.3 ms window, where `chunk_size=1` was a cliff
(+363 ms): an 83.3 ms chunk was just *under* the window, so the first chunk emitted
nothing and TTFA waited for the next. At 42.7 ms every chunk size clears the window, so
the cliff is gone -- but re-measure before relying on `chunk_size=1`. CPU is ~1 ms per
chunk, well under 2% of realtime.

The settings were chosen by ear against the game audio over several rounds. Two copies
beat one; detune and delay were both needed, either alone lost. Detune was
indistinguishable anywhere from 12 to 80 cents and delay only clearly wrong by 22 ms, so
**26 cents and 8/16 ms are weakly determined**; `amount` was the only axis that moved
audibly, and 0.50 is firm. Earlier attempts to derive the filter by measuring dry against
processed game audio failed: the two sets are different takes by the same actor, so the
spectral difference was mostly performance, not effect.

### Pronunciation dictionary and hesitation fillers

Qwen3-TTS has no pronunciation control: no G2P frontend, no lexicon, no phoneme or IPA
input, no SSML. The upstream package has no phonemiser and the tokenizer's 33 added
tokens are all plumbing; text goes verbatim into the chat template. So the only lever on
how a word is said is how it is **spelled**, and it has to change letters — typography
(ALL CAPS, ellipses, em-dash) is measured inert.

`server_voices/pronunciations.yaml` carries two rewrites that both follow from that one
finding, applied to `input` on **both** synthesis endpoints immediately before synthesis.
The first is a flat table of respellings:

```yaml
pronunciations:
  Anby: "Anbee"
  Demara: "Demarra"
```

The second is a list of hesitation fillers, one of which replaces a **medial** ellipsis:

```yaml
fillers: ["uh", "um"]
```

**An ellipsis buys no pause** — that is the same inertness result, and it is the one
upstream complaint with no maintainer answer ([GH discussion #75](https://github.com/QwenLM/Qwen3-TTS/discussions/75),
[HF Base discussion #9](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/discussions/9):
newlines, dots, dashes and underscores all reported to make "no difference"; the two
suggested workarounds are stacked ellipses, which one reporter says *truncates the
audio*, and runs of spaces inside quotes, unverified). What the same spike found does
work is lexical vocalisations — `Haha,` `Ugh,` `Hmm.` were spoken every time. So the
fix spends a word: `I'm the... other thing` → `I'm the, uh, other thing`.

**The delimiting commas are load-bearing and the dashes were not.** Commas move prosody;
hyphens and em-dashes are measured inert, so `- uh -` would be this same word with two
characters that do nothing and might be voiced.

**Only a medial ellipsis is rewritten** — one with a word character either side, and the
surrounding whitespace is swallowed so the replacement supplies its own. A leading or
trailing one keeps its dots: `", uh,"` dangling off the end of a line is worse than the
nothing an ellipsis already does. Since the mark is inert, **every miss is a no-op, not a
regression**, which is why the boundary is deliberately narrow — `Wait...!` and
`the... "other thing"` go unrewritten too. A run of two or more dots, a `…`, or any mix
all match; one bare dot never does, so `3.14` is safe. `\w` is unicode-aware, so
`そう…です` counts as medial for the same reason the respelling boundary avoids `\b`.

The choice is **uniform per occurrence**, from an injectable `random.Random` mirroring
`pick_reference`. Repeat an entry to weight it. Nothing here is language-aware — `uh`/`um`
are English, and a JP/KO deployment should replace the list outright (`えっと`, `어`). Unlike
respelling keys, fillers are emitted and never matched, so they are not restricted to
Latin script.

Respelling runs **before** filling, so an inserted filler can never be caught by a
caller's respelling rule; the reverse is impossible, since a key is Latin letters, spaces,
apostrophes and hyphens and so no replacement can produce an ellipsis.

**`uh` and `um` are unauditioned as a pair**, like `Anbee` and `Demarra` — both are
ordinary English hesitations, but nobody has compared them in these voices.

**Opt-in**, via `serve-http --pronunciations` — bare flag for the bundled table, or a
path. Omitted, nothing is rewritten. Opt-in rather than default so media-worker's dubbing
deployment does not inherit lyrebird's respellings, the same reasoning as `--characters`.
Fillers ride the same flag rather than their own: a dubbed line's `...` is a translator's
punctuation and inserting a filler would rewrite the script and lengthen it against the
original's timing, so the deployment that must not get respellings must not get fillers
either.

Matching is case-insensitive and the replacement is emitted **verbatim**, so `ANBY`
becomes `Anbee` — preserving the caller's capitalisation would be work spent on something
the model ignores. Whole-word, where "word" is bounded by **Latin script** rather than
`\b` or `[A-Za-z]`: `\b` would not match `Anbyさん`, because CJK characters are word
characters, and a bare ASCII class would rewrite inside `Anaïs`. Both matter — this server
serves Japanese and Korean *and* French, Spanish, German, Portuguese and Italian. Digits
are not boundaries (`Anby2` → `Anbee2`); apostrophes are (`Anby's` → `Anbee's`).

Applied in **one pass** over one compiled alternation, so one rule's output can never be
re-matched by another. Keys are `re.escape`d, and the lookup falls back to the matched
text — under `IGNORECASE`, `İ` matches `i` but lowercases to two codepoints, which a bare
dict subscript would turn into a 500.

The pass runs **after** the `MAX_INPUT_CHARS` check on both endpoints, so that bound still
measures what the caller sent; a respelling may push the synthesised text a few characters
over. On the streaming endpoint it also runs after `strip_stage_directions`, so it never
rewrites inside markup that is about to be deleted.

Every malformed entry is **fatal** at load, like `voices.yaml` and unlike the character
library: empty or non-string keys and values, whitespace-padded keys, case-insensitive
collisions, unknown top-level keys, a missing file, a non-list or non-string or
whitespace-padded `fillers` entry. **Non-Latin keys are rejected** rather
than half-supported — the boundary is built from letter lookarounds, so a kana key would
degenerate to a bare substring match and fire inside any longer run. An absent or empty
`pronunciations:` mapping or `fillers:` list is *not* an error; either is how a
deployment turns that half off without dropping the flag.

Keys, replacements and input text are all normalized to **NFC** before matching — at load
for keys and replacements, per-call for `apply`'s `text` argument. Accented Latin can be
encoded two ways (a precomposed codepoint or a base letter plus a combining mark), and
without normalization the two forms just don't compare equal, so a key typed in one form
silently fails to match a caller's text in the other — no error, the word quietly passes
through unrewritten, the worst failure mode for a feature whose only job is fixing
pronunciation. Normalizing both sides to the same form means an accented key matches
regardless of which composition the caller's client happened to send. The zero-entries
guard in `apply` runs first and is unaffected: with no dictionary loaded, the caller's
text is returned untouched, not even normalized.

**`Erridoo` and `Reedoo` are auditioned and correct.** New Eridu is named for the
Sumerian city, /ˈɛrɪduː/ — EH-ri-doo — and `Ridu` is the in-game short form, REE-doo. The
doubled `r` shortens the first vowel and pulls the stress forward, against an English
reader's pull toward eh-RID-yoo or ee-RYE-doo; the doubled `ee` blocks RYE-doo, the
default English reading of an open syllable. `Ridu` cannot fire inside `Eridu` — the word
boundary is a letter lookbehind, so the `E` blocks it, and a test pins this because a key
added later without that boundary would break it silently.

**`Anbee` and `Demarra` are still unauditioned**, first-principles guesses at AN-bee and
de-MAH-ra (Japanese アンビー・デマラ, Chinese 安比・德玛拉). Nobody has heard the model say
either, and no test can check it — if it already reads `Anby` correctly, they make it
worse. Settling them means an audition in the manner of the Billy filter, varying the
reference draw as well as the voice, since two shipped transcripts
(`anby/excited/Galgame_Chapter0_Anbi_05.txt`, `nicole/annoyed/GalGame_Chapter030_Nicole_020_014.txt`)
contain the name and pair it with audio of the actor saying it correctly.

### `GET /v1/voices`

Lists every voice, in the three shapes a client must tell apart. Not gated on warmup — it
reads the registry and needs no GPU, so a UI can populate during startup.

- **character** — all ten emotions with recording counts, `0` meaning it falls back
- **emotive configured voice** — only its own declared emotion names
- **flat voice** — `emotions: null`

`emotion_fallback` distinguishes "falls back to neutral" from "cannot be requested". Counts
are `null` for custom entries, which involve no recordings.

### Audition page

`GET /`, mounted **only when `--characters` is given**, so it never appears on the dubbing
deployment. Self-contained HTML with no external requests. Pick a character and emotion,
type a line, and it streams over `/v1/audio/stream` and plays through WebAudio while
showing TTFA, per-chunk decode time, inter-arrival gap and **playback margin** — delivered
audio minus elapsed time, the quantity that decides whether playback starves. A chunk that
arrives too late to schedule cleanly is flagged as an underrun.

## Gotchas

- **Fixed max-length / StaticCache:** `max_seq_len` (default 2048) bounds the KV buffer; generation stops near that limit. Keep it >= the longest expected utterance in tokens. CUDA graphs capture a specific prefill length on first run.
- **Cold start / warmup:** the first generation captures the graphs and is slow; the service must warm up at startup before serving traffic. Don't benchmark or measure latency on the first call.
- **1.7B only** for this trial — ignore 0.6B benchmarks/results.
- **Quality parity:** outputs are algorithmically equivalent to upstream but not bit-identical. If validating quality, compare against the upstream `../Qwen3-TTS` package or use the parity-mode tests; don't treat divergence from upstream waveforms as a bug.
- **No flash-attn needed:** `attn_implementation="sdpa"` is the expected setting; flash-attn gives no speedup here.
