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

- `generate_voice_clone(text, language, ref_audio=None, ref_text="", xvec_only=False, append_silence=True, instruct=None, voice_clone_prompt=None, non_streaming_mode=None, ...)` and `generate_voice_clone_streaming(..., chunk_size=8)`
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

## Serving as an HTTP API (forward-looking; server not built yet)

Goal: a long-running service that loads the 1.7B model **once**, warms up the CUDA graphs, and keeps it resident on the GPU, exposing an **OpenAI-compatible `POST /v1/audio/speech`** endpoint (`{input, lang_code/voice, response_format:"wav"}`) — matching how media-worker currently calls MeloTTS. media-worker will add a Java `QwenTTSService` and repoint Korean to it. 24 kHz WAV output is fine (media-worker normalizes to 48 kHz). Calls are effectively serialized, so single-stream latency/RTF is what matters.

`examples/openai_server.py` is a reference FastAPI implementation of this contract (`pip install "faster-qwen3-tts[demo]"`), but note it is **voice-clone only** (`generate_voice_clone` / `_streaming`) and routes the request `voice` field through a `--voices voices.json` map of `{ref_audio, ref_text, language}`. To serve male JP/KO via VoiceDesign instead of clone, the server would need extending to call `generate_voice_design`.

## Gotchas

- **Fixed max-length / StaticCache:** `max_seq_len` (default 2048) bounds the KV buffer; generation stops near that limit. Keep it >= the longest expected utterance in tokens. CUDA graphs capture a specific prefill length on first run.
- **Cold start / warmup:** the first generation captures the graphs and is slow; the service must warm up at startup before serving traffic. Don't benchmark or measure latency on the first call.
- **1.7B only** for this trial — ignore 0.6B benchmarks/results.
- **Quality parity:** outputs are algorithmically equivalent to upstream but not bit-identical. If validating quality, compare against the upstream `../Qwen3-TTS` package or use the parity-mode tests; don't treat divergence from upstream waveforms as a bug.
- **No flash-attn needed:** `attn_implementation="sdpa"` is the expected setting; flash-attn gives no speedup here.
