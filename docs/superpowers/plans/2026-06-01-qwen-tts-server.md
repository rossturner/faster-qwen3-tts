# Qwen3-TTS HTTP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a packaged, tested FastAPI server that serves the 14 selected Qwen3-TTS voices over an OpenAI-compatible `POST /v1/audio/speech` endpoint, shipped as a GPU Docker image for media-worker.

**Architecture:** A single-process FastAPI app loads CustomVoice + Base 1.7B models once, pre-bakes clone prompts for the 10 cloned voices from versioned reference clips, warms the CUDA graphs in a background task, and serves complete 24 kHz WAV bytes. All GPU work runs on one dedicated worker thread (serialized). A YAML voice registry maps the 14 voice ids to model/speaker/clone-prompt config.

**Tech Stack:** Python 3.12, FastAPI + uvicorn, PyYAML, `faster_qwen3_tts` (CUDA-graph fork), pytest + httpx (mocked model), Docker (CUDA runtime + mounted HF cache volume).

**Spec:** `docs/superpowers/specs/2026-06-01-qwen-tts-server-design.md`

**Branch:** `qwen-tts-server`. Run everything via the repo venv: `.venv/bin/python`, `.venv/bin/pytest`.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `spikes/two_model_coexistence.py` | Task 0 — validate CustomVoice + Base coexist on one GPU |
| `faster_qwen3_tts/server_voices/voices.yaml` | The 14-voice registry data |
| `faster_qwen3_tts/server_voices/refs/*.wav` | 12 versioned clone-source clips |
| `faster_qwen3_tts/server_voices/README.md` | Provenance for the ref clips |
| `scripts/make_builtin_clone_refs.py` | Generate `aiden_ref.wav` / `sohee_ref.wav` |
| `faster_qwen3_tts/wav_io.py` | float32 → 16-bit WAV encoding |
| `faster_qwen3_tts/voice_registry.py` | Load/resolve/validate the voice registry |
| `faster_qwen3_tts/server.py` | FastAPI app + `ModelManager` (load/warmup/synthesize) |
| `faster_qwen3_tts/cli.py` | Add `serve-http` subcommand (modify) |
| `pyproject.toml` | `server` extra + package-data for `server_voices/` (modify) |
| `.gitignore` | Negations so `server_voices/` wav/yaml are tracked (modify) |
| `Dockerfile`, `.dockerignore`, `docker-compose.example.yml` | The shipped image |
| `tests/test_wav_io.py`, `tests/test_voice_registry.py`, `tests/test_server.py` | Tests (no GPU) |
| `docs/media-worker-qwen-integration.md` | Java handoff doc |

---

### Task 0: Two-model coexistence spike

**Goal:** Prove CustomVoice + Base load and generate together on the one GPU before building anything else; record the decision. If it fails, the spec's Base-only fallback applies and later tasks adjust.

**Files:**
- Create: `spikes/two_model_coexistence.py`
- Create: `docs/superpowers/specs/2026-06-01-coexistence-spike-result.md`

**Acceptance Criteria:**
- [ ] Both models load, generate one clip each, and peak VRAM is reported.
- [ ] Result (pass/fail + peak VRAM) recorded in the spike-result doc.

**Verify:** `.venv/bin/python spikes/two_model_coexistence.py` → prints both durations + peak VRAM, exits 0.

**Steps:**

- [ ] **Step 1: Write the spike script**

```python
# spikes/two_model_coexistence.py
"""Spike: confirm CustomVoice + Base coexist (loaded + generating) on one GPU."""
import torch
from faster_qwen3_tts import FasterQwen3TTS

def load(model_id):
    return FasterQwen3TTS.from_pretrained(
        model_id, device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048,
    )

print("Loading CustomVoice + Base together...")
custom = load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
base = load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

print("Generating on CustomVoice...")
c_audio, sr = custom.generate_custom_voice(
    text="This is a coexistence test.", speaker="aiden",
    language="English", temperature=0.7)
print(f"  custom: {len(c_audio[0])/sr:.2f}s")

print("Generating on Base (clone)...")
b_audio, sr = base.generate_voice_clone(
    text="This is a coexistence test.", language="English",
    ref_audio="voice_design_audition/en_f_v1_ref.wav",
    ref_text="Welcome to the course. Let's get started with today's lesson.",
    xvec_only=False, temperature=0.7)
print(f"  base:   {len(b_audio[0])/sr:.2f}s")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\nPASS — both models coexisted. Peak VRAM allocated: {peak_gb:.2f} GB")
```

- [ ] **Step 2: Run the spike**

Run: `.venv/bin/python spikes/two_model_coexistence.py`
Expected: two durations printed, then `PASS — both models coexisted. Peak VRAM allocated: <N> GB` (expect ~8–12 GB, well under 24).

- [ ] **Step 3: Record the result**

Write `docs/superpowers/specs/2026-06-01-coexistence-spike-result.md` with: outcome (PASS/FAIL), peak VRAM, and a one-line decision: "Proceed with both-models design" or "Fall back to Base-only (clone aiden/sohee; drop CustomVoice loading in Task 5; remove `type: custom` entries from registry, reusing `en_m_clone`/`ko_f_clone`)."

- [ ] **Step 4: Commit**

```bash
git add spikes/two_model_coexistence.py docs/superpowers/specs/2026-06-01-coexistence-spike-result.md
git commit -m "spike: validate two-model coexistence on one GPU"
```

---

### Task 1: Versioned voice registry data + reference clips + provenance

**Goal:** Assemble the source-controlled `server_voices/` directory: the 12 ref clips, the `voices.yaml` registry, and the provenance README; commit the generation scripts.

**Files:**
- Create: `faster_qwen3_tts/server_voices/voices.yaml`
- Create: `faster_qwen3_tts/server_voices/refs/*.wav` (12 files)
- Create: `faster_qwen3_tts/server_voices/README.md`
- Create: `scripts/make_builtin_clone_refs.py`
- Modify: `.gitignore`
- Commit at repo root (do NOT move — they derive the project root from `__file__`):
  `gen_voice_samples.py`, `design_audition_library.py`, `redo_ja_female.py`, `generate_audition_html.py`

**Acceptance Criteria:**
- [ ] `refs/` contains the 10 designed `*_ref.wav` + `aiden_ref.wav` + `sohee_ref.wav`.
- [ ] `voices.yaml` defines all 14 voice ids with valid `type`/fields.
- [ ] `git status` shows the wavs + yaml as tracked (not ignored).
- [ ] README documents per-clip provenance.

**Verify:** `.venv/bin/python -c "import yaml,glob; d=yaml.safe_load(open('faster_qwen3_tts/server_voices/voices.yaml')); assert len(d['voices'])==14; print(sorted(d['voices']))"` → lists 14 ids; `git check-ignore faster_qwen3_tts/server_voices/refs/en_f_v1_ref.wav` → exits non-zero (not ignored).

**Steps:**

- [ ] **Step 1: Add .gitignore negations**

Append to `.gitignore`:
```
# Versioned production voice assets (override the *.wav / *.json globals above)
!faster_qwen3_tts/server_voices/
!faster_qwen3_tts/server_voices/refs/
!faster_qwen3_tts/server_voices/refs/*.wav
!faster_qwen3_tts/server_voices/*.yaml
```

- [ ] **Step 2: Assemble the 10 designed ref clips**

```bash
mkdir -p faster_qwen3_tts/server_voices/refs scripts
cp voice_design_audition/en_f_v1_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/es_m_v2_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/es_f_v3_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/fr_m_v1_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/fr_f_v3_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/zh_m_v3_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/zh_f_v3_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/ja_m_v1_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/ja_f_v3_ref.wav faster_qwen3_tts/server_voices/refs/
cp voice_design_audition/ko_m_v3_ref.wav faster_qwen3_tts/server_voices/refs/
```

- [ ] **Step 3: Write the builtin-clone-ref generator**

```python
# scripts/make_builtin_clone_refs.py
"""Render aiden/sohee (CustomVoice presets) speaking the per-language reference
line, to use as the clone source for the en_m_clone / ko_f_clone voices."""
import os, torch, soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

OUT = "faster_qwen3_tts/server_voices/refs"
INSTRUCT = "Speak in a calm, clear, professional tone suitable for an instructional video."
JOBS = [
    ("aiden", "English", "Welcome to the course. Let's get started with today's lesson.", "aiden_ref.wav"),
    ("sohee", "Korean", "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다.", "sohee_ref.wav"),
]
m = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda",
    dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048)
m.generate_custom_voice(text="Warmup.", speaker="aiden", language="English", max_new_tokens=20)
for speaker, lang, text, fname in JOBS:
    wavs, sr = m.generate_custom_voice(text=text, speaker=speaker, language=lang,
                                       instruct=INSTRUCT, temperature=0.7)
    sf.write(os.path.join(OUT, fname), wavs[0], sr)
    print(f"wrote {fname}: {len(wavs[0])/sr:.2f}s")
```

- [ ] **Step 4: Generate the two builtin-clone refs**

Run: `.venv/bin/python scripts/make_builtin_clone_refs.py`
Expected: `wrote aiden_ref.wav: ~Xs` and `wrote sohee_ref.wav: ~Xs`; both files exist in `refs/`.

- [ ] **Step 5: Write voices.yaml**

```yaml
# faster_qwen3_tts/server_voices/voices.yaml
sample_rate: 24000
default_temperature: 0.7
voices:
  en_m: {type: custom, speaker: aiden, language: English, instruct: "Speak in a calm, clear, professional tone suitable for an instructional video."}
  ko_f: {type: custom, speaker: sohee, language: Korean,  instruct: "Speak in a calm, clear, professional tone suitable for an instructional video."}
  en_f: {type: clone, language: English,  ref_audio: refs/en_f_v1_ref.wav, ref_text: "Welcome to the course. Let's get started with today's lesson."}
  es_m: {type: clone, language: Spanish,  ref_audio: refs/es_m_v2_ref.wav, ref_text: "Bienvenido al curso. Vamos a empezar con la lección de hoy."}
  es_f: {type: clone, language: Spanish,  ref_audio: refs/es_f_v3_ref.wav, ref_text: "Bienvenido al curso. Vamos a empezar con la lección de hoy."}
  fr_m: {type: clone, language: French,   ref_audio: refs/fr_m_v1_ref.wav, ref_text: "Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui."}
  fr_f: {type: clone, language: French,   ref_audio: refs/fr_f_v3_ref.wav, ref_text: "Bienvenue dans ce cours. Commençons la leçon d'aujourd'hui."}
  zh_m: {type: clone, language: Chinese,  ref_audio: refs/zh_m_v3_ref.wav, ref_text: "欢迎来到本课程。让我们开始今天的课程吧。"}
  zh_f: {type: clone, language: Chinese,  ref_audio: refs/zh_f_v3_ref.wav, ref_text: "欢迎来到本课程。让我们开始今天的课程吧。"}
  ja_m: {type: clone, language: Japanese, ref_audio: refs/ja_m_v1_ref.wav, ref_text: "このコースへようこそ。今日のレッスンを始めましょう。"}
  ja_f: {type: clone, language: Japanese, ref_audio: refs/ja_f_v3_ref.wav, ref_text: "このコースへようこそ。今日のレッスンを始めましょう。"}
  ko_m: {type: clone, language: Korean,   ref_audio: refs/ko_m_v3_ref.wav, ref_text: "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다."}
  en_m_clone: {type: clone, language: English, ref_audio: refs/aiden_ref.wav, ref_text: "Welcome to the course. Let's get started with today's lesson."}
  ko_f_clone: {type: clone, language: Korean,  ref_audio: refs/sohee_ref.wav, ref_text: "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다."}
```

- [ ] **Step 6: Write the provenance README**

Create `faster_qwen3_tts/server_voices/README.md` documenting: the binaries are canonical (generation is stochastic, no seed — scripts reproduce the *method* not the clip); per-voice source (designed-then-cloned vs CustomVoice render), the exact instruct/persona prompt and reference line (cross-link `../../voice-mapping.md` and `../../scripts/`), and `temperature=0.7`. Map each ref file → its generating script (`scripts/design_audition_library.py`, `scripts/redo_ja_female.py` for `ja_f_v3`, `scripts/make_builtin_clone_refs.py` for `aiden_ref`/`sohee_ref`).

- [ ] **Step 7: Commit registry, refs, provenance, and generation scripts**

The existing generation scripts stay at repo root (they resolve the project root from `__file__`).
Only the new `make_builtin_clone_refs.py` lives under `scripts/`.
```bash
git add -f faster_qwen3_tts/server_voices/ scripts/make_builtin_clone_refs.py .gitignore voice-mapping.md \
  gen_voice_samples.py design_audition_library.py redo_ja_female.py generate_audition_html.py
git commit -m "feat: versioned voice registry, ref clips, provenance + generation scripts"
```

---

### Task 2: WAV encoding module

**Goal:** A small, GPU-free module that turns a float32 mono array into complete 16-bit PCM WAV bytes with a correct header.

**Files:**
- Create: `faster_qwen3_tts/wav_io.py`
- Test: `tests/test_wav_io.py`

**Acceptance Criteria:**
- [ ] `to_wav_bytes(pcm, sr)` returns bytes parseable by `wave`/`soundfile` with matching sample rate, mono, 16-bit, and correct frame count.

**Verify:** `.venv/bin/pytest tests/test_wav_io.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wav_io.py
import io, wave
import numpy as np
from faster_qwen3_tts.wav_io import to_wav_bytes

def test_wav_roundtrip_header_and_frames():
    sr = 24000
    pcm = np.sin(np.linspace(0, 3.14, sr)).astype(np.float32)  # 1.0s
    data = to_wav_bytes(pcm, sr)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == sr
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == sr  # 1 second

def test_wav_clips_out_of_range():
    data = to_wav_bytes(np.array([2.0, -2.0], dtype=np.float32), 24000)
    with wave.open(io.BytesIO(data)) as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert frames[0] == 32767 and frames[1] == -32768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wav_io.py -v`
Expected: FAIL — `ModuleNotFoundError: faster_qwen3_tts.wav_io`.

- [ ] **Step 3: Write the implementation**

```python
# faster_qwen3_tts/wav_io.py
"""Encode float32 mono audio to complete 16-bit PCM WAV bytes."""
import io, struct
import numpy as np

def _to_pcm16(pcm: np.ndarray) -> bytes:
    return np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()

def to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    raw = _to_pcm16(np.asarray(pcm, dtype=np.float32).flatten())
    n_channels, bits = 1, 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                                    byte_rate, block_align, bits)
    header += b"data" + struct.pack("<I", len(raw))
    return header + raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_wav_io.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/wav_io.py tests/test_wav_io.py
git commit -m "feat: WAV encoding helper with tests"
```

---

### Task 3: Voice registry loader

**Goal:** Load `voices.yaml` into validated `VoiceConfig` objects, resolving `ref_audio` to absolute paths and rejecting malformed/unknown voices. GPU-free.

**Files:**
- Create: `faster_qwen3_tts/voice_registry.py`
- Test: `tests/test_voice_registry.py`

**Acceptance Criteria:**
- [ ] Loads the real `voices.yaml`, exposes all 14 ids.
- [ ] `resolve("zz")` raises `KeyError`; missing required fields raise `ValueError`.
- [ ] `clone` ref paths resolve relative to the yaml's directory; `custom` requires `speaker`.

**Verify:** `.venv/bin/pytest tests/test_voice_registry.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice_registry.py
import pytest
from pathlib import Path
from faster_qwen3_tts.voice_registry import load_registry, VoiceConfig

REAL = Path("faster_qwen3_tts/server_voices/voices.yaml")

def test_loads_all_14_real_voices():
    reg = load_registry(REAL)
    assert len(reg.voices) == 14
    assert reg.resolve("en_m").type == "custom"
    assert reg.resolve("en_m").speaker == "aiden"
    cfg = reg.resolve("ja_f")
    assert cfg.type == "clone"
    assert cfg.ref_audio.is_absolute() and cfg.ref_audio.exists()
    assert cfg.temperature == 0.7

def test_unknown_voice_raises_keyerror():
    reg = load_registry(REAL)
    with pytest.raises(KeyError):
        reg.resolve("nope")

def test_clone_missing_ref_text_raises(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("sample_rate: 24000\ndefault_temperature: 0.7\n"
                 "voices:\n  x: {type: clone, language: English, ref_audio: r.wav}\n")
    with pytest.raises(ValueError):
        load_registry(p)

def test_custom_missing_speaker_raises(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("sample_rate: 24000\ndefault_temperature: 0.7\n"
                 "voices:\n  x: {type: custom, language: English}\n")
    with pytest.raises(ValueError):
        load_registry(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_voice_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: faster_qwen3_tts.voice_registry`.

- [ ] **Step 3: Write the implementation**

```python
# faster_qwen3_tts/voice_registry.py
"""Load and validate the server voice registry (voices.yaml)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import yaml

@dataclass(frozen=True)
class VoiceConfig:
    id: str
    type: str                      # "custom" | "clone"
    language: str
    temperature: float
    speaker: Optional[str] = None        # custom
    instruct: Optional[str] = None       # custom
    ref_audio: Optional[Path] = None     # clone (absolute)
    ref_text: Optional[str] = None       # clone

@dataclass(frozen=True)
class Registry:
    sample_rate: int
    voices: Dict[str, VoiceConfig]
    def resolve(self, voice_id: str) -> VoiceConfig:
        if voice_id not in self.voices:
            raise KeyError(f"Unknown voice id {voice_id!r}. Known: {sorted(self.voices)}")
        return self.voices[voice_id]

def load_registry(path) -> Registry:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    base_dir = path.parent
    default_temp = float(data.get("default_temperature", 0.7))
    voices: Dict[str, VoiceConfig] = {}
    for vid, raw in data["voices"].items():
        vtype = raw.get("type")
        if vtype not in ("custom", "clone"):
            raise ValueError(f"{vid}: type must be custom|clone, got {vtype!r}")
        lang = raw.get("language")
        if not lang:
            raise ValueError(f"{vid}: language is required")
        temp = float(raw.get("temperature", default_temp))
        if vtype == "custom":
            if not raw.get("speaker"):
                raise ValueError(f"{vid}: custom voice requires 'speaker'")
            voices[vid] = VoiceConfig(vid, "custom", lang, temp,
                                      speaker=raw["speaker"], instruct=raw.get("instruct"))
        else:
            if not raw.get("ref_audio") or not raw.get("ref_text"):
                raise ValueError(f"{vid}: clone voice requires 'ref_audio' and 'ref_text'")
            ref = (base_dir / raw["ref_audio"]).resolve()
            voices[vid] = VoiceConfig(vid, "clone", lang, temp,
                                      ref_audio=ref, ref_text=raw["ref_text"])
    return Registry(int(data.get("sample_rate", 24000)), voices)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_voice_registry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/voice_registry.py tests/test_voice_registry.py
git commit -m "feat: voice registry loader with validation + tests"
```

---

### Task 4: FastAPI server (model manager + endpoints)

**Goal:** The server module: a `ModelManager` that loads both models / pre-bakes clone prompts / warms up / synthesizes on a single GPU thread, and a FastAPI app with `GET /health` (warmup-gated) and `POST /v1/audio/speech` (validated, wav-only, complete WAV). Tested with a mocked manager (no GPU).

**Files:**
- Create: `faster_qwen3_tts/server.py`
- Test: `tests/test_server.py`

**Acceptance Criteria:**
- [ ] `build_app(manager, registry)` returns a FastAPI app.
- [ ] `/health` → 503 when `manager.ready` is False, 200 when True.
- [ ] `POST /v1/audio/speech`: empty `input` → 400; unknown `voice` → 400; non-`wav` `response_format` → 400; valid → 200 `audio/wav` with a real WAV body.
- [ ] `ModelManager.synthesize` routes `custom`→`generate_custom_voice`, `clone`→`generate_voice_clone(voice_clone_prompt=…)`.

**Verify:** `.venv/bin/pytest tests/test_server.py -v` → all pass (no GPU/model load).

**Steps:**

- [ ] **Step 1: Write the failing test (mocked manager)**

```python
# tests/test_server.py
import io, wave
import numpy as np
from fastapi.testclient import TestClient
from faster_qwen3_tts.server import build_app
from faster_qwen3_tts.voice_registry import Registry, VoiceConfig

def _registry():
    return Registry(24000, {
        "en_m": VoiceConfig("en_m", "custom", "English", 0.7, speaker="aiden"),
    })

class FakeManager:
    sample_rate = 24000
    def __init__(self): self.ready = True; self.calls = []
    def synthesize(self, cfg, text, temperature, max_new_tokens=None):
        self.calls.append((cfg.id, text))
        return np.zeros(24000, dtype=np.float32)  # 1s silence

def client(manager=None):
    mgr = manager or FakeManager()
    return TestClient(build_app(mgr, _registry())), mgr

def test_health_ready():
    c, _ = client()
    assert c.get("/health").status_code == 200

def test_health_warming():
    mgr = FakeManager(); mgr.ready = False
    c, _ = client(mgr)
    r = c.get("/health")
    assert r.status_code == 503

def test_speech_ok_returns_wav():
    c, mgr = client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "en_m"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 24000
    assert mgr.calls == [("en_m", "hi")]

def test_speech_empty_input_400():
    c, _ = client()
    assert c.post("/v1/audio/speech", json={"input": "  ", "voice": "en_m"}).status_code == 400

def test_speech_unknown_voice_400():
    c, _ = client()
    assert c.post("/v1/audio/speech", json={"input": "hi", "voice": "zz"}).status_code == 400

def test_speech_bad_format_400():
    c, _ = client()
    assert c.post("/v1/audio/speech",
                  json={"input": "hi", "voice": "en_m", "response_format": "mp3"}).status_code == 400

def test_speech_503_when_warming():
    mgr = FakeManager(); mgr.ready = False
    c, _ = client(mgr)
    assert c.post("/v1/audio/speech", json={"input": "hi", "voice": "en_m"}).status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: faster_qwen3_tts.server`.

- [ ] **Step 3: Write the server implementation**

```python
# faster_qwen3_tts/server.py
"""OpenAI-compatible Qwen3-TTS server: 14 voices, dual-model, warmup-gated health."""
from __future__ import annotations
import asyncio, logging, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .voice_registry import Registry, VoiceConfig, load_registry
from .wav_io import to_wav_bytes

logger = logging.getLogger(__name__)

DEFAULT_VOICES = Path(__file__).parent / "server_voices" / "voices.yaml"
MAX_INPUT_CHARS = 2000          # guard against over-long input (model bounded by max_seq_len)
DEFAULT_MAX_NEW_TOKENS = 1024   # ~85s ceiling; prevents runaway decode tying up the GPU


class SpeechRequest(BaseModel):
    input: str
    voice: str
    response_format: str = "wav"
    model: str = "qwen3-tts"
    speed: float = 1.0
    temperature: Optional[float] = None


class ModelManager:
    """Owns both models; serializes all GPU work on one worker thread."""

    def __init__(self, registry: Registry, device="cuda", max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        self.registry = registry
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sample_rate = registry.sample_rate
        self.ready = False
        self._custom = None
        self._base = None
        self._clone_prompts: dict = {}
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpu")

    def _run(self, fn, *a, **k):
        return self._gpu.submit(fn, *a, **k).result()

    def _load_and_warm(self):
        import torch
        from faster_qwen3_tts import FasterQwen3TTS
        needs_custom = any(v.type == "custom" for v in self.registry.voices.values())
        if needs_custom:
            logger.info("Loading CustomVoice...")
            self._custom = FasterQwen3TTS.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device=self.device,
                dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048)
        logger.info("Loading Base...")
        self._base = FasterQwen3TTS.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device=self.device,
            dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048)
        # Pre-bake clone prompts for every clone voice.
        for vid, cfg in self.registry.voices.items():
            if cfg.type == "clone":
                self._clone_prompts[vid] = self._base.model.create_voice_clone_prompt(
                    ref_audio=str(cfg.ref_audio), ref_text=cfg.ref_text, x_vector_only_mode=False)
        # Warm up CUDA graphs on each model.
        if self._custom is not None:
            cv = next(v for v in self.registry.voices.values() if v.type == "custom")
            self._custom.generate_custom_voice(text="Warmup.", speaker=cv.speaker,
                                                language=cv.language, max_new_tokens=20)
        any_clone = next((v for v in self.registry.voices.values() if v.type == "clone"), None)
        if any_clone is not None:
            self._base.generate_voice_clone(
                text="Warmup.", language=any_clone.language,
                voice_clone_prompt=self._clone_prompts[any_clone.id],
                ref_text=any_clone.ref_text, max_new_tokens=20)
        self.ready = True
        logger.info("Warmup complete — server ready.")

    def start_warmup_background(self):
        threading.Thread(target=lambda: self._run(self._load_and_warm),
                         name="tts-warmup", daemon=True).start()

    def _synthesize_blocking(self, cfg: VoiceConfig, text: str, temperature: float):
        if cfg.type == "custom":
            wavs, _ = self._custom.generate_custom_voice(
                text=text, speaker=cfg.speaker, language=cfg.language,
                instruct=cfg.instruct, temperature=temperature,
                max_new_tokens=self.max_new_tokens)
        else:
            wavs, _ = self._base.generate_voice_clone(
                text=text, language=cfg.language,
                voice_clone_prompt=self._clone_prompts[cfg.id], ref_text=cfg.ref_text,
                xvec_only=False, temperature=temperature, max_new_tokens=self.max_new_tokens)
        return np.asarray(wavs[0], dtype=np.float32)

    def synthesize(self, cfg: VoiceConfig, text: str, temperature: float, max_new_tokens=None):
        return self._run(self._synthesize_blocking, cfg, text, temperature)


def build_app(manager, registry: Registry) -> FastAPI:
    app = FastAPI(title="faster-qwen3-tts server")

    @app.get("/health")
    async def health():
        if not manager.ready:
            return JSONResponse({"status": "warming"}, status_code=503)
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        if not manager.ready:
            raise HTTPException(503, "Model warming up")
        text = req.input.strip()
        if not text:
            raise HTTPException(400, "'input' is empty")
        if len(text) > MAX_INPUT_CHARS:
            raise HTTPException(400, f"'input' exceeds {MAX_INPUT_CHARS} chars; chunk upstream")
        if req.response_format.lower() != "wav":
            raise HTTPException(400, "only response_format='wav' is supported")
        try:
            cfg = registry.resolve(req.voice)
        except KeyError as e:
            raise HTTPException(400, str(e))
        temp = req.temperature if req.temperature is not None else cfg.temperature
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, manager.synthesize, cfg, text, temp)
        return Response(content=to_wav_bytes(pcm, manager.sample_rate), media_type="audio/wav")

    return app


def create_app(voices_path=DEFAULT_VOICES, device="cuda", max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
               warmup=True) -> FastAPI:
    registry = load_registry(voices_path)
    manager = ModelManager(registry, device=device, max_new_tokens=max_new_tokens)
    if warmup:
        manager.start_warmup_background()
    return build_app(manager, registry)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_server.py -v`
Expected: PASS (7 tests). (Install test deps first if needed: `.venv/bin/pip install pyyaml httpx`.)

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/server.py tests/test_server.py
git commit -m "feat: FastAPI server with model manager, warmup-gated health, mocked tests"
```

---

### Task 5: `serve-http` CLI command + packaging

**Goal:** Expose the server via `faster-qwen3-tts serve-http`, and package the `server_voices/` data + a `server` extra so the wheel/image ship the registry and refs.

**Files:**
- Modify: `faster_qwen3_tts/cli.py` (add subparser + `cmd_serve_http`)
- Modify: `pyproject.toml` (package-data + `server` optional-deps)

**Acceptance Criteria:**
- [ ] `faster-qwen3-tts serve-http --help` lists `--host/--port/--voices/--device/--max-new-tokens`.
- [ ] `pip wheel` includes `server_voices/voices.yaml` and `refs/*.wav` (package-data configured).

**Verify:** `.venv/bin/faster-qwen3-tts serve-http --help` → shows the options and exits 0.

**Steps:**

- [ ] **Step 1: Add the CLI command**

In `faster_qwen3_tts/cli.py`, add this function near the other `cmd_*`:
```python
def cmd_serve_http(args):
    import uvicorn
    from faster_qwen3_tts.server import create_app, DEFAULT_VOICES
    app = create_app(voices_path=args.voices or DEFAULT_VOICES, device=args.device,
                     max_new_tokens=args.max_new_tokens, warmup=True)
    uvicorn.run(app, host=args.host, port=args.port)
```
And register the subparser inside `build_parser()` (next to the existing `serve` block, before `return p`):
```python
    sp = sub.add_parser("serve-http", help="OpenAI-compatible HTTP server for the 14 production voices")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8092)
    sp.add_argument("--voices", default=None, help="Path to voices.yaml (default: bundled)")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--max-new-tokens", type=int, default=1024)
    sp.set_defaults(fn=cmd_serve_http)
```

- [ ] **Step 2: Add packaging for the voice assets**

In `pyproject.toml`, add a `server` extra and package-data:
```toml
[project.optional-dependencies]
demo = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.24.0",
    "python-multipart>=0.0.7",
]
server = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.24.0",
    "pyyaml>=6.0",
]

[tool.setuptools.package-data]
faster_qwen3_tts = ["server_voices/*.yaml", "server_voices/*.md", "server_voices/refs/*.wav"]
```

- [ ] **Step 3: Reinstall editable + verify the command**

Run: `.venv/bin/pip install -e ".[server]" && .venv/bin/faster-qwen3-tts serve-http --help`
Expected: usage text listing `--host --port --voices --device --max-new-tokens`, exit 0.

- [ ] **Step 4: Verify the full test suite still passes**

Run: `.venv/bin/pytest tests/test_wav_io.py tests/test_voice_registry.py tests/test_server.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/cli.py pyproject.toml
git commit -m "feat: serve-http CLI command + package server_voices data"
```

---

### Task 6: GPU smoke test (opt-in, real models)

**Goal:** An opt-in pytest that loads the real server and generates one clip per voice id — the only test that touches the GPU. Excluded from default CI runs.

**Files:**
- Create: `tests/test_server_gpu.py`
- Modify: `pyproject.toml` (register the `gpu` marker)

**Acceptance Criteria:**
- [ ] Marked `@pytest.mark.gpu`, skipped unless `-m gpu` is passed.
- [ ] When run, every voice id returns a valid non-empty WAV.

**Verify:** `.venv/bin/pytest tests/test_server_gpu.py -m gpu -v` → one pass per voice (slow, needs GPU); `.venv/bin/pytest tests/test_server_gpu.py -v` (no `-m gpu`) → skipped/deselected.

**Steps:**

- [ ] **Step 1: Register the marker in pyproject.toml**

```toml
[tool.pytest.ini_options]
markers = ["gpu: tests that load real models on a CUDA GPU (opt-in, slow)"]
addopts = "-m 'not gpu'"
```

- [ ] **Step 2: Write the GPU smoke test**

```python
# tests/test_server_gpu.py
import io, wave
import pytest
from fastapi.testclient import TestClient
from faster_qwen3_tts.server import create_app, DEFAULT_VOICES
from faster_qwen3_tts.voice_registry import load_registry

pytestmark = pytest.mark.gpu

@pytest.fixture(scope="module")
def client():
    app = create_app(voices_path=DEFAULT_VOICES, warmup=True)
    c = TestClient(app)
    for _ in range(120):  # wait for background warmup
        if c.get("/health").status_code == 200:
            break
        import time; time.sleep(2)
    else:
        pytest.fail("server did not become ready within 240s")
    return c

@pytest.mark.parametrize("voice", list(load_registry(DEFAULT_VOICES).voices))
def test_each_voice_generates_wav(client, voice):
    text = {"Korean": "안녕하세요.", "Japanese": "こんにちは。", "Chinese": "你好。",
            "Spanish": "Hola.", "French": "Bonjour."}.get(
                load_registry(DEFAULT_VOICES).resolve(voice).language, "Hello there.")
    r = client.post("/v1/audio/speech", json={"input": text, "voice": voice})
    assert r.status_code == 200
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getnframes() > 1000
```

- [ ] **Step 3: Verify default run skips GPU; confirm marker works**

Run: `.venv/bin/pytest tests/test_server_gpu.py -v`
Expected: all deselected (`addopts = -m 'not gpu'`).

- [ ] **Step 4: (If a GPU is free) run the smoke test**

Run: `.venv/bin/pytest tests/test_server_gpu.py -m gpu -v`
Expected: one PASS per voice id (slow; loads both models).

- [ ] **Step 5: Commit**

```bash
git add tests/test_server_gpu.py pyproject.toml
git commit -m "test: opt-in GPU smoke test across all voices"
```

---

### Task 7: Docker image + compose example

**Goal:** A GPU Docker image that runs `serve-http`, mounts an HF cache volume for the weights, and is healthy only after warmup. Plus a compose snippet for media-worker.

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.example.yml`

**Acceptance Criteria:**
- [ ] `docker build` succeeds; image contains the package + `server_voices/` (incl. refs).
- [ ] Compose runs with `--gpus`, mounts the HF cache, exposes 8092, healthcheck hits `/health` with a long `start_period`.

**Verify:** `docker build -t qwen-tts .` → builds; `docker run --rm --gpus all -v ~/.cache/huggingface:/hf-cache -p 8092:8092 qwen-tts` then `curl -f localhost:8092/health` returns 200 after warmup.

**Steps:**

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# Dockerfile
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive HF_HOME=/hf-cache PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv curl libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY faster_qwen3_tts ./faster_qwen3_tts
RUN python3.12 -m pip install --break-system-packages \
        torch --index-url https://download.pytorch.org/whl/cu130 && \
    python3.12 -m pip install --break-system-packages ".[server]"
EXPOSE 8092
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8092/health || exit 1
CMD ["faster-qwen3-tts", "serve-http", "--host", "0.0.0.0", "--port", "8092"]
```

- [ ] **Step 2: Write .dockerignore**

```
.venv/
voice_samples/
voice_design_audition/
bench_out/
tests/
docs/
*.html
.git/
__pycache__/
spikes/
```
(Note: `faster_qwen3_tts/server_voices/refs/*.wav` must NOT be ignored — it is under the package dir and required in the image.)

- [ ] **Step 3: Write the compose example**

```yaml
# docker-compose.example.yml — snippet for media-worker's docker-compose
services:
  qwen-tts:
    build: .            # or image: qwen-tts:latest
    ports: ["8092:8092"]
    volumes:
      - ${HOME}/.cache/huggingface:/hf-cache   # pre-populated with the 1.7B Base + CustomVoice repos
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
      start_period: 180s
      retries: 3
```

- [ ] **Step 4: Build the image**

Run: `docker build -t qwen-tts .`
Expected: builds successfully; final image tagged `qwen-tts`.

- [ ] **Step 5: Smoke-run the container**

Run: `docker run -d --name qwen-tts --gpus all -v ${HOME}/.cache/huggingface:/hf-cache -p 8092:8092 qwen-tts`
Then poll: `curl -fs localhost:8092/health` — `503` while warming, `200` once ready (allow ~2 min). Then:
`curl -s localhost:8092/v1/audio/speech -H 'Content-Type: application/json' -d '{"input":"Hello.","voice":"en_m"}' --output /tmp/t.wav && file /tmp/t.wav` → reports WAV.
Cleanup: `docker rm -f qwen-tts`.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.example.yml
git commit -m "feat: GPU Docker image + compose example for media-worker"
```

---

### Task 8: media-worker integration handoff doc

**Goal:** A standalone handoff doc the media-worker team can implement from, derived from spec §12.

**Files:**
- Create: `docs/media-worker-qwen-integration.md`

**Acceptance Criteria:**
- [ ] Documents the request/response contract, the 14 voice ids, the `(Language, Gender)→voice id` mapping, and the exact Java change list (QwenTTSService/Client/Properties, enum + PresetVoiceMapper + CompositeTTSService + QwenHealthIndicator + compose), with the "repoint Korean first" rollout.

**Verify:** `test -f docs/media-worker-qwen-integration.md` and it contains a `(Language, Gender)` mapping table including `ko_m`/`ko_f`.

**Steps:**

- [ ] **Step 1: Write the handoff doc**

Write `docs/media-worker-qwen-integration.md` from spec §12: the wire contract (`POST /v1/audio/speech` `{input, voice, response_format:"wav"}` → raw WAV bytes; `GET /health`); a table mapping each `(Language, Gender)` to a voice id (`(KOREAN, MALE)→ko_m`, `(KOREAN, FEMALE)→ko_f`, `zh-Hans`/`zh-Hant`→`zh_*`, etc., plus the `*_clone` comparison ids); and the numbered Java change list from spec §12 (new `QwenTTSService`/`QwenClient`/`QwenProperties` with a read timeout; `QWEN` in `TtsProvider`, `QWEN_TTS` in `Resource`; `CompositeTTSService` case; `PresetVoiceMapper` Korean rows first; `QwenHealthIndicator`; `application.yml` `tts.qwen.base-url: http://qwen-tts:8092`; the compose service). Note no downstream audio change (worker base64s + ffmpeg-resamples to 48 kHz).

- [ ] **Step 2: Commit**

```bash
git add docs/media-worker-qwen-integration.md
git commit -m "docs: media-worker Qwen TTS integration handoff"
```

---

## Notes for the implementer

- **Run order:** Task 0 first (gates the design). Tasks 2 and 3 are independent and can be done in either order; Task 4 depends on both. Task 5 depends on 4. Task 6/7 depend on 5. Task 8 is independent.
- **If the spike fails:** drop CustomVoice loading in Task 4 (`needs_custom` already guards this), and remove the `en_m`/`ko_f` `type: custom` entries from `voices.yaml`, repointing `en_m`→aiden-clone and `ko_f`→sohee-clone (the `*_clone` refs already exist). Everything else is unchanged.
- **Venv:** all Python/pytest via `.venv/bin/...`. Reinstall editable after Task 5 (`pip install -e ".[server]"`).
- **GPU etiquette:** Tasks 0, 6, 7-step-5 use the GPU — run on a free GPU; never on the first call for latency numbers.
