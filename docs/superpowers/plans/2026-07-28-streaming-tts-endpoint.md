# Streaming TTS Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ross:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a framed streaming synthesis endpoint (`POST /v1/audio/stream`) and an emotion axis to the voice registry, so lyrebird can drive sentence-batched speech within a ~1 s first-audio budget.

**Architecture:** Additive only — the existing `POST /v1/audio/speech` and the flat 12-voice registry are untouched, because media-worker's dubbing pipeline depends on both. Generation runs on the existing single GPU worker thread and pushes chunks through an unbounded queue to the request coroutine, which frames them. Client disconnect sets a cancel event that breaks the decode loop, freeing the GPU for the next beat.

**Tech Stack:** Python 3.12, FastAPI/Starlette, pytest, numpy. No GPU required for any test in this plan — every test uses a fake manager.

**User decisions (already made):**
- Transport is framed chunked HTTP, not WebSocket or raw PCM.
- `chunk_size` defaults to 8 (335 ms TTFA, 407 ms headroom).
- Playback, buffering and lip-sync live in lyrebird, not here.
- No mid-utterance cancellation — whatever lyrebird batches into one request is its cancellation unit.
- Emotion comes from reference clips, not `instruct` (Stage A: instruct moves pace only).
- Multi-sentence prosody benefit is assumed without measurement, by explicit decision.
- All work happens on the `qwen-tts-server` branch.
- Lip-sync ownership boundary deferred.

**Spec:** `docs/superpowers/specs/2026-07-28-lyrebird-streaming-api-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `faster_qwen3_tts/stream_frames.py` (create) | Frame codec — encode/decode only, no I/O, no server knowledge |
| `faster_qwen3_tts/voice_registry.py` (modify) | Gains the emotion axis; flat entries stay valid |
| `faster_qwen3_tts/wav_io.py` (modify) | Expose `to_pcm16` publicly for the streaming path |
| `faster_qwen3_tts/server.py` (modify) | `synthesize_stream` on the manager; `/v1/audio/stream` route |
| `tests/test_stream_frames.py` (create) | Frame codec round-trips and truncation |
| `tests/test_voice_registry.py` (modify) | Emotion resolution and validation |
| `tests/test_server_stream.py` (create) | Endpoint contract against a fake manager |
| `CLAUDE.md` (modify) | Durable documentation of both endpoints |

Stage B (the persona voice and its emotion clips) is **not** in this plan. Every task
here is developed against the existing `en_f` voice and a synthetic test registry.

---

### Task 1: Frame codec

**Goal:** A self-contained encoder/decoder for the wire format, usable by the server and by lyrebird as reference.

**Files:**
- Create: `faster_qwen3_tts/stream_frames.py`
- Test: `tests/test_stream_frames.py`

**Acceptance Criteria:**
- [ ] Frames encode as `1 byte type | 4 byte big-endian length | payload`
- [ ] JSON frames round-trip through the decoder
- [ ] Multiple concatenated frames decode in order
- [ ] Truncated header and truncated payload both raise `ValueError`

**Verify:** `.venv/bin/python -m pytest tests/test_stream_frames.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test**

Create `tests/test_stream_frames.py`:

```python
import pytest

from faster_qwen3_tts.stream_frames import (
    FRAME_AUDIO,
    FRAME_END,
    FRAME_HEADER,
    encode_frame,
    encode_json_frame,
    iter_frames,
)


def test_encode_frame_layout():
    out = encode_frame(FRAME_AUDIO, b"abcd")
    assert out[0] == FRAME_AUDIO
    assert out[1:5] == b"\x00\x00\x00\x04"
    assert out[5:] == b"abcd"


def test_round_trip_multiple_frames_in_order():
    blob = (
        encode_json_frame(FRAME_HEADER, {"sample_rate": 24000})
        + encode_frame(FRAME_AUDIO, b"\x01\x02")
        + encode_json_frame(FRAME_END, {"total_audio_ms": 12})
    )
    frames = list(iter_frames(blob))
    assert [f[0] for f in frames] == [FRAME_HEADER, FRAME_AUDIO, FRAME_END]
    assert frames[1][1] == b"\x01\x02"


def test_json_frame_decodes_back_to_object():
    import json
    frames = list(iter_frames(encode_json_frame(FRAME_HEADER, {"channels": 1})))
    assert json.loads(frames[0][1]) == {"channels": 1}


def test_empty_payload_is_valid():
    frames = list(iter_frames(encode_frame(FRAME_END, b"")))
    assert frames == [(FRAME_END, b"")]


def test_truncated_header_raises():
    with pytest.raises(ValueError, match="truncated frame header"):
        list(iter_frames(encode_frame(FRAME_AUDIO, b"abcd")[:3]))


def test_truncated_payload_raises():
    with pytest.raises(ValueError, match="truncated frame payload"):
        list(iter_frames(encode_frame(FRAME_AUDIO, b"abcd")[:7]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_stream_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'faster_qwen3_tts.stream_frames'`

- [ ] **Step 3: Write the implementation**

Create `faster_qwen3_tts/stream_frames.py`:

```python
"""Wire format for the streaming synthesis endpoint.

Once the first byte of a streaming response is sent the HTTP status is already
committed, so a mid-stream failure cannot be reported as a 5xx. Framing is what makes
a truncated generation distinguishable from a completed one: a stream that ends
without an END frame is a failed beat, not a short one.

Layout per frame:

    1 byte   frame type
    4 bytes  payload length, big-endian
    N bytes  payload
"""
from __future__ import annotations

import json
import struct
from typing import Any, Iterator, Tuple

FRAME_HEADER = 0x01   # JSON: {sample_rate, channels, format}
FRAME_AUDIO = 0x02    # raw s16le PCM
FRAME_MARK = 0x03     # JSON: {chunk_index, decode_ms, prefill_ms, audio_ms_so_far}
FRAME_ERROR = 0x04    # JSON: {message} -- failure after the response began
FRAME_END = 0x05      # JSON: {total_audio_ms, total_decode_ms}

_PREFIX = struct.Struct(">I")


def encode_frame(frame_type: int, payload: bytes) -> bytes:
    return bytes([frame_type]) + _PREFIX.pack(len(payload)) + payload


def encode_json_frame(frame_type: int, obj: Any) -> bytes:
    return encode_frame(frame_type, json.dumps(obj, separators=(",", ":")).encode())


def iter_frames(data: bytes) -> Iterator[Tuple[int, bytes]]:
    """Decode a complete buffer of frames. Raises ValueError on truncation."""
    offset = 0
    while offset < len(data):
        if len(data) - offset < 5:
            raise ValueError("truncated frame header")
        frame_type = data[offset]
        (length,) = _PREFIX.unpack(data[offset + 1:offset + 5])
        end = offset + 5 + length
        if end > len(data):
            raise ValueError("truncated frame payload")
        yield frame_type, data[offset + 5:end]
        offset = end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_stream_frames.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/stream_frames.py tests/test_stream_frames.py
git commit -m "feat(stream): add frame codec for the streaming endpoint"
```

---

### Task 2: Emotion axis in the voice registry

**Goal:** The registry resolves `(voice, emotion)` pairs while the existing 12 flat entries keep working unchanged.

**Files:**
- Modify: `faster_qwen3_tts/voice_registry.py` (whole file rewritten below)
- Test: `tests/test_voice_registry.py` (append)

**Acceptance Criteria:**
- [ ] The real 12-voice registry still loads with `len(reg.voices) == 12`
- [ ] `resolve("nicole")` returns the default emotion's config
- [ ] `resolve("nicole", "amused")` returns that emotion's config
- [ ] Unknown emotion, and an emotion passed for a flat voice, both raise `KeyError`
- [ ] Missing `default_emotion`, or one not present in `emotions`, raises `ValueError`
- [ ] Emotion entries inherit the voice's `language` and may override `temperature`

**Verify:** `.venv/bin/python -m pytest tests/test_voice_registry.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_voice_registry.py`:

```python
EMOTIVE_YAML = """
sample_rate: 24000
default_temperature: 0.7
voices:
  nicole:
    type: clone
    language: English
    default_emotion: neutral
    emotions:
      neutral: {ref_audio: n.wav, ref_text: "hello there"}
      amused:  {ref_audio: a.wav, ref_text: "hello there", temperature: 0.85}
"""


def test_emotive_voice_resolves_default_and_named(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(EMOTIVE_YAML)
    reg = load_registry(p)

    default = reg.resolve("nicole")
    assert default.emotion == "neutral"
    assert default.key == "nicole:neutral"
    assert default.language == "English"
    assert default.temperature == 0.7

    amused = reg.resolve("nicole", "amused")
    assert amused.emotion == "amused"
    assert amused.key == "nicole:amused"
    assert amused.temperature == 0.85


def test_unknown_emotion_raises_keyerror(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(EMOTIVE_YAML)
    with pytest.raises(KeyError):
        load_registry(p).resolve("nicole", "furious")


def test_emotion_on_flat_voice_raises_keyerror():
    with pytest.raises(KeyError):
        load_registry(REAL).resolve("en_m", "amused")


def test_missing_default_emotion_raises(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\nvoices:\n  x:\n    type: clone\n    language: English\n"
        "    emotions:\n      neutral: {ref_audio: n.wav, ref_text: hi}\n"
    )
    with pytest.raises(ValueError, match="default_emotion"):
        load_registry(p)


def test_default_emotion_not_in_emotions_raises(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\nvoices:\n  x:\n    type: clone\n    language: English\n"
        "    default_emotion: calm\n"
        "    emotions:\n      neutral: {ref_audio: n.wav, ref_text: hi}\n"
    )
    with pytest.raises(ValueError, match="default_emotion"):
        load_registry(p)


def test_flat_voice_key_is_the_bare_id():
    assert load_registry(REAL).resolve("en_m").key == "en_m"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_voice_registry.py -v`
Expected: the six new tests FAIL (`TypeError: resolve() takes 2 positional arguments`, `AttributeError: 'VoiceConfig' object has no attribute 'key'`); the four original tests still pass.

- [ ] **Step 3: Write the implementation**

Replace the whole of `faster_qwen3_tts/voice_registry.py` with:

```python
"""Load and validate the server voice registry (voices.yaml).

Two entry shapes are supported, and both must stay supported: the flat form
(`voice -> clip`) that media-worker's 12 dubbing voices use, and the emotive form
(`voice -> emotion -> clip`) that lyrebird uses. Emotive entries are flattened at load
time into the same dict, keyed `"<voice>:<emotion>"`, so every consumer -- including
warmup, which pre-bakes one clone prompt per entry -- works unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

@dataclass(frozen=True)
class VoiceConfig:
    id: str
    type: str
    language: str
    temperature: float
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[Path] = None
    ref_text: Optional[str] = None
    emotion: Optional[str] = None

    @property
    def key(self) -> str:
        """Registry key: bare id for flat voices, '<id>:<emotion>' for emotive ones."""
        return f"{self.id}:{self.emotion}" if self.emotion else self.id

@dataclass(frozen=True)
class Registry:
    sample_rate: int
    voices: Dict[str, VoiceConfig]
    defaults: Dict[str, str] = field(default_factory=dict)

    def resolve(self, voice_id: str, emotion: Optional[str] = None) -> VoiceConfig:
        if emotion is None:
            if voice_id in self.voices:
                return self.voices[voice_id]
            emotion = self.defaults.get(voice_id)
            if emotion is None:
                raise KeyError(
                    f"Unknown voice id {voice_id!r}. Known: {sorted(self.voices)}")
        key = f"{voice_id}:{emotion}"
        if key not in self.voices:
            raise KeyError(
                f"Unknown voice/emotion {voice_id!r}/{emotion!r}. "
                f"Known: {sorted(self.voices)}")
        return self.voices[key]

def _build(vid: str, spec: Dict[str, Any], base_dir: Path, vtype: str, lang: str,
           temp: float, emotion: Optional[str]) -> VoiceConfig:
    if vtype == "custom":
        if not spec.get("speaker"):
            raise ValueError(f"{vid}: custom voice requires 'speaker'")
        return VoiceConfig(vid, "custom", lang, temp, speaker=spec["speaker"],
                           instruct=spec.get("instruct"), emotion=emotion)
    if not spec.get("ref_audio") or not spec.get("ref_text"):
        raise ValueError(f"{vid}: clone voice requires 'ref_audio' and 'ref_text'")
    ref = (base_dir / spec["ref_audio"]).resolve()
    return VoiceConfig(vid, "clone", lang, temp, ref_audio=ref,
                       ref_text=spec["ref_text"], emotion=emotion)

def load_registry(path) -> Registry:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    base_dir = path.parent
    default_temp = float(data.get("default_temperature", 0.7))
    voices: Dict[str, VoiceConfig] = {}
    defaults: Dict[str, str] = {}
    for vid, raw in data["voices"].items():
        vtype = raw.get("type")
        if vtype not in ("custom", "clone"):
            raise ValueError(f"{vid}: type must be custom|clone, got {vtype!r}")
        lang = raw.get("language")
        if not lang:
            raise ValueError(f"{vid}: language is required")
        temp = float(raw.get("temperature", default_temp))

        emotions = raw.get("emotions")
        if emotions is None:
            cfg = _build(vid, raw, base_dir, vtype, lang, temp, None)
            voices[cfg.key] = cfg
            continue

        default_emotion = raw.get("default_emotion")
        if not default_emotion:
            raise ValueError(f"{vid}: voice with 'emotions' requires 'default_emotion'")
        if default_emotion not in emotions:
            raise ValueError(
                f"{vid}: default_emotion {default_emotion!r} is not one of "
                f"{sorted(emotions)}")
        inherited = {k: v for k, v in raw.items()
                     if k not in ("emotions", "default_emotion")}
        for ename, espec in emotions.items():
            merged = {**inherited, **espec}
            cfg = _build(vid, merged, base_dir, vtype, lang,
                         float(merged.get("temperature", temp)), ename)
            voices[cfg.key] = cfg
        defaults[vid] = default_emotion
    return Registry(int(data.get("sample_rate", 24000)), voices, defaults)
```

- [ ] **Step 4: Run the full suite to verify nothing regressed**

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_e2e_parity.py`
Expected: all pass (57 passed, 12 deselected) — the 51 existing tests plus the 6 new ones.

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/voice_registry.py tests/test_voice_registry.py
git commit -m "feat(registry): add emotion axis, keeping flat entries valid"
```

---

### Task 3: Streaming synthesis on the model manager

**Goal:** `ModelManager.synthesize_stream` yields decoded chunks off the GPU worker thread and stops generating when its cancel event is set.

**Files:**
- Modify: `faster_qwen3_tts/server.py` (`ModelManager`)
- Modify: `faster_qwen3_tts/wav_io.py` (expose `to_pcm16`)
- Test: `tests/test_server_stream.py` (create — manager half)

**Acceptance Criteria:**
- [ ] `synthesize_stream` yields `(pcm, timing)` tuples from the model generator
- [ ] Setting the cancel event stops consumption of the model generator
- [ ] An exception in generation propagates to the consumer
- [ ] Clone prompts are keyed by `cfg.key`, so emotions get their own prompt
- [ ] A `custom`-type voice raises `ValueError`
- [ ] `to_pcm16` is importable from `wav_io` and `to_wav_bytes` still works

**Verify:** `.venv/bin/python -m pytest tests/test_server_stream.py -v -k manager` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test**

Create `tests/test_server_stream.py`:

```python
import threading
import time

import numpy as np
import pytest

from faster_qwen3_tts.server import ModelManager
from faster_qwen3_tts.voice_registry import Registry, VoiceConfig


def _clone_cfg(emotion=None):
    return VoiceConfig("nicole", "clone", "English", 0.7,
                       ref_audio="/tmp/x.wav", ref_text="hello", emotion=emotion)


class FakeBase:
    """Stands in for FasterQwen3TTS: yields chunks, records what it was asked for.

    `delay` matters for the cancellation test. The producer runs on its own thread and
    pushes into an unbounded queue, so an instantaneous fake would finish generating
    every chunk before the consumer read its first one -- and the test would prove
    nothing. A small per-chunk delay makes it behave like a real decode.
    """

    def __init__(self, chunks=3, fail_at=None, delay=0.0):
        self.chunks = chunks
        self.fail_at = fail_at
        self.delay = delay
        self.consumed = 0
        self.kwargs = None

    def generate_voice_clone_streaming(self, **kwargs):
        self.kwargs = kwargs
        for i in range(self.chunks):
            if self.fail_at is not None and i == self.fail_at:
                raise RuntimeError("decode exploded")
            if self.delay:
                time.sleep(self.delay)
            self.consumed += 1
            yield np.zeros(2400, dtype=np.float32), 24000, {"chunk_index": i,
                                                            "decode_ms": 1.5}


def _manager(base, cfg):
    registry = Registry(24000, {cfg.key: cfg})
    mgr = ModelManager(registry)
    mgr._base = base
    mgr._clone_prompts = {cfg.key: {"fake": "prompt"}}
    mgr.ready = True
    return mgr


def test_manager_stream_yields_all_chunks():
    cfg = _clone_cfg("amused")
    base = FakeBase(chunks=3)
    mgr = _manager(base, cfg)

    out = list(mgr.synthesize_stream(cfg, "hi there", 0.7, chunk_size=8))

    assert len(out) == 3
    assert out[0][0].shape == (2400,)
    assert out[2][1]["chunk_index"] == 2
    assert base.kwargs["chunk_size"] == 8
    assert base.kwargs["voice_clone_prompt"] == {"fake": "prompt"}
    assert base.kwargs["temperature"] == 0.7


def test_manager_stream_cancel_stops_generation():
    cfg = _clone_cfg()
    base = FakeBase(chunks=50, delay=0.01)
    mgr = _manager(base, cfg)
    cancel = threading.Event()

    got = 0
    for _chunk, _timing in mgr.synthesize_stream(cfg, "hi", 0.7, 8, cancel=cancel):
        got += 1
        if got == 2:
            cancel.set()

    assert got == 2, "the consumer must stop as soon as cancel is set"
    assert base.consumed < 20, "the model generator must not run to completion"


def test_manager_stream_propagates_generation_error():
    cfg = _clone_cfg()
    mgr = _manager(FakeBase(chunks=5, fail_at=2), cfg)

    with pytest.raises(RuntimeError, match="decode exploded"):
        list(mgr.synthesize_stream(cfg, "hi", 0.7, 8))


def test_manager_stream_rejects_custom_voice():
    cfg = VoiceConfig("preset", "custom", "English", 0.7, speaker="aiden")
    mgr = _manager(FakeBase(), cfg)

    with pytest.raises(ValueError, match="clone"):
        list(mgr.synthesize_stream(cfg, "hi", 0.7, 8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_server_stream.py -v`
Expected: FAIL with `AttributeError: 'ModelManager' object has no attribute 'synthesize_stream'`

- [ ] **Step 3: Expose `to_pcm16` in `wav_io.py`**

In `faster_qwen3_tts/wav_io.py`, rename the private helper and point the existing
function at it:

```python
import struct
import numpy as np


def to_pcm16(pcm: np.ndarray) -> bytes:
    return np.clip(pcm * 32768.0, -32768, 32767).astype("<i2").tobytes()


def to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    raw = to_pcm16(np.asarray(pcm, dtype=np.float32).flatten())
    n_channels, bits = 1, 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                                    byte_rate, block_align, bits)
    header += b"data" + struct.pack("<I", len(raw))
    return header + raw
```

- [ ] **Step 4: Add `synthesize_stream` to `ModelManager`**

In `faster_qwen3_tts/server.py`, add `import queue` to the imports (`threading`,
`Optional` and `numpy` are already imported), then insert this method into
`ModelManager` immediately after `synthesize`:

```python
    def synthesize_stream(self, cfg: VoiceConfig, text: str, temperature: float,
                          chunk_size: int, max_new_tokens=None,
                          cancel: Optional[threading.Event] = None):
        """Yield (pcm, timing) chunks as they decode.

        Generation runs on the single GPU worker thread and pushes into an unbounded
        queue; the caller consumes from this generator. The queue is deliberately
        unbounded: a bounded one would block the producer when a cancelled consumer
        stops reading, pinning the GPU thread on a `put` and stalling the next request.

        Setting `cancel` breaks the loop over the model's generator, which stops decode.
        Merely closing the response is not enough -- the GPU is serialised on one
        thread, so a cancelled batch that decodes to completion delays the next beat.
        """
        if cfg.type != "clone":
            raise ValueError(f"streaming supports clone voices only, got {cfg.type!r}")
        tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        q: queue.Queue = queue.Queue()
        done = object()

        def produce():
            try:
                for chunk, _sr, timing in self._base.generate_voice_clone_streaming(
                        text=text, language=cfg.language,
                        voice_clone_prompt=self._clone_prompts[cfg.key],
                        ref_text=cfg.ref_text, xvec_only=False,
                        temperature=temperature, chunk_size=chunk_size,
                        max_new_tokens=tokens):
                    if cancel is not None and cancel.is_set():
                        break
                    q.put((np.asarray(chunk, dtype=np.float32), timing))
            except BaseException as exc:            # surfaced to the consumer below
                q.put(exc)
            finally:
                q.put(done)

        future = self._gpu.submit(produce)
        try:
            while True:
                # Checked before draining the queue, not only in the producer: the
                # producer runs ahead, so a queue full of already-decoded chunks would
                # otherwise keep being yielded long after the caller cancelled.
                if cancel is not None and cancel.is_set():
                    break
                item = q.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Also runs when the caller abandons this generator (client disconnect
            # closes it), which is what guarantees the GPU thread is released.
            if cancel is not None:
                cancel.set()
            future.result()
```

Also change the clone-prompt lookup in `_synthesize_blocking` from `cfg.id` to
`cfg.key`, so emotive voices reach their own prompt:

```python
                voice_clone_prompt=self._clone_prompts[cfg.key], ref_text=cfg.ref_text,
```

And in `_load_and_warm`, key the pre-baked prompts the same way — replace the loop:

```python
        for cfg in self.registry.voices.values():
            if cfg.type == "clone":
                self._clone_prompts[cfg.key] = self._base.model.create_voice_clone_prompt(
                    ref_audio=str(cfg.ref_audio), ref_text=cfg.ref_text, x_vector_only_mode=False)
```

and the clone warmup call below it:

```python
            self._base.generate_voice_clone(
                text="Warmup.", language=any_clone.language,
                voice_clone_prompt=self._clone_prompts[any_clone.key],
                ref_text=any_clone.ref_text, max_new_tokens=20)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_stream.py -v`
Expected: 4 passed

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_e2e_parity.py`
Expected: all pass (61 passed, 12 deselected)

- [ ] **Step 6: Commit**

```bash
git add faster_qwen3_tts/server.py faster_qwen3_tts/wav_io.py tests/test_server_stream.py
git commit -m "feat(server): stream synthesis off the GPU thread with cancellation"
```

---

### Task 4: The `/v1/audio/stream` endpoint

**Goal:** A framed streaming route that validates before the first byte, emits header/audio/mark/end frames, reports mid-stream failure as an error frame, and cancels generation on client disconnect.

**Files:**
- Modify: `faster_qwen3_tts/server.py` (`build_app`, new request model, constants)
- Test: `tests/test_server_stream.py` (append — endpoint half)

**Acceptance Criteria:**
- [ ] Happy path emits header, then one audio + one mark frame per chunk, then end
- [ ] Audio frame payloads are s16le PCM of the right byte length (2 bytes/sample)
- [ ] `emotion` is passed through to registry resolution
- [ ] `400` before any frame for: empty input, over-length input, unknown voice, unknown emotion, custom voice, out-of-range `chunk_size`
- [ ] `503` when warming
- [ ] Mid-stream failure emits an error frame and **no** end frame
- [ ] The manager receives a `threading.Event` as its cancel argument

**Verify:** `.venv/bin/python -m pytest tests/test_server_stream.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server_stream.py`:

```python
from fastapi.testclient import TestClient

from faster_qwen3_tts.server import DEFAULT_CHUNK_SIZE, MAX_INPUT_CHARS, build_app
from faster_qwen3_tts.stream_frames import (
    FRAME_AUDIO, FRAME_END, FRAME_ERROR, FRAME_HEADER, FRAME_MARK, iter_frames,
)


class FakeStreamManager:
    sample_rate = 24000

    def __init__(self, chunks=3, fail_at=None):
        self.ready = True
        self.chunks = chunks
        self.fail_at = fail_at
        self.calls = []
        self.cancel = None

    def synthesize(self, cfg, text, temperature, max_new_tokens=None):
        return np.zeros(24000, dtype=np.float32)

    def synthesize_stream(self, cfg, text, temperature, chunk_size,
                          max_new_tokens=None, cancel=None):
        self.calls.append((cfg.key, text, temperature, chunk_size))
        self.cancel = cancel
        for i in range(self.chunks):
            if self.fail_at is not None and i == self.fail_at:
                raise RuntimeError("decode exploded")
            yield np.zeros(2400, dtype=np.float32), {"chunk_index": i,
                                                     "decode_ms": 1.5,
                                                     "prefill_ms": 0.0}


def _stream_registry():
    neutral = VoiceConfig("nicole", "clone", "English", 0.7,
                          ref_audio="/tmp/n.wav", ref_text="hi", emotion="neutral")
    amused = VoiceConfig("nicole", "clone", "English", 0.7,
                         ref_audio="/tmp/a.wav", ref_text="hi", emotion="amused")
    preset = VoiceConfig("preset", "custom", "English", 0.7, speaker="aiden")
    return Registry(24000,
                    {neutral.key: neutral, amused.key: amused, preset.key: preset},
                    {"nicole": "neutral"})


def _stream_client(manager=None):
    mgr = manager or FakeStreamManager()
    return TestClient(build_app(mgr, _stream_registry())), mgr


def _frames(response):
    return list(iter_frames(response.content))


def test_stream_happy_path_frame_sequence():
    c, mgr = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})

    assert r.status_code == 200
    types = [t for t, _ in _frames(r)]
    assert types == [FRAME_HEADER,
                     FRAME_AUDIO, FRAME_MARK,
                     FRAME_AUDIO, FRAME_MARK,
                     FRAME_AUDIO, FRAME_MARK,
                     FRAME_END]
    assert mgr.calls[0][0] == "nicole:neutral"
    assert mgr.calls[0][3] == DEFAULT_CHUNK_SIZE


def test_stream_header_frame_describes_the_audio():
    import json
    c, _ = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})
    header = json.loads(_frames(r)[0][1])
    assert header == {"sample_rate": 24000, "channels": 1, "format": "s16le"}


def test_stream_audio_frames_are_pcm16():
    c, _ = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})
    audio = [payload for t, payload in _frames(r) if t == FRAME_AUDIO]
    assert all(len(p) == 2400 * 2 for p in audio)


def test_stream_end_frame_totals_the_audio():
    import json
    c, _ = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})
    end = json.loads(_frames(r)[-1][1])
    assert end["total_audio_ms"] == pytest.approx(3 * 2400 / 24000 * 1000)


def test_stream_emotion_is_resolved():
    c, mgr = _stream_client()
    r = c.post("/v1/audio/stream",
               json={"input": "hello", "voice": "nicole", "emotion": "amused"})
    assert r.status_code == 200
    assert mgr.calls[0][0] == "nicole:amused"


def test_stream_passes_a_cancel_event():
    c, mgr = _stream_client()
    c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})
    assert isinstance(mgr.cancel, threading.Event)


def test_stream_midstream_failure_emits_error_frame_and_no_end():
    c, _ = _stream_client(FakeStreamManager(chunks=5, fail_at=2))
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})

    types = [t for t, _ in _frames(r)]
    assert r.status_code == 200, "status is committed before the failure is known"
    assert types[-1] == FRAME_ERROR
    assert FRAME_END not in types


@pytest.mark.parametrize("body,expected", [
    ({"input": "   ", "voice": "nicole"}, 400),
    ({"input": "a" * (MAX_INPUT_CHARS + 1), "voice": "nicole"}, 400),
    ({"input": "hi", "voice": "nope"}, 400),
    ({"input": "hi", "voice": "nicole", "emotion": "furious"}, 400),
    ({"input": "hi", "voice": "preset"}, 400),
    ({"input": "hi", "voice": "nicole", "chunk_size": 0}, 400),
    ({"input": "hi", "voice": "nicole", "chunk_size": 999}, 400),
])
def test_stream_rejects_bad_requests_before_streaming(body, expected):
    c, _ = _stream_client()
    assert c.post("/v1/audio/stream", json=body).status_code == expected


def test_stream_503_when_warming():
    mgr = FakeStreamManager()
    mgr.ready = False
    c, _ = _stream_client(mgr)
    assert c.post("/v1/audio/stream",
                  json={"input": "hi", "voice": "nicole"}).status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_stream.py -v`
Expected: the new tests FAIL with `ImportError: cannot import name 'DEFAULT_CHUNK_SIZE'`

- [ ] **Step 3: Write the implementation**

In `faster_qwen3_tts/server.py`, replace the three existing import lines
(`from fastapi import ...`, `from fastapi.responses import ...`, and
`from .wav_io import to_wav_bytes`) with:

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from .stream_frames import (
    FRAME_AUDIO, FRAME_END, FRAME_ERROR, FRAME_HEADER, FRAME_MARK,
    encode_frame, encode_json_frame,
)
from .wav_io import to_pcm16, to_wav_bytes
```

Add the constants next to `MAX_INPUT_CHARS`:

```python
DEFAULT_CHUNK_SIZE = 8      # 335ms TTFA with 407ms of headroom against a slow chunk
MAX_CHUNK_SIZE = 48
STREAM_MEDIA_TYPE = "application/vnd.lyrebird.tts-stream"
```

Add the request model below `SpeechRequest`:

```python
class StreamRequest(BaseModel):
    input: str
    voice: str
    emotion: Optional[str] = None
    temperature: Optional[float] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
```

Add this route inside `build_app`, after the existing `speech` route:

```python
    @app.post("/v1/audio/stream")
    async def stream(req: StreamRequest, request: Request):
        # Everything that can be rejected must be rejected here: once the first frame
        # is written the status code is committed and 4xx is no longer available.
        if not manager.ready:
            raise HTTPException(503, "Model warming up")
        text = req.input.strip()
        if not text:
            raise HTTPException(400, "'input' is empty")
        if len(text) > MAX_INPUT_CHARS:
            raise HTTPException(400, f"'input' exceeds {MAX_INPUT_CHARS} chars; chunk upstream")
        if not 1 <= req.chunk_size <= MAX_CHUNK_SIZE:
            raise HTTPException(400, f"'chunk_size' must be 1..{MAX_CHUNK_SIZE}")
        try:
            cfg = registry.resolve(req.voice, req.emotion)
        except KeyError as e:
            raise HTTPException(400, str(e))
        if cfg.type != "clone":
            raise HTTPException(400, "streaming supports clone voices only")
        temp = req.temperature if req.temperature is not None else cfg.temperature

        cancel = threading.Event()
        sample_rate = manager.sample_rate

        async def frames():
            yield encode_json_frame(FRAME_HEADER, {
                "sample_rate": sample_rate, "channels": 1, "format": "s16le"})
            audio_ms = decode_ms = 0.0
            try:
                chunks = manager.synthesize_stream(
                    cfg, text, temp, req.chunk_size, cancel=cancel)
                async for pcm, timing in iterate_in_threadpool(chunks):
                    if await request.is_disconnected():
                        cancel.set()
                        return
                    audio_ms += len(pcm) / sample_rate * 1000
                    decode_ms += float(timing.get("decode_ms", 0.0))
                    yield encode_frame(FRAME_AUDIO, to_pcm16(pcm))
                    yield encode_json_frame(FRAME_MARK, {
                        "chunk_index": timing.get("chunk_index"),
                        "decode_ms": timing.get("decode_ms"),
                        "prefill_ms": timing.get("prefill_ms"),
                        "audio_ms_so_far": round(audio_ms, 2),
                    })
                yield encode_json_frame(FRAME_END, {
                    "total_audio_ms": round(audio_ms, 2),
                    "total_decode_ms": round(decode_ms, 2)})
            except Exception as exc:
                logger.exception("streaming synthesis failed")
                cancel.set()
                yield encode_json_frame(FRAME_ERROR, {"message": str(exc)})

        return StreamingResponse(frames(), media_type=STREAM_MEDIA_TYPE)
```


- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_server_stream.py -v`
Expected: all pass (4 manager tests + 15 endpoint tests including parametrised cases)

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_e2e_parity.py`
Expected: all pass, no regressions in `tests/test_server.py`

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/server.py tests/test_server_stream.py
git commit -m "feat(server): add framed POST /v1/audio/stream endpoint"
```

---

### Task 5: Document both endpoints

**Goal:** `CLAUDE.md`'s serving section describes what exists, rather than describing the server as unbuilt.

**Files:**
- Modify: `CLAUDE.md` (the `## Serving as an HTTP API` section)

**Acceptance Criteria:**
- [ ] The section no longer says the server is unbuilt
- [ ] Both endpoints are documented with their intended consumer
- [ ] The frame layout and frame types are listed
- [ ] The registry's two entry shapes are shown
- [ ] Nothing in `CLAUDE.md` references the spec or plan under `docs/superpowers/`

**Verify:** `grep -n "not built yet\|superpowers" CLAUDE.md` → no matches

**Steps:**

- [ ] **Step 1: Replace the serving section**

In `CLAUDE.md`, replace the heading `## Serving as an HTTP API (forward-looking; server not built yet)` and its body with:

```markdown
## Serving as an HTTP API

`faster_qwen3_tts/server.py` loads the 1.7B model once, warms the CUDA graphs, and keeps
it resident. Start it with `faster-qwen3-tts serve-http --voices <voices.yaml>`.
`GET /health` returns 503 while warming and 200 once ready — never route traffic before
200. Two endpoints, two consumers:

### `POST /v1/audio/speech` — whole-file, for media-worker

`{input, voice, response_format:"wav", temperature?}` → one complete 24 kHz mono 16-bit
WAV. Non-streaming: nothing is emitted until generation finishes, which costs ~1.4 s for
a single sentence and scales with length. Right for dubbing, wrong for a live loop.

### `POST /v1/audio/stream` — framed streaming, for live use

`{input, voice, emotion?, temperature?, chunk_size?}` → a stream of length-prefixed
frames, `Content-Type: application/vnd.lyrebird.tts-stream`. First audio arrives in
~335 ms regardless of input length. `chunk_size` defaults to 8; smaller values cut TTFA
but shrink the headroom against a slow chunk (407 ms at 8, 31 ms at 4).

Frame layout is `1 byte type | 4 byte big-endian length | payload`:

| Type | Payload |
|---|---|
| `0x01` header | JSON `{sample_rate, channels, format}` |
| `0x02` audio | raw s16le PCM |
| `0x03` mark | JSON `{chunk_index, decode_ms, prefill_ms, audio_ms_so_far}` |
| `0x04` error | JSON `{message}` — failure after the response began |
| `0x05` end | JSON `{total_audio_ms, total_decode_ms}` |

Framing exists because the HTTP status is committed once the first byte is sent, so a
mid-stream failure cannot be a 5xx. **A stream that ends without an end frame is a
failed generation, not a short one** — clients must treat it that way. Errors detected
before streaming begins still use normal status codes (400/503/500).

Clients cancel by closing the connection, which aborts the decode loop. That matters:
GPU work is serialised on one worker thread, so a cancelled request that kept decoding
would delay the next one. Streaming supports `clone` voices only.

### Voice registry

`voices.yaml` accepts two entry shapes. Flat, which the 12 dubbing voices use:

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
prompt is pre-baked per entry at warmup. Emotion comes from the reference clip because
`instruct` measurably moves only speaking rate on the ICL clone path, not pitch or
energy.
```

- [ ] **Step 2: Verify**

Run: `grep -n "not built yet\|superpowers" CLAUDE.md`
Expected: no output

Run: `.venv/bin/python -m pytest tests/ -q --ignore=tests/test_e2e_parity.py`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the speech and stream endpoints"
```

---

## Not in this plan

- **Stage B** — the persona voice and its emotion clips. Independent of everything here;
  every task above is developed against `en_f` or a synthetic test registry.
- **Any lyrebird-side work** — the `adapter:tts` module and the player sidecar.
- **A GPU integration test** for the streaming path. `tests/test_server_gpu.py` is the
  opt-in home for one; it is worth adding once a real emotive voice exists to stream.
- **Multiple clips per emotion** (the old `EmotionCache` random-pick idea). Deliberate
  YAGNI — the registry shape above extends to it without a rewrite.
- **`instruct` as a secondary pace control.** The spec allows it; nothing needs it yet.
- **Deployment of a second instance.** No code required — `serve-http` already takes
  `--voices` and `--port`, so lyrebird's instance is a second invocation with its own
  registry file, leaving media-worker's untouched.
