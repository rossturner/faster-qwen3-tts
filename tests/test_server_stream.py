import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from faster_qwen3_tts.server import DEFAULT_CHUNK_SIZE, MAX_INPUT_CHARS, ModelManager, build_app
from faster_qwen3_tts.stream_frames import (
    FRAME_AUDIO, FRAME_END, FRAME_ERROR, FRAME_HEADER, FRAME_MARK, iter_frames,
)
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


def test_manager_stream_abandon_without_cancel_stops_generation():
    cfg = _clone_cfg()
    base = FakeBase(chunks=50, delay=0.01)
    mgr = _manager(base, cfg)

    gen = mgr.synthesize_stream(cfg, "hi", 0.7, 8)
    got = 0
    for _chunk, _timing in gen:
        got += 1
        if got == 2:
            break
    gen.close()

    assert got == 2
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
