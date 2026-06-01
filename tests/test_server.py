# tests/test_server.py
import io, wave
import numpy as np
from fastapi.testclient import TestClient
from faster_qwen3_tts.server import build_app, MAX_INPUT_CHARS
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
        return np.zeros(24000, dtype=np.float32)

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


def test_speech_too_long_input_400():
    c, _ = client()
    long_text = "a" * (MAX_INPUT_CHARS + 1)
    assert c.post("/v1/audio/speech", json={"input": long_text, "voice": "en_m"}).status_code == 400


def test_speech_request_temperature_overrides_default():
    captured = {}

    class CapturingManager(FakeManager):
        def synthesize(self, cfg, text, temperature, max_new_tokens=None):
            captured["temperature"] = temperature
            return super().synthesize(cfg, text, temperature, max_new_tokens)

    c, _ = client(CapturingManager())
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "en_m", "temperature": 0.33})
    assert r.status_code == 200
    assert captured["temperature"] == 0.33
