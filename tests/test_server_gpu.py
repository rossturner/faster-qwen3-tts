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
    for _ in range(120):
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
