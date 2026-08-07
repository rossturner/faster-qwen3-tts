# tests/test_server.py
import io, wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from faster_qwen3_tts.server import build_app, MAX_INPUT_CHARS
from faster_qwen3_tts.voice_registry import EMOTIONS, Reference, Registry, VoiceConfig

def _registry():
    return Registry(24000, {
        "en_m": VoiceConfig("en_m", "custom", "English", 0.7, speaker="aiden"),
        "en_f": VoiceConfig("en_f", "clone", "English", 0.7,
                            references=(Reference("ref", Path("/tmp/en_f.wav"), "hello"),)),
    })

class FakeManager:
    sample_rate = 24000
    def __init__(self): self.ready = True; self.error = None; self.calls = []; self.last_reference = None
    def synthesize(self, cfg, reference, text, temperature, max_new_tokens=None):
        self.calls.append((cfg.id, text))
        self.last_reference = reference
        return np.zeros(24000, dtype=np.float32)

def client(manager=None, pronouncer=None):
    from faster_qwen3_tts.pronunciations import EMPTY
    mgr = manager or FakeManager()
    return TestClient(build_app(mgr, _registry(),
                                pronouncer=pronouncer or EMPTY)), mgr

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


def test_speech_passes_the_picked_reference_for_a_clone_voice():
    c, mgr = client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "en_f"})
    assert r.status_code == 200
    assert mgr.last_reference is not None
    assert mgr.last_reference.id == "ref"


def test_speech_request_temperature_overrides_default():
    captured = {}

    class CapturingManager(FakeManager):
        def synthesize(self, cfg, reference, text, temperature, max_new_tokens=None):
            captured["temperature"] = temperature
            return super().synthesize(cfg, reference, text, temperature, max_new_tokens)

    c, _ = client(CapturingManager())
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "en_m", "temperature": 0.33})
    assert r.status_code == 200
    assert captured["temperature"] == 0.33


from faster_qwen3_tts.server import DEFAULT_CHARACTERS, DEFAULT_VOICES, build_registry


def _write_reference_pair(emotion_dir: Path, stem: str = "a", text: str = "hello there") -> None:
    emotion_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(emotion_dir / f"{stem}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 24000 * 5)
    (emotion_dir / f"{stem}.txt").write_text(text)


def test_no_flags_loads_the_bundled_voices_yaml():
    reg = build_registry(None, None)
    assert len(reg.voices) == 12
    assert reg.emotion_fallback_ids == frozenset()


def test_voices_flag_alone_loads_only_that_file():
    reg = build_registry(DEFAULT_VOICES, None)
    assert len(reg.voices) == 12
    assert reg.emotion_fallback_ids == frozenset()


def test_characters_only_does_not_load_the_dubbing_voices(tmp_path):
    """The lyrebird box should not bake 12 clone prompts it never serves."""
    _write_reference_pair(tmp_path / "nicole" / "neutral")

    reg = build_registry(None, tmp_path)
    assert set(reg.voices) == {"nicole:neutral"}
    assert "en_m" not in reg.voices


def test_both_flags_merge_voices_and_characters(tmp_path):
    _write_reference_pair(tmp_path / "nicole" / "neutral")

    reg = build_registry(DEFAULT_VOICES, tmp_path)
    assert "en_m" in reg.voices
    assert "nicole:neutral" in reg.voices
    assert "nicole" in reg.emotion_fallback_ids
    assert "en_m" not in reg.emotion_fallback_ids


def test_colliding_voice_id_across_sources_raises(tmp_path):
    """en_m is both a bundled clone voice and, here, a character id: the two never
    clash as dict keys ('en_m' vs 'en_m:neutral'), but resolve('en_m') would silently
    return the dubbing voice and shadow the character."""
    _write_reference_pair(tmp_path / "en_m" / "neutral")

    with pytest.raises(ValueError, match="en_m"):
        build_registry(DEFAULT_VOICES, tmp_path)


def test_empty_registry_raises(tmp_path):
    with pytest.raises(ValueError, match="no voices"):
        build_registry(None, tmp_path)


def test_bundled_characters_path_points_at_the_shipped_tree():
    assert DEFAULT_CHARACTERS.name == "characters"
    assert DEFAULT_CHARACTERS.parent == DEFAULT_VOICES.parent


def test_health_reports_a_warmup_failure():
    mgr = FakeManager()
    mgr.ready = False
    mgr.error = "create_voice_clone_prompt exploded"
    c, _ = client(mgr)
    r = c.get("/health")
    assert r.status_code == 503
    assert "exploded" in r.json()["error"]


from faster_qwen3_tts.cli import build_parser, cmd_serve_http


def test_bare_characters_flag_parses_to_bundled_sentinel():
    args = build_parser().parse_args(["serve-http", "--characters"])
    assert args.characters == "BUNDLED"


def test_cmd_serve_http_resolves_bare_characters_sentinel_to_bundled(monkeypatch):
    captured = {}

    def fake_create_app(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("faster_qwen3_tts.server.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    args = build_parser().parse_args(["serve-http", "--characters"])
    cmd_serve_http(args)

    assert captured["characters_path"] == DEFAULT_CHARACTERS


def test_cmd_serve_http_passes_an_explicit_characters_path_through_unchanged(monkeypatch, tmp_path):
    captured = {}

    def fake_create_app(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("faster_qwen3_tts.server.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

    args = build_parser().parse_args(["serve-http", "--characters", str(tmp_path)])
    cmd_serve_http(args)

    assert captured["characters_path"] == str(tmp_path)


def _mixed_registry():
    def char(emotion, n):
        refs = tuple(Reference(f"{emotion}{i}", Path(f"/tmp/{emotion}{i}.wav"), "hi")
                     for i in range(n))
        return VoiceConfig("nicole", "clone", "English", 0.7,
                           references=refs, emotion=emotion)
    flat = VoiceConfig("en_m", "clone", "English", 0.7,
                       references=(Reference("ref", Path("/tmp/f.wav"), "hi"),))
    custom_n = VoiceConfig("ono_anna", "custom", "English", 0.7,
                           speaker="ono_anna", instruct="Calm.", emotion="neutral")
    custom_a = VoiceConfig("ono_anna", "custom", "English", 0.7,
                           speaker="ono_anna", instruct="Bright.", emotion="amused")
    voices = {c.key: c for c in (char("neutral", 7), char("amused", 5),
                                 flat, custom_n, custom_a)}
    return Registry(24000, voices,
                    {"nicole": "neutral", "ono_anna": "neutral"},
                    frozenset({"nicole"}))


def _voices_client(ready=True):
    mgr = FakeManager()
    mgr.ready = ready
    return TestClient(build_app(mgr, _mixed_registry()))


def test_voices_lists_a_character_with_all_ten_emotions():
    body = _voices_client().get("/v1/voices").json()
    nicole = next(v for v in body if v["id"] == "nicole")
    assert set(nicole["emotions"]) == set(EMOTIONS)
    assert nicole["emotions"]["neutral"] == 7
    assert nicole["emotions"]["amused"] == 5
    assert nicole["emotions"]["smug"] == 0
    assert nicole["emotion_fallback"] is True
    assert nicole["default_emotion"] == "neutral"


def test_voices_lists_a_flat_voice_with_null_emotions():
    body = _voices_client().get("/v1/voices").json()
    en_m = next(v for v in body if v["id"] == "en_m")
    assert en_m["emotions"] is None
    assert en_m["default_emotion"] is None
    assert en_m["emotion_fallback"] is False


def test_voices_lists_an_emotive_custom_voice_with_its_own_names():
    """Counts are null, not 0: a custom voice has no recordings, and 0 would read as
    'unavailable' when the emotion is perfectly requestable."""
    body = _voices_client().get("/v1/voices").json()
    ono = next(v for v in body if v["id"] == "ono_anna")
    assert ono["emotions"] == {"amused": None, "neutral": None}
    assert ono["emotion_fallback"] is False


def test_voices_answers_before_warmup_completes():
    r = _voices_client(ready=False).get("/v1/voices")
    assert r.status_code == 200


def _speech_client():
    mgr = FakeManager()
    return TestClient(build_app(mgr, _mixed_registry())), mgr


def test_speech_accepts_an_emotion():
    c, _ = _speech_client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "nicole",
                                         "emotion": "amused"})
    assert r.status_code == 200
    assert r.headers["x-tts-emotion"] == "amused"
    assert r.headers["x-tts-requested-emotion"] == "amused"


def test_speech_reports_a_fallback_in_the_headers():
    c, _ = _speech_client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "nicole",
                                         "emotion": "smug"})
    assert r.status_code == 200
    assert r.headers["x-tts-requested-emotion"] == "smug"
    assert r.headers["x-tts-emotion"] == "neutral", "smug is unpopulated"
    assert r.headers["x-tts-reference"].startswith("neutral")


def test_speech_without_emotion_uses_the_default():
    c, _ = _speech_client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "nicole"})
    assert r.headers["x-tts-emotion"] == "neutral"
    assert "x-tts-requested-emotion" not in r.headers


def test_speech_unknown_emotion_400():
    c, _ = _speech_client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "nicole",
                                         "emotion": "furious"})
    assert r.status_code == 400


def test_speech_custom_voice_sets_no_reference_header():
    c, _ = _speech_client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "ono_anna"})
    assert r.status_code == 200
    assert "x-tts-reference" not in r.headers


def test_speech_flat_voice_omits_emotion_headers():
    c, _ = _speech_client()
    r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "en_m"})
    assert r.status_code == 200
    assert "x-tts-emotion" not in r.headers
    assert "x-tts-requested-emotion" not in r.headers
    assert r.headers["x-tts-voice"] == "en_m"
    assert r.headers["x-tts-reference"] == "ref"


def test_page_is_served_when_characters_are_configured():
    mgr = FakeManager()
    c = TestClient(build_app(mgr, _mixed_registry(), serve_page=True))
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/v1/audio/stream" in r.text


def test_page_is_absent_on_a_dubbing_deployment():
    mgr = FakeManager()
    c = TestClient(build_app(mgr, _mixed_registry(), serve_page=False))
    assert c.get("/").status_code == 404


def test_page_makes_no_external_requests():
    """A page that reaches out to a CDN does not work offline on the box, and drags a
    third party into a local dev tool."""
    from faster_qwen3_tts.server import STATIC_DIR
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for marker in ("http://", "https://", "//cdn", "integrity="):
        assert marker not in html, f"external reference {marker!r} in the page"


from faster_qwen3_tts.pronunciations import Pronouncer
from faster_qwen3_tts.server import DEFAULT_PRONUNCIATIONS


def test_speech_applies_pronunciations_before_synthesis():
    c, mgr = client(pronouncer=Pronouncer((("Anby", "Anbee"),)))
    r = c.post("/v1/audio/speech", json={"input": "Hey Anby!", "voice": "en_m"})
    assert r.status_code == 200
    assert mgr.calls[-1][1] == "Hey Anbee!"


def test_speech_length_check_measures_the_raw_input_not_the_substitution():
    # "Anby" -> "Anbee" lengthens the text; the bound protects prefill against the
    # caller's input, so it must not be applied to the rewritten string.
    c, mgr = client(pronouncer=Pronouncer((("Anby", "Anbee"),)))
    body = "Anby " * (MAX_INPUT_CHARS // 5)
    assert len(body.strip()) <= MAX_INPUT_CHARS
    r = c.post("/v1/audio/speech", json={"input": body, "voice": "en_m"})
    assert r.status_code == 200


def test_bare_pronunciations_flag_parses_to_bundled_sentinel():
    args = build_parser().parse_args(["serve-http", "--pronunciations"])
    assert args.pronunciations == "BUNDLED"


def test_cmd_serve_http_resolves_bare_pronunciations_sentinel_to_bundled(monkeypatch):
    captured = {}
    monkeypatch.setattr("faster_qwen3_tts.server.create_app",
                        lambda **kw: captured.update(kw) or object())
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    cmd_serve_http(build_parser().parse_args(["serve-http", "--pronunciations"]))
    assert captured["pronunciations_path"] == DEFAULT_PRONUNCIATIONS


def test_cmd_serve_http_without_the_flag_loads_no_dictionary(monkeypatch):
    captured = {}
    monkeypatch.setattr("faster_qwen3_tts.server.create_app",
                        lambda **kw: captured.update(kw) or object())
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    cmd_serve_http(build_parser().parse_args(["serve-http"]))
    assert captured["pronunciations_path"] is None


def test_bundled_pronunciations_file_loads_and_covers_shipped_names():
    from faster_qwen3_tts.pronunciations import load_pronunciations
    p = load_pronunciations(DEFAULT_PRONUNCIATIONS)
    assert p.apply("Anby Demara") not in ("Anby Demara",)
    assert p.apply("anby") == p.apply("Anby")
    assert p.apply("New Eridu") != "New Eridu"
    assert p.apply("Ridu") != "Ridu"
    # The short form must not fire inside the long one; the letter lookbehind is what
    # stops it, and a key added later without that boundary would break this silently.
    assert p.apply("New Eridu") == p.apply("New Eridu").replace("Reedoo", "")
