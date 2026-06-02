import pytest
from pathlib import Path
from faster_qwen3_tts.voice_registry import load_registry, VoiceConfig

REAL = Path("faster_qwen3_tts/server_voices/voices.yaml")

def test_loads_all_12_real_voices():
    reg = load_registry(REAL)
    assert len(reg.voices) == 12
    assert all(v.type == "clone" for v in reg.voices.values())
    en_m = reg.resolve("en_m")
    assert en_m.type == "clone"
    assert en_m.ref_audio.is_absolute() and en_m.ref_audio.exists()
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
