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


def test_emotion_missing_ref_audio_does_not_inherit_voice_level_clip(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\ndefault_temperature: 0.7\n"
        "voices:\n  nicole:\n    type: clone\n    language: English\n"
        "    ref_audio: leaked.wav\n    ref_text: leaked text\n"
        "    default_emotion: neutral\n"
        "    emotions:\n"
        "      neutral: {ref_audio: n.wav, ref_text: hi}\n"
        "      amused:  {ref_text: hi}\n"
    )
    with pytest.raises(ValueError, match="nicole:amused") as exc_info:
        load_registry(p)
    assert "leaked" not in str(exc_info.value)


def test_duplicate_flattened_key_raises_valueerror(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\ndefault_temperature: 0.7\n"
        "voices:\n"
        "  nicole:\n    type: clone\n    language: English\n"
        "    default_emotion: amused\n"
        "    emotions:\n"
        "      amused: {ref_audio: a.wav, ref_text: hi}\n"
        "  \"nicole:amused\": {type: clone, language: English, ref_audio: r.wav, ref_text: hi}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(p)
