import logging
import os
import wave
import pytest
from pathlib import Path

from faster_qwen3_tts.characters import load_characters


def _wav(path: Path, seconds: float = 5.0, sample_rate: int = 24000):
    """A real, readable wav -- soundfile must be able to report its duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def _pair(root: Path, character: str, emotion: str, stem: str,
          text: str = "hello there", seconds: float = 5.0):
    d = root / character / emotion
    _wav(d / f"{stem}.wav", seconds)
    (d / f"{stem}.txt").write_text(text, encoding="utf-8")


def test_discovers_pairs_with_stem_ids_in_sorted_order(tmp_path):
    _pair(tmp_path, "nicole", "neutral", "b_second")
    _pair(tmp_path, "nicole", "neutral", "a_first")
    reg = load_characters(tmp_path, 0.7)

    cfg = reg.resolve("nicole")
    assert [r.id for r in cfg.references] == ["a_first", "b_second"]
    assert cfg.references[0].text == "hello there"
    assert cfg.references[0].audio.is_absolute()


def test_character_id_is_the_directory_name_and_defaults_apply(tmp_path):
    _pair(tmp_path, "billy", "neutral", "one")
    cfg = load_characters(tmp_path, 0.7).resolve("billy")
    assert cfg.id == "billy"
    assert cfg.type == "clone"
    assert cfg.language == "English"
    assert cfg.temperature == 0.7


def test_orphan_wav_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    _wav(tmp_path / "nicole" / "neutral" / "lonely.wav")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert "lonely.wav" in caplog.text


def test_orphan_transcript_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    (tmp_path / "nicole" / "neutral" / "lonely.txt").write_text("hi")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert "lonely.txt" in caplog.text


def test_empty_transcript_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    _pair(tmp_path, "nicole", "neutral", "blank", text="   ")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert "blank.txt" in caplog.text


def test_unreadable_audio_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    (tmp_path / "nicole" / "neutral" / "junk.wav").write_bytes(b"not a wav")
    (tmp_path / "nicole" / "neutral" / "junk.txt").write_text("hi")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert "junk.wav" in caplog.text


def test_misspelled_emotion_directory_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    _pair(tmp_path, "nicole", "anoyed", "typo")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert set(reg.voices) == {"nicole:neutral"}
    assert "anoyed" in caplog.text


def test_gitkeep_and_stray_files_are_ignored_silently(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    (tmp_path / "nicole" / "neutral" / ".gitkeep").write_text("")
    (tmp_path / "nicole" / "neutral" / "notes.md").write_text("x")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert ".gitkeep" not in caplog.text
    assert "notes.md" not in caplog.text


def test_emotion_with_no_valid_references_produces_no_entry(tmp_path):
    """Load-bearing: an empty entry would be returned by resolve() and then face an
    empty reference tuple, crashing instead of falling back."""
    _pair(tmp_path, "nicole", "neutral", "good")
    (tmp_path / "nicole" / "smug").mkdir(parents=True)
    reg = load_characters(tmp_path, 0.7)
    assert "nicole:smug" not in reg.voices
    assert reg.resolve("nicole", "smug").emotion == "neutral"


def test_character_without_neutral_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "amused", "only")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert reg.voices == {}
    assert "neutral" in caplog.text


def test_entirely_empty_character_is_skipped_quietly(tmp_path, caplog):
    (tmp_path / "anby" / "neutral").mkdir(parents=True)
    _pair(tmp_path, "nicole", "neutral", "good")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert set(reg.voices) == {"nicole:neutral"}
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_character_yaml_overrides_language_and_temperature(tmp_path):
    _pair(tmp_path, "nicole", "neutral", "good")
    (tmp_path / "nicole" / "character.yaml").write_text(
        "language: Japanese\ntemperature: 0.85\n")
    cfg = load_characters(tmp_path, 0.7).resolve("nicole")
    assert (cfg.language, cfg.temperature) == ("Japanese", 0.85)


def test_character_yaml_unknown_key_raises(tmp_path):
    _pair(tmp_path, "nicole", "neutral", "good")
    (tmp_path / "nicole" / "character.yaml").write_text("langauge: Japanese\n")
    with pytest.raises(ValueError, match="langauge"):
        load_characters(tmp_path, 0.7)


def test_a_bad_character_yaml_takes_down_the_whole_load_not_just_that_character(tmp_path):
    """Declared config is fatal, so a typo in one character's character.yaml is loud --
    it fails load_characters() entirely rather than silently costing just that character."""
    _pair(tmp_path, "anby", "neutral", "good")
    _pair(tmp_path, "billy", "neutral", "good")
    _pair(tmp_path, "nicole", "neutral", "good")
    (tmp_path / "nicole" / "character.yaml").write_text("langauge: Japanese\n")
    with pytest.raises(ValueError, match="langauge"):
        load_characters(tmp_path, 0.7)


def test_hidden_and_underscored_directories_are_ignored(tmp_path):
    _pair(tmp_path, "nicole", "neutral", "good")
    _pair(tmp_path, ".scratch", "neutral", "x")
    _pair(tmp_path, "_wip", "neutral", "x")
    assert set(load_characters(tmp_path, 0.7).voices) == {"nicole:neutral"}


def test_out_of_range_duration_warns_but_is_still_used(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "toolong", seconds=45.0)
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["toolong"]
    assert "toolong.wav" in caplog.text


def test_too_short_duration_warns_but_is_still_used(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "tooshort", seconds=1.0)
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["tooshort"]
    assert "tooshort.wav" in caplog.text


def test_registry_declares_the_character_for_fallback_and_defaults(tmp_path):
    _pair(tmp_path, "nicole", "neutral", "good")
    reg = load_characters(tmp_path, 0.7)
    assert reg.emotion_fallback_ids == frozenset({"nicole"})
    assert reg.defaults == {"nicole": "neutral"}
    assert reg.sample_rate == 24000


def test_missing_root_raises(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        load_characters(tmp_path / "nope", 0.7)


def test_non_latin1_reference_stem_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    _pair(tmp_path, "nicole", "neutral", "あ")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert "not safe" in caplog.text


def test_non_latin1_character_directory_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    _pair(tmp_path, "あ", "neutral", "one")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert set(reg.voices) == {"nicole:neutral"}
    assert "not safe" in caplog.text


def test_reference_stem_with_embedded_crlf_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    _pair(tmp_path, "nicole", "neutral", "evil\r\nX-Injected: true")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert "not safe" in caplog.text


def test_reference_stem_with_del_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    _pair(tmp_path, "nicole", "neutral", "evil\x7fstem")
    with caplog.at_level("WARNING"):
        reg = load_characters(tmp_path, 0.7)
    assert [r.id for r in reg.resolve("nicole").references] == ["good"]
    assert "not safe" in caplog.text


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission bits")
def test_unreadable_character_directory_is_skipped_with_a_warning(tmp_path, caplog):
    _pair(tmp_path, "nicole", "neutral", "good")
    locked = tmp_path / "billy"
    _pair(tmp_path, "billy", "neutral", "one")
    os.chmod(locked, 0o000)
    try:
        with caplog.at_level("WARNING"):
            reg = load_characters(tmp_path, 0.7)
    finally:
        os.chmod(locked, 0o755)
    assert set(reg.voices) == {"nicole:neutral"}
    assert "billy" in caplog.text
