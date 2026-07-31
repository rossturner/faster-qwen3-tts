import pytest
from pathlib import Path
from faster_qwen3_tts.voice_registry import (
    EMOTIONS, Reference, Registry, VoiceConfig, load_registry, merge,
    _is_safe_header_value,
)

REAL = Path("faster_qwen3_tts/server_voices/voices.yaml")

def test_loads_all_12_real_voices():
    reg = load_registry(REAL)
    assert len(reg.voices) == 12
    assert all(v.type == "clone" for v in reg.voices.values())
    en_m = reg.resolve("en_m")
    assert en_m.type == "clone"
    assert en_m.references[0].audio.is_absolute() and en_m.references[0].audio.exists()
    cfg = reg.resolve("ja_f")
    assert cfg.type == "clone"
    assert cfg.references[0].audio.is_absolute() and cfg.references[0].audio.exists()
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


def test_custom_emotions_inherit_the_speaker(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\ndefault_temperature: 0.7\n"
        "voices:\n"
        "  ono_anna:\n    type: custom\n    speaker: ono_anna\n    language: English\n"
        "    default_emotion: neutral\n"
        "    emotions:\n"
        "      neutral: {instruct: Calm and even.}\n"
        "      amused:  {instruct: Happy and amused.}\n"
    )
    reg = load_registry(p)
    # The speaker *is* the voice on a custom entry -- only the instruct varies per emotion.
    assert reg.resolve("ono_anna").speaker == "ono_anna"
    assert reg.resolve("ono_anna", "amused").speaker == "ono_anna"
    assert reg.resolve("ono_anna").instruct == "Calm and even."
    assert reg.resolve("ono_anna", "amused").instruct == "Happy and amused."


def test_clone_emotions_still_do_not_inherit_the_clip(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\ndefault_temperature: 0.7\n"
        "voices:\n"
        "  nicole:\n    type: clone\n    language: English\n"
        "    ref_audio: base.wav\n    ref_text: hi\n"
        "    default_emotion: amused\n"
        "    emotions:\n"
        "      amused: {ref_text: hi}\n"
    )
    with pytest.raises(ValueError, match="nicole:amused"):
        load_registry(p)


def test_bundled_lyrebird_registry_loads():
    from pathlib import Path
    import faster_qwen3_tts
    p = Path(faster_qwen3_tts.__file__).parent / "server_voices" / "voices_lyrebird.yaml"
    reg = load_registry(p)
    cfg = reg.resolve("ono_anna")
    assert (cfg.type, cfg.speaker, cfg.language) == ("custom", "ono_anna", "English")


def test_yaml_clone_voice_yields_exactly_one_reference():
    cfg = load_registry(REAL).resolve("en_m")
    assert len(cfg.references) == 1
    assert cfg.references[0].id == "ref"
    assert cfg.references[0].text.startswith("In today's lesson")


def test_pick_reference_returns_none_for_a_custom_voice():
    from faster_qwen3_tts.voice_registry import pick_reference
    cfg = VoiceConfig("x", "custom", "English", 0.7, speaker="aiden")
    assert pick_reference(cfg) is None


def test_pick_reference_is_deterministic_under_a_seeded_rng():
    import random
    from faster_qwen3_tts.voice_registry import Reference, pick_reference
    refs = tuple(Reference(f"r{i}", Path(f"/tmp/r{i}.wav"), "hi") for i in range(5))
    cfg = VoiceConfig("x", "clone", "English", 0.7, references=refs)
    assert pick_reference(cfg, random.Random(7)) is pick_reference(cfg, random.Random(7))
    assert pick_reference(cfg, random.Random(7)) in refs


def test_pick_reference_of_a_single_element_always_yields_it():
    from faster_qwen3_tts.voice_registry import Reference, pick_reference
    only = Reference("only", Path("/tmp/o.wav"), "hi")
    cfg = VoiceConfig("x", "clone", "English", 0.7, references=(only,))
    assert all(pick_reference(cfg) is only for _ in range(20))


def _character_registry():
    """Two emotions of one character, shaped exactly as characters.py will produce."""
    def cfg(emotion, n):
        refs = tuple(Reference(f"t{i}", Path(f"/tmp/{emotion}{i}.wav"), "hi")
                     for i in range(n))
        return VoiceConfig("anby", "clone", "English", 0.7,
                           references=refs, emotion=emotion)
    voices = {c.key: c for c in (cfg("neutral", 3), cfg("amused", 2))}
    return Registry(24000, voices, {"anby": "neutral"}, frozenset({"anby"}))


def test_emotions_are_the_ten_with_neutral_first():
    assert EMOTIONS[0] == "neutral"
    assert len(EMOTIONS) == 10
    assert set(EMOTIONS) == {"neutral", "amused", "smug", "excited", "impressed",
                             "earnest", "deadpan", "annoyed", "panicked", "confused"}


def test_character_falls_back_to_neutral_for_an_unpopulated_emotion():
    cfg = _character_registry().resolve("anby", "smug")
    assert cfg.emotion == "neutral", "smug has no entry, so neutral stands in"


def test_character_exact_emotion_wins_over_the_fallback():
    assert _character_registry().resolve("anby", "amused").emotion == "amused"


def test_character_still_rejects_an_emotion_outside_the_ten():
    with pytest.raises(KeyError):
        _character_registry().resolve("anby", "furious")


def test_emotive_yaml_voice_does_not_fall_back(tmp_path):
    """The YAML path must keep 400ing. On a custom voice a silent fallback would be
    wrong delivery with no error -- the failure this design exists to avoid."""
    p = tmp_path / "v.yaml"
    p.write_text(EMOTIVE_YAML)          # declares neutral + amused only
    with pytest.raises(KeyError):
        load_registry(p).resolve("nicole", "smug")


def test_exact_key_wins_for_a_persona_emotion_outside_the_ten(tmp_path):
    """Emotive YAML emotion names are persona-defined and deliberately unconstrained."""
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\nvoices:\n  nicole:\n    type: clone\n    language: English\n"
        "    default_emotion: neutral\n    emotions:\n"
        "      neutral:  {ref_audio: n.wav, ref_text: hi}\n"
        "      resigned: {ref_audio: r.wav, ref_text: hi}\n"
    )
    assert load_registry(p).resolve("nicole", "resigned").emotion == "resigned"


def test_merge_combines_both_sources():
    merged = merge(load_registry(REAL), _character_registry())
    assert merged.resolve("en_m").id == "en_m"
    assert merged.resolve("anby", "smug").emotion == "neutral"
    assert "anby" in merged.emotion_fallback_ids


def test_merge_rejects_a_voice_id_defined_by_both_sources(tmp_path):
    """A flat 'anby' and a character 'anby:neutral' do not collide as dict keys, but
    resolve('anby') would silently prefer the flat entry and shadow the character."""
    p = tmp_path / "v.yaml"
    p.write_text("sample_rate: 24000\nvoices:\n"
                 "  anby: {type: clone, language: English, ref_audio: r.wav, ref_text: hi}\n")
    with pytest.raises(ValueError, match="anby"):
        merge(load_registry(p), _character_registry())


def test_merge_rejects_a_sample_rate_mismatch():
    with pytest.raises(ValueError, match="sample_rate"):
        merge(Registry(48000, {}, {}), _character_registry())


def test_unsafe_voice_id_raises_valueerror(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\ndefault_temperature: 0.7\n"
        "voices:\n"
        "  \"x\\x7f\": {type: clone, language: English, ref_audio: r.wav, ref_text: hi}\n"
    )
    with pytest.raises(ValueError, match="voice id"):
        load_registry(p)


def test_unsafe_emotion_name_raises_valueerror(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "sample_rate: 24000\ndefault_temperature: 0.7\n"
        "voices:\n"
        "  nicole:\n    type: clone\n    language: English\n"
        "    default_emotion: neutral\n"
        "    emotions:\n"
        "      neutral: {ref_audio: n.wav, ref_text: hi}\n"
        "      \"amused\\r\\nX-Injected: true\": {ref_audio: a.wav, ref_text: hi}\n"
    )
    with pytest.raises(ValueError, match="emotion name"):
        load_registry(p)


def test_is_safe_header_value_matches_uvicorns_actual_rejection_set():
    assert _is_safe_header_value("plain ascii") is True
    assert _is_safe_header_value("has\ta tab") is True
    for bad in ("\x00", "\x07", "\n", "\r", "\x1f", "\x7f"):
        assert _is_safe_header_value(f"x{bad}y") is False
    assert _is_safe_header_value("あ") is False


def test_del_is_rejected_by_the_new_predicate_but_would_have_passed_the_old_one():
    value = "x\x7fy"
    assert _is_safe_header_value(value) is False

    def _old_predicate(v: str) -> bool:
        if "\r" in v or "\n" in v:
            return False
        try:
            v.encode("latin-1")
        except UnicodeEncodeError:
            return False
        return True

    assert _old_predicate(value) is True
