# tests/test_pronunciations.py
import unicodedata

import pytest

from faster_qwen3_tts.pronunciations import EMPTY, Pronouncer, load_pronunciations


def _p(**entries):
    return Pronouncer(tuple(entries.items()))


def _write(tmp_path, body):
    path = tmp_path / "pronunciations.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize("written", ["Anby", "anby", "ANBY", "AnBy"])
def test_matching_ignores_case_and_the_replacement_is_verbatim(written):
    assert _p(Anby="Anbee").apply(f"Tell {written} now") == "Tell Anbee now"


def test_possessive_is_rewritten():
    assert _p(Anby="Anbee").apply("Anby's log") == "Anbee's log"


@pytest.mark.parametrize("text", ["Anbys", "Banby", "Anbyé", "éAnby"])
def test_latin_letter_adjacency_blocks_the_match(text):
    assert _p(Anby="Anbee").apply(text) == text


def test_accented_latin_inside_a_longer_name_is_not_rewritten():
    # The reason the boundary is Latin-script and not [A-Za-z].
    assert _p(Ana="Anna").apply("Anaïs") == "Anaïs"


def test_cjk_adjacency_still_matches():
    # The reason the boundary is not \b: CJK characters are word characters.
    assert _p(Anby="Anbee").apply("Anbyさん") == "Anbeeさん"


def test_digits_are_not_boundaries():
    assert _p(Anby="Anbee").apply("Anby2") == "Anbee2"


def test_substitution_does_not_cascade():
    # One pass: b is output, never re-fed through the b -> c rule.
    assert Pronouncer((("a", "b"), ("b", "c"))).apply("a") == "b"


def test_longest_key_wins_at_a_shared_offset():
    p = Pronouncer((("Anby", "SHORT"), ("Anby Demara", "LONG")))
    assert p.apply("Anby Demara") == "LONG"


def test_regex_metacharacters_in_a_key_match_literally():
    # Unescaped, "Mr." would match "Mrs" -- the dot matches the s, and the following
    # space satisfies the trailing lookaround.
    assert Pronouncer((("Mr.", "Mister"),)).apply("Mrs Smith") == "Mrs Smith"
    assert Pronouncer((("Mr.", "Mister"),)).apply("Mr. Smith") == "Mister Smith"


@pytest.mark.parametrize("text", ["a İ b", "a ſ b"])
def test_unicode_whose_lowercase_is_not_the_pattern_letter_does_not_raise(text):
    # Verified: 'İ' matches the pattern letter 'i' under IGNORECASE but lowercases to
    # two codepoints, and 'ſ' matches 's' but lowercases to itself -- so neither is in
    # the table after the match. A bare dict subscript would KeyError inside re.sub,
    # i.e. a 500 mid-request. Both are standalone here so the match actually fires;
    # inside a word the lookarounds would block it and the path would go untested.
    p = Pronouncer((("i", "EYE"), ("s", "ESS")))
    assert p.apply(text) == text


def test_empty_pronouncer_is_the_identity():
    assert EMPTY.apply("Anby stays") == "Anby stays"


def test_zero_entries_does_not_match_the_empty_string_everywhere():
    assert Pronouncer(()).apply("a b c") == "a b c"


# "e" + combining acute accent (U+0301), spelled as codepoints rather than a pasted
# glyph so the source is unambiguous about which normalization form it is in.
_DECOMPOSED_WORD = "e\u0301clair"


def test_nfc_key_matches_nfd_input():
    key_nfc = unicodedata.normalize("NFC", _DECOMPOSED_WORD)
    text_nfd = unicodedata.normalize("NFD", _DECOMPOSED_WORD)
    assert Pronouncer(((key_nfc, "ECL"),)).apply(text_nfd) == "ECL"


def test_nfd_key_matches_nfc_input_via_load(tmp_path):
    key_nfd = unicodedata.normalize("NFD", _DECOMPOSED_WORD)
    text_nfc = unicodedata.normalize("NFC", _DECOMPOSED_WORD)
    path = _write(tmp_path, f'pronunciations:\n  "{key_nfd}": "ECL"\n')
    assert load_pronunciations(path).apply(text_nfc) == "ECL"


def test_nfc_and_nfd_forms_of_the_same_key_collide_at_load(tmp_path):
    key_nfc = unicodedata.normalize("NFC", _DECOMPOSED_WORD)
    key_nfd = unicodedata.normalize("NFD", _DECOMPOSED_WORD)
    body = f'pronunciations:\n  "{key_nfc}": "a"\n  "{key_nfd}": "b"\n'
    with pytest.raises(ValueError, match="collide"):
        load_pronunciations(_write(tmp_path, body))


def test_zero_entries_pronouncer_leaves_nfd_input_unnormalized():
    text_nfd = unicodedata.normalize("NFD", _DECOMPOSED_WORD)
    assert Pronouncer(()).apply(text_nfd) == text_nfd


def test_load_reads_entries(tmp_path):
    path = _write(tmp_path, 'pronunciations:\n  Anby: "Anbee"\n')
    assert load_pronunciations(path).apply("Anby") == "Anbee"


@pytest.mark.parametrize("body", ["pronunciations:\n", "pronunciations: {}\n", "{}\n"])
def test_absent_or_empty_mapping_loads_to_an_identity(tmp_path, body):
    assert load_pronunciations(_write(tmp_path, body)).apply("Anby") == "Anby"


def test_missing_file_is_fatal(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_pronunciations(tmp_path / "nope.yaml")


@pytest.mark.parametrize("body,match", [
    ('speakers:\n  a: b\n', "unknown top-level key"),
    ('pronunciations: [1, 2]\n', "must be a mapping"),
    ('pronunciations:\n  Anby: 3\n', "must both be strings"),
    ('pronunciations:\n  "": "x"\n', "empty or has leading/trailing whitespace"),
    ('pronunciations:\n  " Anby ": "x"\n', "empty or has leading/trailing whitespace"),
    ('pronunciations:\n  Anby: ""\n', "empty replacement"),
    ('pronunciations:\n  デマラ: "x"\n', "Latin-script"),
    ('pronunciations:\n  Anby: "a"\n  anby: "b"\n', "collide case-insensitively"),
])
def test_malformed_config_is_fatal(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_pronunciations(_write(tmp_path, body))


def test_invalid_yaml_syntax_is_fatal(tmp_path):
    # Unterminated quote raises yaml.YAMLError, which must be re-raised as ValueError
    with pytest.raises(ValueError, match="invalid YAML"):
        load_pronunciations(_write(tmp_path, 'pronunciations:\n  Anby: "x\n'))


@pytest.mark.parametrize("body", ["42\n", "[1, 2, 3]\n"])
def test_non_mapping_top_level_is_fatal(tmp_path, body):
    # A scalar or list at the top level is not a dict, so the isinstance(data, dict) guard rejects it
    with pytest.raises(ValueError, match="top-level document must be a mapping"):
        load_pronunciations(_write(tmp_path, body))
