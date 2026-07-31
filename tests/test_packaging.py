"""Ship checks. None of the character assets or the page were covered by the original
package-data. A bare `*` matches one path segment, so `characters/*.wav` silently matches
nothing when every recording sits two directories deeper -- hence one pattern per depth.
"""
import tomllib
from pathlib import Path

REQUIRED_PACKAGE_DATA = {
    "server_voices/characters/*.md",
    "server_voices/characters/*/*.yaml",
    "server_voices/characters/*/*/*.wav",
    "server_voices/characters/*/*/*.txt",
    "server_static/*.html",
}


def _package_data():
    with open("pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    return set(data["tool"]["setuptools"]["package-data"]["faster_qwen3_tts"])


def test_package_data_declares_every_new_asset_depth():
    missing = REQUIRED_PACKAGE_DATA - _package_data()
    assert not missing, f"package-data does not cover: {sorted(missing)}"


def test_patterns_that_can_match_today_do_match():
    root = Path("faster_qwen3_tts")
    for pattern in ("server_voices/characters/*.md", "server_static/*.html",
                    "server_voices/refs/*.wav", "server_voices/*.yaml"):
        assert list(root.glob(pattern)), f"pattern matches nothing: {pattern}"


def test_manifest_includes_the_character_tree_and_the_page():
    manifest = Path("MANIFEST.in").read_text()
    assert "recursive-include faster_qwen3_tts/server_voices/characters" in manifest
    assert "recursive-include faster_qwen3_tts/server_static" in manifest


def test_dockerignore_does_not_swallow_the_page():
    lines = [ln.strip() for ln in Path(".dockerignore").read_text().splitlines()]
    assert "*.html" in lines, "guard assumes the blanket rule is still there"
    assert "!faster_qwen3_tts/server_static/*.html" in lines
