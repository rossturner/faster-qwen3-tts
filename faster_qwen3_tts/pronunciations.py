"""Respell configured words in request text before it reaches the model.

Qwen3-TTS has no pronunciation control of any kind -- no G2P frontend, no lexicon, no
phoneme or IPA input, no SSML. Text goes verbatim into the chat template. So the only
lever on how a word is said is how it is spelled, and it has to be a change of *letters*:
spikes/emotion/text_markup.py measured typography (ALL CAPS, ellipses, em-dash) as inert.

The shipped case is the name "Anby" from Zenless Zone Zero, which is AN-bee and which the
model would otherwise be free to read as an-BY. Three characters say it, so the table is
global rather than declared per character.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

TOP_LEVEL_KEYS = {"pronunciations"}

# Latin script, not [A-Za-z] and not \b. Two failure modes are being avoided at once:
# \b would not match "Anbyさん", because CJK characters are word characters, so the rule
# would silently stop working in the Japanese and Korean lines this server serves; a bare
# [A-Za-z] lookaround matches "Anbyé" and would rewrite inside "Anaïs", in the French,
# Spanish, German, Portuguese and Italian lines it also serves.
_LETTER = r"A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F"
_KEY_ALLOWED = re.compile(rf"^[{_LETTER} '\-]+$")


@dataclass(frozen=True)
class Pronouncer:
    """A substitution table plus the single pattern that applies it."""

    entries: Tuple[Tuple[str, str], ...] = ()
    _table: Dict[str, str] = field(default_factory=dict, init=False,
                                   compare=False, repr=False)
    _pattern: Optional[re.Pattern] = field(default=None, init=False,
                                           compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_table", {k.lower(): v for k, v in self.entries})
        if not self.entries:
            # An empty alternation compiles to (?:), which matches the empty string at
            # every position that clears the lookarounds.
            return
        # Longest first so alternatives starting at the same offset resolve to the
        # longest. This is narrower than "longest overall": re is leftmost-first across
        # positions, so with keys "Zenless Zone" and "Zone Zero" the input
        # "Zenless Zone Zero" matches the former and the latter never gets a chance.
        keys = sorted((k for k, _ in self.entries), key=len, reverse=True)
        alternation = "|".join(re.escape(k) for k in keys)
        try:
            pattern = re.compile(
                rf"(?<![{_LETTER}])(?:{alternation})(?![{_LETTER}])", re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"pronunciation keys do not compile: {exc}") from exc
        object.__setattr__(self, "_pattern", pattern)

    def _replace(self, match: re.Match) -> str:
        # .get, never [..]: under IGNORECASE 'İ' matches the pattern letter 'i' but
        # 'İ'.lower() is two codepoints, and 'ſ' matches 's' but lowercases to itself.
        # A bare subscript would raise KeyError inside re.sub, i.e. a 500 mid-request.
        matched = match.group(0)
        return self._table.get(matched.lower(), matched)

    def apply(self, text: str) -> str:
        """One pass. A loop of per-entry re.sub would let one rule's output feed the
        next, which is a silent bug the moment the table has a few entries."""
        if self._pattern is None:
            return text
        return self._pattern.sub(self._replace, unicodedata.normalize("NFC", text))


EMPTY = Pronouncer()


def load_pronunciations(path) -> Pronouncer:
    """Read and validate the table. Every fault is fatal.

    This follows voices.yaml's policy rather than the character library's
    skip-and-warn: filesystem-discovered content gets the lenient treatment, deliberate
    config does not.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"pronunciations file {path} does not exist")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level document must be a mapping")
    unknown = set(data) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level key(s) {sorted(unknown)}; "
            f"allowed: {sorted(TOP_LEVEL_KEYS)}")
    raw = data.get("pronunciations") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 'pronunciations' must be a mapping")

    entries = []
    seen: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"{path}: {key!r} -> {value!r}: key and replacement must both be strings")
        key = unicodedata.normalize("NFC", key)
        value = unicodedata.normalize("NFC", value)
        if not key or key != key.strip():
            raise ValueError(f"{path}: {key!r} is empty or has leading/trailing whitespace")
        if not value.strip():
            raise ValueError(f"{path}: {key!r} has an empty replacement")
        if not _KEY_ALLOWED.match(key):
            raise ValueError(
                f"{path}: {key!r} is not a Latin-script key. The word boundary is built "
                f"from letter lookarounds, so a non-Latin key would match as a bare "
                f"substring and fire inside any longer run.")
        lowered = key.lower()
        if lowered in seen:
            raise ValueError(
                f"{path}: {key!r} and {seen[lowered]!r} collide case-insensitively; "
                f"matching is case-insensitive so the table would be ambiguous")
        seen[lowered] = key
        entries.append((key, value))
    return Pronouncer(tuple(entries))
