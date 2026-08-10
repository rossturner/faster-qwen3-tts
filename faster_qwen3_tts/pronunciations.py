"""Rewrite request text before it reaches the model.

Qwen3-TTS has no pronunciation control of any kind -- no G2P frontend, no lexicon, no
phoneme or IPA input, no SSML. Text goes verbatim into the chat template. So the only
lever on how a word is said is how it is spelled, and it has to be a change of *letters*:
spikes/emotion/text_markup.py measured typography (ALL CAPS, ellipses, em-dash) as inert.

Two rewrites live here, both following from that one measurement, and both behind the
single --pronunciations opt-in:

* Respelling. The shipped case is the name "Anby" from Zenless Zone Zero, which is AN-bee
  and which the model would otherwise be free to read as an-BY. Three characters say it,
  so the table is global rather than declared per character.

* Hesitation fillers. An ellipsis buys no pause -- that is the same inertness finding,
  and it is the one upstream complaint with no maintainer answer (GH discussion #75, HF
  Base discussion #9: newlines, dots, dashes and underscores all reported to make "no
  difference"). What the same spike found *does* work is lexical vocalisations: `Haha,`
  `Ugh,` `Hmm.` were spoken every time. So a medial ellipsis is replaced by an actual
  spoken filler, comma-delimited because commas move prosody and dashes do not.
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

TOP_LEVEL_KEYS = {"pronunciations", "fillers"}

# Latin script, not [A-Za-z] and not \b. Two failure modes are being avoided at once:
# \b would not match "Anbyさん", because CJK characters are word characters, so the rule
# would silently stop working in the Japanese and Korean lines this server serves; a bare
# [A-Za-z] lookaround matches "Anbyé" and would rewrite inside "Anaïs", in the French,
# Spanish, German, Portuguese and Italian lines it also serves.
_LETTER = r"A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F"
_KEY_ALLOWED = re.compile(rf"^[{_LETTER} '\-]+$")

# Only a *medial* ellipsis, with a word character either side, and swallowing the
# whitespace around it so the replacement supplies its own. A leading or trailing one is
# left alone: ", uh," dangling off the end of a line is worse than the nothing an
# ellipsis already does, and since the mark is inert, every miss here is a no-op rather
# than a regression. That asymmetry is why the boundary is deliberately narrow -- it also
# means `Wait...!` and `the... "other thing"` go unrewritten, which is the safe direction.
# A run of two or more dots, or a single U+2026, or any mix of the two. One bare dot is
# excluded so "the. other" and "3.14" are untouched.
_ELLIPSIS = re.compile(r"(?<=\w)\s*(?:[.\u2026]{2,}|\u2026)\s*(?=\w)")


@dataclass(frozen=True)
class Pronouncer:
    """A substitution table plus the single pattern that applies it, and the filler list."""

    entries: Tuple[Tuple[str, str], ...] = ()
    fillers: Tuple[str, ...] = ()
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

    def _fill(self, _match: re.Match, rng) -> str:
        return f", {rng.choice(self.fillers)}, "

    def apply(self, text: str, rng: Optional[random.Random] = None) -> str:
        """Respell, then fill. Each is one pass -- a loop of per-entry re.sub would let
        one rule's output feed the next, which is a silent bug the moment the table has a
        few entries.

        Respelling runs first so an inserted filler can never be caught by a caller's
        respelling rule. The reverse cannot happen: keys are Latin letters, spaces,
        apostrophes and hyphens, so no replacement can produce an ellipsis.
        """
        if self._pattern is None and not self.fillers:
            return text
        text = unicodedata.normalize("NFC", text)
        if self._pattern is not None:
            text = self._pattern.sub(self._replace, text)
        if self.fillers:
            chooser = rng or random
            text = _ELLIPSIS.sub(lambda m: self._fill(m, chooser), text)
        return text


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
    return Pronouncer(tuple(entries), _load_fillers(path, data))


def _load_fillers(path: Path, data: dict) -> Tuple[str, ...]:
    """Read the filler list. Absent or empty is how a deployment turns fillers off
    without dropping the --pronunciations flag, exactly as for 'pronunciations'."""
    raw = data.get("fillers") or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'fillers' must be a list")
    fillers: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"{path}: filler {item!r} must be a string")
        item = unicodedata.normalize("NFC", item)
        if not item or item != item.strip():
            raise ValueError(
                f"{path}: filler {item!r} is empty or has leading/trailing whitespace")
        # Not checked against _KEY_ALLOWED: a Japanese or Korean deployment wants
        # "えっと" or "어", and unlike a respelling key this is emitted, never matched, so
        # the letter-lookaround reasoning that forces Latin keys does not apply.
        fillers.append(item)
    # Duplicates are deliberately allowed: repeating an entry is how the list weights the
    # draw, since the choice is uniform over it.
    return tuple(fillers)
