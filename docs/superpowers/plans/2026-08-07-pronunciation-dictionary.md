# Pronunciation Dictionary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ross:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the HTTP server an opt-in, case-insensitive substitution dictionary that respells configured words in request text before synthesis, shipping with `Anby` and `Demara`.

**Architecture:** Three outcomes. First, a self-contained `pronunciations.py` holding a validated loader and a single-pass regex matcher, unit-tested without a GPU. Second, the wiring: the bundled YAML, an opt-in `--pronunciations` flag mirroring `--characters`, application on both synthesis endpoints, and integration coverage through both existing server test harnesses. Third, the CLAUDE.md entry documenting the new config file, flag and text stage.

**Tech Stack:** Python 3.12, `re`, PyYAML, FastAPI, pytest, `fastapi.testclient.TestClient`.

**Global Constraints:**
- Boundary character class is Latin script, exactly `A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F`, in both lookarounds. Not `\b`, not `[A-Za-z]`.
- Matching is one `re.sub` pass over one compiled alternation with `re.IGNORECASE`. Never a loop of per-entry `re.sub` — that cascades.
- Keys are `re.escape`d before joining into the alternation.
- The substitution callback uses `dict.get(matched.lower(), matched)`, never `dict[...]`. A bare subscript is a 500 on `İ` or `ſ` input.
- Zero entries must not compile an alternation; `(?:)` matches the empty string everywhere.
- On `/v1/audio/speech` the substitution runs *after* both the empty check and the `MAX_INPUT_CHARS` check. That bound measures raw caller input on both endpoints.
- `strip_stage_directions` stays stream-only. Do not add it to `/v1/audio/speech`.
- Every load failure is a `ValueError`. Nothing in this feature warns-and-skips.

**User decisions (already made):**
- Dictionary is global, in its own file — not per-character, not a block inside `voices.yaml`.
- Loading is opt-in via `--pronunciations`, mirroring `--characters`' bare-flag `"BUNDLED"` sentinel. The dubbing deployment must not inherit it by default.
- Values ship as a first-principles guess (`Anbee`, `Demarra`) with no audition. The risk is accepted and recorded in the spec.
- Applies to both synthesis endpoints, pronunciation only — the existing stage-direction asymmetry stays.
- Non-Latin keys are fatal at load rather than degenerating to substring matches.
- No `X-TTS-*` header or header-frame field reporting that a substitution occurred.

---

### Task 1: The pronunciations module

**Goal:** A `Pronouncer` that respells configured words in a single regex pass, and a loader that rejects every malformed config with `ValueError`.

**Files:**
- Create: `faster_qwen3_tts/pronunciations.py`
- Test: `tests/test_pronunciations.py`

**Acceptance Criteria:**
- [ ] `Anby`, `anby`, `ANBY`, `AnBy` all become the configured replacement verbatim
- [ ] `Anby's` → `Anbee's`; `Anbys`, `Banby`, `Anbyé` and `Anaïs` (key `Ana`) are untouched
- [ ] `Anbyさん` → `Anbeeさん`; `Anby2` → `Anbee2`
- [ ] With `a → b` and `b → c`, applying to `a` yields `b`, not `c`
- [ ] A key containing regex metacharacters matches literally
- [ ] Input containing `İ` or `ſ` never raises
- [ ] `EMPTY.apply` is the identity, and a file with an absent or empty mapping loads to an equivalent `Pronouncer`
- [ ] Each documented malformed config raises `ValueError`

**Verify:** `.venv/bin/python -m pytest tests/test_pronunciations.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pronunciations.py`:

```python
# tests/test_pronunciations.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pronunciations.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'faster_qwen3_tts.pronunciations'`

- [ ] **Step 3: Write the module**

Create `faster_qwen3_tts/pronunciations.py`:

```python
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
        return self._pattern.sub(self._replace, text)


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
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pronunciations.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/pronunciations.py tests/test_pronunciations.py
git commit -m "feat(server): pronunciation substitution table

Qwen3-TTS has no pronunciation control, so respelling the input is the
only lever. One compiled alternation applied in a single pass, so rules
cannot cascade; a Latin-script boundary so the rule neither dies on CJK
adjacency nor fires inside accented Latin names."
```

---

### Task 2: Wire it into the server and CLI

**Goal:** The dictionary loads from an opt-in `--pronunciations` flag and rewrites request text on both synthesis endpoints.

**Files:**
- Create: `faster_qwen3_tts/server_voices/pronunciations.yaml`
- Modify: `faster_qwen3_tts/server.py` (imports and `DEFAULT_PRONUNCIATIONS` near line 31; `build_app` signature line 287; `/v1/audio/speech` lines 317-321; `/v1/audio/stream` line 355; `create_app` lines 423-429)
- Modify: `faster_qwen3_tts/cli.py` (`cmd_serve_http` lines 308-315; `serve-http` parser near line 411)
- Modify: `tests/test_server.py` (`client()` helper, line 27)
- Modify: `tests/test_server_stream.py` (`_stream_client()` helper, line 204)
- Test: `tests/test_server.py`, `tests/test_server_stream.py`

**Acceptance Criteria:**
- [ ] `/v1/audio/speech` hands the model the substituted text
- [ ] `/v1/audio/stream` hands the model the substituted text
- [ ] An input over `MAX_INPUT_CHARS` is rejected on both endpoints on its raw length, not its substituted length
- [ ] `serve-http --pronunciations` parses to the `"BUNDLED"` sentinel and resolves to the bundled file
- [ ] `serve-http` without the flag passes `pronunciations_path=None`
- [ ] The bundled YAML loads and contains `Anby` and `Demara`
- [ ] Existing server tests still pass unchanged

**Verify:** `.venv/bin/python -m pytest tests/test_server.py tests/test_server_stream.py tests/test_pronunciations.py tests/test_packaging.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
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


def test_bundled_pronunciations_file_loads_and_covers_both_names():
    from faster_qwen3_tts.pronunciations import load_pronunciations
    p = load_pronunciations(DEFAULT_PRONUNCIATIONS)
    assert p.apply("Anby Demara") not in ("Anby Demara",)
    assert p.apply("anby") == p.apply("Anby")
```

Append to `tests/test_server_stream.py`:

```python
from faster_qwen3_tts.pronunciations import Pronouncer


def test_stream_applies_pronunciations_before_synthesis():
    c, mgr = _stream_client(pronouncer=Pronouncer((("Anby", "Anbee"),)))
    r = c.post("/v1/audio/stream", json={"input": "Hey Anby!", "voice": "nicole"})
    assert r.status_code == 200
    assert mgr.calls[-1][1] == "Hey Anbee!"


def test_stream_pronunciations_run_after_stage_directions_are_stripped():
    c, mgr = _stream_client(pronouncer=Pronouncer((("Anby", "Anbee"),)))
    c.post("/v1/audio/stream",
           json={"input": "*waves* Anby is here", "voice": "nicole"})
    assert mgr.calls[-1][1] == "Anbee is here"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server.py tests/test_server_stream.py -v`
Expected: FAIL — `client() got an unexpected keyword argument 'pronouncer'`, and `ImportError` on `DEFAULT_PRONUNCIATIONS`

- [ ] **Step 3: Create the bundled dictionary**

Create `faster_qwen3_tts/server_voices/pronunciations.yaml`:

```yaml
# faster_qwen3_tts/server_voices/pronunciations.yaml
#
# Respellings applied to request text before synthesis. Qwen3-TTS has no pronunciation
# control -- no lexicon, no phonemes, no SSML -- so the spelling of the input is the only
# lever, and it has to change letters: capitalisation and punctuation are measured inert.
#
# Loaded only when serve-http is given --pronunciations. Matching is case-insensitive and
# whole-word; the replacement is emitted exactly as written here.
#
# NOT AUDITIONED. These are first-principles guesses at spellings the model will read as
# AN-bee and de-MAH-ra (Japanese アンビー・デマラ, Chinese 安比・德玛拉 / Ānbǐ Démǎlā).
# Nobody has heard the model say either name. If it already says "Anby" correctly then
# this file makes it worse -- only ears can tell, so change these freely.
pronunciations:
  Anby: "Anbee"
  Demara: "Demarra"
```

- [ ] **Step 4: Wire the server**

In `faster_qwen3_tts/server.py`, add the import beside the other package imports (near line 23):

```python
from .pronunciations import EMPTY as NO_PRONUNCIATIONS, Pronouncer, load_pronunciations
```

Add the constant beside `DEFAULT_CHARACTERS` (line 32):

```python
DEFAULT_PRONUNCIATIONS = Path(__file__).parent / "server_voices" / "pronunciations.yaml"
```

Change `build_app` (line 287):

```python
def build_app(manager, registry: Registry, serve_page: bool = False,
              pronouncer: Pronouncer = NO_PRONUNCIATIONS) -> FastAPI:
```

In `/v1/audio/speech`, insert the substitution *after* the existing empty and length
checks, so lines 317-321 become:

```python
        text = req.input.strip()
        if not text:
            raise HTTPException(400, "'input' is empty")
        if len(text) > MAX_INPUT_CHARS:
            raise HTTPException(400, f"'input' exceeds {MAX_INPUT_CHARS} chars; chunk upstream")
        # After the bound, not before: MAX_INPUT_CHARS guards prefill against what the
        # caller sent, and a respelling can lengthen the text.
        text = pronouncer.apply(text)
```

In `/v1/audio/stream`, line 355 becomes:

```python
        text = pronouncer.apply(strip_stage_directions(req.input))
```

Change `create_app` (lines 423-429):

```python
def create_app(voices_path=None, characters_path=None, device="cuda",
               max_new_tokens=DEFAULT_MAX_NEW_TOKENS, warmup=True,
               pronunciations_path=None) -> FastAPI:
    registry = build_registry(voices_path, characters_path)
    # None means no dictionary, matching voices_path/characters_path: the bundled table
    # is opt-in so the dubbing deployment does not inherit lyrebird's respellings.
    pronouncer = (load_pronunciations(pronunciations_path)
                  if pronunciations_path is not None else NO_PRONUNCIATIONS)
    manager = ModelManager(registry, device=device, max_new_tokens=max_new_tokens)
    if warmup:
        manager.start_warmup_background()
    return build_app(manager, registry, serve_page=characters_path is not None,
                     pronouncer=pronouncer)
```

- [ ] **Step 5: Wire the CLI**

In `faster_qwen3_tts/cli.py`, `cmd_serve_http` (lines 308-315) becomes:

```python
def cmd_serve_http(args):
    import uvicorn
    from faster_qwen3_tts.server import (
        DEFAULT_CHARACTERS, DEFAULT_PRONUNCIATIONS, create_app)
    if args.characters == "BUNDLED":
        args.characters = DEFAULT_CHARACTERS
    if args.pronunciations == "BUNDLED":
        args.pronunciations = DEFAULT_PRONUNCIATIONS
    app = create_app(voices_path=args.voices, characters_path=args.characters,
                     device=args.device, max_new_tokens=args.max_new_tokens,
                     pronunciations_path=args.pronunciations, warmup=True)
    uvicorn.run(app, host=args.host, port=args.port)
```

Add the flag to the `serve-http` parser, after the `--characters` argument (line 411):

```python
    sp.add_argument("--pronunciations", nargs="?", const="BUNDLED", default=None,
                    help="Pronunciation table; bare flag uses the bundled one. "
                         "Omitted, no respelling is applied.")
```

- [ ] **Step 6: Thread the pronouncer through both test harnesses**

In `tests/test_server.py`, `client()` (line 27) becomes:

```python
def client(manager=None, pronouncer=None):
    from faster_qwen3_tts.pronunciations import EMPTY
    mgr = manager or FakeManager()
    return TestClient(build_app(mgr, _registry(),
                                pronouncer=pronouncer or EMPTY)), mgr
```

In `tests/test_server_stream.py`, `_stream_client()` (line 204) becomes:

```python
def _stream_client(manager=None, pronouncer=None):
    from faster_qwen3_tts.pronunciations import EMPTY
    mgr = manager or FakeStreamManager()
    return TestClient(build_app(mgr, _stream_registry(),
                                pronouncer=pronouncer or EMPTY)), mgr
```

- [ ] **Step 7: Run the full server suite**

Run: `.venv/bin/python -m pytest tests/test_server.py tests/test_server_stream.py tests/test_pronunciations.py tests/test_packaging.py -v`
Expected: PASS, including every pre-existing test

- [ ] **Step 8: Commit**

```bash
git add faster_qwen3_tts/server.py faster_qwen3_tts/cli.py \
        faster_qwen3_tts/server_voices/pronunciations.yaml \
        tests/test_server.py tests/test_server_stream.py
git commit -m "feat(server): apply pronunciations on both synthesis endpoints

Opt-in via --pronunciations, mirroring --characters' bare-flag sentinel,
so the dubbing deployment does not inherit lyrebird's respellings. The
pass runs after MAX_INPUT_CHARS on both endpoints, so that bound still
measures what the caller sent."
```

---

### Task 3: Document it

**Goal:** CLAUDE.md describes the new text stage, config file and flag, so the next reader learns the dictionary exists before being surprised by it.

**Files:**
- Modify: `CLAUDE.md` (new subsection after "Per-character audio filters", before "Gotchas")

**Acceptance Criteria:**
- [ ] States that Qwen3-TTS has no pronunciation control and respelling is the only lever
- [ ] States the file location, the flag, and that it is opt-in
- [ ] States the matching rules: case-insensitive, whole-word by Latin-script boundary, replacement verbatim, single pass
- [ ] States that both endpoints apply it, that it runs after `MAX_INPUT_CHARS`, and that non-Latin keys and every other malformed entry are fatal
- [ ] Records that the shipped values are unauditioned

**Verify:** `grep -n "pronunciation" CLAUDE.md` → the new section is present

**Steps:**

- [ ] **Step 1: Add the section**

Insert into `CLAUDE.md` after the "Per-character audio filters" section:

```markdown
### Pronunciation dictionary

Qwen3-TTS has no pronunciation control: no G2P frontend, no lexicon, no phoneme or IPA
input, no SSML. The upstream package has no phonemiser and the tokenizer's 33 added
tokens are all plumbing; text goes verbatim into the chat template. So the only lever on
how a word is said is how it is **spelled**, and it has to change letters — typography
(ALL CAPS, ellipses, em-dash) is measured inert.

`server_voices/pronunciations.yaml` is a flat table of respellings, applied to `input` on
**both** synthesis endpoints immediately before synthesis:

```yaml
pronunciations:
  Anby: "Anbee"
  Demara: "Demarra"
```

**Opt-in**, via `serve-http --pronunciations` — bare flag for the bundled table, or a
path. Omitted, nothing is rewritten. Opt-in rather than default so media-worker's dubbing
deployment does not inherit lyrebird's respellings, the same reasoning as `--characters`.

Matching is case-insensitive and the replacement is emitted **verbatim**, so `ANBY`
becomes `Anbee` — preserving the caller's capitalisation would be work spent on something
the model ignores. Whole-word, where "word" is bounded by **Latin script** rather than
`\b` or `[A-Za-z]`: `\b` would not match `Anbyさん`, because CJK characters are word
characters, and a bare ASCII class would rewrite inside `Anaïs`. Both matter — this server
serves Japanese and Korean *and* French, Spanish, German, Portuguese and Italian. Digits
are not boundaries (`Anby2` → `Anbee2`); apostrophes are (`Anby's` → `Anbee's`).

Applied in **one pass** over one compiled alternation, so one rule's output can never be
re-matched by another. Keys are `re.escape`d, and the lookup falls back to the matched
text — under `IGNORECASE`, `İ` matches `i` but lowercases to two codepoints, which a bare
dict subscript would turn into a 500.

The pass runs **after** the `MAX_INPUT_CHARS` check on both endpoints, so that bound still
measures what the caller sent; a respelling may push the synthesised text a few characters
over. On the streaming endpoint it also runs after `strip_stage_directions`, so it never
rewrites inside markup that is about to be deleted.

Every malformed entry is **fatal** at load, like `voices.yaml` and unlike the character
library: empty or non-string keys and values, whitespace-padded keys, case-insensitive
collisions, unknown top-level keys, a missing file. **Non-Latin keys are rejected** rather
than half-supported — the boundary is built from letter lookarounds, so a kana key would
degenerate to a bare substring match and fire inside any longer run. An absent or empty
`pronunciations:` mapping is *not* an error; it is how a deployment turns the table off
without dropping the flag.

**The shipped values are unauditioned.** `Anbee` and `Demarra` are first-principles
guesses at AN-bee and de-MAH-ra (Japanese アンビー・デマラ, Chinese 安比・德玛拉). Nobody has
heard the model say either name, and no test can check it — if it already reads `Anby`
correctly, this makes it worse. Settling it means an audition in the manner of the Billy
filter, varying the reference draw as well as the voice, since two shipped transcripts
(`anby/excited/Galgame_Chapter0_Anbi_05.txt`, `nicole/annoyed/GalGame_Chapter030_Nicole_020_014.txt`)
contain the name and pair it with audio of the actor saying it correctly.
```

- [ ] **Step 2: Verify and commit**

```bash
grep -n "Pronunciation dictionary" CLAUDE.md
git add CLAUDE.md
git commit -m "docs: record the pronunciation dictionary"
```

---

## Self-Review

**Spec coverage.** Module and matching rules → Task 1. Bundled YAML, opt-in loading, both endpoints, the pinned `MAX_INPUT_CHARS` ordering, both test harnesses → Task 2. The CLAUDE.md deliverable named in the spec's Components → Task 3. The spec's out-of-scope items (reference transcripts, no `X-TTS-*` reporting) correctly have no task.

**Placeholders.** None — every code step carries the actual code, and every test step the actual assertions.

**Type consistency.** `Pronouncer`, `EMPTY`, `load_pronunciations` and `apply()` are named identically in Tasks 1 and 2. `server.py` imports `EMPTY` under the alias `NO_PRONUNCIATIONS` for readability at the two default sites; the tests import the unaliased `EMPTY` from `faster_qwen3_tts.pronunciations`, which is the same object. `DEFAULT_PRONUNCIATIONS` is defined in `server.py` in Task 2 and imported from there by both `cli.py` and the tests.
