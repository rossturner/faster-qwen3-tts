# Character Emotion Reference Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ross:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve filesystem-discovered characters whose emotion comes from a randomly chosen reference recording, with a browser page that auditions them over the streaming endpoint and shows timing.

**Architecture:** A character directory tree compiles into the *existing* `Registry` type — same `<id>:<emotion>` keys the YAML loader already produces — so both endpoints, `resolve()`, warmup and the streaming path work with a widening rather than a parallel system. `VoiceConfig` grows from one reference clip to a tuple of them; the route picks one per request and reports which. A new `characters.py` owns discovery; `voice_registry.py` owns the type, the emotion vocabulary and the merge.

**Tech Stack:** Python 3.12, FastAPI, pydantic, PyYAML, soundfile, pytest. Vanilla HTML/CSS/JS with WebAudio for the test page (no build step, no CDN). Playwright MCP for the browser check.

**User decisions (already made):**
- Three characters: `nicole`, `anby`, `billy`; ten hardcoded emotions (neutral, amused, smug, excited, impressed, earnest, deadpan, annoyed, panicked, confused).
- Characters discovered from directory names, never hardcoded; emotions hardcoded.
- Recordings live in-repo, committed, **plain git not LFS**.
- An emotion with no recordings **falls back to neutral** rather than erroring.
- Per-character settings via optional `character.yaml`; absent file means defaults.
- Expect 5-10 recordings per emotion per character.
- Test page uses the **framed streaming endpoint** and displays timing, not just audio.
- `/v1/audio/speech` gains `emotion` too.
- Page exercised with Playwright MCP.
- Registry approach A (compile into the existing registry), not a parallel character system.
- Playwright and GPU end-to-end checks are **gated on real recordings arriving**; no placeholder or synthetic character is committed to satisfy them.

**Spec:** `docs/superpowers/specs/2026-07-31-character-emotion-library-design.md`

---

## Orientation: what this codebase already does

Read these before Task 1. The plan assumes zero prior context.

- `faster_qwen3_tts/voice_registry.py` — loads `voices.yaml` into a frozen `Registry` of frozen `VoiceConfig`. Two entry shapes: **flat** (`en_m: {...}`, key is the bare id) and **emotive** (`nicole: {emotions: {...}}`, flattened to keys `nicole:neutral`). `Registry.defaults` maps a voice id to its default emotion.
- `faster_qwen3_tts/server.py` — `ModelManager` owns the models and serialises all GPU work on **one worker thread**. Warmup pre-bakes one clone prompt per registry entry into `_clone_prompts`, keyed by `cfg.key`. Two endpoints: `/v1/audio/speech` (whole WAV) and `/v1/audio/stream` (framed: `1 byte type | 4 byte big-endian length | payload`, types `0x01` header, `0x02` audio, `0x03` mark, `0x04` error, `0x05` end).
- **Why emotion comes from a clip:** on the ICL clone path `instruct` moves speaking rate only — pitch and energy come from the reference recording, which outvotes it. That is measured, not assumed (`docs/lyrebird-tts-spike-findings.md`).
- **Two registries ship.** `voices.yaml` (media-worker's 12 dubbing voices, Base only) and `voices_lyrebird.yaml` (the `ono_anna` custom voice, loads CustomVoice). Nothing in this plan may regress the dubbing path.
- Run tests with `.venv/bin/python -m pytest`. GPU tests are marked `gpu`; the non-GPU suite is `-m "not gpu"` and currently passes **114 (12 deselected)**. It takes about **4m20s** — budget for that rather than assuming it hangs, and prefer a single test file (`pytest tests/test_characters.py -q`) during a red-green loop, saving the full run for the end of a task.

## File Structure

| File | Responsibility | Task |
| --- | --- | --- |
| `faster_qwen3_tts/voice_registry.py` (modify) | `Reference` type, `VoiceConfig.references`, `pick_reference`, `EMOTIONS`, `emotion_fallback_ids`, fallback in `resolve()`, `merge()`, `DEFAULT_TEMPERATURE` | 1, 2 |
| `faster_qwen3_tts/characters.py` (create) | Filesystem discovery → `Registry`. All skip/warn rules live here. | 3 |
| `faster_qwen3_tts/server.py` (modify) | Reference plumbing, warmup failure visibility, `/v1/voices`, header-frame fields, `emotion` + `X-TTS-*` on `/v1/audio/speech`, static mount | 1, 4, 5, 6, 7, 8 |
| `faster_qwen3_tts/server_static/index.html` (create) | Self-contained audition page: streaming playback + timing panel | 8 |
| `faster_qwen3_tts/cli.py` (modify) | `--characters` flag | 4 |
| `pyproject.toml`, `MANIFEST.in`, `.dockerignore` (modify) | Ship the new assets | 9 |
| `CLAUDE.md` (modify) | Durable documentation | 10 |
| `tests/test_characters.py` (create) | Discovery rules | 3 |
| `tests/test_voice_registry.py` (modify) | Widening, fallback scoping, merge | 1, 2 |
| `tests/test_server.py`, `tests/test_server_stream.py` (modify) | Doubles, `/v1/voices`, headers, header frame | 1, 5, 6, 7 |
| `tests/test_packaging.py` (create) | Package-data covers the new assets | 9 |

---

### Task 1: Reference type and per-request reference selection

**Goal:** A voice carries a tuple of references instead of one clip, and the route picks one per request and passes it to the manager.

This is one task, not three, because the type change cannot land green in pieces: `VoiceConfig` losing `ref_audio`/`ref_text` breaks `server.py` and two test doubles in the same commit.

**Files:**
- Modify: `faster_qwen3_tts/voice_registry.py:15-30` (`VoiceConfig`), `:53-65` (`_build`), add `Reference` and `pick_reference`
- Modify: `faster_qwen3_tts/server.py:98-111` (warmup), `:119-134` (`_synthesize_blocking`, `synthesize`), `:136-170` (`synthesize_stream`), `:217-235` and `:237-268` (routes)
- Test: `tests/test_voice_registry.py:13,16`, `tests/test_server.py:16`, `tests/test_server_stream.py:17-19,66-80,173-187`

**Acceptance Criteria:**
- [ ] `VoiceConfig.references` is a tuple of `Reference(id, audio, text)`; `ref_audio`/`ref_text` no longer exist
- [ ] A flat or emotive YAML clone voice yields exactly one `Reference` with id `"ref"`
- [ ] `pick_reference` returns `None` for a custom voice and a member of `cfg.references` for a clone voice
- [ ] `_clone_prompts` is keyed `(cfg.key, reference.id)`
- [ ] The `ref_text` sent to the model comes from the chosen reference, not the config
- [ ] Every behavioural assertion in the existing tests is unchanged — only field accesses and double signatures move

**Verify:** `.venv/bin/python -m pytest tests/ -m "not gpu" -q` → `114 passed`

**Steps:**

- [ ] **Step 1: Update the registry test to the new shape (it will fail)**

In `tests/test_voice_registry.py`, replace lines 13 and 16:

```python
    assert en_m.references[0].audio.is_absolute() and en_m.references[0].audio.exists()
```
```python
    assert cfg.references[0].audio.is_absolute() and cfg.references[0].audio.exists()
```

Add at the end of the file:

```python
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
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_voice_registry.py -q`
Expected: FAIL — `AttributeError: 'VoiceConfig' object has no attribute 'references'` and `ImportError: cannot import name 'pick_reference'`

- [ ] **Step 3: Add `Reference`, `references` and `pick_reference`**

In `faster_qwen3_tts/voice_registry.py`, add `import random` to the imports, then replace `VoiceConfig` (lines 15-30) with:

```python
@dataclass(frozen=True)
class Reference:
    """One reference recording and its exact transcript."""
    id: str
    audio: Path
    text: str

@dataclass(frozen=True)
class VoiceConfig:
    id: str
    type: str
    language: str
    temperature: float
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    references: tuple = ()
    emotion: Optional[str] = None

    @property
    def key(self) -> str:
        """Registry key: bare id for flat voices, '<id>:<emotion>' for emotive ones."""
        return f"{self.id}:{self.emotion}" if self.emotion else self.id

def pick_reference(cfg: VoiceConfig, rng: Optional[random.Random] = None) -> Optional[Reference]:
    """Choose one of a voice's references. None for custom voices, which have none."""
    if not cfg.references:
        return None
    return (rng or random).choice(cfg.references)
```

Replace the clone branch of `_build` (lines 61-65) with:

```python
    if not spec.get("ref_audio") or not spec.get("ref_text"):
        raise ValueError(f"{label}: clone voice requires 'ref_audio' and 'ref_text'")
    ref = (base_dir / spec["ref_audio"]).resolve()
    return VoiceConfig(vid, "clone", lang, temp,
                       references=(Reference("ref", ref, spec["ref_text"]),),
                       emotion=emotion)
```

- [ ] **Step 4: Run the registry tests**

Run: `.venv/bin/python -m pytest tests/test_voice_registry.py -q`
Expected: PASS

- [ ] **Step 5: Plumb the reference through the manager**

In `faster_qwen3_tts/server.py`, add `pick_reference` to the `voice_registry` import. Replace the prompt-baking loop and warmup generation (lines 98-111) with:

```python
        for cfg in self.registry.voices.values():
            if cfg.type == "clone":
                for ref in cfg.references:
                    self._clone_prompts[(cfg.key, ref.id)] = \
                        self._base.model.create_voice_clone_prompt(
                            ref_audio=str(ref.audio), ref_text=ref.text,
                            x_vector_only_mode=False)
        if self._custom is not None:
            cv = next(v for v in self.registry.voices.values() if v.type == "custom")
            self._custom.generate_custom_voice(text="Warmup.", speaker=cv.speaker,
                                                language=cv.language, max_new_tokens=20)
        any_clone = next((v for v in self.registry.voices.values() if v.type == "clone"), None)
        if any_clone is not None:
            ref = any_clone.references[0]
            self._base.generate_voice_clone(
                text="Warmup.", language=any_clone.language,
                voice_clone_prompt=self._clone_prompts[(any_clone.key, ref.id)],
                ref_text=ref.text, max_new_tokens=20)
```

Replace `_synthesize_blocking` and `synthesize` (lines 119-134) with:

```python
    def _synthesize_blocking(self, cfg: VoiceConfig, reference, text: str,
                             temperature: float, max_new_tokens: int):
        if cfg.type == "custom":
            wavs, _ = self._custom.generate_custom_voice(
                text=text, speaker=cfg.speaker, language=cfg.language,
                instruct=cfg.instruct, temperature=temperature,
                max_new_tokens=max_new_tokens)
        else:
            wavs, _ = self._base.generate_voice_clone(
                text=text, language=cfg.language,
                voice_clone_prompt=self._clone_prompts[(cfg.key, reference.id)],
                ref_text=reference.text,
                xvec_only=False, temperature=temperature, max_new_tokens=max_new_tokens)
        return np.asarray(wavs[0], dtype=np.float32)

    def synthesize(self, cfg: VoiceConfig, reference, text: str, temperature: float,
                   max_new_tokens=None):
        tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        return self._run(self._synthesize_blocking, cfg, reference, text, temperature, tokens)
```

In `synthesize_stream`, change the signature (line 136) to accept `reference` after `cfg`, and the clone branch to use it:

```python
    def synthesize_stream(self, cfg: VoiceConfig, reference, text: str, temperature: float,
                          chunk_size: int, max_new_tokens=None,
                          cancel: Optional[threading.Event] = None,
                          instruct: Optional[str] = None):
```
```python
        else:
            def stream():
                return self._base.generate_voice_clone_streaming(
                    text=text, language=cfg.language,
                    voice_clone_prompt=self._clone_prompts[(cfg.key, reference.id)],
                    ref_text=reference.text, xvec_only=False,
                    temperature=temperature, chunk_size=chunk_size,
                    max_new_tokens=tokens)
```

Leave the docstring's existing paragraphs intact.

- [ ] **Step 6: Pick the reference in both routes**

In the `/v1/audio/speech` route, after `temp = ...` (line 232), add and thread through:

```python
        reference = pick_reference(cfg)
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, manager.synthesize, cfg, reference, text, temp)
```

In the `/v1/audio/stream` route, after `instruct = ...` (line 257), add:

```python
        reference = pick_reference(cfg)
```

and pass it in the `frames()` generator's call (line 267):

```python
                chunks = manager.synthesize_stream(
                    cfg, reference, text, temp, req.chunk_size, cancel=cancel,
                    instruct=instruct)
```

- [ ] **Step 7: Update the two test doubles**

In `tests/test_server.py`, replace `FakeManager.synthesize` (line 16):

```python
    def synthesize(self, cfg, reference, text, temperature, max_new_tokens=None):
        self.calls.append((cfg.id, text))
        return np.zeros(24000, dtype=np.float32)
```

and `CapturingManager.synthesize` (lines 72-74):

```python
        def synthesize(self, cfg, reference, text, temperature, max_new_tokens=None):
            captured["temperature"] = temperature
            return super().synthesize(cfg, reference, text, temperature, max_new_tokens)
```

In `tests/test_server_stream.py`, add `Path` and `Reference` to the imports, then replace `_clone_cfg` (lines 17-19):

```python
def _clone_cfg(emotion=None):
    return VoiceConfig("nicole", "clone", "English", 0.7,
                       references=(Reference("ref", Path("/tmp/x.wav"), "hello"),),
                       emotion=emotion)
```

Replace the `_manager` prompt dict (line 78) with:

```python
    mgr._clone_prompts = {(cfg.key, "ref"): {"fake": "prompt"}}
```

Every `mgr.synthesize_stream(cfg, ...)` call in this file gains the reference as the second argument — use `pick_reference(cfg)`. For example lines 88, 105, 119, 136, 144, 157 become:

```python
    out = list(mgr.synthesize_stream(cfg, pick_reference(cfg), "hi there", 0.7, chunk_size=8))
```

Replace `FakeStreamManager.synthesize_stream` (line 176):

```python
    def synthesize_stream(self, cfg, reference, text, temperature, chunk_size,
                          max_new_tokens=None, cancel=None, instruct=None):
        self.calls.append((cfg.key, text, temperature, chunk_size))
        self.cancel = cancel
        self.instruct = instruct
        self.reference = reference
```

and the two emotive `VoiceConfig` constructions in `_stream_registry` (lines 190-193):

```python
    neutral = VoiceConfig("nicole", "clone", "English", 0.7,
                          references=(Reference("ref", Path("/tmp/n.wav"), "hi"),),
                          emotion="neutral")
    amused = VoiceConfig("nicole", "clone", "English", 0.7,
                         references=(Reference("ref", Path("/tmp/a.wav"), "hi"),),
                         emotion="amused")
```

- [ ] **Step 8: Run the whole non-GPU suite**

Run: `.venv/bin/python -m pytest tests/ -m "not gpu" -q`
Expected: `114 passed` — the same count as before. A different count means an assertion was lost, not just moved.

- [ ] **Step 9: Commit**

```bash
git add faster_qwen3_tts/voice_registry.py faster_qwen3_tts/server.py tests/
git commit -m "feat(registry): a voice carries many references, the request picks one

VoiceConfig holds a tuple of Reference(id, audio, text) instead of a single
ref_audio/ref_text pair; a YAML clone voice is the one-element case. The route
picks per request and the manager receives it, so the prompt cache keys on
(voice_key, reference_id) and ref_text comes from the chosen reference."
```

---

### Task 2: Emotion vocabulary, scoped fallback, and merge

**Goal:** `resolve()` falls back to neutral for characters only, and two registries can be merged safely.

**Files:**
- Modify: `faster_qwen3_tts/voice_registry.py:32-51` (`Registry`), add `EMOTIONS`, `DEFAULT_TEMPERATURE`, `merge`
- Test: `tests/test_voice_registry.py`

**Acceptance Criteria:**
- [ ] `EMOTIONS` is the ten names in source order, `neutral` first
- [ ] Exact-key lookup precedes any `EMOTIONS` check, so a persona-defined emotion name outside the ten still resolves
- [ ] Fallback fires only for ids in `emotion_fallback_ids`
- [ ] An emotive YAML voice asked for an unentered emotion still raises
- [ ] `merge` detects collisions on **voice id**, not dict key
- [ ] `merge` raises on mismatched `sample_rate`

**Verify:** `.venv/bin/python -m pytest tests/test_voice_registry.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_voice_registry.py`:

```python
from faster_qwen3_tts.voice_registry import (
    EMOTIONS, Reference, Registry, merge,
)


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
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_voice_registry.py -q`
Expected: FAIL — `ImportError: cannot import name 'EMOTIONS'`

- [ ] **Step 3: Implement**

In `faster_qwen3_tts/voice_registry.py`, add after the imports:

```python
DEFAULT_TEMPERATURE = 0.7

# Fixed vocabulary for character voices. Characters are discovered from the filesystem;
# their emotions are not -- an unrecognised emotion is a client error, and one of these
# with no recordings falls back to neutral.
EMOTIONS = ("neutral", "amused", "smug", "excited", "impressed",
            "earnest", "deadpan", "annoyed", "panicked", "confused")
```

Replace `Registry` (lines 32-51) with:

```python
@dataclass(frozen=True)
class Registry:
    sample_rate: int
    voices: Dict[str, VoiceConfig]
    defaults: Dict[str, str] = field(default_factory=dict)
    emotion_fallback_ids: frozenset = frozenset()

    def resolve(self, voice_id: str, emotion: Optional[str] = None) -> VoiceConfig:
        if emotion is None:
            if voice_id in self.voices:
                return self.voices[voice_id]
            emotion = self.defaults.get(voice_id)
            if emotion is None:
                raise KeyError(
                    f"Unknown voice id {voice_id!r}. Known: {sorted(self.voices)}")
        key = f"{voice_id}:{emotion}"
        if key in self.voices:
            return self.voices[key]
        # Only characters fall back. A registry-wide rule would make an emotive YAML
        # voice silently substitute its default where it used to 400 -- on a custom
        # voice that is wrong delivery with no error.
        if voice_id in self.emotion_fallback_ids and emotion in EMOTIONS:
            default = self.defaults.get(voice_id)
            if default is not None and f"{voice_id}:{default}" in self.voices:
                return self.voices[f"{voice_id}:{default}"]
        raise KeyError(
            f"Unknown voice/emotion {voice_id!r}/{emotion!r}. "
            f"Known: {sorted(self.voices)}")
```

Change the `default_temp` line (79) to use the constant:

```python
    default_temp = float(data.get("default_temperature", DEFAULT_TEMPERATURE))
```

Append `merge` at the end of the file:

```python
def _voice_ids(reg: Registry) -> set:
    return {cfg.id for cfg in reg.voices.values()} | set(reg.defaults)

def merge(yaml_registry: Registry, character_registry: Registry) -> Registry:
    """Combine the YAML and character sources into one registry.

    Collisions are detected on voice *id*, not dict key: keys are flattened, so a flat
    `anby` and a character's `anby:neutral` never clash as keys -- yet resolve('anby')
    would return the flat entry and silently shadow the character.
    """
    if yaml_registry.sample_rate != character_registry.sample_rate:
        raise ValueError(
            f"sample_rate mismatch: voices.yaml declares "
            f"{yaml_registry.sample_rate}, characters declare "
            f"{character_registry.sample_rate}")
    clash = _voice_ids(yaml_registry) & _voice_ids(character_registry)
    if clash:
        raise ValueError(f"voice id(s) defined by both sources: {sorted(clash)}")
    return Registry(
        yaml_registry.sample_rate,
        {**yaml_registry.voices, **character_registry.voices},
        {**yaml_registry.defaults, **character_registry.defaults},
        yaml_registry.emotion_fallback_ids | character_registry.emotion_fallback_ids)
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/ -m "not gpu" -q`
Expected: all pass, count up by 9

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/voice_registry.py tests/test_voice_registry.py
git commit -m "feat(registry): scoped emotion fallback and source merging

Fallback is gated on emotion_fallback_ids so only characters use it; exact-key
lookup comes first so persona-defined emotion names outside the ten still
resolve. merge() detects collisions on voice id rather than flattened key,
where a flat entry would otherwise silently shadow a character."
```

---

### Task 3: Character discovery

**Goal:** A directory tree becomes a `Registry`, with every malformed thing logged rather than silently dropped.

**Files:**
- Create: `faster_qwen3_tts/characters.py`
- Test: `tests/test_characters.py`

**Acceptance Criteria:**
- [ ] Pairs discovered; reference id is the file stem; ordering deterministic
- [ ] Orphan `.wav`, orphan `.txt`, empty transcript, unreadable audio → skipped **with a warning**
- [ ] A directory whose name is not one of the ten → skipped with a warning
- [ ] `.gitkeep` and other stray files ignored **silently** (they are structural)
- [ ] An emotion with no valid references produces **no** registry entry
- [ ] A character with references but none under `neutral` → skipped with a warning
- [ ] An entirely empty character directory → skipped **quietly**
- [ ] `character.yaml` honoured; absent → defaults; unknown key → `ValueError`
- [ ] Hidden and underscored directories ignored
- [ ] Out-of-range duration warns but still yields a usable reference

**Verify:** `.venv/bin/python -m pytest tests/test_characters.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_characters.py`:

```python
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
    assert "anby" not in caplog.text


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


def test_registry_declares_the_character_for_fallback_and_defaults(tmp_path):
    _pair(tmp_path, "nicole", "neutral", "good")
    reg = load_characters(tmp_path, 0.7)
    assert reg.emotion_fallback_ids == frozenset({"nicole"})
    assert reg.defaults == {"nicole": "neutral"}
    assert reg.sample_rate == 24000


def test_missing_root_raises(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        load_characters(tmp_path / "nope", 0.7)
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_characters.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'faster_qwen3_tts.characters'`

- [ ] **Step 3: Implement the module**

Create `faster_qwen3_tts/characters.py`:

```python
"""Discover character voices from a directory tree.

A character is a directory, an emotion a subdirectory, and the .wav/.txt pairs inside it
interchangeable takes of that emotion. Characters are found by name so a new one needs no
code or config change; the emotion vocabulary is fixed in voice_registry.EMOTIONS.

Everything malformed is logged and skipped rather than silently dropped -- a misspelled
emotion directory would otherwise cost a whole emotion invisibly. The one exception is a
character.yaml with an unknown key, which is fatal: that file is deliberate content, and a
typo in it should not be shrugged off.

Returns the same Registry type the YAML loader returns, so nothing downstream needs to
tell a character from a configured voice.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import soundfile as sf
import yaml

from .voice_registry import EMOTIONS, Reference, Registry, VoiceConfig

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "English"
SAMPLE_RATE = 24000

# A recording outside this range is a quality concern, not a correctness one: a reference
# contributes ~12 tokens/second to the prefill, so even a minute is far inside
# max_seq_len=2048. Prefill overflow is driven by input text length, which MAX_INPUT_CHARS
# already bounds. So this warns and keeps going.
MIN_REFERENCE_SECONDS = 2.0
MAX_REFERENCE_SECONDS = 30.0

_CHARACTER_YAML_KEYS = {"language", "temperature"}


def _read_character_yaml(path: Path, default_temperature: float) -> Tuple[str, float]:
    if not path.is_file():
        return DEFAULT_LANGUAGE, default_temperature
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - _CHARACTER_YAML_KEYS
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {sorted(unknown)}; "
            f"allowed: {sorted(_CHARACTER_YAML_KEYS)}")
    return (data.get("language", DEFAULT_LANGUAGE),
            float(data.get("temperature", default_temperature)))


def _read_references(emotion_dir: Path) -> Tuple[Reference, ...]:
    wav_stems = {p.stem for p in emotion_dir.glob("*.wav")}
    for txt in sorted(emotion_dir.glob("*.txt")):
        if txt.stem not in wav_stems:
            logger.warning("%s: transcript with no matching .wav, skipping", txt)

    references = []
    for wav in sorted(emotion_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            logger.warning("%s: no matching .txt transcript, skipping", wav)
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("%s: empty transcript, skipping", txt)
            continue
        try:
            duration = sf.info(str(wav)).duration
        except Exception as exc:
            logger.warning("%s: unreadable audio (%s), skipping", wav, exc)
            continue
        if duration <= 0:
            logger.warning("%s: zero-length audio, skipping", wav)
            continue
        if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
            logger.warning(
                "%s: %.1fs is outside the %.0f-%.0fs recommended range; using it anyway",
                wav, duration, MIN_REFERENCE_SECONDS, MAX_REFERENCE_SECONDS)
        references.append(Reference(wav.stem, wav.resolve(), text))
    return tuple(references)


def load_characters(root, default_temperature: float,
                    sample_rate: int = SAMPLE_RATE) -> Registry:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"characters root {root} is not a directory")

    voices: Dict[str, VoiceConfig] = {}
    defaults: Dict[str, str] = {}
    fallback_ids = set()

    for char_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cid = char_dir.name
        if cid.startswith((".", "_")):
            continue
        language, temperature = _read_character_yaml(
            char_dir / "character.yaml", default_temperature)

        for sub in sorted(p for p in char_dir.iterdir() if p.is_dir()):
            if sub.name not in EMOTIONS:
                logger.warning(
                    "%s: %r is not one of the known emotions %s, skipping",
                    char_dir, sub.name, list(EMOTIONS))

        found = {}
        for emotion in EMOTIONS:
            emotion_dir = char_dir / emotion
            if not emotion_dir.is_dir():
                continue
            references = _read_references(emotion_dir)
            if references:
                found[emotion] = references

        if not found:
            continue                    # unpopulated, not broken -- nothing to say
        if "neutral" not in found:
            logger.warning(
                "character %r has recordings but none under 'neutral'; skipping it. "
                "neutral is required because it is what every other emotion falls back "
                "to, so this character would fail unpredictably per request.", cid)
            continue

        for emotion, references in found.items():
            cfg = VoiceConfig(cid, "clone", language, temperature,
                              references=references, emotion=emotion)
            voices[cfg.key] = cfg
        defaults[cid] = "neutral"
        fallback_ids.add(cid)

    return Registry(sample_rate, voices, defaults, frozenset(fallback_ids))
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_characters.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/characters.py tests/test_characters.py
git commit -m "feat(characters): discover characters and their emotions from disk

A character is a directory, an emotion a subdirectory, the .wav/.txt pairs
inside interchangeable takes. Everything malformed is logged and skipped -- a
misspelled emotion directory would otherwise cost a whole emotion invisibly.
An emotion with no valid pairs produces no registry entry at all, which is what
makes the neutral fallback reachable rather than a crash."
```

---

### Task 4: Wire the library into the server and CLI

**Goal:** `serve-http --characters` loads a library, alone or merged with `voices.yaml`, and a warmup failure is visible instead of a silent permanent 503.

**Files:**
- Modify: `faster_qwen3_tts/server.py:28` (constants), `:85-117` (`_load_and_warm`, `start_warmup_background`), `:211-215` (`/health`), `:295-301` (`create_app`)
- Modify: `faster_qwen3_tts/cli.py:307-312` (`cmd_serve_http`), `:405-410` (parser)
- Test: `tests/test_server.py`

**Acceptance Criteria:**
- [ ] Neither flag → bundled `voices.yaml`, exactly as today
- [ ] `--voices` only → that file only
- [ ] `--characters` only → characters only; the bundled `voices.yaml` is **not** loaded
- [ ] `--characters` bare → the bundled `server_voices/characters`
- [ ] Both → merged, collisions raise
- [ ] An empty merged registry raises at startup
- [ ] A warmup exception is logged and reported by `/health` rather than leaving it at a bare 503 forever

**Verify:** `.venv/bin/python -m pytest tests/test_server.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
import pytest
from faster_qwen3_tts.server import DEFAULT_CHARACTERS, DEFAULT_VOICES, build_registry


def test_no_flags_loads_the_bundled_voices_yaml():
    reg = build_registry(None, None)
    assert len(reg.voices) == 12
    assert reg.emotion_fallback_ids == frozenset()


def test_characters_only_does_not_load_the_dubbing_voices(tmp_path):
    """The lyrebird box should not bake 12 clone prompts it never serves."""
    import wave
    d = tmp_path / "nicole" / "neutral"
    d.mkdir(parents=True)
    with wave.open(str(d / "a.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 24000 * 5)
    (d / "a.txt").write_text("hello there")

    reg = build_registry(None, tmp_path)
    assert set(reg.voices) == {"nicole:neutral"}
    assert "en_m" not in reg.voices


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
```

Add `error = None` to `FakeManager.__init__` in the same file:

```python
    def __init__(self): self.ready = True; self.error = None; self.calls = []
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -q`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_CHARACTERS'`

- [ ] **Step 3: Implement in `server.py`**

Add next to `DEFAULT_VOICES` (line 28):

```python
DEFAULT_CHARACTERS = Path(__file__).parent / "server_voices" / "characters"
```

Add `DEFAULT_TEMPERATURE` and `merge` to the existing `from .voice_registry import ...` line, and add `from .characters import load_characters` beside it — `server.py` uses relative imports for its own package.

Add above `create_app`:

```python
def build_registry(voices_path, characters_path) -> Registry:
    """Resolve the two flags into one registry.

    The bundled voices.yaml is a default, not a floor: it loads when nothing was asked
    for, or when it was asked for. `--characters` alone means characters alone, so the
    lyrebird deployment does not carry media-worker's dubbing voices.
    """
    if voices_path is None and characters_path is None:
        voices_path = DEFAULT_VOICES
    registry = load_registry(voices_path) if voices_path is not None else None
    if characters_path is not None:
        characters = load_characters(characters_path, DEFAULT_TEMPERATURE)
        registry = characters if registry is None else merge(registry, characters)
    if not registry.voices:
        raise ValueError(f"no voices configured (characters_path={characters_path})")
    return registry
```

Replace `create_app` (lines 295-301) with:

```python
def create_app(voices_path=None, characters_path=None, device="cuda",
               max_new_tokens=DEFAULT_MAX_NEW_TOKENS, warmup=True) -> FastAPI:
    registry = build_registry(voices_path, characters_path)
    manager = ModelManager(registry, device=device, max_new_tokens=max_new_tokens)
    if warmup:
        manager.start_warmup_background()
    return build_app(manager, registry, serve_page=characters_path is not None)
```

Add `self.error = None` to `ModelManager.__init__` (after `self.ready = False`), and replace `start_warmup_background` (lines 115-117) with:

```python
    def start_warmup_background(self):
        def run():
            try:
                self._run(self._load_and_warm)
            except BaseException as exc:
                # Without this the thread dies silently and /health returns a bare 503
                # forever, which reads identically to "still warming".
                logger.exception("warmup failed")
                self.error = f"{type(exc).__name__}: {exc}"
        threading.Thread(target=run, name="tts-warmup", daemon=True).start()
```

Replace `/health` (lines 211-215) with:

```python
    @app.get("/health")
    async def health():
        if getattr(manager, "error", None):
            return JSONResponse({"status": "failed", "error": manager.error},
                                status_code=503)
        if not manager.ready:
            return JSONResponse({"status": "warming"}, status_code=503)
        return {"status": "ok"}
```

Change `build_app`'s signature to accept the page flag (Task 8 uses it; accept and ignore it for now is **not** acceptable — mount it in Task 8):

```python
def build_app(manager, registry: Registry, serve_page: bool = False) -> FastAPI:
```

- [ ] **Step 4: Implement in `cli.py`**

Replace `cmd_serve_http` (lines 307-312):

```python
def cmd_serve_http(args):
    import uvicorn
    from faster_qwen3_tts.server import create_app
    app = create_app(voices_path=args.voices, characters_path=args.characters,
                     device=args.device, max_new_tokens=args.max_new_tokens, warmup=True)
    uvicorn.run(app, host=args.host, port=args.port)
```

Add to the `serve-http` parser, after the `--voices` argument (line 408):

```python
    sp.add_argument("--characters", nargs="?", const="BUNDLED", default=None,
                    help="Character library directory; bare flag uses the bundled one. "
                         "Given alone, voices.yaml is not loaded.")
```

and resolve the sentinel at the top of `cmd_serve_http`:

```python
    from faster_qwen3_tts.server import DEFAULT_CHARACTERS
    if args.characters == "BUNDLED":
        args.characters = DEFAULT_CHARACTERS
```

- [ ] **Step 5: Run**

Run: `.venv/bin/python -m pytest tests/ -m "not gpu" -q`
Expected: all pass

- [ ] **Step 6: Check the flag combinations by hand**

Run: `.venv/bin/faster-qwen3-tts serve-http --help`
Expected: `--characters` appears with `[CHARACTERS]` optional-value syntax

- [ ] **Step 7: Commit**

```bash
git add faster_qwen3_tts/server.py faster_qwen3_tts/cli.py tests/test_server.py
git commit -m "feat(server): --characters loads a library, alone or merged

The bundled voices.yaml is a default rather than a floor -- --characters alone
means characters alone, so the lyrebird deployment does not bake 12 dubbing
prompts it never serves. A warmup exception now surfaces through /health
instead of leaving it at a bare 503 indistinguishable from still warming."
```

---

### Task 5: `GET /v1/voices`

**Goal:** Clients can discover voices, their emotions and how many recordings back each.

**Files:**
- Modify: `faster_qwen3_tts/server.py` (add the route and a describe helper in `build_app`)
- Test: `tests/test_server.py`

**Acceptance Criteria:**
- [ ] A character enumerates all ten `EMOTIONS` with reference counts, `0` where absent
- [ ] An emotive YAML voice lists only its own declared emotion names
- [ ] A flat voice reports `emotions: null`
- [ ] Custom entries report `null` per emotion, not `0` — there are no recordings involved, and `0` would read as "unavailable"
- [ ] `emotion_fallback` distinguishes "falls back to neutral" from "cannot be requested"
- [ ] Answers **before** warmup completes — it reads the registry and needs no GPU

**Verify:** `.venv/bin/python -m pytest tests/test_server.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
from faster_qwen3_tts.voice_registry import EMOTIONS, Reference, Registry, VoiceConfig


def _mixed_registry():
    def char(emotion, n):
        refs = tuple(Reference(f"t{i}", Path(f"/tmp/{emotion}{i}.wav"), "hi")
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
    from pathlib import Path  # noqa: F401 -- used by _mixed_registry
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
```

Add `from pathlib import Path` to the imports at the top of `tests/test_server.py`.

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -k voices -q`
Expected: FAIL — 404, the route does not exist

- [ ] **Step 3: Implement**

Add to `faster_qwen3_tts/server.py`, above `build_app`:

```python
def describe_voices(registry: Registry) -> list:
    """One entry per voice id, in the three shapes a client has to tell apart.

    A character enumerates all ten emotions so the caller can see which ones will fall
    back; an emotive configured voice lists only what it declares; a flat voice has no
    emotion axis at all. Counts are reference recordings, and null for custom entries --
    they have no recordings, and reporting 0 would read as "unavailable".
    """
    by_id: dict = {}
    for cfg in registry.voices.values():
        by_id.setdefault(cfg.id, []).append(cfg)

    described = []
    for vid in sorted(by_id):
        cfgs = by_id[vid]
        first = cfgs[0]
        fallback = vid in registry.emotion_fallback_ids

        def count(cfg):
            return None if cfg.type == "custom" else len(cfg.references)

        if first.emotion is None:
            emotions = None
        elif fallback:
            present = {c.emotion: count(c) for c in cfgs}
            emotions = {e: present.get(e, 0) for e in EMOTIONS}
        else:
            emotions = {c.emotion: count(c) for c in sorted(cfgs, key=lambda c: c.emotion)}

        described.append({
            "id": vid,
            "type": first.type,
            "language": first.language,
            "default_emotion": registry.defaults.get(vid),
            "emotion_fallback": fallback,
            "emotions": emotions,
        })
    return described
```

Add `EMOTIONS` to the `voice_registry` import. Add the route inside `build_app`, after `/health`:

```python
    @app.get("/v1/voices")
    async def voices():
        # Deliberately not gated on readiness: this reads the registry, needs no GPU, and
        # a client should be able to populate its UI during the ~50s warmup.
        return describe_voices(registry)
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_server.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/server.py tests/test_server.py
git commit -m "feat(server): GET /v1/voices for discovery

Three shapes, because there are three kinds of entry: a character enumerates
all ten emotions with recording counts so a caller can see which fall back, an
emotive configured voice lists only its declared names, and a flat voice has no
emotion axis. Not gated on warmup -- it needs no GPU."
```

---

### Task 6: `emotion` on `/v1/audio/speech`

**Goal:** The whole-file endpoint takes an emotion and reports what it actually used.

**Files:**
- Modify: `faster_qwen3_tts/server.py:50-56` (`SpeechRequest`), `:217-235` (route)
- Test: `tests/test_server.py`

**Acceptance Criteria:**
- [ ] `emotion` accepted and resolved; omitting it resolves through `default_emotion`
- [ ] An unknown emotion is a 400
- [ ] `X-TTS-Voice`, `X-TTS-Emotion`, `X-TTS-Requested-Emotion`, `X-TTS-Reference` set
- [ ] Headers whose value would be null are omitted rather than sent empty
- [ ] A fallback is visible by comparing `X-TTS-Emotion` to `X-TTS-Requested-Emotion`

**Verify:** `.venv/bin/python -m pytest tests/test_server.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
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
    assert r.headers["x-tts-reference"].startswith("t")


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
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -k speech -q`
Expected: FAIL — `KeyError: 'x-tts-emotion'`

- [ ] **Step 3: Implement**

Add `emotion` to `SpeechRequest`:

```python
class SpeechRequest(BaseModel):
    input: str
    voice: str
    emotion: Optional[str] = None
    response_format: str = "wav"
    model: str = "qwen3-tts"
    speed: float = 1.0
    temperature: Optional[float] = None
```

Replace the resolve-and-return part of the `/v1/audio/speech` route:

```python
        try:
            cfg = registry.resolve(req.voice, req.emotion)
        except KeyError as e:
            raise HTTPException(400, str(e))
        temp = req.temperature if req.temperature is not None else cfg.temperature
        reference = pick_reference(cfg)
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, manager.synthesize, cfg, reference, text, temp)
        # The body is raw WAV with no envelope, so what actually got used goes in headers.
        # A fallback is otherwise invisible: the audio is simply the wrong emotion.
        headers = {"X-TTS-Voice": cfg.id}
        if cfg.emotion:
            headers["X-TTS-Emotion"] = cfg.emotion
        if req.emotion:
            headers["X-TTS-Requested-Emotion"] = req.emotion
        if reference is not None:
            headers["X-TTS-Reference"] = reference.id
        return Response(content=to_wav_bytes(pcm, manager.sample_rate),
                        media_type="audio/wav", headers=headers)
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_server.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/server.py tests/test_server.py
git commit -m "feat(server): emotion on /v1/audio/speech, reported in headers

The body is raw WAV with no envelope, so the resolved emotion and reference go
in X-TTS-* headers. Without them a fallback is invisible -- the caller just
gets the wrong emotion and no way to tell."
```

---

### Task 7: Report voice, emotion and reference in the stream header frame

**Goal:** A streaming client can see which character, emotion and recording produced the audio, and whether the emotion fell back.

**Files:**
- Modify: `faster_qwen3_tts/server.py:262-264` (header frame)
- Test: `tests/test_server_stream.py`

**Acceptance Criteria:**
- [ ] Header frame carries `voice`, `emotion`, `requested_emotion`, `reference`
- [ ] `requested_emotion` is `null` when the caller omitted `emotion`
- [ ] `reference` is `null` for a custom voice
- [ ] The existing three keys are unchanged, so current clients are unaffected

**Verify:** `.venv/bin/python -m pytest tests/test_server_stream.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server_stream.py`:

```python
def test_stream_header_frame_reports_voice_emotion_and_reference():
    import json
    c, _ = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole",
                                         "emotion": "amused"})
    header = json.loads(_frames(r)[0][1])
    assert header["sample_rate"] == 24000
    assert header["voice"] == "nicole"
    assert header["emotion"] == "amused"
    assert header["requested_emotion"] == "amused"
    assert header["reference"] == "ref"


def test_stream_header_frame_requested_emotion_is_null_when_omitted():
    import json
    c, _ = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})
    header = json.loads(_frames(r)[0][1])
    assert header["requested_emotion"] is None
    assert header["emotion"] == "neutral"


def test_stream_header_frame_reference_is_null_for_a_custom_voice():
    import json
    c, _ = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "preset"})
    header = json.loads(_frames(r)[0][1])
    assert header["reference"] is None
```

Update the existing `test_stream_header_frame_describes_the_audio` to assert only the audio keys, since the frame now carries more:

```python
def test_stream_header_frame_describes_the_audio():
    import json
    c, _ = _stream_client()
    r = c.post("/v1/audio/stream", json={"input": "hello", "voice": "nicole"})
    header = json.loads(_frames(r)[0][1])
    assert header["sample_rate"] == 24000
    assert header["channels"] == 1
    assert header["format"] == "s16le"
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_server_stream.py -k header -q`
Expected: FAIL — `KeyError: 'voice'`

- [ ] **Step 3: Implement**

Replace the header frame in the `/v1/audio/stream` route's `frames()` generator:

```python
            yield encode_json_frame(FRAME_HEADER, {
                "sample_rate": sample_rate, "channels": 1, "format": "s16le",
                "voice": cfg.id, "emotion": cfg.emotion,
                "requested_emotion": req.emotion,
                "reference": reference.id if reference is not None else None})
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/ -m "not gpu" -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add faster_qwen3_tts/server.py tests/test_server_stream.py
git commit -m "feat(stream): header frame reports voice, emotion and reference

Additive JSON keys, so existing clients are unaffected. Comparing emotion to
requested_emotion is how a caller sees the neutral fallback fire; the reference
id is what makes a bad-sounding take diagnosable when selection is random."
```

---

### Task 8: The audition page

**Goal:** A self-contained page that streams a character line, plays it, and shows TTFA, per-chunk timing and playback margin.

**Files:**
- Create: `faster_qwen3_tts/server_static/index.html`
- Modify: `faster_qwen3_tts/server.py` (mount in `build_app` when `serve_page`)
- Test: `tests/test_server.py`

**Acceptance Criteria:**
- [ ] `GET /` serves the page when `serve_page=True`, and 404s when not
- [ ] Page makes no external requests (no CDN, no fonts, no analytics)
- [ ] `AudioContext` is created on the first Generate click, not at page load
- [ ] Scheduling clamps to `currentTime` and marks the chunk as an underrun when it does
- [ ] `prefill_ms` renders blank on chunks after the first, never `undefined`
- [ ] A stream ending without an end frame is reported as a failed generation
- [ ] The resolved reference and any emotion fallback are displayed

**Verify:** `.venv/bin/python -m pytest tests/test_server.py -k page -q` → pass; open the page manually in Task 11

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_server.py -k page -q`
Expected: FAIL — `ImportError: cannot import name 'STATIC_DIR'`

- [ ] **Step 3: Create the page**

Create `faster_qwen3_tts/server_static/index.html`:

```html
<!doctype html>
<meta charset="utf-8">
<title>Character audition</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px;
         max-width: 1000px; }
  h1 { font-size: 18px; margin: 0 0 16px; }
  .row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap;
         margin-bottom: 12px; }
  label { display: block; font-size: 12px; opacity: .7; margin-bottom: 2px; }
  select, input, textarea, button { font: inherit; padding: 6px 8px; }
  textarea { width: 100%; height: 70px; box-sizing: border-box; }
  button { cursor: pointer; }
  button[disabled] { cursor: default; opacity: .5; }
  #status { margin: 12px 0; min-height: 1.5em; }
  .err { color: #c00; }
  .warn { color: #b60; }
  #ttfa { font-size: 28px; font-weight: 600; }
  #ttfa small { font-size: 12px; font-weight: 400; opacity: .7; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; }
  th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid #8883;
           font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  tr.underrun td { background: #c0000018; }
  .wrap { overflow-x: auto; }
</style>

<h1>Character audition</h1>

<div class="row">
  <div><label for="voice">Character</label><select id="voice"></select></div>
  <div><label for="emotion">Emotion</label><select id="emotion"></select></div>
  <div><label for="chunk">Chunk size</label>
       <input id="chunk" type="number" value="4" min="1" max="48" style="width:5em"></div>
  <div><button id="go">Generate</button></div>
  <div><button id="stop" disabled>Stop</button></div>
</div>

<textarea id="text">Haha, no. I have read this same error message four times now and it still says nothing useful.</textarea>

<div id="status"></div>
<div id="ttfa"></div>
<div class="wrap"><table id="chunks"><thead><tr>
  <th>#</th><th>decode ms</th><th>prefill ms</th><th>gap ms</th>
  <th>audio ms</th><th>margin ms</th>
</tr></thead><tbody></tbody></table></div>

<script>
const $ = (id) => document.getElementById(id);
let voices = [], ctl = null, ctx = null;

function setStatus(msg, cls) {
  $('status').className = cls || '';
  $('status').textContent = msg;
}

async function loadVoices() {
  voices = await (await fetch('/v1/voices')).json();
  $('voice').innerHTML = voices
    .map(v => `<option value="${v.id}">${v.id}</option>`).join('');
  onVoiceChange();
}

function onVoiceChange() {
  const v = voices.find(x => x.id === $('voice').value);
  const sel = $('emotion');
  if (!v || !v.emotions) { sel.innerHTML = '<option value="">—</option>'; return; }
  sel.innerHTML = Object.entries(v.emotions).map(([name, count]) => {
    // count null means a custom voice: no recordings involved, always available.
    const falls = v.emotion_fallback && count === 0;
    const label = falls ? `${name} (falls back to neutral)`
                        : count === null ? name : `${name} (${count})`;
    return `<option value="${name}">${label}</option>`;
  }).join('');
}

async function pollHealth() {
  try {
    const r = await fetch('/health');
    if (r.status === 200) { $('go').disabled = false; setStatus('ready'); return; }
    const body = await r.json();
    if (body.status === 'failed') {
      setStatus('server warmup failed: ' + body.error, 'err');
      return;                                  // no point retrying a dead warmup
    }
    setStatus('server warming up…', 'warn');
  } catch (e) {
    setStatus('server unreachable', 'err');
  }
  setTimeout(pollHealth, 2000);
}

function readFrames(reader) {
  // 1 byte type | 4 byte big-endian length | payload
  let buf = new Uint8Array(0);
  return {
    async *[Symbol.asyncIterator]() {
      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        const next = new Uint8Array(buf.length + value.length);
        next.set(buf); next.set(value, buf.length);
        buf = next;
        while (buf.length >= 5) {
          const len = new DataView(buf.buffer, buf.byteOffset, 5).getUint32(1);
          if (buf.length < 5 + len) break;
          const frame = { type: buf[0], payload: buf.slice(5, 5 + len),
                          at: performance.now() };
          buf = buf.slice(5 + len);
          yield frame;
        }
      }
    }
  };
}

function makePlayer() {
  let nextStart = 0, playbackStart = 0, started = false;
  return {
    get playbackStart() { return playbackStart; },
    push(payload) {
      const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
      const n = payload.byteLength / 2;
      const f32 = new Float32Array(n);
      for (let i = 0; i < n; i++) f32[i] = view.getInt16(i * 2, true) / 32768;
      const audio = ctx.createBuffer(1, n, 24000);
      audio.copyToChannel(f32, 0);
      const src = ctx.createBufferSource();
      src.buffer = audio;
      src.connect(ctx.destination);
      const now = ctx.currentTime;
      if (!started) { nextStart = now + 0.02; playbackStart = performance.now();
                      started = true; }
      // Without the clamp a late chunk is scheduled in the past, start() fires it
      // immediately, and the audio overlaps -- hiding the very underrun this page
      // exists to show.
      let underrun = false;
      if (nextStart < now) { nextStart = now; underrun = true; }
      src.start(nextStart);
      nextStart += audio.duration;
      return { underrun, seconds: audio.duration };
    }
  };
}

async function generate() {
  $('go').disabled = true; $('stop').disabled = false;
  $('chunks').tBodies[0].innerHTML = ''; $('ttfa').textContent = '';
  setStatus('generating…');

  // Constructed on the gesture: an ungestured AudioContext is blocked and warns.
  if (!ctx) ctx = new AudioContext();
  await ctx.resume();

  const player = makePlayer();
  ctl = new AbortController();
  const body = { input: $('text').value, voice: $('voice').value,
                 chunk_size: Number($('chunk').value) };
  if ($('emotion').value) body.emotion = $('emotion').value;

  const sent = performance.now();
  let ttfa = null, delivered = 0, prev = null, sawEnd = false, header = null;
  try {
    const res = await fetch('/v1/audio/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal: ctl.signal });
    if (!res.ok) { setStatus(`${res.status}: ${await res.text()}`, 'err'); return; }

    let pendingUnderrun = false;
    for await (const f of readFrames(res.body.getReader())) {
      if (f.type === 1) {
        header = JSON.parse(new TextDecoder().decode(f.payload));
        const fell = header.requested_emotion &&
                     header.requested_emotion !== header.emotion;
        setStatus(`${header.voice} · ${header.emotion}` +
                  (fell ? ` (asked for ${header.requested_emotion}, fell back)` : '') +
                  (header.reference ? ` · reference "${header.reference}"` : ''),
                  fell ? 'warn' : '');
      } else if (f.type === 2) {
        if (ttfa === null) {
          ttfa = f.at - sent;
          $('ttfa').innerHTML = `${ttfa.toFixed(0)} ms ` +
            `<small>time to first audio — varies with the chosen reference, ` +
            `since prefill scales with clip length</small>`;
        }
        const r = player.push(f.payload);
        delivered += r.seconds * 1000;
        pendingUnderrun = r.underrun;
      } else if (f.type === 3) {
        const m = JSON.parse(new TextDecoder().decode(f.payload));
        const gap = prev === null ? 0 : f.at - prev;
        prev = f.at;
        const margin = delivered - (f.at - player.playbackStart);
        const row = $('chunks').tBodies[0].insertRow();
        if (pendingUnderrun) row.className = 'underrun';
        [m.chunk_index,
         m.decode_ms == null ? '' : m.decode_ms.toFixed(1),
         m.prefill_ms == null ? '' : m.prefill_ms.toFixed(1),
         gap.toFixed(0),
         delivered.toFixed(0),
         margin.toFixed(0)].forEach(v => row.insertCell().textContent = v);
      } else if (f.type === 4) {
        const e = JSON.parse(new TextDecoder().decode(f.payload));
        setStatus('generation failed: ' + e.message, 'err');
      } else if (f.type === 5) {
        sawEnd = true;
        const e = JSON.parse(new TextDecoder().decode(f.payload));
        setStatus(`done · ${e.total_audio_ms.toFixed(0)} ms audio · ` +
                  `${e.total_decode_ms.toFixed(0)} ms decode` +
                  (header && header.reference ? ` · reference "${header.reference}"` : ''));
      }
    }
    // The wire contract: no end frame means the generation failed, not that it was short.
    if (!sawEnd && !ctl.signal.aborted) {
      setStatus('stream ended without an end frame — failed generation', 'err');
    }
  } catch (e) {
    if (e.name !== 'AbortError') setStatus('error: ' + e.message, 'err');
  } finally {
    $('go').disabled = false; $('stop').disabled = true; ctl = null;
  }
}

$('voice').addEventListener('change', onVoiceChange);
$('go').addEventListener('click', generate);
$('stop').addEventListener('click', () => { if (ctl) { ctl.abort();
  setStatus('cancelled'); } });
$('go').disabled = true;
loadVoices();
pollHealth();
</script>
```

- [ ] **Step 4: Mount it**

In `faster_qwen3_tts/server.py`, add next to the other path constants:

```python
STATIC_DIR = Path(__file__).parent / "server_static"
```

and inside `build_app`, after the `/v1/voices` route:

```python
    if serve_page:
        # Only when a character library is configured. "Mount if the file exists" would
        # be no guard at all -- the file is committed, so the dubbing deployment would
        # serve an unauthenticated dev page.
        @app.get("/", response_class=HTMLResponse)
        async def page():
            return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
```

Add `HTMLResponse` to the `fastapi.responses` import.

- [ ] **Step 5: Run**

Run: `.venv/bin/python -m pytest tests/ -m "not gpu" -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add faster_qwen3_tts/server_static/index.html faster_qwen3_tts/server.py tests/test_server.py
git commit -m "feat(server): audition page over the streaming endpoint

Self-contained, no CDN. Shows TTFA, per-chunk decode/gap/margin and the
resolved reference. Scheduling clamps to currentTime and flags the chunk --
without the clamp a late chunk plays in the past and overlaps, hiding the
underrun the panel exists to reveal. Mounted only when --characters is given."
```

---

### Task 9: Ship the new assets

**Goal:** The character library, `character.yaml` files and the page are actually in a built wheel, sdist and Docker image.

**Files:**
- Modify: `pyproject.toml:50-51`, `MANIFEST.in`, `.dockerignore`
- Test: `tests/test_packaging.py`

**Acceptance Criteria:**
- [ ] `package-data` names each depth explicitly — setuptools globs do **not** recurse
- [ ] `MANIFEST.in` includes the character tree and the page
- [ ] `.dockerignore`'s `*.html` no longer swallows the page
- [ ] A test fails if any of these regress

**Verify:** `.venv/bin/python -m pytest tests/test_packaging.py -q` → pass

**Steps:**

- [ ] **Step 1: Write the failing test**

Create `tests/test_packaging.py`:

```python
"""Ship checks. None of the character assets or the page are covered by the original
package-data, and setuptools globs do not recurse -- so these must be named per depth.
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
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -q`
Expected: FAIL — `package-data does not cover: [...]`

- [ ] **Step 3: Fix `pyproject.toml`**

Replace lines 50-51:

```toml
[tool.setuptools.package-data]
faster_qwen3_tts = [
    "server_voices/*.yaml",
    "server_voices/*.md",
    "server_voices/refs/*.wav",
    "server_voices/characters/*.md",
    "server_voices/characters/*/*.yaml",
    "server_voices/characters/*/*/*.wav",
    "server_voices/characters/*/*/*.txt",
    "server_static/*.html",
]
```

- [ ] **Step 4: Fix `MANIFEST.in`**

Add after the existing `recursive-include faster_qwen3_tts *.py` line:

```
recursive-include faster_qwen3_tts/server_voices/characters *.wav *.txt *.yaml *.md
recursive-include faster_qwen3_tts/server_static *.html
```

- [ ] **Step 5: Fix `.dockerignore`**

Add immediately after the `*.html` line:

```
!faster_qwen3_tts/server_static/*.html
```

- [ ] **Step 6: Run**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml MANIFEST.in .dockerignore tests/test_packaging.py
git commit -m "build: ship the character library and the audition page

setuptools package-data globs do not recurse, so each depth is named. The
sdist carried only *.py, and .dockerignore's blanket *.html would have dropped
the page from an image while the server's existence check made its absence look
deliberate."
```

---

### Task 10: Durable documentation

**Goal:** `CLAUDE.md` describes the character library, the new endpoint and flag, so the next person does not have to read the spec (which is transient and will be deleted).

**Files:**
- Modify: `CLAUDE.md` (the "Serving as an HTTP API" section)

**Acceptance Criteria:**
- [ ] A "Character reference library" subsection covers the layout, the ten emotions, the neutral fallback and the neutral-required invariant
- [ ] `GET /v1/voices` documented with its three response shapes
- [ ] `emotion` documented on both endpoints, with the `X-TTS-*` headers
- [ ] The stream header-frame table updated with the four new fields
- [ ] `--characters` documented, including that it alone suppresses the bundled `voices.yaml`
- [ ] The audition page documented, including that it is only mounted with `--characters`
- [ ] No reference to the spec or plan files anywhere in `CLAUDE.md`

**Verify:** `grep -c "docs/superpowers" CLAUDE.md` → `0`

**Steps:**

- [ ] **Step 1: Update the frame table**

In `CLAUDE.md`, replace the `0x01` header row of the frame-layout table:

```
| `0x01` header | JSON `{sample_rate, channels, format, voice, emotion, requested_emotion, reference}` |
```

and add below the table:

> `emotion` is what was actually used and `requested_emotion` what was asked for; they
> differ when a character's emotion had no recordings and neutral stood in. `reference`
> names the recording chosen for this request, and is null for custom voices.

- [ ] **Step 2: Add the library section**

Add after the "Voice registry" section:

````markdown
### Character reference library

A third voice source, alongside the two registries: a directory tree where a character is
a directory and an emotion a subdirectory holding interchangeable `.wav`/`.txt` pairs.
Each request picks one pair at random and clones from it. Load it with
`serve-http --characters` (bare flag uses the bundled
`faster_qwen3_tts/server_voices/characters`).

```
characters/nicole/neutral/calm_intro.wav
characters/nicole/neutral/calm_intro.txt
characters/nicole/character.yaml          # optional: language, temperature
```

Characters are discovered by directory name — adding one needs no code or config change.
The ten emotion names are fixed in `voice_registry.EMOTIONS`: neutral, amused, smug,
excited, impressed, earnest, deadpan, annoyed, panicked, confused.

**Emotion has to come from the clip** on this path: `instruct` moves only speaking rate on
an ICL clone, so a reference recording is the only thing that moves pitch and energy. That
is measured, not assumed — see `docs/lyrebird-tts-spike-findings.md`.

Three request outcomes, deliberately distinct:

- an emotion outside the ten is a **400**
- one of the ten with no recordings **falls back to neutral**, reported in the response
- a character with recordings but none under `neutral` is **skipped at startup** — neutral
  is what everything else falls back to, so without it the character would fail
  unpredictably per request

Malformed content is logged and skipped, never silently dropped: orphan `.wav` or `.txt`,
empty transcript, unreadable audio, or a directory whose name is not one of the ten. An
unknown key in `character.yaml` is fatal, because that file is deliberate content.

**`--characters` alone does not load `voices.yaml`.** The bundled registry is a default,
not a floor — it loads when neither flag is given, or when `--voices` is given. This keeps
media-worker's deployment from baking clone prompts for a library it never serves, and
keeps the lyrebird deployment from carrying twelve dubbing voices.

**Warmup bakes one clone prompt per recording**, ~101 ms each (measured, RTX 4090, 1.7B
Base, bf16), so a 150-300 recording library adds 15-30 s to startup on top of the model
load. Pre-baking is deliberate: selection is random, so lazy baking would make early
requests pay ~101 ms on a ~260 ms TTFA budget until the cache happened to fill. A warmup
failure is reported by `/health` as `{"status": "failed", "error": ...}`.

### `GET /v1/voices`

Lists every voice, in the three shapes a client must tell apart. Not gated on warmup — it
reads the registry and needs no GPU, so a UI can populate during startup.

- **character** — all ten emotions with recording counts, `0` meaning it falls back
- **emotive configured voice** — only its own declared emotion names
- **flat voice** — `emotions: null`

`emotion_fallback` distinguishes "falls back to neutral" from "cannot be requested". Counts
are `null` for custom entries, which involve no recordings.

### Audition page

`GET /`, mounted **only when `--characters` is given**, so it never appears on the dubbing
deployment. Self-contained HTML with no external requests. Pick a character and emotion,
type a line, and it streams over `/v1/audio/stream` and plays through WebAudio while
showing TTFA, per-chunk decode time, inter-arrival gap and **playback margin** — delivered
audio minus elapsed time, the quantity that decides whether playback starves. A chunk that
arrives too late to schedule cleanly is flagged as an underrun.
````

- [ ] **Step 3: Document `emotion` on the endpoints**

In the `/v1/audio/speech` section, replace the request line with:

> `{input, voice, emotion?, response_format:"wav", temperature?}` → one complete 24 kHz
> mono 16-bit WAV. The resolved voice, emotion, requested emotion and reference come back
> as `X-TTS-Voice`, `X-TTS-Emotion`, `X-TTS-Requested-Emotion` and `X-TTS-Reference`
> headers — the body has no envelope to carry them, and without them an emotion fallback
> is invisible.

- [ ] **Step 4: Verify no transient references**

Run: `grep -c "docs/superpowers" CLAUDE.md`
Expected: `0`

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the character library, /v1/voices and the audition page"
```

---

## Deferred verification — gated on real recordings

**These are not part of implementation completion.** By explicit decision, no placeholder
or synthetic character is committed, so they cannot run until real `.wav`/`.txt` pairs are
in `faster_qwen3_tts/server_voices/characters/<character>/<emotion>/`. Every unit and
server-level behaviour above is covered without them; what is deferred is the first
end-to-end run.

Run once at least `nicole/neutral` is populated:

**Browser check (Playwright MCP)** against `serve-http --characters` on a free GPU:

- page loads; character and emotion dropdowns populate from `/v1/voices`
- an emotion with zero recordings is labelled as falling back
- Generate yields a TTFA figure, per-chunk rows and an end-frame total
- the resolved reference id is shown, and a fallback is visibly noted
- a rejected request (emotion outside the ten) shows an inline error, not silence
- the browser console is clean

**GPU check** — add to `tests/test_server_gpu.py`:

```python
def test_character_streams_end_to_end(tmp_path):
    from faster_qwen3_tts.server import create_app, DEFAULT_CHARACTERS
    app = create_app(characters_path=DEFAULT_CHARACTERS, warmup=True)
    c = TestClient(app)
    for _ in range(120):
        if c.get("/health").status_code == 200:
            break
        import time; time.sleep(2)
    else:
        pytest.fail("server did not become ready within 240s")
    r = c.post("/v1/audio/stream", json={"input": "Hello there.", "voice": "nicole"})
    assert r.status_code == 200
    types = [t for t, _ in iter_frames(r.content)]
    assert types[0] == FRAME_HEADER and types[-1] == FRAME_END
```

Neither check can tell you whether the audio *sounds* right. That stays a human judgement,
which is the page's actual purpose — the automation covers the plumbing around the ear.

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task: §1 layout → Task 3; §2 emotions
and fallback → Tasks 2, 3; §3 registry → Tasks 1, 2; §4 loader → Task 3; §5 selection and
manager signatures → Task 1; §6 endpoints → Tasks 5, 6, 7, 8; §7 warmup → Tasks 1, 4;
§8 CLI and packaging → Tasks 4, 9; §9 page → Task 8; §10 testing → distributed, with the
gated portion recorded above; §12 risks → no task needed.

**One spec correction found while planning.** §6's example showed an emotive custom voice
reporting `{"neutral": 1, "amused": 1}`. A custom voice has no references, so a literal
count would be `0` — which reads as "unavailable" for an emotion that is perfectly
requestable. The plan reports `null` for custom entries instead, and Task 5's test pins it.

**2. Placeholder scan.** Clean. An earlier draft split one test in Task 3 Step 1 across a
placeholder line and a follow-up snippet; that was rewritten as four plain assertions.
Every step now carries complete, runnable code — no "TBD", no "add validation", no "similar
to Task N".

**3. Type consistency.** `Reference(id, audio, text)` is used identically in Tasks 1, 2, 3
and 5. `pick_reference(cfg, rng=None)` is defined in Task 1 and called in Tasks 1 and 6.
`load_characters(root, default_temperature, sample_rate=24000)` is defined in Task 3 and
called in Task 4. `merge(yaml_registry, character_registry)` is defined in Task 2 and
called in Task 4. `build_app(manager, registry, serve_page=False)` gains its parameter in
Task 4 and uses it in Task 8. `synthesize(cfg, reference, text, temperature,
max_new_tokens=None)` and `synthesize_stream(cfg, reference, text, temperature,
chunk_size, ...)` are fixed in Task 1 and matched by the doubles in the same task.
`describe_voices(registry)` and `build_registry(voices_path, characters_path)` are each
defined and used within one task. `_clone_prompts` is keyed `(cfg.key, reference.id)`
consistently from Task 1 onward.
