"""Load and validate the server voice registry (voices.yaml).

Two entry shapes are supported, and both must stay supported: the flat form
(`voice -> clip`) that media-worker's 12 dubbing voices use, and the emotive form
(`voice -> emotion -> clip`) that lyrebird uses. Emotive entries are flattened at load
time into the same dict, keyed `"<voice>:<emotion>"`, so every consumer -- including
warmup, which pre-bakes one clone prompt per entry -- works unchanged.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

DEFAULT_TEMPERATURE = 0.7

_UNSAFE_HEADER_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _is_safe_header_value(value: str) -> bool:
    """Voice ids, emotion names and reference ids are echoed verbatim into HTTP
    response headers."""
    if _UNSAFE_HEADER_CHARS.search(value):
        return False
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True

# Fixed vocabulary for character voices. Characters are discovered from the filesystem;
# their emotions are not -- an unrecognised emotion is a client error, and one of these
# with no recordings falls back to neutral.
EMOTIONS = ("neutral", "amused", "smug", "excited", "impressed",
            "earnest", "deadpan", "annoyed", "panicked", "confused")

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
    references: tuple[Reference, ...] = ()
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

@dataclass(frozen=True)
class Registry:
    sample_rate: int
    voices: Dict[str, VoiceConfig]
    defaults: Dict[str, str] = field(default_factory=dict)
    emotion_fallback_ids: frozenset[str] = frozenset()

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

def _build(vid: str, spec: Dict[str, Any], base_dir: Path, vtype: str, lang: str,
           temp: float, emotion: Optional[str]) -> VoiceConfig:
    label = f"{vid}:{emotion}" if emotion else vid
    if vtype == "custom":
        if not spec.get("speaker"):
            raise ValueError(f"{label}: custom voice requires 'speaker'")
        return VoiceConfig(vid, "custom", lang, temp, speaker=spec["speaker"],
                           instruct=spec.get("instruct"), emotion=emotion)
    if not spec.get("ref_audio") or not spec.get("ref_text"):
        raise ValueError(f"{label}: clone voice requires 'ref_audio' and 'ref_text'")
    ref = (base_dir / spec["ref_audio"]).resolve()
    return VoiceConfig(vid, "clone", lang, temp,
                       references=(Reference("ref", ref, spec["ref_text"]),),
                       emotion=emotion)

_INHERITABLE_KEYS = {
    # The clip *is* the emotion on a clone voice, so ref_audio/ref_text must not inherit
    # -- an emotion that silently reuses the voice-level clip is two identical emotions.
    "clone": ("instruct",),
    # The speaker *is* the voice on a custom voice; the emotion varies only the instruct.
    "custom": ("speaker", "instruct"),
}

def load_registry(path) -> Registry:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    base_dir = path.parent
    default_temp = float(data.get("default_temperature", DEFAULT_TEMPERATURE))
    voices: Dict[str, VoiceConfig] = {}
    defaults: Dict[str, str] = {}
    for vid, raw in data["voices"].items():
        if not _is_safe_header_value(vid):
            raise ValueError(f"{vid}: voice id is not safe as an HTTP header value")
        vtype = raw.get("type")
        if vtype not in ("custom", "clone"):
            raise ValueError(f"{vid}: type must be custom|clone, got {vtype!r}")
        lang = raw.get("language")
        if not lang:
            raise ValueError(f"{vid}: language is required")
        temp = float(raw.get("temperature", default_temp))

        emotions = raw.get("emotions")
        if emotions is None:
            cfg = _build(vid, raw, base_dir, vtype, lang, temp, None)
            if cfg.key in voices:
                raise ValueError(f"duplicate voice key {cfg.key!r}")
            voices[cfg.key] = cfg
            continue

        default_emotion = raw.get("default_emotion")
        if not default_emotion:
            raise ValueError(f"{vid}: voice with 'emotions' requires 'default_emotion'")
        if default_emotion not in emotions:
            raise ValueError(
                f"{vid}: default_emotion {default_emotion!r} is not one of "
                f"{sorted(emotions)}")
        inherited = {k: v for k in _INHERITABLE_KEYS[vtype]
                     if (v := raw.get(k)) is not None}
        for ename, espec in emotions.items():
            if not _is_safe_header_value(ename):
                raise ValueError(
                    f"{vid}:{ename}: emotion name is not safe as an HTTP header value")
            merged = {**inherited, **espec}
            cfg = _build(vid, merged, base_dir, vtype, lang,
                         float(espec.get("temperature", temp)), ename)
            if cfg.key in voices:
                raise ValueError(f"duplicate voice key {cfg.key!r}")
            voices[cfg.key] = cfg
        defaults[vid] = default_emotion
    return Registry(int(data.get("sample_rate", 24000)), voices, defaults)

def _voice_ids(reg: Registry) -> set[str]:
    return {cfg.id for cfg in reg.voices.values()} | set(reg.defaults)

def merge(yaml_registry: Registry, character_registry: Registry) -> Registry:
    """Combine the YAML and character sources into one registry.

    Collisions are detected on voice *id*, not dict key: keys are flattened, so a flat
    `anby` and a character's `anby:neutral` never clash as keys -- yet resolve('anby')
    would return the flat entry and silently shadow the character.
    """
    if yaml_registry.sample_rate != character_registry.sample_rate:
        raise ValueError(
            f"sample_rate mismatch: the YAML registry declares "
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
