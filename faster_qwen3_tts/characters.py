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
from typing import Dict, Tuple

import soundfile as sf
import yaml

from .voice_registry import EMOTIONS, Reference, Registry, VoiceConfig, _is_safe_header_value

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
        if not _is_safe_header_value(wav.stem):
            logger.warning(
                "%s: filename is not safe as an HTTP header value, skipping", wav)
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
        if not _is_safe_header_value(cid):
            logger.warning(
                "%s: directory name is not safe as an HTTP header value, skipping",
                char_dir)
            continue
        try:
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
        except OSError as exc:
            # e.g. a directory chmod'd unreadable -- one bad character must not take
            # down the whole registry build at server startup.
            logger.warning("%s: unreadable (%s), skipping", char_dir, exc)
            continue

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
