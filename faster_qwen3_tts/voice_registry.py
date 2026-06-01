"""Load and validate the server voice registry (voices.yaml)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import yaml

@dataclass(frozen=True)
class VoiceConfig:
    id: str
    type: str
    language: str
    temperature: float
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[Path] = None
    ref_text: Optional[str] = None

@dataclass(frozen=True)
class Registry:
    sample_rate: int
    voices: Dict[str, VoiceConfig]
    def resolve(self, voice_id: str) -> VoiceConfig:
        if voice_id not in self.voices:
            raise KeyError(f"Unknown voice id {voice_id!r}. Known: {sorted(self.voices)}")
        return self.voices[voice_id]

def load_registry(path) -> Registry:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    base_dir = path.parent
    default_temp = float(data.get("default_temperature", 0.7))
    voices: Dict[str, VoiceConfig] = {}
    for vid, raw in data["voices"].items():
        vtype = raw.get("type")
        if vtype not in ("custom", "clone"):
            raise ValueError(f"{vid}: type must be custom|clone, got {vtype!r}")
        lang = raw.get("language")
        if not lang:
            raise ValueError(f"{vid}: language is required")
        temp = float(raw.get("temperature", default_temp))
        if vtype == "custom":
            if not raw.get("speaker"):
                raise ValueError(f"{vid}: custom voice requires 'speaker'")
            voices[vid] = VoiceConfig(vid, "custom", lang, temp,
                                      speaker=raw["speaker"], instruct=raw.get("instruct"))
        else:
            if not raw.get("ref_audio") or not raw.get("ref_text"):
                raise ValueError(f"{vid}: clone voice requires 'ref_audio' and 'ref_text'")
            ref = (base_dir / raw["ref_audio"]).resolve()
            voices[vid] = VoiceConfig(vid, "clone", lang, temp,
                                      ref_audio=ref, ref_text=raw["ref_text"])
    return Registry(int(data.get("sample_rate", 24000)), voices)
