# faster_qwen3_tts/server.py
"""OpenAI-compatible Qwen3-TTS server: 14 voices, dual-model, warmup-gated health."""
from __future__ import annotations
import asyncio, logging, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .voice_registry import Registry, VoiceConfig, load_registry
from .wav_io import to_wav_bytes

logger = logging.getLogger(__name__)

DEFAULT_VOICES = Path(__file__).parent / "server_voices" / "voices.yaml"
MAX_INPUT_CHARS = 2000
DEFAULT_MAX_NEW_TOKENS = 1024


class SpeechRequest(BaseModel):
    input: str
    voice: str
    response_format: str = "wav"
    model: str = "qwen3-tts"
    speed: float = 1.0
    temperature: Optional[float] = None


class ModelManager:
    """Owns both models; serializes all GPU work on one worker thread."""

    def __init__(self, registry: Registry, device="cuda", max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        self.registry = registry
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.sample_rate = registry.sample_rate
        self.ready = False
        self._custom = None
        self._base = None
        self._clone_prompts: dict = {}
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpu")

    def _run(self, fn, *a, **k):
        return self._gpu.submit(fn, *a, **k).result()

    def _load_and_warm(self):
        import torch
        from faster_qwen3_tts import FasterQwen3TTS
        needs_custom = any(v.type == "custom" for v in self.registry.voices.values())
        if needs_custom:
            logger.info("Loading CustomVoice...")
            self._custom = FasterQwen3TTS.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device=self.device,
                dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048)
        logger.info("Loading Base...")
        self._base = FasterQwen3TTS.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device=self.device,
            dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048)
        for vid, cfg in self.registry.voices.items():
            if cfg.type == "clone":
                self._clone_prompts[vid] = self._base.model.create_voice_clone_prompt(
                    ref_audio=str(cfg.ref_audio), ref_text=cfg.ref_text, x_vector_only_mode=False)
        if self._custom is not None:
            cv = next(v for v in self.registry.voices.values() if v.type == "custom")
            self._custom.generate_custom_voice(text="Warmup.", speaker=cv.speaker,
                                                language=cv.language, max_new_tokens=20)
        any_clone = next((v for v in self.registry.voices.values() if v.type == "clone"), None)
        if any_clone is not None:
            self._base.generate_voice_clone(
                text="Warmup.", language=any_clone.language,
                voice_clone_prompt=self._clone_prompts[any_clone.id],
                ref_text=any_clone.ref_text, max_new_tokens=20)
        self.ready = True
        logger.info("Warmup complete — server ready.")

    def start_warmup_background(self):
        threading.Thread(target=lambda: self._run(self._load_and_warm),
                         name="tts-warmup", daemon=True).start()

    def _synthesize_blocking(self, cfg: VoiceConfig, text: str, temperature: float):
        if cfg.type == "custom":
            wavs, _ = self._custom.generate_custom_voice(
                text=text, speaker=cfg.speaker, language=cfg.language,
                instruct=cfg.instruct, temperature=temperature,
                max_new_tokens=self.max_new_tokens)
        else:
            wavs, _ = self._base.generate_voice_clone(
                text=text, language=cfg.language,
                voice_clone_prompt=self._clone_prompts[cfg.id], ref_text=cfg.ref_text,
                xvec_only=False, temperature=temperature, max_new_tokens=self.max_new_tokens)
        return np.asarray(wavs[0], dtype=np.float32)

    def synthesize(self, cfg: VoiceConfig, text: str, temperature: float, max_new_tokens=None):
        return self._run(self._synthesize_blocking, cfg, text, temperature)


def build_app(manager, registry: Registry) -> FastAPI:
    app = FastAPI(title="faster-qwen3-tts server")

    @app.get("/health")
    async def health():
        if not manager.ready:
            return JSONResponse({"status": "warming"}, status_code=503)
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        if not manager.ready:
            raise HTTPException(503, "Model warming up")
        text = req.input.strip()
        if not text:
            raise HTTPException(400, "'input' is empty")
        if len(text) > MAX_INPUT_CHARS:
            raise HTTPException(400, f"'input' exceeds {MAX_INPUT_CHARS} chars; chunk upstream")
        if req.response_format.lower() != "wav":
            raise HTTPException(400, "only response_format='wav' is supported")
        try:
            cfg = registry.resolve(req.voice)
        except KeyError as e:
            raise HTTPException(400, str(e))
        temp = req.temperature if req.temperature is not None else cfg.temperature
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, manager.synthesize, cfg, text, temp)
        return Response(content=to_wav_bytes(pcm, manager.sample_rate), media_type="audio/wav")

    return app


def create_app(voices_path=DEFAULT_VOICES, device="cuda", max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
               warmup=True) -> FastAPI:
    registry = load_registry(voices_path)
    manager = ModelManager(registry, device=device, max_new_tokens=max_new_tokens)
    if warmup:
        manager.start_warmup_background()
    return build_app(manager, registry)
