# faster_qwen3_tts/server.py
"""OpenAI-compatible Qwen3-TTS server: registry-driven voices, warmup-gated health.

Loads Base (1.7B) always; loads CustomVoice only if the registry has a `type: custom`
voice. The shipped registry is all clones, so the server runs Base-only.
"""
from __future__ import annotations
import asyncio, logging, queue, re, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from .stream_frames import (
    FRAME_AUDIO, FRAME_END, FRAME_ERROR, FRAME_HEADER, FRAME_MARK,
    encode_frame, encode_json_frame,
)
from .voice_registry import Registry, VoiceConfig, load_registry
from .wav_io import to_pcm16, to_wav_bytes

logger = logging.getLogger(__name__)

DEFAULT_VOICES = Path(__file__).parent / "server_voices" / "voices.yaml"
MAX_INPUT_CHARS = 2000
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 8      # 335ms TTFA with 407ms of headroom against a slow chunk
MAX_CHUNK_SIZE = 48
STREAM_MEDIA_TYPE = "application/vnd.lyrebird.tts-stream"

_STAGE_DIRECTION = re.compile(r"\*[^*]*\*|\[[^\]]*\]|<[^>]*>")


def strip_stage_directions(text: str) -> str:
    """Remove `*action*`, `[action]` and `<action>` markup before it reaches the model.

    Qwen3-TTS supports no markup: measured, these are ignored where they are harmless and
    spoken aloud where they are not, and which one happens varies between takes of the
    same input. An LLM writing in-character dialogue emits them unprompted, so they are
    stripped rather than trusted. Parentheses are deliberately left alone -- a
    parenthetical aside is usually speech the caller meant to keep.
    """
    return re.sub(r"\s{2,}", " ", _STAGE_DIRECTION.sub(" ", text)).strip()


class SpeechRequest(BaseModel):
    input: str
    voice: str
    response_format: str = "wav"
    model: str = "qwen3-tts"
    speed: float = 1.0
    temperature: Optional[float] = None


class StreamRequest(BaseModel):
    input: str
    voice: str
    emotion: Optional[str] = None
    instruct: Optional[str] = None
    temperature: Optional[float] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE


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
        for cfg in self.registry.voices.values():
            if cfg.type == "clone":
                self._clone_prompts[cfg.key] = self._base.model.create_voice_clone_prompt(
                    ref_audio=str(cfg.ref_audio), ref_text=cfg.ref_text, x_vector_only_mode=False)
        if self._custom is not None:
            cv = next(v for v in self.registry.voices.values() if v.type == "custom")
            self._custom.generate_custom_voice(text="Warmup.", speaker=cv.speaker,
                                                language=cv.language, max_new_tokens=20)
        any_clone = next((v for v in self.registry.voices.values() if v.type == "clone"), None)
        if any_clone is not None:
            self._base.generate_voice_clone(
                text="Warmup.", language=any_clone.language,
                voice_clone_prompt=self._clone_prompts[any_clone.key],
                ref_text=any_clone.ref_text, max_new_tokens=20)
        self.ready = True
        logger.info("Warmup complete — server ready.")

    def start_warmup_background(self):
        threading.Thread(target=lambda: self._run(self._load_and_warm),
                         name="tts-warmup", daemon=True).start()

    def _synthesize_blocking(self, cfg: VoiceConfig, text: str, temperature: float, max_new_tokens: int):
        if cfg.type == "custom":
            wavs, _ = self._custom.generate_custom_voice(
                text=text, speaker=cfg.speaker, language=cfg.language,
                instruct=cfg.instruct, temperature=temperature,
                max_new_tokens=max_new_tokens)
        else:
            wavs, _ = self._base.generate_voice_clone(
                text=text, language=cfg.language,
                voice_clone_prompt=self._clone_prompts[cfg.key], ref_text=cfg.ref_text,
                xvec_only=False, temperature=temperature, max_new_tokens=max_new_tokens)
        return np.asarray(wavs[0], dtype=np.float32)

    def synthesize(self, cfg: VoiceConfig, text: str, temperature: float, max_new_tokens=None):
        tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        return self._run(self._synthesize_blocking, cfg, text, temperature, tokens)

    def synthesize_stream(self, cfg: VoiceConfig, text: str, temperature: float,
                          chunk_size: int, max_new_tokens=None,
                          cancel: Optional[threading.Event] = None,
                          instruct: Optional[str] = None):
        """Yield (pcm, timing) chunks as they decode.

        Generation runs on the single GPU worker thread and pushes into an unbounded
        queue; the caller consumes from this generator. The queue is deliberately
        unbounded: a bounded one would block the producer when a cancelled consumer
        stops reading, pinning the GPU thread on a `put` and stalling the next request.

        Clone voices replay a pre-baked ICL prompt; custom voices pass `instruct` through,
        which is the only path where it moves anything beyond speaking rate.

        Setting `cancel` breaks the loop over the model's generator, which stops decode.
        Merely closing the response is not enough -- the GPU is serialised on one
        thread, so a cancelled batch that decodes to completion delays the next beat.
        `cancel` is optional only for convenience: an internal event stands in when it is
        omitted, so abandoning this generator releases the GPU thread either way.
        """
        tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        if cfg.type == "custom":
            def stream():
                return self._custom.generate_custom_voice_streaming(
                    text=text, speaker=cfg.speaker, language=cfg.language,
                    instruct=instruct, temperature=temperature,
                    chunk_size=chunk_size, max_new_tokens=tokens)
        else:
            def stream():
                return self._base.generate_voice_clone_streaming(
                    text=text, language=cfg.language,
                    voice_clone_prompt=self._clone_prompts[cfg.key],
                    ref_text=cfg.ref_text, xvec_only=False,
                    temperature=temperature, chunk_size=chunk_size,
                    max_new_tokens=tokens)
        stop = cancel if cancel is not None else threading.Event()
        q: queue.Queue = queue.Queue()
        done = object()

        def produce():
            try:
                for chunk, _sr, timing in stream():
                    if stop.is_set():
                        break
                    q.put((np.asarray(chunk, dtype=np.float32), timing))
            except BaseException as exc:            # surfaced to the consumer below
                q.put(exc)
            finally:
                q.put(done)

        future = self._gpu.submit(produce)
        try:
            while True:
                # Checked before draining the queue, not only in the producer: the
                # producer runs ahead, so a queue full of already-decoded chunks would
                # otherwise keep being yielded long after the caller cancelled.
                if stop.is_set():
                    break
                item = q.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Also runs when the caller abandons this generator (client disconnect
            # closes it, or an early break/GC without a caller-supplied `cancel`),
            # which is what guarantees the GPU thread is released.
            stop.set()
            future.result()


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
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(None, manager.synthesize, cfg, text, temp)
        return Response(content=to_wav_bytes(pcm, manager.sample_rate), media_type="audio/wav")

    @app.post("/v1/audio/stream")
    async def stream(req: StreamRequest, request: Request):
        # Everything that can be rejected must be rejected here: once the first frame
        # is written the status code is committed and 4xx is no longer available.
        if not manager.ready:
            raise HTTPException(503, "Model warming up")
        if len(req.input) > MAX_INPUT_CHARS:
            raise HTTPException(400, f"'input' exceeds {MAX_INPUT_CHARS} chars; chunk upstream")
        text = strip_stage_directions(req.input)
        if not text:
            raise HTTPException(400, "'input' is empty")
        if not 1 <= req.chunk_size <= MAX_CHUNK_SIZE:
            raise HTTPException(400, f"'chunk_size' must be 1..{MAX_CHUNK_SIZE}")
        try:
            cfg = registry.resolve(req.voice, req.emotion)
        except KeyError as e:
            raise HTTPException(400, str(e))
        temp = req.temperature if req.temperature is not None else cfg.temperature
        # A free-text instruct overrides the emotion handle's registry string, so the
        # caller can direct a line the persona has no declared handle for.
        instruct = req.instruct if req.instruct is not None else cfg.instruct

        cancel = threading.Event()
        sample_rate = manager.sample_rate

        async def frames():
            yield encode_json_frame(FRAME_HEADER, {
                "sample_rate": sample_rate, "channels": 1, "format": "s16le"})
            audio_ms = decode_ms = 0.0
            try:
                chunks = manager.synthesize_stream(
                    cfg, text, temp, req.chunk_size, cancel=cancel, instruct=instruct)
                async for pcm, timing in iterate_in_threadpool(chunks):
                    if await request.is_disconnected():
                        cancel.set()
                        return
                    audio_ms += len(pcm) / sample_rate * 1000
                    decode_ms += float(timing.get("decode_ms", 0.0))
                    yield encode_frame(FRAME_AUDIO, to_pcm16(pcm))
                    yield encode_json_frame(FRAME_MARK, {
                        "chunk_index": timing.get("chunk_index"),
                        "decode_ms": timing.get("decode_ms"),
                        "prefill_ms": timing.get("prefill_ms"),
                        "audio_ms_so_far": round(audio_ms, 2),
                    })
                yield encode_json_frame(FRAME_END, {
                    "total_audio_ms": round(audio_ms, 2),
                    "total_decode_ms": round(decode_ms, 2)})
            except Exception as exc:
                logger.exception("streaming synthesis failed")
                cancel.set()
                yield encode_json_frame(FRAME_ERROR, {"message": str(exc)})

        return StreamingResponse(frames(), media_type=STREAM_MEDIA_TYPE)

    return app


def create_app(voices_path=DEFAULT_VOICES, device="cuda", max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
               warmup=True) -> FastAPI:
    registry = load_registry(voices_path)
    manager = ModelManager(registry, device=device, max_new_tokens=max_new_tokens)
    if warmup:
        manager.start_warmup_background()
    return build_app(manager, registry)
