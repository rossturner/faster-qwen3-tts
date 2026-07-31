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
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from .stream_frames import (
    FRAME_AUDIO, FRAME_END, FRAME_ERROR, FRAME_HEADER, FRAME_MARK,
    encode_frame, encode_json_frame,
)
from .characters import load_characters
from .voice_registry import (
    DEFAULT_TEMPERATURE, EMOTIONS, Reference, Registry, VoiceConfig, load_registry, merge, pick_reference,
)
from .wav_io import to_pcm16, to_wav_bytes

logger = logging.getLogger(__name__)

DEFAULT_VOICES = Path(__file__).parent / "server_voices" / "voices.yaml"
DEFAULT_CHARACTERS = Path(__file__).parent / "server_voices" / "characters"
STATIC_DIR = Path(__file__).parent / "server_static"
MAX_INPUT_CHARS = 2000
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 4      # 260ms TTFA; worst observed gap 159ms against a 320ms chunk
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
    emotion: Optional[str] = None
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
        self.error = None
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
        self.ready = True
        logger.info("Warmup complete — server ready.")

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

    def _synthesize_blocking(self, cfg: VoiceConfig, reference: Optional[Reference], text: str,
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

    def synthesize(self, cfg: VoiceConfig, reference: Optional[Reference], text: str, temperature: float,
                   max_new_tokens=None):
        tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        return self._run(self._synthesize_blocking, cfg, reference, text, temperature, tokens)

    def synthesize_stream(self, cfg: VoiceConfig, reference: Optional[Reference], text: str, temperature: float,
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
                    voice_clone_prompt=self._clone_prompts[(cfg.key, reference.id)],
                    ref_text=reference.text, xvec_only=False,
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


def build_app(manager, registry: Registry, serve_page: bool = False) -> FastAPI:
    app = FastAPI(title="faster-qwen3-tts server")

    @app.get("/health")
    async def health():
        if getattr(manager, "error", None):
            return JSONResponse({"status": "failed", "error": manager.error},
                                status_code=503)
        if not manager.ready:
            return JSONResponse({"status": "warming"}, status_code=503)
        return {"status": "ok"}

    @app.get("/v1/voices")
    async def voices():
        # Deliberately not gated on readiness: this reads the registry, needs no GPU, and
        # a client should be able to populate its UI during the ~50s warmup.
        return describe_voices(registry)

    if serve_page:
        # Only when a character library is configured. "Mount if the file exists" would
        # be no guard at all -- the file is committed, so the dubbing deployment would
        # serve an unauthenticated dev page.
        @app.get("/", response_class=HTMLResponse)
        async def page():
            return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

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
        reference = pick_reference(cfg)

        cancel = threading.Event()
        sample_rate = manager.sample_rate

        async def frames():
            yield encode_json_frame(FRAME_HEADER, {
                "sample_rate": sample_rate, "channels": 1, "format": "s16le",
                "voice": cfg.id, "emotion": cfg.emotion,
                "requested_emotion": req.emotion,
                "reference": reference.id if reference is not None else None})
            audio_ms = decode_ms = 0.0
            try:
                chunks = manager.synthesize_stream(
                    cfg, reference, text, temp, req.chunk_size, cancel=cancel,
                    instruct=instruct)
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


def create_app(voices_path=None, characters_path=None, device="cuda",
               max_new_tokens=DEFAULT_MAX_NEW_TOKENS, warmup=True) -> FastAPI:
    registry = build_registry(voices_path, characters_path)
    manager = ModelManager(registry, device=device, max_new_tokens=max_new_tokens)
    if warmup:
        manager.start_warmup_background()
    return build_app(manager, registry, serve_page=characters_path is not None)
