"""Experiment 2: is a streaming endpoint necessary for lyrebird's ~1s budget?

Non-streaming RTF is ~3.76, so synthesis costs (duration / 3.76) before any byte
exists. This measures, for beat-realistic texts against a pinned clone prompt:

  non-streaming   wall time to the complete waveform -- which is also time-to-first-byte,
                  since nothing is emitted until it is done.
  streaming       time to first audio chunk (TTFA), per-chunk inter-arrival times, and
                  the realtime margin: audio produced so far minus wall time elapsed.

The realtime margin is the one that decides playback quality. TTFA can look fine while
generation falls behind mid-sentence, which a listener hears as a stutter. A margin that
shrinks as the sentence proceeds is a failure even when TTFA passes.

Usage:
    MODEL_SIZE=1.7B .venv/bin/python spikes/streaming/ttfa_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.voice_registry import load_registry

REPO = Path(__file__).resolve().parents[2]
VOICES = REPO / "faster_qwen3_tts" / "server_voices" / "voices.yaml"
VOICE_ID = "en_f"          # an existing pinned voice; no new design needed here
SAMPLE_RATE = 24000
REPEATS = 3

# Beat-realistic: what the wording LLM would hand Voice for one beat.
TEXTS = {
    "1 sentence": "Okay chat, that boss fight was way closer than I want to admit.",
    "2 sentences": "Okay chat, that boss fight was way closer than I want to admit. "
                   "I genuinely thought we were going to lose the run right there.",
    "3 sentences": "Okay chat, that boss fight was way closer than I want to admit. "
                   "I genuinely thought we were going to lose the run right there. "
                   "Someone in chat said parry and honestly, that saved me.",
}


def load():
    registry = load_registry(VOICES)
    cfg = registry.resolve(VOICE_ID)
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048,
    )
    reference = cfg.references[0]
    prompt = model.model.create_voice_clone_prompt(
        ref_audio=str(reference.audio), ref_text=reference.text, x_vector_only_mode=False)
    return model, prompt, cfg


def warm(model, prompt, cfg) -> None:
    """Capture graphs on both paths before measuring anything."""
    reference = cfg.references[0]
    model.generate_voice_clone(text="Warming up.", language=cfg.language,
                               voice_clone_prompt=prompt, ref_text=reference.text,
                               max_new_tokens=32)
    for _ in model.generate_voice_clone_streaming(
            text="Warming up.", language=cfg.language, voice_clone_prompt=prompt,
            ref_text=reference.text, chunk_size=4, max_new_tokens=32):
        pass


def time_non_streaming(model, prompt, cfg, text: str) -> dict:
    reference = cfg.references[0]
    start = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=text, language=cfg.language, voice_clone_prompt=prompt,
        ref_text=reference.text, xvec_only=False, temperature=0.7)
    elapsed = time.perf_counter() - start
    audio_s = len(wavs[0]) / sr
    return {"first_byte_s": round(elapsed, 4), "audio_s": round(audio_s, 3),
            "rtf": round(audio_s / elapsed, 2)}


def time_streaming(model, prompt, cfg, text: str, chunk_size: int) -> dict:
    reference = cfg.references[0]
    start = time.perf_counter()
    ttfa = None
    arrivals, margins = [], []
    audio_s = 0.0
    for chunk, sr, _timing in model.generate_voice_clone_streaming(
            text=text, language=cfg.language, voice_clone_prompt=prompt,
            ref_text=reference.text, chunk_size=chunk_size, temperature=0.7):
        now = time.perf_counter() - start
        if ttfa is None:
            ttfa = now
        arrivals.append(now)
        audio_s += len(chunk) / sr
        # positive margin = generation is ahead of playback
        margins.append(audio_s - (now - ttfa))
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {
        "ttfa_s": round(ttfa, 4),
        "audio_s": round(audio_s, 3),
        "total_s": round(arrivals[-1], 4),
        "rtf": round(audio_s / arrivals[-1], 2),
        "max_gap_ms": round(max(gaps) * 1000, 1) if gaps else 0.0,
        "min_margin_s": round(min(margins), 3),
        "final_margin_s": round(margins[-1], 3),
    }


def main() -> None:
    model, prompt, cfg = load()
    print("warming up (captures CUDA graphs; not measured)...", flush=True)
    warm(model, prompt, cfg)

    results = []
    for label, text in TEXTS.items():
        print(f"\n=== {label} ({len(text)} chars) ===", flush=True)

        runs = [time_non_streaming(model, prompt, cfg, text) for _ in range(REPEATS)]
        best = min(r["first_byte_s"] for r in runs)
        worst = max(r["first_byte_s"] for r in runs)
        print(f"  non-streaming   first byte {best:.3f}-{worst:.3f}s  "
              f"for {runs[0]['audio_s']:.2f}s of audio  (RTF {runs[0]['rtf']})")
        results.append({"text": label, "mode": "non-streaming", "runs": runs})

        for chunk_size in (2, 4, 8):
            runs = [time_streaming(model, prompt, cfg, text, chunk_size)
                    for _ in range(REPEATS)]
            ttfas = [r["ttfa_s"] for r in runs]
            print(f"  streaming cs={chunk_size:<2}  TTFA {min(ttfas):.3f}-{max(ttfas):.3f}s  "
                  f"max gap {max(r['max_gap_ms'] for r in runs):6.1f}ms  "
                  f"min margin {min(r['min_margin_s'] for r in runs):+.2f}s  "
                  f"(RTF {runs[0]['rtf']})")
            results.append({"text": label, "mode": f"streaming-cs{chunk_size}",
                            "runs": runs})

    out = Path(__file__).parent / "ttfa_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
