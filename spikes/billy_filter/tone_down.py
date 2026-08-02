"""A fresh spread for toning Billy's filter down, one axis at a time.

Uses the SHIPPED filter (faster_qwen3_tts.audio_filter.StreamingChorus), fed in 333 ms
chunks -- the same class and the same chunking as /v1/audio/stream, so what is auditioned
is the production path rather than an offline stand-in.

Clones are generated directly rather than pulled from the server, because the server
already applies Billy's filter and there is no way to ask it for the dry signal.

Every variant is LEVEL-MATCHED. makeup_db compensates the level a particular `amount`
loses and does not track it, so lowering `amount` with the shipped 5.7 dB would make the
quieter settings louder -- and louder reliably wins a blind A/B for reasons that have
nothing to do with the effect. Makeup is therefore re-measured per variant, over the same
clones, and the value is printed so the winner can go straight into character.yaml.

Four emotions, because the filter has only ever been judged on calm conversational lines
and it may well behave differently on a shout.

Usage:
    .venv/bin/python spikes/billy_filter/tone_down.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.audio_filter import ChorusSpec, StreamingChorus

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "faster_qwen3_tts" / "server_voices" / "characters" / "billy"
OUT = Path(__file__).parent / "out" / "toned"
SR = 24000
CHUNK = int(SR * 4 / 12)          # chunk_size=4 at 12 Hz codec rate = 333 ms

LINES = {
    "neutral": "Boss, I've been going over the route again, and there's a shortcut "
               "through the lower level nobody seems to use.",
    "excited": "Whoa, that's the exact move from episode three seventy-two! I cannot "
               "believe you actually pulled it off!",
    "smug": "Oh, I had that covered about ten minutes ago. You're welcome, by the way.",
    "panicked": "Wait, wait, that wasn't supposed to happen! Somebody tell me that "
                "wasn't supposed to happen!",
}

CURRENT = ChorusSpec()            # what ships today

AXES = {
    "A_amount": [("a030", ChorusSpec(amount=0.30)),
                 ("a040", ChorusSpec(amount=0.40)),
                 ("a055_CURRENT", CURRENT)],
    "B_cents": [("c12", ChorusSpec(cents=(12.0, -12.0))),
                ("c18", ChorusSpec(cents=(18.0, -18.0))),
                ("c26_CURRENT", CURRENT)],
    "C_delay": [("d04", ChorusSpec(delays_ms=(4.0, 8.0))),
                ("d06", ChorusSpec(delays_ms=(6.0, 12.0))),
                ("d08_CURRENT", CURRENT)],
}


def stream_filter(x, spec, makeup_db=None):
    """Run through the shipped filter in production-sized chunks."""
    if makeup_db is not None:
        spec = ChorusSpec(spec.cents, spec.delays_ms, spec.amount, spec.window_ms,
                          makeup_db)
    ch = StreamingChorus(SR, spec)
    out = [ch.process(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)]
    out.append(ch.flush())
    return np.concatenate([p for p in out if len(p)])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def save(path, x):
    p = float(np.abs(x).max())
    if p > 1.0:                      # only ever from an unmatched makeup; keep it honest
        print(f"    ! {path.name} peaked at {p:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1.0, 1.0), SR)


def main():
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)

    rng = random.Random(11)
    dry = {}
    for emotion, text in LINES.items():
        refs = sorted((LIB / emotion).glob("*.wav"))
        ref = rng.choice(refs)
        prompt = model.model.create_voice_clone_prompt(
            ref_audio=str(ref), ref_text=ref.with_suffix(".txt").read_text().strip(),
            x_vector_only_mode=False)
        wavs, sr = model.generate_voice_clone(
            text=text, language="English", voice_clone_prompt=prompt, temperature=0.7)
        assert sr == SR
        dry[emotion] = np.asarray(wavs[0], np.float32)
        save(OUT / "00_reference" / f"{emotion}_dry.wav", dry[emotion])
        print(f"  {emotion:9s} {len(dry[emotion])/SR:5.2f}s   ref={ref.stem[-16:]}")
    del model
    torch.cuda.empty_cache()

    print(f"\n{'axis / setting':22s} {'makeup':>8s}  {'level vs dry':>13s}")
    for axis, variants in AXES.items():
        for name, spec in variants:
            # measure the level this setting loses, across all four clones at once
            raw = {e: stream_filter(dry[e], spec, makeup_db=0.0) for e in LINES}
            ratio = np.mean([rms(raw[e]) / rms(dry[e]) for e in LINES])
            makeup_db = -20 * np.log10(ratio)
            outs = {e: stream_filter(dry[e], spec, makeup_db=makeup_db) for e in LINES}
            got = np.mean([20 * np.log10(rms(outs[e]) / rms(dry[e])) for e in LINES])
            for e, y in outs.items():
                save(OUT / axis / f"{e}__{name}.wav", y)
            print(f"  {axis}/{name:14s} {makeup_db:+7.2f}dB {got:+12.2f}dB")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
