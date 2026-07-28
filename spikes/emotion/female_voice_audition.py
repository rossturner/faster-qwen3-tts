"""Audition the four built-in female voices for emotional range, all speaking English.

CustomVoice is the only path where `instruct` demonstrably moves pitch and loudness
(spikes/emotion/instruct_inertness.py). This renders every built-in female voice across
three sentences x three emotions so the range can be judged by ear, not by metric.

Three of the four voices are native to Chinese/Japanese/Korean; all are forced to
English here, because that is the language the persona needs and the accent they carry
doing it is part of what is being judged.

Usage:
    .venv/bin/python spikes/emotion/female_voice_audition.py
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from instruct_inertness import measure  # noqa: E402

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

OUT = Path(__file__).parent / "female_voices"
LANGUAGE = "English"
TEMPERATURE = 0.7
GAP_SECONDS = 0.5

VOICES = {
    "vivian": "Chinese - bright, slightly edgy young female",
    "serena": "Chinese - warm, gentle young female",
    "ono_anna": "Japanese - playful, light and nimble",
    "sohee": "Korean - warm, rich emotion",
}

SENTENCES = {
    "s1": "Okay, so I just spent the entire afternoon trying to get this thing to work.",
    "s2": "I genuinely cannot believe that just happened, and I have no idea what to say.",
    "s3": "Right, let's take a look at this together and see where it takes us.",
}

EMOTIONS = {
    "excited": "Speak with bright, excited energy, quick and animated, with a real lift "
               "in the voice, like you cannot wait to get the words out.",
    "sad": "Speak sadly and quietly, slow and subdued, with a low, heavy, downcast "
           "delivery and the energy drained out of it.",
    "amused": "Speak with warm amusement, light and playful and teasing, like you are "
              "holding back a laugh the whole way through.",
}


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    print("Loading CustomVoice 1.7B...", flush=True)
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)
    model.generate_custom_voice(text="Warmup.", speaker="serena", language=LANGUAGE,
                                max_new_tokens=20)

    manifest = {}
    for speaker, blurb in VOICES.items():
        print(f"\n{speaker}  ({blurb})", flush=True)
        tour, sr = [], None
        manifest[speaker] = {}
        for emotion, instruct in EMOTIONS.items():
            manifest[speaker][emotion] = {}
            for sid, text in SENTENCES.items():
                wavs, sr = model.generate_custom_voice(
                    text=text, speaker=speaker, language=LANGUAGE,
                    instruct=instruct, temperature=TEMPERATURE)
                audio = np.asarray(wavs[0], dtype=np.float32)
                write_wav(OUT / speaker / f"{emotion}_{sid}.wav", audio, sr)
                manifest[speaker][emotion][sid] = measure(audio, sr, len(text))
                tour.append(audio)
                tour.append(np.zeros(int(GAP_SECONDS * sr), dtype=np.float32))
            m = manifest[speaker][emotion]
            print(f"    {emotion:<9} "
                  f"dur {np.mean([m[s]['duration_s'] for s in SENTENCES]):5.2f}s   "
                  f"pitch {np.nanmean([m[s]['f0_median'] for s in SENTENCES]):6.1f}Hz   "
                  f"loudness {np.mean([m[s]['rms'] for s in SENTENCES]):.4f}", flush=True)
        write_wav(OUT / f"{speaker}_tour.wav", np.concatenate(tour), sr)

    (OUT / "manifest.json").write_text(json.dumps(
        {"language": LANGUAGE, "temperature": TEMPERATURE, "voices": VOICES,
         "sentences": SENTENCES, "emotions": EMOTIONS, "measurements": manifest}, indent=2))

    print("\nRange per voice (spread across the three emotions, averaged over sentences):")
    print(f"  {'voice':<10} {'pitch range':>12} {'loudness range':>16} {'pace range':>12}")
    for speaker, by_emotion in manifest.items():
        def spread(key):
            per = [np.nanmean([by_emotion[e][s][key] for s in SENTENCES]) for e in EMOTIONS]
            return max(per) - min(per)
        print(f"  {speaker:<10} {spread('f0_median'):>10.1f}Hz "
              f"{spread('rms'):>16.4f} {spread('chars_per_s'):>10.2f}c/s")

    print(f"\n  36 clips in {OUT}/<voice>/, plus <voice>_tour.wav to listen straight through")


if __name__ == "__main__":
    main()
