"""Wide-range audition of the four built-in female voices, all speaking English.

Follow-up to female_voice_audition.py, which established that these voices respond to
instructions but used only three emotions and short sentences. This widens the emotional
range -- weighted toward the everyday registers a persona actually spends most of its
time in, not just the peaks -- and lengthens the passages so delivery can be judged over
a paragraph rather than a line.

`neutral` passes no instruction at all, so it is the honest baseline the rest are read
against.

Usage:
    .venv/bin/python spikes/emotion/female_voice_range.py
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

OUT = Path(__file__).parent / "female_voices_range"
LANGUAGE = "English"
TEMPERATURE = 0.7
GAP_SECONDS = 0.6
MAX_NEW_TOKENS = 1024

VOICES = {
    "vivian": "Chinese - bright, slightly edgy young female",
    "ono_anna": "Japanese - playful, light and nimble",
    "serena": "Chinese - warm, gentle young female",
    "sohee": "Korean - warm, rich emotion",
}

PASSAGES = {
    "p1": "Okay, so I spent basically the whole afternoon on this, and I want to walk "
          "you through what actually happened, because honestly, the ending is not "
          "what I expected when I started.",
    "p2": "Someone in chat just asked me whether I had ever tried any of this before, "
          "and the honest answer is no, not once, which is probably going to become "
          "very obvious in about thirty seconds.",
    "p3": "Let me get this straight. We started with a simple plan, we followed every "
          "single step exactly as it was written down, and somehow we have ended up "
          "here, looking at this.",
}

EMOTIONS = {
    "neutral": None,
    "calm": "Speak calmly and evenly, relaxed and unhurried, with a steady, grounded "
            "delivery and no particular emotional colour.",
    "conversational": "Speak in a casual, chatty way, like you are talking to a friend, "
                      "with a natural informal rhythm and easy, loose phrasing.",
    "warm": "Speak warmly and kindly, gentle and sincere, with a soft, caring delivery "
            "and an open, generous tone.",
    "curious": "Speak with genuine curiosity, thinking aloud as you go, engaged and "
               "questioning, with a light rising inflection.",
    "excited": "Speak with bright, excited energy, quick and animated, with a real lift "
               "in the voice, like you cannot wait to get the words out.",
    "amused": "Speak with warm amusement, light and playful and teasing, like you are "
              "holding back a laugh the whole way through.",
    "exasperated": "Speak with weary exasperation, dry and a little fed up, with a flat, "
                   "put-upon delivery and a sigh underneath it.",
    "sad": "Speak sadly and quietly, slow and subdued, low and downcast, with the "
           "energy drained out of it.",
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

    clips, manifest, sr = {}, {}, None
    for speaker, blurb in VOICES.items():
        print(f"\n{speaker}  ({blurb})", flush=True)
        manifest[speaker] = {}
        for ei, (emotion, instruct) in enumerate(EMOTIONS.items()):
            manifest[speaker][emotion] = {}
            for pi, (pid, text) in enumerate(PASSAGES.items()):
                wavs, sr = model.generate_custom_voice(
                    text=text, speaker=speaker, language=LANGUAGE, instruct=instruct,
                    temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
                audio = np.asarray(wavs[0], dtype=np.float32)
                write_wav(OUT / speaker / f"{ei:02d}_{emotion}_{pid}.wav", audio, sr)
                clips[(speaker, emotion, pid)] = audio
                manifest[speaker][emotion][pid] = measure(audio, sr, len(text))
            m = manifest[speaker][emotion]
            print(f"    {emotion:<15} "
                  f"dur {np.mean([m[p]['duration_s'] for p in PASSAGES]):5.2f}s   "
                  f"pitch {np.nanmean([m[p]['f0_median'] for p in PASSAGES]):6.1f}Hz   "
                  f"loudness {np.mean([m[p]['rms'] for p in PASSAGES]):.4f}", flush=True)

    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
    for speaker in VOICES:
        tour = [c for e in EMOTIONS for p in PASSAGES
                for c in (clips[(speaker, e, p)], gap)]
        write_wav(OUT / "tours_by_voice" / f"{speaker}.wav", np.concatenate(tour), sr)
    for emotion in EMOTIONS:
        tour = [c for s in VOICES for c in (clips[(s, emotion, "p1")], gap)]
        write_wav(OUT / "tours_by_emotion" / f"{emotion}.wav", np.concatenate(tour), sr)

    (OUT / "manifest.json").write_text(json.dumps(
        {"language": LANGUAGE, "temperature": TEMPERATURE, "voices": VOICES,
         "passages": PASSAGES, "emotions": EMOTIONS,
         "tour_order": {"by_voice": [f"{e}/{p}" for e in EMOTIONS for p in PASSAGES],
                        "by_emotion": list(VOICES)},
         "measurements": manifest}, indent=2))

    print("\nRange per voice (spread across all nine emotions, averaged over passages):")
    print(f"  {'voice':<10} {'pitch':>10} {'loudness':>12} {'pace':>12}")
    for speaker, by_emotion in manifest.items():
        def spread(key):
            per = [np.nanmean([by_emotion[e][p][key] for p in PASSAGES]) for e in EMOTIONS]
            return max(per) - min(per)
        print(f"  {speaker:<10} {spread('f0_median'):>8.1f}Hz "
              f"{spread('rms'):>12.4f} {spread('chars_per_s'):>9.2f}c/s")

    print(f"\n  {len(clips)} clips in {OUT}/")
    print("    tours_by_voice/<voice>.wav      all nine emotions, in the order above")
    print("    tours_by_emotion/<emotion>.wav  the four voices back to back on passage 1")


if __name__ == "__main__":
    main()
