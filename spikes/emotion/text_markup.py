"""Which orthographic devices in the input text actually do anything?

Qwen3-TTS has no markup support: the tokenizer's added tokens are all plumbing, and the
text goes verbatim into the chat template with no preprocessing. The 86 inline tags
([laughing], [gasp]) belong to the hosted Qwen-Audio-3.0-TTS, a different model. So
whatever a laugh or a hesitation costs has to come from how the words are written.

Each device is paired with a control -- same instruct, same sentence, device removed --
because otherwise an effect from the instruct is indistinguishable from an effect from
the device. Device cells get two takes since sampling is unseeded.

The outcome for each is one of: vocalised, silently ignored, read aloud literally, or
prosody changed without a sound. The third is the dangerous one -- a stage direction the
LLM emits would be spoken on stream.

Instructs are clip-caption style (delivery attributes only, no speaker description --
the speaker is fixed by the Ono_Anna token).

Usage:
    .venv/bin/python spikes/emotion/text_markup.py
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

OUT = Path(__file__).parent / "text_markup"
SPEAKER = "ono_anna"
LANGUAGE = "English"
TEMPERATURE = 0.7
DEVICE_TAKES = 2
GAP_SECONDS = 0.6

A_INSTRUCT = "Happy and amused, laughing, fast-paced and bright with high pitch."
A_TAIL = "Okay, whoever just sent that, I need you to know I nearly dropped my drink."

# (group, device, instruct, control text, device text)
CELLS = [
    ("A", "brackets_square", A_INSTRUCT, A_TAIL, f"[laughs] {A_TAIL}"),
    ("A", "brackets_round", A_INSTRUCT, A_TAIL, f"(laughs) {A_TAIL}"),
    ("A", "asterisks", A_INSTRUCT, A_TAIL, f"*laughs* {A_TAIL}"),
    ("A", "angle", A_INSTRUCT, A_TAIL, f"<laugh> {A_TAIL}"),

    ("B", "haha",
     "Happy and playful, teasing, at a moderate pace.",
     "No, absolutely not, we are not doing that again after last time.",
     "Haha, no, absolutely not, we are not doing that again after last time."),
    ("B", "ugh",
     "Tired and annoyed, slow and heavy with low pitch and low energy.",
     "I have read this same error message four times now and it still says nothing useful.",
     "Ugh, I have read this same error message four times now and it still says nothing useful."),
    ("B", "hmm",
     "Thoughtful and calm, slow and quiet with long pauses.",
     "Okay, hang on, let me actually think about that one properly for a second.",
     "Hmm. Okay, hang on, let me actually think about that one properly for a second."),
    ("B", "oh",
     "Surprised and startled, fast with a sudden rise in pitch.",
     "I genuinely did not expect it to do that, that is not what I clicked.",
     "Oh! Oh no, I genuinely did not expect it to do that, that is not what I clicked."),

    ("C", "ellipses",
     "Hesitant and reluctant, slow and drawn out.",
     "I mean, I suppose, if you really want me to, I could try it one more time.",
     "I mean... I suppose... if you really want me to, I could try it one more time."),
    ("C", "em_dash",
     "Excited, fast-paced and slightly breathless with high pitch.",
     "And then it just stopped, no, wait, that's not even the worst part of this.",
     "And then it just— no, wait, that's not even the worst part of this."),
    ("C", "caps",
     "Angry and indignant, loud and sharply stressed with high pitch.",
     "That is not what I said, and I have the receipts to prove it.",
     "That is NOT what I said, and I have the receipts to prove it."),
    ("C", "elongation",
     "Relaxed and happy, slow and soft with a warm tone.",
     "That was so much closer than it had any right to be, honestly.",
     "That was sooo much closer than it had any right to be, honestly."),
]


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
    model.generate_custom_voice(text="Warmup.", speaker=SPEAKER, language=LANGUAGE,
                                max_new_tokens=20)

    def gen(text, instruct):
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=SPEAKER, language=LANGUAGE, instruct=instruct,
            temperature=TEMPERATURE)
        return np.asarray(wavs[0], dtype=np.float32), sr

    manifest, sr = {}, None
    for group, device, instruct, control_text, device_text in CELLS:
        control, sr = gen(control_text, instruct)
        write_wav(OUT / group / device / "control.wav", control, sr)

        takes = []
        for take in range(DEVICE_TAKES):
            audio, sr = gen(device_text, instruct)
            write_wav(OUT / group / device / f"take{take}.wav", audio, sr)
            takes.append(audio)

        gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
        tour = [c for a in [control] + takes for c in (a, gap)]
        write_wav(OUT / f"tour_{device}.wav", np.concatenate(tour), sr)

        manifest[device] = {
            "group": group, "instruct": instruct,
            "control_text": control_text, "device_text": device_text,
            "control": measure(control, sr, len(control_text)),
            "takes": [measure(a, sr, len(device_text)) for a in takes],
        }
        c = manifest[device]["control"]["duration_s"]
        t = np.mean([m["duration_s"] for m in manifest[device]["takes"]])
        print(f"  {device:<16} control {c:5.2f}s   device {t:5.2f}s   "
              f"delta {t - c:+5.2f}s", flush=True)

    (OUT / "results.json").write_text(json.dumps(
        {"speaker": SPEAKER, "language": LANGUAGE, "temperature": TEMPERATURE,
         "measurements": manifest}, indent=2))

    print(f"\n  {len(CELLS) * (1 + DEVICE_TAKES)} clips in {OUT}/")
    print("    tour_<device>.wav   control, then the two device takes")
    print("  Listening for: vocalised / ignored / read aloud literally / prosody only.")


if __name__ == "__main__":
    main()
