"""Do bracketed non-verbal tags do anything on their own?

Follow-up to text_markup.py, which was confounded: the laugh notations were paired with
an instruct that already said "laughing", so the controls laughed too and the tags could
not be credited with anything. It did establish the safety property -- none of the four
notations was ever read aloud.

Here the instruct is deliberately neutral, so any vocalisation has to come from the text.
The mid-sentence cell is the one that matters most: if a tag works there, it gives
localised control within an utterance, which the instruct cannot do.

Usage:
    .venv/bin/python spikes/emotion/text_tags_neutral.py
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

OUT = Path(__file__).parent / "text_tags_neutral"
SPEAKER = "ono_anna"
LANGUAGE = "English"
TEMPERATURE = 0.7
TAKES = 2
GAP_SECONDS = 0.6

INSTRUCT = "Calm and even, moderate pace, neutral tone."
BASE = "That is genuinely the worst idea anyone has suggested all week."

CELLS = {
    "control": BASE,
    "laughs_start": f"[laughs] {BASE}",
    "laughs_mid": "That is genuinely the worst idea — [laughs] — anyone has suggested all week.",
    "laughs_asterisk": f"*laughs* {BASE}",
    "sighs_start": f"[sighs] {BASE}",
    "gasps_start": f"[gasps] {BASE}",
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
    model.generate_custom_voice(text="Warmup.", speaker=SPEAKER, language=LANGUAGE,
                                max_new_tokens=20)

    clips, manifest, sr = {}, {}, None
    for cell, text in CELLS.items():
        takes = 1 if cell == "control" else TAKES
        audios = []
        for take in range(takes):
            wavs, sr = model.generate_custom_voice(
                text=text, speaker=SPEAKER, language=LANGUAGE, instruct=INSTRUCT,
                temperature=TEMPERATURE)
            audio = np.asarray(wavs[0], dtype=np.float32)
            write_wav(OUT / cell / f"take{take}.wav", audio, sr)
            audios.append(audio)
        clips[cell] = audios
        manifest[cell] = {"text": text,
                          "takes": [measure(a, sr, len(text)) for a in audios]}
        print(f"  {cell:<16} {np.mean([m['duration_s'] for m in manifest[cell]['takes']]):5.2f}s",
              flush=True)

    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
    for cell, audios in clips.items():
        if cell == "control":
            continue
        tour = [c for a in clips["control"] + audios for c in (a, gap)]
        write_wav(OUT / f"tour_{cell}.wav", np.concatenate(tour), sr)

    (OUT / "results.json").write_text(json.dumps(
        {"speaker": SPEAKER, "instruct": INSTRUCT, "temperature": TEMPERATURE,
         "measurements": manifest}, indent=2))

    print(f"\n  {sum(len(v) for v in clips.values())} clips in {OUT}/")
    print("    tour_<cell>.wav   the control, then that cell's two takes")


if __name__ == "__main__":
    main()
