"""Can a designed voice borrow Ono_Anna's emotional performance?

A voice clone prompt carries two separable things: `ref_spk_embedding`, a compact vector
of who the speaker is, and `ref_code`, the reference recording as codec tokens. Normally
both come from one clip. This builds the prompt from an emotional Ono_Anna clip and
swaps in the designed persona's identity vector, to see which one the model follows.

Three outcomes, all distinguishable by ear:
  - sounds like Ono_Anna, emotional   -> the recording won; no gain over using her
  - sounds like the persona, flat     -> the identity won; no gain over the pinned clone
  - sounds like the persona, emotional-> what we want

Stage A found the reference recording overrules everything else, so the first outcome is
the expected one. Each emotion is rendered as a tour of persona control, two spliced
takes, then the Ono_Anna control, so the spliced takes can be placed between the two
identity endpoints.

Usage:
    .venv/bin/python spikes/emotion/xvec_splice.py
"""

from __future__ import annotations

import json
import sys
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from instruct_inertness import measure  # noqa: E402

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "xvec_splice"
LANGUAGE = "English"
TEMPERATURE = 0.7
SPLICE_TAKES = 2
GAP_SECONDS = 0.6

PERSONA_CLIP = HERE / "voicedesign_directions_v2" / "neutral" / "take0.wav"
PERSONA_TEXT = ("Someone in chat just asked me whether I had ever tried any of this "
                "before, and the honest answer is no, not once.")

ONO_DIR = HERE / "female_voices_range" / "ono_anna"
ONO_TEXT = ("Someone in chat just asked me whether I had ever tried any of this before, "
            "and the honest answer is no, not once, which is probably going to become "
            "very obvious in about thirty seconds.")

EMOTIONS = {
    "warm": "03_warm_p2.wav",
    "curious": "04_curious_p2.wav",
    "excited": "05_excited_p2.wav",
    "amused": "06_amused_p2.wav",
    "exasperated": "07_exasperated_p2.wav",
    "sad": "08_sad_p2.wav",
}

TARGET = ("I honestly did not see that coming, and now I have absolutely no idea what "
          "to do next.")


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    print("Loading Base...", flush=True)
    base = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)

    def gen(prompt, ref_text):
        wavs, sr = base.generate_voice_clone(
            text=TARGET, language=LANGUAGE, voice_clone_prompt=prompt,
            ref_text=ref_text, temperature=TEMPERATURE)
        return np.asarray(wavs[0], dtype=np.float32), sr

    persona_items = base.model.create_voice_clone_prompt(
        ref_audio=str(PERSONA_CLIP), ref_text=PERSONA_TEXT)
    persona_xvec = persona_items[0].ref_spk_embedding

    print("persona control (identity endpoint A)...", flush=True)
    persona_audio, sr = gen(persona_items, PERSONA_TEXT)
    write_wav(OUT / "control_persona.wav", persona_audio, sr)
    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)

    manifest = {"control_persona": measure(persona_audio, sr, len(TARGET))}
    for emotion, filename in EMOTIONS.items():
        print(f"{emotion}...", flush=True)
        items = base.model.create_voice_clone_prompt(
            ref_audio=str(ONO_DIR / filename), ref_text=ONO_TEXT)

        ono_audio, sr = gen(items, ONO_TEXT)
        write_wav(OUT / emotion / "control_onoanna.wav", ono_audio, sr)

        spliced = [replace(items[0], ref_spk_embedding=persona_xvec)]
        splices = []
        for take in range(SPLICE_TAKES):
            audio, sr = gen(spliced, ONO_TEXT)
            write_wav(OUT / emotion / f"splice_take{take}.wav", audio, sr)
            splices.append(audio)

        tour = [persona_audio, gap] + [c for a in splices for c in (a, gap)] + [ono_audio, gap]
        write_wav(OUT / f"tour_{emotion}.wav", np.concatenate(tour), sr)

        manifest[emotion] = {
            "control_onoanna": measure(ono_audio, sr, len(TARGET)),
            "splice": [measure(a, sr, len(TARGET)) for a in splices],
        }
        p, o = manifest["control_persona"], manifest[emotion]["control_onoanna"]
        s = np.nanmean([m["f0_median"] for m in manifest[emotion]["splice"]])
        print(f"    pitch  persona {p['f0_median']:6.1f}   splice {s:6.1f}   "
              f"ono_anna {o['f0_median']:6.1f} Hz", flush=True)

    (OUT / "results.json").write_text(json.dumps(
        {"persona_clip": str(PERSONA_CLIP), "target": TARGET,
         "emotions": EMOTIONS, "measurements": manifest}, indent=2))

    print(f"\n  clips in {OUT}/")
    print("    tour_<emotion>.wav   persona control, two spliced takes, Ono_Anna control")
    print("  If the spliced takes sit at the Ono_Anna end, the recording won and the "
          "splice is dead.")


if __name__ == "__main__":
    main()
