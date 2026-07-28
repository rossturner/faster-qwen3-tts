"""Experiment 3, Stage A: is `instruct` inert on the ICL clone path?

`instruct` is plumbed into the Base clone path (upstream PR #40), but the tests only
assert the instruct tokens are prepended to the talker input -- nothing shows the model
obeys them. This measures whether it does.

Design, per the spike spec:

  4 conditions x 5 runs on the pinned clone, so an instruct effect can be told apart
  from the run-to-run variation of unseeded sampling.

  A positive control on CustomVoice, where Alibaba documents instruct as supported.
  Without it a null result is ambiguous: inert instruct and insensitive metrics look
  identical.

  Decision rule fixed in advance: instruct is EFFECTIVE on a metric when the spread of
  condition means exceeds 2x the mean within-condition spread.

Usage:
    .venv/bin/python spikes/emotion/instruct_inertness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.voice_registry import load_registry

REPO = Path(__file__).resolve().parents[2]
VOICES = REPO / "faster_qwen3_tts" / "server_voices" / "voices.yaml"
OUT = Path(__file__).parent / "out"
VOICE_ID = "en_f"
RUNS = 5
TEMPERATURE = 0.7

TEXT = ("I really did not expect that to happen, and now I have no idea what to do next.")

CONDITIONS = {
    "none": None,
    "slow_sad": "Speak slowly and sadly, in a low, subdued, downcast tone.",
    "fast_excited": "Speak quickly and excitedly, in a bright, energetic, animated tone.",
    "whispered": "Speak very quietly, in a soft, hushed whisper.",
}


def f0_stats(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Median and IQR of frame-wise F0 via autocorrelation.

    Deliberately simple and dependency-free -- absolute accuracy does not matter, only
    that the same estimator is applied to every condition so differences are comparable.
    """
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 400), int(sr / 60)          # 60-400 Hz search range
    pitches = []
    for start in range(0, len(audio) - frame, hop):
        seg = audio[start:start + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.01:      # skip silence
            continue
        seg = seg - seg.mean()
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        if corr[0] <= 0:
            continue
        window = corr[lo:hi]
        if len(window) == 0:
            continue
        peak = int(np.argmax(window)) + lo
        if corr[peak] / corr[0] > 0.3:             # require a confident peak
            pitches.append(sr / peak)
    if len(pitches) < 5:
        return float("nan"), float("nan")
    return float(np.median(pitches)), float(np.subtract(*np.percentile(pitches, [75, 25])))


def measure(audio: np.ndarray, sr: int, n_chars: int) -> dict:
    duration = len(audio) / sr
    voiced = audio[np.abs(audio) > 0.01]
    median_f0, iqr_f0 = f0_stats(audio, sr)
    return {
        "duration_s": round(duration, 3),
        "chars_per_s": round(n_chars / duration, 2),
        "rms": round(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))), 5),
        "voiced_frac": round(len(voiced) / len(audio), 3),
        "f0_median": round(median_f0, 1),
        "f0_iqr": round(iqr_f0, 1),
    }


METRICS = ["duration_s", "chars_per_s", "rms", "f0_median", "voiced_frac"]


def verdict(by_condition: dict[str, list[dict]]) -> dict:
    """Between-condition spread vs within-condition spread, per metric."""
    out = {}
    for metric in METRICS:
        means, within = [], []
        for runs in by_condition.values():
            values = [r[metric] for r in runs if not np.isnan(r[metric])]
            if not values:
                continue
            means.append(float(np.mean(values)))
            within.append(float(np.std(values)))
        if len(means) < 2:
            continue
        between = float(np.std(means))
        mean_within = float(np.mean(within))
        ratio = between / mean_within if mean_within > 0 else float("inf")
        out[metric] = {"between": round(between, 4), "within": round(mean_within, 4),
                       "ratio": round(ratio, 2), "effective": bool(ratio > 2.0)}
    return out


def run_clone(model, prompt, cfg) -> dict:
    results = {}
    for name, instruct in CONDITIONS.items():
        runs = []
        for i in range(RUNS):
            wavs, sr = model.generate_voice_clone(
                text=TEXT, language=cfg.language, voice_clone_prompt=prompt,
                ref_text=cfg.ref_text, xvec_only=False, instruct=instruct,
                temperature=TEMPERATURE)
            audio = np.asarray(wavs[0], dtype=np.float32)
            runs.append(measure(audio, sr, len(TEXT)))
            _write(OUT / "clone" / f"{name}_{i}.wav", audio, sr)
        results[name] = runs
        m = runs[0]
        print(f"    {name:<14} dur {np.mean([r['duration_s'] for r in runs]):5.2f}s  "
              f"f0 {np.nanmean([r['f0_median'] for r in runs]):6.1f}Hz  "
              f"rms {np.mean([r['rms'] for r in runs]):.4f}", flush=True)
    return results


def run_control(cfg_language: str) -> dict:
    """Positive control: same instructs on CustomVoice, where instruct is supported."""
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)
    results = {}
    for name, instruct in CONDITIONS.items():
        runs = []
        for i in range(RUNS):
            wavs, sr = model.generate_custom_voice(
                text=TEXT, speaker="aiden", language=cfg_language,
                instruct=instruct, temperature=TEMPERATURE)
            audio = np.asarray(wavs[0], dtype=np.float32)
            runs.append(measure(audio, sr, len(TEXT)))
            _write(OUT / "control" / f"{name}_{i}.wav", audio, sr)
        results[name] = runs
        print(f"    {name:<14} dur {np.mean([r['duration_s'] for r in runs]):5.2f}s  "
              f"f0 {np.nanmean([r['f0_median'] for r in runs]):6.1f}Hz  "
              f"rms {np.mean([r['rms'] for r in runs]):.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return results


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def report(label: str, v: dict) -> bool:
    print(f"\n  {label}: between-condition spread vs within-condition noise")
    any_effective = False
    for metric, s in v.items():
        flag = "EFFECTIVE" if s["effective"] else "-"
        any_effective |= s["effective"]
        print(f"    {metric:<14} between {s['between']:>8.4f}  within {s['within']:>8.4f}  "
              f"ratio {s['ratio']:>6.2f}  {flag}")
    return any_effective


def main() -> None:
    registry = load_registry(VOICES)
    cfg = registry.resolve(VOICE_ID)
    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048)
    prompt = model.model.create_voice_clone_prompt(
        ref_audio=str(cfg.ref_audio), ref_text=cfg.ref_text, x_vector_only_mode=False)

    print(f"clone path ({VOICE_ID}, ICL), {RUNS} runs x {len(CONDITIONS)} conditions:")
    clone = run_clone(model, prompt, cfg)
    del model
    torch.cuda.empty_cache()

    print("\ncontrol (CustomVoice/aiden, instruct officially supported):")
    control = run_control(cfg.language)

    clone_v, control_v = verdict(clone), verdict(control)
    clone_effective = report("CLONE", clone_v)
    control_effective = report("CONTROL", control_v)

    print("\n  VERDICT: ", end="")
    if clone_effective:
        print("instruct MOVES the clone path -- usable for emotion.")
    elif control_effective:
        print("instruct is INERT on the clone path (control responded, so the metrics "
              "can detect a real effect).\n           Stage B must use reference clips.")
    else:
        print("INCONCLUSIVE -- the control did not respond either, so the measurement "
              "is at fault,\n           not the engine. Do not report this as a Qwen finding.")

    (Path(__file__).parent / "inertness_results.json").write_text(json.dumps(
        {"clone": clone, "control": control,
         "verdict": {"clone": clone_v, "control": control_v}}, indent=2))
    print(f"\n  wavs in {OUT}/")


if __name__ == "__main__":
    main()
