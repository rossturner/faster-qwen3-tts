# spikes/two_model_coexistence.py
"""Spike: confirm CustomVoice + Base coexist (loaded + generating) on one GPU."""
import torch
from faster_qwen3_tts import FasterQwen3TTS

def load(model_id):
    return FasterQwen3TTS.from_pretrained(
        model_id, device="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=2048,
    )

print("Loading CustomVoice + Base together...")
custom = load("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
base = load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

print("Generating on CustomVoice...")
c_audio, sr = custom.generate_custom_voice(
    text="This is a coexistence test.", speaker="aiden",
    language="English", temperature=0.7)
print(f"  custom: {len(c_audio[0])/sr:.2f}s")

print("Generating on Base (clone)...")
b_audio, sr = base.generate_voice_clone(
    text="This is a coexistence test.", language="English",
    ref_audio="voice_design_audition/en_f_v1_ref.wav",
    ref_text="Welcome to the course. Let's get started with today's lesson.",
    xvec_only=False, temperature=0.7)
print(f"  base:   {len(b_audio[0])/sr:.2f}s")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\nPASS — both models coexisted. Peak VRAM allocated: {peak_gb:.2f} GB")
