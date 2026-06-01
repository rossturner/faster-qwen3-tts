# scripts/make_builtin_clone_refs.py
"""Render aiden/sohee (CustomVoice presets) speaking the per-language reference
line, to use as the clone source for the en_m_clone / ko_f_clone voices."""
import os, torch, soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

OUT = "faster_qwen3_tts/server_voices/refs"
INSTRUCT = "Speak in a calm, clear, professional tone suitable for an instructional video."
JOBS = [
    ("aiden", "English", "Welcome to the course. Let's get started with today's lesson.", "aiden_ref.wav"),
    ("sohee", "Korean", "이 강좌에 오신 것을 환영합니다. 오늘 수업을 시작해 보겠습니다.", "sohee_ref.wav"),
]
m = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda",
    dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048)
m.generate_custom_voice(text="Warmup.", speaker="aiden", language="English", max_new_tokens=20)
for speaker, lang, text, fname in JOBS:
    wavs, sr = m.generate_custom_voice(text=text, speaker=speaker, language=lang,
                                       instruct=INSTRUCT, temperature=0.7)
    sf.write(os.path.join(OUT, fname), wavs[0], sr)
    print(f"wrote {fname}: {len(wavs[0])/sr:.2f}s")
