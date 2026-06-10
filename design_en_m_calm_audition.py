#!/usr/bin/env python3
"""English-male replacement audition (calm-leaning wide net).

The shipped en_m (friendly_casual) reads as 'almost right but too energetic'. This
designs a spread of calmer English-male personas, renders one VoiceDesign reference
clip per persona, then clones the first 4 chunks of a real media-worker dub off each
ref (design-once -> pin -> clone), so each candidate is judged exactly as it would
serve. Outputs an audition.html for listening by ear.
"""
import os
import gc
import json
import html
import torch
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "voice_design_en_m_calm")
os.makedirs(OUT_DIR, exist_ok=True)

# Clean, longer reference line used to design each ref clip (longer ref = better clone basis).
REF_TEXT = "In today's lesson we'll work through each step slowly and carefully, so take your time and follow along at your own pace as we go."

# Judging material: first 4 chunks of a real en dub (takaillust drawing tutorial).
TTS_JSON = "/home/ross/workspace/prentora/media-worker/app/workspace/0610-080933-dub-takaillust-312dc79f/14-tts-en/tts.json"
chunks = json.load(open(TTS_JSON))["chunks"][:4]
SAMPLE_LINES = [c["text"] for c in chunks]

# Clear/present, natural-paced spread. instruct is English-only and independent of output language.
# NOT energetic/lively/upbeat, but NOT deep/soft/slow either — aim for clear, present, natural pace.
PERSONAS = [
    ("clear_articulate",  "A clear, articulate adult man's voice in a natural mid-range with clean diction and a natural, everyday speaking pace, confident and easy to follow, suitable for an instructional video."),
    ("bright_natural",    "A bright, natural younger-adult man's voice in a clear mid-range, friendly and approachable, with a relaxed but flowing pace and good presence, suitable for an online tutorial."),
    ("crisp_professional","A crisp, professional adult man's voice in a confident mid-range, precise and clear with an even, natural pace, suitable for a corporate training video."),
    ("warm_engaged",      "A warm, engaged adult man's voice in a natural mid-range, attentive and present, speaking at a natural conversational pace, like an interested teacher explaining a topic."),
    ("friendly_present",  "A relaxed, friendly adult man's voice in a natural mid-range with a warm, conversational tone and good presence, moving along at a natural, comfortable pace, suitable for a tutorial."),
    ("confident_mid",     "A confident, steady adult man's voice in a clear mid-range, assured and grounded, with a natural, brisk-but-composed pace, suitable for an instructional video."),
    ("modern_explainer",  "A clear, modern adult man's voice in a natural mid-range with clean articulation and a natural, conversational pace, in a contemporary online-explainer style, suitable for a tutorial."),
    ("light_tenor",       "A light, clear adult man's voice in a higher mid/tenor range, natural and approachable, with even articulation and a natural pace, suitable for an online tutorial."),
]

print(f"Personas: {len(PERSONAS)} | sample lines: {len(SAMPLE_LINES)}")
for i, l in enumerate(SAMPLE_LINES):
    print(f"  chunk{i}: {l}")

print("\nLoading VoiceDesign 1.7B...")
design = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
design.generate_voice_design(text="Test.", instruct=PERSONAS[0][1], language="English", max_new_tokens=20)

refs = {}
for n, (persona, instruct) in enumerate(PERSONAS, 1):
    wavs, sr = design.generate_voice_design(text=REF_TEXT, instruct=instruct, language="English", temperature=0.7)
    p = os.path.join(OUT_DIR, f"{persona}_ref.wav")
    sf.write(p, wavs[0], sr)
    refs[persona] = p
    print(f"  [design {n:2d}/{len(PERSONAS)}] {persona:20s} {len(wavs[0])/sr:4.1f}s")

del design
gc.collect()
torch.cuda.empty_cache()

print("\nLoading Base 1.7B...")
base = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=2048,
)
print("Warmup...")
base.generate_voice_clone(text="Test.", language="English",
                          ref_audio=refs[PERSONAS[0][0]], ref_text=REF_TEXT, max_new_tokens=20)

for persona, instruct in PERSONAS:
    for i, text in enumerate(SAMPLE_LINES):
        wavs, sr = base.generate_voice_clone(
            text=text, language="English",
            ref_audio=refs[persona], ref_text=REF_TEXT,
            xvec_only=False, temperature=0.7,
        )
        sf.write(os.path.join(OUT_DIR, f"{persona}_chunk{i}.wav"), wavs[0], sr)
    print(f"  [clone {persona:20s}] {len(SAMPLE_LINES)} chunks done")

# --- audition.html ---
rows = []
for persona, instruct in PERSONAS:
    cells = [f'<td class="ref"><b>{html.escape(persona)}</b><div class="ins">{html.escape(instruct)}</div>'
             f'<div class="r">ref:<br><audio controls src="{persona}_ref.wav"></audio></div></td>']
    for i in range(len(SAMPLE_LINES)):
        cells.append(f'<td><audio controls src="{persona}_chunk{i}.wav"></audio></td>')
    rows.append("<tr>" + "".join(cells) + "</tr>")

heads = ['<th>persona / instruct</th>'] + [
    f'<th>chunk {i}<div class="ln">{html.escape(l)}</div></th>' for i, l in enumerate(SAMPLE_LINES)]
page = f"""<!doctype html><meta charset=utf-8><title>en_m calm audition</title>
<style>
body{{font:14px/1.4 system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:18px}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #333;padding:8px;vertical-align:top}}
th{{background:#1c1c1c;text-align:left}} .ref{{width:300px;background:#181818}}
.ins{{color:#9bb;font-size:12px;margin:4px 0}} .ln{{color:#9a9;font-weight:normal;font-size:12px;margin-top:4px}}
.r{{margin-top:6px;color:#aaa}} audio{{width:230px;height:34px}}
</style>
<h1>English-male replacement audition — calm wide net</h1>
<p>Design (VoiceDesign) &rarr; ref clip &rarr; Base clone of 4 real dub chunks. Current shipped voice = <code>friendly_casual</code> (too energetic). temperature=0.7, unseeded single takes.</p>
<table><thead><tr>{''.join(heads)}</tr></thead><tbody>{''.join(rows)}</tbody></table>
"""
with open(os.path.join(OUT_DIR, "audition.html"), "w") as f:
    f.write(page)

print(f"\nDone -> {OUT_DIR}")
print(f"Audition page: {os.path.join(OUT_DIR, 'audition.html')}")
