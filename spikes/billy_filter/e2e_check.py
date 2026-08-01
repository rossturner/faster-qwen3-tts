"""End-to-end: is the filter actually applied, over both endpoints, on the live server?

Billy declares the chorus in his character.yaml; nicole declares nothing. Comparing the
two isolates the filter from everything else the server does.

Writes the audio out so it can be listened to, because "the bytes differ" is not the same
claim as "it sounds like Billy".
"""
from __future__ import annotations

import io
import json
import struct
import time
import wave
from pathlib import Path

import numpy as np
import requests

BASE = "http://localhost:8092"
OUT = Path(__file__).parent / "out" / "e2e"
LINE = ("Boss, I've been thinking about that mech we saw yesterday, and I'm almost "
        "certain I've seen it somewhere before.")


def whole_file(voice, emotion="neutral"):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/audio/speech", json={
        "input": LINE, "voice": voice, "emotion": emotion, "response_format": "wav"})
    r.raise_for_status()
    ms = (time.perf_counter() - t0) * 1000
    with wave.open(io.BytesIO(r.content)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return pcm, sr, ms, r.headers.get("X-TTS-Reference")


def stream(voice, emotion="neutral", chunk_size=4):
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/audio/stream", json={
        "input": LINE, "voice": voice, "emotion": emotion, "chunk_size": chunk_size},
        stream=True)
    r.raise_for_status()
    buf, audio, ttfa, end, header = b"", [], None, None, None
    for piece in r.iter_content(4096):
        buf += piece
        while len(buf) >= 5:
            kind = buf[0]
            (ln,) = struct.unpack(">I", buf[1:5])
            if len(buf) < 5 + ln:
                break
            payload, buf = buf[5:5 + ln], buf[5 + ln:]
            if kind == 1:
                header = json.loads(payload)
            elif kind == 2:
                if ttfa is None:
                    ttfa = (time.perf_counter() - t0) * 1000
                audio.append(np.frombuffer(payload, np.int16).astype(np.float32) / 32768)
            elif kind == 4:
                raise RuntimeError(json.loads(payload)["message"])
            elif kind == 5:
                end = json.loads(payload)
    return (np.concatenate(audio) if audio else np.zeros(0)), ttfa, end, header


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'voice':8s} {'path':7s} {'samples':>8s} {'audio_ms':>9s} {'ttfa_ms':>8s}  end-frame")
    results = {}
    for voice in ("billy", "nicole"):
        pcm, sr, ms, ref = whole_file(voice)
        results[(voice, "whole")] = pcm
        import soundfile as sf
        sf.write(OUT / f"{voice}_speech.wav", pcm, sr)
        print(f"{voice:8s} {'speech':7s} {len(pcm):8d} {len(pcm)/sr*1000:9.1f} {ms:8.1f}  n/a  ref={ref}")

        pcm, ttfa, end, header = stream(voice)
        results[(voice, "stream")] = pcm
        sf.write(OUT / f"{voice}_stream.wav", pcm, sr)
        ok = end is not None
        print(f"{voice:8s} {'stream':7s} {len(pcm):8d} {len(pcm)/sr*1000:9.1f} "
              f"{ttfa:8.1f}  {'yes' if ok else 'MISSING'}  "
              f"declared={end['total_audio_ms'] if ok else '-'}")

    # A stream and a whole file are different generations, so they cannot be compared
    # sample for sample. What must hold is that the durations agree: the flush is what
    # stops the streaming path losing its last window.
    for voice in ("billy", "nicole"):
        w, s = results[(voice, "whole")], results[(voice, "stream")]
        print(f"\n{voice}: whole {len(w)/24000*1000:.0f} ms vs stream "
              f"{len(s)/24000*1000:.0f} ms")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
