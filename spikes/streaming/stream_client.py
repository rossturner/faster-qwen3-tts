"""Drive /v1/audio/stream end to end and report time-to-first-audio.

The library-level TTFA figures in docs/lyrebird-tts-spike-findings.md were measured on
the clone path, where a ~7s reference clip sits in the prefill context. A custom voice
carries no reference audio, so this measures the path lyrebird will actually use,
including HTTP transport, which the spike explicitly excluded.

Usage:
    .venv/bin/python spikes/streaming/stream_client.py [--port 8093] [--instruct "..."]
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import wave
from pathlib import Path

import requests

FRAME_NAMES = {1: "header", 2: "audio", 3: "mark", 4: "error", 5: "end"}


def read_frames(response):
    """Yield (type, payload, arrival_time) as frames complete on the wire."""
    buf = bytearray()
    for block in response.iter_content(chunk_size=1):
        buf += block
        while len(buf) >= 5:
            ftype = buf[0]
            (length,) = struct.unpack(">I", buf[1:5])
            if len(buf) < 5 + length:
                break
            payload = bytes(buf[5:5 + length])
            del buf[:5 + length]
            yield ftype, payload, time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--voice", default="ono_anna")
    ap.add_argument("--instruct", default="Tired and annoyed, slow and heavy with low pitch and low energy.")
    ap.add_argument("--text", default="Haha, no. I have read this same error message four times now and it still says nothing useful.")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--out", default="spikes/streaming/stream_out.wav")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/audio/stream"
    body = {"input": args.text, "voice": args.voice, "instruct": args.instruct,
            "chunk_size": args.chunk_size}
    print(f"POST {url}\n  voice    {args.voice}\n  instruct {args.instruct}\n"
          f"  text     {args.text}\n")

    sent = time.perf_counter()
    r = requests.post(url, json=body, stream=True, timeout=120)
    print(f"status {r.status_code}  content-type {r.headers.get('content-type')}")
    r.raise_for_status()

    pcm, sample_rate, ttfa, saw_end, marks = bytearray(), None, None, False, []
    for ftype, payload, at in read_frames(r):
        name = FRAME_NAMES.get(ftype, f"unknown({ftype})")
        if ftype == 1:
            header = json.loads(payload)
            sample_rate = header["sample_rate"]
            print(f"  {(at - sent) * 1000:7.1f} ms  header  {header}")
        elif ftype == 2:
            if ttfa is None:
                ttfa = at - sent
                print(f"  {ttfa * 1000:7.1f} ms  audio   FIRST AUDIO ({len(payload)} bytes)")
            pcm += payload
        elif ftype == 3:
            marks.append(json.loads(payload))
        elif ftype == 4:
            print(f"  {(at - sent) * 1000:7.1f} ms  ERROR   {json.loads(payload)}")
        elif ftype == 5:
            saw_end = True
            print(f"  {(at - sent) * 1000:7.1f} ms  end     {json.loads(payload)}")

    total = time.perf_counter() - sent
    if not saw_end:
        print("\n  NO END FRAME -- this is a failed generation, not a short one.")
        raise SystemExit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))

    audio_s = len(pcm) / 2 / sample_rate
    print(f"\n  chunks      {len(marks)}")
    print(f"  TTFA        {ttfa * 1000:.1f} ms   (incl. HTTP transport)")
    print(f"  wall        {total:.2f} s for {audio_s:.2f} s of audio  "
          f"(RTF {audio_s / total:.2f})")
    print(f"  wrote       {out}")


if __name__ == "__main__":
    main()
