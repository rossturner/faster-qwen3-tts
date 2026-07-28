"""Does a smaller chunk_size actually risk starving playback?

The spike's headroom figures (407 ms at cs=8, 31 ms at cs=4) were measured on the clone
path, library-level, and were worst *inter-arrival gaps* rather than steady-state slack.
This measures the quantity that decides the question, on the custom path over HTTP:

  margin(i) = audio delivered by chunk i  -  time elapsed since playback began

Playback starts when the first audio frame lands, so margin(0) is the first chunk's own
duration and the run starves if the margin ever reaches zero. The spike found the margin
only ever grows after the first chunk; if that holds, a smaller chunk is exposed only at
the opening and buys lower TTFA on every beat thereafter.

`--load` spawns a competing CUDA process. It is a crude proxy for the LLM co-tenant --
continuous large matmuls, not a real inference workload -- but contention is the stall
source that matters and is otherwise entirely unmeasured.

Usage:
    .venv/bin/python spikes/streaming/chunk_margin.py [--port 8093] [--load]
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import sys
import time

import requests

TEXT = ("Okay, so I spent basically the whole afternoon on this, and I want to walk you "
        "through what actually happened, because honestly, the ending is not what I "
        "expected when I started, and I think you are going to enjoy it.")
INSTRUCT = "Tired and annoyed, slow and heavy with low pitch and low energy."

LOAD_SRC = """
import torch, time
a = torch.randn(6144, 6144, device='cuda', dtype=torch.bfloat16)
b = torch.randn(6144, 6144, device='cuda', dtype=torch.bfloat16)
while True:
    for _ in range(40):
        a = (a @ b).relu() * 0.001
    torch.cuda.synchronize()
"""


def run_once(url: str, chunk_size: int, sample_rate: int = 24000) -> dict:
    """One request; returns the margin curve and the gaps that produced it."""
    sent = time.perf_counter()
    r = requests.post(url, json={"input": TEXT, "voice": "ono_anna",
                                 "instruct": INSTRUCT, "chunk_size": chunk_size},
                      stream=True, timeout=180)
    r.raise_for_status()

    buf, arrivals, first_audio, saw_end = bytearray(), [], None, False
    for block in r.iter_content(chunk_size=8192):
        buf += block
        while len(buf) >= 5:
            ftype = buf[0]
            (n,) = struct.unpack(">I", buf[1:5])
            if len(buf) < 5 + n:
                break
            payload = bytes(buf[5:5 + n])
            del buf[:5 + n]
            now = time.perf_counter()
            if ftype == 2:
                if first_audio is None:
                    first_audio = now
                arrivals.append((now, len(payload) / 2 / sample_rate))
            elif ftype == 5:
                saw_end = True
    r.close()
    if not saw_end:
        raise RuntimeError("stream ended without an end frame")

    delivered, margins, gaps, prev = 0.0, [], [], first_audio
    for at, duration in arrivals:
        delivered += duration
        margins.append(delivered - (at - first_audio))
        gaps.append(at - prev)
        prev = at
    return {"ttfa_ms": (first_audio - sent) * 1000, "chunks": len(arrivals),
            "chunk_s": arrivals[0][1], "margins": margins, "gaps": gaps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[4, 8])
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/audio/stream"

    load = None
    if args.load:
        load = subprocess.Popen([sys.executable, "-c", LOAD_SRC],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("competing CUDA process started (proxy for the LLM co-tenant)\n")
        time.sleep(15)

    try:
        print(f"{'cs':<4} {'TTFA':>9} {'chunk':>8} {'min margin':>12} {'end margin':>12} "
              f"{'worst gap':>11} {'starved':>8}")
        for cs in args.chunk_sizes:
            run_once(url, cs)                      # settle
            runs = [run_once(url, cs) for _ in range(args.runs)]
            ttfa = statistics.mean(r["ttfa_ms"] for r in runs)
            chunk_s = runs[0]["chunk_s"]
            min_margin = min(min(r["margins"]) for r in runs)
            end_margin = statistics.mean(r["margins"][-1] for r in runs)
            worst_gap = max(max(r["gaps"][1:]) for r in runs)
            starved = sum(1 for r in runs if min(r["margins"]) <= 0)
            print(f"{cs:<4} {ttfa:>7.0f}ms {chunk_s * 1000:>6.0f}ms "
                  f"{min_margin * 1000:>10.0f}ms {end_margin * 1000:>10.0f}ms "
                  f"{worst_gap * 1000:>9.0f}ms {starved:>5}/{len(runs)}")

        print("\n  margin curve, first run of each (ms):")
        for cs in args.chunk_sizes:
            m = run_once(url, cs)["margins"]
            print(f"    cs={cs:<3} " + " ".join(f"{x * 1000:.0f}" for x in m[:12]))
    finally:
        if load is not None:
            load.terminate()
            load.wait(timeout=10)


if __name__ == "__main__":
    main()
