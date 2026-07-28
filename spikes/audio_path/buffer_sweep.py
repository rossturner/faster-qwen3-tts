"""Experiment 1, path A: how much audio sits in flight on the WSLg path?

`pulse_probe.py overhead` showed the writer finishing ~1s before the audio does,
which means ~1s of submitted audio is still buffered when the write completes.
That buffer is added latency: a sentence handed to the player is not heard until
the buffer ahead of it drains.

This measures the depth directly, and whether ffmpeg's pulse buffer options move it.
Depth is inferred as (audio duration - wall time to write it all), which is only
meaningful when the writer is the thing being blocked -- so the tone must be longer
than the buffer.

Usage:
    .venv/bin/python spikes/audio_path/buffer_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

SAMPLE_RATE = 24000
DURATION = 6.0
REPEATS = 3


def tone(seconds: float) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    return (0.25 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype("<i2").tobytes()


def measure(extra_opts: list[str]) -> float:
    """Return seconds of audio still in flight when the writer finishes."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        "-f", "pulse", *extra_opts, "lyrebird-spike",
    ]
    pcm = tone(DURATION)
    start = time.perf_counter()
    proc = subprocess.run(cmd, input=pcm, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return float("nan")
    return DURATION - elapsed


def main() -> None:
    configs = [
        ("default", []),
        ("buffer_duration=500ms", ["-buffer_duration", "500"]),
        ("buffer_duration=200ms", ["-buffer_duration", "200"]),
        ("buffer_duration=100ms", ["-buffer_duration", "100"]),
        ("buffer_duration=50ms", ["-buffer_duration", "50"]),
    ]
    print(f"in-flight audio when the writer finishes ({DURATION}s tone, "
          f"{REPEATS} runs, first run discarded as cold):\n")
    for label, opts in configs:
        depths = [measure(opts) for _ in range(REPEATS + 1)][1:]
        if any(np.isnan(d) for d in depths):
            print(f"  {label:<24} UNSUPPORTED")
            continue
        print(f"  {label:<24} {np.mean(depths) * 1000:7.1f}ms  "
              f"(min {min(depths) * 1000:.1f}  max {max(depths) * 1000:.1f})")
    print("\nLower is less added latency. A floor that does not move with the option "
          "\nis the WSLg/RDP transport's own buffering, not something we control.")


if __name__ == "__main__":
    if sys.platform != "linux":
        sys.exit("run inside WSL")
    main()
