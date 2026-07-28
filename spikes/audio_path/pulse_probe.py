"""Experiment 1, path A: WSLg PulseAudio -> Windows default device.

Two measurements:

  overhead  Play tones of several durations, all submitted up front. Wall time minus
            audio duration is the fixed pipeline overhead (startup + buffer drain).
            It should be roughly constant across durations; if it scales with
            duration, playback is not running at realtime and something is wrong.

  stream    Feed PCM to the player in small chunks at realtime pace, the way streamed
            TTS would arrive, and look for stalls. WSLg gives no underrun counter, so
            a stall shows up as wall time materially exceeding the audio duration.

Usage:
    .venv/bin/python spikes/audio_path/pulse_probe.py overhead
    .venv/bin/python spikes/audio_path/pulse_probe.py stream --seconds 60
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

SAMPLE_RATE = 24000  # match the TTS output rate
CHUNK_MS = 160       # ~ one streaming chunk of audio


def tone(seconds: float, freq: float = 440.0, beep_every: float | None = None) -> np.ndarray:
    """A continuous tone; optionally a pitch jump each `beep_every` seconds.

    The pitch jumps make dropouts audible to a listener during the human check.
    """
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f = np.full_like(t, freq)
    if beep_every:
        f[((t % beep_every) < 0.12)] = freq * 1.5
    wave = 0.25 * np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE)
    return (wave * 32767).astype("<i2")


def player() -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        "-f", "pulse", "lyrebird-spike",
    ]


def play_all_at_once(pcm: np.ndarray) -> float:
    start = time.perf_counter()
    proc = subprocess.run(player(), input=pcm.tobytes(), stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:400]}")
    return elapsed


def play_streamed(pcm: np.ndarray, chunk_ms: int = CHUNK_MS) -> tuple[float, list[float]]:
    """Write chunks at realtime pace; record how long each write blocked."""
    frames = int(SAMPLE_RATE * chunk_ms / 1000)
    proc = subprocess.Popen(player(), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    write_stalls = []
    start = time.perf_counter()
    for i in range(0, len(pcm), frames):
        target = start + (i / SAMPLE_RATE)
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        w0 = time.perf_counter()
        proc.stdin.write(pcm[i:i + frames].tobytes())
        proc.stdin.flush()
        write_stalls.append(time.perf_counter() - w0)
    proc.stdin.close()
    proc.wait()
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed: {proc.stderr.read().decode(errors='replace')[:400]}")
    return elapsed, write_stalls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["overhead", "stream"])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if args.mode == "overhead":
        results = []
        for duration in (1.0, 5.0):
            for run in range(args.repeats):
                pcm = tone(duration)
                elapsed = play_all_at_once(pcm)
                results.append({"duration_s": duration, "run": run,
                                "wall_s": round(elapsed, 4),
                                "overhead_s": round(elapsed - duration, 4)})
                print(f"  {duration:>4.1f}s tone -> wall {elapsed:6.3f}s  "
                      f"overhead {elapsed - duration:+.3f}s", flush=True)
        by_duration = {}
        for r in results:
            by_duration.setdefault(r["duration_s"], []).append(r["overhead_s"])
        print("\noverhead by duration (should be ~constant if playback is realtime):")
        for duration, overheads in sorted(by_duration.items()):
            print(f"  {duration:>4.1f}s: mean {np.mean(overheads):+.3f}s  "
                  f"min {min(overheads):+.3f}s  max {max(overheads):+.3f}s")
        print("\n" + json.dumps(results))
    else:
        pcm = tone(args.seconds, beep_every=5.0)
        elapsed, stalls = play_streamed(pcm)
        drift = elapsed - args.seconds
        print(f"  submitted {args.seconds:.1f}s of audio at realtime pace")
        print(f"  wall {elapsed:.3f}s  drift {drift:+.3f}s")
        print(f"  slowest write blocked {max(stalls) * 1000:.1f}ms  "
              f"(mean {np.mean(stalls) * 1000:.2f}ms)")
        print(f"\n  verdict: {'STALLED' if drift > 1.0 else 'no stall detected'}")


if __name__ == "__main__":
    main()
