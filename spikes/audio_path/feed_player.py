"""Experiment 1, path B: the WSL-side feeder for win_player.py.

Listens, launches the Windows player, streams PCM to it at realtime pace, and
measures on a single clock:

  start_latency  first byte written -> player reports the device consumed it.
                 This is the fixed cost to subtract from lyrebird's ~1s budget.
  underruns      counted by the player's audio callback -- a real counter, unlike
                 path A, where nothing downstream could be observed at all.

Usage:
    .venv/bin/python spikes/audio_path/feed_player.py --seconds 20
    .venv/bin/python spikes/audio_path/feed_player.py --wav some.wav
"""

from __future__ import annotations

import argparse
import socket
import threading
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
CHUNK_MS = 160
PORT = 47811


def wsl_ip() -> str:
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def windows_path(p: Path) -> str:
    out = subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def tone(seconds: float) -> np.ndarray:
    """Continuous tone with a pitch jump every 5s, so dropouts are audible."""
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f = np.where((t % 5.0) < 0.12, 660.0, 440.0)
    return (0.25 * np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE) * 32767).astype("<i2")


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            raise SystemExit(f"expected {SAMPLE_RATE}Hz mono, got "
                             f"{w.getframerate()}Hz {w.getnchannels()}ch")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--wav", type=Path)
    ap.add_argument("--realtime", action="store_true", default=True,
                    help="pace writes like streamed TTS rather than dumping")
    ap.add_argument("--hostapi", default="WASAPI",
                    help="Windows host API for the player; '-' for PortAudio's default (MME)")
    ap.add_argument("--latency", default="low", help="PortAudio latency hint")
    ap.add_argument("--prefill", type=float, default=0.0,
                    help="ms of audio to buffer before playback starts (jitter buffer)")
    args = ap.parse_args()

    pcm = load_wav(args.wav) if args.wav else tone(args.seconds)
    duration = len(pcm) / SAMPLE_RATE

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(1)
    server.settimeout(30)

    player_script = windows_path(Path(__file__).parent / "win_player.py")
    proc = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command",
         f"python '{player_script}' {wsl_ip()} {PORT} {args.hostapi} {args.latency} "
         f"{args.prefill}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        conn, _ = server.accept()
    except socket.timeout:
        proc.kill()
        out, err = proc.communicate()
        raise SystemExit("player never connected.\n"
                         f"stdout: {out.decode(errors='replace')[:500]}\n"
                         f"stderr: {err.decode(errors='replace')[:500]}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    frames = int(SAMPLE_RATE * CHUNK_MS / 1000)
    replies = b""
    marks: dict = {}

    def reader():
        """Timestamp replies the instant they arrive.

        Polling for the ack inside the send loop would quantise start latency to the
        chunk interval -- which is how a flat ~160ms appeared on every host API before.
        """
        nonlocal replies
        while True:
            try:
                chunk = conn.recv(256)
            except OSError:
                return
            if not chunk:
                return
            replies += chunk
            if b"START" in replies and "start" not in marks:
                marks["start"] = time.perf_counter()
            if b"DONE" in replies:
                marks["done"] = time.perf_counter()
                return

    t0 = time.perf_counter()
    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    try:
        for i in range(0, len(pcm), frames):
            if args.realtime:
                target = t0 + (i / SAMPLE_RATE)
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
            conn.sendall(pcm[i:i + frames].tobytes())
        conn.shutdown(socket.SHUT_WR)
    except (ConnectionResetError, BrokenPipeError):
        proc.wait(timeout=10)
        err = proc.stderr.read().decode(errors="replace")
        raise SystemExit(f"player died mid-stream:\n{err[-1200:]}")

    reader_thread.join(timeout=20)
    elapsed = time.perf_counter() - t0
    conn.close()
    server.close()
    proc.wait(timeout=10)

    text = replies.decode(errors="replace")
    device_latency = underruns = played_frames = device_rate = None
    starved = gaps = None
    for line in text.splitlines():
        if line.startswith("START"):
            device_latency = float(line.split()[1])
        elif line.startswith("DONE"):
            _, underruns, played_frames, device_rate, starved, gaps = line.split()

    start_latency = (marks["start"] - t0) if "start" in marks else None
    print(f"  submitted        {duration:.2f}s of audio"
          f"{' at realtime pace' if args.realtime else ''}")
    print(f"  start latency    {start_latency * 1000:.1f}ms  (send -> device consumed)"
          if start_latency else "  start latency    NOT REPORTED")
    print(f"  device latency   {device_latency:.1f}ms (PortAudio buffer, adds on top)"
          if device_latency is not None else "  device latency   unknown")
    print(f"  underruns        {underruns} (PortAudio flag)")
    if starved is not None and device_rate:
        print(f"  starvation       {int(starved) / int(device_rate) * 1000:.0f}ms of "
              f"silence in {gaps} gaps  <-- the metric that matters")
    if played_frames and device_rate:
        print(f"  played           {int(played_frames) / int(device_rate):.2f}s "
              f"@ {device_rate}Hz")
    print(f"  wall             {elapsed:.2f}s  drift {elapsed - duration:+.2f}s")

    if proc.returncode != 0:
        print(f"\n  player stderr: {proc.stderr.read().decode(errors='replace')[:500]}")


if __name__ == "__main__":
    main()
