"""Experiment 1, path B: the Windows-side audio player sidecar.

Runs on the Windows host under Windows Python. **Connects out to a listener in WSL**
rather than listening itself -- outbound connections need no Windows Firewall rule,
inbound ones prompt. The WSL side is the server; this pulls PCM from it.

Speaks a deliberately minimal protocol so the spike measures audio, not framing:

    <-  raw s16le PCM, 24 kHz mono, until the server closes
    ->  "START <device_latency_ms>\n"   once the device consumes the first samples
    ->  "DONE <underruns> <frames>\n"   once the stream has drained

The server times its own send -> START on a single clock, which sidesteps the fact
that WSL and Windows clocks are not comparable.

Resampling is mandatory, not an optimisation: WASAPI shared mode only accepts the
device's mix rate (48 kHz here) and rejects the TTS's 24 kHz outright with
"Invalid sample rate". MME accepts 24 kHz but costs ~180ms of buffer.

Usage (driven by feed_player.py):
    python win_player.py <wsl-ip> <port> [hostapi|-] [latency]
"""

from __future__ import annotations

import queue
import socket
import sys
import threading

import numpy as np
import sounddevice as sd

SRC_RATE = 24000
CHANNELS = 1
BLOCK_MS = 20


class Resampler:
    """Linear-interpolating resampler that survives chunk boundaries.

    Tracks position in the global input stream so successive chunks join without a
    discontinuity at each seam -- a per-chunk resampler clicks audibly at every join,
    which would be indistinguishable from the dropouts this experiment is counting.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        self.ratio = dst_rate / src_rate
        self.buf = np.zeros(0, dtype=np.float32)
        self.base = 0      # global input index of buf[0]
        self.n_out = 0     # next output sample index

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        if len(self.buf) < 2:
            return np.zeros(0, dtype=np.int16)
        last_pos = self.base + len(self.buf) - 1
        n_max = int(np.floor(last_pos * self.ratio))
        if n_max < self.n_out:
            return np.zeros(0, dtype=np.int16)
        out_idx = np.arange(self.n_out, n_max + 1)
        positions = out_idx / self.ratio - self.base
        out = np.interp(positions, np.arange(len(self.buf)), self.buf)
        self.n_out = n_max + 1
        consumed = int(np.floor(positions[-1]))
        if consumed > 0:
            self.buf = self.buf[consumed:]
            self.base += consumed
        return out.astype(np.int16)


def resolve_device(hostapi_name: str | None):
    """Pick the default output device of a named host API (e.g. 'WASAPI').

    Windows' default host API is MME, whose buffers are ~90-180ms -- most of a live
    latency budget spent before any sound exists. WASAPI reports ~2.7ms.
    """
    if not hostapi_name:
        return None
    for api in sd.query_hostapis():
        if hostapi_name.lower() in api["name"].lower():
            if api["default_output_device"] < 0:
                raise SystemExit(f"host API {api['name']} has no default output")
            return api["default_output_device"]
    raise SystemExit(f"no host API matching {hostapi_name!r}")


def main() -> None:
    host, port = sys.argv[1], int(sys.argv[2])
    hostapi = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    latency = sys.argv[4] if len(sys.argv) > 4 else "low"
    prefill_ms = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    device = resolve_device(hostapi)
    info = sd.query_devices(device if device is not None else sd.default.device[1])
    dst_rate = int(info["default_samplerate"])
    resample = Resampler(SRC_RATE, dst_rate) if dst_rate != SRC_RATE else None

    sock = socket.create_connection((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    pcm_q: queue.Queue = queue.Queue()
    state = {"underruns": 0, "frames": 0, "starved": 0, "gaps": 0,
             "started": False, "eof": False}
    pending = np.zeros(0, dtype=np.int16)
    first_audio = threading.Event()
    prefill_frames = int(dst_rate * prefill_ms / 1000)

    def callback(outdata, frames, _time, status):
        nonlocal pending
        if status.output_underflow:
            state["underruns"] += 1
        while len(pending) < max(frames, prefill_frames if not state["started"] else 0):
            try:
                pending = np.concatenate([pending, pcm_q.get_nowait()])
            except queue.Empty:
                break
        # Hold playback until the jitter buffer is primed. Starting on the first sample
        # means the device and the feeder run at the same rate with zero slack, so any
        # jitter starves the stream.
        if not state["started"]:
            if len(pending) < prefill_frames and not state["eof"]:
                outdata[:] = b"\x00" * (frames * 2)
                return
            state["started"] = True
            first_audio.set()
        take = min(frames, len(pending))
        buf = np.zeros(frames, dtype=np.int16)
        buf[:take] = pending[:take]
        pending = pending[take:]
        outdata[:] = buf.tobytes()
        state["frames"] += take
        # Zero-fill after playback has begun is starvation. PortAudio's output_underflow
        # flag does NOT catch this -- the callback returned on time, just with silence --
        # so counting it here is the only way to see it.
        if take < frames and not state["eof"]:
            state["starved"] += frames - take
            state["gaps"] += 1
        if state["eof"] and take == 0:
            raise sd.CallbackStop

    stream = sd.RawOutputStream(
        samplerate=dst_rate, channels=CHANNELS, dtype="int16",
        blocksize=int(dst_rate * BLOCK_MS / 1000), device=device,
        latency=latency, callback=callback)

    def report_start():
        first_audio.wait()
        sock.sendall(f"START {stream.latency * 1000:.1f}\n".encode())

    threading.Thread(target=report_start, daemon=True).start()

    tail = b""
    with stream:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            tail += data
            usable = len(tail) - (len(tail) % 2)
            samples = np.frombuffer(tail[:usable], dtype="<i2")
            tail = tail[usable:]
            pcm_q.put(resample(samples) if resample else samples.copy())
        state["eof"] = True
        while not pcm_q.empty() or len(pending):
            sd.sleep(20)
        sd.sleep(int(stream.latency * 1000) + 50)

    sock.sendall(f"DONE {state['underruns']} {state['frames']} {dst_rate} "
                 f"{state['starved']} {state['gaps']}\n".encode())
    sock.close()


if __name__ == "__main__":
    main()
