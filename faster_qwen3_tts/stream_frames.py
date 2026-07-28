"""Wire format for the streaming synthesis endpoint.

Once the first byte of a streaming response is sent the HTTP status is already
committed, so a mid-stream failure cannot be reported as a 5xx. Framing is what makes
a truncated generation distinguishable from a completed one: a stream that ends
without an END frame is a failed beat, not a short one.

Layout per frame:

    1 byte   frame type
    4 bytes  payload length, big-endian
    N bytes  payload
"""
from __future__ import annotations

import json
import struct
from typing import Any, Iterator, Tuple

FRAME_HEADER = 0x01   # JSON: {sample_rate, channels, format}
FRAME_AUDIO = 0x02    # raw s16le PCM
FRAME_MARK = 0x03     # JSON: {chunk_index, decode_ms, prefill_ms, audio_ms_so_far}
FRAME_ERROR = 0x04    # JSON: {message} -- failure after the response began
FRAME_END = 0x05      # JSON: {total_audio_ms, total_decode_ms}

_PREFIX = struct.Struct(">I")


def encode_frame(frame_type: int, payload: bytes) -> bytes:
    return bytes([frame_type]) + _PREFIX.pack(len(payload)) + payload


def encode_json_frame(frame_type: int, obj: Any) -> bytes:
    return encode_frame(frame_type, json.dumps(obj, separators=(",", ":")).encode())


def iter_frames(data: bytes) -> Iterator[Tuple[int, bytes]]:
    """Decode a complete buffer of frames. Raises ValueError on truncation."""
    offset = 0
    while offset < len(data):
        if len(data) - offset < 5:
            raise ValueError("truncated frame header")
        frame_type = data[offset]
        (length,) = _PREFIX.unpack(data[offset + 1:offset + 5])
        end = offset + 5 + length
        if end > len(data):
            raise ValueError("truncated frame payload")
        yield frame_type, data[offset + 5:end]
        offset = end
