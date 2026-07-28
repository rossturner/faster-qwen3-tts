import pytest

from faster_qwen3_tts.stream_frames import (
    FRAME_AUDIO,
    FRAME_END,
    FRAME_HEADER,
    encode_frame,
    encode_json_frame,
    iter_frames,
)


def test_encode_frame_layout():
    out = encode_frame(FRAME_AUDIO, b"abcd")
    assert out[0] == FRAME_AUDIO
    assert out[1:5] == b"\x00\x00\x00\x04"
    assert out[5:] == b"abcd"


def test_round_trip_multiple_frames_in_order():
    blob = (
        encode_json_frame(FRAME_HEADER, {"sample_rate": 24000})
        + encode_frame(FRAME_AUDIO, b"\x01\x02")
        + encode_json_frame(FRAME_END, {"total_audio_ms": 12})
    )
    frames = list(iter_frames(blob))
    assert [f[0] for f in frames] == [FRAME_HEADER, FRAME_AUDIO, FRAME_END]
    assert frames[1][1] == b"\x01\x02"


def test_json_frame_decodes_back_to_object():
    import json
    frames = list(iter_frames(encode_json_frame(FRAME_HEADER, {"channels": 1})))
    assert json.loads(frames[0][1]) == {"channels": 1}


def test_empty_payload_is_valid():
    frames = list(iter_frames(encode_frame(FRAME_END, b"")))
    assert frames == [(FRAME_END, b"")]


def test_truncated_header_raises():
    with pytest.raises(ValueError, match="truncated frame header"):
        list(iter_frames(encode_frame(FRAME_AUDIO, b"abcd")[:4]))


def test_truncated_payload_raises():
    with pytest.raises(ValueError, match="truncated frame payload"):
        list(iter_frames(encode_frame(FRAME_AUDIO, b"abcd")[:8]))
