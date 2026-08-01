"""The chorus filter, and the character.yaml surface that declares it.

The load-bearing property is chunk invariance: the streaming endpoint chunks however the
caller asks, so a filter whose output depended on chunk size would seam audibly several
times a second. That is asserted against the real chunk sizes the server uses, not a
convenient one.
"""
import numpy as np
import pytest
import yaml

from faster_qwen3_tts.audio_filter import ChorusSpec, StreamingChorus
from faster_qwen3_tts.characters import load_characters


def _speech_like(n, sr=24000, seed=0):
    """A voiced signal with harmonics -- silence would hide pitch-shift bugs."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    f0 = 130 + 12 * np.sin(2 * np.pi * 1.7 * t)
    sig = sum(np.sin(2 * np.pi * h * np.cumsum(f0) / sr) / h for h in range(1, 9))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)
    out = sig * env + 0.01 * rng.standard_normal(n)
    # Scaled to a realistic peak: summing harmonics reaches ~2.7, which no real signal
    # does, and would sit permanently inside the soft knee.
    return (out / np.abs(out).max() * 0.6).astype(np.float32)


@pytest.mark.parametrize("chunk", [2000, 6984, 512, 1])
def test_output_is_identical_however_it_is_chunked(chunk):
    x = _speech_like(24000)
    whole = StreamingChorus(24000).process(x)
    streamed = StreamingChorus(24000)
    pieces = [streamed.process(x[i:i + chunk]) for i in range(0, len(x), chunk)]
    got = np.concatenate([p for p in pieces if len(p)])
    n = min(len(whole), len(got))
    assert n > 0
    assert np.abs(whole[:n] - got[:n]).max() < 1e-6


def test_apply_preserves_length_and_changes_the_audio():
    x = _speech_like(12000)
    y = StreamingChorus(24000).apply(x)
    assert len(y) == len(x)
    # the tail must actually be flushed, not zero-padded away
    assert np.abs(y[-1024:]).max() > 0.01
    # and the start must be real audio, not the ramp-in window
    assert np.abs(y[:1024]).max() > 0.01
    assert np.abs(y - x).max() > 0.01


def test_flush_releases_the_window_the_filter_was_holding():
    x = _speech_like(12000)
    c = StreamingChorus(24000)
    streamed = len(c.process(x))
    tail = len(c.flush())
    assert tail > 0, "without a flush the last window of every line is lost"
    assert streamed + tail >= len(x)


def test_amount_zero_leaves_the_signal_alone():
    """Pins the alignment: with no wet signal and no makeup the output must equal the
    input sample for sample, which only holds if the ramp-in window is dropped.

    makeup_db is passed as 0 because it is a declared constant measured for the default
    amount, not something derived from it -- see the coupling test below.
    """
    x = _speech_like(12000)
    y = StreamingChorus(24000, ChorusSpec(amount=0.0, makeup_db=0.0)).apply(x)
    assert len(y) == len(x)
    assert np.abs(y - x).max() < 1e-5


def test_makeup_is_declared_not_derived_from_amount():
    """A footgun worth pinning: makeup_db compensates the level lost at a PARTICULAR
    amount. Changing amount without re-measuring makeup leaves the voice mismatched, so
    the two must be edited together."""
    quiet = StreamingChorus(24000, ChorusSpec(amount=0.05))
    assert quiet.makeup == pytest.approx(10 ** (5.7 / 20), abs=1e-6), \
        "makeup does not track amount; character.yaml must set both"


def test_the_filter_is_loudness_neutral():
    """Billy sits next to unfiltered characters, so the filter must not change level.

    It is not neutral by construction: the copies are pitch-shifted, so they are
    decorrelated from the dry and sum as power rather than amplitude -- measured at
    -5.66 dB before makeup, which was audible as Billy being quieter than everyone else.
    """
    x = _speech_like(24000 * 2) * 0.3
    y = StreamingChorus(24000).apply(x)
    change = 20 * np.log10(np.sqrt((y ** 2).mean()) / np.sqrt((x ** 2).mean()))
    assert abs(change) < 1.0, f"level moved by {change:+.2f} dB"


def test_makeup_gain_cannot_push_past_full_scale():
    """The chorus raises crest factor as well as lowering RMS, so restoring the level
    overshoots on transients. The soft knee has to catch that without hard clipping."""
    x = _speech_like(24000) * 0.95
    y = StreamingChorus(24000).apply(x)
    assert np.abs(y).max() <= 1.0


def test_soft_clip_is_memoryless_so_it_cannot_break_chunk_invariance():
    x = _speech_like(24000) * 0.95           # loud enough to be driven into the knee
    whole = StreamingChorus(24000).process(x)
    c = StreamingChorus(24000)
    got = np.concatenate([p for p in (c.process(x[i:i + 700])
                                      for i in range(0, len(x), 700)) if len(p)])
    n = min(len(whole), len(got))
    assert np.abs(whole[:n] - got[:n]).max() < 1e-6


def test_latency_is_reported_in_milliseconds():
    assert ChorusSpec().latency_ms(24000) == pytest.approx(42.67, abs=0.01)
    assert ChorusSpec().latency_ms(48000) == pytest.approx(42.67, abs=0.01)


def test_the_window_is_the_same_duration_at_every_sample_rate():
    """The bug this pins: a window fixed in SAMPLES is a different filter per rate.
    2048 samples is 42.7 ms at 48 kHz but 85.3 ms at 24 kHz, and those were heard as two
    different settings -- one chosen, one rejected as too far apart."""
    spec = ChorusSpec()
    assert spec.n_fft(24000) == 1024
    assert spec.n_fft(48000) == 2048
    assert spec.latency_ms(24000) == pytest.approx(spec.latency_ms(48000), abs=0.1)


@pytest.mark.parametrize("bad, message", [
    ({"amount": 1.4}, "between 0 and 1"),
    ({"cents": [26, -26], "delays_ms": [8]}, "one delay per copy"),
    ({"window_ms": 900}, "between 5 and 500"),
    ({"makeup_db": 40}, "between -24 and 24"),
    ({"type": "flanger"}, "unknown filter type"),
    ({"wobble": 3}, "unknown filter key"),
    ({"cents": []}, "at least one entry"),
    ({"delays_ms": [-1], "cents": [26]}, "must not be negative"),
])
def test_bad_filter_config_is_rejected_with_a_useful_message(bad, message):
    with pytest.raises(ValueError, match=message):
        ChorusSpec.from_config(bad, "character.yaml")


def _character(tmp_path, yaml_text=None):
    d = tmp_path / "billy" / "neutral"
    d.mkdir(parents=True)
    import soundfile as sf
    sf.write(d / "a.wav", _speech_like(24000 * 3), 24000)
    (d / "a.txt").write_text("hello there")
    if yaml_text is not None:
        (tmp_path / "billy" / "character.yaml").write_text(yaml_text)
    return tmp_path


def test_character_yaml_declares_the_filter(tmp_path):
    root = _character(tmp_path, yaml.safe_dump(
        {"language": "English",
         "filter": {"type": "chorus", "cents": [26, -26], "delays_ms": [8, 16],
                    "amount": 0.55, "window_ms": 42.7, "makeup_db": 5.7}}))
    cfg = load_characters(root, 0.7).resolve("billy", "neutral")
    assert cfg.audio_filter == ChorusSpec((26.0, -26.0), (8.0, 16.0), 0.55, 42.7, 5.7)


def test_a_character_without_a_filter_gets_none(tmp_path):
    cfg = load_characters(_character(tmp_path), 0.7).resolve("billy", "neutral")
    assert cfg.audio_filter is None


def test_a_broken_filter_block_is_fatal_not_skipped(tmp_path):
    """character.yaml is deliberate config, so a typo must not be shrugged off."""
    root = _character(tmp_path, yaml.safe_dump({"filter": {"amount": 9}}))
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_characters(root, 0.7)


def test_shipped_billy_declares_the_settled_filter():
    from pathlib import Path
    p = (Path(__file__).resolve().parents[1] / "faster_qwen3_tts" / "server_voices" /
         "characters" / "billy" / "character.yaml")
    spec = ChorusSpec.from_config(yaml.safe_load(p.read_text())["filter"], str(p))
    assert spec == ChorusSpec((26.0, -26.0), (8.0, 16.0), 0.55, 42.7, 5.7)
    assert spec.n_fft(24000) == 1024, "the server runs at 24 kHz"
