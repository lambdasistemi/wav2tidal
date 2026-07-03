"""Feature extraction on synthetic fixtures (T015)."""

from __future__ import annotations

import math
import re

import numpy as np

from wav2tidal.core.dsp.features import (
    _brt_digit,
    descriptor_text_v2,
    estimate_key,
    quantized_arc,
    segment_arcs,
    segment_pitch_classes,
    slice_features,
    track_descriptors,
)
from wav2tidal.core.pattern.key import PITCH_NAMES, parse_key
from wav2tidal.io.wav import clip_fraction, is_silent

SR = 8000  # keep tests fast


def sine(freq: float, sr: int = 22050, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_blocks_are_fixed_length_regardless_of_clip_length():
    short = slice_features(sine(440, seconds=1.0), 22050)
    long = slice_features(sine(440, seconds=3.0), 22050)
    for key in ("mfcc", "chroma", "spectral"):
        assert short[key].shape == long[key].shape


def test_a440_chroma_peaks_at_A():
    feats = slice_features(sine(440.0), 22050)
    # chroma block is [mean(12), std(12)]; mean peak should be pitch-class A (idx 9)
    chroma_mean = feats["chroma"][:12]
    assert int(np.argmax(chroma_mean)) == 9


def test_key_estimate_reports_strength():
    label, strength = estimate_key(sine(440.0), 22050)
    assert isinstance(label, str)
    assert 0.0 <= strength <= 1.0


def test_track_descriptors_shape():
    td = track_descriptors(sine(220.0), 22050)
    assert set(td) == {
        "tempo_bpm",
        "tempo_confidence",
        "key",
        "key_strength",
        "onset_rate",
    }
    assert 0.0 <= td["tempo_confidence"] <= 1.0


def test_silence_and_clip_detection():
    assert is_silent(np.zeros(1000, dtype=np.float32))
    assert not is_silent(sine(440.0))
    clipped = np.ones(1000, dtype=np.float32)
    assert clip_fraction(clipped) > 0.5
    assert clip_fraction(sine(440.0) * 0.5) == 0.0


# ---------------------------------------------------------------------------
# quantized_arc
# ---------------------------------------------------------------------------


def test_quantized_arc_all_zero():
    assert quantized_arc(np.zeros(8)) == "00000000"


def test_quantized_arc_uniform():
    # All equal → all 9 (each value = max, so 9/9*9 = 9)
    result = quantized_arc(np.ones(4))
    assert result == "9999"


def test_quantized_arc_scaling():
    # [0, 1, 2, 3] → 0/3*9=0, 1/3*9=3, 2/3*9=6, 3/3*9=9
    result = quantized_arc(np.array([0.0, 1.0, 2.0, 3.0]))
    assert result == "0369"


def test_quantized_arc_length():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = quantized_arc(vals)
    assert len(result) == 8
    assert result.isdigit()


# ---------------------------------------------------------------------------
# _brt_digit log mapping
# ---------------------------------------------------------------------------


def test_brt_digit_lower_bound():
    assert _brt_digit(200.0) == 0


def test_brt_digit_upper_bound():
    assert _brt_digit(8000.0) == 9


def test_brt_digit_clamp_below():
    assert _brt_digit(10.0) == 0  # below 200 Hz → clamped to 0


def test_brt_digit_clamp_above():
    assert _brt_digit(20000.0) == 9  # above 8 kHz → clamped to 9


def test_brt_digit_zero_hz():
    assert _brt_digit(0.0) == 0


def test_brt_digit_midpoint():
    # geometric midpoint: sqrt(200 * 8000) ≈ 1265 Hz → digit 4 or 5
    mid_hz = math.sqrt(200.0 * 8000.0)
    d = _brt_digit(mid_hz)
    assert d in (4, 5)


# ---------------------------------------------------------------------------
# segment_pitch_classes
# ---------------------------------------------------------------------------


def _pure_tone_segment(freq: float, n_samples: int, sr: int) -> np.ndarray:
    t = np.arange(n_samples) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_segment_pitch_classes_silence_yields_dash():
    y = np.zeros(SR * 4, dtype=np.float32)
    result = segment_pitch_classes(y, SR, n_segments=8)
    assert all(p == "-" for p in result)


def test_segment_pitch_classes_returns_valid_names():
    y = sine(440.0, sr=SR, seconds=4.0)
    result = segment_pitch_classes(y, SR, n_segments=8)
    assert len(result) == 8
    valid = set(PITCH_NAMES) | {"-"}
    for p in result:
        assert p in valid


def test_segment_pitch_classes_a440_dominant_pitch():
    # A 440 Hz → pitch class A should dominate most segments
    y = sine(440.0, sr=SR, seconds=4.0)
    result = segment_pitch_classes(y, SR, n_segments=8)
    assert result.count("A") >= 6


def test_segment_pitch_classes_count():
    y = sine(261.63, sr=SR, seconds=4.0)  # ≈ C4
    result = segment_pitch_classes(y, SR, n_segments=8)
    assert len(result) == 8


def test_segment_pitch_classes_mixed_silence_and_tone():
    silence = np.zeros(SR * 2, dtype=np.float32)
    tone = sine(440.0, sr=SR, seconds=2.0)
    # first half silence, second half A 440 — 4 segments each
    y = np.concatenate([silence, tone])
    result = segment_pitch_classes(y, SR, n_segments=8)
    # first 4 segments should be "-"
    assert all(p == "-" for p in result[:4])
    # last 4 segments should be a valid pitch name
    assert all(p in PITCH_NAMES for p in result[4:])


# ---------------------------------------------------------------------------
# segment_arcs
# ---------------------------------------------------------------------------


def test_segment_arcs_keys():
    y = sine(440.0, sr=SR, seconds=4.0)
    arcs = segment_arcs(y, SR, n_segments=8)
    assert set(arcs) == {"dyn", "ons", "brt"}


def test_segment_arcs_lengths():
    y = sine(440.0, sr=SR, seconds=4.0)
    arcs = segment_arcs(y, SR, n_segments=8)
    assert len(arcs["dyn"]) == 8
    assert len(arcs["ons"]) == 8
    assert len(arcs["brt"]) == 8


def test_segment_arcs_digits_only():
    y = sine(440.0, sr=SR, seconds=4.0)
    arcs = segment_arcs(y, SR, n_segments=8)
    for key in ("dyn", "ons", "brt"):
        assert arcs[key].isdigit(), f"{key!r} arc contains non-digit: {arcs[key]!r}"


def test_segment_arcs_silence_dyn_zero():
    y = np.zeros(SR * 4, dtype=np.float32)
    arcs = segment_arcs(y, SR, n_segments=8)
    assert arcs["dyn"] == "00000000"


# ---------------------------------------------------------------------------
# descriptor_text_v2
# ---------------------------------------------------------------------------

_V2_PATTERN = re.compile(
    r"^tempo=\d+ key=\S+ pit:(\S+ ){7}\S+ dyn:\d{8} ons:\d{8} brt:\d{8}$"
)


def test_descriptor_text_v2_format():
    y = sine(440.0, sr=SR, seconds=4.0)
    desc = descriptor_text_v2(y, SR, n_segments=8)
    assert _V2_PATTERN.match(desc), f"v2 descriptor does not match pattern: {desc!r}"


def test_descriptor_text_v2_deterministic():
    y = sine(440.0, sr=SR, seconds=4.0)
    d1 = descriptor_text_v2(y, SR)
    d2 = descriptor_text_v2(y, SR)
    assert d1 == d2


def test_descriptor_text_v2_has_tempo_and_key():
    y = sine(440.0, sr=SR, seconds=4.0)
    desc = descriptor_text_v2(y, SR)
    assert desc.startswith("tempo=")
    assert "key=" in desc


def test_descriptor_text_v2_parse_key_compat():
    """parse_key must extract the key label from a v2 descriptor (issue #67)."""
    y = sine(440.0, sr=SR, seconds=4.0)
    desc = descriptor_text_v2(y, SR)
    key_label = parse_key(desc)
    # parse_key returns None only for "N/A"; a real tone should have a key
    # (we just assert the function runs and returns a str or None — not None
    # for a deterministic non-silent input would be ideal, but key detection
    # on a short 8kHz sine can legitimately return N/A so we accept either)
    assert key_label is None or isinstance(key_label, str)


def test_descriptor_text_v2_key_compat_nontrivial():
    """parse_key on a v2 descriptor produced from a real tone returns a string."""
    # Use higher sr so chroma is reliable
    y_hq = sine(440.0, sr=22050, seconds=4.0)
    desc = descriptor_text_v2(y_hq, 22050)
    assert _V2_PATTERN.match(desc), f"v2 pattern mismatch: {desc!r}"
    key_label = parse_key(desc)
    assert key_label is not None
    assert isinstance(key_label, str)
