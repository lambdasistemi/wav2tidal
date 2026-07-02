"""Feature extraction on synthetic fixtures (T015)."""

from __future__ import annotations

import numpy as np

from wav2tidal.core.dsp.features import (
    estimate_key,
    slice_features,
    track_descriptors,
)
from wav2tidal.io.wav import clip_fraction, is_silent


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
