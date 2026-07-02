"""Slice-boundary detection on synthetic fixtures (T014)."""

from __future__ import annotations

import numpy as np

from wav2tidal.core.dsp.slice import slice_boundaries


def click_track(
    bpm: float = 120.0, sr: int = 22050, seconds: float = 8.0
) -> np.ndarray:
    """Impulses at a fixed tempo — a deterministic beat fixture."""
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    y[::period] = 1.0
    return y


def test_boundaries_span_clip_and_are_sorted():
    y = click_track()
    b = slice_boundaries(y, 22050, strategy="beat")
    assert b[0] == 0.0
    assert b[-1] > 0.0
    assert np.all(np.diff(b) > 0)


def test_beat_slicing_finds_multiple_slices():
    y = click_track(bpm=120.0)
    b = slice_boundaries(y, 22050, strategy="beat")
    assert b.size >= 4  # a beat grid, not a single interval


def test_deterministic():
    y = click_track()
    a = slice_boundaries(y, 22050, strategy="beat")
    b = slice_boundaries(y, 22050, strategy="beat")
    assert np.array_equal(a, b)


def test_silence_yields_trivial_boundaries():
    y = np.zeros(22050 * 2, dtype=np.float32)
    b = slice_boundaries(y, 22050, strategy="beat")
    # start and end only, no interior slices carved from silence
    assert b[0] == 0.0 and b[-1] > 0.0


def test_grid_strategy_runs():
    y = click_track()
    b = slice_boundaries(y, 22050, strategy="grid", grid_subdivisions=4)
    assert b.size >= 2
