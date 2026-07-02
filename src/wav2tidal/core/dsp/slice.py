"""Slice-boundary detection (T018).

Pure functions over a mono float signal. Two-tier strategy with a
documented fallback (research R4): compute the onset envelope once, take
beat boundaries (``beat_track``) or onset-refined boundaries
(``onset_detect(backtrack=True)``), and fall back to fixed tempo
subdivisions or energy segmentation when no reliable beat is found.

Deterministic given fixed inputs and pinned librosa. No RNG.
"""

from __future__ import annotations

import librosa
import numpy as np


def _beats_ok(beats: np.ndarray, duration: float) -> bool:
    # A usable beat grid has at least a few beats spread over the clip.
    return beats.size >= 4 and (beats[-1] - beats[0]) > 0.25 * duration


def slice_boundaries(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    strategy: str = "beat",
    beats_per_slice: int = 1,
    grid_subdivisions: int = 4,
    silence_top_db: float = 40.0,
) -> np.ndarray:
    """Return slice-boundary times in seconds, inclusive of 0.0 and the end.

    ``strategy`` is "beat", "onset", or "grid"; any strategy falls back to
    energy segmentation when it yields no usable boundaries. The returned
    array always starts at 0.0 and ends at the clip duration, with at least
    one interval.
    """
    y = np.asarray(y, dtype=np.float32)
    duration = len(y) / sr
    if duration <= 0:
        return np.array([0.0])

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    times: np.ndarray | None = None
    if strategy == "beat":
        _tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            units="time",
            trim=False,
        )
        beats = np.atleast_1d(beats)
        if _beats_ok(beats, duration):
            times = beats[:: max(1, beats_per_slice)]
    elif strategy == "onset":
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            units="time",
            backtrack=True,
        )
        onsets = np.atleast_1d(onsets)
        if onsets.size >= 2:
            times = onsets
    elif strategy == "grid":
        times = _grid(onset_env, sr, hop_length, duration, grid_subdivisions)
    else:
        raise ValueError(f"unknown slice strategy: {strategy!r}")

    if times is None or np.asarray(times).size < 1:
        times = _energy_segments(y, sr, hop_length, silence_top_db)

    return _finalize(np.asarray(times, dtype=np.float64), duration)


def _grid(onset_env, sr, hop_length, duration, subdivisions) -> np.ndarray:
    tempo = librosa.feature.rhythm.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    bpm = float(np.atleast_1d(tempo)[0]) or 120.0
    beat_seconds = 60.0 / bpm
    step = beat_seconds / max(1, subdivisions)
    if step <= 0:
        return np.array([0.0])
    return np.arange(0.0, duration, step)


def _energy_segments(y, sr, hop_length, top_db) -> np.ndarray:
    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=2 * hop_length, hop_length=hop_length
    )
    if intervals.size == 0:
        return np.array([0.0])
    return intervals[:, 0].astype(np.float64) / sr


def _finalize(times: np.ndarray, duration: float) -> np.ndarray:
    times = times[(times >= 0.0) & (times < duration)]
    times = np.unique(np.concatenate([[0.0], times, [duration]]))
    return times
