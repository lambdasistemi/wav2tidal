"""Offline session assembly (US3-2, issue #51).

For the file-in/file-out replay milestone: the pursuit engine's winning
scenes are rendered individually (NRT); this stitches them into one
timeline — each winner placed at its generation's start time, summed
where they overlap, peak-normalized once at the end so generations are
level-consistent with each other. Pure numpy — no IO.
"""

from __future__ import annotations

import numpy as np

_NORM_PEAK = 0.891  # -1 dBFS, matching the renderers


def assemble(
    placements: list[tuple[float, np.ndarray]],
    total_seconds: float,
    sr: int,
    peak: float = _NORM_PEAK,
) -> np.ndarray:
    """Sum ``(start_s, stereo_or_mono_signal)`` clips onto one canvas.

    Returns float32 stereo of ``total_seconds`` (clips are clipped at the
    canvas end; mono clips are duplicated to stereo).
    """
    n = max(1, int(round(total_seconds * sr)))
    out = np.zeros((n, 2), dtype=np.float64)
    for start_s, clip in placements:
        c = np.asarray(clip, dtype=np.float64)
        if c.ndim == 1:
            c = np.stack([c, c], axis=1)
        start = int(round(start_s * sr))
        if start >= n or c.size == 0:
            continue
        end = min(n, start + len(c))
        out[start:end] += c[: end - start]
    m = float(np.abs(out).max())
    if m > 0:
        out *= peak / m
    return out.astype(np.float32)
