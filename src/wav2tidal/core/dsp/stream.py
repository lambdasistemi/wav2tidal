"""Sliding-window analysis stream for the live pursuit loop (US3-1).

Pure functions over numpy arrays — no file, audio-device, or process IO.
Embed callables are injected by the pipeline layer so this module stays
importable without torch/transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .features import (
    chroma_sequence,
    descriptor_text,
    descriptor_text_v2,
    estimate_tempo,
    mean_chroma,
    modulation_spectrum,
)


@dataclass(frozen=True)
class AnalysisWindow:
    """One analysis window's snapshot for the pursuit engine.

    ``energy`` is the RMS amplitude of the window samples. It is deliberately
    NOT part of ``descriptor_text``: the trained ByT5 model's input format must
    not change mid-corpus. The pursuit engine consumes ``energy`` numerically to
    steer ensemble gain, voice-count, and trajectory depth (see
    specs/001-corpus-to-live-pipeline/us3-live-loop-design.md — "the energy arc").
    """

    t0: float
    t1: float
    descriptor: str
    tempo: float
    energy: float
    # Excluded from eq/hash: numpy arrays are not hashable and equality is
    # element-wise.  Callers compare embeddings via input_jump or np.array_equal.
    embedding: np.ndarray = field(repr=False, compare=False, hash=False)
    # Harmonic target for the pursuit score (issue #59).  Filled by windows()
    # via mean_chroma(); defaults to an empty array so existing direct
    # constructions remain valid without passing this field.
    chroma: np.ndarray = field(
        repr=False,
        compare=False,
        hash=False,
        default_factory=lambda: np.empty(0, dtype=np.float64),
    )
    # Change-aware scoring fields (issue #69).  Filled by windows() via
    # chroma_sequence() and modulation_spectrum(); default to empty arrays so
    # direct AnalysisWindow constructions remain valid without these fields.
    chroma_seq: np.ndarray = field(
        repr=False,
        compare=False,
        hash=False,
        default_factory=lambda: np.empty(0, dtype=np.float64),
    )
    modspec: np.ndarray = field(
        repr=False,
        compare=False,
        hash=False,
        default_factory=lambda: np.empty(0, dtype=np.float64),
    )


def windows(
    y: np.ndarray,
    sr: int,
    *,
    window_s: float = 4.0,
    hop_s: float = 2.0,
    hop_length: int = 512,
    embed: Callable[[np.ndarray, int], np.ndarray | None] | None = None,
    descriptor_version: int = 1,
) -> list[AnalysisWindow]:
    """Slide an analysis window over mono audio ``y`` sampled at ``sr`` Hz.

    ``window_s`` is the window duration in seconds; ``hop_s`` is the step
    between successive window starts.  A partial trailing window is included
    iff its length is >= half a window (``window_s / 2``).

    ``embed`` is an injected callable ``(y, sr) -> np.ndarray | None``.  Pass
    ``None`` for the CI-safe, embedding-free path; if the callable returns
    ``None`` the window's embedding is a zero-length array.

    ``descriptor_version`` selects the descriptor format: 1 (default) uses the
    v1 bucketed format; 2 uses ``descriptor_text_v2`` (pitch contour + arcs,
    issue #67).  Default stays 1 until the retrain ships (#68).
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    win_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)
    half_win = win_samples // 2

    result: list[AnalysisWindow] = []
    start = 0
    while start < n:
        end = min(start + win_samples, n)
        if end - start < half_win:
            break
        win = y[start:end]
        t0 = start / sr
        t1 = end / sr

        bpm, _ = estimate_tempo(win, sr, hop_length)
        energy = float(np.sqrt(np.mean(np.square(win.astype(np.float64)))))
        if descriptor_version == 2:
            desc = descriptor_text_v2(win, sr, hop_length)
        else:
            desc = descriptor_text(win, sr, hop_length)

        raw_emb = embed(win, sr) if embed is not None else None
        emb = (
            np.empty(0, dtype=np.float64)
            if raw_emb is None
            else np.asarray(raw_emb, dtype=np.float64)
        )

        chroma = mean_chroma(win, sr, hop_length)
        chroma_seq = chroma_sequence(win, sr, hop_length)
        modspec = modulation_spectrum(win, sr)

        result.append(
            AnalysisWindow(
                t0=t0,
                t1=t1,
                descriptor=desc,
                tempo=bpm,
                energy=energy,
                embedding=emb,
                chroma=chroma,
                chroma_seq=chroma_seq,
                modspec=modspec,
            )
        )
        start += hop_samples

    return result


def input_jump(
    prev: AnalysisWindow,
    cur: AnalysisWindow,
    *,
    threshold: float = 0.35,
) -> bool:
    """Return True when the input changed drastically between two adjacent windows.

    Three independent signals — any one being True constitutes a jump:

    - **Cosine distance** between embeddings > ``threshold`` (tested only when
      both windows carry a non-empty embedding; DSP-only windows fall back to
      the remaining two gates).
    - **Relative tempo change** > 25 %:  ``|cur.tempo - prev.tempo| / prev.tempo``.
    - **Relative energy change** > 150 %: ``|cur.energy - prev.energy| / prev.energy``.
      A drop to near-silence or an explosion in level both qualify.
    """
    # Embedding gate — cosine distance (only when both windows are embedded)
    if prev.embedding.size > 0 and cur.embedding.size > 0:
        norm_p = np.linalg.norm(prev.embedding)
        norm_c = np.linalg.norm(cur.embedding)
        if norm_p > 1e-8 and norm_c > 1e-8:
            cos_sim = float(np.dot(prev.embedding, cur.embedding) / (norm_p * norm_c))
            if 1.0 - cos_sim > threshold:
                return True

    # Tempo gate (> 25 % relative change)
    if abs(cur.tempo - prev.tempo) / max(abs(prev.tempo), 1e-8) > 0.25:
        return True

    # Energy gate (> 150 % relative change)
    if abs(cur.energy - prev.energy) / max(abs(prev.energy), 1e-8) > 1.50:
        return True

    return False
