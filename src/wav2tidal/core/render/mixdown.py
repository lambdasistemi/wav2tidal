"""Offline slice-mixdown renderer (T032, FR-012).

Deterministic mono render of a scheduled pattern by mixing bank slices at
their event times, applying playback rate (speed) and gain. Pan is carried
in the schedule for the live path but folded to mono here (v1 renders mono;
the embedder downmixes anyway). No live audio infrastructure required.

``Banks`` is a pure in-memory sample set so the renderer stays testable
without disk; the IO loader lives in ``io.banks``.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .schedule import Event


@dataclass(frozen=True)
class Banks:
    """In-memory sample banks: name -> ordered slice signals, at ``sr``."""

    sr: int
    data: dict[str, list[np.ndarray]]

    def inventory(self) -> dict[str, int]:
        return {name: len(slices) for name, slices in self.data.items()}

    def get(self, bank: str, index: int) -> np.ndarray:
        slices = self.data[bank]
        return slices[index % len(slices)]  # SuperDirt wraps out-of-range :n


def _apply_speed(slice_: np.ndarray, speed: float, sr: int) -> np.ndarray:
    if speed == 1.0 or slice_.size == 0:
        return slice_
    orig_sr = max(1, int(round(sr * speed)))  # speed>1 shortens (higher pitch)
    return librosa.resample(slice_, orig_sr=orig_sr, target_sr=sr, res_type="soxr_hq")


def render(
    events: list[Event], banks: Banks, total_seconds: float, sr: int
) -> np.ndarray:
    """Render events to a mono float32 signal of length ``total_seconds``."""
    out = np.zeros(max(1, int(round(total_seconds * sr))), dtype=np.float64)
    for ev in events:
        clip = _apply_speed(banks.get(ev.bank, ev.index), ev.speed, sr) * ev.gain
        start = int(round(ev.start * sr))
        if start >= out.size:
            continue
        end = min(out.size, start + clip.size)
        out[start:end] += clip[: end - start]
    return np.clip(out, -1.0, 1.0).astype(np.float32)
