"""WAV IO edge (T017).

The only place audio enters the system. Reads with soundfile, downmixes to
mono, resamples to the pipeline's target rate, and validates (silence /
clipping / corruption). Corrupt files raise ``LibsndfileError`` here and are
handled by the caller; the pure core never sees a bad signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# re-exported so callers catch corruption without importing soundfile
LibsndfileError = sf.LibsndfileError


@dataclass(frozen=True)
class LoadedAudio:
    y: np.ndarray  # mono float32 at target_sr
    sr: int  # == target_sr
    orig_sr: int
    orig_channels: int


def to_mono(y: np.ndarray) -> np.ndarray:
    """Downmix to mono by channel mean. Accepts (n,) or (n, channels)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    return y.mean(axis=1).astype(np.float32)


def is_silent(y: np.ndarray, rms_threshold: float = 1e-4) -> bool:
    if y.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(y)))) < rms_threshold


def clip_fraction(y: np.ndarray, eps: float = 1e-4) -> float:
    """Fraction of samples at full scale (a clipping proxy)."""
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= 1.0 - eps))


def read_wav(path: str | Path, target_sr: int) -> LoadedAudio:
    """Read, downmix to mono, resample to target_sr. Raises on unreadable files."""
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    orig_channels = data.shape[1]
    mono = to_mono(data)
    if orig_sr != target_sr:
        mono = librosa.resample(
            mono, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq"
        )
    return LoadedAudio(
        y=mono, sr=target_sr, orig_sr=orig_sr, orig_channels=orig_channels
    )


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(y, dtype=np.float32), sr, subtype="PCM_16")
