"""Feature extraction and descriptor blocks (T019).

Pure functions mapping a mono float signal to fixed-length hand-crafted
descriptor blocks (research R4). Per-frame features are pooled with
mean+std so a clip of any length yields the same-shaped vector.

librosa exposes neither a tempo *confidence* nor a key estimator, so both
are derived here: confidence from the onset-autocorrelation peak, key from
a Krumhansl-Schmuckler correlation over the pooled chromagram. Both report
their strength honestly rather than feigning precision.
"""

from __future__ import annotations

import librosa
import librosa.feature.rhythm  # noqa: F401  (register lazy-loaded submodule)
import numpy as np

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (major, minor), rotated to find the key.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _pool(feat: np.ndarray) -> np.ndarray:
    """Mean+std pooling over the time axis -> a fixed-length vector."""
    feat = np.atleast_2d(feat)
    return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])


def slice_features(
    clip: np.ndarray, sr: int, hop_length: int = 512, n_mfcc: int = 20
) -> dict[str, np.ndarray]:
    """Hand-crafted descriptor blocks for one clip (pure)."""
    clip = np.asarray(clip, dtype=np.float32)
    if clip.size < hop_length:
        clip = np.pad(clip, (0, hop_length - clip.size))
    mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=clip, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=clip, sr=sr, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=clip, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=clip, sr=sr, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=clip, hop_length=hop_length)
    spectral = np.vstack([centroid, bandwidth, rolloff, flatness])
    return {
        "mfcc": _pool(mfcc),
        "chroma": _pool(chroma),
        "spectral": _pool(spectral),
    }


def onset_rate(clip: np.ndarray, sr: int, hop_length: int = 512) -> float:
    """Rhythmic density: onsets per second (R4)."""
    duration = len(clip) / sr
    if duration <= 0:
        return 0.0
    onsets = librosa.onset.onset_detect(
        y=clip, sr=sr, hop_length=hop_length, units="time"
    )
    return float(np.atleast_1d(onsets).size) / duration


def estimate_tempo(
    clip: np.ndarray, sr: int, hop_length: int = 512
) -> tuple[float, float]:
    """Return (bpm, confidence). Confidence is the normalized onset-autocorr peak."""
    onset_env = librosa.onset.onset_strength(y=clip, sr=sr, hop_length=hop_length)
    tempo = librosa.feature.rhythm.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    bpm = float(np.atleast_1d(tempo)[0])
    if onset_env.size < 2 or float(np.max(np.abs(onset_env))) < 1e-8:
        return bpm, 0.0
    ac = librosa.autocorrelate(onset_env)
    if ac[0] <= 0:
        return bpm, 0.0
    ac_norm = ac / ac[0]
    # peak strength away from lag 0 as a rough confidence proxy
    confidence = float(
        np.clip(np.max(ac_norm[1:]) if ac_norm.size > 1 else 0.0, 0.0, 1.0)
    )
    return bpm, confidence


def _chroma(clip: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """CQT chroma (better for key), with an STFT fallback for short/low-sr clips."""
    try:
        return librosa.feature.chroma_cqt(y=clip, sr=sr, hop_length=hop_length)
    except librosa.util.exceptions.ParameterError:
        return librosa.feature.chroma_stft(y=clip, sr=sr, hop_length=hop_length)


def estimate_key(clip: np.ndarray, sr: int, hop_length: int = 512) -> tuple[str, float]:
    """Krumhansl-Schmuckler key estimate -> (label, correlation strength)."""
    chroma = _chroma(clip, sr, hop_length)
    profile = chroma.mean(axis=1)
    if float(np.linalg.norm(profile)) < 1e-8:
        return "N/A", 0.0
    best_label, best_corr = "N/A", -2.0
    for i in range(12):
        maj = np.corrcoef(np.roll(_KS_MAJOR, i), profile)[0, 1]
        minr = np.corrcoef(np.roll(_KS_MINOR, i), profile)[0, 1]
        if maj > best_corr:
            best_label, best_corr = f"{_PITCH_CLASSES[i]}", float(maj)
        if minr > best_corr:
            best_label, best_corr = f"{_PITCH_CLASSES[i]}m", float(minr)
    return best_label, float(np.clip(best_corr, 0.0, 1.0))


def track_descriptors(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """Track-level scalar descriptors (tempo/confidence/key/density)."""
    bpm, tempo_conf = estimate_tempo(y, sr, hop_length)
    key, key_strength = estimate_key(y, sr, hop_length)
    return {
        "tempo_bpm": bpm,
        "tempo_confidence": tempo_conf,
        "key": key,
        "key_strength": key_strength,
        "onset_rate": onset_rate(y, sr, hop_length),
    }


_DENSITY_EDGES = (2.0, 6.0)  # onsets/sec -> lo / mid / hi
_BRIGHT_EDGES = (1500.0, 3000.0, 5000.0, 7000.0)  # spectral centroid Hz -> 1..5


def _bucket(value: float, edges) -> int:
    return sum(1 for e in edges if value >= e)


def _motion_label(ratio: float, wobble: float) -> str:
    if ratio >= 1.25:
        return "rising"
    if ratio <= 0.8:
        return "falling"
    return "wobbly" if wobble >= 0.15 else "steady"


def descriptor_text(audio: np.ndarray, sr: int, hop_length: int = 512) -> str:
    """Compact, bucketed description of a rendered clip — the model input.

    ``motion`` is the movement-aware field (issue #30): the direction/
    oscillation of the spectral-centroid track, so a filter sweep and a
    fixed timbre of equal average brightness get different descriptions.
    """
    bpm, _ = estimate_tempo(audio, sr, hop_length)
    key, _ = estimate_key(audio, sr, hop_length)
    density = onset_rate(audio, sr, hop_length)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    density_label = ("lo", "mid", "hi")[_bucket(density, _DENSITY_EDGES)]
    brightness = _bucket(centroid, _BRIGHT_EDGES) + 1
    motion = _motion_label(*centroid_motion(audio, sr, hop_length))
    return (
        f"tempo={int(round(bpm))} density={density_label} "
        f"key={key} brightness={brightness}/5 motion={motion}"
    )


def centroid_motion(
    clip: np.ndarray, sr: int, hop_length: int = 512
) -> tuple[float, float]:
    """Timbre movement of a clip: (trend ratio, wobble coefficient).

    Trend ratio = mean centroid of the last third over the first third
    (>1 = brightening); wobble = normalized std of the detrended centroid
    track (how much the brightness oscillates around its trend). The
    movement-aware descriptor buckets these (issue #30) — a static
    brightness figure cannot distinguish a filter sweep from a fixed
    timbre.
    """
    import librosa

    cent = librosa.feature.spectral_centroid(y=clip, sr=sr, hop_length=hop_length)[0]
    cent = cent[cent > 0]
    if cent.size < 9:
        return 1.0, 0.0
    third = cent.size // 3
    start, end = float(cent[:third].mean()), float(cent[-third:].mean())
    ratio = end / start if start > 0 else 1.0
    x = np.arange(cent.size, dtype=np.float64)
    trend = np.polyval(np.polyfit(x, cent, 1), x)
    wobble = float(np.std(cent - trend) / max(np.mean(cent), 1e-9))
    return ratio, wobble
