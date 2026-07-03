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

import math

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


def mean_chroma(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Mean-over-time CQT chroma, L2-normalised — harmonic fingerprint (issue #59).

    Returns a 12-element float64 array (one coefficient per pitch class, C..B).
    All-zero input, or a clip whose chroma norm is negligibly small, returns a
    zero vector so callers can detect the unavailable case with a norm check.

    Uses the same ``_chroma`` helper (CQT with STFT fallback) as
    ``estimate_key`` so the two are consistent across the codebase.

    Torch-free — safe to import and call without GPU dependencies.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size < hop_length:
        y = np.pad(y, (0, hop_length - y.size))
    chroma = _chroma(y, sr, hop_length)  # (12, T)
    profile = chroma.mean(axis=1).astype(np.float64)  # (12,)
    norm = float(np.linalg.norm(profile))
    if norm < 1e-8:
        return np.zeros(12, dtype=np.float64)
    return profile / norm


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


# Absolute brightness log-scale bounds (Hz) for descriptor v2 (issue #67).
_BRT_LOG_MIN = math.log(200.0)
_BRT_LOG_MAX = math.log(8000.0)


def _brt_digit(hz: float) -> int:
    """Map a spectral centroid in Hz to a 0–9 digit on a log scale.

    200 Hz → 0, 8 kHz → 9; values outside the range are clamped.
    """
    if hz <= 0.0:
        return 0
    t = (math.log(hz) - _BRT_LOG_MIN) / (_BRT_LOG_MAX - _BRT_LOG_MIN) * 9.0
    return int(round(max(0.0, min(9.0, t))))


def segment_pitch_classes(
    y: np.ndarray,
    sr: int,
    n_segments: int = 8,
    hop_length: int = 512,
) -> tuple[str, ...]:
    """Dominant pitch class per equal-length segment (issue #67, research R3).

    Splits ``y`` into ``n_segments`` equal pieces; for each piece takes the
    mean CQT chromagram frame and returns the PITCH_NAMES name of the
    argmax.  A segment whose chroma L2-norm is below 1e-8 (silence) yields
    ``"-"``.
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    seg_len = max(n // n_segments, 1)
    result: list[str] = []
    for i in range(n_segments):
        seg = y[i * seg_len : (i + 1) * seg_len]
        if seg.size == 0:
            seg = np.zeros(hop_length, dtype=np.float32)
        elif seg.size < hop_length:
            seg = np.pad(seg, (0, hop_length - seg.size))
        ch = _chroma(seg, sr, hop_length)  # (12, T)
        profile = ch.mean(axis=1)
        if float(np.linalg.norm(profile)) < 1e-8:
            result.append("-")
        else:
            result.append(_PITCH_CLASSES[int(np.argmax(profile))])
    return tuple(result)


def quantized_arc(values: np.ndarray) -> str:
    """Map a 1-D array of non-negative per-segment values to a digit string.

    Each value is scaled relative to the array maximum and rounded to 0–9.
    An all-zero array yields all ``"0"`` digits (issue #67, research R3).
    """
    values = np.asarray(values, dtype=np.float64)
    max_val = float(np.max(values))
    if max_val < 1e-12:
        return "0" * len(values)
    return "".join(str(int(round(float(v) * 9.0 / max_val))) for v in values)


def segment_arcs(
    y: np.ndarray,
    sr: int,
    n_segments: int = 8,
    hop_length: int = 512,
) -> dict[str, str]:
    """Per-segment energy, onset-density, and brightness arcs (issue #67).

    Returns a dict with three keys:
    - ``"dyn"``: RMS per segment, quantized relative to the segment-max.
    - ``"ons"``: onset count per segment, quantized relative to the segment-max.
    - ``"brt"``: mean spectral centroid per segment mapped LOG-scale from
      200 Hz to 8 kHz → digit 0–9 (absolute, not relative).
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    seg_len = max(n // n_segments, 1)

    rms_vals = np.zeros(n_segments, dtype=np.float64)
    ons_vals = np.zeros(n_segments, dtype=np.float64)
    brt_digits: list[int] = []

    for i in range(n_segments):
        seg = y[i * seg_len : (i + 1) * seg_len]
        if seg.size == 0:
            seg = np.zeros(hop_length, dtype=np.float32)
        elif seg.size < hop_length:
            seg = np.pad(seg, (0, hop_length - seg.size))

        rms_vals[i] = float(np.sqrt(np.mean(np.square(seg.astype(np.float64)))))

        onsets = librosa.onset.onset_detect(
            y=seg, sr=sr, hop_length=hop_length, units="frames"
        )
        ons_vals[i] = float(len(onsets))

        cent = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=hop_length)
        hz = float(np.mean(cent[cent > 0])) if np.any(cent > 0) else 0.0
        brt_digits.append(_brt_digit(hz))

    return {
        "dyn": quantized_arc(rms_vals),
        "ons": quantized_arc(ons_vals),
        "brt": "".join(str(d) for d in brt_digits),
    }


def descriptor_text_v2(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_segments: int = 8,
) -> str:
    """Descriptor v2: pitch contour + quantized arcs (issue #67, research R3).

    Format (single spaces, fields in this order)::

        tempo=<int> key=<label> pit:<8 names> dyn:<8 d> ons:<8 d> brt:<8 d>

    ``tempo`` and ``key`` are computed identically to ``descriptor_text`` v1.
    ``pit`` is the argmax pitch class per equal-length segment (dominant
    harmony skeleton; ``"-"`` for silent segments).
    ``dyn``/``ons``/``brt`` are energy, onset-density, and brightness arcs
    quantized to single digits 0–9.
    """
    bpm, _ = estimate_tempo(audio, sr, hop_length)
    key, _ = estimate_key(audio, sr, hop_length)
    pit = segment_pitch_classes(audio, sr, n_segments=n_segments, hop_length=hop_length)
    arcs = segment_arcs(audio, sr, n_segments=n_segments, hop_length=hop_length)
    pit_str = " ".join(pit)
    return (
        f"tempo={int(round(bpm))} key={key} "
        f"pit:{pit_str} dyn:{arcs['dyn']} ons:{arcs['ons']} brt:{arcs['brt']}"
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
