"""Tests for chroma_sequence and modulation_spectrum (issue #69)."""

from __future__ import annotations

import numpy as np
import pytest

from wav2tidal.core.dsp.features import chroma_sequence, modulation_spectrum

SR = 22050  # sufficient for 8 kHz upper band edge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def sine(freq: float, seconds: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def am_tone(
    carrier: float,
    mod_rate: float,
    seconds: float,
    amp: float = 0.3,
    mod_depth: float = 0.8,
    sr: int = SR,
) -> np.ndarray:
    """Amplitude-modulated sine: carrier * (1 + mod_depth * sin(2π * mod_rate * t))."""
    t = np.arange(int(sr * seconds)) / sr
    carrier_sig = amp * np.sin(2 * np.pi * carrier * t)
    envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_rate * t)
    return (carrier_sig * envelope).astype(np.float32)


# ---------------------------------------------------------------------------
# chroma_sequence
# ---------------------------------------------------------------------------


def test_chroma_sequence_shape_default():
    """chroma_sequence returns (12, 32) for a normal audio clip."""
    y = sine(440.0, 2.0)
    c = chroma_sequence(y, SR)
    assert c.shape == (12, 32)
    assert c.dtype == np.float64


def test_chroma_sequence_shape_custom_n_frames():
    """n_frames parameter controls the number of output columns."""
    y = sine(440.0, 2.0)
    c = chroma_sequence(y, SR, n_frames=16)
    assert c.shape == (12, 16)


def test_chroma_sequence_shape_short_clip():
    """Shape is (12, 32) even for very short input (padded internally)."""
    y = sine(440.0, 0.05)
    c = chroma_sequence(y, SR)
    assert c.shape == (12, 32)


def test_chroma_sequence_normalized_columns():
    """Each non-zero column has L2 norm ≈ 1.0."""
    y = sine(440.0, 2.0)
    c = chroma_sequence(y, SR)
    for j in range(c.shape[1]):
        col_norm = float(np.linalg.norm(c[:, j]))
        if col_norm > 1e-8:
            assert col_norm == pytest.approx(1.0, abs=1e-6)


def test_chroma_sequence_silence_zero_or_normalized():
    """Silence may yield zero or near-zero columns — no crash, correct shape."""
    y = silence(2.0)
    c = chroma_sequence(y, SR)
    assert c.shape == (12, 32)
    # All columns should be zero or unit-norm (silence → zero expected)
    for j in range(c.shape[1]):
        col_norm = float(np.linalg.norm(c[:, j]))
        assert col_norm < 1.0 + 1e-6


def test_chroma_sequence_harmonic_movement():
    """An A-then-E signal shows argmax switching across frame halves.

    First half of the clip: A4 (440 Hz, pitch class 9).
    Second half: E4 (329.63 Hz, pitch class 4).
    The first-half frame argmax and the second-half frame argmax must differ,
    confirming that chroma_sequence captures harmonic movement.
    """
    dur = 2.0
    a_half = sine(440.0, dur)
    e_half = sine(329.63, dur)
    y = np.concatenate([a_half, e_half])

    c = chroma_sequence(y.astype(np.float32), SR)
    assert c.shape == (12, 32)

    n = c.shape[1]
    first_half = c[:, : n // 2]
    second_half = c[:, n // 2 :]

    # Only consider non-zero columns
    def dominant_pc(block: np.ndarray) -> int | None:
        valid = [
            block[:, j]
            for j in range(block.shape[1])
            if np.linalg.norm(block[:, j]) > 1e-8
        ]
        if not valid:
            return None
        mean_col = np.mean(np.stack(valid, axis=1), axis=1)
        return int(np.argmax(mean_col))

    pc_first = dominant_pc(first_half)
    pc_second = dominant_pc(second_half)

    assert (
        pc_first is not None and pc_second is not None
    ), "All columns zero — can't check harmonic movement"
    assert pc_first != pc_second, (
        f"Expected different dominant pitch classes in each half; "
        f"got {pc_first} and {pc_second}"
    )


# ---------------------------------------------------------------------------
# modulation_spectrum
# ---------------------------------------------------------------------------


def test_modulation_spectrum_shape():
    """modulation_spectrum returns (n_bands * n_points,) for default params."""
    y = am_tone(440.0, 4.0, 4.0)
    m = modulation_spectrum(y, SR)
    assert m.shape == (8 * 24,)
    assert m.dtype == np.float64


def test_modulation_spectrum_shape_custom():
    """Custom n_bands and n_points reshape the output accordingly."""
    y = am_tone(440.0, 4.0, 2.0)
    m = modulation_spectrum(y, SR, n_bands=4, n_points=12)
    assert m.shape == (4 * 12,)


def test_modulation_spectrum_unit_norm():
    """Non-silence output is L2-normalised (unit norm)."""
    y = am_tone(440.0, 4.0, 4.0)
    m = modulation_spectrum(y, SR)
    norm = float(np.linalg.norm(m))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_modulation_spectrum_silence_zero():
    """Silent input yields a zero vector."""
    y = silence(4.0)
    m = modulation_spectrum(y, SR)
    assert m.shape == (8 * 24,)
    assert np.allclose(m, 0.0)


def test_modulation_spectrum_am_discrimination():
    """A 4 Hz AM tone scores higher cosine with another 4 Hz AM (different
    carrier, same log-spaced band) than with an unmodulated tone at the same
    carrier.

    With default n_bands=8, the [~299, ~518] Hz band contains both A4 (440 Hz)
    and E4 (330 Hz).  Both AM signals activate the same band with a 4 Hz
    envelope peak; the unmodulated 440 Hz sine has a flat envelope (only DC,
    which falls below the 0.5 Hz lower bound) → near-zero modulation spectrum.

    This demonstrates that modulation_spectrum is sensitive to AM rate, not
    carrier pitch (issue #69 / `.llm/research-seed-encoding-2-extraction-tools.md`).
    """
    dur = 4.0
    # Both 440 Hz and 330 Hz fall in the same log-spaced band [~299, ~518] Hz.
    ref = am_tone(440.0, 4.0, dur)  # A4 carrier, 4 Hz AM
    same_mod = am_tone(330.0, 4.0, dur)  # E4 carrier (same band), 4 Hz AM
    no_mod = sine(440.0, dur)  # A4 carrier, no AM → flat envelope → near-zero spec

    m_ref = modulation_spectrum(ref, SR)
    m_same = modulation_spectrum(same_mod, SR)
    m_no = modulation_spectrum(no_mod, SR)

    cos_same = float(np.dot(m_ref, m_same))  # both already unit-norm
    cos_no = float(np.dot(m_ref, m_no))  # m_no ≈ zero → cos_no ≈ 0

    assert cos_same > cos_no, (
        f"Expected AM-matched score ({cos_same:.4f}) > unmodulated ({cos_no:.4f}); "
        "modulation_spectrum failed to discriminate AM rate"
    )
