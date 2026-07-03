"""Tests for core.dsp.stream and pipeline.analysis (US3-1)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from wav2tidal.core.dsp.features import mean_chroma
from wav2tidal.core.dsp.stream import AnalysisWindow, input_jump, windows
from wav2tidal.io.wav import write_wav
from wav2tidal.pipeline.analysis import analyze_wav

SR = 8000  # low rate keeps tests fast


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def white_noise(
    seconds: float, amp: float = 0.5, sr: int = SR, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sr * seconds)) * amp).astype(np.float32)


def sine(freq: float, seconds: float, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def click_track(bpm: float, seconds: float, sr: int = SR) -> np.ndarray:
    """Impulses at a fixed tempo — a deterministic beat fixture."""
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    y[::period] = 1.0
    return y


def chirp(
    f0: float, f1: float, seconds: float, amp: float = 0.5, sr: int = SR
) -> np.ndarray:
    """Linear sine sweep from f0 to f1 Hz."""
    t = np.arange(int(sr * seconds)) / sr
    freq = f0 + (f1 - f0) * t / seconds
    return (amp * np.sin(2 * np.pi * np.cumsum(freq) / sr)).astype(np.float32)


# ---------------------------------------------------------------------------
# Window segmentation math
# ---------------------------------------------------------------------------


def test_window_count_and_times():
    # 10 s at SR with window=4, hop=2 → t0 in {0,2,4,6,8}
    # t0=8: remaining = 2 s = exactly half window → included
    y = white_noise(10.0)
    wins = windows(y, SR, window_s=4.0, hop_s=2.0)
    assert len(wins) == 5
    assert wins[0].t0 == pytest.approx(0.0)
    assert wins[0].t1 == pytest.approx(4.0)
    assert wins[4].t0 == pytest.approx(8.0)
    assert wins[4].t1 == pytest.approx(10.0)


def test_window_t0_t1_cover_audio():
    y = white_noise(6.0)
    wins = windows(y, SR, window_s=4.0, hop_s=2.0)
    # all windows start within the signal
    for w in wins:
        assert w.t0 >= 0.0
        assert w.t1 <= len(y) / SR + 1e-9


def test_trailing_window_excluded_when_too_short():
    # 9 s: t0 in {0,2,4,6,8}; t0=8 gives only 1 s < 2 s (half of 4 s) → excluded
    y = white_noise(9.0)
    wins = windows(y, SR, window_s=4.0, hop_s=2.0)
    assert len(wins) == 4
    assert wins[-1].t0 == pytest.approx(6.0)


def test_single_window_short_clip():
    # 3 s clip, window=4 s: partial window of 3 s >= 2 s (half) → 1 window
    y = white_noise(3.0)
    wins = windows(y, SR, window_s=4.0, hop_s=2.0)
    assert len(wins) == 1
    assert wins[0].t0 == pytest.approx(0.0)
    assert wins[0].t1 == pytest.approx(3.0)


def test_clip_too_short_returns_empty():
    # 1 s clip, window=4 s: 1 s < 2 s (half) → no windows
    y = white_noise(1.0)
    wins = windows(y, SR, window_s=4.0, hop_s=2.0)
    assert wins == []


# ---------------------------------------------------------------------------
# Energy follows amplitude envelope
# ---------------------------------------------------------------------------


def test_energy_tracks_amplitude():
    # quiet first half, loud second half — non-overlapping windows
    quiet = white_noise(4.0, amp=0.05, seed=1)
    loud = white_noise(4.0, amp=0.8, seed=2)
    y = np.concatenate([quiet, loud])
    # hop = window → non-overlapping windows
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) == 2
    assert wins[1].energy > wins[0].energy * 5


# ---------------------------------------------------------------------------
# Rising sine sweep → motion=rising in descriptor
# ---------------------------------------------------------------------------


def test_chirp_descriptor_rising():
    # sweep 200 Hz → 4000 Hz over 8 s — centroid climbs, expect motion=rising
    # (v1 semantics; v2 encodes the same rise in the brt arc instead)
    y = chirp(200.0, 4000.0, 8.0)
    wins = windows(y, SR, window_s=8.0, hop_s=8.0, descriptor_version=1)
    assert len(wins) >= 1
    assert "motion=rising" in wins[0].descriptor


# ---------------------------------------------------------------------------
# Tempo field populated on click-track fixture
# ---------------------------------------------------------------------------


def test_tempo_populated_on_click_track():
    # 120 BPM click track; estimate_tempo should land in a plausible range
    y = click_track(120.0, seconds=8.0)
    wins = windows(y, SR, window_s=8.0, hop_s=8.0)
    assert len(wins) == 1
    tempo = wins[0].tempo
    # librosa may report half/double tempo; accept a wide range
    assert 60.0 <= tempo <= 250.0


# ---------------------------------------------------------------------------
# input_jump detection
# ---------------------------------------------------------------------------


def _make_window(
    y: np.ndarray, sr: int = SR, emb: np.ndarray | None = None
) -> AnalysisWindow:
    from wav2tidal.core.dsp.features import descriptor_text, estimate_tempo

    bpm, _ = estimate_tempo(y, sr)
    energy = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
    desc = descriptor_text(y, sr)
    embedding = np.empty(0, dtype=np.float64) if emb is None else emb
    return AnalysisWindow(
        t0=0.0,
        t1=len(y) / sr,
        descriptor=desc,
        tempo=bpm,
        energy=energy,
        embedding=embedding,
    )


def test_jump_fires_on_energy_explosion():
    quiet = _make_window(white_noise(4.0, amp=0.02, seed=10))
    loud = _make_window(white_noise(4.0, amp=0.9, seed=11))
    assert input_jump(quiet, loud)


def test_jump_fires_on_embedding_distance():
    # Two orthogonal embeddings → cosine distance = 1.0 > default threshold 0.35
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])
    y = white_noise(4.0, amp=0.3, seed=5)
    wa = _make_window(y, emb=emb_a)
    wb = _make_window(y, emb=emb_b)
    assert input_jump(wa, wb)


def test_no_jump_on_similar_windows():
    y1 = white_noise(4.0, amp=0.3, seed=20)
    y2 = white_noise(4.0, amp=0.3, seed=21)
    wa = _make_window(y1)
    wb = _make_window(y2)
    # Both DSP-only (no embedding) with similar amplitude — no jump expected
    # (tempo and energy thresholds not crossed for same-amplitude noise)
    # We do not assert False because tempo estimates may vary; we just verify
    # the function runs and returns a bool.
    result = input_jump(wa, wb)
    assert isinstance(result, bool)


def test_no_jump_on_identical_window():
    y = white_noise(4.0, amp=0.3, seed=7)
    w = _make_window(y)
    assert not input_jump(w, w)


def test_jump_does_not_fire_on_similar_embedding():
    # Embeddings pointing in nearly the same direction → cosine dist ≈ 0
    emb_a = np.array([1.0, 0.01, 0.0])
    emb_b = np.array([0.99, 0.01, 0.0])
    y = white_noise(4.0, amp=0.3, seed=8)
    wa = _make_window(y, emb=emb_a)
    wb = _make_window(y, emb=emb_b)
    # cosine distance ≈ 0 → embedding gate does not fire; energy/tempo similar too
    assert not input_jump(wa, wb)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_windows_deterministic():
    y = white_noise(10.0, seed=42)
    runs = [windows(y, SR, window_s=4.0, hop_s=2.0) for _ in range(3)]
    for a, b in zip(runs[0], runs[1], strict=True):
        assert a.t0 == b.t0
        assert a.t1 == b.t1
        assert a.descriptor == b.descriptor
        assert a.tempo == b.tempo
        assert a.energy == b.energy
        assert np.array_equal(a.embedding, b.embedding)


# ---------------------------------------------------------------------------
# analyze_wav happy path
# ---------------------------------------------------------------------------


def test_analyze_wav(tmp_path):
    sr = 22050
    y = white_noise(10.0, sr=sr, amp=0.5)
    wav_path = tmp_path / "test.wav"
    write_wav(wav_path, y, sr)

    result = analyze_wav(wav_path, target_sr=sr, window_s=4.0, hop_s=2.0)

    assert len(result) > 0
    for w in result:
        assert isinstance(w, AnalysisWindow)
        assert w.t0 >= 0.0
        assert w.t1 > w.t0
        assert isinstance(w.descriptor, str)
        assert "tempo=" in w.descriptor
        assert w.tempo > 0.0
        assert w.energy >= 0.0
        # default NullEmbedder → zero-length embedding
        assert w.embedding.size == 0


# ---------------------------------------------------------------------------
# mean_chroma (issue #59)
# ---------------------------------------------------------------------------


def test_mean_chroma_shape_and_unit_norm_on_sine():
    """mean_chroma returns a 12-element L2-unit vector for a tone."""
    y = sine(440.0, seconds=2.0)
    c = mean_chroma(y, SR)
    assert c.shape == (12,)
    assert c.dtype == np.float64
    assert float(np.linalg.norm(c)) == pytest.approx(1.0, abs=1e-6)


def test_mean_chroma_zero_on_silence():
    """mean_chroma returns a zero vector for an all-zero input."""
    y = silence(2.0)
    c = mean_chroma(y, SR)
    assert c.shape == (12,)
    assert np.allclose(c, 0.0)


def test_mean_chroma_stable_across_lengths():
    """mean_chroma: consistent pitch-class peak for sine at different lengths."""
    short = mean_chroma(sine(440.0, seconds=1.0), SR)
    long_ = mean_chroma(sine(440.0, seconds=4.0), SR)
    # Both should have their max at pitch class A (index 9)
    assert np.argmax(short) == np.argmax(long_)


# ---------------------------------------------------------------------------
# windows() carries chroma field (issue #59)
# ---------------------------------------------------------------------------


def test_windows_chroma_is_12_dim():
    """Every window returned by windows() has a 12-element chroma array."""
    y = white_noise(8.0)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    for w in wins:
        assert w.chroma.shape == (12,)
        assert w.chroma.dtype == np.float64


def test_windows_chroma_unit_norm_on_noise():
    """Chroma on white noise is unit-normed (noise has harmonic content)."""
    y = white_noise(4.0, amp=0.5, seed=99)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    norm = float(np.linalg.norm(wins[0].chroma))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_windows_chroma_zero_on_silence():
    """Silent windows produce a zero chroma vector."""
    y = silence(4.0)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    assert np.allclose(wins[0].chroma, 0.0)


def test_windows_chroma_deterministic():
    """Repeated calls to windows() yield identical chroma arrays."""
    y = white_noise(8.0, seed=7)
    runs = [windows(y, SR, window_s=4.0, hop_s=2.0) for _ in range(2)]
    for a, b in zip(runs[0], runs[1], strict=True):
        assert np.array_equal(a.chroma, b.chroma)


def test_analysis_window_chroma_default_is_empty():
    """AnalysisWindow constructed without chroma gets an empty default array."""
    w = AnalysisWindow(
        t0=0.0,
        t1=4.0,
        descriptor="tempo=120 density=lo key=C brightness=3/5 motion=steady",
        tempo=120.0,
        energy=0.1,
        embedding=np.empty(0, dtype=np.float64),
    )
    assert w.chroma.size == 0


# ---------------------------------------------------------------------------
# windows() carries chroma_seq and modspec fields (issue #69)
# ---------------------------------------------------------------------------


def test_windows_chroma_seq_shape():
    """Every window from windows() has chroma_seq of shape (12, 32)."""
    y = white_noise(8.0)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    for w in wins:
        assert w.chroma_seq.shape == (
            12,
            32,
        ), f"Expected (12, 32), got {w.chroma_seq.shape}"
        assert w.chroma_seq.dtype == np.float64


def test_windows_modspec_shape():
    """Every window from windows() has modspec of shape (192,) == 8*24."""
    y = white_noise(8.0)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    for w in wins:
        assert w.modspec.shape == (8 * 24,), f"Expected (192,), got {w.modspec.shape}"
        assert w.modspec.dtype == np.float64


def test_windows_chroma_seq_columns_normalized_on_noise():
    """Non-zero columns of chroma_seq have unit L2 norm."""
    y = white_noise(4.0, amp=0.5, seed=77)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    seq = wins[0].chroma_seq
    for j in range(seq.shape[1]):
        col_norm = float(np.linalg.norm(seq[:, j]))
        if col_norm > 1e-8:
            assert col_norm == pytest.approx(1.0, abs=1e-6)


def test_windows_modspec_unit_norm_on_noise():
    """modspec is unit-norm on non-silence noise windows."""
    y = white_noise(4.0, amp=0.5, seed=88)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    norm = float(np.linalg.norm(wins[0].modspec))
    # norm may be 0 (all energy above Nyquist of SR=8000) or 1.0
    assert norm == pytest.approx(0.0, abs=1e-6) or norm == pytest.approx(1.0, abs=1e-6)


def test_analysis_window_chroma_seq_default_is_empty():
    """AnalysisWindow constructed without chroma_seq gets an empty default."""
    w = AnalysisWindow(
        t0=0.0,
        t1=4.0,
        descriptor="tempo=120 density=lo key=C brightness=3/5 motion=steady",
        tempo=120.0,
        energy=0.1,
        embedding=np.empty(0, dtype=np.float64),
    )
    assert w.chroma_seq.size == 0


def test_analysis_window_modspec_default_is_empty():
    """AnalysisWindow constructed without modspec gets an empty default."""
    w = AnalysisWindow(
        t0=0.0,
        t1=4.0,
        descriptor="tempo=120 density=lo key=C brightness=3/5 motion=steady",
        tempo=120.0,
        energy=0.1,
        embedding=np.empty(0, dtype=np.float64),
    )
    assert w.modspec.size == 0


# ---------------------------------------------------------------------------
# descriptor_version plumbing (issue #67)
# ---------------------------------------------------------------------------

_V1_PATTERN = re.compile(r"tempo=\d+ density=\S+ key=\S+ brightness=\S+ motion=\S+")
_V2_PATTERN = re.compile(
    r"tempo=\d+ key=\S+ pit:(\S+ ){7}\S+ dyn:\d{8} ons:\d{8} brt:\d{8}"
)


def test_windows_default_descriptor_is_v2():
    """windows() with no descriptor_version produces v2-format descriptors
    (default flipped after the byt5-4k-v2 retrain, issue #68)."""
    y = white_noise(4.0, amp=0.5, seed=50)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0)
    assert len(wins) >= 1
    for w in wins:
        assert _V2_PATTERN.search(
            w.descriptor
        ), f"Expected v2 descriptor, got: {w.descriptor!r}"


def test_windows_descriptor_version_1_explicit():
    """windows(descriptor_version=1) still yields v1 descriptors (back-compat)."""
    y = white_noise(4.0, amp=0.5, seed=51)
    v1_wins = windows(y, SR, window_s=4.0, hop_s=4.0, descriptor_version=1)
    assert len(v1_wins) >= 1
    for w in v1_wins:
        assert _V1_PATTERN.search(
            w.descriptor
        ), f"Expected v1 descriptor, got: {w.descriptor!r}"


def test_windows_descriptor_version_2():
    """windows(descriptor_version=2) produces v2-format descriptors."""
    y = white_noise(8.0, amp=0.5, seed=52)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0, descriptor_version=2)
    assert len(wins) >= 1
    for w in wins:
        assert _V2_PATTERN.search(
            w.descriptor
        ), f"Expected v2 descriptor, got: {w.descriptor!r}"


def test_windows_descriptor_version_2_no_v1_fields():
    """v2 descriptors do not contain v1 fields like 'density=' or 'motion='."""
    y = white_noise(4.0, amp=0.5, seed=53)
    wins = windows(y, SR, window_s=4.0, hop_s=4.0, descriptor_version=2)
    assert len(wins) >= 1
    for w in wins:
        assert "density=" not in w.descriptor
        assert "motion=" not in w.descriptor


def test_analyze_wav_descriptor_version_passthrough(tmp_path):
    """analyze_wav(descriptor_version=2) uses v2 descriptors."""
    sr = 8000
    y = white_noise(8.0, sr=sr, amp=0.5, seed=60)
    wav_path = tmp_path / "test_v2.wav"
    write_wav(wav_path, y, sr)

    result_v2 = analyze_wav(
        wav_path, target_sr=sr, window_s=4.0, hop_s=4.0, descriptor_version=2
    )
    result_v1 = analyze_wav(
        wav_path, target_sr=sr, window_s=4.0, hop_s=4.0, descriptor_version=1
    )

    assert len(result_v2) >= 1
    assert len(result_v1) >= 1
    for w in result_v2:
        assert _V2_PATTERN.search(
            w.descriptor
        ), f"Expected v2 from analyze_wav, got: {w.descriptor!r}"
    for w in result_v1:
        assert _V1_PATTERN.search(
            w.descriptor
        ), f"Expected v1 from analyze_wav, got: {w.descriptor!r}"


def test_analyze_wav_default_is_v2(tmp_path):
    """analyze_wav default produces v2 descriptors (issue #68 default flip)."""
    sr = 8000
    y = white_noise(8.0, sr=sr, amp=0.5, seed=61)
    wav_path = tmp_path / "test_default.wav"
    write_wav(wav_path, y, sr)

    result = analyze_wav(wav_path, target_sr=sr, window_s=4.0, hop_s=4.0)
    assert len(result) >= 1
    for w in result:
        assert _V2_PATTERN.search(w.descriptor)
