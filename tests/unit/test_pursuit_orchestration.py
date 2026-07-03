"""Tests for pipeline/pursuit.py — orchestration layer (US3-3).

All IO is faked:
  - ``fake_render`` writes a short deterministic WAV (sine at hash-derived freq).
  - ``fake_embed`` returns a normalised 3-vector derived from the audio.
  - ``fake_propose`` returns a fixed valid scene text.
No SuperCollider or torch is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wav2tidal.core.dsp.features import chroma_sequence, mean_chroma
from wav2tidal.core.dsp.stream import AnalysisWindow
from wav2tidal.core.pattern.model import Scene
from wav2tidal.core.pattern.validate import Sources
from wav2tidal.core.pursuit import PursuitConfig
from wav2tidal.io.wav import write_wav
from wav2tidal.pipeline.pursuit import (
    GenerationRecord,
    run_pursuit,
    write_session_log,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

SOURCES = Sources()  # full Super* palette, no sample banks

_SR = 8000  # low rate keeps tests fast
_DUR_S = 0.05  # tiny write keeps I/O fast


def fake_render(
    scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
) -> Path:
    """Write a short mono sine whose frequency depends on the scene text."""
    freq = 200 + abs(hash(scene.to_text())) % 500
    n = int(_SR * _DUR_S)
    t = np.arange(n) / _SR
    y = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    write_wav(out_wav, y, _SR)
    return out_wav


def fake_embed(y: np.ndarray, sr: int) -> np.ndarray:
    """Deterministic 3-vector from audio: [mean_abs, std, zcr_proxy]."""
    mean_a = float(np.mean(np.abs(y))) if y.size else 0.0
    std_v = float(np.std(y)) if y.size else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(y))))) if y.size > 1 else 0.0
    v = np.array([mean_a, std_v, zcr], dtype=np.float64)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def fake_propose(_descriptor: str) -> str:
    """Return a fixed valid scene text (two supersaw voices)."""
    return "scene voice supersaw # note -12 voice supersaw # note 0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window(
    t0: float = 0.0,
    t1: float = 4.0,
    tempo: float = 120.0,
    energy: float = 0.1,
    emb: np.ndarray | None = None,
) -> AnalysisWindow:
    return AnalysisWindow(
        t0=t0,
        t1=t1,
        descriptor="tempo=120 density=medium motion=steady",
        tempo=tempo,
        energy=energy,
        embedding=(
            np.empty(0, dtype=np.float64)
            if emb is None
            else np.asarray(emb, dtype=np.float64)
        ),
    )


def _windows(n: int) -> list[AnalysisWindow]:
    return [_window(t0=float(i * 2), t1=float(i * 2 + 4)) for i in range(n)]


# ---------------------------------------------------------------------------
# run_pursuit — basic behaviour
# ---------------------------------------------------------------------------


def test_run_pursuit_record_count(tmp_path):
    """One record per analysis window."""
    ws = _windows(3)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=PursuitConfig(k_candidates=3),
    )
    assert len(records) == 3


def test_run_pursuit_winner_wav_exists(tmp_path):
    """Winner WAV file is written to disk for every successful generation."""
    ws = _windows(4)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=PursuitConfig(k_candidates=3),
    )
    for r in records:
        if r.winner_index >= 0:
            assert r.wav_path is not None
            assert Path(r.wav_path).exists()


def test_run_pursuit_first_generation_is_propose(tmp_path):
    """First generation must use propose mode (no scene yet)."""
    ws = _windows(1)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
    )
    assert records[0].mode == "propose"


def test_run_pursuit_cps_correct(tmp_path):
    """Records carry the correct cps value derived from window tempo."""
    ws = [_window(tempo=120.0)]
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=PursuitConfig(beats_per_cycle=4.0),
    )
    # 120 BPM / 60 / 4 = 0.5 cps
    assert records[0].cps == pytest.approx(0.5)


def test_run_pursuit_candidate_texts_present(tmp_path):
    """candidate_texts tuple is non-empty and matches pool size."""
    cfg = PursuitConfig(k_candidates=4)
    ws = _windows(1)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=cfg,
    )
    assert len(records[0].candidate_texts) > 0
    assert len(records[0].scores) == len(records[0].candidate_texts)


# ---------------------------------------------------------------------------
# run_pursuit — session log
# ---------------------------------------------------------------------------


def test_run_pursuit_session_log_json_round_trips(tmp_path):
    """write_session_log produces valid JSON that round-trips."""
    ws = _windows(3)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=PursuitConfig(k_candidates=3),
    )
    log_path = tmp_path / "session.json"
    write_session_log(records, log_path)

    loaded = json.loads(log_path.read_text())
    assert len(loaded) == 3
    for i, d in enumerate(loaded):
        assert d["t0"] == pytest.approx(records[i].t0)
        assert d["winner_index"] == records[i].winner_index
        assert d["mode"] in ("propose", "mutate")
        assert isinstance(d["candidate_texts"], list)
        assert isinstance(d["scores"], list)


# ---------------------------------------------------------------------------
# run_pursuit — failing render (partial failure)
# ---------------------------------------------------------------------------


def test_run_pursuit_one_failing_render_degrades_gracefully(tmp_path):
    """A single candidate failing still leaves a winner from the rest."""
    call_count: list[int] = [0]

    def first_fails(
        scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            raise RuntimeError("fake render error")
        return fake_render(scene, out_wav, duration_s, cps, seed)

    ws = _windows(1)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=first_fails,
        embed=fake_embed,
        cfg=PursuitConfig(k_candidates=4),
    )
    assert len(records) == 1
    # At least one candidate succeeded → winner_index >= 0
    assert records[0].winner_index >= 0


# ---------------------------------------------------------------------------
# run_pursuit — all-fail generation
# ---------------------------------------------------------------------------


def test_run_pursuit_all_fail_winner_index_minus_one(tmp_path):
    """When every render fails, winner_index=-1 and wav_path=None."""

    def always_fail(
        scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        raise RuntimeError("always fail")

    ws = _windows(1)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=always_fail,
        embed=fake_embed,
        cfg=PursuitConfig(k_candidates=2),
    )
    assert len(records) == 1
    assert records[0].winner_index == -1
    assert records[0].wav_path is None


def test_run_pursuit_all_fail_then_recover(tmp_path):
    """After an all-fail gen, the loop continues in propose mode (scene=None)."""

    def conditional_render(
        scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        if "gen0000" in out_wav.name:
            raise RuntimeError("first gen all fail")
        return fake_render(scene, out_wav, duration_s, cps, seed)

    ws = _windows(2)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=conditional_render,
        embed=fake_embed,
        cfg=PursuitConfig(k_candidates=2),
    )
    assert len(records) == 2
    assert records[0].winner_index == -1
    # Second window: state.scene is still None → must be in propose mode
    assert records[1].mode == "propose"


# ---------------------------------------------------------------------------
# run_pursuit — proposer integration
# ---------------------------------------------------------------------------


def test_run_pursuit_with_proposer(tmp_path):
    """Propose callback is exercised on the first generation."""
    proposal_calls: list[str] = []

    def tracking_propose(descriptor: str) -> str:
        proposal_calls.append(descriptor)
        return fake_propose(descriptor)

    ws = _windows(2)
    run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        propose=tracking_propose,
        cfg=PursuitConfig(k_candidates=3, patience=10),
    )
    # Proposer was called at least once (first window always proposes)
    assert len(proposal_calls) >= 1


# ---------------------------------------------------------------------------
# run_pursuit — stagnation triggers re-propose
# ---------------------------------------------------------------------------


def test_run_pursuit_stagnation_triggers_propose(tmp_path):
    """After cfg.patience stagnant generations, mode flips back to propose."""
    # Embed always returns the same vector → cosine similarity is 1.0 every time
    # → score never improves beyond the first win, but stagnation accumulates.
    fixed_emb = np.array([1.0, 0.0, 0.0])

    def fixed_embed(y: np.ndarray, sr: int) -> np.ndarray:
        return fixed_emb

    ws = [_window(emb=fixed_emb) for _ in range(5)]
    cfg = PursuitConfig(k_candidates=3, patience=2)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fixed_embed,
        cfg=cfg,
    )
    modes = [r.mode for r in records]
    # Generation 0 → propose (cold start)
    assert modes[0] == "propose"
    # After patience=2 stagnant gens, a propose must appear again
    assert "propose" in modes[2:]


# ---------------------------------------------------------------------------
# run_pursuit — input jump triggers re-propose
# ---------------------------------------------------------------------------


def test_run_pursuit_input_jump_triggers_propose(tmp_path):
    """Orthogonal embeddings between consecutive windows trigger propose."""
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])
    # Window 1 and 2 have orthogonal embeddings → jump
    ws = [
        _window(t0=0.0, t1=4.0, emb=emb_a),
        _window(t0=2.0, t1=6.0, emb=emb_b),
    ]
    cfg = PursuitConfig(k_candidates=2, patience=10, jump_threshold=0.35)
    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=cfg,
    )
    assert len(records) == 2
    assert records[0].mode == "propose"  # cold start
    assert records[1].mode == "propose"  # jump detected


# ---------------------------------------------------------------------------
# write_session_log
# ---------------------------------------------------------------------------


def test_write_session_log_creates_parent(tmp_path):
    """write_session_log creates parent directories if absent."""
    records = [
        GenerationRecord(
            t0=0.0,
            t1=4.0,
            descriptor="",
            tempo=120.0,
            energy=0.1,
            mode="propose",
            candidate_texts=("scene voice supersaw",),
            scores=(0.5,),
            winner_index=0,
            cps=0.5,
            wav_path=None,
        )
    ]
    nested = tmp_path / "a" / "b" / "c" / "log.json"
    write_session_log(records, nested)
    assert nested.exists()
    loaded = json.loads(nested.read_text())
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Harmonic A/B: in-key candidate wins via chroma (issue #59)
# ---------------------------------------------------------------------------

_SR_H = 8000  # low rate — fast test


def _sine_wav(freq: float, dur_s: float = 0.25, sr: int = _SR_H) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_harmonic_ab_in_key_candidate_wins(tmp_path):
    """In-key candidate beats out-of-key when CLAP embeddings are identical.

    Candidate 0 → 440 Hz sine (A4, in-key); candidate 1 → 370 Hz sine
    (F#4, out-of-key relative to A).  Target chroma is built from a 440 Hz
    reference.  With equal timbre embeddings, the chroma term lifts cand0.
    (issue #59)
    """
    # Target chroma — computed from an A4 reference sine
    ref_y = _sine_wav(440.0)
    target_chroma = mean_chroma(ref_y, _SR_H)
    assert float(np.linalg.norm(target_chroma)) > 0.5, "reference chroma is zero"

    # Both candidates get an identical timbre embedding (CLAP deaf to harmony)
    fixed_emb = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def fixed_embed(y: np.ndarray, sr: int) -> np.ndarray:
        return fixed_emb

    def harmonic_render(
        scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        # cand0 → in-key (440 Hz); cand1 → out-of-key (370 Hz)
        freq = 440.0 if "cand0" in out_wav.name else 370.0
        write_wav(out_wav, _sine_wav(freq), _SR_H)
        return out_wav

    win = AnalysisWindow(
        t0=0.0,
        t1=4.0,
        descriptor="tempo=120 density=lo key=A brightness=3/5 motion=steady",
        tempo=120.0,
        energy=0.1,
        embedding=fixed_emb,
        chroma=target_chroma,
    )

    records = run_pursuit(
        [win],
        SOURCES,
        tmp_path / "out",
        render=harmonic_render,
        embed=fixed_embed,
        cfg=PursuitConfig(k_candidates=2, w_timbre=0.0, w_harmony=1.0),
    )

    assert len(records) == 1
    assert (
        records[0].winner_index == 0
    ), f"In-key candidate (0) should win; scores={records[0].scores}"


def test_harmonic_scoring_window_without_chroma_degrades_to_timbre(tmp_path):
    """Window without chroma (empty default) falls back to timbre-only; no crash.

    Uses the existing _window() helper which does not set chroma, so the field
    defaults to an empty array.  Scoring must not raise and must still pick a
    winner based on the timbre embedding alone.  (issue #59)
    """
    ws = _windows(1)  # _window() produces AnalysisWindow with empty chroma
    # Sanity: the default chroma must be empty
    assert ws[0].chroma.size == 0

    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=PursuitConfig(k_candidates=3, w_timbre=0.5, w_harmony=0.5),
    )
    assert len(records) == 1
    # Should not crash and should produce a winner
    assert records[0].winner_index >= 0


# ---------------------------------------------------------------------------
# chroma_seq movement A→E: correct direction beats reversed (issue #69)
# ---------------------------------------------------------------------------

_SR_SEQ = 22050  # sufficient for CQT at both 440 Hz and 330 Hz


def _two_tone_wav(
    freq1: float, freq2: float, half_dur_s: float = 1.0, sr: int = _SR_SEQ
) -> np.ndarray:
    """Concatenate two equal-length sines: freq1 then freq2."""
    n = int(sr * half_dur_s)
    t = np.arange(n) / sr
    y1 = (0.3 * np.sin(2 * np.pi * freq1 * t)).astype(np.float32)
    y2 = (0.3 * np.sin(2 * np.pi * freq2 * t)).astype(np.float32)
    return np.concatenate([y1, y2])


def test_chroma_seq_movement_correct_direction_wins(tmp_path):
    """A candidate whose harmony moves A→E beats one moving E→A.

    The target window has chroma_seq built from an A-then-E signal.  Candidate
    0 renders A→E (matching direction); candidate 1 renders E→A (reversed).
    With w_harmony_seq=1.0 and all other weights=0, candidate 0 must win.
    (issue #69)
    """
    # A4 = 440 Hz (pitch class 9), E4 = 329.63 Hz (pitch class 4).
    target_audio = _two_tone_wav(440.0, 329.63)
    target_seq = chroma_sequence(target_audio, _SR_SEQ)

    # Equal (dummy) embeddings so timbre/harmony/modspec components don't decide.
    fixed_emb = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def fixed_embed(y: np.ndarray, sr: int) -> np.ndarray:
        return fixed_emb

    def seq_render(
        scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        # cand0 → A→E (matching target); cand1 → E→A (reversed)
        if "cand0" in out_wav.name:
            y = _two_tone_wav(440.0, 329.63)
        else:
            y = _two_tone_wav(329.63, 440.0)
        write_wav(out_wav, y, _SR_SEQ)
        return out_wav

    win = AnalysisWindow(
        t0=0.0,
        t1=4.0,
        descriptor="tempo=120 density=lo key=A brightness=3/5 motion=steady",
        tempo=120.0,
        energy=0.1,
        embedding=fixed_emb,
        chroma_seq=target_seq,
    )

    records = run_pursuit(
        [win],
        SOURCES,
        tmp_path / "out",
        render=seq_render,
        embed=fixed_embed,
        cfg=PursuitConfig(
            k_candidates=2,
            w_timbre=0.0,
            w_harmony=0.0,
            w_harmony_seq=1.0,
            w_modspec=0.0,
        ),
    )

    assert len(records) == 1
    assert (
        records[0].winner_index == 0
    ), f"A→E candidate (0) should win; scores={records[0].scores}"


def test_chroma_seq_missing_fields_degrade_gracefully(tmp_path):
    """Windows without chroma_seq/modspec fields (empty defaults) do not crash.

    Scoring falls back to the available components (timbre + harmony or none).
    The loop must still complete and pick a winner.  (issue #69)
    """
    # _window() produces AnalysisWindow with default-empty chroma_seq and modspec.
    ws = _windows(1)
    assert ws[0].chroma_seq.size == 0
    assert ws[0].modspec.size == 0

    records = run_pursuit(
        ws,
        SOURCES,
        tmp_path / "out",
        render=fake_render,
        embed=fake_embed,
        cfg=PursuitConfig(
            k_candidates=3,
            w_timbre=0.3,
            w_harmony=0.2,
            w_harmony_seq=0.25,
            w_modspec=0.25,
        ),
    )
    assert len(records) == 1
    assert records[0].winner_index >= 0
