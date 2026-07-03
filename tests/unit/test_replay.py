"""Tests for pipeline/replay.py and the 'replay' CLI stage (US3-4).

All IO is faked:
  - FakeEmbedder.embed returns a deterministic 4-vector (no torch).
  - _fake_render writes a short sine WAV (no SuperCollider).
  - CLI handler test patches wav2tidal.pipeline.replay.replay.
No SuperCollider or torch is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from wav2tidal.cli import build_parser
from wav2tidal.core.pattern.model import Scene
from wav2tidal.core.pursuit import PursuitConfig
from wav2tidal.io.wav import write_wav
from wav2tidal.pipeline.replay import _ASSEMBLY_SR, replay

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_RENDER_SR = 8000  # low rate keeps writes fast
_RENDER_DUR_S = 0.05


class FakeEmbedder:
    """Deterministic 4-vector embedder; no torch dependency."""

    embedder_id = "fake"

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        v = np.array([float(y.size), float(np.mean(np.abs(y))), 0.0, 1.0])
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-12 else v


def _fake_render(
    scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
) -> Path:
    """Write a short mono sine; frequency varies by scene text."""
    freq = 200 + abs(hash(scene.to_text())) % 300
    n = int(_RENDER_SR * _RENDER_DUR_S)
    t = np.arange(n) / _RENDER_SR
    y = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    write_wav(out_wav, y, _RENDER_SR)
    return out_wav


def _fake_embed(y: np.ndarray, sr: int) -> np.ndarray:
    """Deterministic 3-vector from audio statistics."""
    v = np.array([float(np.mean(np.abs(y))), float(np.std(y)), 0.0])
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else v


def _make_input_wav(tmp_path: Path, duration_s: float = 6.0, sr: int = 44100) -> Path:
    """Write a sine tone WAV at ``sr`` as a stand-in input mix."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "input.wav"
    write_wav(p, y, sr)
    return p


# ---------------------------------------------------------------------------
# Happy-path end-to-end
# ---------------------------------------------------------------------------


def test_replay_returns_out_wav_path(tmp_path: Path) -> None:
    """replay() returns the resolved out_wav Path."""
    input_wav = _make_input_wav(tmp_path)
    out_wav = tmp_path / "out.wav"

    result = replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_fake_render,
        embed=_fake_embed,
    )

    assert result == out_wav


def test_replay_out_wav_is_stereo(tmp_path: Path) -> None:
    """Output WAV is two-channel (stereo)."""
    input_wav = _make_input_wav(tmp_path)
    out_wav = tmp_path / "out.wav"

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_fake_render,
        embed=_fake_embed,
    )

    info = sf.info(str(out_wav))
    assert info.channels == 2


def test_replay_out_wav_duration_matches_last_window_t1(tmp_path: Path) -> None:
    """Output WAV duration equals the last analysis window's t1 (±1 sample)."""
    input_wav = _make_input_wav(tmp_path, duration_s=6.0)
    out_wav = tmp_path / "out.wav"

    from wav2tidal.pipeline.analysis import analyze_wav

    windows = analyze_wav(input_wav, embedder=FakeEmbedder())
    total_s = windows[-1].t1
    expected_frames = round(total_s * _ASSEMBLY_SR)

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_fake_render,
        embed=_fake_embed,
    )

    info = sf.info(str(out_wav))
    assert abs(info.frames - expected_frames) <= 1


def test_replay_session_log_exists_and_is_valid_json(tmp_path: Path) -> None:
    """Session log is written next to out_wav and contains valid JSON."""
    input_wav = _make_input_wav(tmp_path)
    out_wav = tmp_path / "out.wav"

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_fake_render,
        embed=_fake_embed,
    )

    log_path = out_wav.with_suffix(".session.json")
    assert log_path.exists()
    records = json.loads(log_path.read_text())
    assert isinstance(records, list)
    assert len(records) > 0


def test_replay_session_log_one_record_per_window(tmp_path: Path) -> None:
    """Session log has exactly one record per analysis window."""
    input_wav = _make_input_wav(tmp_path, duration_s=6.0)
    out_wav = tmp_path / "out.wav"

    from wav2tidal.pipeline.analysis import analyze_wav

    windows = analyze_wav(input_wav, embedder=FakeEmbedder())

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_fake_render,
        embed=_fake_embed,
    )

    log_path = out_wav.with_suffix(".session.json")
    records = json.loads(log_path.read_text())
    assert len(records) == len(windows)


def test_replay_work_dir_populated(tmp_path: Path) -> None:
    """work_dir is created and contains candidate WAV files."""
    input_wav = _make_input_wav(tmp_path)
    out_wav = tmp_path / "out.wav"
    work_dir = tmp_path / "work"

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_fake_render,
        embed=_fake_embed,
        work_dir=work_dir,
    )

    assert work_dir.exists()
    assert len(list(work_dir.glob("*.wav"))) > 0


# ---------------------------------------------------------------------------
# Too-short / empty input
# ---------------------------------------------------------------------------


def test_replay_too_short_input_raises_value_error(tmp_path: Path) -> None:
    """replay() raises ValueError when input is too short to produce windows.

    At default window_s=4 s and hop_s=2 s, the minimum viable input length
    is half a window ≈ 2 s at the analysis rate.  A 0.5 s input is well
    below that threshold.
    """
    input_wav = _make_input_wav(tmp_path, duration_s=0.5)
    out_wav = tmp_path / "out.wav"

    with pytest.raises(ValueError, match="No analysis windows"):
        replay(
            input_wav,
            out_wav,
            embedder=FakeEmbedder(),
            render=_fake_render,
            embed=_fake_embed,
        )


# ---------------------------------------------------------------------------
# winner_index = -1 handling (failed generation skipped in assembly)
# ---------------------------------------------------------------------------


def test_replay_all_fail_still_writes_output(tmp_path: Path) -> None:
    """When all renders fail (winner_index=-1) output is still written (silent)."""
    input_wav = _make_input_wav(tmp_path)
    out_wav = tmp_path / "out.wav"

    def _always_fail(
        scene: Scene, out_w: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        raise RuntimeError("forced failure")

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_always_fail,
        embed=_fake_embed,
    )

    assert out_wav.exists()
    info = sf.info(str(out_wav))
    assert info.channels == 2


def test_replay_partial_fail_skips_failed_gens(tmp_path: Path) -> None:
    """Records with winner_index=-1 are absent from assembly; others contribute."""
    input_wav = _make_input_wav(tmp_path, duration_s=8.0)
    out_wav = tmp_path / "out.wav"

    def _gen0_fails(
        scene: Scene, out_w: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        if "gen0000" in out_w.name:
            raise RuntimeError("gen0 forced fail")
        return _fake_render(scene, out_w, duration_s, cps, seed)

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_gen0_fails,
        embed=_fake_embed,
    )

    assert out_wav.exists()
    log_path = out_wav.with_suffix(".session.json")
    records = json.loads(log_path.read_text())
    assert records[0]["winner_index"] == -1
    # Output is still stereo
    info = sf.info(str(out_wav))
    assert info.channels == 2


# ---------------------------------------------------------------------------
# Custom log path
# ---------------------------------------------------------------------------


def test_replay_custom_log_path(tmp_path: Path) -> None:
    """log_path parameter overrides the default session log location."""
    input_wav = _make_input_wav(tmp_path)
    out_wav = tmp_path / "out.wav"
    log_path = tmp_path / "custom" / "session.json"

    replay(
        input_wav,
        out_wav,
        embedder=FakeEmbedder(),
        render=_fake_render,
        embed=_fake_embed,
        log_path=log_path,
    )

    assert log_path.exists()
    json.loads(log_path.read_text())  # must be valid JSON


# ---------------------------------------------------------------------------
# CLI tests (follows test_cli_scaffold.py conventions)
# ---------------------------------------------------------------------------


def test_cli_replay_stage_present() -> None:
    """'replay' stage is registered in the parser."""
    parser = build_parser()
    stages = set(parser._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
    assert "replay" in stages


def test_cli_replay_parser_defaults() -> None:
    """Parser sets expected defaults for replay flags."""
    parser = build_parser()
    args = parser.parse_args(["replay", "--input", "x.wav"])
    assert args.input == "x.wav"
    assert args.out == "reinterpretation.wav"
    assert args.embedder == "clap"
    assert args.k == 8
    assert args.seed == 0
    assert args.checkpoint is None


def test_cli_replay_handler_invokes_replay_with_mapped_args(tmp_path: Path) -> None:
    """CLI handler calls pipeline.replay.replay with correctly mapped arguments."""
    parser = build_parser()
    in_path = tmp_path / "in.wav"
    out_path = tmp_path / "out.wav"
    in_path.touch()  # is_file() check in handler requires the file to exist

    args = parser.parse_args(
        [
            "replay",
            "--input",
            str(in_path),
            "--out",
            str(out_path),
            "--embedder",
            "null",
            "--seed",
            "42",
            "--k",
            "4",
        ]
    )

    captured: dict = {}

    def _fake_replay(inp, out, **kw):  # type: ignore[no-untyped-def]
        captured["input"] = Path(inp)
        captured["out"] = Path(out)
        captured.update(kw)
        return Path(out)

    with patch("wav2tidal.pipeline.replay.replay", _fake_replay):
        code = args.handler(args)

    assert code == 0
    assert captured["input"] == in_path
    assert captured["out"] == out_path
    assert captured.get("embedder_kind") == "null"
    assert captured.get("seed") == 42
    assert captured.get("cfg") == PursuitConfig(k_candidates=4)
