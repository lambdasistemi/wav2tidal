"""End-to-end ingest: banks load, idempotency and incrementality (T016, FR-006)."""

from __future__ import annotations

import numpy as np

from wav2tidal.core.config import IngestConfig
from wav2tidal.io.wav import write_wav
from wav2tidal.pipeline.ingest import ingest


def _tone(freq, sr=22050, seconds=3.0):
    t = np.arange(int(sr * seconds)) / sr
    # tone + a click grid so beat tracking finds slices
    click = np.zeros_like(t)
    click[:: int(sr * 0.5)] = 1.0
    return (0.4 * np.sin(2 * np.pi * freq * t) + 0.6 * click).astype(np.float32)


def _corpus(tmp_path, sr=22050):
    d = tmp_path / "corpus"
    d.mkdir()
    write_wav(d / "a.wav", _tone(220), sr)
    write_wav(d / "b.wav", _tone(440), sr)
    return d


def _cfg():
    return IngestConfig(target_sr=22050, embedder="null", min_slice_seconds=0.05)


def test_ingest_produces_banks(tmp_path):
    corpus = _corpus(tmp_path)
    root = tmp_path / "ws"
    report = ingest(corpus, root, _cfg())
    assert len(report.processed) == 2
    assert report.n_slices > 0
    wavs = list((root / "banks").rglob("*.wav"))
    assert len(wavs) == report.n_slices
    # SuperDirt layout: folder = bank name, 0-indexed files
    for bank in report.banks:
        files = sorted((root / "banks" / bank).glob("*.wav"))
        assert files[0].name.startswith("0000_")


def test_reingest_unchanged_is_noop(tmp_path):
    corpus = _corpus(tmp_path)
    root = tmp_path / "ws"
    ingest(corpus, root, _cfg())
    again = ingest(corpus, root, _cfg())
    assert again.processed == []
    assert all(reason.startswith("unchanged") for _, reason in again.skipped)


def test_adding_a_file_processes_only_it(tmp_path):
    corpus = _corpus(tmp_path)
    root = tmp_path / "ws"
    ingest(corpus, root, _cfg())
    write_wav(corpus / "c.wav", _tone(330), 22050)
    report = ingest(corpus, root, _cfg())
    assert report.processed == [str(corpus / "c.wav")]


def test_corrupt_file_skipped_not_fatal(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "bad.wav").write_bytes(b"not a wav file")
    root = tmp_path / "ws"
    report = ingest(corpus, root, _cfg())
    assert any("bad.wav" in p for p, _ in report.skipped)
    assert len(report.processed) == 2
