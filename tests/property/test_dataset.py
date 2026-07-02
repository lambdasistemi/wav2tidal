"""Synthetic dataset reproducibility (T028, SC-008)."""

from __future__ import annotations

import numpy as np

from wav2tidal.core.config import DatasetConfig
from wav2tidal.io.wav import write_wav
from wav2tidal.pipeline.dataset import synth_dataset


def _make_banks(root, sr=8000):
    banks = root / "banks"
    for name, n in (("bd", 2), ("sn", 2), ("hh", 3)):
        for i in range(n):
            sig = (
                np.random.default_rng(hash((name, i)) % 2**32).standard_normal(sr) * 0.3
            ).astype("float32")
            write_wav(banks / name / f"{i:04d}_{name}.wav", sig, sr)


def _cfg(**kw):
    base = dict(size=6, seed=0, cps=1.0, n_cycles=1, target_sr=8000, hop_length=256)
    base.update(kw)
    return DatasetConfig(**base)


def test_dataset_written(tmp_path):
    _make_banks(tmp_path)
    result = synth_dataset(tmp_path, _cfg())
    assert result.n_pairs == 6
    lines = (result.path / "pairs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 6
    assert (result.path / "config.json").exists()


def test_same_seed_reproduces_identical_dataset(tmp_path):
    _make_banks(tmp_path)
    a = synth_dataset(tmp_path, _cfg(seed=0)).path / "pairs.jsonl"
    text_a = a.read_text()
    # regenerate into a fresh root with the same banks + seed
    other = tmp_path / "other"
    _make_banks(other)
    b = synth_dataset(other, _cfg(seed=0)).path / "pairs.jsonl"
    assert b.read_text() == text_a


def test_different_seed_differs(tmp_path):
    _make_banks(tmp_path)
    a = (synth_dataset(tmp_path, _cfg(seed=0)).path / "pairs.jsonl").read_text()
    b = (synth_dataset(tmp_path, _cfg(seed=1)).path / "pairs.jsonl").read_text()
    assert a != b


def test_no_banks_raises(tmp_path):
    (tmp_path / "banks").mkdir()
    import pytest

    with pytest.raises(ValueError):
        synth_dataset(tmp_path, _cfg())
