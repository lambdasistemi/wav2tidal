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


# -- v2 synth mode (config_dataset, issue #21) --------------------------------

import json  # noqa: E402

from wav2tidal.pipeline.dataset import config_dataset  # noqa: E402


class _FakeRenderers:
    """Injected renderer fakes: write a deterministic tone per job."""

    def __init__(self, sr=8000):
        self.sr = sr
        self.rt_jobs: list = []
        self.nrt_calls: list = []

    def _write(self, path, seconds, hz):
        t = np.arange(int(seconds * self.sr)) / self.sr
        write_wav(path, (0.2 * np.sin(2 * np.pi * hz * t)).astype("float32"), self.sr)

    def rt_batch(self, jobs, banks_dir=None):
        self.rt_jobs.append((len(jobs), banks_dir))
        for out, seconds, events in jobs:
            self._write(out, seconds, 440 + 10 * len(events))
        return [j[0] for j in jobs]

    def nrt_events(self, events, seconds, out):
        self.nrt_calls.append(len(events))
        self._write(out, seconds, 220 + 10 * len(events))
        return out


def _run_synth(root, **kw):
    _make_banks(root)
    fakes = _FakeRenderers()
    cfg = _cfg(mode="synth", size=12, rt_batch_size=4, **kw)
    result = config_dataset(
        root, cfg, rt_batch=fakes.rt_batch, nrt_events=fakes.nrt_events
    )
    return result, fakes


def test_synth_dataset_written_with_renderer_column(tmp_path):
    result, fakes = _run_synth(tmp_path)
    rows = [
        json.loads(line)
        for line in (result.path / "pairs.jsonl").read_text().strip().splitlines()
    ]
    assert len(rows) == 12
    assert {r["renderer"] for r in rows} <= {"mix", "nrt", "rt"}
    assert all(r["output"].startswith('d1 $ s "') for r in rows)
    # every pair's captured audio is kept as provenance
    assert len(list((result.path / "audio").glob("*.wav"))) == 12
    meta = json.loads((result.path / "config.json").read_text())
    assert "reproducibility" in meta and "rt" in meta["reproducibility"]
    assert meta["sources"]["banks"] == {"bd": 2, "hh": 3, "sn": 2}


def test_synth_config_texts_reproducible_from_seed(tmp_path):
    a, _ = _run_synth(tmp_path)
    other = tmp_path / "other"
    b, _ = _run_synth(other)
    outs = lambda r: [  # noqa: E731
        json.loads(x)["output"]
        for x in (r.path / "pairs.jsonl").read_text().strip().splitlines()
    ]
    assert outs(a) == outs(b)


def test_synth_rt_jobs_are_batched(tmp_path):
    _, fakes = _run_synth(tmp_path)
    if fakes.rt_jobs:  # seed-dependent, but batches never exceed the cap
        assert all(n <= 4 for n, _ in fakes.rt_jobs)
        assert all(banks_dir is not None for _, banks_dir in fakes.rt_jobs)


def test_synth_mode_works_without_banks(tmp_path):
    fakes = _FakeRenderers()
    cfg = _cfg(mode="synth", size=6, rt_batch_size=4)
    result = config_dataset(
        tmp_path, cfg, rt_batch=fakes.rt_batch, nrt_events=fakes.nrt_events
    )
    assert result.n_pairs == 6
    rows = [
        json.loads(line)
        for line in (result.path / "pairs.jsonl").read_text().strip().splitlines()
    ]
    assert {r["renderer"] for r in rows} <= {"nrt", "rt"}  # no banks -> no mix
