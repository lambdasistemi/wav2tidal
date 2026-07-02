"""Config round-trip and validation (T013)."""

from __future__ import annotations

import pytest

from wav2tidal.core.config import DEFAULT_TARGET_SR, IngestConfig


def test_defaults():
    cfg = IngestConfig()
    assert cfg.target_sr == DEFAULT_TARGET_SR
    assert cfg.embedder == "null"


def test_roundtrip_identity():
    cfg = IngestConfig(target_sr=24000, seed=7, slice_strategy="onset")
    assert IngestConfig.from_dict(cfg.to_dict()) == cfg


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        IngestConfig.from_dict({"nonsense": 1})
