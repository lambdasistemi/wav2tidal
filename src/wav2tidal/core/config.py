"""Typed configuration for the pipeline stages (T010).

Config is loaded from committed YAML (FR-026) and carries the seed and the
shared ``target_sr`` constant that couples ingestion (R4) and embedding
(R1). The dataclasses are pure; only ``load_ingest_config`` touches the
filesystem, at the edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# Sample rate the pipeline normalizes everything to. CLAP expects 48 kHz
# (research R1); keep it here so ingestion and embedding never disagree.
DEFAULT_TARGET_SR = 48000


@dataclass(frozen=True)
class IngestConfig:
    """Knobs for the ingest stage. See configs/ingest.yaml for the shipped defaults."""

    target_sr: int = DEFAULT_TARGET_SR
    hop_length: int = 512
    # slice strategy: "beat" (beat_track), "onset" (onset_detect+backtrack),
    # or "grid" (fixed subdivisions of the estimated tempo). Falls back to
    # energy segmentation when no reliable beat is found.
    slice_strategy: str = "beat"
    beats_per_slice: int = 1
    grid_subdivisions: int = 4
    # thresholds
    silence_top_db: float = 40.0
    min_slice_seconds: float = 0.05
    clip_fraction: float = 0.001  # fraction of |y|>=1-eps samples => clipped
    # embedding: "null" (handcrafted-only, offline/CI-safe) or "clap"
    embedder: str = "null"
    n_mfcc: int = 20
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ingest config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ingest_config(path: str | Path | None) -> IngestConfig:
    """Load an IngestConfig from a YAML file, or defaults when path is None."""
    if path is None:
        return IngestConfig()
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return IngestConfig.from_dict(data)


@dataclass(frozen=True)
class DatasetConfig:
    """Knobs for synthetic (descriptor -> pattern) dataset generation (FR-014)."""

    size: int = 1000
    seed: int = 0
    cps: float = 0.5  # cycles/second (~120 BPM at 4 beats/cycle)
    n_cycles: int = 2
    target_sr: int = DEFAULT_TARGET_SR
    hop_length: int = 512
    max_events_per_cycle: int = 64
    max_nesting_depth: int = 4
    # v2 synth path (design-change-001, issue #21):
    # "slices" = v1 sample patterns, pure numpy render (CI-safe);
    # "synth"  = grammar-v2 configs routed per-config to mix/NRT/RT
    #            (needs SuperCollider + SuperDirt for the NRT/RT part).
    mode: str = "slices"
    tail_seconds: float = 2.0  # render past the last cycle (release/FX tails)
    rt_batch_size: int = 16  # RT jobs per booted SuperDirt (boot ~15 s amortized)
    # Hybrid corpus mix (issue #30): fraction of items that are parameter
    # scenes (grammar v3) vs v2 event lines. 0.0 reproduces the pure
    # event-line dataset.
    scene_ratio: float = 0.7
    automation_tick: float = 0.05  # trajectory tick for scene renders (s)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Knobs for ByT5 fine-tuning (T035, FR-015). See configs/train.yaml."""

    dataset: str = "synth_n400_seed0"  # datasets/<id> under the root
    model_name: str = "google/byt5-small"
    revision: str = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"  # byt5-small pin
    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    max_input_len: int = 96
    max_target_len: int = 768
    val_fraction: float = 0.1
    seed: int = 0
    out_dir: str = "checkpoints/byt5"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_train_config(path: str | Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return TrainConfig.from_dict(data)


def load_dataset_config(path: str | Path | None) -> DatasetConfig:
    if path is None:
        return DatasetConfig()
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return DatasetConfig.from_dict(data)
