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
