"""Synthetic dataset generation (T033, FR-014).

Generates seeded (style-descriptor-text -> pattern-text) pairs: sample a
valid pattern over the ingested banks, render it offline, describe the
render, and pair the description with the pattern. Deterministic from
(config, seed) — regenerating yields byte-identical pairs (SC-008) — and
the config is embedded in the artifact.

The descriptor text is the ByT5 model's input surface (T035); keep it
stable and bucketed so the mapping is learnable.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.config import DatasetConfig
from ..core.dsp.features import estimate_key, estimate_tempo, onset_rate
from ..core.pattern.generate import Diversity, generate_pattern
from ..core.pattern.validate import PatternBounds, validate
from ..core.render.mixdown import Banks, render
from ..core.render.schedule import schedule_events
from ..io.banks import load_banks
from ..io.storage import Workspace

_DENSITY_EDGES = (2.0, 6.0)  # onsets/sec -> lo / mid / hi
_BRIGHT_EDGES = (1500.0, 3000.0, 5000.0, 7000.0)  # spectral centroid Hz -> 1..5


def _bucket(value: float, edges) -> int:
    return sum(1 for e in edges if value >= e)


def descriptor_text(audio: np.ndarray, sr: int, hop_length: int = 512) -> str:
    """Compact, bucketed description of a rendered clip — the model input."""
    import librosa

    bpm, _ = estimate_tempo(audio, sr, hop_length)
    key, _ = estimate_key(audio, sr, hop_length)
    density = onset_rate(audio, sr, hop_length)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    density_label = ("lo", "mid", "hi")[_bucket(density, _DENSITY_EDGES)]
    brightness = _bucket(centroid, _BRIGHT_EDGES) + 1
    return (
        f"tempo={int(round(bpm))} density={density_label} "
        f"key={key} brightness={brightness}/5"
    )


@dataclass
class DatasetResult:
    path: Path
    n_pairs: int


def dataset_id(cfg: DatasetConfig) -> str:
    return f"n{cfg.size}_seed{cfg.seed}"


def synth_dataset(root: Path, cfg: DatasetConfig) -> DatasetResult:
    ws = Workspace(root)
    banks: Banks = load_banks(ws.banks, cfg.target_sr)
    inv = banks.inventory()
    if not inv:
        raise ValueError(f"no banks at {ws.banks} — run `wav2tidal ingest` first")

    rng = random.Random(cfg.seed)
    div = Diversity()
    bounds = PatternBounds(cfg.max_events_per_cycle, cfg.max_nesting_depth)
    total_seconds = cfg.n_cycles / cfg.cps

    out_dir = root / "datasets" / dataset_id(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"

    n = 0
    with open(pairs_path, "w") as fh:
        while n < cfg.size:
            pattern = generate_pattern(rng, inv, div)
            if not validate(pattern, inv, bounds).valid:
                continue  # generator is valid by construction; belt and braces
            events = schedule_events(pattern, cfg.cps, cfg.n_cycles)
            audio = render(events, banks, total_seconds, cfg.target_sr)
            fh.write(
                json.dumps(
                    {
                        "input": descriptor_text(audio, cfg.target_sr, cfg.hop_length),
                        "output": pattern.to_text(),
                    }
                )
                + "\n"
            )
            n += 1

    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
    )
    return DatasetResult(path=out_dir, n_pairs=n)
