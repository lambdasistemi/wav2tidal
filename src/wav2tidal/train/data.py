"""Training-pair loading and splitting (T035 support, issue #22).

Pure helpers over the dataset artifact (``pairs.jsonl`` + ``config.json``
from ``pipeline.dataset``): load rows, reconstruct the source inventory
the corpus was generated against (the validator needs it for eval), and
make a deterministic train/val split. No torch here — CI-testable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..core.pattern.validate import Sources


def load_pairs(dataset_dir: str | Path) -> list[dict]:
    """Rows of ``pairs.jsonl``: {input, output, renderer, kind}."""
    path = Path(dataset_dir) / "pairs.jsonl"
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def load_sources(dataset_dir: str | Path) -> Sources:
    """The inventory embedded in the artifact's config.json."""
    meta = json.loads((Path(dataset_dir) / "config.json").read_text())
    src = meta.get("sources", {})
    return Sources(
        banks=dict(src.get("banks", {})),
        synths=frozenset(src.get("synths", ())),
        custom=frozenset(src.get("custom", ())),
    )


def split_pairs(
    pairs: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffled split; validation gets at least one row."""
    if not pairs:
        raise ValueError("no pairs to split")
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(len(pairs) * val_fraction))
    val_idx = set(order[:n_val])
    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]
    return train, val
