"""Style descriptor and similarity (T011).

A ``StyleDescriptor`` is the uniform representation of "how audio sounds"
(FR-004/FR-005), used identically for corpus audio, rendered clips, and
captured live audio. It assembles an optional neural embedding (CLAP, R1)
with hand-crafted DSP blocks (R4) into a single comparable vector.

Descriptors are only comparable within the same ``embedder_id`` and
``sr_used`` — mixing rates or models silently corrupts similarity, so we
reject it loudly instead. Pure numpy; no IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-8


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v


@dataclass(frozen=True)
class StyleDescriptor:
    """A fixed-length, comparable representation of an audio fragment."""

    vector: np.ndarray  # assembled + L2-normalized; the comparison surface
    embedder_id: str  # e.g. "null" or "laion/larger_clap_music@<sha>"
    sr_used: int
    blocks: dict[str, np.ndarray] = field(default_factory=dict)  # raw, for inspection

    def compatible_with(self, other: StyleDescriptor) -> bool:
        return self.embedder_id == other.embedder_id and self.sr_used == other.sr_used


def assemble_descriptor(
    handcrafted: dict[str, np.ndarray],
    embedder_id: str,
    sr_used: int,
    embedding: np.ndarray | None = None,
    block_weights: dict[str, float] | None = None,
) -> StyleDescriptor:
    """Build a descriptor from hand-crafted blocks and an optional embedding.

    Each block is L2-normalized independently (so no single block dominates
    by scale), optionally weighted, then concatenated with the (already
    normalized) embedding. The full vector is L2-normalized so cosine
    similarity reduces to a dot product.
    """
    weights = block_weights or {}
    parts: list[np.ndarray] = []
    if embedding is not None:
        parts.append(_l2_normalize(np.asarray(embedding, dtype=np.float64)))
    for name in sorted(handcrafted):
        block = _l2_normalize(np.asarray(handcrafted[name], dtype=np.float64))
        parts.append(block * float(weights.get(name, 1.0)))
    vector = _l2_normalize(np.concatenate(parts)) if parts else np.zeros(0)
    return StyleDescriptor(
        vector=vector,
        embedder_id=embedder_id,
        sr_used=sr_used,
        blocks=dict(handcrafted),
    )


def similarity(a: StyleDescriptor, b: StyleDescriptor) -> float:
    """Cosine similarity of two descriptors. Raises on incompatible descriptors."""
    if not a.compatible_with(b):
        raise ValueError(
            f"incompatible descriptors: {a.embedder_id}@{a.sr_used} "
            f"vs {b.embedder_id}@{b.sr_used}"
        )
    if a.vector.shape != b.vector.shape or a.vector.size == 0:
        raise ValueError("descriptor vectors have mismatched or empty shape")
    return float(np.dot(a.vector, b.vector))


@dataclass
class ProfileIndex:
    """A nearest-neighbour index over descriptors sharing one embedder/sr."""

    ids: list[str]
    matrix: np.ndarray  # (n, d), rows already L2-normalized
    embedder_id: str
    sr_used: int

    @classmethod
    def build(cls, items: list[tuple[str, StyleDescriptor]]) -> ProfileIndex:
        if not items:
            raise ValueError("cannot build an index from zero descriptors")
        first = items[0][1]
        for _id, d in items:
            if not d.compatible_with(first):
                raise ValueError("all descriptors in an index must be compatible")
        ids = [i for i, _ in items]
        matrix = np.stack([d.vector for _, d in items])
        return cls(ids, matrix, first.embedder_id, first.sr_used)

    def nearest(self, query: StyleDescriptor, k: int = 5) -> list[tuple[str, float]]:
        if query.embedder_id != self.embedder_id or query.sr_used != self.sr_used:
            raise ValueError("query descriptor is incompatible with the index")
        scores = self.matrix @ query.vector
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order]
