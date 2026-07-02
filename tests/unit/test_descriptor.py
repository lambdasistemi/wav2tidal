"""Descriptor + similarity invariants (T013)."""

from __future__ import annotations

import numpy as np
import pytest

from wav2tidal.core.descriptor.types import (
    ProfileIndex,
    assemble_descriptor,
    similarity,
)


def _desc(vec, embedder_id="null", sr=48000):
    return assemble_descriptor({"a": np.asarray(vec, dtype=float)}, embedder_id, sr)


def test_self_similarity_is_one():
    d = _desc([1.0, 2.0, 3.0])
    assert similarity(d, d) == pytest.approx(1.0)


def test_similarity_is_symmetric():
    a, b = _desc([1.0, 0.0, 1.0]), _desc([0.0, 1.0, 1.0])
    assert similarity(a, b) == pytest.approx(similarity(b, a))


def test_cross_embedder_comparison_rejected():
    a, b = _desc([1.0, 2.0], embedder_id="null"), _desc([1.0, 2.0], embedder_id="clap")
    with pytest.raises(ValueError):
        similarity(a, b)


def test_cross_samplerate_comparison_rejected():
    a, b = _desc([1.0, 2.0], sr=48000), _desc([1.0, 2.0], sr=24000)
    with pytest.raises(ValueError):
        similarity(a, b)


def test_nearest_ranks_similar_above_dissimilar():
    q = _desc([1.0, 0.0, 0.0])
    near = _desc([0.9, 0.1, 0.0])
    far = _desc([0.0, 0.0, 1.0])
    index = ProfileIndex.build([("near", near), ("far", far)])
    ranked = index.nearest(q, k=2)
    assert ranked[0][0] == "near"
    assert ranked[0][1] > ranked[1][1]


def test_assemble_with_embedding_concatenates():
    emb = np.array([1.0, 0.0])
    d = assemble_descriptor({"a": np.array([0.0, 3.0])}, "clap", 48000, embedding=emb)
    assert d.vector.size == 4
    assert d.embedder_id == "clap"
