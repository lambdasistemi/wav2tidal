"""Generator + mutation validity and determinism (T026)."""

from __future__ import annotations

import random

from wav2tidal.core.pattern.generate import generate_pattern, mutate
from wav2tidal.core.pattern.validate import validate

BANKS = {"bd": 4, "sn": 2, "hh": 8}


def test_generated_patterns_always_valid():
    rng = random.Random(0)
    for _ in range(200):
        p = generate_pattern(rng, BANKS)
        assert validate(p, BANKS).valid, p.mini


def test_generation_is_deterministic_from_seed():
    a = [generate_pattern(random.Random(0), BANKS) for _ in range(5)]
    b = [generate_pattern(random.Random(0), BANKS) for _ in range(5)]
    assert [p.to_text() for p in a] == [p.to_text() for p in b]


def test_mutations_stay_valid():
    rng = random.Random(1)
    p = generate_pattern(rng, BANKS)
    for _ in range(100):
        p = mutate(rng, p, BANKS)
        assert validate(p, BANKS).valid, p.mini


def test_mutation_marks_source():
    rng = random.Random(2)
    p = mutate(rng, generate_pattern(rng, BANKS), BANKS)
    assert p.source == "mutation"


def test_generate_without_banks_raises():
    import pytest

    with pytest.raises(ValueError):
        generate_pattern(random.Random(0), {})
