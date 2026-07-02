"""Generator + mutation validity and determinism (T026 + grammar v2)."""

from __future__ import annotations

import random

from wav2tidal.core.pattern.generate import (
    generate_config,
    generate_pattern,
    mutate,
    mutate_config,
)
from wav2tidal.core.pattern.validate import Sources, validate

BANKS = {"bd": 4, "sn": 2, "hh": 8}
SOURCES = Sources(banks=BANKS, custom=frozenset({"mydef"}))


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


# -- grammar-v2 config space -------------------------------------------------


def test_generated_configs_always_valid():
    rng = random.Random(0)
    for _ in range(300):
        p = generate_config(rng, SOURCES)
        v = validate(p, SOURCES)
        assert v.valid, (p.to_text(), v.reason)


def test_config_generation_is_deterministic_from_seed():
    a = [generate_config(random.Random(7), SOURCES) for _ in range(10)]
    b = [generate_config(random.Random(7), SOURCES) for _ in range(10)]
    assert [p.to_text() for p in a] == [p.to_text() for p in b]


def test_config_space_covers_all_source_kinds():
    rng = random.Random(1)
    texts = " ".join(generate_config(rng, SOURCES).to_text() for _ in range(200))
    assert "super" in texts  # synth palette
    assert "bd" in texts  # corpus samples through FX
    assert "mydef" in texts  # custom synthdefs


def test_config_mutations_stay_valid():
    rng = random.Random(2)
    p = generate_config(rng, SOURCES)
    for _ in range(300):
        p = mutate_config(rng, p, SOURCES)
        v = validate(p, SOURCES)
        assert v.valid, (p.to_text(), v.reason)
    assert p.source == "mutation"


def test_generate_config_without_sources_raises():
    import pytest

    with pytest.raises(ValueError):
        generate_config(random.Random(0), Sources(banks={}, synths=frozenset()))


# -- grammar-v3 parameter scenes ----------------------------------------------


def test_generated_scenes_always_valid():
    from wav2tidal.core.pattern.generate import generate_scene
    from wav2tidal.core.pattern.model import parse_scene_text
    from wav2tidal.core.pattern.validate import validate_scene

    rng = random.Random(0)
    for _ in range(300):
        s = generate_scene(rng, SOURCES)
        v = validate_scene(s, SOURCES)
        assert v.valid, (s.to_text(), v.reason)
        assert parse_scene_text(s.to_text()).to_text() == s.to_text()


def test_scene_generation_is_deterministic_from_seed():
    from wav2tidal.core.pattern.generate import generate_scene

    a = [generate_scene(random.Random(5), SOURCES) for _ in range(10)]
    b = [generate_scene(random.Random(5), SOURCES) for _ in range(10)]
    assert [s.to_text() for s in a] == [s.to_text() for s in b]


def test_scene_space_covers_shapes_and_layers():
    from wav2tidal.core.pattern.generate import generate_scene

    rng = random.Random(1)
    texts = " ".join(generate_scene(rng, SOURCES).to_text() for _ in range(200))
    for token in ("ramp", "sine", "walk", "steps", "layer", "mod room"):
        assert token in texts, token


def test_scene_mutations_stay_valid():
    from wav2tidal.core.pattern.generate import generate_scene, mutate_scene
    from wav2tidal.core.pattern.validate import validate_scene

    rng = random.Random(2)
    s = generate_scene(rng, SOURCES)
    for _ in range(300):
        s = mutate_scene(rng, s, SOURCES)
        v = validate_scene(s, SOURCES)
        assert v.valid, (s.to_text(), v.reason)
    assert s.source == "mutation"


def test_generate_scene_without_synths_raises():
    import pytest

    from wav2tidal.core.pattern.generate import generate_scene

    with pytest.raises(ValueError):
        generate_scene(random.Random(0), Sources(banks={"bd": 4}, synths=frozenset()))
