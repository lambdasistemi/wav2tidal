"""Param table integrity + grammar/table vocabulary sync (grammar v2, R7)."""

from __future__ import annotations

import random
import re

from wav2tidal.core.pattern import params
from wav2tidal.core.pattern.grammar import grammar_path


def _grammar_param_names() -> set[str]:
    text = grammar_path().read_text()
    m = re.search(r"PARAM:(.*?)\n\nVOWEL", text, re.S)
    assert m, "PARAM terminal not found in grammar"
    return set(re.findall(r'"([a-z0-9]+)"', m.group(1)))


def test_grammar_and_table_vocabularies_agree():
    # the .lark file is the syntactic source of truth, params.py the
    # semantic one — they must never drift apart
    assert _grammar_param_names() | {"vowel"} == set(params.PARAM_ORDER)


def test_every_synth_param_has_a_spec():
    for synth, overrides in params.SYNTHS.items():
        for name in overrides:
            assert name in params.PARAMS, f"{synth}.{name} missing from PARAMS"


def test_override_ranges_are_ordered():
    for synth, overrides in params.SYNTHS.items():
        for name, rng in overrides.items():
            if rng is not None:
                assert rng[0] < rng[1], f"{synth}.{name} has empty range {rng}"


def test_samples_pass_their_own_check():
    rng = random.Random(0)
    for name, s in params.PARAMS.items():
        for _ in range(20):
            assert s.in_range(s.sample(rng)), name
    for synth in params.SYNTHS:
        for name in params.SYNTHS[synth]:
            lo, hi = params.synth_range(synth, name)
            for _ in range(20):
                v = params.spec(name).sample(rng, lo, hi)
                assert params.check_value(name, v, {synth}), (synth, name, v)


def test_effective_range_intersects_listing_synths():
    # supernoise widens resonance to 0..1; with supersaw in the same config
    # the intersection falls back to the stricter default
    assert params.effective_range("resonance", {"supernoise"}) == (0.0, 1.0)
    lo, hi = params.effective_range("resonance", {"supernoise", "supersaw"})
    assert (lo, hi) == (0.0, 0.8)
    assert params.check_value("resonance", 0.95, {"supernoise"})
    assert not params.check_value("resonance", 0.95, {"supernoise", "supersaw"})


def test_applicability_rules():
    assert params.applicable("cutoff", {"bd"})  # event FX apply to samples
    assert params.applicable("room", set())  # global sends apply anywhere
    assert params.applicable("lfo", {"supersaw"})
    assert not params.applicable("lfo", {"bd"})  # synth param needs a synth
    assert not params.applicable("lfo", {"supersaw", "superkick"})  # all must list
    assert not params.applicable("nosuch", {"supersaw"})


def test_vowel_is_choice():
    s = params.spec("vowel")
    assert s.in_range("a") and not s.in_range("x")


def test_midicps_reference_points():
    assert abs(params.midicps(69) - 440.0) < 1e-9
    assert abs(params.midicps(60) - 261.6255653) < 1e-6
