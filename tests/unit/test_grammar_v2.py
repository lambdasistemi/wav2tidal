"""Grammar v2 full-line membership + config text round-trip (FR-008)."""

from __future__ import annotations

import pytest
from lark.exceptions import LarkError

from wav2tidal.core.pattern.grammar import line_controls, parse_line
from wav2tidal.core.pattern.model import Pattern, parse_pattern_text
from wav2tidal.core.pattern.validate import PatternBounds, Sources, validate

SOURCES = Sources(banks={"bd": 4}, custom=frozenset({"mydef"}))


@pytest.mark.parametrize(
    "text",
    [
        'd1 $ s "supersaw"',
        'd1 $ s "supersaw supersaw:7 ~" # note 7 # cutoff 1200 # resonance 0.3',
        'd1 $ s "superkick(3,8) superhat*2" # shape 0.5 # room 0.4 # size 0.7',
        'd1 $ s "bd:3 ~ [bd bd]" # vowel a # crush 4.5 # delaytime 0.25',
        'd1 $ s "mydef" # note -12 # hcutoff 2000 # delayfeedback 0.5',
        'd1 $ s "bd sn" # gain 1 # speed 1 # pan 0.5',  # v1 lines still parse
    ],
)
def test_line_membership(text):
    parse_line(text)


@pytest.mark.parametrize(
    "text",
    [
        'd1 $ s "supersaw" # nosuch 1',  # unknown control name
        'd1 $ s "supersaw" # vowel x',  # not a vowel
        'd1 $ s "supersaw" # cutoff',  # missing value
        'd2 $ s "supersaw"',  # only the d1 channel
        'd1 $ n "supersaw"',  # only s patterns
        'd1 $ s "supersaw" cutoff 300',  # controls need '#'
    ],
)
def test_line_rejection(text):
    with pytest.raises(LarkError):
        parse_line(text)


def test_line_controls_extraction():
    t = parse_line('d1 $ s "supersaw" # note 7 # vowel e # room 0.4')
    assert line_controls(t) == {"note": 7.0, "room": 0.4, "vowel": "e"}


def test_v2_text_roundtrip():
    p = Pattern(
        mini="supersaw supersaw:7 ~",
        controls={"note": 7.0, "cutoff": 1200.0, "vowel": "a", "room": 0.4},
    )
    q = parse_pattern_text(p.to_text())
    assert (q.mini, q.controls) == (p.mini, p.controls)


def test_canonical_control_order_is_stable():
    a = Pattern("supersaw", {"room": 0.4, "note": 7.0, "cutoff": 800.0})
    b = Pattern("supersaw", {"note": 7.0, "cutoff": 800.0, "room": 0.4})
    assert a.to_text() == b.to_text()
    assert a.to_text().index("note") < a.to_text().index("cutoff")
    assert a.to_text().index("cutoff") < a.to_text().index("room")


def test_validate_synth_config():
    p = Pattern("supersaw supersaw:7", {"note": 7.0, "cutoff": 1200.0, "lfo": 2.0})
    assert validate(p, SOURCES).valid


def test_validate_rejects_inapplicable_synth_param():
    v = validate(Pattern("superkick", {"lfo": 2.0}), SOURCES)
    assert not v.valid and "not applicable" in v.reason


def test_validate_rejects_out_of_range():
    v = validate(Pattern("supersaw", {"cutoff": 99999.0}), SOURCES)
    assert not v.valid and "out of range" in v.reason


def test_validate_rejects_unknown_control():
    v = validate(Pattern("supersaw", {"warp": 1.0}), SOURCES)
    assert not v.valid


def test_validate_custom_def_gets_core_and_fx_only():
    assert validate(Pattern("mydef", {"cutoff": 500.0}), SOURCES).valid
    v = validate(Pattern("mydef", {"lfo": 1.0}), SOURCES)
    assert not v.valid and "not applicable" in v.reason


def test_validate_synth_selector_bounded():
    assert validate(Pattern("supersaw:7", {}), SOURCES).valid
    v = validate(Pattern("supersaw:99", {}), SOURCES)
    assert not v.valid and "out of range" in v.reason


def test_validate_control_count_bounded():
    controls = {"note": 1.0, "gain": 1.0, "pan": 0.5, "cutoff": 500.0}
    v = validate(Pattern("supersaw", controls), SOURCES, PatternBounds(max_controls=3))
    assert not v.valid and "controls" in v.reason


def test_per_synth_range_override_applies():
    # supernoise widens resonance to 0..1
    assert validate(Pattern("supernoise", {"resonance": 0.95}), SOURCES).valid
    assert not validate(Pattern("supersaw", {"resonance": 0.95}), SOURCES).valid
