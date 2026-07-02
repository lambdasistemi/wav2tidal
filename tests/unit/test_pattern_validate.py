"""Pattern grammar + validation (T025)."""

from __future__ import annotations

import pytest

from wav2tidal.core.pattern.model import Pattern, parse_pattern_text
from wav2tidal.core.pattern.validate import PatternBounds, validate

BANKS = {"bd": 4, "sn": 2, "hh": 8}


def _p(mini, **controls):
    return Pattern(mini=mini, controls=controls)


@pytest.mark.parametrize(
    "mini",
    [
        "bd sn hh",
        "bd:3 ~ sn:1",
        "[bd sn] hh*2",
        "bd(3,8)",
        "[bd sn, hh hh]",
        "bd*2 ~ [sn:1 hh:7]",
    ],
)
def test_valid_patterns_accepted(mini):
    assert validate(_p(mini), BANKS).valid


def test_syntax_error_rejected():
    assert not validate(_p("bd ["), BANKS).valid


def test_unknown_bank_rejected():
    v = validate(_p("kick sn"), BANKS)
    assert not v.valid and "unknown bank" in v.reason


def test_index_out_of_range_rejected():
    v = validate(_p("sn:5"), BANKS)  # sn has size 2
    assert not v.valid and "out of range" in v.reason


def test_nesting_bound_enforced():
    v = validate(_p("[[[[bd]]]]"), BANKS, PatternBounds(max_nesting_depth=2))
    assert not v.valid and "nesting" in v.reason


def test_event_bound_enforced():
    v = validate(_p("bd*8 bd*8"), BANKS, PatternBounds(max_events_per_cycle=4))
    assert not v.valid and "events/cycle" in v.reason


def test_text_roundtrip():
    p = Pattern(mini="bd:1 ~ sn", controls={"gain": 1.0, "speed": 2.0, "pan": 0.5})
    assert parse_pattern_text(p.to_text()) == Pattern(
        mini="bd:1 ~ sn",
        controls={"gain": 1.0, "speed": 2.0, "pan": 0.5},
        source="model",
    )


def test_malformed_line_raises():
    with pytest.raises(ValueError):
        parse_pattern_text("play some jazz")
