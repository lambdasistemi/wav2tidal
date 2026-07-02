"""Scheduler + renderer determinism and timing (T027)."""

from __future__ import annotations

import numpy as np

from wav2tidal.core.pattern.model import Pattern
from wav2tidal.core.render.mixdown import Banks, render
from wav2tidal.core.render.schedule import bjorklund, schedule_events


def _banks(sr=8000):
    # distinct short clicks per bank so mixes are identifiable
    return Banks(
        sr=sr,
        data={
            "bd": [np.ones(200, dtype=np.float32) * 0.9],
            "sn": [
                np.ones(100, dtype=np.float32) * 0.5,
                np.ones(100, dtype=np.float32) * 0.3,
            ],
        },
    )


def test_schedule_places_events_in_order():
    ev = schedule_events(Pattern("bd sn"), cps=1.0, n_cycles=1)
    assert [e.bank for e in ev] == ["bd", "sn"]
    assert ev[0].start == 0.0
    assert ev[1].start > ev[0].start


def test_rest_produces_no_event():
    ev = schedule_events(Pattern("bd ~ sn"), cps=1.0, n_cycles=1)
    assert len(ev) == 2


def test_fast_multiplies_events():
    ev = schedule_events(Pattern("bd*4"), cps=1.0, n_cycles=1)
    assert len(ev) == 4


def test_euclid_onset_count():
    ev = schedule_events(Pattern("bd(3,8)"), cps=1.0, n_cycles=1)
    assert len(ev) == 3


def test_bjorklund_shape():
    p = bjorklund(3, 8)
    assert len(p) == 8 and sum(p) == 3


def test_render_is_deterministic():
    p = Pattern("bd sn:1 [bd sn]", controls={"gain": 1.0, "speed": 1.0})
    banks = _banks()
    ev = schedule_events(p, cps=2.0, n_cycles=2)
    a = render(ev, banks, total_seconds=1.0, sr=banks.sr)
    b = render(ev, banks, total_seconds=1.0, sr=banks.sr)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
    assert np.abs(a).max() > 0.0  # produced sound


def test_speed_changes_slice_length():
    banks = _banks()
    fast = schedule_events(Pattern("bd", controls={"speed": 2.0}), cps=1.0)
    a = render(fast, banks, total_seconds=1.0, sr=banks.sr)
    # a 200-sample clip at speed 2 becomes ~100 samples; energy still present
    assert np.count_nonzero(a) <= 120
