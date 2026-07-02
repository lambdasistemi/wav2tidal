"""Trajectory sampling: shapes -> timed knots (issue #29)."""

from __future__ import annotations

import math

from wav2tidal.core.pattern.model import Trajectory
from wav2tidal.core.pattern.trajectory import knots


def _k(shape, args, **kw):
    defaults = dict(duration=4.0, cps=0.5, tick=0.5, lo=0.0, hi=1.0, log=False)
    defaults.update(kw)
    return knots(Trajectory("x", shape, tuple(args)), **defaults)


def test_ramp_endpoints_and_linearity():
    ks = _k("ramp", (0.0, 1.0))
    assert ks[0] == (0.0, 0.0) and ks[-1] == (4.0, 1.0)
    assert abs(dict(ks)[2.0] - 0.5) < 1e-9


def test_ramp_log_space_for_log_params():
    ks = _k("ramp", (100.0, 10000.0), lo=60.0, hi=12000.0, log=True)
    assert abs(dict(ks)[2.0] - 1000.0) < 1e-6  # geometric midpoint


def test_sine_oscillates_around_center():
    ks = _k("sine", (0.5, 0.3, 0.25), tick=1.0)  # period 4s
    vals = dict(ks)
    assert abs(vals[1.0] - 0.8) < 1e-9  # peak at quarter period
    assert abs(vals[3.0] - 0.2) < 1e-9  # trough


def test_walk_is_seeded_and_bounded():
    a = _k("walk", (0.5, 0.4, 0.5, 7.0))
    b = _k("walk", (0.5, 0.4, 0.5, 7.0))
    c = _k("walk", (0.5, 0.4, 0.5, 8.0))
    assert a == b and a != c
    assert all(0.1 - 1e-9 <= v <= 0.9 + 1e-9 for _, v in a)


def test_steps_hold_per_cycle():
    ks = _k("steps", (0.1, 0.9), tick=0.25)  # cycle = 2s -> 1s per step
    vals = dict(ks)
    assert vals[0.0] == 0.1
    assert vals[1.0] == 0.9
    assert vals[2.0] == 0.1  # repeats next cycle


def test_values_clamped_and_deduped():
    ks = _k("sine", (0.5, 0.0, 1.0))  # zero depth -> constant
    assert ks == [(0.0, 0.5)]  # duplicates dropped
    ks = _k("ramp", (0.0, 2.0), hi=1.0)  # escapes hi -> clamped
    assert max(v for _, v in ks) == 1.0


def test_math_sanity_full_period():
    ks = _k("sine", (0.5, 0.2, 0.5), tick=0.125)
    for t, v in ks:
        assert abs(v - (0.5 + 0.2 * math.sin(2 * math.pi * 0.5 * t))) < 1e-6
