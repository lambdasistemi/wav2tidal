"""Renderer routing + event mapping (core/pattern/dirt.py, issue #21)."""

from __future__ import annotations

from wav2tidal.core.pattern.dirt import MIX, NRT, RT, render_events, route
from wav2tidal.core.pattern.model import Pattern
from wav2tidal.core.pattern.validate import Sources

SOURCES = Sources(banks={"bd": 4}, custom=frozenset({"mydef"}))


def _r(mini, **controls):
    return route(Pattern(mini, controls), SOURCES)


def test_global_send_always_routes_rt():
    assert _r("supersaw", room=0.4) == RT
    assert _r("bd", delaytime=0.2) == RT


def test_banks_only_plain_controls_route_mix():
    assert _r("bd bd:2 ~", gain=1.0, speed=2.0, pan=0.5) == MIX


def test_banks_with_fx_route_rt():
    # the numpy mixdown has no FX DSP; NRT has no sample buffers
    assert _r("bd", cutoff=800.0) == RT
    assert _r("bd", note=7.0) == RT
    assert _r("bd", attack=0.1, release=0.5) == RT


def test_synth_bare_def_controls_route_nrt():
    assert _r("supersaw", note=7.0, lfo=2.0, pan=0.3) == NRT
    assert _r("supersaw supersaw:7") == NRT
    assert _r("mydef", note=-12.0) == NRT  # custom defs: core args only


def test_synth_needing_module_chain_routes_rt():
    # event FX / envelope / gain go through SuperDirt's dirt_* chain,
    # which the NRT score does not build yet (issue #24)
    assert _r("supersaw", cutoff=800.0) == RT
    assert _r("supersaw", vowel="a") == RT
    assert _r("supersaw", attack=0.1, release=0.5) == RT
    assert _r("supersaw", gain=1.1) == RT
    assert _r("mydef", lfo=2.0) == RT  # not a bare-def arg for a custom def


def test_mixed_banks_and_synths_route_rt():
    assert _r("bd supersaw") == RT


def test_multi_synth_nrt_needs_intersection():
    # lfo is a supersaw arg but not a superkick arg
    assert _r("supersaw superkick", lfo=2.0) == RT
    assert _r("supersaw superkick", note=3.0) == NRT


def test_render_events_rt_carries_controls_and_sustain():
    p = Pattern("supersaw bd", {"note": 7.0, "room": 0.4})
    evs = render_events(p, SOURCES, cps=1.0, n_cycles=1, mode=RT)
    by_name = {name: params for _, name, params in evs}
    assert by_name["supersaw"]["room"] == 0.4
    assert by_name["supersaw"]["sustain"] == 0.5  # slot duration
    assert "sustain" not in by_name["bd"]  # samples keep buffer-length default


def test_render_events_nrt_maps_bare_def_args():
    p = Pattern("supersaw:7 ~", {"note": 12.0, "lfo": 2.0, "cutoff": 500.0})
    evs = render_events(p, SOURCES, cps=1.0, n_cycles=1, mode=NRT)
    assert len(evs) == 1
    t, name, params = evs[0]
    assert (t, name) == (0.0, "supersaw")
    assert params["n"] == 7.0 and params["lfo"] == 2.0
    assert "cutoff" not in params and "freq" in params and "sustain" in params


def test_render_events_are_timed():
    p = Pattern("supersaw supersaw", {})
    evs = render_events(p, SOURCES, cps=0.5, n_cycles=1, mode=RT)
    assert [t for t, _, _ in evs] == [0.0, 1.0]
