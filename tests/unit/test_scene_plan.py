"""Scene routing + render-plan compilation + script builders (issue #29)."""

from __future__ import annotations

import pytest

from wav2tidal.core.pattern.dirt import NRT, RT, scene_plan, scene_route
from wav2tidal.core.pattern.model import Pattern, Scene, Trajectory, Voice
from wav2tidal.core.pattern.params import midicps
from wav2tidal.core.pattern.validate import Sources
from wav2tidal.io.superdirt import (
    build_nrt_scene_script,
    build_rt_scene_batch_script,
)

SOURCES = Sources(banks={"bd": 4}, custom=frozenset({"mydef"}))


def _scene(voices, layer=None):
    return Scene(voices=tuple(voices), layer=layer)


def _voice(**kw):
    kw.setdefault("source_name", "supersaw")
    return Voice(**kw)


def test_route_nrt_for_bare_scene():
    s = _scene(
        [
            _voice(
                controls={"note": -12.0},
                mods=(Trajectory("cutoff", "ramp", (200.0, 2000.0)),),
            )
        ]
    )
    assert scene_route(s, SOURCES) == NRT


def test_route_only_layer_forces_rt():
    # global FX render in NRT since issue #40 (the scene graph owns them)
    assert scene_route(_scene([_voice(controls={"room": 0.4})]), SOURCES) == NRT
    assert (
        scene_route(
            _scene([_voice(mods=(Trajectory("delaytime", "ramp", (0.1, 0.5)),))]),
            SOURCES,
        )
        == NRT
    )
    assert scene_route(_scene([_voice()], layer=Pattern("bd", {})), SOURCES) == RT


def test_route_rejects_vowel_voice():
    with pytest.raises(ValueError):
        scene_route(_scene([_voice(controls={"vowel": "a"})]), SOURCES)


def _plan(scene, duration=4.0, cps=0.5, tick=0.5):
    return scene_plan(scene, SOURCES, duration, cps, tick)


def test_plan_spawns_fx_chain_for_modulated_activator():
    plan = _plan(
        _scene([_voice(mods=(Trajectory("cutoff", "ramp", (200.0, 2000.0)),))])
    )
    chain = plan.chains[0]
    assert [n.synth for n in chain] == ["supersaw", "dirt_lpf"]
    assert chain[1].is_fx
    # t=0 knot lands in creation args; later knots in the automation
    assert chain[1].params["cutoff"] == 200.0
    assert any(
        ref == "v0_lpf" and arg == "cutoff" for _, ref, arg, _ in plan.automation
    )


def test_plan_note_trajectory_becomes_freq():
    plan = _plan(_scene([_voice(mods=(Trajectory("note", "ramp", (-12.0, 0.0)),))]))
    src = plan.chains[0][0]
    assert abs(src.params["freq"] - midicps(48.0)) < 1e-6
    rows = [r for r in plan.automation if r[2] == "freq"]
    assert rows and abs(rows[-1][3] - midicps(60.0)) < 1e-6


def test_plan_resonance_fans_out_to_source_and_lpf():
    plan = _plan(
        _scene(
            [
                _voice(
                    controls={"cutoff": 800.0},
                    mods=(Trajectory("resonance", "sine", (0.4, 0.3, 0.5)),),
                )
            ]
        )
    )
    refs = {(ref, arg) for _, ref, arg, _ in plan.automation}
    assert ("v0", "resonance") in refs  # supersaw lists resonance
    assert ("v0_lpf", "resonance") in refs  # and the chained filter gets it


def test_plan_bandf_renames_to_bandqf():
    plan = _plan(_scene([_voice(mods=(Trajectory("bandf", "ramp", (200.0, 2000.0)),))]))
    assert plan.chains[0][1].synth == "dirt_bpf"
    assert "bandqf" in plan.chains[0][1].params
    assert all(arg == "bandqf" for _, ref, arg, _ in plan.automation if "bpf" in ref)


def test_plan_globals_and_layer():
    plan = _plan(
        _scene(
            [
                _voice(
                    controls={"room": 0.4},
                    mods=(Trajectory("size", "ramp", (0.2, 0.9)),),
                )
            ],
            layer=Pattern("bd bd:2", {"gain": 0.9}),
        )
    )
    assert plan.globals_static["room"] == 0.4
    assert any(
        ref == "g_reverb" and arg == "size" for _, ref, arg, _ in plan.automation
    )
    assert len(plan.layer_events) == 4  # 2 events x 2 cycles


def test_plan_automation_is_time_sorted():
    plan = _plan(
        _scene(
            [
                _voice(
                    mods=(
                        Trajectory("cutoff", "ramp", (200.0, 2000.0)),
                        Trajectory("pan", "sine", (0.5, 0.4, 0.5)),
                    )
                )
            ]
        )
    )
    times = [t for t, *_ in plan.automation]
    assert times == sorted(times)


# -- script builders ----------------------------------------------------------


def test_nrt_scene_script_structure():
    plan = _plan(
        _scene(
            [
                _voice(
                    controls={"note": -12.0},
                    mods=(Trajectory("cutoff", "ramp", (200.0, 2000.0)),),
                )
            ]
        )
    )
    s = build_nrt_scene_script(
        plan, "/tmp/out.wav", "/tmp/score.osc", ["/q/lib.scd", "/q/core.scd"]
    )
    assert "SynthDef(\\w2t_route" in s and "RandSeed.ir(1, seed)" in s
    assert "SynthDescLib.global[\\dirt_lpf2]" in s
    assert "[\\s_new, \\supersaw, 2000, 1, 0," in s
    assert "[\\s_new, \\dirt_lpf2, 2001, 1, 0," in s
    assert "\\n_set, 2001, \\cutoff," in s
    assert "duration: 4" in s


def test_rt_scene_batch_script_structure():
    plan = _plan(
        _scene(
            [
                _voice(
                    controls={"room": 0.5},
                    mods=(Trajectory("cutoff", "ramp", (200.0, 2000.0)),),
                )
            ],
            layer=Pattern("bd", {}),
        )
    )
    s = build_rt_scene_batch_script([("/tmp/a.wav", plan)], banks_dir="/ws/banks")
    assert s.count("SuperDirt(2, s)") == 1
    assert '~g_reverb = Synth("dirt_reverb"' in s and "\\room, 0.5" in s
    assert '~n_v0 = Synth.tail(~orbit.group, "supersaw"' in s
    assert '~n_v0_lpf = Synth.tail(~orbit.group, "dirt_lpf2"' in s
    assert "~n_v0_route = Synth.tail(~orbit.group, \\w2t_route" in s
    assert "~n_v0_lpf.set(\\cutoff," in s
    assert '"/dirt/play"' in s  # the layer
    assert "~n_v0.free;" in s and "~g_reverb.free;" in s
    assert "WAV2TIDAL_JOB_0_DONE" in s


def test_nrt_scene_script_includes_global_fx_graph():
    plan = _plan(
        _scene(
            [
                _voice(
                    controls={"room": 0.5},
                    mods=(Trajectory("size", "ramp", (0.2, 0.9)),),
                )
            ]
        )
    )
    s = build_nrt_scene_script(
        plan, "/tmp/out.wav", "/tmp/score.osc", ["/q/lib.scd", "/q/core.scd"]
    )
    assert "SynthDescLib.global[\\dirt_reverb2]" in s
    assert "\\s_new, \\dirt_reverb2" in s and "\\room, 0.5" in s
    assert "SynthDef(\\w2t_monitor" in s and "\\s_new, \\w2t_monitor" in s
    # route feeds the dry bus, size trajectory automates the reverb node
    assert "\\out, 8]]" in s
    assert "\\size," in s


def test_nrt_scene_script_without_globals_has_no_fx_graph():
    plan = _plan(
        _scene([_voice(mods=(Trajectory("cutoff", "ramp", (200.0, 2000.0)),))])
    )
    s = build_nrt_scene_script(
        plan, "/tmp/out.wav", "/tmp/score.osc", ["/q/lib.scd", "/q/core.scd"]
    )
    assert "dirt_reverb" not in s and "dirt_delay" not in s
    assert "\\s_new, \\w2t_monitor" in s  # monitor always sums dry+effect
