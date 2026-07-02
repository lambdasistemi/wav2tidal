"""Grammar v3 scenes: membership, round-trip, validation (design-change-002)."""

from __future__ import annotations

import pytest
from lark.exceptions import LarkError

from wav2tidal.core.pattern.grammar import parse_scene
from wav2tidal.core.pattern.model import (
    Pattern,
    Scene,
    Trajectory,
    Voice,
    parse_scene_text,
)
from wav2tidal.core.pattern.validate import SceneBounds, Sources, validate_scene

SOURCES = Sources(banks={"bd": 4}, custom=frozenset({"mydef"}))

TEXT = (
    "scene voice supersaw # note -12 # lfo 0.5 "
    "mod cutoff sine 800 600 0.25 mod resonance ramp 0.2 0.6 "
    "voice superhammond:3 # vibrato 0.5 mod room walk 0.4 0.3 0.5 7 "
    'layer d1 $ s "bd(3,8) [bd bd]" # gain 0.9'
)


@pytest.mark.parametrize(
    "text",
    [
        TEXT,
        "scene voice supersaw",
        "scene voice mydef # note -12 mod pan sine 0.5 0.4 0.1",
        "scene voice supersaw mod note steps -12 -5 0 3",
        "scene voice supersaw # voice 0.5 mod voice sine 0.5 0.3 0.2",  # param 'voice'
    ],
)
def test_scene_membership(text):
    parse_scene(text)


@pytest.mark.parametrize(
    "text",
    [
        "scene",  # no voices
        "scene voice supersaw mod cutoff sine 800 600",  # sine needs 3 args
        "scene voice supersaw mod cutoff wave 1 2 3",  # unknown shape
        "scene voice supersaw mod cutoff walk 1 2 3 4.5",  # seed must be INT
        'scene layer d1 $ s "bd"',  # layer without voices
        'scene voice supersaw layer d1 $ s "bd" voice superpwm',  # layer must be last
    ],
)
def test_scene_rejection(text):
    with pytest.raises(LarkError):
        parse_scene(text)


def test_scene_text_roundtrip():
    scene = parse_scene_text(TEXT)
    assert len(scene.voices) == 2
    assert scene.voices[0].controls == {"note": -12.0, "lfo": 0.5}
    assert scene.voices[0].mods[0].shape in ("sine", "ramp")
    assert scene.voices[1].n == 3
    assert scene.layer is not None and scene.layer.mini == "bd(3,8) [bd bd]"
    again = parse_scene_text(scene.to_text())
    assert again == scene


def _scene(**voice_kw):
    return Scene(voices=(Voice(**voice_kw),))


def test_validate_scene_accepts_valid():
    s = _scene(
        source_name="supersaw",
        controls={"note": -12.0},
        mods=(Trajectory("cutoff", "sine", (800.0, 600.0, 0.25)),),
    )
    assert validate_scene(s, SOURCES).valid
    assert validate_scene(parse_scene_text(TEXT), SOURCES).valid


def test_validate_scene_rejects_bank_voice():
    v = validate_scene(_scene(source_name="bd"), SOURCES)
    assert not v.valid and "unknown voice source" in v.reason


def test_validate_scene_rejects_inapplicable_mod():
    s = _scene(source_name="superkick", mods=(Trajectory("lfo", "ramp", (0.0, 2.0)),))
    v = validate_scene(s, SOURCES)
    assert not v.valid and "not applicable" in v.reason


def test_validate_scene_rejects_non_modulatable_param():
    # coarse is an integer, gain/attack trigger-only (dirt_gate declares
    # gain \ir); vowel can't even be written (not in the PARAM terminal)
    for traj in (
        Trajectory("coarse", "ramp", (1.0, 8.0)),
        Trajectory("gain", "ramp", (0.6, 1.0)),
        Trajectory("attack", "ramp", (0.0, 0.3)),
    ):
        v = validate_scene(_scene(source_name="supersaw", mods=(traj,)), SOURCES)
        assert not v.valid and "not modulatable" in v.reason, traj.param
    v = validate_scene(
        _scene(source_name="supersaw", mods=(Trajectory("vowel", "ramp", (0.0, 1.0)),)),
        SOURCES,
    )
    assert not v.valid and "syntax" in v.reason  # vowel mod is unwritable


def test_validate_scene_rejects_out_of_range_shape():
    s = _scene(
        source_name="supersaw",
        mods=(Trajectory("resonance", "sine", (0.5, 0.6, 0.25)),),  # 0.5±0.6 escapes
    )
    v = validate_scene(s, SOURCES)
    assert not v.valid and "invalid" in v.reason


def test_validate_scene_rejects_double_set_param():
    s = _scene(
        source_name="supersaw",
        controls={"cutoff": 500.0},
        mods=(Trajectory("cutoff", "ramp", (200.0, 2000.0)),),
    )
    v = validate_scene(s, SOURCES)
    assert not v.valid and "twice" in v.reason


def test_validate_scene_bounds():
    voices = tuple(Voice(source_name="supersaw") for _ in range(5))
    v = validate_scene(Scene(voices=voices), SOURCES, SceneBounds(max_voices=4))
    assert not v.valid and "voices" in v.reason


def test_validate_scene_checks_layer():
    s = Scene(
        voices=(Voice(source_name="supersaw"),),
        layer=Pattern("nosuchbank", {}),
    )
    v = validate_scene(s, SOURCES)
    assert not v.valid and "layer" in v.reason


def test_global_mod_allowed_on_any_voice():
    s = _scene(
        source_name="mydef",
        mods=(Trajectory("room", "ramp", (0.0, 0.8)),),
    )
    assert validate_scene(s, SOURCES).valid
