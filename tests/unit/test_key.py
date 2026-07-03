"""Tests for core/pattern/key.py — pitch classes, note snapping (issue #58)."""

from __future__ import annotations

from wav2tidal.core.pattern.key import (
    PITCH_NAMES,
    parse_key,
    pitch_classes,
    snap_note,
    snap_scene,
)
from wav2tidal.core.pattern.model import Scene, Trajectory, Voice
from wav2tidal.core.pattern.params import spec

_NOTE_LO = int(spec("note").lo)  # -24
_NOTE_HI = int(spec("note").hi)  # 24

# ---------------------------------------------------------------------------
# PITCH_NAMES
# ---------------------------------------------------------------------------


def test_pitch_names_length():
    assert len(PITCH_NAMES) == 12


def test_pitch_names_no_flats():
    """All names are sharps-only (matching features.py _PITCH_CLASSES)."""
    for name in PITCH_NAMES:
        assert "b" not in name


def test_pitch_names_c_is_zero():
    assert PITCH_NAMES[0] == "C"


# ---------------------------------------------------------------------------
# pitch_classes()
# ---------------------------------------------------------------------------


def test_pitch_classes_c_major():
    pcs = pitch_classes("C")
    assert pcs == frozenset({0, 2, 4, 5, 7, 9, 11})


def test_pitch_classes_a_major():
    # A major: A B C# D E F# G# → pcs 9 11 1 2 4 6 8
    pcs = pitch_classes("A")
    assert pcs == frozenset({9, 11, 1, 2, 4, 6, 8})


def test_pitch_classes_f_sharp_major():
    # F# major: F# G# A# B C# D# F → pcs 6 8 10 11 1 3 5
    pcs = pitch_classes("F#")
    assert pcs == frozenset({6, 8, 10, 11, 1, 3, 5})


def test_pitch_classes_a_minor():
    # A natural minor: A B C D E F G → pcs 9 11 0 2 4 5 7
    pcs = pitch_classes("Am")
    assert pcs == frozenset({9, 11, 0, 2, 4, 5, 7})


def test_pitch_classes_f_sharp_minor():
    # F# natural minor: F# G# A B C# D E → pcs 6 8 9 11 1 2 4
    pcs = pitch_classes("F#m")
    assert pcs == frozenset({6, 8, 9, 11, 1, 2, 4})


def test_pitch_classes_d_minor():
    # D natural minor: D E F G A A# C → pcs 2 4 5 7 9 10 0
    pcs = pitch_classes("Dm")
    assert pcs == frozenset({2, 4, 5, 7, 9, 10, 0})


def test_pitch_classes_na_returns_none():
    assert pitch_classes("N/A") is None


def test_pitch_classes_empty_returns_none():
    assert pitch_classes("") is None


def test_pitch_classes_unknown_returns_none():
    assert pitch_classes("Xb") is None
    assert pitch_classes("Hmaj") is None


def test_pitch_classes_major_has_7_notes():
    for name in PITCH_NAMES:
        pcs = pitch_classes(name)
        assert pcs is not None and len(pcs) == 7


def test_pitch_classes_minor_has_7_notes():
    for name in PITCH_NAMES:
        pcs = pitch_classes(name + "m")
        assert pcs is not None and len(pcs) == 7


# ---------------------------------------------------------------------------
# parse_key()
# ---------------------------------------------------------------------------


def test_parse_key_extracts_major():
    assert (
        parse_key("tempo=141 density=lo key=F# brightness=1/5 motion=falling") == "F#"
    )


def test_parse_key_extracts_minor():
    assert (
        parse_key("tempo=141 density=lo key=F#m brightness=1/5 motion=falling") == "F#m"
    )


def test_parse_key_na_returns_none():
    assert (
        parse_key("tempo=120 density=hi key=N/A brightness=3/5 motion=steady") is None
    )


def test_parse_key_missing_token_returns_none():
    assert parse_key("tempo=90 density=lo brightness=2/5 motion=rising") is None


def test_parse_key_c_major():
    assert (
        parse_key("tempo=120 density=medium key=C brightness=mid motion=steady") == "C"
    )


def test_parse_key_empty_string_returns_none():
    assert parse_key("") is None


# ---------------------------------------------------------------------------
# snap_note()
# ---------------------------------------------------------------------------

# C major pitch classes: {0, 2, 4, 5, 7, 9, 11}
_C_MAJOR = frozenset({0, 2, 4, 5, 7, 9, 11})
# F# minor pitch classes: {1, 2, 4, 6, 8, 9, 11}
_FS_MINOR = frozenset({1, 2, 4, 6, 8, 9, 11})


def test_snap_note_in_key_unchanged():
    """A value already on an in-key pitch class stays unchanged."""
    # C is in C major (pc 0)
    assert snap_note(0.0, _C_MAJOR) == 0.0
    # E is in C major (pc 4)
    assert snap_note(4.0, _C_MAJOR) == 4.0
    # G is in C major (pc 7)
    assert snap_note(7.0, _C_MAJOR) == 7.0


def test_snap_note_out_of_key_snapped():
    """C# (pc 1) is not in C major; snaps to C (0) or D (2), tie → C (lower)."""
    result = snap_note(1.0, _C_MAJOR)
    assert result == 0.0  # tie resolves downward: 0 < 2


def test_snap_note_f_sharp_in_c_major():
    """F# (pc 6) is not in C major; F=5, G=7 both distance 1, tie → F (lower)."""
    result = snap_note(6.0, _C_MAJOR)
    assert result == 5.0


def test_snap_note_bb_in_c_major():
    """A# (pc 10) is not in C major; A=9, B=11 both distance 1, tie → A (lower)."""
    result = snap_note(10.0, _C_MAJOR)
    assert result == 9.0


def test_snap_note_negative_value():
    """Negative note values work: -12 is C (pc=0), stays for C major."""
    assert snap_note(-12.0, _C_MAJOR) == -12.0


def test_snap_note_negative_out_of_key():
    """−11 is C# (pc=1), not in C major; snaps to −12 (C) or −10 (D), tie → −12."""
    result = snap_note(-11.0, _C_MAJOR)
    assert result == -12.0


def test_snap_note_returns_integer_valued_float():
    result = snap_note(1.7, _C_MAJOR)
    assert result == float(int(result))


def test_snap_note_fractional_rounds_first():
    """1.7 rounds to 2 (D, pc 2 ∈ C major) → stays 2."""
    assert snap_note(1.7, _C_MAJOR) == 2.0


def test_snap_note_clamp_at_hi():
    """Values that would snap above 24 are clamped to 24."""
    # 24 = C (pc=0), in C major → stays
    assert snap_note(24.0, _C_MAJOR) == 24.0
    # 23 = B (pc=11), in C major → stays
    assert snap_note(23.0, _C_MAJOR) == 23.0


def test_snap_note_clamp_at_lo():
    """Values that would snap below -24 are clamped to -24."""
    # -24 = C (pc=0), in C major → stays
    assert snap_note(-24.0, _C_MAJOR) == -24.0


def test_snap_note_result_in_bounds():
    """snap_note result is always within spec bounds."""
    for v in range(-26, 27):
        result = snap_note(float(v), _C_MAJOR)
        assert _NOTE_LO <= result <= _NOTE_HI


def test_snap_note_f_sharp_minor():
    """F# (pc=6) is in F#m {1,2,4,6,8,9,11} → unchanged."""
    assert snap_note(6.0, _FS_MINOR) == 6.0


def test_snap_note_d_sharp_not_in_f_sharp_minor():
    """D# (pc=3) not in F#m; D=2, E=4 both distance 1, tie → 2 (D, lower)."""
    assert snap_note(3.0, _FS_MINOR) == 2.0


# ---------------------------------------------------------------------------
# snap_scene()
# ---------------------------------------------------------------------------


def _voice(note: float | None = None, mods: tuple[Trajectory, ...] = ()) -> Voice:
    controls: dict = {"note": note} if note is not None else {}
    return Voice(source_name="supersaw", n=0, controls=controls, mods=mods)


def _scene(*voices: Voice) -> Scene:
    return Scene(voices=voices, layer=None, source="sampled")


def test_snap_scene_label_none_unchanged():
    """None label → scene returned unchanged."""
    v = _voice(note=3.0)
    s = _scene(v)
    out = snap_scene(s, None)
    assert out is s


def test_snap_scene_na_unchanged():
    """'N/A' yields no pitch classes → scene returned unchanged."""
    v = _voice(note=3.0)
    s = _scene(v)
    out = snap_scene(s, "N/A")
    assert out is s


def test_snap_scene_static_note_snapped():
    """Static note out-of-key is snapped."""
    # D# (pc=3) not in C major; snaps to D (2) or E (4), tie → D (2)
    v = _voice(note=3.0)
    s = _scene(v)
    out = snap_scene(s, "C")
    assert out.voices[0].controls["note"] == 2.0


def test_snap_scene_static_note_in_key_unchanged():
    """Static note already in key is left unchanged."""
    v = _voice(note=4.0)  # E, pc=4 ∈ C major
    s = _scene(v)
    out = snap_scene(s, "C")
    assert out.voices[0].controls["note"] == 4.0


def test_snap_scene_no_note_control_unchanged():
    """Voice with no 'note' control passes through untouched."""
    v = Voice(source_name="supersaw", n=0, controls={"gain": 1.0}, mods=())
    s = _scene(v)
    out = snap_scene(s, "C")
    assert "note" not in out.voices[0].controls


def test_snap_scene_steps_all_snapped():
    """steps trajectory: every arg is snapped to key."""
    # D# (3) and F# (6) not in C major
    traj = Trajectory(param="note", shape="steps", args=(3.0, 6.0, 4.0))
    v = _voice(mods=(traj,))
    s = _scene(v)
    out = snap_scene(s, "C")
    new_traj = out.voices[0].mods[0]
    c_major = frozenset({0, 2, 4, 5, 7, 9, 11})
    for a in new_traj.args:
        assert int(a) % 12 in c_major


def test_snap_scene_steps_values():
    """steps args snap to nearest in-key semitone."""
    # 3 → 2 (D), 6 → 5 (F), 4 stays (E)
    traj = Trajectory(param="note", shape="steps", args=(3.0, 6.0, 4.0))
    v = _voice(mods=(traj,))
    s = _scene(v)
    out = snap_scene(s, "C")
    new_args = out.voices[0].mods[0].args
    assert new_args == (2.0, 5.0, 4.0)


def test_snap_scene_sine_first_arg_only():
    """sine: only center (first arg) is snapped; depth and rate are untouched."""
    # center=3.0 (D#, not in C major) → snaps to 2.0; depth=1.0, rate=0.5 unchanged
    traj = Trajectory(param="note", shape="sine", args=(3.0, 1.0, 0.5))
    v = _voice(mods=(traj,))
    s = _scene(v)
    out = snap_scene(s, "C")
    new_args = out.voices[0].mods[0].args
    assert new_args[0] == 2.0  # snapped center
    assert new_args[1] == 1.0  # depth unchanged
    assert new_args[2] == 0.5  # rate unchanged


def test_snap_scene_ramp_both_endpoints():
    """ramp is a portamento between two pitches: both endpoints snapped."""
    traj = Trajectory(param="note", shape="ramp", args=(3.0, 6.0))
    v = _voice(mods=(traj,))
    s = _scene(v)
    out = snap_scene(s, "C")
    new_args = out.voices[0].mods[0].args
    assert new_args[0] == 2.0  # snapped start (D# -> D)
    assert new_args[1] == 5.0  # snapped end (F# -> F, tie resolves down)


def test_snap_scene_walk_first_arg_only():
    """walk: only center (first arg) is snapped; depth, rate, seed untouched."""
    traj = Trajectory(param="note", shape="walk", args=(3.0, 1.0, 0.5, 42.0))
    v = _voice(mods=(traj,))
    s = _scene(v)
    out = snap_scene(s, "C")
    new_args = out.voices[0].mods[0].args
    assert new_args[0] == 2.0  # snapped center
    assert new_args[1:] == (1.0, 0.5, 42.0)  # depth, rate, seed unchanged


def test_snap_scene_non_note_trajectory_unchanged():
    """Trajectories for params other than 'note' are not touched."""
    traj = Trajectory(param="cutoff", shape="sine", args=(3.0, 1.0, 0.5))
    v = _voice(mods=(traj,))
    s = _scene(v)
    out = snap_scene(s, "C")
    assert out.voices[0].mods[0].args == (3.0, 1.0, 0.5)


def test_snap_scene_preserves_source():
    s = Scene(voices=(_voice(note=3.0),), layer=None, source="model")
    out = snap_scene(s, "C")
    assert out.source == "model"


def test_snap_scene_preserves_layer():
    from wav2tidal.core.pattern.model import Pattern

    layer = Pattern(mini="bd", controls={})
    s = Scene(voices=(_voice(note=3.0),), layer=layer, source="sampled")
    out = snap_scene(s, "C")
    assert out.layer is layer


def test_snap_scene_multiple_voices():
    """All voices are snapped independently."""
    v1 = _voice(note=3.0)  # D# → snaps to D (2)
    v2 = _voice(note=6.0)  # F# → snaps to F (5)
    s = _scene(v1, v2)
    out = snap_scene(s, "C")
    assert out.voices[0].controls["note"] == 2.0
    assert out.voices[1].controls["note"] == 5.0


def test_snap_scene_f_sharp_minor():
    """Snap to F# minor key ({1,2,4,6,8,9,11})."""
    # note=3 (D#) not in F#m; D=2 ✓, E=4 ✓ both distance 1, tie → 2
    v = _voice(note=3.0)
    s = _scene(v)
    out = snap_scene(s, "F#m")
    assert out.voices[0].controls["note"] == 2.0
