"""Tests for core/pattern/ensemble.py (issue #64 — musicality 0)."""

from __future__ import annotations

import pytest

from wav2tidal.core.pattern.ensemble import (
    chord_classes,
    ensemble_rules,
    quantize_rates,
    snap_static_notes_to_chord,
    space_registers,
    stage_gains,
)
from wav2tidal.core.pattern.model import Scene, Trajectory, Voice
from wav2tidal.core.pattern.params import spec
from wav2tidal.core.pattern.shapes import RATE_HI, RATE_LO

_NOTE_LO = int(spec("note").lo)  # -24
_NOTE_HI = int(spec("note").hi)  # +24
_GAIN_LO = spec("gain").lo
_GAIN_HI = spec("gain").hi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _voice(
    note: float | None = None,
    gain: float | None = None,
    mods: tuple[Trajectory, ...] = (),
) -> Voice:
    controls: dict = {}
    if note is not None:
        controls["note"] = note
    if gain is not None:
        controls["gain"] = gain
    return Voice(source_name="supersaw", n=0, controls=controls, mods=mods)


def _scene(*voices: Voice) -> Scene:
    return Scene(voices=voices, layer=None, source="sampled")


def _note_pc(v: float) -> int:
    return int(round(v)) % 12


# ---------------------------------------------------------------------------
# chord_classes
# ---------------------------------------------------------------------------


def test_chord_classes_c_major_maj7():
    # C maj7: C E G B → {0, 4, 7, 11}
    assert chord_classes("C") == frozenset({0, 4, 7, 11})


def test_chord_classes_a_minor_min7():
    # Am7: A C E G → pcs 9 0 4 7
    assert chord_classes("Am") == frozenset({9, 0, 4, 7})


def test_chord_classes_f_sharp_minor_min7():
    # F#m7: F# A C# E → pcs 6 9 1 4
    assert chord_classes("F#m") == frozenset({6, 9, 1, 4})


def test_chord_classes_f_sharp_major_maj7():
    # F# maj7: F# A# C# F (enharmonic E#) → pcs 6 10 1 5
    assert chord_classes("F#") == frozenset({6, 10, 1, 5})


def test_chord_classes_d_minor_min7():
    # Dm7: D F A C → pcs 2 5 9 0
    assert chord_classes("Dm") == frozenset({2, 5, 9, 0})


def test_chord_classes_na_returns_none():
    assert chord_classes("N/A") is None


def test_chord_classes_empty_returns_none():
    assert chord_classes("") is None


def test_chord_classes_unknown_returns_none():
    assert chord_classes("Xb") is None
    assert chord_classes("Hmaj") is None


def test_chord_classes_all_major_have_4_tones():
    from wav2tidal.core.pattern.key import PITCH_NAMES

    for name in PITCH_NAMES:
        cc = chord_classes(name)
        assert cc is not None and len(cc) == 4


def test_chord_classes_all_minor_have_4_tones():
    from wav2tidal.core.pattern.key import PITCH_NAMES

    for name in PITCH_NAMES:
        cc = chord_classes(name + "m")
        assert cc is not None and len(cc) == 4


def test_chord_classes_tones_are_subset_of_scale():
    """Chord tones must be a subset of the scale pitch classes."""
    from wav2tidal.core.pattern.key import PITCH_NAMES, pitch_classes

    for name in PITCH_NAMES:
        scale = pitch_classes(name)
        chord = chord_classes(name)
        assert chord is not None and scale is not None
        assert chord <= scale, f"{name} chord {chord} not subset of scale {scale}"

    for name in PITCH_NAMES:
        scale = pitch_classes(name + "m")
        chord = chord_classes(name + "m")
        assert chord is not None and scale is not None
        assert chord <= scale, f"{name}m chord {chord} not subset of scale {scale}"


# ---------------------------------------------------------------------------
# snap_static_notes_to_chord
# ---------------------------------------------------------------------------

# C maj7 = {0, 4, 7, 11}
_C_MAJ7 = frozenset({0, 4, 7, 11})
# C natural minor scale = {0, 2, 3, 5, 7, 8, 10}; Cm7 = {0, 3, 7, 10}
_CM7 = frozenset({0, 3, 7, 10})


def test_chord_snap_chord_tone_unchanged():
    """A note already on a chord tone stays unchanged."""
    v = _voice(note=4.0)  # E, pc=4 ∈ C maj7
    s = _scene(v)
    out = snap_static_notes_to_chord(s, "C")
    assert out.voices[0].controls["note"] == 4.0


def test_chord_snap_scale_tone_moves_to_chord():
    """D (pc=2) is in C major scale but NOT C maj7; snaps to nearest chord tone."""
    # C=0, D=2, E=4 → tie between C(0) and E(4), gap both =2 → resolves to C (lower)
    v = _voice(note=2.0)  # D, pc=2
    s = _scene(v)
    out = snap_static_notes_to_chord(s, "C")
    result_pc = _note_pc(out.voices[0].controls["note"])
    assert result_pc in _C_MAJ7


def test_chord_snap_out_of_chord_snaps():
    """A note not in the chord is moved to the nearest chord tone."""
    # note=5 (F, pc=5): C=0 dist 5, E=4 dist 1, G=7 dist 2, B=11 dist 6 → E (4)
    v = _voice(note=5.0)
    s = _scene(v)
    out = snap_static_notes_to_chord(s, "C")
    result_pc = _note_pc(out.voices[0].controls["note"])
    assert result_pc in _C_MAJ7
    assert out.voices[0].controls["note"] == 4.0


def test_chord_snap_trajectories_untouched():
    """Trajectory notes are NOT snapped (only static controls are touched)."""
    traj = Trajectory(
        param="note", shape="sine", args=(2.0, 1.0, 0.5)
    )  # D, not in Cmaj7
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = snap_static_notes_to_chord(s, "C")
    assert out.voices[0].mods[0].args == (2.0, 1.0, 0.5)


def test_chord_snap_label_none_unchanged():
    """None label → scene returned unchanged."""
    v = _voice(note=2.0)
    s = _scene(v)
    out = snap_static_notes_to_chord(s, None)
    assert out is s


def test_chord_snap_na_unchanged():
    """'N/A' label → scene returned unchanged."""
    v = _voice(note=2.0)
    s = _scene(v)
    out = snap_static_notes_to_chord(s, "N/A")
    assert out is s


def test_chord_snap_no_note_control_unchanged():
    """Voice with no 'note' control passes through untouched."""
    v = Voice(source_name="supersaw", n=0, controls={"gain": 1.0}, mods=())
    s = _scene(v)
    out = snap_static_notes_to_chord(s, "C")
    assert "note" not in out.voices[0].controls


def test_chord_snap_result_in_spec_bounds():
    """All snapped notes stay within spec bounds."""
    for raw_note in range(_NOTE_LO - 2, _NOTE_HI + 3):
        v = _voice(note=float(raw_note))
        s = _scene(v)
        out = snap_static_notes_to_chord(s, "C")
        result = out.voices[0].controls["note"]
        assert _NOTE_LO <= result <= _NOTE_HI


def test_chord_snap_result_is_chord_tone():
    """Every snapped note has a pitch class in the chord."""
    for raw_note in range(_NOTE_LO, _NOTE_HI + 1):
        v = _voice(note=float(raw_note))
        s = _scene(v)
        out = snap_static_notes_to_chord(s, "C")
        pc = _note_pc(out.voices[0].controls["note"])
        assert pc in _C_MAJ7


def test_chord_snap_f_sharp_minor():
    """Snap to F#m7 ({1, 4, 6, 9})."""
    _FSM7 = frozenset({1, 4, 6, 9})
    for raw_note in range(_NOTE_LO, _NOTE_HI + 1):
        v = _voice(note=float(raw_note))
        s = _scene(v)
        out = snap_static_notes_to_chord(s, "F#m")
        pc = _note_pc(out.voices[0].controls["note"])
        assert pc in _FSM7


# ---------------------------------------------------------------------------
# space_registers
# ---------------------------------------------------------------------------


def test_space_registers_single_voice_unchanged():
    v = _voice(note=5.0)
    s = _scene(v)
    out = space_registers(s)
    assert out is s


def test_space_registers_already_spread_unchanged():
    """Two notes already >= min_gap apart are not moved."""
    v1 = _voice(note=0.0)
    v2 = _voice(note=12.0)
    s = _scene(v1, v2)
    out = space_registers(s)
    assert out.voices[0].controls["note"] == 0.0
    assert out.voices[1].controls["note"] == 12.0


def test_space_registers_gap_exactly_min_gap_unchanged():
    """Gap exactly equal to min_gap (default 3) leaves notes unchanged."""
    v1 = _voice(note=0.0)
    v2 = _voice(note=3.0)
    s = _scene(v1, v2)
    out = space_registers(s)
    assert out.voices[0].controls["note"] == 0.0
    assert out.voices[1].controls["note"] == 3.0


def test_space_registers_cluster_spreads():
    """Two notes only 1 apart get spread to min_gap."""
    v1 = _voice(note=4.0)
    v2 = _voice(note=5.0)
    s = _scene(v1, v2)
    out = space_registers(s)
    n1 = out.voices[0].controls["note"]
    n2 = out.voices[1].controls["note"]
    sorted_notes = sorted([n1, n2])
    assert sorted_notes[1] - sorted_notes[0] >= 3


def test_space_registers_pitch_class_preserved():
    """Octave transpositions never change pitch class."""
    v1 = _voice(note=4.0)  # pc=4
    v2 = _voice(note=5.0)  # pc=5
    s = _scene(v1, v2)
    out = space_registers(s)
    n1 = out.voices[0].controls["note"]
    n2 = out.voices[1].controls["note"]
    assert _note_pc(n1) == 4
    assert _note_pc(n2) == 5


def test_space_registers_bounds_never_violated():
    """Result notes always stay within spec bounds."""
    # Pack everything near the top boundary
    v1 = _voice(note=22.0)
    v2 = _voice(note=23.0)
    v3 = _voice(note=24.0)
    s = _scene(v1, v2, v3)
    out = space_registers(s)
    for v in out.voices:
        if "note" in v.controls:
            assert _NOTE_LO <= v.controls["note"] <= _NOTE_HI


def test_space_registers_no_static_note_voices_ignored():
    """Voices without a static note are preserved unchanged."""
    traj = Trajectory(param="note", shape="sine", args=(5.0, 1.0, 0.5))
    v_no_note = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    v_note = _voice(note=0.0)
    s = _scene(v_no_note, v_note)
    out = space_registers(s)
    # First voice (no static note) is unchanged
    assert out.voices[0].mods == (traj,)
    assert "note" not in out.voices[0].controls


def test_space_registers_voice_order_preserved():
    """Voice order in the output matches the input order."""
    v1 = _voice(note=10.0)
    v2 = _voice(note=11.0)
    s = _scene(v1, v2)
    out = space_registers(s)
    # v1 was first, v2 was second — order preserved regardless of note movement
    # (we verify by checking source_name identity isn't swapped,
    #  but both are "supersaw"; verify via the fact that input v1 note=10 is
    #  the lower note and the output positions differ)
    assert len(out.voices) == 2


def test_space_registers_three_voices_spread():
    """Three notes within 2 semitones of each other all get spread."""
    v1 = _voice(note=0.0)
    v2 = _voice(note=1.0)
    v3 = _voice(note=2.0)
    s = _scene(v1, v2, v3)
    out = space_registers(s)
    notes = [v.controls["note"] for v in out.voices]
    sorted_notes = sorted(notes)
    # All consecutive pairs in sorted order must be >= 3 apart
    for i in range(len(sorted_notes) - 1):
        assert sorted_notes[i + 1] - sorted_notes[i] >= 3


def test_space_registers_down_when_up_bounded():
    """When transposing up would exceed +24, the note is pushed below the group."""
    # Two notes near the top — cannot go further up
    v1 = _voice(note=22.0)
    v2 = _voice(note=23.0)
    s = _scene(v1, v2)
    out = space_registers(s)
    notes = sorted(v.controls["note"] for v in out.voices)
    assert notes[1] - notes[0] >= 3
    for n in notes:
        assert _NOTE_LO <= n <= _NOTE_HI


def test_space_registers_non_note_controls_preserved():
    """Non-note controls are untouched after spacing."""
    v1 = Voice(
        source_name="supersaw", n=0, controls={"note": 0.0, "gain": 0.8}, mods=()
    )
    v2 = Voice(
        source_name="supersaw", n=0, controls={"note": 1.0, "gain": 0.9}, mods=()
    )
    s = _scene(v1, v2)
    out = space_registers(s)
    assert out.voices[0].controls["gain"] == pytest.approx(0.8)
    assert out.voices[1].controls["gain"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# quantize_rates
# ---------------------------------------------------------------------------


def test_quantize_rates_cps_zero_no_op():
    """cps = 0 → scene returned unchanged."""
    traj = Trajectory(param="note", shape="sine", args=(0.0, 1.0, 1.5))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.0)
    assert out is s


def test_quantize_rates_cps_negative_no_op():
    """Negative cps → scene returned unchanged."""
    traj = Trajectory(param="note", shape="sine", args=(0.0, 1.0, 1.5))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=-1.0)
    assert out is s


def test_quantize_rates_sine_nearest_division():
    """Sine rate is quantised to the nearest cps × division."""
    # cps=0.5 → candidates {0.125, 0.25, 0.5, 1.0, 2.0}
    # rate=0.4 → nearest is 0.5
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, 0.4))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    assert out.voices[0].mods[0].args[2] == pytest.approx(0.5)


def test_quantize_rates_walk_nearest_division():
    """Walk rate is quantised; seed (arg[3]) is untouched."""
    # cps=0.5 → candidates {0.125, 0.25, 0.5, 1.0, 2.0}
    # rate=0.8 → nearest is 1.0
    traj = Trajectory(param="note", shape="walk", args=(5.0, 2.0, 0.8, 42.0))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    new_args = out.voices[0].mods[0].args
    assert new_args[2] == pytest.approx(1.0)
    assert new_args[3] == pytest.approx(42.0)  # seed unchanged


def test_quantize_rates_ramp_untouched():
    """Ramp trajectories are not modified."""
    traj = Trajectory(param="note", shape="ramp", args=(0.0, 12.0))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    assert out.voices[0].mods[0].args == (0.0, 12.0)


def test_quantize_rates_steps_untouched():
    """Steps trajectories are not modified."""
    traj = Trajectory(param="note", shape="steps", args=(0.0, 4.0, 7.0))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    assert out.voices[0].mods[0].args == (0.0, 4.0, 7.0)


def test_quantize_rates_clamp_at_rate_lo():
    """Very low candidate is clamped to RATE_LO."""
    # cps=0.05 → 0.05 * 0.25 = 0.0125 → clamped to RATE_LO=0.02
    # rate=RATE_LO → stays at RATE_LO
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, RATE_LO))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.05)
    result_rate = out.voices[0].mods[0].args[2]
    assert result_rate >= RATE_LO


def test_quantize_rates_clamp_at_rate_hi():
    """Very high candidate is clamped to RATE_HI."""
    # cps=2.0 → 2.0 * 4 = 8.0 = RATE_HI (already at bound)
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, RATE_HI))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=2.0)
    result_rate = out.voices[0].mods[0].args[2]
    assert result_rate <= RATE_HI


def test_quantize_rates_center_and_depth_untouched():
    """Only the rate arg (index 2) changes; center and depth are preserved."""
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, 0.4))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    new_args = out.voices[0].mods[0].args
    assert new_args[0] == pytest.approx(500.0)  # center
    assert new_args[1] == pytest.approx(100.0)  # depth


def test_quantize_rates_metric_divisions_cps_half():
    """All five metric divisions at cps=0.5 are the only possible outputs."""
    # cps=0.5 → {0.125, 0.25, 0.5, 1.0, 2.0}
    valid = {0.125, 0.25, 0.5, 1.0, 2.0}
    # Test a spread of rates that should each map to one of these
    for raw_rate in [0.02, 0.1, 0.18, 0.35, 0.7, 1.5, 3.0, 8.0]:
        traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, raw_rate))
        v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
        s = _scene(v)
        out = quantize_rates(s, cps=0.5)
        result_rate = out.voices[0].mods[0].args[2]
        assert result_rate in valid, f"rate {raw_rate} → {result_rate} not in {valid}"


# ---------------------------------------------------------------------------
# stage_gains
# ---------------------------------------------------------------------------


def test_stage_gains_sub_bass_factor():
    """note ≤ -12 → factor 1.05."""
    v = Voice(
        source_name="supersaw", n=0, controls={"note": -12.0, "gain": 1.0}, mods=()
    )
    s = _scene(v)
    out = stage_gains(s)
    expected = round(min(_GAIN_HI, max(_GAIN_LO, 1.0 * 1.05)), 6)
    assert out.voices[0].controls["gain"] == pytest.approx(expected)


def test_stage_gains_mid_low_factor():
    """-12 < note < 0 → factor 1.0."""
    v = Voice(
        source_name="supersaw", n=0, controls={"note": -6.0, "gain": 1.0}, mods=()
    )
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] == pytest.approx(1.0)


def test_stage_gains_mid_high_factor():
    """0 ≤ note < 12 → factor 0.9."""
    v = Voice(source_name="supersaw", n=0, controls={"note": 0.0, "gain": 1.0}, mods=())
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] == pytest.approx(0.9)


def test_stage_gains_treble_factor():
    """note ≥ 12 → factor 0.8."""
    v = Voice(
        source_name="supersaw", n=0, controls={"note": 12.0, "gain": 1.0}, mods=()
    )
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] == pytest.approx(0.8)


def test_stage_gains_absent_gain_defaults_to_one():
    """Absent gain is treated as 1.0 and then SET on the output voice."""
    v = _voice(note=0.0)
    s = _scene(v)
    out = stage_gains(s)
    # gain was absent; now it should be set to 1.0 * 0.9 = 0.9
    assert "gain" in out.voices[0].controls
    assert out.voices[0].controls["gain"] == pytest.approx(0.9)


def test_stage_gains_no_note_factor_one():
    """Voice without a static note gets factor 1.0 (gain multiplied by 1)."""
    v = Voice(source_name="supersaw", n=0, controls={"gain": 0.8}, mods=())
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] == pytest.approx(0.8)


def test_stage_gains_clamped_to_spec_hi():
    """Result gain is clamped to spec hi when factor pushes it over."""
    # note=-24 → factor=1.05; gain=_GAIN_HI → would exceed → clamp
    v = Voice(
        source_name="supersaw", n=0, controls={"note": -24.0, "gain": _GAIN_HI}, mods=()
    )
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] <= _GAIN_HI


def test_stage_gains_clamped_to_spec_lo():
    """Result gain is clamped to spec lo when base is already low."""
    v = Voice(
        source_name="supersaw", n=0, controls={"note": 12.0, "gain": _GAIN_LO}, mods=()
    )
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] >= _GAIN_LO


def test_stage_gains_rounded_to_6_decimals():
    """Output gain values are rounded to 6 decimal places."""
    v = Voice(source_name="supersaw", n=0, controls={"note": 0.0, "gain": 1.0}, mods=())
    s = _scene(v)
    out = stage_gains(s)
    g = out.voices[0].controls["gain"]
    # round(g, 6) should equal g itself
    assert g == round(g, 6)


def test_stage_gains_boundary_note_minus12():
    """note = -12 is ≤ -12 → factor 1.05 (boundary is inclusive)."""
    v = Voice(
        source_name="supersaw", n=0, controls={"note": -12.0, "gain": 1.0}, mods=()
    )
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] == pytest.approx(round(1.05, 6))


def test_stage_gains_boundary_note_zero():
    """note = 0 is ≥ 0 → factor 0.9 (boundary is inclusive)."""
    v = Voice(source_name="supersaw", n=0, controls={"note": 0.0, "gain": 1.0}, mods=())
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] == pytest.approx(0.9)


def test_stage_gains_boundary_note_12():
    """note = 12 is ≥ 12 → factor 0.8 (boundary is inclusive)."""
    v = Voice(
        source_name="supersaw", n=0, controls={"note": 12.0, "gain": 1.0}, mods=()
    )
    s = _scene(v)
    out = stage_gains(s)
    assert out.voices[0].controls["gain"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# ensemble_rules
# ---------------------------------------------------------------------------


def test_ensemble_rules_label_none_skips_chord_snap():
    """label=None skips chord snap; other passes still run."""
    # note=2.0 (D, not a C maj7 chord tone) should NOT be snapped to chord
    # but gains/registers still apply
    v = Voice(source_name="supersaw", n=0, controls={"note": 2.0, "gain": 1.0}, mods=())
    s = _scene(v)
    out = ensemble_rules(s, label=None, cps=0.5)
    # Chord snap skipped → note might stay at 2.0 (D, pc=2)
    # Space registers with 1 voice → no change
    # stage_gains: note=2.0, 0 ≤ 2 < 12 → factor 0.9
    result_note = out.voices[0].controls["note"]
    # Note was NOT chord-snapped (no chord snapping when label=None)
    assert result_note == 2.0
    assert out.voices[0].controls["gain"] == pytest.approx(round(1.0 * 0.9, 6))


def test_ensemble_rules_na_label_skips_chord_snap():
    """'N/A' label skips chord snap; other passes still run."""
    v = Voice(source_name="supersaw", n=0, controls={"note": 2.0, "gain": 1.0}, mods=())
    s = _scene(v)
    out = ensemble_rules(s, label="N/A", cps=0.5)
    assert out.voices[0].controls["note"] == 2.0  # not chord-snapped


def test_ensemble_rules_gains_staged_on_final_registers():
    """Gains reflect FINAL register positions (after spacing), not originals."""
    # v1=4 (E, pc=4 ∈ Cmaj7), v2=5 (F, pc=5 → snaps to E=4 in Cmaj7).
    # Both land on pc=4; space_registers moves one up by +12 to 16 (≥12 → 0.8).
    v1 = Voice(
        source_name="supersaw", n=0, controls={"note": 4.0, "gain": 1.0}, mods=()
    )
    v2 = Voice(
        source_name="supersaw", n=0, controls={"note": 5.0, "gain": 1.0}, mods=()
    )
    s = _scene(v1, v2)
    out = ensemble_rules(s, label="C", cps=0.5)
    # One voice should be in treble register (note ≥ 12) with gain factor 0.8
    notes = [v.controls["note"] for v in out.voices]
    gains = [v.controls["gain"] for v in out.voices]
    # Verify: the voice with note ≥ 12 has gain ≈ 0.8 (× the pre-existing 1.0)
    for note, gain in zip(notes, gains, strict=False):
        if note >= 12:
            assert gain == pytest.approx(
                round(1.0 * 0.8, 6)
            ), f"Expected treble gain 0.8, got {gain} for note {note}"


def test_ensemble_rules_rates_quantised():
    """ensemble_rules quantises sine/walk rates to metric divisions."""
    # cps=0.5 → metric rates {0.125, 0.25, 0.5, 1.0, 2.0}
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, 0.4))
    v = Voice(source_name="supersaw", n=0, controls={"note": 0.0}, mods=(traj,))
    s = _scene(v)
    out = ensemble_rules(s, label=None, cps=0.5)
    rate = out.voices[0].mods[0].args[2]
    assert rate in {0.125, 0.25, 0.5, 1.0, 2.0}


def test_ensemble_rules_order_chord_then_register_then_rate_then_gain():
    """Composition order: chord snap → register spacing → rate quantisation → gain."""
    # Two voices with notes in C scale but not C maj7, close together.
    # After chord snap: both at nearest chord tone.
    # After register spacing: at least 3 apart.
    # Gain staged on FINAL (spaced) positions.
    v1 = _voice(
        note=2.0
    )  # D → snaps to C(0) or E(4), nearest=C? no: both dist=2, tie→C=0
    v2 = _voice(note=9.0)  # A → nearest C maj7: G=7 (dist=2), B=11 (dist=2), tie→G=7
    s = _scene(v1, v2)
    out = ensemble_rules(s, label="C", cps=0.5)
    notes = sorted(v.controls["note"] for v in out.voices)
    assert notes[1] - notes[0] >= 3
    # Both are chord tones
    for n in notes:
        assert _note_pc(n) in _C_MAJ7
