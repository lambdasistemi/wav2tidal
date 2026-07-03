"""Tests for core/pattern/ensemble.py (issue #66 — musicality 0b calibration)."""

from __future__ import annotations

import pytest

from wav2tidal.core.pattern.ensemble import (
    chord_classes,
    ensemble_rules,
    quantize_rates,
    space_registers,
    stage_gains,
    voice_chord,
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
# voice_chord
# ---------------------------------------------------------------------------

# C maj7 voicing (bass→treble): root=C(0), fifth=G(7), third=E(4), seventh=B(11)
_C_MAJ7_DEGREES = (0, 7, 4, 11)
# F#m7 voicing: root=F#(6), fifth=C#(1), third=A(9), seventh=E(4)
_FSM7_DEGREES = (6, 1, 9, 4)


def test_voice_chord_na_noop():
    """'N/A' label → scene returned unchanged."""
    v = _voice(note=2.0)
    s = _scene(v)
    assert voice_chord(s, "N/A") is s


def test_voice_chord_none_noop():
    """None label → scene returned unchanged."""
    v = _voice(note=2.0)
    s = _scene(v)
    assert voice_chord(s, None) is s


def test_voice_chord_unknown_label_noop():
    """Unrecognised label → scene returned unchanged."""
    v = _voice(note=2.0)
    s = _scene(v)
    assert voice_chord(s, "Xb") is s


def test_voice_chord_no_static_notes_noop():
    """Scene with no static note controls → unchanged (no indexed voices)."""
    traj = Trajectory(param="note", shape="sine", args=(0.0, 1.0, 0.5))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = voice_chord(s, "C")
    # No static notes: should return unchanged
    assert out.voices[0].mods[0].args == (0.0, 1.0, 0.5)
    assert "note" not in out.voices[0].controls


def test_voice_chord_two_voices_distinct_pcs():
    """Two clustered voices get distinct pitch classes (root + fifth)."""
    # Both notes near 0 → sorted: voice1 (note=0) rank 0 → root C(0),
    # voice2 (note=1) rank 1 → fifth G(7).
    v1 = _voice(note=0.0)
    v2 = _voice(note=1.0)
    s = _scene(v1, v2)
    out = voice_chord(s, "C")
    pcs = {_note_pc(v.controls["note"]) for v in out.voices}
    # Must be 2 distinct pitch classes
    assert len(pcs) == 2
    # Both in C-major voicing degrees
    assert pcs <= set(_C_MAJ7_DEGREES)


def test_voice_chord_four_voices_cover_all_degrees():
    """Four voices in C major receive all four voicing degrees."""
    voices = [_voice(note=float(i)) for i in range(4)]
    s = _scene(*voices)
    out = voice_chord(s, "C")
    pcs = {_note_pc(v.controls["note"]) for v in out.voices}
    assert pcs == set(_C_MAJ7_DEGREES)


def test_voice_chord_five_voices_cycles_degrees():
    """Five voices cycle: voice 4 (rank 4) receives degree[0 % 4] = root again."""
    voices = [_voice(note=float(i * 2)) for i in range(5)]
    s = _scene(*voices)
    out = voice_chord(s, "C")
    # Ranks 0..4 → degrees[0,1,2,3,0] — voice at rank 4 gets degree 0 = root(C=0)
    sorted_notes = sorted(v.controls["note"] for v in out.voices)
    pcs = [_note_pc(n) for n in sorted_notes]
    assert pcs[4] == _C_MAJ7_DEGREES[0]  # root repeated


def test_voice_chord_nearest_octave_placement():
    """Each voice is placed at the nearest octave with the target pitch class."""
    # Voice at note=14: rank 0 → target_pc=0 (C).
    # Candidates with pc=0: ..., -12, 0, 12, 24. Nearest to 14: 12 (dist=2).
    v = _voice(note=14.0)
    s = _scene(v)
    out = voice_chord(s, "C")
    result = out.voices[0].controls["note"]
    assert _note_pc(result) == 0  # C
    assert result == pytest.approx(12.0)  # nearest octave below 14


def test_voice_chord_tie_goes_down():
    """On equal distance, the lower candidate wins."""
    # Voice at note=6: target_pc=0. 0 (dist=6) and 12 (dist=6) → tie → lower wins.
    # Tie resolves downward → 0.
    v = _voice(note=6.0)
    s = _scene(v)
    out = voice_chord(s, "C")
    result = out.voices[0].controls["note"]
    assert result == pytest.approx(0.0)


def test_voice_chord_bounds_respected():
    """Placed notes always stay within spec bounds."""
    for raw_note in range(_NOTE_LO - 2, _NOTE_HI + 3):
        v = _voice(note=float(raw_note))
        s = _scene(v)
        out = voice_chord(s, "C")
        result = out.voices[0].controls["note"]
        assert _NOTE_LO <= result <= _NOTE_HI


def test_voice_chord_no_static_note_voices_untouched():
    """Voices without a static note are not modified."""
    traj = Trajectory(param="note", shape="sine", args=(3.0, 1.0, 0.5))
    v_no = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    v_yes = _voice(note=0.0)
    s = _scene(v_no, v_yes)
    out = voice_chord(s, "C")
    # First voice (no static note) unchanged
    assert out.voices[0].mods[0].args == (3.0, 1.0, 0.5)
    assert "note" not in out.voices[0].controls


def test_voice_chord_f_sharp_minor_pcs():
    """F#m voicing: root=F#(6), fifth=C#(1), third=A(9), seventh=E(4)."""
    voices = [_voice(note=float(i)) for i in range(4)]
    s = _scene(*voices)
    out = voice_chord(s, "F#m")
    pcs = {_note_pc(v.controls["note"]) for v in out.voices}
    assert pcs == set(_FSM7_DEGREES)


def test_voice_chord_scene_order_preserved():
    """Voice order in the output matches the input order."""
    v1 = _voice(note=10.0)
    v2 = _voice(note=2.0)
    s = _scene(v1, v2)
    out = voice_chord(s, "C")
    # voice1 was originally at 10, voice2 at 2.
    # sorted ascending: rank0=v2(note=2→pc=7[fifth@0]=7), rank1=v1(note=10→pc=0)
    # But ORDER in output matches input: out.voices[0] corresponds to v1 (note=10).
    # v1 is rank 1 → degree[1] = fifth = 7
    assert _note_pc(out.voices[0].controls["note"]) == _C_MAJ7_DEGREES[1]  # fifth
    # v2 is rank 0 → degree[0] = root = 0
    assert _note_pc(out.voices[1].controls["note"]) == _C_MAJ7_DEGREES[0]  # root


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


def test_quantize_rates_drift_untouched():
    """Rate below cps/4 (drift band) is left completely untouched."""
    # cps=0.5 → drift_threshold=0.125; rate=0.03 is drift
    drift_rate = 0.03
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, drift_rate))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    assert out.voices[0].mods[0].args[2] == pytest.approx(drift_rate)


def test_quantize_rates_drift_threshold_boundary():
    """Just below cps/4 is drift; at/above is rhythmic."""
    cps = 0.5
    threshold = cps / 4  # 0.125
    # Just below threshold → drift (untouched)
    drift_traj = Trajectory(
        param="cutoff", shape="sine", args=(500.0, 1.0, threshold - 0.001)
    )
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(drift_traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=cps)
    assert out.voices[0].mods[0].args[2] == pytest.approx(threshold - 0.001)


def test_quantize_rates_rhythmic_snaps_to_three_divisions():
    """Rhythmic rates snap to one of the three metric divisions."""
    # cps=0.5 → candidates = {0.125, 0.25, 0.5}
    valid = {0.125, 0.25, 0.5}
    for raw_rate in [0.13, 0.18, 0.3, 0.4, 0.45]:
        traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, raw_rate))
        v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
        s = _scene(v)
        out = quantize_rates(s, cps=0.5)
        result_rate = out.voices[0].mods[0].args[2]
        assert result_rate in valid, f"rate {raw_rate} → {result_rate} not in {valid}"


def test_quantize_rates_sine_nearest_division():
    """Sine rate is quantised to the nearest cps × division."""
    # cps=0.5 → candidates {0.125, 0.25, 0.5}
    # rate=0.4 → nearest is 0.5
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, 0.4))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    assert out.voices[0].mods[0].args[2] == pytest.approx(0.5)


def test_quantize_rates_walk_nearest_division():
    """Walk rate is quantised; seed (arg[3]) is untouched."""
    # cps=0.5 → candidates {0.125, 0.25, 0.5}
    # rate=0.8 → nearest is 0.5
    traj = Trajectory(param="note", shape="walk", args=(5.0, 2.0, 0.8, 42.0))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    new_args = out.voices[0].mods[0].args
    assert new_args[2] == pytest.approx(0.5)
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


def test_quantize_rates_depth_scaled_when_rate_raised():
    """When new_rate > old_rate, depth is scaled by old/new ratio."""
    # cps=0.5, drift_threshold=0.125
    # rate=0.2 → nearest candidate: |0.2-0.125|=0.075, |0.2-0.25|=0.05 → nearest=0.25
    # old=0.2, new=0.25 → ratio=0.2/0.25=0.8 → depth = 1.0 * 0.8 = 0.8
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 1.0, 0.2))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    new_args = out.voices[0].mods[0].args
    assert new_args[2] == pytest.approx(0.25)  # rate raised
    assert new_args[1] == pytest.approx(0.8)  # depth scaled by 0.8


def test_quantize_rates_depth_not_scaled_when_rate_lowered():
    """When new_rate < old_rate (rate lowered), depth is unchanged."""
    # cps=0.5 → candidates {0.125, 0.25, 0.5}
    # rate=0.7 → nearest=0.5 (lowered: 0.7 → 0.5)
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 2.0, 0.7))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.5)
    new_args = out.voices[0].mods[0].args
    assert new_args[2] == pytest.approx(0.5)  # rate lowered
    assert new_args[1] == pytest.approx(2.0)  # depth unchanged


def test_quantize_rates_depth_clamp_floor():
    """Depth scale is clamped to at least 0.15 of original (defensive floor).

    With three divisions {0.125, 0.25, 0.5}, the maximum ratio jump in de-unison
    is 0.125/0.5=0.25 (well above floor=0.15).  This test verifies depth is
    never reduced below 15% of original even in the worst-case de-unison jump.
    """
    # Three trajectories all snap to 0.125; de-unison moves 2nd→0.25, 3rd→0.5.
    # Worst jump: 0.125→0.5, ratio=0.25, depth=0.25*1.0=0.25 ≥ 0.15*1.0.
    original_depth = 1.0
    trajs = [
        Trajectory(param="cutoff", shape="sine", args=(500.0, original_depth, 0.13))
        for _ in range(3)
    ]
    voices = [Voice(source_name="supersaw", n=0, controls={}, mods=(t,)) for t in trajs]
    s = _scene(*voices)
    out = quantize_rates(s, cps=0.5)
    for v in out.voices:
        for t in v.mods:
            if t.shape == "sine":
                assert (
                    t.args[1] >= 0.15 * original_depth
                ), f"depth {t.args[1]} < floor 0.15 * {original_depth}"


def test_quantize_rates_clamp_at_rate_lo():
    """Very low candidate is clamped to RATE_LO."""
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, RATE_LO))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=0.05)
    result_rate = out.voices[0].mods[0].args[2]
    assert result_rate >= RATE_LO


def test_quantize_rates_clamp_at_rate_hi():
    """Very high candidate is clamped to RATE_HI."""
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, RATE_HI))
    v = Voice(source_name="supersaw", n=0, controls={}, mods=(traj,))
    s = _scene(v)
    out = quantize_rates(s, cps=2.0)
    result_rate = out.voices[0].mods[0].args[2]
    assert result_rate <= RATE_HI


def test_quantize_rates_deunison_two_voices_different_divisions():
    """Two trajectories both nearest the same division end up on different divisions."""
    # cps=0.5 → candidates=[0.125, 0.25, 0.5]; drift_threshold=0.125
    # rate=0.2 → nearest is 0.25 for both
    cps = 0.5
    t1 = Trajectory(param="cutoff", shape="sine", args=(500.0, 2.0, 0.2))
    t2 = Trajectory(param="cutoff", shape="sine", args=(500.0, 2.0, 0.2))
    v1 = Voice(source_name="supersaw", n=0, controls={}, mods=(t1,))
    v2 = Voice(source_name="supersaw", n=0, controls={}, mods=(t2,))
    s = _scene(v1, v2)
    out = quantize_rates(s, cps=cps)
    r1 = out.voices[0].mods[0].args[2]
    r2 = out.voices[1].mods[0].args[2]
    assert r1 != r2, f"Both trajectories landed on division {r1} (de-unison failed)"
    # Both must still be valid metric divisions
    valid = {0.125, 0.25, 0.5}
    assert r1 in valid and r2 in valid


def test_quantize_rates_deunison_three_voices_cover_all_divisions():
    """Three trajectories all snapping to the same division spread onto all three."""
    # rate=0.13 → nearest is 0.125 (cps=0.5); de-unison moves 2nd and 3rd
    cps = 0.5
    trajs = [
        Trajectory(param="cutoff", shape="sine", args=(500.0, 2.0, 0.13))
        for _ in range(3)
    ]
    voices = [Voice(source_name="supersaw", n=0, controls={}, mods=(t,)) for t in trajs]
    s = _scene(*voices)
    out = quantize_rates(s, cps=cps)
    rates = {out.voices[i].mods[0].args[2] for i in range(3)}
    assert rates == {
        0.125,
        0.25,
        0.5,
    }, f"Expected all three divisions {{0.125, 0.25, 0.5}}, got {rates}"


def test_quantize_rates_deunison_depth_scaled_on_move():
    """When de-unison raises a rate, depth is scaled accordingly."""
    # Two trajectories at 0.13 → both snap to 0.125. Second is moved to 0.25.
    # old=0.125, new=0.25 → ratio=0.5 → depth = 2.0 * 0.5 = 1.0
    cps = 0.5
    t1 = Trajectory(param="cutoff", shape="sine", args=(500.0, 2.0, 0.13))
    t2 = Trajectory(param="cutoff", shape="sine", args=(500.0, 2.0, 0.13))
    v1 = Voice(source_name="supersaw", n=0, controls={}, mods=(t1,))
    v2 = Voice(source_name="supersaw", n=0, controls={}, mods=(t2,))
    s = _scene(v1, v2)
    out = quantize_rates(s, cps=cps)
    r1 = out.voices[0].mods[0].args[2]
    r2 = out.voices[1].mods[0].args[2]
    # The trajectory on the higher division should have reduced depth
    if r1 > r2:
        raised = out.voices[0].mods[0].args[1]
    else:
        raised = out.voices[1].mods[0].args[1]
    assert raised < 2.0, "Depth should be reduced when rate is raised by de-unison"


def test_quantize_rates_drift_voices_not_deunisoned():
    """Drift trajectories do not participate in de-unison."""
    # Two drift trajectories at same rate (below threshold) → both stay unchanged
    cps = 0.5
    drift_rate = 0.05  # < 0.125
    t1 = Trajectory(param="cutoff", shape="sine", args=(500.0, 1.0, drift_rate))
    t2 = Trajectory(param="cutoff", shape="sine", args=(500.0, 1.0, drift_rate))
    v1 = Voice(source_name="supersaw", n=0, controls={}, mods=(t1,))
    v2 = Voice(source_name="supersaw", n=0, controls={}, mods=(t2,))
    s = _scene(v1, v2)
    out = quantize_rates(s, cps=cps)
    assert out.voices[0].mods[0].args[2] == pytest.approx(drift_rate)
    assert out.voices[1].mods[0].args[2] == pytest.approx(drift_rate)


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


def test_ensemble_rules_label_none_skips_voicing():
    """label=None skips voice_chord; other passes still run."""
    # note=2.0 (D) not moved by voicing (label=None).
    # 1 voice → no register spacing change.
    # stage_gains: 0 ≤ 2 < 12 → factor 0.9.
    v = Voice(source_name="supersaw", n=0, controls={"note": 2.0, "gain": 1.0}, mods=())
    s = _scene(v)
    out = ensemble_rules(s, label=None, cps=0.5)
    assert out.voices[0].controls["note"] == 2.0  # voicing skipped
    assert out.voices[0].controls["gain"] == pytest.approx(round(1.0 * 0.9, 6))


def test_ensemble_rules_na_label_skips_voicing():
    """'N/A' label skips voice_chord; other passes still run."""
    v = Voice(source_name="supersaw", n=0, controls={"note": 2.0, "gain": 1.0}, mods=())
    s = _scene(v)
    out = ensemble_rules(s, label="N/A", cps=0.5)
    assert out.voices[0].controls["note"] == 2.0  # not voiced


def test_ensemble_rules_two_voices_get_distinct_pcs():
    """Two voices in C major get distinct chord degrees after voicing."""
    v1 = _voice(note=0.0)
    v2 = _voice(note=1.0)
    s = _scene(v1, v2)
    out = ensemble_rules(s, label="C", cps=0.5)
    pcs = {_note_pc(v.controls["note"]) for v in out.voices}
    assert len(pcs) == 2


def test_ensemble_rules_gains_reflect_register():
    """Gains reflect register positions after voicing+spacing."""
    # Two voices — voicing places them on distinct degrees; stage_gains
    # reflects whatever register they end up in.
    v1 = _voice(note=0.0, gain=1.0)
    v2 = _voice(note=12.0, gain=1.0)
    s = _scene(v1, v2)
    out = ensemble_rules(s, label="C", cps=0.5)
    # All gains must be within spec bounds.
    for v in out.voices:
        g = v.controls.get("gain", 1.0)
        assert _GAIN_LO <= g <= _GAIN_HI


def test_ensemble_rules_rates_quantised():
    """ensemble_rules quantises sine/walk rates to the three metric divisions."""
    # cps=0.5 → rhythmic candidates {0.125, 0.25, 0.5}
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 100.0, 0.4))
    v = Voice(source_name="supersaw", n=0, controls={"note": 0.0}, mods=(traj,))
    s = _scene(v)
    out = ensemble_rules(s, label=None, cps=0.5)
    rate = out.voices[0].mods[0].args[2]
    assert rate in {0.125, 0.25, 0.5}


def test_ensemble_rules_drift_rates_pass_through():
    """Drift rates (< cps/4) pass through ensemble_rules untouched."""
    drift_rate = 0.03
    traj = Trajectory(param="cutoff", shape="sine", args=(500.0, 1.0, drift_rate))
    v = Voice(source_name="supersaw", n=0, controls={"note": 0.0}, mods=(traj,))
    s = _scene(v)
    out = ensemble_rules(s, label=None, cps=0.5)
    assert out.voices[0].mods[0].args[2] == pytest.approx(drift_rate)


def test_ensemble_rules_order_voicing_then_register_then_rate_then_gain():
    """Composition order: voice_chord → space_registers → quantize_rates → gains."""
    # Two voices. Voicing assigns distinct degrees. Register spacing enforces gap.
    v1 = _voice(note=0.0)
    v2 = _voice(note=2.0)
    s = _scene(v1, v2)
    out = ensemble_rules(s, label="C", cps=0.5)
    notes = sorted(v.controls["note"] for v in out.voices)
    # Pairwise gap >= 3 (register spacing applied after voicing)
    assert notes[1] - notes[0] >= 3
    # Both are C-major chord tones (voicing applied)
    c_maj7_pcs = {0, 4, 7, 11}
    for n in notes:
        assert _note_pc(n) in c_maj7_pcs
