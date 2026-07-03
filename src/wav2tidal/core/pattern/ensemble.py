"""Intrinsic ensemble rules (issue #64 — musicality 0).

Axis-1 principle: output must be music by its own rules; the seed only
steers where.  Key-locking (#58) put every voice in one scale; these passes
give the simultaneous voices an internal grammar:

1. **Chord snap** — static drone notes land on tonic-seventh chord tones
   (maj7 / min7 degrees), not just any scale tone.  Trajectory notes keep
   the full scale (melodic freedom).
2. **Register spacing** — voices with static notes are spread by
   octave-transposition (pitch-class preserving) to a pairwise minimum gap.
3. **Metric rates** — sine/walk trajectory rates quantise to musical
   divisions of the TidalCycles cycle (cps × {1/4, 1/2, 1, 2, 4}).
4. **Gain staging** — low voices carry, high voices sparkle.

Pure — no IO.  Apply via ``ensemble_rules`` after ``snap_scene``.
"""

from __future__ import annotations

from wav2tidal.core.pattern.key import PITCH_NAMES, snap_note
from wav2tidal.core.pattern.model import Scene, Trajectory, Voice
from wav2tidal.core.pattern.params import spec
from wav2tidal.core.pattern.shapes import RATE_HI, RATE_LO

_NOTE_LO: int = int(spec("note").lo)  # -24
_NOTE_HI: int = int(spec("note").hi)  # +24

# Seventh-chord intervals in semitones above the tonic.
_MAJ7_INTERVALS: frozenset[int] = frozenset({0, 4, 7, 11})
_MIN7_INTERVALS: frozenset[int] = frozenset({0, 3, 7, 10})

# Metric division multipliers for rate quantisation.
_METRIC_MULTS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


# ---------------------------------------------------------------------------
# chord_classes
# ---------------------------------------------------------------------------


def chord_classes(label: str) -> frozenset[int] | None:
    """Return tonic-seventh chord pitch classes (0=C … 11=B) for ``label``.

    ``label`` is a key name as produced by ``estimate_key`` in features.py,
    e.g. ``"F#"`` for F# major (→ maj7) or ``"F#m"`` for F# minor (→ min7).
    ``"N/A"`` or any unrecognisable string yields ``None``.

    Major label → maj7 degrees {0, 4, 7, 11} rotated by the tonic.
    Minor label (trailing "m") → min7 degrees {0, 3, 7, 10} rotated by tonic.
    """
    if not label or label == "N/A":
        return None
    if label.endswith("m"):
        name, intervals = label[:-1], _MIN7_INTERVALS
    else:
        name, intervals = label, _MAJ7_INTERVALS
    if name not in PITCH_NAMES:
        return None
    tonic = PITCH_NAMES.index(name)
    return frozenset((tonic + iv) % 12 for iv in intervals)


# ---------------------------------------------------------------------------
# snap_static_notes_to_chord
# ---------------------------------------------------------------------------


def snap_static_notes_to_chord(scene: Scene, label: str | None) -> Scene:
    """Snap every voice's static ``"note"`` control to the nearest chord tone.

    Trajectory notes are **not** touched — they keep the scale from
    ``snap_scene`` (melodic freedom).  Returns ``scene`` unchanged when
    ``label`` yields no chord classes (``None`` / ``"N/A"`` / unknown).

    Reuses ``snap_note`` from ``key`` with the chord pitch-class set rather
    than the scale pitch-class set.  Pure — returns a new ``Scene``.
    """
    if label is None:
        return scene
    classes = chord_classes(label)
    if classes is None:
        return scene

    new_voices: list[Voice] = []
    for v in scene.voices:
        new_controls: dict = dict(v.controls)
        if "note" in new_controls and isinstance(new_controls["note"], (int, float)):
            new_controls["note"] = snap_note(float(new_controls["note"]), classes)
        new_voices.append(
            Voice(
                source_name=v.source_name,
                n=v.n,
                controls=new_controls,
                mods=v.mods,
            )
        )
    return Scene(voices=tuple(new_voices), layer=scene.layer, source=scene.source)


# ---------------------------------------------------------------------------
# space_registers
# ---------------------------------------------------------------------------


def space_registers(scene: Scene, min_gap: int = 3) -> Scene:
    """Spread static-note voices by octave transposition to enforce minimum gap.

    Algorithm (deterministic, one greedy pass):

    1. Collect voices that carry a static numeric ``"note"`` control.
    2. Sort them by note value (ascending).
    3. Walk from lowest to highest.  When the current note is within
       ``min_gap`` semitones of the most-recently placed note:

       a. **Try UP** — add 12 repeatedly until the gap holds or the note
          exceeds +24.
       b. **Try DOWN** (only if UP was bounded) — subtract 12 from the
          original note until the new value is at least ``min_gap`` below the
          global minimum of all notes placed so far and ≥ −24.
       c. If neither works, leave the note unchanged.

    Octave transpositions preserve pitch class, so chord / scale membership
    (established by earlier passes) is never disturbed.  Voice ORDER in the
    scene is preserved; only note values change.  Spec bounds [−24, +24] are
    never violated.
    """
    # Gather (scene_index, note) for voices with a static numeric note.
    indexed: list[tuple[int, float]] = []
    for i, v in enumerate(scene.voices):
        note_val = v.controls.get("note")
        if isinstance(note_val, (int, float)) and not isinstance(note_val, bool):
            indexed.append((i, float(note_val)))

    if len(indexed) <= 1:
        return scene  # nothing to spread

    # Sort ascending by note to process from lowest to highest.
    indexed_sorted = sorted(indexed, key=lambda x: x[1])

    placed: dict[int, float] = {}  # scene_index → final note
    prev_note: float | None = None
    prev_lowest: float | None = None

    for orig_idx, note in indexed_sorted:
        if prev_note is None:
            placed[orig_idx] = note
            prev_note = note
            prev_lowest = note
            continue

        gap = note - prev_note
        if gap >= min_gap:
            placed[orig_idx] = note
            prev_note = note
            # prev_lowest stays (we're ascending, note >= prev_lowest)
            continue

        # Try transposing UP.
        found = False
        candidate = note + 12
        while candidate <= _NOTE_HI:
            if candidate - prev_note >= min_gap:
                placed[orig_idx] = candidate
                prev_note = candidate
                found = True
                break
            candidate += 12

        if not found:
            # Try placing DOWN, below the global minimum placed so far.
            assert prev_lowest is not None
            candidate = note - 12
            while candidate >= _NOTE_LO:
                if prev_lowest - candidate >= min_gap:
                    placed[orig_idx] = candidate
                    prev_note = candidate
                    prev_lowest = candidate
                    found = True
                    break
                candidate -= 12

        if not found:
            # Leave unchanged (bounds prevent any valid octave step).
            placed[orig_idx] = note
            prev_note = note

    # Rebuild scene, updating only the note values that changed.
    new_voices: list[Voice] = []
    for i, v in enumerate(scene.voices):
        if i in placed:
            new_note = placed[i]
            new_controls = {**v.controls, "note": new_note}
            new_voices.append(
                Voice(
                    source_name=v.source_name,
                    n=v.n,
                    controls=new_controls,
                    mods=v.mods,
                )
            )
        else:
            new_voices.append(v)

    return Scene(voices=tuple(new_voices), layer=scene.layer, source=scene.source)


# ---------------------------------------------------------------------------
# quantize_rates
# ---------------------------------------------------------------------------


def quantize_rates(scene: Scene, cps: float) -> Scene:
    """Quantise sine/walk trajectory rates to metric divisions of the cycle.

    Candidate rates: ``cps × {0.25, 0.5, 1, 2, 4}``, clamped into
    ``[RATE_LO, RATE_HI]`` (shapes.py bounds).  For each ``sine`` or ``walk``
    trajectory the existing rate arg (index 2) is replaced by the nearest
    valid metric rate (ties go to the lower candidate).

    ``cps ≤ 0`` is a guard: returns ``scene`` unchanged.
    ``ramp`` and ``steps`` trajectories are not touched.
    The ``walk`` seed (arg index 3) is never modified.
    """
    if cps <= 0:
        return scene

    # Precompute clamped metric candidates (sorted ascending).
    candidates: list[float] = sorted(
        {max(RATE_LO, min(RATE_HI, cps * m)) for m in _METRIC_MULTS}
    )

    def _nearest(rate: float) -> float:
        return min(candidates, key=lambda c: (abs(c - rate), c))

    def _quantize_traj(traj: Trajectory) -> Trajectory:
        if traj.shape not in ("sine", "walk") or len(traj.args) < 3:
            return traj
        new_args = traj.args[:2] + (_nearest(traj.args[2]),) + traj.args[3:]
        return Trajectory(param=traj.param, shape=traj.shape, args=new_args)

    new_voices: list[Voice] = []
    for v in scene.voices:
        new_mods = tuple(_quantize_traj(t) for t in v.mods)
        new_voices.append(
            Voice(
                source_name=v.source_name,
                n=v.n,
                controls=v.controls,
                mods=new_mods,
            )
        )
    return Scene(voices=tuple(new_voices), layer=scene.layer, source=scene.source)


# ---------------------------------------------------------------------------
# stage_gains
# ---------------------------------------------------------------------------


def _register_factor(note: float) -> float:
    """Map a static note value to a gain multiplier by register."""
    if note <= -12:
        return 1.05
    if note < 0:
        return 1.0
    if note < 12:
        return 0.9
    return 0.8


def stage_gains(scene: Scene) -> Scene:
    """Multiply each voice's gain by a register factor derived from its note.

    Register factors (applied to the static gain; default 1.0 when absent):

    * note ≤ −12   → 1.05  (sub-bass / bass: carry)
    * −12 < note < 0 → 1.0 (mid-low: neutral)
    * 0 ≤ note < 12  → 0.9 (mid-high: pull back)
    * note ≥ 12    → 0.8  (treble: sparkle, not blare)

    Voices without a static numeric note use factor 1.0 (no register
    context).  The result is clamped to ``spec("gain")`` bounds and rounded
    to 6 decimal places (matching ``apply_energy``'s convention).  Pure.
    """
    gain_spec = spec("gain")
    g_lo, g_hi = gain_spec.lo, gain_spec.hi

    new_voices: list[Voice] = []
    for v in scene.voices:
        base_gain = float(v.controls.get("gain", 1.0))
        note_val = v.controls.get("note")
        if isinstance(note_val, (int, float)) and not isinstance(note_val, bool):
            factor = _register_factor(float(note_val))
        else:
            factor = 1.0
        new_gain = round(min(g_hi, max(g_lo, base_gain * factor)), 6)
        new_controls = {**v.controls, "gain": new_gain}
        new_voices.append(
            Voice(
                source_name=v.source_name,
                n=v.n,
                controls=new_controls,
                mods=v.mods,
            )
        )
    return Scene(voices=tuple(new_voices), layer=scene.layer, source=scene.source)


# ---------------------------------------------------------------------------
# ensemble_rules
# ---------------------------------------------------------------------------


def ensemble_rules(scene: Scene, label: str | None, cps: float) -> Scene:
    """Apply all four intrinsic ensemble passes in order.

    Passes (applied left to right):

    1. ``snap_static_notes_to_chord`` — static notes → tonic-seventh chord
       tones.  Skipped when ``label`` is ``None`` or unknown; the remaining
       passes still run.
    2. ``space_registers`` — octave-spread voices with static notes to a
       pairwise minimum gap of 3 semitones.
    3. ``quantize_rates`` — sine/walk rates → nearest metric division.
    4. ``stage_gains`` — multiply gains by register factor.

    Pure.  Returns a new ``Scene``; ``scene`` is not mutated.
    """
    s = snap_static_notes_to_chord(scene, label)
    s = space_registers(s)
    s = quantize_rates(s, cps)
    s = stage_gains(s)
    return s
