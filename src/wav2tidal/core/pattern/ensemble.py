"""Intrinsic ensemble rules (issue #64 — musicality 0).

Axis-1 principle: output must be music by its own rules; the seed only
steers where.  Key-locking (#58) put every voice in one scale; these passes
give the simultaneous voices an internal grammar:

1. **Chord voicing** — static drone notes are assigned distinct chord degrees
   (root, fifth, third, seventh) sorted bass-up, each placed at the octave
   nearest its original note.  Guarantees ≥ 2 distinct pitch classes whenever
   ≥ 2 static-note voices exist.
2. **Register spacing** — voices with static notes are spread by
   octave-transposition (pitch-class preserving) to a pairwise minimum gap.
3. **Metric rates** — sine/walk trajectory rates quantise to musical
   divisions of the TidalCycles cycle (cps × {1/4, 1/2, 1}).  Rates below
   cps/4 are drift and are left untouched.  Depth scales with rate when a
   rate is raised (depth/rate trade).  De-unison assigns distinct divisions.
4. **Gain staging** — low voices carry, high voices sparkle.

Pure — no IO.  Apply via ``ensemble_rules`` after ``snap_scene``.
"""

from __future__ import annotations

import math

from wav2tidal.core.pattern.key import PITCH_NAMES
from wav2tidal.core.pattern.model import Scene, Trajectory, Voice
from wav2tidal.core.pattern.params import spec
from wav2tidal.core.pattern.shapes import RATE_HI, RATE_LO

_NOTE_LO: int = int(spec("note").lo)  # -24
_NOTE_HI: int = int(spec("note").hi)  # +24

# Seventh-chord intervals in semitones above the tonic.
_MAJ7_INTERVALS: frozenset[int] = frozenset({0, 4, 7, 11})
_MIN7_INTERVALS: frozenset[int] = frozenset({0, 3, 7, 10})

# Voicing order (bass up): root, fifth, third, seventh — semitones above tonic.
_MAJ_VOICING: tuple[int, ...] = (0, 7, 4, 11)
_MIN_VOICING: tuple[int, ...] = (0, 7, 3, 10)

# Metric division multipliers (reduced set — drift below cps/4 is excluded).
_METRIC_MULTS: tuple[float, ...] = (0.25, 0.5, 1.0)

# Local 3-significant-figure rounding (same convention as shapes._round3).
_round3 = lambda v: float(f"{v:.3g}")  # noqa: E731


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
# voice_chord
# ---------------------------------------------------------------------------


def _nearest_in_pc(orig: float, target_pc: int) -> float:
    """Return the note nearest to *orig* that has pitch class *target_pc*.

    Searches all integers n in [_NOTE_LO, _NOTE_HI] with n % 12 == target_pc;
    ties resolve downward (lower value wins).
    """
    k_min = math.ceil((_NOTE_LO - target_pc) / 12)
    k_max = math.floor((_NOTE_HI - target_pc) / 12)
    candidates = [target_pc + 12 * k for k in range(k_min, k_max + 1)]
    if not candidates:
        return orig
    return float(min(candidates, key=lambda c: (abs(c - orig), c)))


def voice_chord(scene: Scene, label: str | None) -> Scene:
    """Assign distinct chord degrees to static-note voices (voicing pass).

    Voicing order (bass to treble): root, fifth, third, seventh — voice i
    (sorted ascending by current note) receives degree ``i % 4``.  Each
    voice's new note is placed at the octave nearest its original note while
    keeping the target pitch class.

    Degrees for C major: [C, G, E, B] (pcs 0, 7, 4, 11).
    Degrees for C minor: [C, G, Eb, Bb] (pcs 0, 7, 3, 10).

    Voices without a static numeric ``"note"`` control are untouched.  Scene
    order is preserved.  Returns *scene* unchanged when *label* is ``None``
    or unrecognisable.  Pure.
    """
    if label is None:
        return scene
    if label.endswith("m"):
        name, voicing = label[:-1], _MIN_VOICING
    else:
        name, voicing = label, _MAJ_VOICING
    if name not in PITCH_NAMES:
        return scene
    tonic = PITCH_NAMES.index(name)
    degrees = tuple((tonic + iv) % 12 for iv in voicing)

    # Collect (scene_index, note) for voices with a static numeric note.
    indexed: list[tuple[int, float]] = []
    for i, v in enumerate(scene.voices):
        note_val = v.controls.get("note")
        if isinstance(note_val, (int, float)) and not isinstance(note_val, bool):
            indexed.append((i, float(note_val)))

    if not indexed:
        return scene

    # Sort ascending by note (bass-first voicing assignment).
    indexed_sorted = sorted(indexed, key=lambda x: x[1])

    new_notes: dict[int, float] = {}
    for rank, (orig_idx, orig_note) in enumerate(indexed_sorted):
        target_pc = degrees[rank % 4]
        new_notes[orig_idx] = _nearest_in_pc(orig_note, target_pc)

    # Rebuild scene preserving order.
    new_voices: list[Voice] = []
    for i, v in enumerate(scene.voices):
        if i in new_notes:
            new_voices.append(
                Voice(
                    source_name=v.source_name,
                    n=v.n,
                    controls={**v.controls, "note": new_notes[i]},
                    mods=v.mods,
                )
            )
        else:
            new_voices.append(v)
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


def _with_depth_trade(traj: Trajectory, old_rate: float, new_rate: float) -> Trajectory:
    """Scale depth when *new_rate* > *old_rate* (depth/rate trade).

    Multiplier = old_rate / new_rate, clamped to [0.15, 1.0].  Depth is
    rounded to 3 significant figures (same convention as shapes._round3).
    Returns *traj* unchanged when new_rate ≤ old_rate (no increase).
    """
    if new_rate <= old_rate:
        return traj
    ratio = max(0.15, min(1.0, old_rate / new_rate))
    new_depth = _round3(traj.args[1] * ratio)
    return Trajectory(
        param=traj.param,
        shape=traj.shape,
        args=(traj.args[0], new_depth) + traj.args[2:],
    )


def quantize_rates(scene: Scene, cps: float) -> Scene:
    """Quantise sine/walk trajectory rates to metric divisions of the cycle.

    Candidate rates: ``cps × {0.25, 0.5, 1.0}``, clamped to
    ``[RATE_LO, RATE_HI]``.  Rates below ``cps / 4`` are **drift** and are
    left completely untouched.

    When a rate is raised to a higher candidate, depth (arg index 1) is
    scaled by ``old_rate / new_rate`` (clamped to [0.15, 1.0]) to keep
    per-tick velocity bounded — fast-shallow or slow-deep.

    **De-unison**: after quantisation, trajectories are walked in (voice,
    mod) order and each division may only be claimed once.  When a
    trajectory lands on an already-used division it is moved to the nearest
    unused one; if all are used, it cycles to index ``(i + 1) % 3``.  The
    depth/rate trade applies again if this raises the rate.

    ``cps ≤ 0`` → scene returned unchanged.
    ``ramp`` / ``steps`` trajectories are not touched.
    The ``walk`` seed (arg index 3) is never modified.
    """
    if cps <= 0:
        return scene

    drift_threshold = cps / 4.0

    # Precompute 3-division candidates (de-duplicated, sorted ascending).
    candidates: list[float] = sorted(
        {max(RATE_LO, min(RATE_HI, cps * m)) for m in _METRIC_MULTS}
    )

    def _nearest(rate: float) -> float:
        return min(candidates, key=lambda c: (abs(c - rate), c))

    # Mutable copy of each voice's mods list.
    new_mods: list[list[Trajectory]] = [list(v.mods) for v in scene.voices]

    # Track which (vi, mi) pairs are rhythmic (rate ≥ drift_threshold).
    rhythmic: set[tuple[int, int]] = set()

    # --- First pass: quantise rhythmic rates and apply depth/rate trade. ---
    for vi, v in enumerate(scene.voices):
        for mi, traj in enumerate(v.mods):
            if traj.shape not in ("sine", "walk") or len(traj.args) < 3:
                continue
            old_rate = traj.args[2]
            if old_rate < drift_threshold:
                continue  # drift — leave untouched
            rhythmic.add((vi, mi))
            new_rate = _nearest(old_rate)
            updated = Trajectory(
                param=traj.param,
                shape=traj.shape,
                args=traj.args[:2] + (new_rate,) + traj.args[3:],
            )
            updated = _with_depth_trade(updated, old_rate, new_rate)
            new_mods[vi][mi] = updated

    # --- De-unison pass: assign distinct divisions in (voice, mod) order. ---
    used: set[float] = set()
    for vi in range(len(scene.voices)):
        for mi in range(len(new_mods[vi])):
            if (vi, mi) not in rhythmic:
                continue
            traj = new_mods[vi][mi]
            rate = traj.args[2]

            if rate not in used:
                used.add(rate)
            else:
                # Move to nearest unused division.
                unused = [c for c in candidates if c not in used]
                if unused:
                    new_rate = min(unused, key=lambda c: (abs(c - rate), c))
                else:
                    # All used: cycle to next index.
                    cur_idx = next(
                        (i for i, c in enumerate(candidates) if c == rate), 0
                    )
                    new_rate = candidates[(cur_idx + 1) % len(candidates)]

                old_rate = rate
                updated = Trajectory(
                    param=traj.param,
                    shape=traj.shape,
                    args=traj.args[:2] + (new_rate,) + traj.args[3:],
                )
                updated = _with_depth_trade(updated, old_rate, new_rate)
                new_mods[vi][mi] = updated
                used.add(new_rate)

    # Rebuild scene.
    new_voices: list[Voice] = []
    for vi, v in enumerate(scene.voices):
        new_voices.append(
            Voice(
                source_name=v.source_name,
                n=v.n,
                controls=v.controls,
                mods=tuple(new_mods[vi]),
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

    1. ``voice_chord`` — assign distinct chord degrees (root, fifth, third,
       seventh) to static-note voices sorted bass-up; each note placed at the
       nearest octave.  Skipped when ``label`` is ``None`` or unknown.
    2. ``space_registers`` — octave-spread voices with static notes to a
       pairwise minimum gap of 3 semitones.
    3. ``quantize_rates`` — sine/walk rates → nearest metric division
       (cps × {0.25, 0.5, 1.0}); drift rates (< cps/4) untouched; depth
       scaled when rate is raised; de-unison assigns distinct divisions.
    4. ``stage_gains`` — multiply gains by register factor.

    Pure.  Returns a new ``Scene``; ``scene`` is not mutated.
    """
    s = voice_chord(scene, label)
    s = space_registers(s)
    s = quantize_rates(s, cps)
    s = stage_gains(s)
    return s
