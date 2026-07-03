"""Key-snapping utilities (issue #58 — musicality 1: key-locked candidates).

The bt.wav listening test showed the output lacks a tonal centre: scene
voice ``note`` controls are sampled/mutated freely with no awareness of the
key detected in the analysis window.  These pure helpers extract the
detected key from a descriptor string and snap every candidate's note
values to the pitch classes of that key before the candidate exits the
pursuit pipeline.

Pitch-class convention: note 0 ≡ C (SuperDirt semitone offset), so
``round(note) % 12`` gives the pitch class of an integer note value, using
Python's non-negative modulo for negative note values.
"""

from __future__ import annotations

from wav2tidal.core.pattern.model import Scene, Trajectory, Voice
from wav2tidal.core.pattern.params import spec

# Pitch-class names matching features.py ``_PITCH_CLASSES`` — sharps only,
# index i is pitch class i (C=0, C#=1, …, B=11).
PITCH_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

# Scale intervals in semitones above the tonic (index 0 of the rotated scale).
_MAJOR_INTERVALS: frozenset[int] = frozenset({0, 2, 4, 5, 7, 9, 11})
_MINOR_INTERVALS: frozenset[int] = frozenset({0, 2, 3, 5, 7, 8, 10})  # natural minor

_NOTE_LO: int = int(spec("note").lo)  # −24
_NOTE_HI: int = int(spec("note").hi)  # +24


def pitch_classes(label: str) -> frozenset[int] | None:
    """Return the set of pitch-class indices (0=C … 11=B) for ``label``.

    ``label`` is a key name as produced by ``estimate_key`` in features.py,
    e.g. ``"F#"`` for F# major, ``"F#m"`` for F# natural minor.
    ``"N/A"`` or any unrecognisable string yields ``None``.
    """
    if not label or label == "N/A":
        return None
    if label.endswith("m"):
        name, intervals = label[:-1], _MINOR_INTERVALS
    else:
        name, intervals = label, _MAJOR_INTERVALS
    if name not in PITCH_NAMES:
        return None
    tonic = PITCH_NAMES.index(name)
    return frozenset((tonic + interval) % 12 for interval in intervals)


def parse_key(descriptor: str) -> str | None:
    """Extract the ``key=<label>`` token from a descriptor string.

    Returns the label string (e.g. ``"F#m"``) or ``None`` when the token
    is absent or the value is ``"N/A"``.

    Example::

        >>> parse_key("tempo=141 density=lo key=F#m brightness=1/5 motion=falling")
        'F#m'
        >>> parse_key("tempo=120 density=hi key=N/A brightness=3/5 motion=steady")
        >>> parse_key("tempo=90 density=lo brightness=2/5 motion=rising")
    """
    for token in descriptor.split():
        if token.startswith("key="):
            value = token[4:]
            return None if value == "N/A" else value
    return None


def snap_note(value: float, classes: frozenset[int]) -> float:
    """Snap ``value`` to the nearest in-key semitone (integer-valued float).

    Searches outward from ``round(value)``, checking the lower candidate
    before the upper on each radius so ties resolve downward.  The result
    is clamped to ``spec("note")`` bounds ``[−24, 24]``.

    ``classes`` must be non-empty; behaviour is undefined for the empty set
    (the function will return the clamped rounded value unchanged).
    """
    start = max(_NOTE_LO, min(_NOTE_HI, int(round(value))))
    if start % 12 in classes:
        return float(start)
    for delta in range(1, (_NOTE_HI - _NOTE_LO) + 1):
        lo_cand = start - delta
        hi_cand = start + delta
        lo_ok = lo_cand >= _NOTE_LO and lo_cand % 12 in classes
        hi_ok = hi_cand <= _NOTE_HI and hi_cand % 12 in classes
        if lo_ok:
            return float(lo_cand)
        if hi_ok:
            return float(hi_cand)
    return float(start)  # unreachable for non-empty classes


def snap_scene(scene: Scene, label: str | None) -> Scene:
    """Snap all voice note values in ``scene`` to the key named by ``label``.

    If ``label`` is ``None`` or yields no pitch classes (e.g. "N/A"), the
    scene is returned unchanged.

    For each voice:

    * Static ``"note"`` control — snapped to the nearest in-key semitone.
    * ``Trajectory`` with ``param="note"``:

      - ``"steps"`` shape: every arg is snapped independently (each step
        is a discrete pitch choice).
      - ``"ramp"`` shape: both args are snapped — a note ramp is a
        portamento between two pitches, so both endpoints are chord tones.
      - ``"sine"`` / ``"walk"`` shapes: only the **first arg** (the centre
        point) is snapped; the remaining args (range width, rate, seed)
        are left untouched.  Continuous shapes glide by design — anchoring
        the tonal centre keeps gravitational pull toward the key without
        freezing the motion.

    Pure: returns a new ``Scene`` preserving ``source`` and ``layer``.
    """
    if label is None:
        return scene
    classes = pitch_classes(label)
    if classes is None:
        return scene

    new_voices: list[Voice] = []
    for v in scene.voices:
        # Snap static note control.
        new_controls: dict = dict(v.controls)
        if "note" in new_controls and isinstance(new_controls["note"], (int, float)):
            new_controls["note"] = snap_note(float(new_controls["note"]), classes)

        # Snap note trajectories.
        new_mods: list[Trajectory] = []
        for traj in v.mods:
            if traj.param == "note":
                if traj.shape in ("steps", "ramp"):
                    # steps: discrete pitch choices; ramp: portamento between
                    # two pitches — every arg is itself a pitch.
                    new_args = tuple(snap_note(a, classes) for a in traj.args)
                else:
                    # sine / walk: snap first arg (centre) only.
                    if traj.args:
                        new_args = (snap_note(traj.args[0], classes),) + traj.args[1:]
                    else:
                        new_args = traj.args
                new_mods.append(
                    Trajectory(param=traj.param, shape=traj.shape, args=new_args)
                )
            else:
                new_mods.append(traj)

        new_voices.append(
            Voice(
                source_name=v.source_name,
                n=v.n,
                controls=new_controls,
                mods=tuple(new_mods),
            )
        )

    return Scene(voices=tuple(new_voices), layer=scene.layer, source=scene.source)
