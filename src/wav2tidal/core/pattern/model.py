"""Pattern value type and its Tidal-line text form (T030 support).

A ``Pattern`` is the mini-notation string plus controls. Its canonical
text is the single-channel Tidal line the model emits and the live driver
sends:

    d1 $ s "supersaw supersaw:7 ~" # note 7 # cutoff 1200 # room 0.4

The full line is governed by ``grammar/pattern_subset.lark`` v2 (FR-008);
control semantics (ranges, scopes, per-synth applicability) live in
``params.py``. Controls are emitted in the fixed ``PARAM_ORDER`` so text
is canonical/round-trippable. Pure — no IO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .params import PARAM_ORDER, PARAMS

# Kept for the v1 sample path (generator/mutation defaults).
CONTROL_ORDER = ("gain", "speed", "pan")

_LINE = re.compile(r'^d1 \$ s "(?P<mini>[^"]*)"(?P<rest>.*)$')
_CTRL = re.compile(r"#\s*(?P<key>[a-z][a-z0-9]*)\s+(?P<val>-?\d+(?:\.\d+)?|[aeiou]\b)")


def _fmt(v: float | str) -> str:
    return v if isinstance(v, str) else f"{v:g}"


@dataclass(frozen=True)
class Pattern:
    mini: str
    controls: dict[str, float | str] = field(default_factory=dict)
    source: str = "unknown"  # sampled | model | mutation | unknown

    def to_text(self) -> str:
        parts = [f'd1 $ s "{self.mini.strip()}"']
        for key in PARAM_ORDER:
            if key in self.controls:
                parts.append(f"# {key} {_fmt(self.controls[key])}")
        return " ".join(parts)


def parse_pattern_text(text: str, source: str = "model") -> Pattern:
    """Parse a Tidal line into a Pattern. Raises ValueError on a malformed line."""
    m = _LINE.match(text.strip())
    if not m:
        raise ValueError(f"not a supported d1 pattern line: {text!r}")
    controls: dict[str, float | str] = {}
    for cm in _CTRL.finditer(m.group("rest")):
        key, val = cm.group("key"), cm.group("val")
        if key not in PARAMS:
            raise ValueError(f"unsupported control {key!r}")
        controls[key] = val if key == "vowel" else float(val)
    return Pattern(mini=m.group("mini").strip(), controls=controls, source=source)


# -- Parameter scenes (grammar v3, design-change-002) ------------------------


@dataclass(frozen=True)
class Trajectory:
    """One modulated param: ``mod <param> <shape> <args...>``.

    ``args`` are the shape's positional numbers (walk's trailing seed is
    carried as a float, emitted as an int by ``%g``).
    """

    param: str
    shape: str  # ramp | sine | walk | steps
    args: tuple[float, ...]

    def to_text(self) -> str:
        return f"mod {self.param} {self.shape} " + " ".join(_fmt(a) for a in self.args)


@dataclass(frozen=True)
class Voice:
    """A sustained source: synth/custom def + static controls + trajectories."""

    source_name: str
    n: int = 0
    controls: dict[str, float | str] = field(default_factory=dict)
    mods: tuple[Trajectory, ...] = ()

    def to_text(self) -> str:
        head = f"voice {self.source_name}" + (f":{self.n}" if self.n else "")
        parts = [head]
        for key in PARAM_ORDER:
            if key in self.controls:
                parts.append(f"# {key} {_fmt(self.controls[key])}")
        parts += [m.to_text() for m in sorted(self.mods, key=lambda m: m.param)]
        return " ".join(parts)


@dataclass(frozen=True)
class Scene:
    """A parameter scene: 1..4 voices + an optional v2 event-line layer."""

    voices: tuple[Voice, ...]
    layer: Pattern | None = None
    source: str = "unknown"  # sampled | model | mutation | unknown

    def to_text(self) -> str:
        parts = ["scene"] + [v.to_text() for v in self.voices]
        if self.layer is not None:
            parts.append(f"layer {self.layer.to_text()}")
        return " ".join(parts)


def parse_scene_text(text: str, source: str = "model") -> Scene:
    """Parse a scene config into a Scene. Raises LarkError on invalid text."""
    from .grammar import line_controls, mini_text, parse_scene

    tree = parse_scene(text.strip())
    voices: list[Voice] = []
    layer = None
    for node in tree.children:
        if node.data == "scene_voice":
            event = node.children[0]
            name = str(event.children[0])
            n = int(event.children[1]) if len(event.children) > 1 else 0
            controls: dict[str, float | str] = {}
            mods: list[Trajectory] = []
            for child in node.children[1:]:
                if child.data == "control_num":
                    key, value = (str(t) for t in child.children)
                    controls[key] = float(value)
                elif child.data == "control_vowel":
                    controls["vowel"] = str(child.children[0])
                else:  # traj
                    param, shape_node = child.children
                    mods.append(
                        Trajectory(
                            param=str(param),
                            shape=shape_node.data.removeprefix("shape_"),
                            args=tuple(float(str(t)) for t in shape_node.children),
                        )
                    )
            voices.append(Voice(name, n, controls, tuple(mods)))
        else:  # scene_layer -> line
            line = node.children[0]
            layer = Pattern(
                mini=mini_text(line.children[0]),
                controls=line_controls(line),
                source=source,
            )
    return Scene(voices=tuple(voices), layer=layer, source=source)
