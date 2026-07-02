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
