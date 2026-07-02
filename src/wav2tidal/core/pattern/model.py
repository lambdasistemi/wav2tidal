"""Pattern value type and its Tidal-line text form (T030 support).

A ``Pattern`` is the mini-notation string plus global controls. Its
canonical text is the single-channel Tidal line the model emits and the
live driver sends:

    d1 $ s "bd:0 ~ [sn sn] hh*2" # gain 1.0 # speed 1.0 # pan 0.5

The mini-notation is governed by ``grammar/pattern_subset.lark`` (FR-008);
the controls are a fixed, ordered set for v1. Pure — no IO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Controls emitted, in a fixed order, so text is canonical/round-trippable.
CONTROL_ORDER = ("gain", "speed", "pan")

_LINE = re.compile(r'^d1 \$ s "(?P<mini>[^"]*)"(?P<rest>.*)$')
_CTRL = re.compile(r"#\s*(?P<key>[a-z]+)\s+(?P<val>-?\d+(?:\.\d+)?)")


def _fmt(v: float) -> str:
    return f"{v:g}"


@dataclass(frozen=True)
class Pattern:
    mini: str
    controls: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"  # sampled | model | mutation | unknown

    def to_text(self) -> str:
        parts = [f'd1 $ s "{self.mini.strip()}"']
        for key in CONTROL_ORDER:
            if key in self.controls:
                parts.append(f"# {key} {_fmt(self.controls[key])}")
        return " ".join(parts)


def parse_pattern_text(text: str, source: str = "model") -> Pattern:
    """Parse a Tidal line into a Pattern. Raises ValueError on a malformed line."""
    m = _LINE.match(text.strip())
    if not m:
        raise ValueError(f"not a supported d1 pattern line: {text!r}")
    controls: dict[str, float] = {}
    for cm in _CTRL.finditer(m.group("rest")):
        key = cm.group("key")
        if key not in CONTROL_ORDER:
            raise ValueError(f"unsupported control {key!r}")
        controls[key] = float(cm.group("val"))
    return Pattern(mini=m.group("mini").strip(), controls=controls, source=source)
