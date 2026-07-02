"""Pattern validation (T029, FR-009/010).

Every pattern — sampled, model-generated, or mutated — passes through
here before it is rendered, trained on, or sent live. Checks, in order:
syntactic membership of the full config line in grammar v2, source
references that exist in the inventory, control applicability and ranges
against the param table, and complexity within configured bounds. Invalid
patterns never reach audio or training data (FR-010).

Pure: takes a Pattern and a source inventory, returns a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..render.schedule import schedule_events
from .grammar import LarkError, bank_refs, nesting_depth, parse_line
from .model import Pattern
from .params import SYNTH_NAMES, applicable, check_value

# The :INT selector on a synth or custom source is its `n` knob, not a
# sample index — bounded, like the `n` param spec.
_MAX_SOURCE_N = 24


@dataclass(frozen=True)
class Sources:
    """The source inventory a config may reference (FR deltas, R7).

    ``banks``: ingested sample banks {name: size}; ``synths``: the Super*
    palette (defaults to the full param-table set); ``custom``: user
    SynthDef names (core + event-FX params only — their own args are
    unknown to the table).
    """

    banks: dict[str, int] = field(default_factory=dict)
    synths: frozenset[str] = SYNTH_NAMES
    custom: frozenset[str] = frozenset()

    @classmethod
    def banks_only(cls, banks: dict[str, int]) -> Sources:
        return cls(banks=banks, synths=frozenset())

    def names(self) -> frozenset[str]:
        return frozenset(self.banks) | self.synths | self.custom


@dataclass(frozen=True)
class PatternBounds:
    max_events_per_cycle: int = 64
    max_nesting_depth: int = 4
    max_controls: int = 16


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reason: str | None = None
    events_per_cycle: int = 0
    nesting_depth: int = 0


def validate(
    pattern: Pattern,
    sources: Sources | dict[str, int],
    bounds: PatternBounds | None = None,
) -> Verdict:
    """Validate against grammar v2, the source inventory, and bounds.

    Passing a plain ``{name: size}`` dict keeps the v1 sample-path
    behaviour: banks only, synth names rejected.
    """
    if isinstance(sources, dict):
        sources = Sources.banks_only(sources)
    bounds = bounds or PatternBounds()
    try:
        tree = parse_line(pattern.to_text())
    except LarkError as e:
        return Verdict(False, f"syntax: {e.__class__.__name__}")

    used: set[str] = set()
    for name, index in bank_refs(tree):
        used.add(name)
        if name in sources.banks:
            if index >= sources.banks[name]:
                return Verdict(
                    False,
                    f"index {index} out of range for bank {name}"
                    f" (size {sources.banks[name]})",
                )
        elif name in sources.synths or name in sources.custom:
            if index > _MAX_SOURCE_N:
                return Verdict(False, f"n {index} out of range for {name}")
        else:
            return Verdict(False, f"unknown source: {name}")

    if len(pattern.controls) > bounds.max_controls:
        return Verdict(
            False, f"{len(pattern.controls)} controls exceed {bounds.max_controls}"
        )
    for key, value in pattern.controls.items():
        if not applicable(key, used):
            return Verdict(False, f"control {key!r} not applicable to {sorted(used)}")
        if not check_value(key, value, used):
            return Verdict(False, f"control {key!r} = {value!r} out of range")

    depth = nesting_depth(tree)
    if depth > bounds.max_nesting_depth:
        return Verdict(
            False, f"nesting depth {depth} exceeds {bounds.max_nesting_depth}", 0, depth
        )

    n_events = len(schedule_events(pattern, cps=1.0, n_cycles=1))
    if n_events > bounds.max_events_per_cycle:
        return Verdict(
            False,
            f"{n_events} events/cycle exceeds {bounds.max_events_per_cycle}",
            n_events,
            depth,
        )
    return Verdict(True, None, n_events, depth)
