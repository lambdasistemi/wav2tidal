"""Pattern validation (T029, FR-009/010).

Every pattern — sampled, model-generated, or mutated — passes through
here before it is rendered, trained on, or sent live. Checks, in order:
syntactic membership in the pattern subset, bank references that exist,
and complexity within configured bounds. Invalid patterns never reach
audio or training data (FR-010).

Pure: takes a Pattern and a bank inventory, returns a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..render.schedule import schedule_events
from .grammar import LarkError, bank_refs, nesting_depth, parse_mini
from .model import Pattern


@dataclass(frozen=True)
class PatternBounds:
    max_events_per_cycle: int = 64
    max_nesting_depth: int = 4


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reason: str | None = None
    events_per_cycle: int = 0
    nesting_depth: int = 0


def validate(
    pattern: Pattern,
    banks: dict[str, int],
    bounds: PatternBounds | None = None,
) -> Verdict:
    """Validate against the grammar, the bank inventory {name: size}, and bounds."""
    bounds = bounds or PatternBounds()
    try:
        tree = parse_mini(pattern.mini)
    except LarkError as e:
        return Verdict(False, f"syntax: {e.__class__.__name__}")

    for name, index in bank_refs(tree):
        if name not in banks:
            return Verdict(False, f"unknown bank: {name}")
        if index >= banks[name]:
            return Verdict(
                False,
                f"index {index} out of range for bank {name} (size {banks[name]})",
            )

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
