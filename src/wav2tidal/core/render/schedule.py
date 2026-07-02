"""Event scheduling — the shared timing core (T031, FR-013).

Turns a ``Pattern`` into a list of timed sample events. This is the ONE
place mini-notation timing is interpreted; the offline renderer (T032) and
the live path (US3) both consume its output, so an offline render and live
playback of the same pattern agree by construction (FR-013).

Pure and deterministic: a function of (pattern, cps, n_cycles).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pattern.grammar import parse_mini
from ..pattern.model import Pattern

_DEFAULTS = {"gain": 1.0, "speed": 1.0, "pan": 0.5}


@dataclass(frozen=True)
class Event:
    start: float  # seconds from the render origin
    bank: str
    index: int
    speed: float
    gain: float
    pan: float
    duration: float  # seconds until the slot ends (informational)


def bjorklund(k: int, n: int) -> list[bool]:
    """Euclidean rhythm: k onsets distributed as evenly as possible over n steps."""
    if n <= 0 or k <= 0:
        return [False] * max(n, 0)
    k = min(k, n)
    pattern: list[bool] = []
    bucket = 0
    for _ in range(n):
        bucket += k
        if bucket >= n:
            bucket -= n
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern


def schedule_events(pattern: Pattern, cps: float, n_cycles: int = 1) -> list[Event]:
    """Schedule a pattern over ``n_cycles`` at ``cps`` cycles/second."""
    if cps <= 0:
        raise ValueError("cps must be positive")
    cycle_dur = 1.0 / cps
    ctrl = {**_DEFAULTS, **pattern.controls}
    tree = parse_mini(pattern.mini)
    sequence = tree.children[0]  # start -> sequence
    events: list[Event] = []

    def emit(node, s: float, e: float) -> None:
        events.append(
            Event(
                start=s,
                bank=str(node.children[0]),
                index=int(node.children[1]) if len(node.children) > 1 else 0,
                speed=ctrl["speed"],
                gain=ctrl["gain"],
                pan=ctrl["pan"],
                duration=e - s,
            )
        )

    def sched_atom(node, s: float, e: float, ci: int) -> None:
        if node.data == "rest":
            return
        if node.data == "event":
            emit(node, s, e)
            return
        if node.data == "group":
            for seq in node.children[0].children:  # stack -> sequences (overlaid)
                sched_seq(seq, s, e, ci)

    def sched_element(node, s: float, e: float, ci: int) -> None:
        if node.data == "modified":
            atom, mod = node.children[0], str(node.children[1])
            if mod.startswith("*"):
                k = int(mod[1:])
                step = (e - s) / k
                for j in range(k):
                    sched_atom(atom, s + j * step, s + (j + 1) * step, ci)
            elif mod.startswith("/"):
                k = int(mod[1:])
                if ci % k == 0:
                    sched_atom(atom, s, e, ci)
            elif mod.startswith("("):
                nums = mod[1:-1].split(",")
                onsets, steps = int(nums[0]), int(nums[1])
                step = (e - s) / steps
                for j, hit in enumerate(bjorklund(onsets, steps)):
                    if hit:
                        sched_atom(atom, s + j * step, s + (j + 1) * step, ci)
        else:
            sched_atom(node, s, e, ci)

    def sched_seq(seq, s: float, e: float, ci: int) -> None:
        children = seq.children
        n = len(children)
        if n == 0:
            return
        slot = (e - s) / n
        for i, el in enumerate(children):
            sched_element(el, s + i * slot, s + (i + 1) * slot, ci)

    for ci in range(n_cycles):
        base = ci * cycle_dur
        sched_seq(sequence, base, base + cycle_dur, ci)
    return events
