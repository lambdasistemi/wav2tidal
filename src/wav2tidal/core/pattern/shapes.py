"""Trajectory shapes (grammar v3, design-change-002).

The modulation-shape vocabulary for parameter scenes: what each shape's
arguments mean, when they are valid against a param's (lo, hi) range from
the table, and how to sample valid arguments. Shared by the generator,
the validator, and — in US2-scene-2 — the trajectory sampler that turns a
shape into timed values for the renderers and the live /ctrl stream.

Shapes (args are positional in the scene text):

- ``ramp v0 v1``               linear from v0 to v1 over the scene
- ``sine center depth rate``   LFO around center, +/- depth, rate in Hz
- ``walk center depth rate seed``  seeded random walk within +/- depth
- ``steps v1 v2 ...``          per-cycle value pattern (1..MAX_STEPS)

Only continuous/log params are modulatable (v1 of the vocabulary); the
lin/log tag decides interpolation space downstream. Pure — no IO.
"""

from __future__ import annotations

import random

# LFO/walk rate bounds (Hz): slow drift up to audible wobble.
RATE_LO = 0.02
RATE_HI = 8.0
MAX_STEPS = 8
MAX_SEED = 65535

SHAPES = ("ramp", "sine", "walk", "steps")

# Fixed arity per shape; None = variadic (steps).
ARITY: dict[str, int | None] = {"ramp": 2, "sine": 3, "walk": 4, "steps": None}


def valid_args(shape: str, args: tuple[float, ...], lo: float, hi: float) -> bool:
    """Are ``args`` a valid instance of ``shape`` for a param in [lo, hi]?"""
    n = ARITY.get(shape, -1)
    if n is None:
        if not 1 <= len(args) <= MAX_STEPS:
            return False
        return all(lo <= v <= hi for v in args)
    if n == -1 or len(args) != n:
        return False
    if shape == "ramp":
        v0, v1 = args
        return lo <= v0 <= hi and lo <= v1 <= hi
    if shape == "sine":
        center, depth, rate = args
        return (
            depth >= 0
            and lo <= center - depth
            and center + depth <= hi
            and RATE_LO <= rate <= RATE_HI
        )
    if shape == "walk":
        center, depth, rate, seed = args
        return (
            depth >= 0
            and lo <= center - depth
            and center + depth <= hi
            and RATE_LO <= rate <= RATE_HI
            and seed == int(seed)
            and 0 <= seed <= MAX_SEED
        )
    return False


def _round3(v: float) -> float:
    return float(f"{v:.3g}")


def sample_args(
    rng: random.Random, shape: str, lo: float, hi: float
) -> tuple[float, ...]:
    """Sample valid-by-construction args for ``shape`` over [lo, hi]."""
    span = hi - lo

    def point() -> float:
        return _round3(rng.uniform(lo, hi))

    if shape == "ramp":
        return (point(), point())
    if shape == "steps":
        return tuple(point() for _ in range(rng.randint(2, MAX_STEPS)))
    # sine / walk: pick a center, then a depth within the headroom; the
    # 0.95 cap keeps 3-sig-fig rounding from drifting past the bounds
    center = _round3(rng.uniform(lo + 0.1 * span, hi - 0.1 * span))
    headroom = min(center - lo, hi - center)
    depth = _round3(rng.uniform(0.1, 0.95) * headroom)
    rate = _round3(RATE_LO * (RATE_HI / RATE_LO) ** rng.random())  # log-uniform in Hz
    if shape == "sine":
        return (center, depth, rate)
    return (center, depth, rate, float(rng.randint(0, MAX_SEED)))
