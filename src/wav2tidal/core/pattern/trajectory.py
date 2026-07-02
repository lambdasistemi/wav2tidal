"""Trajectory sampling: shape -> timed values (US2-scene-2, issue #29).

Turns a ``Trajectory`` (shape + args, grammar v3) into the discrete
``(time, value)`` knots the renderers automate with (NRT ``n_set`` score
rows, RT tick messages) and the live agent streams over ``/ctrl``.

Semantics:

- ``ramp v0 v1``    linear over the scene duration; for log-tagged params
                    the interpolation runs in log space (a cutoff ramp
                    sweeps musically, not arithmetically).
- ``sine c d r``    c + d * sin(2*pi*r*t) — linear space; the validator
                    guarantees c±d stays inside the param range.
- ``walk c d r s``  seeded sample-and-hold random walk, linearly
                    interpolated between knots 1/r seconds apart.
- ``steps v...``    the values spread evenly over ONE CYCLE (1/cps) and
                    held — Tidal-like, repeating for the whole scene.

Pure and deterministic: a function of (trajectory, duration, cps, tick).
Values are clamped to [lo, hi] as a final guard.
"""

from __future__ import annotations

import math
import random

from .model import Trajectory


def knots(
    traj: Trajectory,
    duration: float,
    cps: float,
    tick: float,
    lo: float,
    hi: float,
    log: bool = False,
) -> list[tuple[float, float]]:
    """Sample ``traj`` every ``tick`` seconds over ``duration``.

    Returns (t, value) pairs starting at t=0; consecutive duplicate values
    are dropped (no-op ``n_set`` rows would only bloat the score).
    """
    if duration <= 0 or tick <= 0:
        raise ValueError("duration and tick must be positive")
    n = int(duration / tick)
    times = [i * tick for i in range(n + 1)]
    fn = _SHAPES[traj.shape]
    values = fn(traj.args, times, duration, cps, lo, hi, log)
    out: list[tuple[float, float]] = []
    last: float | None = None
    for t, v in zip(times, values, strict=True):
        v = float(f"{min(hi, max(lo, v)):.6g}")  # kill float-noise, aid dedupe
        if last is None or v != last:
            out.append((t, v))
            last = v
    return out


def _ramp(args, times, duration, cps, lo, hi, log):
    v0, v1 = args
    if log and v0 > 0 and v1 > 0:
        a, b = math.log(v0), math.log(v1)
        return [math.exp(a + (b - a) * (t / duration)) for t in times]
    return [v0 + (v1 - v0) * (t / duration) for t in times]


def _sine(args, times, duration, cps, lo, hi, log):
    center, depth, rate = args
    return [center + depth * math.sin(2.0 * math.pi * rate * t) for t in times]


def _walk(args, times, duration, cps, lo, hi, log):
    center, depth, rate, seed = args
    rng = random.Random(int(seed))
    seg = 1.0 / rate
    n_knots = int(duration / seg) + 2
    points = [center + rng.uniform(-depth, depth) for _ in range(n_knots)]
    out = []
    for t in times:
        i = int(t / seg)
        frac = (t - i * seg) / seg
        out.append(points[i] + (points[i + 1] - points[i]) * frac)
    return out


def _steps(args, times, duration, cps, lo, hi, log):
    cycle = 1.0 / cps
    step = cycle / len(args)
    return [args[int((t % cycle) / step) % len(args)] for t in times]


_SHAPES = {"ramp": _ramp, "sine": _sine, "walk": _walk, "steps": _steps}
