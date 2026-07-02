"""Procedural pattern generator and mutation (T030).

Seeded, pure functions that emit patterns in the subset by construction —
they never produce something the validator would reject (given a non-empty
bank inventory). This is both the synthetic-dataset source (FR-011) and the
mutation operator the live evolution reuses (FR-022).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .model import CONTROL_ORDER, Pattern


@dataclass(frozen=True)
class Diversity:
    n_steps_choices: tuple[int, ...] = (4, 8)
    rest_prob: float = 0.25
    modifier_prob: float = 0.2
    group_prob: float = 0.1
    speed_choices: tuple[float, ...] = (0.5, 1.0, 1.0, 1.0, 2.0)
    gain_range: tuple[float, float] = (0.8, 1.1)
    euclid_choices: tuple[tuple[int, int], ...] = ((3, 8), (5, 8), (3, 4))
    fast_choices: tuple[int, ...] = (2, 3)


def _event_token(rng: random.Random, banks: dict[str, int]) -> str:
    name = rng.choice(sorted(banks))
    index = rng.randrange(banks[name])
    return name if index == 0 else f"{name}:{index}"


def _rand_step(rng: random.Random, banks: dict[str, int], div: Diversity) -> str:
    if rng.random() < div.rest_prob:
        return "~"
    if rng.random() < div.group_prob:
        inner = " ".join(_event_token(rng, banks) for _ in range(2))
        return f"[{inner}]"
    token = _event_token(rng, banks)
    if rng.random() < div.modifier_prob:
        if rng.random() < 0.5:
            return f"{token}*{rng.choice(div.fast_choices)}"
        k, n = rng.choice(div.euclid_choices)
        return f"{token}({k},{n})"
    return token


def _controls(rng: random.Random, div: Diversity) -> dict[str, float]:
    return {
        "gain": round(rng.uniform(*div.gain_range), 2),
        "speed": rng.choice(div.speed_choices),
        "pan": round(rng.uniform(0.0, 1.0), 2),
    }


def generate_pattern(
    rng: random.Random, banks: dict[str, int], div: Diversity | None = None
) -> Pattern:
    if not banks:
        raise ValueError("cannot generate a pattern with no banks")
    div = div or Diversity()
    n = rng.choice(div.n_steps_choices)
    mini = " ".join(_rand_step(rng, banks, div) for _ in range(n))
    return Pattern(mini=mini, controls=_controls(rng, div), source="sampled")


def _top_tokens(mini: str) -> list[str]:
    """Split a mini-notation string into top-level steps, respecting [] and ()."""
    tokens, depth, cur = [], 0, ""
    for ch in mini:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == " " and depth == 0:
            if cur:
                tokens.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        tokens.append(cur)
    return tokens


def mutate(
    rng: random.Random,
    pattern: Pattern,
    banks: dict[str, int],
    div: Diversity | None = None,
) -> Pattern:
    """Apply one small mutation, preserving validity (FR-022)."""
    div = div or Diversity()
    if rng.random() < 0.4:  # tweak one control
        controls = dict(pattern.controls)
        controls.update(_controls(rng, div))
        key = rng.choice(CONTROL_ORDER)
        controls[key] = _controls(rng, div)[key]
        return Pattern(pattern.mini, controls, source="mutation")
    tokens = _top_tokens(pattern.mini) or [_rand_step(rng, banks, div)]
    i = rng.randrange(len(tokens))
    tokens[i] = _rand_step(rng, banks, div)
    return Pattern(" ".join(tokens), dict(pattern.controls), source="mutation")
