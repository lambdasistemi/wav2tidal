"""Procedural pattern generator and mutation (T030, grammar v2).

Seeded, pure functions that emit patterns in the subset by construction —
they never produce something the validator would reject (given a non-empty
inventory). This is both the synthetic-dataset source (FR-011) and the
mutation operator the live evolution reuses (FR-022).

Two levels: the v1 sample path (``generate_pattern``/``mutate`` over a
bank inventory, unchanged) and the v2 synth+FX config space
(``generate_config``/``mutate_config`` over a ``Sources`` inventory,
design-change-001) — sources from the Super* palette, sample banks, or
custom synthdefs, with controls drawn from the ``params`` table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .model import CONTROL_ORDER, Pattern, Scene, Trajectory, Voice
from .params import (
    EVENT_FX,
    GLOBAL_FX,
    SYNTHS,
    effective_range,
    modulatable,
    spec,
)
from .shapes import sample_args
from .validate import Sources


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


# -- Grammar-v2 synth+FX config space (design-change-001, research R7) ------

_GEN_MAX_N = 12  # :n selector ceiling on synth/custom sources (< validator's 24)


@dataclass(frozen=True)
class ConfigDiversity:
    """Sampling knobs for the v2 config space (all bounded by the table)."""

    n_steps_choices: tuple[int, ...] = (1, 2, 4, 8)
    rest_prob: float = 0.2
    selector_prob: float = 0.3  # name:n on synth/custom events
    modifier_prob: float = 0.15
    fast_choices: tuple[int, ...] = (2, 3)
    euclid_choices: tuple[tuple[int, int], ...] = ((3, 8), (5, 8), (3, 4))
    bank_prob: float = 0.25  # corpus-samples-through-FX lines
    note_prob: float = 0.7
    gain_prob: float = 0.3
    pan_prob: float = 0.3
    envelope_prob: float = 0.25
    n_fx_choices: tuple[int, ...] = (0, 1, 1, 2, 3)
    n_synth_param_choices: tuple[int, ...] = (0, 1, 2, 3)
    reverb_prob: float = 0.35
    delay_prob: float = 0.25
    with_size_prob: float = 0.7
    with_feedback_prob: float = 0.7


_EVENT_FX_NAMES = tuple(s.name for s in EVENT_FX)


def _sample_source_step(rng: random.Random, name: str, div: ConfigDiversity) -> str:
    if rng.random() < div.rest_prob:
        return "~"
    token = name
    if rng.random() < div.selector_prob:
        token = f"{name}:{rng.randrange(_GEN_MAX_N + 1)}"
    if rng.random() < div.modifier_prob:
        if rng.random() < 0.5:
            return f"{token}*{rng.choice(div.fast_choices)}"
        k, n = rng.choice(div.euclid_choices)
        return f"{token}({k},{n})"
    return token


def _sample_value(rng: random.Random, name: str, sources: set[str]):
    s = spec(name)
    if s.kind in ("choice",):
        return s.sample(rng)
    return s.sample(rng, *effective_range(name, sources))


def _sample_controls(
    rng: random.Random, synth: str | None, div: ConfigDiversity
) -> dict[str, float | str]:
    sources = {synth} if synth is not None else set()
    controls: dict[str, float | str] = {}
    if rng.random() < div.note_prob:
        controls["note"] = spec("note").sample(rng)
    if rng.random() < div.gain_prob:
        controls["gain"] = spec("gain").sample(rng)
    if rng.random() < div.pan_prob:
        controls["pan"] = spec("pan").sample(rng)
    if rng.random() < div.envelope_prob:
        controls["attack"] = spec("attack").sample(rng)
        controls["release"] = spec("release").sample(rng)

    fx_pool = [n for n in _EVENT_FX_NAMES if n not in controls]
    for name in rng.sample(fx_pool, k=min(rng.choice(div.n_fx_choices), len(fx_pool))):
        controls[name] = _sample_value(rng, name, sources)

    if synth is not None:
        pool = [n for n in sorted(SYNTHS[synth]) if n not in controls]
        k = min(rng.choice(div.n_synth_param_choices), len(pool))
        for name in rng.sample(pool, k=k):
            controls[name] = _sample_value(rng, name, sources)

    if rng.random() < div.reverb_prob:
        controls["room"] = spec("room").sample(rng)
        if rng.random() < div.with_size_prob:
            controls["size"] = spec("size").sample(rng)
    if rng.random() < div.delay_prob:
        controls["delaytime"] = spec("delaytime").sample(rng)
        if rng.random() < div.with_feedback_prob:
            controls["delayfeedback"] = spec("delayfeedback").sample(rng)
    return controls


def generate_config(
    rng: random.Random, sources: Sources, div: ConfigDiversity | None = None
) -> Pattern:
    """Sample a valid-by-construction synth+FX config (grammar v2).

    Emits either a single-synth line (one Super* or custom source, with its
    applicable params) or a corpus-sample line routed through the FX space.
    """
    div = div or ConfigDiversity()
    pool = sorted(sources.synths | sources.custom)
    use_banks = sources.banks and (not pool or rng.random() < div.bank_prob)
    if not pool and not use_banks:
        raise ValueError("cannot generate a config with no sources")

    if use_banks:
        base = generate_pattern(rng, sources.banks)
        controls = dict(base.controls)
        controls.update(_sample_controls(rng, None, div))
        return Pattern(base.mini, controls, source="sampled")

    name = rng.choice(pool)
    synth = name if name in SYNTHS else None  # custom defs: core+FX only
    n = rng.choice(div.n_steps_choices)
    steps = [_sample_source_step(rng, name, div) for _ in range(n)]
    if all(s == "~" for s in steps):
        steps[rng.randrange(len(steps))] = name  # never an all-rest line
    return Pattern(" ".join(steps), _sample_controls(rng, synth, div), source="sampled")


def mutate_config(
    rng: random.Random,
    pattern: Pattern,
    sources: Sources,
    div: ConfigDiversity | None = None,
) -> Pattern:
    """One small validity-preserving mutation in the v2 space (FR-022).

    Either resamples one control value within its table range, adds or
    drops a control, or rewrites one mini step — reusing only source names
    already in the pattern, so param applicability is preserved.
    """
    div = div or ConfigDiversity()
    names = sorted({name for name, _ in _refs(pattern.mini)})
    in_line = set(names)
    synths_in = [n for n in names if n in SYNTHS]
    synth = synths_in[0] if len(synths_in) == 1 else None
    roll = rng.random()

    if roll < 0.5 and pattern.controls:  # resample one control value
        controls = dict(pattern.controls)
        key = rng.choice(sorted(controls))
        controls[key] = _sample_value(rng, key, in_line)
        return Pattern(pattern.mini, controls, source="mutation")

    if roll < 0.75:  # add or drop a control
        fresh = _sample_controls(rng, synth, div)
        controls = dict(pattern.controls)
        added = [k for k in fresh if k not in controls]
        if controls and (not added or rng.random() < 0.5):
            del controls[rng.choice(sorted(controls))]
        elif added:
            key = rng.choice(added)
            controls[key] = fresh[key]
        return Pattern(pattern.mini, controls, source="mutation")

    tokens = _top_tokens(pattern.mini)
    i = rng.randrange(len(tokens))
    tokens[i] = _sample_source_step(rng, rng.choice(names), div)
    if all(t == "~" for t in tokens):
        tokens[i] = rng.choice(names)
    return Pattern(" ".join(tokens), dict(pattern.controls), source="mutation")


def _refs(mini: str) -> list[tuple[str, int]]:
    from .grammar import bank_refs, parse_mini

    return bank_refs(parse_mini(mini))


# -- Parameter scenes (grammar v3, design-change-002) ------------------------


@dataclass(frozen=True)
class SceneDiversity:
    """Sampling knobs for the scene space (bounded by table + shapes)."""

    n_voices_choices: tuple[int, ...] = (1, 2, 2, 3, 3, 4)
    note_prob: float = 0.9
    drone_note_range: tuple[float, float] = (-24.0, 7.0)  # drones sit low
    selector_prob: float = 0.15  # name:n variants
    n_static_choices: tuple[int, ...] = (0, 1, 1, 2)
    n_mods_choices: tuple[int, ...] = (1, 1, 2, 2, 3)
    shape_choices: tuple[str, ...] = ("ramp", "sine", "sine", "walk", "walk", "steps")
    global_mod_prob: float = 0.35  # one orbit-FX trajectory somewhere
    layer_prob: float = 0.3  # rhythmic sample layer (needs banks)


_GLOBAL_NAMES = tuple(s.name for s in GLOBAL_FX)


def _modulatable(source: str | None) -> list[str]:
    """Params a voice on ``source`` may modulate (no globals) — the
    table's ``modulatable`` rule over core + event-FX + the synth's own."""
    pool = [n for n in ("note", "pan") if modulatable(n)]
    pool += [s.name for s in EVENT_FX if modulatable(s.name)]
    if source is not None:
        pool += [n for n in sorted(SYNTHS[source]) if modulatable(n)]
    return list(dict.fromkeys(pool))  # e.g. resonance is both FX and synth param


def _sample_traj(
    rng: random.Random, param: str, source: str | None, div: SceneDiversity
):
    lo, hi = effective_range(param, {source} if source else set())
    shape = rng.choice(div.shape_choices)
    return Trajectory(param=param, shape=shape, args=sample_args(rng, shape, lo, hi))


def _sample_voice(rng: random.Random, name: str, div: SceneDiversity) -> Voice:
    source = name if name in SYNTHS else None  # custom defs: core+FX only
    n = rng.randrange(_GEN_MAX_N + 1) if rng.random() < div.selector_prob else 0
    controls: dict[str, float | str] = {}
    if rng.random() < div.note_prob:
        controls["note"] = spec("note").sample(rng, *div.drone_note_range)
    static_pool = list(
        dict.fromkeys(
            p
            for p in (
                _EVENT_FX_NAMES + (tuple(sorted(SYNTHS[source])) if source else ())
            )
            if p not in controls
        )
    )
    k = min(rng.choice(div.n_static_choices), len(static_pool))
    for p in rng.sample(static_pool, k=k):
        controls[p] = _sample_value(rng, p, {name})
    mod_pool = [p for p in _modulatable(source) if p not in controls]
    k = min(rng.choice(div.n_mods_choices), len(mod_pool))
    mods = tuple(_sample_traj(rng, p, source, div) for p in rng.sample(mod_pool, k=k))
    return Voice(source_name=name, n=n, controls=controls, mods=mods)


def generate_scene(
    rng: random.Random, sources: Sources, div: SceneDiversity | None = None
) -> Scene:
    """Sample a valid-by-construction parameter scene (grammar v3).

    1..4 drone voices with static params + shaped trajectories, optionally
    one global-FX trajectory, optionally a rhythmic sample layer.
    """
    div = div or SceneDiversity()
    pool = sorted(sources.synths | sources.custom)
    if not pool:
        raise ValueError("cannot generate a scene with no synth/custom sources")
    n = rng.choice(div.n_voices_choices)
    voices = [_sample_voice(rng, rng.choice(pool), div) for _ in range(n)]
    if rng.random() < div.global_mod_prob:
        i = rng.randrange(len(voices))
        v = voices[i]
        used = set(v.controls) | {m.param for m in v.mods}
        candidates = [g for g in _GLOBAL_NAMES if g not in used]
        if candidates and len(v.mods) < 4:
            src = v.source_name if v.source_name in SYNTHS else None
            traj = _sample_traj(rng, rng.choice(candidates), src, div)
            voices[i] = Voice(v.source_name, v.n, v.controls, v.mods + (traj,))
    layer = None
    if sources.banks and rng.random() < div.layer_prob:
        layer = generate_pattern(rng, sources.banks)
    return Scene(voices=tuple(voices), layer=layer, source="sampled")


def mutate_scene(
    rng: random.Random,
    scene: Scene,
    sources: Sources,
    div: SceneDiversity | None = None,
) -> Scene:
    """One small validity-preserving mutation in the scene space (FR-022):
    resample a trajectory's args, swap its shape, tweak a static control,
    or nudge a voice's note."""
    div = div or SceneDiversity()
    voices = list(scene.voices)
    i = rng.randrange(len(voices))
    v = voices[i]
    src = v.source_name if v.source_name in SYNTHS else None
    roll = rng.random()

    if roll < 0.5 and v.mods:  # resample or reshape one trajectory
        j = rng.randrange(len(v.mods))
        mods = list(v.mods)
        mods[j] = _sample_traj(rng, mods[j].param, src, div)
        voices[i] = Voice(v.source_name, v.n, v.controls, tuple(mods))
    elif roll < 0.75 and v.controls:  # tweak one static control
        controls = dict(v.controls)
        key = rng.choice(sorted(controls))
        controls[key] = (
            spec("note").sample(rng, *div.drone_note_range)
            if key == "note"
            else _sample_value(rng, key, {v.source_name})
        )
        voices[i] = Voice(v.source_name, v.n, controls, v.mods)
    elif "note" not in {m.param for m in v.mods}:  # nudge the note (or set one)
        controls = dict(v.controls)
        controls["note"] = spec("note").sample(rng, *div.drone_note_range)
        voices[i] = Voice(v.source_name, v.n, controls, v.mods)
    elif v.mods:  # note is modulated: resample that trajectory instead
        mods = list(v.mods)
        j = rng.randrange(len(mods))
        mods[j] = _sample_traj(rng, mods[j].param, src, div)
        voices[i] = Voice(v.source_name, v.n, v.controls, tuple(mods))
    return Scene(tuple(voices), scene.layer, source="mutation")
