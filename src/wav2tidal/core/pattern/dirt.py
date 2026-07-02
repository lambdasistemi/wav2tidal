"""Config -> SuperDirt renderer params (design-change-001, research R7).

Pure mapping from one scheduled event of a grammar-v2 config to the flat
``{param: value}`` dict the ``io/superdirt.py`` renderers consume:

- ``rt_params``   for ``rt_render`` (/dirt/play through a booted SuperDirt):
  the full vocabulary passes through — SuperDirt's event routing dispatches
  per-event FX modules and forwards global sends to the orbit.
- ``nrt_params``  for ``nrt_render`` (tier-1 Score.recordNRT of the bare
  source synthdef): only args the synthdef itself consumes — core args
  (freq derived from note/n, pan, sustain, speed, accelerate) plus the
  synth's own params. Event-FX and global params are dropped: the NRT
  score plays the source def directly, without SuperDirt's module chain.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..render.schedule import schedule_events
from .grammar import bank_refs, parse_mini
from .model import Pattern
from .params import GLOBAL, PARAMS, SYNTH_NAMES, SYNTHS, midicps
from .validate import Sources

# Source-synth args every Super* def accepts (default-synths-extra.scd).
_NRT_CORE = ("pan", "speed", "accelerate")


def rt_params(
    controls: Mapping[str, float | str], n: int = 0
) -> dict[str, float | str]:
    """Params for one /dirt/play event: all controls, plus the event's
    ``:n`` selector (sample index or synth pitch knob) when set."""
    params = dict(controls)
    if n and "n" not in params:
        params["n"] = n
    return params


def nrt_params(
    synth: str,
    controls: Mapping[str, float | str],
    sustain: float,
    n: int = 0,
) -> dict[str, float]:
    """Args for a bare ``s_new`` of ``synth`` in an NRT score.

    ``note``/``n`` become ``freq`` the way SuperDirt's event does
    (midinote = note + 60 at the default octave 5); drum-style defs that
    read ``n`` directly get it too. ``sustain`` scales the def's envelope
    and must match the scheduled event duration.
    """
    own = SYNTHS.get(synth, {})
    params: dict[str, float] = {"sustain": float(sustain)}
    note = float(controls.get("note", 0.0))
    n_val = float(controls.get("n", n))
    params["freq"] = midicps(60.0 + note)
    if n_val:
        params["n"] = n_val
    for key in _NRT_CORE:
        if key in controls:
            params[key] = float(controls[key])
    for key in own:
        if key in controls:
            params[key] = float(controls[key])
    return params


# -- Renderer routing (US2-synth-4, issue #21) -------------------------------
#
# Fidelity rule: a config may only go to a renderer that realizes EVERY
# control it sets, otherwise the descriptor would describe audio the config
# text does not match — mislabeled training pairs.

MIX = "mix"  # pure numpy slice mixdown (CI-safe; samples, gain/speed/pan only)
NRT = "nrt"  # headless deterministic Score.recordNRT (bare source defs)
RT = "rt"  # booted SuperDirt capture (full chain incl. global FX)

_MIX_FAITHFUL = frozenset({"gain", "speed", "pan"})
_NRT_COMMON = frozenset({"note", "n", "pan", "speed", "accelerate"})


def route(pattern: Pattern, sources: Sources) -> str:
    """Pick the cheapest renderer that faithfully realizes the config.

    - any global send -> RT (no NRT support for orbit FX, R7 tier 2)
    - banks only: gain/speed/pan -> MIX; any FX/note/envelope -> RT
      (the numpy mixdown implements no FX DSP; NRT has no sample buffers)
    - synth/custom only: controls the bare defs consume -> NRT; anything
      needing SuperDirt's module chain (event FX, envelope, gain) -> RT
      until the NRT score chains dirt_* effect synths (issue #24)
    - banks mixed with synths -> RT
    """
    names = {name for name, _ in bank_refs(parse_mini(pattern.mini))}
    if any(k in PARAMS and PARAMS[k].scope == GLOBAL for k in pattern.controls):
        return RT
    non_banks = names - set(sources.banks)
    if not non_banks:
        return MIX if set(pattern.controls) <= _MIX_FAITHFUL else RT
    if names & set(sources.banks):
        return RT
    allowed = set(_NRT_COMMON)
    synths = non_banks & SYNTH_NAMES
    if synths:
        allowed |= frozenset.intersection(*(frozenset(SYNTHS[s]) for s in synths))
    return NRT if set(pattern.controls) <= allowed else RT


def render_events(
    pattern: Pattern, sources: Sources, cps: float, n_cycles: int, mode: str
) -> list[tuple[float, str, dict[str, float | str]]]:
    """Schedule a config and map each event to renderer params.

    RT events carry the full control set (+ ``sustain`` = slot duration for
    synth/custom sources; samples keep SuperDirt's buffer-length default).
    NRT events carry the bare-synthdef args (``nrt_params``).
    """
    out: list[tuple[float, str, dict[str, float | str]]] = []
    for ev in schedule_events(pattern, cps, n_cycles):
        if mode == NRT:
            out.append(
                (
                    ev.start,
                    ev.bank,
                    nrt_params(ev.bank, pattern.controls, ev.duration, ev.index),
                )
            )
        else:
            params = rt_params(pattern.controls, n=ev.index)
            if ev.bank not in sources.banks:
                params.setdefault("sustain", ev.duration)
            out.append((ev.start, ev.bank, params))
    return out
