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

from .params import SYNTHS, midicps

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
