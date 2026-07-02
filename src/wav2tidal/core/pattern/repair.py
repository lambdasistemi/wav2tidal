"""Config repair: coerce a nearly-valid generated config into the space.

The model's greedy failures are overwhelmingly termination artifacts
(too many voices, a param set twice, values a hair out of range), not
gibberish. Repair applies the minimal normalizations — truncate voices to
the bound, drop duplicate/colliding/non-modulatable mods, clamp numeric
statics into their effective range — and re-validates. Returns None when
the text is beyond repair (unparseable, unknown sources, broken shapes).

Used by eval (the ``repaired_valid`` metric) and later by the live loop
(US3): a repaired config is always safe to send. Pure — no IO.
"""

from __future__ import annotations

from .grammar import LarkError
from .model import Pattern, Scene, Voice, parse_pattern_text, parse_scene_text
from .params import PARAMS, effective_range, modulatable
from .shapes import valid_args
from .validate import (
    PatternBounds,
    SceneBounds,
    Sources,
    validate,
    validate_scene,
)


def _clamp_controls(controls: dict, source_names: set[str]) -> dict:
    out: dict = {}
    for key, value in controls.items():
        spec = PARAMS.get(key)
        if spec is None:
            continue
        if isinstance(value, str) or spec.kind in ("choice", "integer"):
            if spec.in_range(value):
                out[key] = value
            continue
        lo, hi = effective_range(key, source_names)
        out[key] = min(hi, max(lo, float(value)))
    return out


def _repair_voice(voice: Voice, bounds: SceneBounds) -> Voice:
    controls = _clamp_controls(voice.controls, {voice.source_name})
    mods = []
    seen: set[str] = set()
    for mod in voice.mods:
        if mod.param in seen or mod.param in controls:
            continue
        if not modulatable(mod.param):
            continue
        lo, hi = effective_range(mod.param, {voice.source_name})
        if not valid_args(mod.shape, mod.args, lo, hi):
            continue
        seen.add(mod.param)
        mods.append(mod)
    return Voice(
        voice.source_name,
        voice.n,
        controls,
        tuple(mods[: bounds.max_mods_per_voice]),
    )


def repair_config(
    text: str,
    sources: Sources,
    scene_bounds: SceneBounds | None = None,
    pattern_bounds: PatternBounds | None = None,
) -> str | None:
    """Return a valid config text derived from ``text``, or None."""
    scene_bounds = scene_bounds or SceneBounds()
    pattern_bounds = pattern_bounds or PatternBounds()
    text = text.strip()
    try:
        if text.startswith("scene "):
            scene = parse_scene_text(text)
            voices = tuple(
                _repair_voice(v, scene_bounds)
                for v in scene.voices[: scene_bounds.max_voices]
            )
            layer = scene.layer
            if layer is not None and not validate(layer, sources).valid:
                layer = None
            repaired = Scene(voices=voices, layer=layer, source="repair")
            if validate_scene(repaired, sources, scene_bounds).valid:
                return repaired.to_text()
            return None
        if text.startswith("d1 "):
            pattern = parse_pattern_text(text)
            names = set()
            for token in pattern.mini.replace("[", " ").replace("]", " ").split():
                names.add(token.split(":")[0].split("(")[0].rstrip("*/0123456789,"))
            repaired = Pattern(
                pattern.mini,
                _clamp_controls(pattern.controls, names & sources.names()),
                source="repair",
            )
            if validate(repaired, sources, pattern_bounds).valid:
                return repaired.to_text()
            return None
    except (LarkError, ValueError):
        return None
    return None
