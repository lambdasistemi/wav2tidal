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
from .params import GLOBAL, PARAMS, SYNTH_NAMES, SYNTHS, effective_range, midicps
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


# -- Scene render plans (US2-scene-2, issue #29) ------------------------------
#
# Scenes cannot ride /dirt/play: SuperDirt spawns nodes internally, so
# there would be nothing to n_set trajectories on. Both renderers build
# the voice graph themselves with known node handles — which also lets us
# chain the per-voice dirt_* effect synths (R7 tier-1) in module order.

from dataclasses import dataclass  # noqa: E402

from .model import Scene  # noqa: E402
from .trajectory import knots  # noqa: E402

# SuperDirt's per-event module dispatch order (core-modules.scd), reduced
# to the chainable subset: (module, def base name, activator, def args as
# {event param: def arg}). dirt_vowel needs the Vowel quark's formant
# tables — unrenderable in our chain for now (excluded by the generator).
_MODULES: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    ("shape", "dirt_shape", "shape", {"shape": "shape"}),
    ("hpf", "dirt_hpf", "hcutoff", {"hcutoff": "hcutoff", "hresonance": "hresonance"}),
    ("bpf", "dirt_bpf", "bandf", {"bandf": "bandqf", "bandq": "bandq"}),
    ("crush", "dirt_crush", "crush", {"crush": "crush"}),
    ("coarse", "dirt_coarse", "coarse", {"coarse": "coarse"}),
    ("lpf", "dirt_lpf", "cutoff", {"cutoff": "cutoff", "resonance": "resonance"}),
    (
        "envelope",
        "dirt_envelope",
        "attack",
        {"attack": "attack", "release": "release"},
    ),
)
_GLOBAL_REFS = {
    "room": ("g_reverb", "room"),
    "size": ("g_reverb", "size"),
    "delaytime": ("g_delay", "delaytime"),
    "delayfeedback": ("g_delay", "delayfeedback"),
}


@dataclass(frozen=True)
class NodePlan:
    """One synth to spawn: stable ``ref`` for automation targeting."""

    ref: str
    synth: str  # def base name; fx defs get the numChannels suffix
    params: dict[str, float | str]
    is_fx: bool = False


@dataclass(frozen=True)
class ScenePlan:
    chains: tuple[tuple[NodePlan, ...], ...]  # per voice: source, then fx
    automation: tuple[tuple[float, str, str, float], ...]  # t, ref, arg, value
    layer_events: tuple[tuple[float, str, dict[str, float | str]], ...]
    globals_static: dict[str, float]  # room/size/delaytime/delayfeedback
    duration: float
    tick: float


def scene_route(scene: Scene, sources: Sources) -> str:
    """NRT (deterministic, faster than real time) unless the scene needs
    the live engine: only a sample *layer* forces RT (buffers + SuperDirt
    event machinery). Global FX render in NRT since issue #40 — the scene
    graph owns its reverb/delay/monitor nodes, so they are ordinary
    seeded synths in the score. Raises ValueError for scenes we cannot
    render faithfully yet."""
    for voice in scene.voices:
        if "vowel" in voice.controls:
            raise ValueError("vowel on a scene voice is not renderable yet")
    return RT if scene.layer is not None else NRT


def _voice_params(voice, synth: str | None) -> dict[str, float]:
    """Creation args for the bare source def (cf. nrt_params)."""
    params: dict[str, float] = {}
    note = float(voice.controls.get("note", 0.0))
    params["freq"] = midicps(60.0 + note)
    n_val = float(voice.controls.get("n", voice.n))
    if n_val:
        params["n"] = n_val
    for key in _NRT_CORE:
        if key in voice.controls:
            params[key] = float(voice.controls[key])
    if synth is not None:
        for key in SYNTHS[synth]:
            if key in voice.controls:
                params[key] = float(voice.controls[key])
    return params


def scene_plan(
    scene: Scene,
    sources: Sources,
    duration: float,
    cps: float,
    tick: float = 0.05,
) -> ScenePlan:
    """Compile a scene into spawn chains + merged tick automation.

    Every modulated param contributes its t=0 knot to the creation args
    (no first-tick jump) and its later knots to the automation, addressed
    by (ref, def arg). ``resonance`` may fan out to both the source def
    and the chained dirt_lpf, as in SuperDirt's one-namespace routing.
    """
    chains: list[tuple[NodePlan, ...]] = []
    automation: list[tuple[float, str, str, float]] = []
    globals_static: dict[str, float] = {}

    for i, voice in enumerate(scene.voices):
        synth = voice.source_name if voice.source_name in SYNTHS else None
        ref = f"v{i}"
        source_params = _voice_params(voice, synth)
        # everything set on this voice, static or modulated — module
        # activation looks here, so a cutoff *trajectory* spawns the lpf
        present: dict[str, object] = dict(voice.controls)
        for mod in voice.mods:
            present.setdefault(mod.param, None)

        # trajectory targets: (ref, def arg) pairs per modulated param
        targets: dict[str, list[tuple[str, str]]] = {}
        for mod in voice.mods:
            p = mod.param
            if p in _GLOBAL_REFS:
                targets[p] = [_GLOBAL_REFS[p]]
                continue
            t: list[tuple[str, str]] = []
            if p == "note":
                t.append((ref, "freq"))
            elif p == "pan":
                t.append((ref, "pan"))
            elif synth is not None and p in SYNTHS[synth]:
                t.append((ref, p))
            for _, def_name, activator, args in _MODULES:
                if p in args and activator in present:
                    t.append((f"{ref}_{def_name.removeprefix('dirt_')}", args[p]))
            targets[p] = t

        # fx chain: a module is spawned when its activator is present
        fx: list[NodePlan] = []
        for _, def_name, activator, args in _MODULES:
            active = activator in present or (
                def_name == "dirt_envelope" and "release" in present
            )
            if not active:
                continue
            fx_params = {
                arg: float(voice.controls[p])
                for p, arg in args.items()
                if p in voice.controls
            }
            fx.append(
                NodePlan(
                    f"{ref}_{def_name.removeprefix('dirt_')}", def_name, fx_params, True
                )
            )

        # sample each trajectory; t=0 goes to creation args, rest to ticks
        for mod in voice.mods:
            lo, hi = effective_range(mod.param, {voice.source_name})
            is_log = PARAMS[mod.param].kind == "log"
            for t, value in knots(mod, duration, cps, tick, lo, hi, is_log):
                for tref, arg in targets[mod.param]:
                    v = midicps(60.0 + value) if arg == "freq" else value
                    if t == 0.0:
                        if tref == ref:
                            source_params[arg] = v
                        elif tref.startswith(f"{ref}_"):
                            for j, node in enumerate(fx):
                                if node.ref == tref:
                                    fx[j] = NodePlan(
                                        node.ref,
                                        node.synth,
                                        {**node.params, arg: v},
                                        True,
                                    )
                        else:
                            globals_static[arg] = v
                    else:
                        automation.append((t, tref, arg, v))

        for k, v in voice.controls.items():
            if k in _GLOBAL_REFS:
                gref, arg = _GLOBAL_REFS[k]
                globals_static[arg] = float(v)
        source_params["sustain"] = float(duration)
        chains.append((NodePlan(ref, voice.source_name, source_params), *fx))

    layer_events: tuple = ()
    if scene.layer is not None:
        layer_events = tuple(
            render_events(scene.layer, sources, cps, max(1, int(duration * cps)), RT)
        )
    automation.sort(key=lambda row: (row[0], row[1], row[2]))
    return ScenePlan(
        chains=tuple(chains),
        automation=tuple(automation),
        layer_events=layer_events,
        globals_static=globals_static,
        duration=duration,
        tick=tick,
    )
