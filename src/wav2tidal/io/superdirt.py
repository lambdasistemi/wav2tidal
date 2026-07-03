"""SuperDirt NRT rendering (tier-1, design-change-001 / research R7).

Renders a single SuperDirt source synth + params to a WAV **headless and
deterministically** via SuperCollider Non-Real-Time mode — no audio device,
CI-unfriendly only in that it needs SuperCollider installed. Proven on the
box: a real ``supersaw`` renders byte-identically across runs.

``build_nrt_script`` is pure (synth+params -> sclang source) and unit-
tested; ``nrt_render`` is the IO edge that runs ``sclang-with-superdirt``.
The global-FX (reverb/delay) chain is out of scope here — that needs
real-time capture (US2-synth-2), per the design-change ADR.

Environment (until the flake pins these):
- ``WAV2TIDAL_SCLANG``           path to a sclang that has SuperDirt on its
  class path (the ``sclang-with-superdirt`` wrapper). Falls back to PATH.
- ``WAV2TIDAL_SUPERDIRT_QUARK``  path to the SuperDirt quark root (for the
  synthdef library files).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_SCLANG_ENV = "WAV2TIDAL_SCLANG"
_QUARK_ENV = "WAV2TIDAL_SUPERDIRT_QUARK"


def _fmt(v: float | int | str) -> str:
    if isinstance(v, str):  # e.g. vowel "a" (grammar v2)
        return f'"{v}"'
    return f"{v:g}" if isinstance(v, float) else str(v)


def _dirt_args(synth: str, params: dict[str, float | str]) -> str:
    parts = [f'\\s, "{synth}"']
    for k, v in sorted(params.items()):
        parts.append(f"\\{k}, {_fmt(v)}")
    return ", ".join(parts)


def build_rt_script(
    synth: str,
    params: dict[str, float | str],
    seconds: float,
    out_wav: str,
    port: int = 57120,
    server_port: int = 57110,
) -> str:
    """Pure: sclang source that boots SuperDirt, plays one /dirt/play event
    through an orbit (so the full FX chain — filters, reverb, delay — applies),
    and records the wet output bus. This is the real-time renderer (T US2-synth-2);
    unlike NRT it captures global FX, at the cost of wall-clock time + determinism.
    """
    play = (
        f'NetAddr("127.0.0.1", {port})'
        f'.sendMsg("/dirt/play", {_dirt_args(synth, params)}, \\orbit, 0);'
    )
    return f"""(
s = Server(\\w2t, NetAddr("127.0.0.1", {server_port}), s.options);
Server.default = s;             // fleet: dedicated scsynth per instance
s.options.numWireBufs = 512;    // superfm needs many interconnects
s.options.memSize = 131072;     // GVerb defs (superprimes/...) need RT memory
s.waitForBoot {{
    ~dirt = SuperDirt(2, s);
    ~dirt.loadSynthDefs;
    s.sync;
    ~dirt.start({port}, [0]);
    s.sync;
    "WAV2TIDAL_RT_READY".postln;
    s.record("{out_wav}", numChannels: 2);
    s.sync;
    {play}
    {_fmt(float(seconds))}.wait;
    s.stopRecording;
    0.5.wait;
    "WAV2TIDAL_RT_OK".postln;
    0.exit;
}};
)
"""


def rt_render(
    synth: str,
    params: dict[str, float | str],
    seconds: float,
    out_wav: str | Path,
    *,
    sclang: str | None = None,
    sink: str | None = "w2t_rt",
    timeout: float = 180.0,
) -> Path:
    """Render one synth+FX event to ``out_wav`` via a booted SuperDirt (real time).

    Captures the full orbit output including global FX (reverb/delay). If ``sink``
    is set and PipeWire is available, routes SuperCollider to a temporary null sink
    (best-effort) so playback does not reach the speakers.
    """
    sclang = _resolve_sclang(sclang)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    sink_id = _make_null_sink(sink) if sink else None
    if sink_id:
        env["SC_JACK_DEFAULT_OUTPUTS"] = f"{sink}:playback_FL,{sink}:playback_FR"
    try:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "rt.scd"
            script.write_text(build_rt_script(synth, params, seconds, str(out_wav)))
            proc = _run_sclang([sclang, str(script)], timeout, env)
    finally:
        if sink_id:
            _unload_null_sink(sink_id)

    if "WAV2TIDAL_RT_OK" not in proc.stdout or not out_wav.exists():
        raise RuntimeError(
            f"RT render failed for {synth}\nstdout tail:\n{proc.stdout[-800:]}"
        )
    return out_wav


def _make_null_sink(name: str) -> str | None:
    """Load a per-render null sink; returns the pactl module id (fleet-safe:
    unloading by module TYPE would tear down every other instance's sink)."""
    if not shutil.which("pactl"):
        return None
    r = subprocess.run(
        ["pactl", "load-module", "module-null-sink", f"sink_name={name}"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _unload_null_sink(module_id: str) -> None:
    if shutil.which("pactl"):
        subprocess.run(["pactl", "unload-module", module_id], capture_output=True)


# A renderable event: (time seconds, sound name, /dirt/play or s_new params).
RenderEvent = tuple[float, str, dict[str, float | str]]


# Global delay params are realized by a per-job synth, not the event path:
# SuperDirt's orbit-owned dirt_delay is created paused at boot and — verified
# on the box (issue #21 diagnostics) — never produces output after the
# event-driven resume, while a synth created WITH its args sounds correctly.
# A fresh per-job instance also guarantees an empty delay line per render.
_JOB_DELAY = ("delaytime", "delayfeedback")


def build_rt_batch_script(
    jobs: list[tuple[str, float, list[RenderEvent]]],
    port: int = 57120,
    banks_dir: str | None = None,
    server_port: int = 57110,
) -> str:
    """Pure: sclang source that boots SuperDirt ONCE and renders many jobs.

    Each job is ``(out_wav, seconds, events)``; events are scheduled with
    Routine waits and recorded to their own file. Booting per render costs
    ~15 s, so batch rendering is what keeps RT dataset generation inside
    the SC-010 time budget (design-change-001: wall-clock-bound).

    Reverb (room/size) rides the events — the orbit reverb resumes fine.
    Delay is spawned per job as a fresh ``dirt_delay`` synth with creation
    args (see ``_JOB_DELAY`` note) and freed afterwards.
    """
    load_samples = f'~dirt.loadSoundFiles("{banks_dir}/*");\n    ' if banks_dir else ""
    blocks = []
    for i, (out_wav, seconds, events) in enumerate(jobs):
        delay = {k: p[k] for _, _, p in events for k in _JOB_DELAY if k in p}
        lines = [
            f's.record("{out_wav}", numChannels: 2);',
            "s.sync;",
        ]
        if delay:
            dargs = ", ".join(f"\\{k}, {_fmt(v)}" for k, v in sorted(delay.items()))
            lines.append(
                '~w2t_delay = Synth("dirt_delay" ++ ~dirt.numChannels, '
                f"[\\dryBus, ~orbit.dryBus.index, \\effectBus, "
                f"~orbit.globalEffectBus.index, \\delaySend, 1, \\delayAmp, 1, "
                f"{dargs}], ~orbit.group, \\addAfter);"
            )
        t = 0.0
        for ev_t, sound, params in sorted(events, key=lambda e: e[0]):
            if ev_t > t:
                lines.append(f"{_fmt(float(ev_t - t))}.wait;")
                t = ev_t
            play = {k: v for k, v in params.items() if k not in _JOB_DELAY}
            lines.append(
                f'addr.sendMsg("/dirt/play", {_dirt_args(sound, play)}, \\orbit, 0);'
            )
        if seconds > t:
            lines.append(f"{_fmt(float(seconds - t))}.wait;")
        lines.append("s.stopRecording;")
        if delay:
            lines.append("~w2t_delay.free;")
        lines += [
            "0.8.wait;",  # let the recorder close the file before the next job
            f'"WAV2TIDAL_JOB_{i}_DONE".postln;',
        ]
        blocks.append("\n        ".join(lines))
    body = "\n        ".join(blocks)
    return f"""(
s = Server(\\w2t, NetAddr("127.0.0.1", {server_port}), s.options);
Server.default = s;             // fleet: dedicated scsynth per instance
s.options.numWireBufs = 512;    // superfm needs many interconnects
s.options.memSize = 131072;     // GVerb defs (superprimes/...) need RT memory
s.waitForBoot {{
    ~dirt = SuperDirt(2, s);
    ~dirt.loadSynthDefs;
    {load_samples}s.sync;
    ~dirt.start({port}, [0]);
    s.sync;
    ~orbit = ~dirt.orbits[0];
    ~orbit.globalEffects.do {{ |fx|
        if(fx.name == \\dirt_monitor) {{ fx.synth.free }};
    }};
    SynthDef(\\w2t_monitor, {{ |dryBus, effectBus, outBus = 0|
        Out.ar(outBus, Limiter.ar(In.ar(dryBus, 2) + In.ar(effectBus, 2)))
    }}).add;
    s.sync;
    ~w2t_monitor = Synth.tail(s.defaultGroup, \\w2t_monitor,
        [\\dryBus, ~orbit.dryBus.index, \\effectBus, ~orbit.globalEffectBus.index]);
    s.sync;
    "WAV2TIDAL_RT_READY".postln;
    Routine({{
        var addr = NetAddr("127.0.0.1", {port});
        {body}
        "WAV2TIDAL_RT_OK".postln;
        0.exit;
    }}).play;
}};
)
"""


def rt_render_batch(
    jobs: list[tuple[str | Path, float, list[RenderEvent]]],
    *,
    banks_dir: str | Path | None = None,
    sclang: str | None = None,
    sink: str | None = "w2t_rt",
    port: int = 57120,
    server_port: int = 57110,
    timeout: float | None = None,
) -> list[Path]:
    """Render many event-sequences through ONE booted SuperDirt (real time).

    Returns the output paths in job order. Timeout scales with the summed
    job durations plus boot and per-job recorder gaps.
    """
    sclang = _resolve_sclang(sclang)
    outs = [Path(o) for o, _, _ in jobs]
    for o in outs:
        o.parent.mkdir(parents=True, exist_ok=True)
    norm = [(str(o), s, e) for o, (_, s, e) in zip(outs, jobs, strict=True)]
    if timeout is None:
        timeout = 90.0 + 1.5 * (sum(s for _, s, _ in jobs) + 2.5 * len(jobs))

    env = dict(os.environ)
    sink_id = _make_null_sink(sink) if sink else None
    if sink_id:
        env["SC_JACK_DEFAULT_OUTPUTS"] = f"{sink}:playback_FL,{sink}:playback_FR"
    try:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "rt_batch.scd"
            script.write_text(
                build_rt_batch_script(
                    norm,
                    port=port,
                    banks_dir=str(banks_dir) if banks_dir else None,
                    server_port=server_port,
                )
            )
            proc = _run_sclang([sclang, str(script)], timeout, env)
    finally:
        if sink_id:
            _unload_null_sink(sink_id)

    missing = [str(o) for o in outs if not o.exists()]
    if "WAV2TIDAL_RT_OK" not in proc.stdout or missing:
        raise RuntimeError(
            f"RT batch render failed ({len(missing)} of {len(outs)} outputs missing)"
            f"\nstdout tail:\n{proc.stdout[-800:]}"
        )
    return outs


def build_nrt_events_script(
    events: list[RenderEvent],
    seconds: float,
    out_wav: str,
    osc_path: str,
    synthdef_files: list[str],
    sr: int = 44100,
    seed: int = 1917,
) -> str:
    """Pure: sclang source that NRT-renders a timed event sequence.

    Like ``build_nrt_script`` but with one ``s_new`` score row per event
    (bare source synthdefs — the tier-1 subset; no module chain, issue #24).
    A ``RandSeed`` synth at score time 0 seeds the server RNG (rand ID 0,
    shared by all defs), so noise-carrying defs (superkick's WhiteNoise
    click, supersnare, superhat, …) render byte-identically (R7).
    """
    loads = "\n".join(f'"{f}".load;' for f in synthdef_files)
    synths = sorted({s for _, s, _ in events})
    rows = [
        "[0.0, [\\d_recv, "
        "SynthDef(\\w2t_seed, { |seed| RandSeed.ir(1, seed); "
        "FreeSelf.kr(Impulse.kr(0)) }).asBytes]]"
    ]
    rows += [
        f"[0.0, [\\d_recv, SynthDescLib.global[\\{s}].def.asBytes]]" for s in synths
    ]
    rows.append(f"[0.0, [\\s_new, \\w2t_seed, 999, 0, 0, \\seed, {int(seed)}]]")
    for i, (t, synth, params) in enumerate(sorted(events, key=lambda e: e[0])):
        args = " ".join(f"\\{k}, {_fmt(v)}," for k, v in sorted(params.items()))
        rows.append(
            f"[{_fmt(float(t))}, [\\s_new, \\{synth}, {1000 + i}, 0, 0, {args}]]"
        )
    rows.append(f"[{_fmt(float(seconds))}, [\\c_set, 0, 0]]")
    score = ",\n        ".join(rows)
    return f"""(
~dirt = 0 ! 2; // .numChannels -> 2 (an Event can't fake a real method)
{loads}
Score.program = Server.program;
Score.recordNRT(
    [
        {score}
    ],
    "{osc_path}", "{out_wav}", nil,
    {sr}, "WAV", "int16",
    ServerOptions.new.numOutputBusChannels_(2)
        .numWireBufs_(512).memSize_(131072), duration: {_fmt(float(seconds))},
    action: {{ "WAV2TIDAL_NRT_OK".postln; 0.exit }}
);
)
"""


def nrt_render_events(
    events: list[RenderEvent],
    seconds: float,
    out_wav: str | Path,
    *,
    sr: int = 44100,
    seed: int = 1917,
    sclang: str | None = None,
    synthdef_files: list[str] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Render a timed synth-event sequence to ``out_wav`` via NRT (tier 1)."""
    sclang = _resolve_sclang(sclang)
    synthdef_files = synthdef_files or _default_synthdef_files()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "nrt.scd"
        osc = Path(td) / "score.osc"
        script.write_text(
            build_nrt_events_script(
                events, seconds, str(out_wav), str(osc), synthdef_files, sr, seed
            )
        )
        proc = _run_sclang([sclang, str(script)], timeout)
    if "WAV2TIDAL_NRT_OK" not in proc.stdout or not out_wav.exists():
        raise RuntimeError(
            f"NRT events render failed\nstdout tail:\n{proc.stdout[-800:]}"
        )
    return out_wav


def build_nrt_script(
    synth: str,
    params: dict[str, float | str],
    seconds: float,
    out_wav: str,
    osc_path: str,
    synthdef_files: list[str],
    sr: int = 44100,
) -> str:
    """Pure: produce the sclang source that NRT-renders one synth event.

    Loads the SuperDirt synthdef library files (with a faked ``~dirt`` env so
    they compile without a booted server), then ``Score.recordNRT`` a single
    ``s_new`` of ``synth`` with ``params``.
    """
    loads = "\n".join(f'"{f}".load;' for f in synthdef_files)
    args = " ".join(f"\\{k}, {_fmt(v)}," for k, v in sorted(params.items()))
    return f"""(
~dirt = 0 ! 2; // .numChannels -> 2 (an Event can't fake a real method)
{loads}
Score.program = Server.program;
Score.recordNRT(
    [
        [0.0, [\\d_recv, SynthDescLib.global[\\{synth}].def.asBytes]],
        [0.0, [\\s_new, \\{synth}, 1000, 0, 0, {args}]],
        [{_fmt(float(seconds))}, [\\c_set, 0, 0]]
    ],
    "{osc_path}", "{out_wav}", nil,
    {sr}, "WAV", "int16",
    ServerOptions.new.numOutputBusChannels_(2)
        .numWireBufs_(512).memSize_(131072), duration: {_fmt(float(seconds))},
    action: {{ "WAV2TIDAL_NRT_OK".postln; 0.exit }}
);
)
"""


def _run_sclang(
    cmd: list[str], timeout: float, env: dict | None = None
) -> subprocess.CompletedProcess:
    """Run sclang in its own process group and kill the WHOLE group on
    timeout — the wrapper is a shell script, so a plain kill orphans
    sclang/scsynth, which then hold the SuperDirt UDP port hostage."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        out, err = proc.communicate()
        raise RuntimeError(
            f"sclang timed out after {timeout:.0f}s (process group killed)"
            f"\nstdout tail:\n{out[-800:]}"
        ) from None
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _resolve_sclang(sclang: str | None) -> str:
    sclang = (
        sclang or os.environ.get(_SCLANG_ENV) or shutil.which("sclang-with-superdirt")
    )
    if not sclang or not Path(sclang).exists():
        raise RuntimeError(
            f"sclang-with-superdirt not found; set ${_SCLANG_ENV} to its path"
        )
    return sclang


def _default_synthdef_files() -> list[str]:
    quark = os.environ.get(_QUARK_ENV)
    if not quark:
        raise RuntimeError(f"set ${_QUARK_ENV} to the SuperDirt quark root")
    lib = Path(quark) / "library" / "default-synths-extra.scd"
    if not lib.exists():
        raise RuntimeError(f"SuperDirt synthdef library not found at {lib}")
    return [str(lib)]


def nrt_render(
    synth: str,
    params: dict[str, float | str],
    seconds: float,
    out_wav: str | Path,
    *,
    sr: int = 44100,
    sclang: str | None = None,
    synthdef_files: list[str] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Render one synth event to ``out_wav`` via SuperCollider NRT. Returns the path."""
    sclang = _resolve_sclang(sclang)
    synthdef_files = synthdef_files or _default_synthdef_files()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "nrt.scd"
        osc = Path(td) / "score.osc"
        script.write_text(
            build_nrt_script(
                synth, params, seconds, str(out_wav), str(osc), synthdef_files, sr
            )
        )
        proc = _run_sclang([sclang, str(script)], timeout)
    if "WAV2TIDAL_NRT_OK" not in proc.stdout or not out_wav.exists():
        raise RuntimeError(
            f"NRT render failed for {synth}\nstdout tail:\n{proc.stdout[-800:]}"
        )
    return out_wav


# -- Scene rendering (US2-scene-2, issue #29) ---------------------------------
#
# Scenes spawn their own voice graphs (known node ids) so trajectories can
# n_set running synths; per-voice dirt_* FX are chained in module order
# (R7 tier-1). Rendered output is peak-normalized (design-change-002: the
# loudness fix) so descriptors are level-comparable across configs.

_NRT_BASE_NODE = 2000
_NRT_BASE_BUS = 16  # after the 2 hardware output channels; NRT has 1024
_NORM_PEAK = 0.891  # -1 dBFS


def _flatten_plan(plan) -> tuple[list, dict[str, int]]:
    """Assign node ids in spawn order; returns (nodes, ref -> id)."""
    nodes = []
    ids: dict[str, int] = {}
    nid = _NRT_BASE_NODE
    for i, chain in enumerate(plan.chains):
        for node in chain:
            ids[node.ref] = nid
            nodes.append((nid, i, node))
            nid += 1
        ids[f"v{i}_route"] = nid
        nid += 1
    ids["g_reverb"] = nid
    ids["g_delay"] = nid + 1
    return nodes, ids


def _def_name(node) -> str:
    return f"{node.synth}2" if node.is_fx else node.synth


def build_nrt_scene_script(
    plan,
    out_wav: str,
    osc_path: str,
    synthdef_files: list[str],
    sr: int = 44100,
    seed: int = 1917,
) -> str:
    """Pure: sclang source that NRT-renders a scene plan.

    Each voice runs on a private bus through its FX chain (addToTail keeps
    creation order = execution order), then a ``w2t_route`` copier sums it
    to the output. Trajectories are ``n_set`` score rows. Deterministic:
    RandSeed at t=0 + fixed node/bus allocation.
    """
    loads = "\n".join(f'"{f}".load;' for f in synthdef_files)
    nodes, ids = _flatten_plan(plan)
    defs = sorted({_def_name(n) for _, _, n in nodes})
    rows = [
        "[0.0, [\\d_recv, "
        "SynthDef(\\w2t_seed, { |seed| RandSeed.ir(1, seed); "
        "FreeSelf.kr(Impulse.kr(0)) }).asBytes]]",
        "[0.0, [\\d_recv, "
        "SynthDef(\\w2t_route, { |bus, out = 0| "
        "Out.ar(out, In.ar(bus, 2)) }).asBytes]]",
    ]
    rows += [f"[0.0, [\\d_recv, SynthDescLib.global[\\{d}].def.asBytes]]" for d in defs]
    rows.append(f"[0.0, [\\s_new, \\w2t_seed, 999, 0, 0, \\seed, {int(seed)}]]")
    for nid, i, node in nodes:
        bus = _NRT_BASE_BUS + 2 * i
        args = " ".join(
            f"\\{k}, {_fmt(v)}," for k, v in sorted({**node.params, "out": bus}.items())
        )
        rows.append(f"[0.0, [\\s_new, \\{_def_name(node)}, {nid}, 1, 0, {args}]]")
    for i in range(len(plan.chains)):
        bus = _NRT_BASE_BUS + 2 * i
        rows.append(
            f"[0.0, [\\s_new, \\w2t_route, {ids[f'v{i}_route']}, 1, 0,"
            f" \\bus, {bus}, \\out, 0]]"
        )
    for t, ref, arg, value in plan.automation:
        if ref in ("g_reverb", "g_delay"):
            continue  # global FX are RT-only; scene_route sent us here NRT-clean
        rows.append(
            f"[{_fmt(float(t))}, [\\n_set, {ids[ref]}, \\{arg}, {_fmt(value)}]]"
        )
    rows.append(f"[{_fmt(float(plan.duration))}, [\\c_set, 0, 0]]")
    score = ",\n        ".join(rows)
    return f"""(
~dirt = 0 ! 2; // .numChannels -> 2 (an Event can't fake a real method)
{loads}
Score.program = Server.program;
Score.recordNRT(
    [
        {score}
    ],
    "{osc_path}", "{out_wav}", nil,
    {sr}, "WAV", "int16",
    ServerOptions.new.numOutputBusChannels_(2)
        .numWireBufs_(512).memSize_(131072), duration: {_fmt(float(plan.duration))},
    action: {{ "WAV2TIDAL_NRT_OK".postln; 0.exit }}
);
)
"""


def _scene_synthdef_files() -> list[str]:
    quark = os.environ.get(_QUARK_ENV)
    if not quark:
        raise RuntimeError(f"set ${_QUARK_ENV} to the SuperDirt quark root")
    files = [
        Path(quark) / "library" / "default-synths-extra.scd",
        Path(quark) / "synths" / "core-synths.scd",  # dirt_* effect defs
    ]
    for f in files:
        if not f.exists():
            raise RuntimeError(f"SuperDirt synthdef file not found at {f}")
    return [str(f) for f in files]


def _normalize_wav(path: Path, peak: float = _NORM_PEAK) -> None:
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float64")
    m = float(np.abs(y).max())
    if m > 0:
        y = y * (peak / m)
    # PCM_24, not FLOAT: libsndfile stamps float WAVs with a timestamped
    # PEAK chunk, which would break byte-determinism of NRT renders
    sf.write(str(path), y, sr, subtype="PCM_24")


def nrt_render_scene(
    plan,
    out_wav: str | Path,
    *,
    sr: int = 44100,
    seed: int = 1917,
    sclang: str | None = None,
    synthdef_files: list[str] | None = None,
    normalize: float | None = _NORM_PEAK,
    timeout: float = 180.0,
) -> Path:
    """Render a scene plan via NRT (deterministic) and peak-normalize."""
    sclang = _resolve_sclang(sclang)
    synthdef_files = synthdef_files or _scene_synthdef_files()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "scene.scd"
        osc = Path(td) / "score.osc"
        script.write_text(
            build_nrt_scene_script(
                plan, str(out_wav), str(osc), synthdef_files, sr, seed
            )
        )
        proc = _run_sclang([sclang, str(script)], timeout)
    if "WAV2TIDAL_NRT_OK" not in proc.stdout or not out_wav.exists():
        raise RuntimeError(
            f"NRT scene render failed\nstdout tail:\n{proc.stdout[-800:]}"
        )
    if normalize:
        _normalize_wav(out_wav, normalize)
    return out_wav


def build_rt_scene_batch_script(
    jobs: list[tuple[str, object]],
    port: int = 57120,
    banks_dir: str | None = None,
    server_port: int = 57110,
) -> str:
    """Pure: sclang source rendering scene plans through one booted SuperDirt.

    Voice chains spawn inside the orbit group writing to private buses,
    routed onto the orbit dry bus (so a sample layer via /dirt/play shares
    the same space). Global reverb/delay are OUR per-job instances with
    creation args (the orbit-owned paused ones never sound — R7 addendum),
    freed per job so delay lines start empty. Trajectories are timed
    ``set`` lines in the Routine.
    """
    load_samples = f'~dirt.loadSoundFiles("{banks_dir}/*");\n    ' if banks_dir else ""
    blocks = []
    for j, (out_wav, plan) in enumerate(jobs):
        # dirt_monitor/dirt_rms pause themselves after 4 s of orbit silence
        # (DirtPause); vanilla SuperDirt resumes them on every /dirt/play, so
        # a quiet scene tail would otherwise silence all later jobs
        lines = []
        n_voices = len(plan.chains)
        for i in range(n_voices):
            lines.append(f"~b{i} = Bus.audio(s, 2);")
        lines.append(f's.record("{out_wav}", numChannels: 2);')
        lines.append("s.sync;")  # recorder MUST run before any voice fires
        g = dict(plan.globals_static)
        needs_reverb = any(k in g for k in ("room", "size")) or any(
            ref == "g_reverb" for _, ref, _, _ in plan.automation
        )
        needs_delay = any(k in g for k in ("delaytime", "delayfeedback")) or any(
            ref == "g_delay" for _, ref, _, _ in plan.automation
        )
        if needs_reverb:
            args = ", ".join(f"\\{k}, {_fmt(g[k])}" for k in ("room", "size") if k in g)
            lines.append(
                '~g_reverb = Synth("dirt_reverb" ++ ~dirt.numChannels, '
                f"[\\dryBus, ~orbit.dryBus.index, \\effectBus, "
                f"~orbit.globalEffectBus.index{', ' + args if args else ''}], "
                "~orbit.group, \\addAfter);"
            )
        if needs_delay:
            args = ", ".join(
                f"\\{k}, {_fmt(g[k])}" for k in ("delaytime", "delayfeedback") if k in g
            )
            lines.append(
                '~g_delay = Synth("dirt_delay" ++ ~dirt.numChannels, '
                f"[\\dryBus, ~orbit.dryBus.index, \\effectBus, "
                f"~orbit.globalEffectBus.index, \\delaySend, 1, \\delayAmp, 1"
                f"{', ' + args if args else ''}], ~orbit.group, \\addAfter);"
            )
        for i, chain in enumerate(plan.chains):
            for node in chain:
                args = " ".join(
                    f"\\{k}, {_fmt(v)}," for k, v in sorted(node.params.items())
                )
                lines.append(
                    f'~n_{node.ref} = Synth.tail(~orbit.group, "{_def_name(node)}",'
                    f" [{args} \\out, ~b{i}.index]);"
                )
            lines.append(
                f"~n_v{i}_route = Synth.tail(~orbit.group, \\w2t_route,"
                f" [\\bus, ~b{i}.index, \\out, ~orbit.dryBus.index]);"
            )

        timeline: list[tuple[float, str]] = []
        for t, ref, arg, value in plan.automation:
            var = f"~{ref}" if ref.startswith("g_") else f"~n_{ref}"
            timeline.append((t, f"{var}.set(\\{arg}, {_fmt(value)});"))
        for t, sound, params in plan.layer_events:
            play_args = _dirt_args(sound, params)
            timeline.append(
                (t, f'addr.sendMsg("/dirt/play", {play_args}, \\orbit, 0);')
            )
        timeline.sort(key=lambda x: x[0])
        t_now = 0.0
        for t, line in timeline:
            if t > t_now:
                lines.append(f"{_fmt(float(t - t_now))}.wait;")
                t_now = t
            lines.append(line)
        if plan.duration > t_now:
            lines.append(f"{_fmt(float(plan.duration - t_now))}.wait;")
        lines.append("s.stopRecording;")
        for i, chain in enumerate(plan.chains):
            for node in chain:
                lines.append(f"~n_{node.ref}.free;")
            lines.append(f"~n_v{i}_route.free;")
            lines.append(f"~b{i}.free;")
        if needs_reverb:
            lines.append("~g_reverb.free;")
        if needs_delay:
            lines.append("~g_delay.free;")
        lines.append("0.8.wait;")
        lines.append(f'"WAV2TIDAL_JOB_{j}_DONE".postln;')
        blocks.append("\n        ".join(lines))
    body = "\n        ".join(blocks)
    return f"""(
s = Server(\\w2t, NetAddr("127.0.0.1", {server_port}), s.options);
Server.default = s;             // fleet: dedicated scsynth per instance
s.options.numWireBufs = 512;    // superfm needs many interconnects
s.options.memSize = 131072;     // GVerb defs (superprimes/...) need RT memory
s.waitForBoot {{
    ~dirt = SuperDirt(2, s);
    ~dirt.loadSynthDefs;
    {load_samples}s.sync;
    ~dirt.start({port}, [0]);
    s.sync;
    ~orbit = ~dirt.orbits[0];
    // dirt_monitor pauses itself after 4 s of orbit silence (DirtPause)
    // and — like the orbit delay, R7 addendum — never sounds again after a
    // resume on this box. Replace it with an unpausable monitor.
    ~orbit.globalEffects.do {{ |fx|
        if(fx.name == \\dirt_monitor) {{ fx.synth.free }};
    }};
    SynthDef(\\w2t_monitor, {{ |dryBus, effectBus, outBus = 0|
        Out.ar(outBus, Limiter.ar(In.ar(dryBus, 2) + In.ar(effectBus, 2)))
    }}).add;
    SynthDef(\\w2t_route, {{ |bus, out = 0| Out.ar(out, In.ar(bus, 2)) }}).add;
    s.sync;
    ~w2t_monitor = Synth.tail(s.defaultGroup, \\w2t_monitor,
        [\\dryBus, ~orbit.dryBus.index, \\effectBus, ~orbit.globalEffectBus.index]);
    s.sync;
    "WAV2TIDAL_RT_READY".postln;
    Routine({{
        var addr = NetAddr("127.0.0.1", {port});
        {body}
        "WAV2TIDAL_RT_OK".postln;
        0.exit;
    }}).play;
}};
)
"""


def rt_render_scene_batch(
    jobs: list[tuple[str | Path, object]],
    *,
    banks_dir: str | Path | None = None,
    sclang: str | None = None,
    sink: str | None = "w2t_rt",
    port: int = 57120,
    server_port: int = 57110,
    normalize: float | None = _NORM_PEAK,
    timeout: float | None = None,
) -> list[Path]:
    """Render scene plans through ONE booted SuperDirt; peak-normalize."""
    sclang = _resolve_sclang(sclang)
    outs = [Path(o) for o, _ in jobs]
    for o in outs:
        o.parent.mkdir(parents=True, exist_ok=True)
    norm = [(str(o), plan) for o, (_, plan) in zip(outs, jobs, strict=True)]
    if timeout is None:
        total = sum(plan.duration for _, plan in jobs)
        timeout = 90.0 + 1.5 * (total + 2.5 * len(jobs))

    env = dict(os.environ)
    sink_id = _make_null_sink(sink) if sink else None
    if sink_id:
        env["SC_JACK_DEFAULT_OUTPUTS"] = f"{sink}:playback_FL,{sink}:playback_FR"
    try:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "rt_scenes.scd"
            script.write_text(
                build_rt_scene_batch_script(
                    norm,
                    port=port,
                    banks_dir=str(banks_dir) if banks_dir else None,
                    server_port=server_port,
                )
            )
            proc = _run_sclang([sclang, str(script)], timeout, env)
    finally:
        if sink_id:
            _unload_null_sink(sink_id)

    missing = [str(o) for o in outs if not o.exists()]
    if "WAV2TIDAL_RT_OK" not in proc.stdout or missing:
        raise RuntimeError(
            f"RT scene batch failed ({len(missing)} of {len(outs)} outputs missing)"
            f"\nstdout tail:\n{proc.stdout[-800:]}"
        )
    if normalize:
        for o in outs:
            _normalize_wav(o, normalize)
    return outs
