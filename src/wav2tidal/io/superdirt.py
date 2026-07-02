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
    routed = _make_null_sink(sink) if sink else False
    if routed:
        env["SC_JACK_DEFAULT_OUTPUTS"] = f"{sink}:playback_FL,{sink}:playback_FR"
    try:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "rt.scd"
            script.write_text(build_rt_script(synth, params, seconds, str(out_wav)))
            proc = subprocess.run(
                [sclang, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
    finally:
        if routed:
            _unload_null_sink()

    if "WAV2TIDAL_RT_OK" not in proc.stdout or not out_wav.exists():
        raise RuntimeError(
            f"RT render failed for {synth}\nstdout tail:\n{proc.stdout[-800:]}"
        )
    return out_wav


def _make_null_sink(name: str) -> bool:
    if not shutil.which("pactl"):
        return False
    r = subprocess.run(
        ["pactl", "load-module", "module-null-sink", f"sink_name={name}"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _unload_null_sink() -> None:
    if shutil.which("pactl"):
        subprocess.run(
            ["pactl", "unload-module", "module-null-sink"], capture_output=True
        )


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
        lines = [f's.record("{out_wav}", numChannels: 2);', "s.sync;"]
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
s.waitForBoot {{
    ~dirt = SuperDirt(2, s);
    ~dirt.loadSynthDefs;
    {load_samples}s.sync;
    ~dirt.start({port}, [0]);
    s.sync;
    ~orbit = ~dirt.orbits[0];
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
    routed = _make_null_sink(sink) if sink else False
    if routed:
        env["SC_JACK_DEFAULT_OUTPUTS"] = f"{sink}:playback_FL,{sink}:playback_FR"
    try:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "rt_batch.scd"
            script.write_text(
                build_rt_batch_script(
                    norm, banks_dir=str(banks_dir) if banks_dir else None
                )
            )
            proc = subprocess.run(
                [sclang, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
    finally:
        if routed:
            _unload_null_sink()

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
~dirt = (numChannels: 2);
{loads}
Score.program = Server.program;
Score.recordNRT(
    [
        {score}
    ],
    "{osc_path}", "{out_wav}", nil,
    {sr}, "WAV", "int16",
    ServerOptions.new.numOutputBusChannels_(2), duration: {_fmt(float(seconds))},
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
        proc = subprocess.run(
            [sclang, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
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
~dirt = (numChannels: 2);
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
    ServerOptions.new.numOutputBusChannels_(2), duration: {_fmt(float(seconds))},
    action: {{ "WAV2TIDAL_NRT_OK".postln; 0.exit }}
);
)
"""


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
        proc = subprocess.run(
            [sclang, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if "WAV2TIDAL_NRT_OK" not in proc.stdout or not out_wav.exists():
        raise RuntimeError(
            f"NRT render failed for {synth}\nstdout tail:\n{proc.stdout[-800:]}"
        )
    return out_wav
