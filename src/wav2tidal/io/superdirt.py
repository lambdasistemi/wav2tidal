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
