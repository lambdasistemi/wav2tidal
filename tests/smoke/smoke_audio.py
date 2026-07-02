"""Audio-path smoke gate — tier-1 NRT synth rendering (US2-synth-1).

Renders a real SuperDirt ``supersaw`` via NRT and checks it is non-silent
and deterministic (two renders -> identical bytes). Needs SuperCollider +
SuperDirt; set WAV2TIDAL_SCLANG and WAV2TIDAL_SUPERDIRT_QUARK (or have
sclang-with-superdirt on PATH). NOT part of CI (constitution IV).

Run:  just smoke-audio     (exit 0 = PASS, 1 = FAIL)
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

from wav2tidal.io.superdirt import nrt_render, rt_render


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        print("FAIL: soundfile missing", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a.wav"
        b = Path(td) / "b.wav"
        params = {"freq": 220, "sustain": 1.0, "pan": 0.5}
        try:
            nrt_render("supersaw", params, 1.4, a)
            nrt_render("supersaw", params, 1.4, b)
        except RuntimeError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 1

        y, sr = sf.read(str(a))
        peak = float(abs(y).max())
        print(f"supersaw: sr={sr} shape={y.shape} peak={peak:.3f}")
        if peak < 0.01:
            print("FAIL: rendered audio is silent", file=sys.stderr)
            return 1
        if _md5(a) != _md5(b):
            print("FAIL: NRT render is not deterministic", file=sys.stderr)
            return 1
        print("  NRT: non-silent + deterministic OK")

        # Stage 2 — real-time SuperDirt render WITH global FX (reverb + filter).
        rt = Path(td) / "rt.wav"
        try:
            rt_render(
                "supersaw",
                {"note": 0, "sustain": 1.5, "cutoff": 500, "room": 0.7, "size": 0.9},
                3.0,
                rt,
            )
        except RuntimeError as e:
            print(f"FAIL (rt): {e}", file=sys.stderr)
            return 1
        yr, srr = sf.read(str(rt))
        env = abs(yr).mean(axis=1) if yr.ndim > 1 else abs(yr)
        import numpy as np

        active = np.where(env > 0.005)[0]
        span = (active[-1] - active[0]) / srr if active.size else 0.0
        peak = float(abs(yr).max())
        print(f"  RT: span={span:.2f}s peak={peak:.3f} tail={span > 1.7}")
        if peak < 0.01 or span < 1.7:
            print("FAIL: RT render silent or missing FX tail", file=sys.stderr)
            return 1

        # Stage 3 — multi-event NRT score, seeded determinism incl. a
        # noise-carrying def (superkick's WhiteNoise click) — issue #21.
        from wav2tidal.io.superdirt import nrt_render_events, rt_render_batch

        ev_a, ev_b = Path(td) / "ev_a.wav", Path(td) / "ev_b.wav"
        events = [
            (0.0, "supersaw", {"freq": 220.0, "sustain": 1.0}),
            (1.0, "superkick", {"n": 5.0, "sustain": 0.5}),
        ]
        try:
            nrt_render_events(events, 2.0, ev_a)
            nrt_render_events(events, 2.0, ev_b)
        except RuntimeError as e:
            print(f"FAIL (nrt events): {e}", file=sys.stderr)
            return 1
        if _md5(ev_a) != _md5(ev_b):
            print("FAIL: seeded multi-event NRT not deterministic", file=sys.stderr)
            return 1
        print("  NRT events: multi-event + noise def deterministic OK")

        # Stage 4 — RT batch (one boot, several jobs) + the global-delay
        # workaround: the delayed job must ring past the dry control.
        dly, dry = Path(td) / "dly.wav", Path(td) / "dry.wav"
        note = {"note": 0.0, "sustain": 1.0}
        try:
            rt_render_batch(
                [
                    (
                        dly,
                        4.0,
                        [
                            (
                                0.0,
                                "superchip",
                                dict(note, delaytime=0.35, delayfeedback=0.7),
                            )
                        ],
                    ),
                    (dry, 4.0, [(0.0, "superchip", dict(note))]),
                ]
            )
        except RuntimeError as e:
            print(f"FAIL (rt batch): {e}", file=sys.stderr)
            return 1

        def tail_rms(p: Path) -> float:
            y, sr = sf.read(str(p))
            e = abs(y).mean(axis=1) if y.ndim > 1 else abs(y)
            seg = e[int(1.2 * sr) : int(3.5 * sr)]
            return float(np.sqrt(np.mean(seg**2)))

        t_dly, t_dry = tail_rms(dly), tail_rms(dry)
        print(f"  RT batch: delay tail rms={t_dly:.4f} dry tail rms={t_dry:.4f}")
        if t_dly < 10 * max(t_dry, 1e-6) or t_dly < 0.01:
            print("FAIL: global delay tail missing in RT batch", file=sys.stderr)
            return 1

        # Stage 5 — parameter scenes (issue #29): NRT scene automation is
        # byte-deterministic and the cutoff ramp audibly sweeps via RT.
        import librosa

        from wav2tidal.core.pattern.dirt import scene_plan
        from wav2tidal.core.pattern.model import Scene, Trajectory, Voice
        from wav2tidal.core.pattern.validate import Sources
        from wav2tidal.io.superdirt import nrt_render_scene, rt_render_scene_batch

        scene = Scene(
            voices=(
                Voice(
                    "supersaw",
                    controls={"note": -12.0},
                    mods=(Trajectory("cutoff", "ramp", (300.0, 6000.0)),),
                ),
            )
        )
        plan = scene_plan(scene, Sources(), 4.0, 0.5, 0.1)
        sc_a, sc_b = Path(td) / "sc_a.wav", Path(td) / "sc_b.wav"
        try:
            nrt_render_scene(plan, sc_a)
            nrt_render_scene(plan, sc_b)
        except RuntimeError as e:
            print(f"FAIL (nrt scene): {e}", file=sys.stderr)
            return 1
        if _md5(sc_a) != _md5(sc_b):
            print("FAIL: NRT scene automation not deterministic", file=sys.stderr)
            return 1
        rt_scene = Scene(
            voices=(
                Voice(
                    "supersaw",
                    controls={"note": -12.0, "room": 0.15},
                    mods=(Trajectory("cutoff", "ramp", (300.0, 6000.0)),),
                ),
            )
        )
        sc_rt = Path(td) / "sc_rt.wav"
        try:
            rt_render_scene_batch(
                [(sc_rt, scene_plan(rt_scene, Sources(), 5.0, 0.5, 0.1))]
            )
        except RuntimeError as e:
            print(f"FAIL (rt scene): {e}", file=sys.stderr)
            return 1
        ysc, ssr = sf.read(str(sc_rt))
        mono = (ysc.mean(axis=1) if ysc.ndim > 1 else ysc).astype("float32")
        cent = librosa.feature.spectral_centroid(y=mono, sr=ssr)[0]
        m = len(cent)
        c0, c1 = float(cent[: m // 5].mean()), float(
            cent[3 * m // 5 : 4 * m // 5].mean()
        )
        print(f"  scenes: NRT deterministic OK; RT sweep centroid {c0:.0f}->{c1:.0f}Hz")
        if abs(ysc).max() < 0.1 or c1 < 1.3 * c0:
            print("FAIL: RT scene silent or cutoff sweep not audible", file=sys.stderr)
            return 1

    print(
        "PASS: NRT (deterministic, multi-event, scenes) + RT capture"
        " + batches with global FX + scene automation all work."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
