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

    print("PASS: SuperDirt NRT (deterministic) + real-time FX capture both work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
