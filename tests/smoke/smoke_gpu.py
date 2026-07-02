"""FR-018 GPU/training feasibility gate (T007 scaffold).

Run in the training shell only:  nix develop .#training -c just smoke-gpu

Two steps, from research.md R2:
  1. torch sees the gfx1151 GPU and rocBLAS runs a matmul.
  2. one bf16 training step (LoRA/seq2seq) completes — no bitsandbytes.

Prints PASS/FAIL with an actionable fix on FAIL. NOT part of CI
(constitution IV: CI never needs ROCm). Exit 0 = PASS, 1 = FAIL.

This is a scaffold stub: it performs step 1 (cheap, no model download) and
leaves the one-step train check to T035 once the ByT5 trainer exists.
"""

from __future__ import annotations

import sys


def _fail(msg: str, fix: str) -> int:
    print(f"FAIL: {msg}\n  fix: {fix}", file=sys.stderr)
    return 1


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        return _fail(
            "torch not importable",
            "run inside the training shell: nix develop .#training -c just smoke-gpu",
        )

    hip = getattr(torch.version, "hip", None)
    if not hip:
        return _fail(
            "torch is not a ROCm build (torch.version.hip is None)",
            "use torchWithRocm from the training shell, not a CPU torch",
        )
    if not torch.cuda.is_available():
        return _fail(
            "torch.cuda.is_available() is False",
            "ensure /dev/kfd and /dev/dri are accessible and the user is in "
            "the render+video groups; do NOT set HSA_OVERRIDE_GFX_VERSION on a "
            "native gfx1151 build",
        )

    dev = torch.cuda.get_device_name(0)
    x = torch.randn(4096, 4096, device="cuda")
    s = float((x @ x).sum().item())  # exercises rocBLAS
    ok = s == s  # not NaN
    print(f"torch={torch.__version__} hip={hip} device={dev} matmul_sum={s:.3e}")
    if not ok:
        return _fail("rocBLAS matmul produced NaN", "check ROCm install / arch build")

    print("PASS: torch sees gfx1151 and rocBLAS runs.")
    print("NOTE: one-step train check lands with the ByT5 trainer (T035).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
