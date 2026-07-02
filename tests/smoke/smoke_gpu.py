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
    print(f"torch={torch.__version__} hip={hip} device={dev} matmul_sum={s:.3e}")
    if s != s:  # NaN
        return _fail("rocBLAS matmul produced NaN", "check ROCm install / arch build")

    # Step 2 — a real seq2seq training step (autograd + optimizer) in bf16, the
    # ByT5 training path (T035). Random-init tiny T5; no network needed.
    try:
        from transformers import T5Config, T5ForConditionalGeneration
    except ModuleNotFoundError:
        print(
            "PASS(partial): GPU + rocBLAS OK; install transformers for the train check."
        )
        return 0

    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=384,
        d_model=128,
        d_ff=256,
        d_kv=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        pad_token_id=0,
        decoder_start_token_id=0,
        eos_token_id=1,
    )
    model = T5ForConditionalGeneration(cfg).to("cuda", dtype=torch.bfloat16).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 384, (4, 32), device="cuda")
    labels = torch.randint(0, 384, (4, 16), device="cuda")
    losses = []
    for _ in range(3):
        opt.zero_grad()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.item()))
    if any(loss_val != loss_val for loss_val in losses):
        return _fail("training loss is NaN", "bf16 instability on this ROCm build")
    print(f"seq2seq bf16 train loss: {' -> '.join(f'{v:.3f}' for v in losses)}")
    print("PASS: gfx1151 runs rocBLAS and a ByT5 seq2seq training step in bf16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
