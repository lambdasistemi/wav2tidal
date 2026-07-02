"""ByT5 fine-tuning: descriptor text -> config text (T035, issue #22).

Full fine-tune of ``google/byt5-small`` in bf16 on the gfx1151 ROCm
substrate proven by ``just smoke-gpu`` (R2/PR #7). A compact manual
training loop instead of ``Seq2SeqTrainer``: the training shell carries
no ``accelerate`` (which modern Trainer requires), and the loop gives
exact control over seeding — same transformers/ByT5 tooling either way.

torch/transformers are imported lazily so the pure core and CI never
touch them. Run inside ``nix develop .#training``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from ..core.config import TrainConfig
from .data import load_pairs, load_sources, split_pairs
from .metrics import validity_report


def _hf_home(root: Path) -> str:
    # project-local cache (R1 offline-pinning policy); override via env
    return os.environ.get("HF_HOME", str(root / ".hf_cache"))


def train_model(root: str | Path, cfg: TrainConfig) -> Path:
    """Train + evaluate; returns the checkpoint dir (model + eval.json)."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    root = Path(root)
    os.environ.setdefault("HF_HOME", _hf_home(root))
    dataset_dir = root / "datasets" / cfg.dataset
    pairs = load_pairs(dataset_dir)
    sources = load_sources(dataset_dir)
    train_rows, val_rows = split_pairs(pairs, cfg.val_fraction, cfg.seed)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.revision)
    model = T5ForConditionalGeneration.from_pretrained(
        cfg.model_name, revision=cfg.revision
    )
    # fp32 master weights + bf16 autocast for compute: a full bf16 cast
    # silently drops optimizer updates below one bf16 ulp and training
    # plateaus (observed: train loss stuck ~0.79 while underfitting).
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    autocast = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
    )

    def encode(rows: list[dict]):
        enc = tok(
            [r["input"] for r in rows],
            padding=True,
            truncation=True,
            max_length=cfg.max_input_len,
            return_tensors="pt",
        )
        lab = tok(
            [r["output"] for r in rows],
            padding=True,
            truncation=True,
            max_length=cfg.max_target_len,
            return_tensors="pt",
        ).input_ids
        lab[lab == tok.pad_token_id] = -100
        return enc.input_ids.to(device), enc.attention_mask.to(device), lab.to(device)

    order_rng = random.Random(cfg.seed)
    history = []
    for epoch in range(cfg.epochs):
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            batch = [train_rows[i] for i in order[start : start + cfg.batch_size]]
            ids, mask, labels = encode(batch)
            opt.zero_grad()
            with autocast:
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.item()))
        train_loss = sum(losses) / len(losses)
        model.eval()
        with torch.no_grad(), autocast:
            ids, mask, labels = encode(val_rows)
            val_loss = float(
                model(input_ids=ids, attention_mask=mask, labels=labels).loss.item()
            )
        model.train()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}", flush=True)

    # greedy decode on the held-out rows -> the FR-015 validity metrics
    model.eval()
    outputs = []
    with torch.no_grad(), autocast:
        for start in range(0, len(val_rows), cfg.batch_size):
            batch = val_rows[start : start + cfg.batch_size]
            enc = tok(
                [r["input"] for r in batch],
                padding=True,
                truncation=True,
                max_length=cfg.max_input_len,
                return_tensors="pt",
            ).to(device)
            gen = model.generate(
                **enc, max_new_tokens=cfg.max_target_len, do_sample=False
            )
            outputs += tok.batch_decode(gen, skip_special_tokens=True)
    report = validity_report(outputs, [r["output"] for r in val_rows], sources)

    out_dir = root / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    resolved = getattr(model.config, "_commit_hash", None)
    (out_dir / "eval.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "model_revision_resolved": resolved,
                "dataset": str(dataset_dir),
                "n_train": len(train_rows),
                "n_val": len(val_rows),
                "history": history,
                "validity": report,
                "samples": [
                    {"input": r["input"], "reference": r["output"], "generated": o}
                    for r, o in list(zip(val_rows, outputs, strict=True))[:8]
                ],
            },
            indent=2,
        )
    )
    print(f"validity: {report}", flush=True)
    return out_dir


def generate_config_text(
    checkpoint: str | Path, descriptor: str, max_new_tokens: int = 256
) -> str:
    """Greedy descriptor -> config text from a saved checkpoint."""
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device).eval()
    enc = tok([descriptor], return_tensors="pt").to(device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ),
    ):
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]
