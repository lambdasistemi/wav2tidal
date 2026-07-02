"""wav2tidal command-line entry point.

Subcommand surface per specs/001-corpus-to-live-pipeline/contracts/cli.md.
Each stage is invoked with a config file; no stage needs source edits
(FR-026). This scaffold wires the surface; stages are implemented in later
tasks (T017+) and currently exit 2 ("not implemented").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__

_NOT_IMPLEMENTED = 2


def _todo(stage: str, task: str):
    def run(_args: argparse.Namespace) -> int:
        print(f"wav2tidal {stage}: not implemented yet ({task})", file=sys.stderr)
        return _NOT_IMPLEMENTED

    return run


def _run_ingest(args: argparse.Namespace) -> int:
    from .core.config import load_ingest_config
    from .pipeline.ingest import ingest

    corpus = Path(args.corpus) if args.corpus else None
    if corpus is None or not corpus.is_dir():
        print("ingest: --corpus must be an existing directory", file=sys.stderr)
        return 1
    cfg = load_ingest_config(args.config)
    if args.seed is not None:
        cfg = cfg.__class__.from_dict({**cfg.to_dict(), "seed": args.seed})
    report = ingest(corpus, Path(args.root), cfg)
    print(report.summary())
    return 0


def _run_profile(args: argparse.Namespace) -> int:
    import numpy as np

    from .core.config import load_ingest_config
    from .core.descriptor.types import assemble_descriptor
    from .core.dsp.features import slice_features, track_descriptors
    from .io.embedder import make_embedder
    from .io.storage import Workspace
    from .io.wav import read_wav
    from .pipeline import profile as profile_io

    if not args.query:
        print("profile: --query is required (track id or audio file)", file=sys.stderr)
        return 1
    ws = Workspace(Path(args.root))
    cfg = load_ingest_config(args.config)
    try:
        index = profile_io.load_index(ws)
    except FileNotFoundError:
        print(
            "profile: no profile found — run `wav2tidal ingest` first", file=sys.stderr
        )
        return 1

    resolved = profile_io.resolve_query_id(ws, args.query)
    if resolved is not None:
        row = index.ids.index(resolved)
        from .core.descriptor.types import StyleDescriptor

        qdesc = StyleDescriptor(index.matrix[row], index.embedder_id, index.sr_used)
    else:
        qpath = Path(args.query)
        if not qpath.is_file():
            print(
                f"profile: query {args.query!r} not in profile and not a file",
                file=sys.stderr,
            )
            return 1
        loaded = read_wav(qpath, cfg.target_sr)
        td = track_descriptors(loaded.y, loaded.sr, cfg.hop_length)
        block = slice_features(loaded.y, loaded.sr, cfg.hop_length, cfg.n_mfcc)
        block["rhythm"] = np.array([td["tempo_bpm"] / 300.0, td["onset_rate"] / 20.0])
        emb = make_embedder(cfg.embedder).embed(loaded.y, loaded.sr)
        qdesc = assemble_descriptor(
            block, index.embedder_id, index.sr_used, embedding=emb
        )

    for tid, score in index.nearest(qdesc, k=args.k):
        print(f"{score:.4f}  {tid}")
    return 0


_HANDLERS = {"ingest": _run_ingest, "profile": _run_profile}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wav2tidal", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"wav2tidal {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def opt(name, **kw):
        return (name, kw)

    seed = opt("--seed", type=int)

    # (stage, task, help, [argument specs]) — surface per contracts/cli.md.
    stages = [
        (
            "ingest",
            "T022",
            "WAVs -> SuperDirt banks + style profile",
            [opt("--config"), opt("--corpus"), opt("--root", default="."), seed],
        ),
        (
            "profile",
            "T023",
            "nearest-neighbour query over the style profile",
            [
                opt("--query"),
                opt("--k", type=int, default=5),
                opt("--config"),
                opt("--root", default="."),
            ],
        ),
        (
            "dataset",
            "T033",
            "synthesize (descriptor -> pattern) pairs",
            [opt("--config"), seed, opt("--resume", action="store_true")],
        ),
        (
            "smoke-gpu",
            "T007",
            "FR-018 GPU/training feasibility gate",
            [opt("--config")],
        ),
        (
            "train",
            "T035",
            "fine-tune ByT5 descriptor->pattern model",
            [opt("--config"), seed],
        ),
        ("eval", "T036", "produce held-out evaluation report", [opt("--checkpoint")]),
        (
            "generate",
            "T037",
            "target audio -> N ranked candidate patterns",
            [
                opt("--target", nargs="+"),
                opt("--n", type=int, default=8),
                opt("--checkpoint"),
            ],
        ),
        (
            "live",
            "T044",
            "live evolving session driving SuperDirt",
            [opt("--config"), opt("--target", nargs="+"), seed],
        ),
        ("doctor", "T045", "environment preflight for the live session", []),
    ]

    for name, task, help_, args in stages:
        p = sub.add_parser(name, help=help_)
        p.set_defaults(handler=_HANDLERS.get(name, _todo(name, task)))
        for flag, kw in args:
            p.add_argument(flag, **kw)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
