"""wav2tidal command-line entry point.

Subcommand surface per specs/001-corpus-to-live-pipeline/contracts/cli.md.
Each stage is invoked with a config file; no stage needs source edits
(FR-026). This scaffold wires the surface; stages are implemented in later
tasks (T017+) and currently exit 2 ("not implemented").
"""

from __future__ import annotations

import argparse
import sys

from . import __version__

_NOT_IMPLEMENTED = 2


def _todo(stage: str, task: str):
    def run(_args: argparse.Namespace) -> int:
        print(f"wav2tidal {stage}: not implemented yet ({task})", file=sys.stderr)
        return _NOT_IMPLEMENTED

    return run


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
            [opt("--config"), opt("--corpus"), seed],
        ),
        (
            "profile",
            "T023",
            "nearest-neighbour query over the style profile",
            [opt("--query"), opt("--k", type=int, default=5)],
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
        p.set_defaults(handler=_todo(name, task))
        for flag, kw in args:
            p.add_argument(flag, **kw)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
