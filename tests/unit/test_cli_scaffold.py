"""Scaffold smoke: the CLI surface exists and every stage is wired."""

from __future__ import annotations

import pytest

from wav2tidal.cli import build_parser


def test_version_is_wired():
    parser = build_parser()
    with pytest.raises(SystemExit) as e:
        parser.parse_args(["--version"])
    assert e.value.code == 0


def test_all_stages_present():
    parser = build_parser()
    stages = set(parser._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
    assert {
        "ingest",
        "profile",
        "dataset",
        "smoke-gpu",
        "train",
        "eval",
        "generate",
        "live",
        "doctor",
    } <= stages


def test_unimplemented_stage_exits_2():
    parser = build_parser()
    args = parser.parse_args(["train", "--config", "x.yaml"])
    assert args.handler(args) == 2
