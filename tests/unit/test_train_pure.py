"""Training data split + validity metrics — pure, no torch (issue #22)."""

from __future__ import annotations

import json

import pytest

from wav2tidal.core.pattern.validate import Sources
from wav2tidal.train.data import load_pairs, load_sources, split_pairs
from wav2tidal.train.metrics import check_output, validity_report

SOURCES = Sources(banks={"bd": 4}, custom=frozenset())


def _rows(n):
    return [
        {"input": f"d{i}", "output": f'd1 $ s "bd:{i % 4}"', "kind": "line"}
        for i in range(n)
    ]


def test_split_is_deterministic_and_disjoint():
    rows = _rows(20)
    a = split_pairs(rows, 0.2, seed=7)
    b = split_pairs(rows, 0.2, seed=7)
    assert a == b
    train, val = a
    assert len(val) == 4 and len(train) == 16
    ids = {r["input"] for r in train} | {r["input"] for r in val}
    assert len(ids) == 20


def test_split_val_never_empty():
    _, val = split_pairs(_rows(3), 0.01, seed=0)
    assert len(val) == 1


def test_load_pairs_and_sources(tmp_path):
    (tmp_path / "pairs.jsonl").write_text(
        json.dumps({"input": "x", "output": "y", "kind": "line"}) + "\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"sources": {"banks": {"bd": 4}, "synths": ["supersaw"], "custom": []}}
        )
    )
    assert load_pairs(tmp_path)[0]["input"] == "x"
    src = load_sources(tmp_path)
    assert src.banks == {"bd": 4} and src.synths == frozenset({"supersaw"})


@pytest.mark.parametrize(
    ("text", "grammar", "valid"),
    [
        ('d1 $ s "bd bd:2" # gain 1 # pan 0.5', True, True),
        ("scene voice supersaw # note -12 mod cutoff ramp 200 2000", True, True),
        ('d1 $ s "nosuchbank"', True, False),  # parses, fails inventory
        ('d1 $ s "bd" # cutoff 99999', True, False),  # out of range
        ("scene voice supersaw mod cutoff sine 1 2", False, False),  # arity
        ("play some jazz", False, False),
        ("scene voice bd", True, False),  # banks are not voices
    ],
)
def test_check_output(text, grammar, valid):
    g, v = check_output(text, SOURCES)
    assert (g, v) == (grammar, valid), text


def test_validity_report_aggregates():
    outs = [
        'd1 $ s "bd"',
        "scene voice supersaw # note -12 mod cutoff ramp 200 2000",
        "garbage",
    ]
    refs = ['d1 $ s "bd"', "x", "y"]
    r = validity_report(outs, refs, SOURCES)
    assert r["n"] == 3
    assert abs(r["grammar_valid"] - 2 / 3) < 1e-9
    assert abs(r["validator_valid"] - 2 / 3) < 1e-9
    assert abs(r["exact_match"] - 1 / 3) < 1e-9


def test_validity_report_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        validity_report(["a"], [], SOURCES)


def test_repair_truncates_and_dedupes_scene():
    from wav2tidal.core.pattern.model import parse_scene_text
    from wav2tidal.core.pattern.repair import repair_config

    src = Sources(banks={"bd": 4})
    text = "scene " + " ".join(
        "voice supersaw # note -12 mod cutoff ramp 200 2000 mod cutoff ramp 300 400"
        for _ in range(6)
    )
    fixed = repair_config(text, src)
    assert fixed is not None
    scene = parse_scene_text(fixed)
    assert len(scene.voices) == 4
    assert all(len(v.mods) == 1 for v in scene.voices)


def test_repair_clamps_line_controls():
    from wav2tidal.core.pattern.repair import repair_config

    src = Sources(banks={"bd": 4})
    fixed = repair_config('d1 $ s "bd" # cutoff 99999 # gain 0.1', src)
    assert fixed is not None and "cutoff 12000" in fixed and "gain 0.5" in fixed


def test_repair_gives_up_on_garbage():
    from wav2tidal.core.pattern.repair import repair_config

    src = Sources(banks={"bd": 4})
    assert repair_config("play some jazz", src) is None
    assert repair_config('d1 $ s "nosuchbank"', src) is None
