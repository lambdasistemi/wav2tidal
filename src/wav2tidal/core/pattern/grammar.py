"""Load and parse the pattern-subset grammar (FR-008).

Locates ``grammar/pattern_subset.lark`` (overridable via
``WAV2TIDAL_GRAMMAR``) by walking up from this file, so it works from a
source checkout or any worktree. Exposes a cached parser plus tree helpers
(source references, controls, nesting depth) used by both the validator
and the scheduler — the one place the grammar is interpreted.

Grammar v2 has two start rules: ``line`` (the full Tidal config line —
what the model emits and the validator checks) and ``mini`` (the quoted
mini-notation alone — what the scheduler interprets).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from lark import Lark, Tree
from lark.exceptions import LarkError

__all__ = [
    "LarkError",
    "grammar_path",
    "get_parser",
    "parse_mini",
    "parse_line",
    "bank_refs",
    "line_controls",
    "nesting_depth",
]


def grammar_path() -> Path:
    override = os.environ.get("WAV2TIDAL_GRAMMAR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "grammar" / "pattern_subset.lark"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "grammar/pattern_subset.lark not found (set WAV2TIDAL_GRAMMAR)"
    )


@lru_cache(maxsize=1)
def get_parser() -> Lark:
    return Lark(grammar_path().read_text(), parser="earley", start=["line", "mini"])


def parse_mini(mini: str) -> Tree:
    """Parse a mini-notation string. Raises LarkError on invalid input."""
    return get_parser().parse(mini, start="mini")


def parse_line(text: str) -> Tree:
    """Parse a full config line (``d1 $ s "..." # p v ...``). Raises LarkError."""
    return get_parser().parse(text, start="line")


def line_controls(tree: Tree) -> dict[str, float | str]:
    """The ``# param value`` controls of a parsed line, in source order."""
    controls: dict[str, float | str] = {}
    for node in tree.find_data("control_num"):
        name, value = (str(t) for t in node.children)
        controls[name] = float(value)
    for node in tree.find_data("control_vowel"):
        controls["vowel"] = str(node.children[0])
    return controls


def bank_refs(tree: Tree) -> list[tuple[str, int]]:
    """All (bank, index) sample references in a parse tree."""
    refs: list[tuple[str, int]] = []
    for node in tree.find_data("event"):
        tokens = list(node.children)
        name = str(tokens[0])
        index = int(tokens[1]) if len(tokens) > 1 else 0
        refs.append((name, index))
    return refs


def nesting_depth(tree: Tree) -> int:
    """Maximum group-nesting depth (0 for a flat sequence)."""

    def depth(node) -> int:
        if not isinstance(node, Tree):
            return 0
        inc = 1 if node.data == "group" else 0
        child_depths = [depth(c) for c in node.children] or [0]
        return inc + max(child_depths)

    return depth(tree)
