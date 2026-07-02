"""Load and parse the pattern-subset grammar (FR-008).

Locates ``grammar/pattern_subset.lark`` (overridable via
``WAV2TIDAL_GRAMMAR``) by walking up from this file, so it works from a
source checkout or any worktree. Exposes a cached parser plus tree helpers
(bank references, nesting depth) used by both the validator and the
scheduler — the one place the grammar is interpreted.
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
    "bank_refs",
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
    return Lark(grammar_path().read_text(), parser="earley")


def parse_mini(mini: str) -> Tree:
    """Parse a mini-notation string. Raises LarkError on invalid input."""
    return get_parser().parse(mini)


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
