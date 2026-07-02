"""Model-output validity metrics (T036, FR-015 gate; issue #22).

Pure text metrics over generated configs — the eval half of the ≥95%
valid requirement. Grammar membership uses the same lark grammar the
generator emits; semantic validity uses the same validator the dataset
gate uses; there is no separate "eval grammar" to drift.

Dispatch by prefix: ``scene …`` is a grammar-v3 scene, ``d1 $ s …`` a v2
event line. Anything else is invalid outright.
"""

from __future__ import annotations

from ..core.pattern.grammar import LarkError
from ..core.pattern.model import parse_pattern_text, parse_scene_text
from ..core.pattern.validate import Sources, validate, validate_scene


def check_output(text: str, sources: Sources) -> tuple[bool, bool]:
    """(grammar_valid, validator_valid) for one generated config."""
    text = text.strip()
    try:
        if text.startswith("scene "):
            scene = parse_scene_text(text)
            return True, validate_scene(scene, sources).valid
        if text.startswith("d1 "):
            pattern = parse_pattern_text(text)
            # parse_pattern_text is lenient; full membership via validate
            verdict = validate(pattern, sources)
            if verdict.reason and verdict.reason.startswith("syntax"):
                return False, False
            return True, verdict.valid
        return False, False
    except (LarkError, ValueError):
        return False, False


def validity_report(
    outputs: list[str], references: list[str], sources: Sources
) -> dict:
    """Aggregate metrics for a batch of (generated, reference) configs."""
    if not outputs or len(outputs) != len(references):
        raise ValueError("outputs and references must be same non-zero length")
    grammar = 0
    valid = 0
    exact = 0
    for out, ref in zip(outputs, references, strict=True):
        g, v = check_output(out, sources)
        grammar += g
        valid += v
        exact += out.strip() == ref.strip()
    n = len(outputs)
    return {
        "n": n,
        "grammar_valid": grammar / n,
        "validator_valid": valid / n,
        "exact_match": exact / n,
    }
