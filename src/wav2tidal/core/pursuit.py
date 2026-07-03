"""Pursuit-engine pure policy (US3-3).

Propose / shadow-audition / select loop: decide which mode to use, build
candidate pools, score by cosine similarity, and advance state.  No file,
process, or torch IO — all heavy callables are injected by the
orchestration layer (pipeline/pursuit.py).

References:
  specs/001-corpus-to-live-pipeline/us3-live-loop-design.md
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .dsp.stream import AnalysisWindow, input_jump
from .pattern.generate import SceneDiversity, generate_scene, mutate_scene
from .pattern.model import Scene, Voice, parse_scene_text
from .pattern.params import spec
from .pattern.repair import repair_config
from .pattern.validate import Sources

# Energy arc: RMS breakpoints → gain multiplier (linear interp).
# Design: "input loudness steers ensemble gain/voice-count/trajectory depth"
# (us3-live-loop-design § "What must never be lost — the energy arc").
_ENERGY_BREAKPOINTS: tuple[tuple[float, float], ...] = (
    (0.00, 0.5),
    (0.05, 0.8),
    (0.15, 1.0),
    (0.30, 1.2),
)

# Fullness north star: never collapse the ensemble below this many voices.
_MIN_VOICES_FLOOR: int = 2


@dataclass(frozen=True)
class PursuitConfig:
    """Knobs for the pursuit engine (US3-3).

    ``k_candidates``: shadow-audition pool size per generation.
    ``patience``: stagnation limit before forcing a fresh proposal (FR-022).
    ``jump_threshold``: cosine-distance threshold forwarded to ``input_jump``.
    ``scene_duration_s``: NRT render duration per candidate (seconds).
    ``beats_per_cycle``: TidalCycles cycle length in beats.
    ``min_voices``: minimum voice count enforced by ``apply_energy``.
    """

    k_candidates: int = 8
    patience: int = 3
    jump_threshold: float = 0.35
    scene_duration_s: float = 4.0
    beats_per_cycle: float = 4.0
    min_voices: int = 2


@dataclass(frozen=True)
class PursuitState:
    """Immutable pursuit-loop state between generations.

    ``scene``: current playing scene; None before the first proposal.
    ``best_score``: highest cosine similarity achieved so far.
    ``stagnation``: consecutive generations without a score improvement.
    ``generation``: 0-indexed counter of completed successful generations.
    """

    scene: Scene | None
    best_score: float
    stagnation: int
    generation: int

    @classmethod
    def initial(cls) -> PursuitState:
        """Return the zero-state at loop start."""
        return cls(scene=None, best_score=float("-inf"), stagnation=0, generation=0)


Mode = Literal["propose", "mutate"]


def decide(
    state: PursuitState,
    prev_window: AnalysisWindow | None,
    window: AnalysisWindow,
    cfg: PursuitConfig,
) -> Mode:
    """Choose propose or mutate for this generation.

    Returns ``"propose"`` when:
    - no scene exists yet (cold start),
    - the input jumped since the previous window (``input_jump``), or
    - the score has stagnated for ``≥ cfg.patience`` generations.

    Otherwise returns ``"mutate"`` (evolve the current scene).
    """
    if state.scene is None:
        return "propose"
    if prev_window is not None and input_jump(
        prev_window, window, threshold=cfg.jump_threshold
    ):
        return "propose"
    if state.stagnation >= cfg.patience:
        return "propose"
    return "mutate"


def score(candidate_emb: np.ndarray, target_emb: np.ndarray) -> float:
    """Cosine similarity between two embeddings.

    Returns 0.0 when either array is empty or has zero norm (safe fallback
    so a missing CLAP embedding does not crash the loop).
    """
    if candidate_emb.size == 0 or target_emb.size == 0:
        return 0.0
    cn = float(np.linalg.norm(candidate_emb))
    tn = float(np.linalg.norm(target_emb))
    if cn < 1e-12 or tn < 1e-12:
        return 0.0
    c = candidate_emb.astype(np.float64)
    t = target_emb.astype(np.float64)
    return float(np.dot(c, t) / (cn * tn))


def select(scores: list[float]) -> int:
    """Return the index of the highest score; ties go to the lowest index.

    Raises ``ValueError`` on an empty list.
    """
    if not scores:
        raise ValueError("select() called with an empty scores list")
    best = max(scores)
    return scores.index(best)


def tempo_to_cps(bpm: float, beats_per_cycle: float = 4.0) -> float:
    """Convert BPM to TidalCycles ``cps``, clamped to [0.125, 2.0].

    Guards ``bpm ≤ 0`` by returning 0.5 (a safe mid-tempo fallback).
    """
    if bpm <= 0:
        return 0.5
    cps = bpm / 60.0 / beats_per_cycle
    return max(0.125, min(2.0, cps))


def _interp_gain(energy: float) -> float:
    """Map RMS energy to a gain multiplier via ``_ENERGY_BREAKPOINTS``."""
    bps = _ENERGY_BREAKPOINTS
    if energy <= bps[0][0]:
        return bps[0][1]
    if energy >= bps[-1][0]:
        return bps[-1][1]
    for (e0, g0), (e1, g1) in zip(bps, bps[1:], strict=False):
        if e0 <= energy <= e1:
            t = (energy - e0) / (e1 - e0)
            return g0 + t * (g1 - g0)
    return bps[-1][1]  # unreachable; satisfies type checker


def apply_energy(scene: Scene, energy: float) -> Scene:
    """Steer ensemble gain (and optionally voice count) from window RMS energy.

    Maps ``energy`` through ``_ENERGY_BREAKPOINTS`` to a factor, then
    multiplies each voice's "gain" control (default 1.0 when absent) and
    clamps the result to ``spec("gain")`` bounds.

    Very low energy (below the midpoint of the two lowest breakpoints)
    trims the trailing voice, but never below ``_MIN_VOICES_FLOOR`` so the
    ensemble stays full (north star).  Pure and deterministic.
    """
    gain_spec = spec("gain")
    g_lo, g_hi = gain_spec.lo, gain_spec.hi
    factor = _interp_gain(energy)

    # Trim at very low energy — midpoint between breakpoints 0 and 1.
    low_threshold = (_ENERGY_BREAKPOINTS[0][0] + _ENERGY_BREAKPOINTS[1][0]) / 2.0
    voices: list[Voice] = list(scene.voices)
    if energy < low_threshold and len(voices) > _MIN_VOICES_FLOOR:
        voices = voices[: max(_MIN_VOICES_FLOOR, len(voices) - 1)]

    new_voices: list[Voice] = []
    for v in voices:
        base = float(v.controls.get("gain", 1.0))
        new_gain = round(min(g_hi, max(g_lo, base * factor)), 6)
        new_voices.append(
            Voice(
                source_name=v.source_name,
                n=v.n,
                controls={**v.controls, "gain": new_gain},
                mods=v.mods,
            )
        )
    return Scene(voices=tuple(new_voices), layer=scene.layer, source=scene.source)


def nrt_safe(scene: Scene) -> Scene:
    """Drop the scene layer so it can be shadow-rendered via NRT.

    A sample layer forces RT (SuperDirt buffer machinery); shadow audition
    is NRT-only per design (us3-live-loop-design § "audition:").
    """
    return Scene(voices=scene.voices, layer=None, source=scene.source)


def make_candidates(
    rng: random.Random,
    mode: Mode,
    state: PursuitState,
    window: AnalysisWindow,
    sources: Sources,
    propose: Callable[[str], str | None] | None,
    cfg: PursuitConfig,
    div: SceneDiversity | None = None,
) -> list[Scene]:
    """Build a candidate pool for shadow audition.

    *propose mode*: if ``propose`` is provided, calls it with
    ``window.descriptor`` and pipes the result through ``repair_config``;
    a successful repair becomes the first candidate.  Remaining slots are
    filled with mutations of ``state.scene`` (when one exists) or fresh
    ``generate_scene`` samples.

    *mutate mode*: all candidates are mutations of ``state.scene``
    (non-None is asserted here — the caller must guarantee this).

    Every candidate is passed through ``nrt_safe`` then
    ``apply_energy(scene, window.energy)``.  Duplicates (by ``to_text()``)
    are replaced up to ``4 * k`` attempts to keep the pool full.
    """
    k = cfg.k_candidates
    max_attempts = 4 * k

    def _finalize(s: Scene) -> Scene:
        return apply_energy(nrt_safe(s), window.energy)

    seen: set[str] = set()
    pool: list[Scene] = []

    def _add(s: Scene) -> bool:
        s = _finalize(s)
        txt = s.to_text()
        if txt in seen:
            return False
        seen.add(txt)
        pool.append(s)
        return True

    if mode == "propose":
        # ByT5 proposal → repair → first candidate (when available).
        if propose is not None:
            raw = propose(window.descriptor)
            if raw is not None:
                repaired = repair_config(raw, sources)
                if repaired is not None:
                    try:
                        _add(parse_scene_text(repaired, source="model"))
                    except Exception:
                        pass

        # Fill remaining slots.
        attempts = 0
        while len(pool) < k and attempts < max_attempts:
            attempts += 1
            if state.scene is not None:
                _add(mutate_scene(rng, state.scene, sources, div))
            else:
                _add(generate_scene(rng, sources, div))

    else:  # mutate
        assert state.scene is not None, "mutate mode requires a current scene"
        attempts = 0
        while len(pool) < k and attempts < max_attempts:
            attempts += 1
            _add(mutate_scene(rng, state.scene, sources, div))

    return pool


def advance(
    state: PursuitState,
    winner: Scene,
    winner_score: float,
) -> PursuitState:
    """Return the next state after a generation completes successfully.

    Increments ``generation``; resets stagnation and updates ``best_score``
    when the winner improves on the running best.
    """
    improved = winner_score > state.best_score
    return PursuitState(
        scene=winner,
        best_score=winner_score if improved else state.best_score,
        stagnation=0 if improved else state.stagnation + 1,
        generation=state.generation + 1,
    )


@dataclass(frozen=True)
class GenerationRecord:
    """Log entry for one generation of the pursuit loop.

    ``winner_index`` is -1 when all candidate renders failed.
    ``wav_path`` is None until the orchestration layer fills it in (or when
    winner_index is -1).
    """

    t0: float
    t1: float
    descriptor: str
    tempo: float
    energy: float
    mode: Mode
    candidate_texts: tuple[str, ...]
    scores: tuple[float, ...]
    winner_index: int
    cps: float
    wav_path: str | None

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for the session log."""
        return {
            "t0": self.t0,
            "t1": self.t1,
            "descriptor": self.descriptor,
            "tempo": self.tempo,
            "energy": self.energy,
            "mode": self.mode,
            "candidate_texts": list(self.candidate_texts),
            "scores": list(self.scores),
            "winner_index": self.winner_index,
            "cps": self.cps,
            "wav_path": self.wav_path,
        }
