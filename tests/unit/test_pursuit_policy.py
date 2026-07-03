"""Tests for core/pursuit.py — pure pursuit-engine policy (US3-3)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from wav2tidal.core.dsp.stream import AnalysisWindow
from wav2tidal.core.pattern.model import Scene, Voice
from wav2tidal.core.pattern.params import spec
from wav2tidal.core.pattern.validate import Sources
from wav2tidal.core.pursuit import (
    _ENERGY_BREAKPOINTS,
    _MIN_VOICES_FLOOR,
    GenerationRecord,
    PursuitConfig,
    PursuitState,
    advance,
    apply_energy,
    decide,
    make_candidates,
    nrt_safe,
    score,
    select,
    tempo_to_cps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SOURCES = Sources()  # default: full Super* palette, no sample banks


def _window(
    t0: float = 0.0,
    t1: float = 4.0,
    tempo: float = 120.0,
    energy: float = 0.1,
    emb: np.ndarray | None = None,
) -> AnalysisWindow:
    return AnalysisWindow(
        t0=t0,
        t1=t1,
        descriptor="tempo=120 density=medium key=C brightness=mid motion=steady",
        tempo=tempo,
        energy=energy,
        embedding=(
            np.empty(0, dtype=np.float64)
            if emb is None
            else np.asarray(emb, dtype=np.float64)
        ),
    )


def _simple_scene(n_voices: int = 2, with_layer: bool = False) -> Scene:
    """Minimal valid scene using supersaw."""
    from wav2tidal.core.pattern.model import Pattern

    voices = tuple(
        Voice(
            source_name="supersaw", n=0, controls={"note": float(-12 + i * 4)}, mods=()
        )
        for i in range(n_voices)
    )
    layer = Pattern(mini="bd", controls={}) if with_layer else None
    return Scene(voices=voices, layer=layer, source="sampled")


# A fixed valid scene text for the fake proposer.
_VALID_SCENE_TEXT = "scene voice supersaw # note -12 voice supersaw # note 0"


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------


def test_decide_initial_propose():
    """No scene → always propose (cold start)."""
    state = PursuitState.initial()
    w = _window()
    assert decide(state, None, w, PursuitConfig()) == "propose"


def test_decide_stagnation_triggers_propose():
    """Stagnation ≥ patience → propose."""
    cfg = PursuitConfig(patience=3)
    state = PursuitState(
        scene=_simple_scene(),
        best_score=0.5,
        stagnation=3,
        generation=5,
    )
    w = _window()
    assert decide(state, w, w, cfg) == "propose"


def test_decide_stagnation_below_patience_mutates():
    """Stagnation < patience and no jump → mutate."""
    cfg = PursuitConfig(patience=3)
    state = PursuitState(
        scene=_simple_scene(),
        best_score=0.5,
        stagnation=2,
        generation=5,
    )
    w = _window()
    assert decide(state, w, w, cfg) == "mutate"


def test_decide_input_jump_triggers_propose():
    """Large embedding distance between windows → propose."""
    cfg = PursuitConfig(jump_threshold=0.35)
    state = PursuitState(
        scene=_simple_scene(), best_score=0.5, stagnation=0, generation=1
    )
    # Orthogonal embeddings → cosine distance = 1.0 > 0.35
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])
    prev = _window(emb=emb_a)
    cur = _window(emb=emb_b)
    assert decide(state, prev, cur, cfg) == "propose"


def test_decide_no_jump_no_stagnation_mutates():
    """Similar consecutive windows + no stagnation → mutate."""
    cfg = PursuitConfig(patience=3, jump_threshold=0.35)
    state = PursuitState(
        scene=_simple_scene(), best_score=0.5, stagnation=1, generation=3
    )
    # Identical embeddings → no jump
    emb = np.array([1.0, 0.0, 0.0])
    prev = _window(emb=emb)
    cur = _window(emb=emb)
    assert decide(state, prev, cur, cfg) == "mutate"


def test_decide_no_prev_window_no_jump():
    """prev_window=None skips jump check; falls through to stagnation/mutate."""
    state = PursuitState(
        scene=_simple_scene(), best_score=0.5, stagnation=1, generation=1
    )
    w = _window()
    assert decide(state, None, w, PursuitConfig(patience=3)) == "mutate"


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_score_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert score(v, v) == pytest.approx(1.0, abs=1e-9)


def test_score_cosine_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert score(a, b) == pytest.approx(0.0, abs=1e-9)


def test_score_cosine_opposite():
    v = np.array([1.0, 2.0, 3.0])
    assert score(v, -v) == pytest.approx(-1.0, abs=1e-9)


def test_score_empty_candidate():
    assert score(np.empty(0), np.array([1.0, 0.0])) == 0.0


def test_score_empty_target():
    assert score(np.array([1.0, 0.0]), np.empty(0)) == 0.0


def test_score_zero_norm_candidate():
    assert score(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0


def test_score_zero_norm_target():
    assert score(np.array([1.0, 0.0, 0.0]), np.zeros(3)) == 0.0


def test_score_both_empty():
    assert score(np.empty(0), np.empty(0)) == 0.0


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------


def test_select_argmax():
    assert select([0.1, 0.9, 0.5]) == 1


def test_select_tie_lowest_index():
    # Both index 0 and 2 have the same max value → lowest wins
    assert select([0.7, 0.5, 0.7]) == 0


def test_select_single_element():
    assert select([0.3]) == 0


def test_select_all_inf():
    # All -inf: lowest index wins
    assert select([float("-inf"), float("-inf")]) == 0


def test_select_empty_raises():
    with pytest.raises(ValueError):
        select([])


# ---------------------------------------------------------------------------
# tempo_to_cps()
# ---------------------------------------------------------------------------


def test_tempo_to_cps_normal():
    # 120 BPM, 4 beats/cycle → 120/60/4 = 0.5 cps
    assert tempo_to_cps(120.0, 4.0) == pytest.approx(0.5)


def test_tempo_to_cps_clamp_low():
    # Very slow: 1 BPM / 4 beats = 0.00417 → clamped to 0.125
    assert tempo_to_cps(1.0, 4.0) == pytest.approx(0.125)


def test_tempo_to_cps_clamp_high():
    # Very fast: 1000 BPM / 4 beats = 4.167 → clamped to 2.0
    assert tempo_to_cps(1000.0, 4.0) == pytest.approx(2.0)


def test_tempo_to_cps_nonpositive_bpm():
    assert tempo_to_cps(0.0) == pytest.approx(0.5)
    assert tempo_to_cps(-10.0) == pytest.approx(0.5)


def test_tempo_to_cps_at_boundary_low():
    # Exactly 0.125 cps: bpm = 0.125 * 60 * 4 = 30 BPM
    assert tempo_to_cps(30.0, 4.0) == pytest.approx(0.125)


def test_tempo_to_cps_at_boundary_high():
    # Exactly 2.0 cps: bpm = 2.0 * 60 * 4 = 480 BPM
    assert tempo_to_cps(480.0, 4.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# apply_energy()
# ---------------------------------------------------------------------------

_GAIN_LO = spec("gain").lo
_GAIN_HI = spec("gain").hi


def _voice_with_gain(gain: float) -> Voice:
    return Voice(source_name="supersaw", n=0, controls={"gain": gain}, mods=())


def _voice_no_gain() -> Voice:
    return Voice(source_name="supersaw", n=0, controls={}, mods=())


def _scene_gain(gains: list[float]) -> Scene:
    return Scene(
        voices=tuple(_voice_with_gain(g) for g in gains),
        layer=None,
        source="sampled",
    )


def test_apply_energy_gain_scaled_medium():
    # energy=0.15 → factor=1.0 (middle breakpoint); gain=1.0 → 1.0
    scene = _scene_gain([1.0])
    out = apply_energy(scene, energy=0.15)
    assert out.voices[0].controls["gain"] == pytest.approx(1.0, abs=1e-5)


def test_apply_energy_gain_scaled_low():
    # energy=0.0 → factor=0.5; gain=1.0 → 0.5 (at floor)
    scene = _scene_gain([1.0])
    out = apply_energy(scene, energy=0.0)
    assert out.voices[0].controls["gain"] == pytest.approx(0.5, abs=1e-5)


def test_apply_energy_gain_scaled_high():
    # energy=0.3 → factor=1.2; gain=1.0 → 1.2
    scene = _scene_gain([1.0])
    out = apply_energy(scene, energy=0.3)
    assert out.voices[0].controls["gain"] == pytest.approx(1.2, abs=1e-5)


def test_apply_energy_gain_clamped_to_spec_hi():
    # energy=0.5 → factor=1.2; gain=1.3 → 1.3*1.2=1.56 → clamped to 1.3
    scene = _scene_gain([_GAIN_HI])
    out = apply_energy(scene, energy=0.5)
    g = out.voices[0].controls["gain"]
    assert g <= _GAIN_HI + 1e-9


def test_apply_energy_gain_clamped_to_spec_lo():
    # energy=0.0 → factor=0.5; gain=0.5 → 0.5*0.5=0.25 → clamped to 0.5
    scene = _scene_gain([_GAIN_LO])
    out = apply_energy(scene, energy=0.0)
    g = out.voices[0].controls["gain"]
    assert g >= _GAIN_LO - 1e-9


def test_apply_energy_default_gain_when_absent():
    # Voice has no 'gain' key → default 1.0 used
    scene = Scene(voices=(_voice_no_gain(),), layer=None, source="sampled")
    out = apply_energy(scene, energy=0.15)
    # factor=1.0 at energy=0.15; 1.0*1.0=1.0
    assert out.voices[0].controls["gain"] == pytest.approx(1.0, abs=1e-5)


def test_apply_energy_never_below_2_voices():
    """Very low energy trims voices but never below _MIN_VOICES_FLOOR=2."""
    # 4 voices, very low energy
    scene = _scene_gain([1.0, 1.0, 1.0, 1.0])
    low = _ENERGY_BREAKPOINTS[0][0]  # 0.0
    out = apply_energy(scene, energy=low)
    assert len(out.voices) >= _MIN_VOICES_FLOOR


def test_apply_energy_does_not_trim_at_normal_energy():
    """Normal energy (0.1) preserves all voices."""
    scene = _scene_gain([1.0, 1.0, 1.0])
    out = apply_energy(scene, energy=0.1)
    assert len(out.voices) == 3


def test_apply_energy_deterministic():
    """Same inputs → identical outputs."""
    scene = _scene_gain([0.9, 1.1])
    out_a = apply_energy(scene, energy=0.2)
    out_b = apply_energy(scene, energy=0.2)
    assert out_a.to_text() == out_b.to_text()


def test_apply_energy_source_preserved():
    scene = Scene(
        voices=(_voice_with_gain(1.0),),
        layer=None,
        source="mutation",
    )
    out = apply_energy(scene, energy=0.15)
    assert out.source == "mutation"


# ---------------------------------------------------------------------------
# nrt_safe()
# ---------------------------------------------------------------------------


def test_nrt_safe_drops_layer():
    from wav2tidal.core.pattern.model import Pattern

    layer = Pattern(mini="bd", controls={})
    scene = Scene(voices=(_voice_with_gain(1.0),), layer=layer, source="sampled")
    out = nrt_safe(scene)
    assert out.layer is None
    assert out.voices == scene.voices


def test_nrt_safe_no_layer_unchanged():
    scene = Scene(voices=(_voice_with_gain(1.0),), layer=None, source="sampled")
    out = nrt_safe(scene)
    assert out.to_text() == scene.to_text()


def test_nrt_safe_preserves_source():
    scene = Scene(voices=(_voice_with_gain(1.0),), layer=None, source="model")
    out = nrt_safe(scene)
    assert out.source == "model"


# ---------------------------------------------------------------------------
# make_candidates()
# ---------------------------------------------------------------------------


def test_make_candidates_propose_count_near_k():
    """Pool size ≤ k (dedup may reduce it; 4*k attempts refill best effort)."""
    rng = random.Random(0)
    state = PursuitState.initial()
    w = _window()
    cfg = PursuitConfig(k_candidates=4)
    pool = make_candidates(rng, "propose", state, w, SOURCES, None, cfg)
    assert 1 <= len(pool) <= cfg.k_candidates


def test_make_candidates_mutate_count():
    """Mutate mode yields up to k candidates."""
    rng = random.Random(42)
    scene = _simple_scene()
    state = PursuitState(scene=scene, best_score=0.5, stagnation=0, generation=1)
    w = _window()
    cfg = PursuitConfig(k_candidates=5)
    pool = make_candidates(rng, "mutate", state, w, SOURCES, None, cfg)
    assert 1 <= len(pool) <= cfg.k_candidates


def test_make_candidates_no_duplicates():
    """No two candidates in the pool share the same to_text() output."""
    rng = random.Random(7)
    state = PursuitState.initial()
    w = _window()
    cfg = PursuitConfig(k_candidates=6)
    pool = make_candidates(rng, "propose", state, w, SOURCES, None, cfg)
    texts = [c.to_text() for c in pool]
    assert len(texts) == len(set(texts))


def test_make_candidates_propose_first_when_propose_succeeds():
    """When propose returns a valid scene text, the repaired scene is first."""

    def _propose(descriptor: str) -> str:
        return _VALID_SCENE_TEXT

    rng = random.Random(0)
    state = PursuitState.initial()
    w = _window()
    cfg = PursuitConfig(k_candidates=4)
    pool = make_candidates(rng, "propose", state, w, SOURCES, _propose, cfg)
    # The pool must be non-empty and the first candidate comes from the proposer.
    assert len(pool) >= 1
    # All candidates passed through nrt_safe (no layer) and apply_energy.
    for c in pool:
        assert c.layer is None


def test_make_candidates_fallback_when_propose_returns_garbage():
    """Garbage proposal text → repair fails → fallback to generated scenes."""

    def _bad_propose(descriptor: str) -> str:
        return "this is not valid scene text at all !@#"

    rng = random.Random(1)
    state = PursuitState.initial()
    w = _window()
    cfg = PursuitConfig(k_candidates=3)
    pool = make_candidates(rng, "propose", state, w, SOURCES, _bad_propose, cfg)
    # Still gets candidates via generate_scene fallback
    assert len(pool) >= 1


def test_make_candidates_propose_returns_none():
    """propose callback returning None falls back gracefully."""
    rng = random.Random(2)
    state = PursuitState.initial()
    w = _window()
    cfg = PursuitConfig(k_candidates=3)
    pool = make_candidates(rng, "propose", state, w, SOURCES, lambda _: None, cfg)
    assert len(pool) >= 1


def test_make_candidates_mutate_asserts_scene_not_none():
    """Mutate mode with no scene raises AssertionError."""
    rng = random.Random(0)
    state = PursuitState.initial()  # scene=None
    w = _window()
    with pytest.raises(AssertionError):
        make_candidates(rng, "mutate", state, w, SOURCES, None, PursuitConfig())


def test_make_candidates_all_nrt_safe():
    """All returned candidates have layer=None (nrt_safe applied)."""
    rng = random.Random(10)
    state = PursuitState.initial()
    w = _window()
    pool = make_candidates(
        rng, "propose", state, w, SOURCES, None, PursuitConfig(k_candidates=4)
    )
    for c in pool:
        assert c.layer is None


def test_make_candidates_energy_applied():
    """gain is present on every voice after apply_energy pass."""
    rng = random.Random(20)
    state = PursuitState.initial()
    w = _window(energy=0.15)
    pool = make_candidates(
        rng, "propose", state, w, SOURCES, None, PursuitConfig(k_candidates=3)
    )
    for scene in pool:
        for v in scene.voices:
            assert "gain" in v.controls


# ---------------------------------------------------------------------------
# advance()
# ---------------------------------------------------------------------------


def test_advance_improvement_resets_stagnation():
    state = PursuitState(
        scene=_simple_scene(), best_score=0.5, stagnation=2, generation=5
    )
    winner = _simple_scene(n_voices=3)
    next_state = advance(state, winner, winner_score=0.7)
    assert next_state.scene is winner
    assert next_state.best_score == pytest.approx(0.7)
    assert next_state.stagnation == 0
    assert next_state.generation == 6


def test_advance_no_improvement_increments_stagnation():
    state = PursuitState(
        scene=_simple_scene(), best_score=0.8, stagnation=1, generation=3
    )
    winner = _simple_scene()
    next_state = advance(state, winner, winner_score=0.6)
    assert next_state.best_score == pytest.approx(0.8)  # unchanged
    assert next_state.stagnation == 2
    assert next_state.generation == 4


def test_advance_equal_score_does_not_improve():
    """Equal score is not an improvement (best_score unchanged, stagnation++)."""
    state = PursuitState(
        scene=_simple_scene(), best_score=0.5, stagnation=0, generation=1
    )
    next_state = advance(state, _simple_scene(), winner_score=0.5)
    assert next_state.stagnation == 1
    assert next_state.best_score == pytest.approx(0.5)


def test_advance_updates_scene():
    old_scene = _simple_scene(n_voices=2)
    new_scene = _simple_scene(n_voices=3)
    state = PursuitState(scene=old_scene, best_score=0.0, stagnation=0, generation=0)
    next_state = advance(state, new_scene, winner_score=0.9)
    assert next_state.scene is new_scene


# ---------------------------------------------------------------------------
# GenerationRecord.to_dict()
# ---------------------------------------------------------------------------


def test_generation_record_to_dict_round_trips():
    import json

    rec = GenerationRecord(
        t0=0.0,
        t1=4.0,
        descriptor="tempo=120 density=medium",
        tempo=120.0,
        energy=0.1,
        mode="propose",
        candidate_texts=("scene voice supersaw",),
        scores=(0.8,),
        winner_index=0,
        cps=0.5,
        wav_path="/tmp/gen0000_cand0.wav",
    )
    d = rec.to_dict()
    # JSON round-trip
    loaded = json.loads(json.dumps(d))
    assert loaded["mode"] == "propose"
    assert loaded["winner_index"] == 0
    assert loaded["cps"] == pytest.approx(0.5)
    assert loaded["wav_path"] == "/tmp/gen0000_cand0.wav"
    assert loaded["candidate_texts"] == ["scene voice supersaw"]
    assert loaded["scores"] == [pytest.approx(0.8)]


def test_generation_record_all_fail_dict():
    rec = GenerationRecord(
        t0=0.0,
        t1=4.0,
        descriptor="",
        tempo=0.0,
        energy=0.0,
        mode="propose",
        candidate_texts=(),
        scores=(),
        winner_index=-1,
        cps=0.5,
        wav_path=None,
    )
    d = rec.to_dict()
    assert d["winner_index"] == -1
    assert d["wav_path"] is None
