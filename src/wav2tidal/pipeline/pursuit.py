"""Pursuit-engine orchestration: shadow renders, scoring, session log (US3-3).

Thin IO layer over the pure policy in ``core/pursuit.py``.  Heavy
callables — NRT rendering, embedding, ByT5 proposal — are injected so CI
runs end-to-end with fakes (no SuperCollider, no torch required).

References:
  specs/001-corpus-to-live-pipeline/us3-live-loop-design.md
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..core.dsp.features import chroma_sequence, mean_chroma, modulation_spectrum
from ..core.pattern.dirt import scene_plan
from ..core.pattern.model import Scene
from ..core.pattern.validate import Sources
from ..core.pursuit import (
    GenerationRecord,
    PursuitConfig,
    PursuitState,
    advance,
    combined_score,
    decide,
    make_candidates,
    select,
    seq_similarity,
    tempo_to_cps,
    vec_similarity,
)
from ..io.superdirt import nrt_render_scene
from ..io.wav import read_wav

log = logging.getLogger(__name__)

# Render callable signature:
#   (scene, out_wav, duration_s, cps, seed) → Path
RenderFn = Callable[[Scene, Path, float, float, int], Path]

# Embed callable signature:
#   (audio_array_float32, sample_rate) → embedding or None
EmbedFn = Callable[[np.ndarray, int], np.ndarray | None]

# Sample rate used when reading back candidate renders for scoring.
# 48 kHz because ClapEmbedder.embed requires CLAP_SR audio — read_wav
# resamples the 44.1 kHz NRT renders; targets from analyze_wav are 48 kHz too.
_SCORE_SR = 48000


def default_render(sources: Sources) -> RenderFn:
    """Return a ``RenderFn`` closure wrapping ``scene_plan`` + ``nrt_render_scene``.

    This is the production render path; inject fakes in tests.
    """

    def _render(
        scene: Scene, out_wav: Path, duration_s: float, cps: float, seed: int
    ) -> Path:
        plan = scene_plan(scene, sources, duration_s, cps)
        return nrt_render_scene(plan, out_wav, seed=seed)

    return _render


def run_pursuit(
    windows: list,  # list[AnalysisWindow]
    sources: Sources,
    out_dir: Path,
    *,
    render: RenderFn,
    embed: EmbedFn,
    propose: Callable[[str], str | None] | None = None,
    cfg: PursuitConfig | None = None,
    seed: int = 0,
) -> list[GenerationRecord]:
    """Run the propose / shadow-audition / select pursuit loop.

    One generation per analysis window:

    1. ``decide`` — choose propose or mutate mode.
    2. ``make_candidates`` — build the pool.
    3. Render all candidates concurrently (``ThreadPoolExecutor``,
       ``max_workers ≤ 8``).  Per-candidate seeds are derived
       deterministically from ``(seed, generation, j)``.
    4. Read each rendered WAV, embed it, and cosine-score against
       ``window.embedding``.
    5. ``select`` winner; ``advance`` state; append ``GenerationRecord``.

    A candidate whose render raises is logged at WARNING and scored
    ``-inf``.  When every candidate in a generation fails, the record's
    ``winner_index`` is -1 and the pursuit state is left completely
    unchanged (the next generation will retry in the same mode).
    """
    if cfg is None:
        cfg = PursuitConfig()

    import random as _random

    rng = _random.Random(seed)
    state = PursuitState.initial()
    records: list[GenerationRecord] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_window = None

    for gen_i, window in enumerate(windows):
        cps = tempo_to_cps(window.tempo, cfg.beats_per_cycle)
        mode = decide(state, prev_window, window, cfg)
        candidates = make_candidates(rng, mode, state, window, sources, propose, cfg)

        n_cands = len(candidates)
        n_workers = min(n_cands, 8) if n_cands > 0 else 1

        # Bind gen_i and cfg in the closure explicitly to avoid late-binding.
        def _render_cand(
            j: int,
            scene: Scene,
            _gen_i: int = gen_i,
            _seed: int = seed,
            _cps: float = cps,
            _dur: float = cfg.scene_duration_s,
        ) -> Path:
            out_wav = out_dir / f"gen{_gen_i:04d}_cand{j}.wav"
            cand_seed = _seed ^ (_gen_i * 997 + j * 31)
            return render(scene, out_wav, _dur, _cps, cand_seed)

        wav_paths: list[Path | None] = [None] * n_cands
        scores_list: list[float] = [float("-inf")] * n_cands

        if n_cands > 0:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                future_to_j = {
                    executor.submit(_render_cand, j, scene): j
                    for j, scene in enumerate(candidates)
                }
                for fut in concurrent.futures.as_completed(future_to_j):
                    j = future_to_j[fut]
                    try:
                        wav_paths[j] = fut.result()
                    except Exception as exc:
                        log.warning("Render failed gen=%d cand=%d: %s", gen_i, j, exc)

            _empty = np.empty(0, dtype=np.float64)
            # Score each successfully rendered candidate.
            for j, wav_path in enumerate(wav_paths):
                if wav_path is None:
                    continue
                try:
                    loaded = read_wav(wav_path, _SCORE_SR)
                    raw_emb = embed(loaded.y, loaded.sr)
                    cand_emb = (
                        np.asarray(raw_emb, dtype=np.float64)
                        if raw_emb is not None
                        else _empty
                    )
                    cand_chroma = mean_chroma(loaded.y, loaded.sr)
                    cand_seq = chroma_sequence(loaded.y, loaded.sr)
                    cand_modspec = modulation_spectrum(loaded.y, loaded.sr)
                    target_chroma = getattr(window, "chroma", _empty)
                    target_seq = getattr(window, "chroma_seq", _empty)
                    target_modspec = getattr(window, "modspec", _empty)
                    scores_list[j] = combined_score(
                        [
                            (vec_similarity(cand_emb, window.embedding), cfg.w_timbre),
                            (vec_similarity(cand_chroma, target_chroma), cfg.w_harmony),
                            (seq_similarity(cand_seq, target_seq), cfg.w_harmony_seq),
                            (
                                vec_similarity(cand_modspec, target_modspec),
                                cfg.w_modspec,
                            ),
                        ]
                    )
                except Exception as exc:
                    log.warning("Score failed gen=%d cand=%d: %s", gen_i, j, exc)

        candidate_texts = tuple(c.to_text() for c in candidates)
        all_fail = not n_cands or all(s == float("-inf") for s in scores_list)

        if all_fail:
            records.append(
                GenerationRecord(
                    t0=window.t0,
                    t1=window.t1,
                    descriptor=window.descriptor,
                    tempo=window.tempo,
                    energy=window.energy,
                    mode=mode,
                    candidate_texts=candidate_texts,
                    scores=tuple(scores_list),
                    winner_index=-1,
                    cps=cps,
                    wav_path=None,
                )
            )
            # State left completely unchanged (next generation retries same mode).
            prev_window = window
            continue

        winner_idx = select(scores_list)
        winner_scene = candidates[winner_idx]
        winner_score = scores_list[winner_idx]
        state = advance(state, winner_scene, winner_score)

        records.append(
            GenerationRecord(
                t0=window.t0,
                t1=window.t1,
                descriptor=window.descriptor,
                tempo=window.tempo,
                energy=window.energy,
                mode=mode,
                candidate_texts=candidate_texts,
                scores=tuple(scores_list),
                winner_index=winner_idx,
                cps=cps,
                wav_path=(
                    str(wav_paths[winner_idx]) if wav_paths[winner_idx] else None
                ),
            )
        )
        prev_window = window

    return records


def write_session_log(records: list[GenerationRecord], path: Path) -> None:
    """Write the session log as a JSON array of generation records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in records], indent=2))


def make_proposer(
    checkpoint: str | Path, max_new_tokens: int = 768
) -> Callable[[str], str]:
    """Load a ByT5 checkpoint once; return a greedy descriptor → config-text closure.

    ``torch`` and ``transformers`` are imported *inside* this function so
    the module never touches them at import time — mirroring
    ``train/trainer.py § generate_config_text`` (the production path).
    """
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    checkpoint = str(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device).eval()

    def _propose(descriptor: str) -> str:
        enc = tok([descriptor], return_tensors="pt").to(device)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device == "cuda"),
            ),
        ):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.batch_decode(gen, skip_special_tokens=True)[0]

    return _propose
