"""Session replay — compose analysis, pursuit, and assembly (US3-4).

Thin composition layer: reads an input WAV, runs the pursuit engine over
its analysis windows, stitches the winning scene renders into one output
timeline, and writes a session log.  All heavy callables (render, embed,
propose) can be injected for CI/test runs without SuperCollider or torch.

References:
  specs/001-corpus-to-live-pipeline/us3-live-loop-design.md
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from ..core.pattern.validate import Sources
from ..core.pursuit import PursuitConfig
from ..core.render.assemble import assemble
from ..io.embedder import Embedder, make_embedder
from ..io.wav import read_wav
from .analysis import analyze_wav
from .pursuit import (
    EmbedFn,
    RenderFn,
    default_render,
    make_proposer,
    run_pursuit,
    write_session_log,
)

log = logging.getLogger(__name__)

# Candidate renders are scored at 44100 Hz (pursuit.py _SCORE_SR); the
# assembly reads them back at the same rate for consistency.
_ASSEMBLY_SR = 44100

# Match the repo convention: _normalize_wav in io/superdirt.py uses PCM_24.
_PCM_SUBTYPE = "PCM_24"


def replay(
    input_wav: str | Path,
    out_wav: str | Path,
    *,
    checkpoint: str | Path | None = None,
    embedder_kind: str = "clap",
    cfg: PursuitConfig | None = None,
    seed: int = 0,
    work_dir: Path | None = None,
    log_path: Path | None = None,
    render: RenderFn | None = None,
    embed: EmbedFn | None = None,
    propose: Callable[[str], str | None] | None = None,
    embedder: Embedder | None = None,
) -> Path:
    """Mix WAV in → reinterpretation WAV out + session log (US3-4).

    1. Analyse ``input_wav`` into windows (default: CLAP or NullEmbedder).
    2. Run the propose / shadow-audition / select pursuit loop.
    3. Stitch winning scene renders into one output timeline via
       :func:`~wav2tidal.core.render.assemble.assemble`.
    4. Write ``out_wav`` as PCM_24 stereo at 44100 Hz and a JSON session
       log alongside it.

    All heavy callables — ``render``, ``embed``, ``propose`` — can be
    injected so tests run end-to-end without SuperCollider or torch.

    Parameters
    ----------
    input_wav:
        Path to the input mix WAV (any rate; resampled internally).
    out_wav:
        Output path for the assembled reinterpretation.
    checkpoint:
        ByT5 checkpoint directory for the neural proposer; ``None`` runs
        the mutation-only path (no ByT5).
    embedder_kind:
        ``"clap"`` or ``"null"``; ignored when ``embedder`` is injected.
    cfg:
        Pursuit engine configuration; defaults to ``PursuitConfig()``.
    seed:
        RNG seed for deterministic candidate generation.
    work_dir:
        Directory for candidate render files; defaults to
        ``out_wav.parent / (out_wav.stem + "_work")``.
    log_path:
        Session log path; defaults to ``out_wav.with_suffix(".session.json")``.
    render, embed, propose, embedder:
        Injectable callables / embedder for testing.  When provided they
        bypass the production defaults.

    Returns
    -------
    Path
        Resolved ``out_wav`` path.

    Raises
    ------
    ValueError
        When ``input_wav`` is too short (< half a window ≈ 2 s) or silent.
    """
    out_wav = Path(out_wav)

    # 1. Embedder — injection keeps tests torch-free
    emb: Embedder = embedder if embedder is not None else make_embedder(embedder_kind)

    # 2. Analyse
    win_list = analyze_wav(input_wav, embedder=emb)
    if not win_list:
        raise ValueError(
            f"No analysis windows produced from {input_wav!r}. "
            "The file may be too short (< 2 s), silent, or unreadable."
        )

    # 3. Sources — full synth palette; no sample banks needed (shadow
    #    audition strips the layer so banks are never referenced).
    sources = Sources()

    # 4. Callables — use injected fakes when provided, else production defaults
    _render: RenderFn = render if render is not None else default_render(sources)
    _embed: EmbedFn = embed if embed is not None else emb.embed
    _propose: Callable[[str], str | None] | None = propose
    if _propose is None and checkpoint is not None:
        _propose = make_proposer(checkpoint)

    # 5. Work directory for candidate renders
    _work_dir: Path = (
        work_dir if work_dir is not None else out_wav.parent / (out_wav.stem + "_work")
    )

    # 6. Pursuit loop — one generation per analysis window
    records = run_pursuit(
        win_list,
        sources,
        _work_dir,
        render=_render,
        embed=_embed,
        propose=_propose,
        cfg=cfg if cfg is not None else PursuitConfig(),
        seed=seed,
    )

    # 7. Assemble — place each winner at its generation's t0; skip failures.
    #    The output timeline mirrors the input; the DJ-set delay happens at
    #    playback, not in the file (us3-live-loop-design § "file-out first").
    placements: list[tuple[float, np.ndarray]] = []
    for record in records:
        if record.winner_index >= 0 and record.wav_path is not None:
            loaded = read_wav(record.wav_path, _ASSEMBLY_SR)
            placements.append((record.t0, loaded.y))

    total_seconds = win_list[-1].t1
    assembled = assemble(placements, total_seconds, _ASSEMBLY_SR)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), assembled, _ASSEMBLY_SR, subtype=_PCM_SUBTYPE)

    # 8. Session log
    _log_path: Path = (
        log_path if log_path is not None else out_wav.with_suffix(".session.json")
    )
    write_session_log(records, _log_path)

    # One-line summary for operators
    n_ok = sum(1 for r in records if r.winner_index >= 0)
    n_fail = len(records) - n_ok
    log.info(
        "replay: %d generations (%d failed); output %.2f s → %s",
        len(records),
        n_fail,
        total_seconds,
        out_wav,
    )

    return out_wav
