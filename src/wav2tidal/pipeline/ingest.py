"""Ingest orchestration (T022).

Corpus WAVs -> beat-sliced SuperDirt banks + style profile. Idempotent and
incremental via the corpus manifest (FR-006): unchanged files keep their
banks and descriptors; only new/changed files are reprocessed. Unreadable,
silent, or corrupt files are skipped and reported, never fatal (FR-001).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.config import IngestConfig
from ..core.descriptor.types import ProfileIndex, StyleDescriptor, assemble_descriptor
from ..core.dsp.features import slice_features, track_descriptors
from ..core.dsp.slice import slice_boundaries
from ..io import wav
from ..io.banks import write_bank
from ..io.embedder import Embedder, make_embedder
from ..io.storage import (
    Workspace,
    build_manifest,
    diff_manifest,
    load_manifest,
    save_manifest,
)
from . import profile as profile_io


@dataclass
class IngestReport:
    processed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    banks: list[str] = field(default_factory=list)
    n_slices: int = 0

    def summary(self) -> str:
        lines = [
            f"processed {len(self.processed)} file(s), "
            f"{len(self.banks)} bank(s), {self.n_slices} slice(s)",
        ]
        for p, r in self.skipped:
            lines.append(f"  skipped {p}: {r}")
        for p, r in self.warnings:
            lines.append(f"  warning {p}: {r}")
        return "\n".join(lines)


def discover_wavs(corpus: Path) -> list[Path]:
    return sorted(p for p in corpus.rglob("*.wav") if p.is_file())


def _unique_bank_name(stem: str, taken: set[str]) -> str:
    base = "".join(c if c.isalnum() else "" for c in stem).lower()[:24] or "bank"
    name, i = base, 1
    while name in taken:
        i += 1
        name = f"{base}{i}"
    taken.add(name)
    return name


def _rhythm_block(desc: dict) -> np.ndarray:
    # inject tempo/density that neural/spectral blocks encode weakly (R1)
    return np.array([desc["tempo_bpm"] / 300.0, desc["onset_rate"] / 20.0])


def ingest(
    corpus: Path,
    root: Path,
    cfg: IngestConfig,
    embedder: Embedder | None = None,
) -> IngestReport:
    ws = Workspace(root)
    ws.ensure()
    embedder = embedder or make_embedder(cfg.embedder)
    report = IngestReport()

    files = discover_wavs(corpus)
    new_manifest = build_manifest(files)
    old_manifest = load_manifest(ws.manifest_path)
    changed = set(diff_manifest(old_manifest, new_manifest))

    taken_banks: set[str] = set()
    track_items: list[tuple[str, StyleDescriptor]] = []
    tracks: list[dict] = []
    slices_meta: list[dict] = []

    for path in files:
        key = str(path)
        if key not in changed and key in old_manifest:
            report.skipped.append((key, "unchanged (incremental)"))
            continue
        try:
            loaded = wav.read_wav(path, cfg.target_sr)
        except wav.LibsndfileError as e:
            report.skipped.append((key, f"unreadable: {e}"))
            continue
        if wav.is_silent(loaded.y):
            report.skipped.append((key, "silent"))
            continue
        cf = wav.clip_fraction(loaded.y)
        if cf > cfg.clip_fraction:
            report.warnings.append((key, f"clipping fraction {cf:.4f}"))

        bank = _unique_bank_name(path.stem, taken_banks)
        boundaries = slice_boundaries(
            loaded.y,
            loaded.sr,
            hop_length=cfg.hop_length,
            strategy=cfg.slice_strategy,
            beats_per_slice=cfg.beats_per_slice,
            grid_subdivisions=cfg.grid_subdivisions,
            silence_top_db=cfg.silence_top_db,
        )
        clips = _cut(loaded.y, loaded.sr, boundaries, cfg.min_slice_seconds)
        if not clips:
            report.skipped.append((key, "no non-silent slices"))
            continue

        write_bank(ws.banks, bank, [(c, path.stem) for _, c in clips], loaded.sr)
        report.banks.append(bank)
        report.n_slices += len(clips)
        report.processed.append(key)

        track_id = new_manifest[key]["hash"][:16]
        td = track_descriptors(loaded.y, loaded.sr, cfg.hop_length)
        block = slice_features(loaded.y, loaded.sr, cfg.hop_length, cfg.n_mfcc)
        block["rhythm"] = _rhythm_block(td)
        emb = embedder.embed(loaded.y, loaded.sr)
        desc = assemble_descriptor(
            block, embedder.embedder_id, loaded.sr, embedding=emb
        )
        track_items.append((track_id, desc))
        tracks.append(
            {
                "id": track_id,
                "source_path": key,
                "source_stem": path.stem,
                "bank": bank,
                "n_slices": len(clips),
                "orig_sr": loaded.orig_sr,
                "orig_channels": loaded.orig_channels,
                **td,
            }
        )
        for i, (start, _clip) in enumerate(clips):
            slices_meta.append(
                {"track_id": track_id, "bank": bank, "index": i, "start_s": start}
            )

    if track_items:
        index = ProfileIndex.build(track_items)
        profile_io.save_profile(ws, tracks, slices_meta, index)
    save_manifest(ws.manifest_path, new_manifest)
    return report


def _cut(
    y: np.ndarray, sr: int, boundaries: np.ndarray, min_seconds: float
) -> list[tuple[float, np.ndarray]]:
    """Cut y at boundary times; drop slices shorter than min or silent."""
    out: list[tuple[float, np.ndarray]] = []
    for start_t, end_t in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end_t - start_t < min_seconds:
            continue
        a, b = int(start_t * sr), int(end_t * sr)
        clip = y[a:b]
        if clip.size == 0 or wav.is_silent(clip):
            continue
        out.append((float(start_t), clip))
    return out
