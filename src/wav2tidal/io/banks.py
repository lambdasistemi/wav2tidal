"""SuperDirt sample-bank writer (T020).

Writes slices to ``banks/<bank>/NN_*.wav`` in the layout SuperDirt loads
natively (research R5, verified against DirtSoundLibrary.sc): the folder
basename is the sample name, files are sorted alphabetically, and that
0-based order is the ``:n`` selector. The ``NN_`` numeric prefix pins the
order so re-ingest is stable (FR-006).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .wav import write_wav


def slice_filename(index: int, source_stem: str) -> str:
    """Zero-padded, order-stable filename; index defines the SuperDirt :n."""
    safe = "".join(c if c.isalnum() else "-" for c in source_stem)[:32]
    return f"{index:04d}_{safe}.wav"


def write_bank(
    banks_root: Path,
    bank: str,
    slices: list[tuple[np.ndarray, str]],
    sr: int,
) -> list[Path]:
    """Write ``slices`` (audio, source_stem) into ``banks_root/bank``.

    Returns the written paths in bank order. The bank directory is rewritten
    wholesale for this bank's set of slices so alphabetical order == the
    intended :n index.
    """
    bank_dir = banks_root / bank
    bank_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, (audio, stem) in enumerate(slices):
        path = bank_dir / slice_filename(index, stem)
        write_wav(path, audio, sr)
        written.append(path)
    return written
