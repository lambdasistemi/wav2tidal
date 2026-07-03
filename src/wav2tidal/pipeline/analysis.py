"""Input analysis stream — thin IO wrapper for the pursuit loop (US3-1).

Reads a WAV file and returns a list of AnalysisWindow objects ready for the
pursuit engine.  Audio IO and embedder construction happen here; the pure
sliding-window logic lives in core.dsp.stream.
"""

from __future__ import annotations

from pathlib import Path

from ..core.dsp.stream import AnalysisWindow, windows
from ..io.embedder import Embedder, NullEmbedder
from ..io.wav import read_wav


def analyze_wav(
    path: str | Path,
    *,
    target_sr: int = 48000,
    window_s: float = 4.0,
    hop_s: float = 2.0,
    hop_length: int = 512,
    embedder: Embedder | None = None,
    descriptor_version: int = 1,
) -> list[AnalysisWindow]:
    """Read ``path``, slide the analysis window, and return AnalysisWindow objects.

    ``embedder`` defaults to ``NullEmbedder`` (hand-crafted DSP descriptors only,
    no neural embedding, no torch dependency) — the offline/CI-safe path.
    Pass a ``ClapEmbedder`` instance to add CLAP embeddings.

    ``descriptor_version`` is forwarded to ``windows()``; see its docstring.
    Default is 1 until the retrain ships (#68).
    """
    emb: Embedder = embedder if embedder is not None else NullEmbedder()

    loaded = read_wav(path, target_sr)

    def _embed(y, sr):
        return emb.embed(y, sr)

    return windows(
        loaded.y,
        loaded.sr,
        window_s=window_s,
        hop_s=hop_s,
        hop_length=hop_length,
        embed=_embed,
        descriptor_version=descriptor_version,
    )
