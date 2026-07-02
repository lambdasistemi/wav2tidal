"""Audio embedders (T021).

An embedder maps a mono float clip at the target rate to a fixed-length
neural embedding. Two implementations:

- ``NullEmbedder`` — no neural embedding; descriptors use hand-crafted DSP
  blocks only. Offline, CPU, dependency-free — the CI/test default.
- ``ClapEmbedder`` — LAION-CLAP ``laion/larger_clap_music`` (Apache-2.0,
  research R1), run on CPU, pinned by revision and forced offline at
  runtime. Loaded lazily so importing this module never pulls torch.

Both expose ``embedder_id`` (part of a descriptor's compatibility key) and
``embed(y, sr) -> np.ndarray | None``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# CLAP's expected input rate; ingestion must resample to this before embedding.
CLAP_SR = 48000
_CLAP_MODEL = "laion/larger_clap_music"


class Embedder(Protocol):
    embedder_id: str

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray | None: ...


class NullEmbedder:
    """Hand-crafted-only descriptors (no neural embedding)."""

    embedder_id = "null"

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray | None:
        return None


class ClapEmbedder:
    """LAION-CLAP music embedder, CPU, offline. Lazily loads the model."""

    def __init__(self, revision: str | None = None, model_name: str = _CLAP_MODEL):
        self._model_name = model_name
        self._revision = revision
        self.embedder_id = f"{model_name}@{revision or 'unpinned'}"
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        kwargs = {"local_files_only": True}
        if self._revision:
            kwargs["revision"] = self._revision
        self._model = ClapModel.from_pretrained(self._model_name, **kwargs).eval()
        self._processor = ClapProcessor.from_pretrained(self._model_name, **kwargs)
        self._torch = torch

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != CLAP_SR:
            raise ValueError(f"ClapEmbedder needs {CLAP_SR} Hz audio, got {sr}")
        self._ensure_loaded()
        inputs = self._processor(
            audios=np.asarray(y), sampling_rate=CLAP_SR, return_tensors="pt"
        )
        with self._torch.no_grad():
            feats = self._model.get_audio_features(**inputs)
        return feats.squeeze(0).cpu().numpy().astype(np.float64)


def make_embedder(kind: str, revision: str | None = None) -> Embedder:
    if kind == "null":
        return NullEmbedder()
    if kind == "clap":
        return ClapEmbedder(revision=revision)
    raise ValueError(f"unknown embedder: {kind!r} (expected 'null' or 'clap')")
