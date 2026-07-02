"""wav2tidal — learn musical style from WAV corpora and drive TidalCycles live.

Package layout (see specs/001-corpus-to-live-pipeline/plan.md):

- ``core``     pure functions, no IO (constitution I) — the whole test surface
- ``io``       impure edges (WAV, SuperDirt banks, GHCi, PipeWire, model load)
- ``train``    ByT5 seq2seq fine-tune + grammar-constrained decoding
- ``pipeline`` stage orchestration (ingest, dataset, train, eval, generate, live)
- ``cli``      subcommand entry point
"""

__version__ = "0.1.0"
