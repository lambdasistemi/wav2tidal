# wav2tidal

Learn musical style from a local WAV corpus and drive
[TidalCycles](https://tidalcycles.org) live.

wav2tidal ingests a collection of recordings into beat-sliced, SuperDirt-
loadable sample banks plus a style profile; synthesizes training data by
rendering random valid patterns through a deterministic slice-mixdown
renderer; fine-tunes a local model that maps a style descriptor to a
TidalCycles pattern; and runs a live evolutionary agent that plays through
SuperDirt and steers its patterns toward a chosen target style.

Everything runs locally and offline — the corpus never leaves the machine.

## Status

Early development. The design lives under
[`specs/001-corpus-to-live-pipeline/`](https://github.com/lambdasistemi/wav2tidal/tree/main/specs/001-corpus-to-live-pipeline):
spec, verified research, data model, contracts, quickstart, and the task
breakdown.

## Development

```bash
nix develop        # CPU dev shell (ingestion, embeddings, tests)
just ci            # lint + test, mirrors GitHub CI
```

Training uses a separate, hardware-gated shell:

```bash
nix develop .#training -c just smoke-gpu   # verify gfx1151 ROCm first (FR-018)
```

See the [quickstart](https://github.com/lambdasistemi/wav2tidal/blob/main/specs/001-corpus-to-live-pipeline/quickstart.md)
for the full corpus → live-session path.
