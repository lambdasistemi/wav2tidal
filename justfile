# wav2tidal — task recipes. Run inside `nix develop`.

# Run the full CI gate locally (mirrors GitHub CI exactly).
ci: format-check lint test

# Auto-format.
format:
    ruff check --fix src tests
    black src tests

# Check formatting without modifying (used by CI).
format-check:
    black --check src tests

# Lint.
lint:
    ruff check src tests

# Pure-core unit + property tests (CPU only, no GPU/audio — constitution IV).
test:
    pytest

# One-time pretrained-model fetch (CLAP), pinned by SHA into ./.hf_cache.
fetch-models:
    @echo "TODO(T021): fetch laion/larger_clap_music @ pinned SHA into ./.hf_cache"

# FR-018 GPU smoke test — run in the training shell: nix develop .#training -c just smoke-gpu
smoke-gpu:
    python tests/smoke/smoke_gpu.py

# Environment preflight for the live session (GHCi/SuperDirt/PipeWire/models).
doctor:
    @echo "TODO(T045): wav2tidal doctor preflight"

# Full reproducible pipeline: ingest -> dataset -> train -> eval (from configs).
pipeline:
    @echo "TODO(T048): chain ingest/dataset/train/eval from configs/"

# Docs.
build-docs:
    nix develop github:paolino/dev-assets?dir=mkdocs --quiet -c mkdocs build --strict

serve-docs:
    nix develop github:paolino/dev-assets?dir=mkdocs --quiet -c mkdocs serve

deploy-docs:
    nix develop github:paolino/dev-assets?dir=mkdocs --quiet -c mkdocs gh-deploy --force
