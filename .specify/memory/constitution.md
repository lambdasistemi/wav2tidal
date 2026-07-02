# wav2tidal Constitution

wav2tidal learns musical style from a WAV corpus and drives TidalCycles
live: ingest (slice WAVs into SuperDirt sample banks + extract features),
synthesize training data (pattern sampler + pure-Python slice mixdown
renderer), fine-tune a small local LLM (feature descriptor → Tidal
mini-notation), and run a live evolutionary agent (generate → play via
SuperDirt → capture → CLAP-score against target → mutate).

## Core Principles

### I. Pure Core, Impure Shell
DSP feature extraction, pattern generation, event scheduling, and the
mixdown renderer are pure functions over arrays and data structures.
IO — audio files, PipeWire capture, the ghci/Tidal subprocess, model
inference — lives at the edges and calls into the pure core. Pure
functions are tested directly; the impure layer is covered by local
E2E only.

### II. Patterns Are Validated Data
Generated Tidal code is data, not trusted text. Every pattern must pass
validation (parse / bounded complexity / known sample banks) before
being rendered for training or sent to a live Tidal session. The live
agent never sends unvalidated model output to SuperDirt.

### III. Reproducibility
Training runs and dataset generation are seeded and parameterized by
config files committed to the repo. Corpus WAVs, sample banks, datasets,
and model checkpoints are local artifacts — never committed. A run must
be reproducible from (config, seed, corpus path).

### IV. Tests Run Without GPU or Audio Hardware
Unit and property tests exercise the pure core on CPU with synthetic
audio fixtures; CI never requires ROCm, a sound card, or SuperCollider.
GPU training and SuperDirt playback are verified by local smoke tests
(documented `just` recipes), not CI.

### V. Nix-First
The flake dev shell provides every tool (Python env, just, formatters).
CI runs `nix develop -c just ci` on the self-hosted nixos runner; local
and CI environments are identical. No pip install outside the flake.

## Domain Constraints

- Target hardware for training/inference: Strix Halo gfx1151 under
  ROCm. Feasibility of each training feature is gated by a smoke test
  on that hardware before the feature is specced as depending on it.
- Pretrained encoders (CLAP/MERT) are used for audio embeddings;
  training audio encoders from scratch is out of scope.
- Live rendering uses real SuperDirt; offline/training rendering uses
  the pure slice mixdown renderer. The two must share the same event
  scheduling core so training audio approximates live audio.
- The live agent is soft-real-time: generation and scoring happen
  between pattern swaps on cycle boundaries; it must never block or
  glitch the audio stream.
- Corpus audio may be copyrighted: it stays on local disk, out of git,
  out of logs, and out of published artifacts.

## Development Workflow

- Every ticket goes through speckit: specify → plan → tasks → implement.
- PRs only; linear history; rebase merges; CI green before merge
  (merge-guard). One worktree per issue.
- Conventional commits; each commit addresses a single concern.
- `just ci` locally before every push; it mirrors GitHub CI exactly.
- Investigation notes in the main repo's `.llm/` (gitignored).

## Governance

This constitution gates speckit planning decisions: plans and specs
must comply or explicitly amend it first. Amendments are made by PR
that updates this file, with the reasoning in the PR description.
