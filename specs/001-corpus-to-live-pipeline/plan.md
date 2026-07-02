# Implementation Plan: Corpus-to-Live Pipeline (wav2tidal v1)

**Branch**: `001-corpus-to-live-pipeline` | **Date**: 2026-07-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `./spec.md`

## Summary

wav2tidal ingests a local WAV corpus into SuperDirt-loadable, beat-sliced
sample banks plus a style profile; synthesizes a seeded training dataset
by rendering random valid TidalCycles patterns through a pure offline
slice-mixdown renderer; trains a local model mapping style descriptors to
pattern text; and runs a live evolutionary agent that plays through
SuperDirt and steers a population of validated patterns toward a target
style. Python, pure-core/impure-shell, offline, single Linux workstation.

Technical approach is grounded in six verified research tasks
([research.md](./research.md)). Load-bearing decisions: **GPU (gfx1151
ROCm) is used only for training**; embedding (CLAP, Apache-2.0) and all
DSP run on **CPU**; the **evolution score comes from the deterministic
offline render**, not live capture (live capture is a verification
channel); a single versioned **pattern-subset EBNF grammar** is shared by
generator, validator, and constrained decoder.

## Technical Context

**Language/Version**: Python (3.11+ per nixpkgs), pure-core + impure-shell.

**Primary Dependencies** (all VERIFIED present in nixpkgs on the target
box except where noted): `soundfile`, `librosa` 0.11.0, `soxr`, `numpy`,
`scipy` (ingestion/DSP); `transformers` + `laion/larger_clap_music`
(embeddings, CPU); `torchWithRocm` 2.11.0 `.override{gpuTargets=
["gfx1151"]}`, `peft`, `trl`, `outlines` (training/decoding — torch-ROCm
build NEEDS-HARDWARE-TEST); `pw-record`/`soundcard` (capture; `JACK-Client`
and `pipewire-python` are NOT packaged — avoided).

**Storage**: local files only (FR-007) — parquet/npy/faiss for the
profile, on-disk WAV banks in SuperDirt layout, dataset/checkpoint/session
directories with embedded `(config, seed)`. No database, no network.

**Testing**: pytest on the pure core with synthetic in-memory fixtures
(click track, sine, silence, clipped, corrupt); property tests for
determinism and pattern validation; GPU/audio behavior via documented
local smoke tests (`smoke-gpu`, `doctor`) — **CI never needs ROCm or
audio hardware** (constitution IV).

**Target Platform**: single NixOS workstation ("sesimbra", gfx1151); the
user's TidalCycles+SuperDirt+PipeWire stack is a driven dependency.

**Project Type**: single Python project (CLI with per-stage subcommands).

**Performance Goals**: ingest 100 tracks/~8h in <30 min (SC-001); full
ingest→dataset→train→eval <24 h (SC-010); live session 30 min continuous,
no dropouts, cycle-boundary swaps (SC-005); controls effective within one
cycle (SC-007).

**Constraints**: offline after one-time model fetch (FR-027); corpus and
derivatives local-only (FR-007); soft-real-time live agent (never glitch
audio); reproducible from `(config, seed)` (SC-008); ≥95 % generated-
pattern validity (SC-003).

**Scale/Scope**: 10–500 tracks; single user; four prioritized user
stories (P1 ingest → P2 one-shot → P3 live → P4 reproducibility).

## Constitution Check

*GATE: re-checked after Phase 1 design. All pass.*

- **I. Pure core, impure shell** — PASS. DSP, embedding math, pattern
  generation/validation, event scheduling, and mixdown render are pure
  numpy functions ([contracts/interfaces.md](./contracts/interfaces.md));
  IO (WAV, GHCi subprocess, PipeWire, model load) is at the edges.
- **II. Patterns are validated data** — PASS. `validate()` gates every
  pattern before render/train/play (FR-009/010); the live agent never
  sends unvalidated output to SuperDirt.
- **III. Reproducibility** — PASS. Seeds + committed configs; artifacts
  embed `(config, seed)`; corpus/derivatives gitignored; SC-008 automated
  check.
- **IV. Tests run without GPU or audio** — PASS. Pure-core pytest on CPU
  with synthetic fixtures; GPU/audio only in documented local smoke tests,
  not CI.
- **V. Nix-first** — PASS. Flake dev shell provides everything; CI runs
  `nix develop -c just ci` on the nixos runner.
- **Domain constraints** — PASS. gfx1151 gated by the FR-018 smoke test
  (R2); pretrained encoders only (R1, no from-scratch encoder); offline
  renderer and live playback share one scheduling core (FR-013,
  `schedule_events`); soft-real-time agent scores the offline render to
  stay off the audio thread (R6); corpus stays local (FR-007).

No violations → Complexity Tracking omitted.

## Open decisions surfaced to the user (do not silently resolve)

1. **Learned-model fork (FR-015) — RESOLVED (user, 2026-07-02): ByT5-small
   seq2seq** (`google/byt5-small`, Apache-2.0). Tokenizer-free byte-level
   string→string, architecturally correct for descriptor→mini-notation and
   won't mangle pattern punctuation. The LoRA'd-tiny-LLM path is dropped
   for v1 (its only edge — free-text prompt understanding — is out of
   scope). The procedural generator + grammar + validator + evolution
   remain model-agnostic; only `train`/`generate` bind to ByT5. Note:
   torch-ROCm on gfx1151 (R2) is still the training substrate; byt5-small
   (~300 M) trains comfortably in bf16 on 128 GB (full fine-tune viable;
   LoRA optional). The FR-018 GPU smoke test still gates training.
2. **FR-021 — RESOLVED (user, 2026-07-02): keep live capture
   authoritative.** The evolution loop scores the **live-captured** audio
   of the agent's own output (PipeWire null-sink monitor, R6), as the spec
   originally states — chosen for fidelity to "listens to its own output"
   over the simpler offline-render shortcut. The deterministic offline
   render (R6) is retained as (a) the dataset-generation signal (FR-012,
   unchanged) and (b) a fallback score when capture is unavailable and a
   cross-check that offline≈live (FR-013). Accepted tradeoff: added
   real-time capture complexity and weaker reproducibility of live
   sessions (per-session logs remain, SC-008 still applies to the offline
   dataset/train path). No spec edit needed — FR-021 stands as written.

## Project Structure

### Documentation (this feature)

```text
specs/001-corpus-to-live-pipeline/
├── plan.md              # this file
├── spec.md              # feature spec
├── research.md          # Phase 0 — six verified research tasks
├── data-model.md        # Phase 1 — entities & state transitions
├── quickstart.md        # Phase 1 — newcomer path, doc-as-script
├── contracts/           # Phase 1 — CLI + interface/protocol contracts
│   ├── cli.md
│   └── interfaces.md
└── tasks.md             # Phase 2 — /speckit.tasks (not yet created)
```

### Source Code (repository root)

```text
src/wav2tidal/
├── core/                     # PURE — no IO (constitution I)
│   ├── dsp/                  # normalize, slice_boundaries, features (R4)
│   ├── descriptor/           # assemble descriptor, similarity (R1)
│   ├── pattern/              # subset grammar, generate, validate, mutate (R3/R5)
│   └── render/               # schedule_events (SHARED timing), mixdown (R4/R6)
├── io/                       # IMPURE edges
│   ├── wav.py                # soundfile read/write
│   ├── banks.py             # SuperDirt bank layout writer (R5)
│   ├── embedder.py          # CLAP load+embed, offline-pinned (R1)
│   ├── tidal.py             # GHCi stdin driver + OSC control (R5)
│   └── capture.py           # PipeWire null-sink + pw-record (R6)
├── train/                    # LoRA/seq2seq + constrained decode (R2/R3)
├── pipeline/                 # ingest, dataset, train, eval, generate, live
└── cli.py                    # subcommands (contracts/cli.md)

grammar/pattern_subset.lark    # THE versioned grammar artifact (FR-008)
configs/                       # committed per-stage config files (FR-026)
tests/
├── unit/                     # pure-core, synthetic fixtures, CPU-only
├── property/                 # determinism + validation invariants
└── smoke/                    # GPU (FR-018) + audio doctor — local, not CI
```

**Structure Decision**: single Python project. The `core/` vs `io/` split
is the physical enforcement of constitution I — `core/` imports no IO and
is the entire unit-test surface. `grammar/pattern_subset.lark` is the one
artifact the generator, validator, and decoder all derive from (FR-008),
preventing the three from drifting.

## Phase 2 note

`/speckit.tasks` will decompose this by user-story priority (P1→P4) so each
story is an independently testable, independently valuable slice
(constitution + spec). Story 1 (ingest) ships standalone value with zero
ML; Story 4 (reproducibility) is cross-cutting and gated by SC-008/SC-009.
