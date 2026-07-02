# Tasks: Corpus-to-Live Pipeline (wav2tidal v1)

> **Amended by [design-change-001-sound-first.md](./design-change-001-sound-first.md)
> (2026-07-02).** Done: Phase 1 scaffold, US1 ingest (PR #4), US2a pattern
> engine (PR #6), GPU training gate (PR #7). **User Story 2 is reshaped
> around synth control** — the ADR's "Reshaped US2 task plan" supersedes
> Phase 4 below for the synth path (audio-path smoke gate → grammar v2 →
> tier-1 NRT + SuperDirt RT-capture renderers → dataset → ByT5 training).
> Phase 4 below remains the record for the slice-based path (US2a).

**Input**: Design documents from `/specs/001-corpus-to-live-pipeline/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: INCLUDED — the constitution (IV) mandates a pure-core test
surface (pytest + property tests on synthetic fixtures, CPU-only). GPU and
audio behaviour is covered by local smoke tests, never CI.

**Organization**: by user story priority (P1→P4). Phase 1 = the issue #1
scaffold. Phase 2 = foundational artifacts shared by all stories (grammar,
config, storage, descriptor core). Each story phase is an independently
testable increment.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: parallelizable (different files, no incomplete deps)
- **[Story]**: US1..US4; Setup/Foundational/Polish carry no story label

## Path conventions (from plan.md)
`src/wav2tidal/{core,io,train,pipeline}/`, `src/wav2tidal/cli.py`,
`grammar/pattern_subset.lark`, `configs/`, `tests/{unit,property,smoke}/`,
`docs/`, repo-root `flake.nix`/`justfile`/`pyproject.toml`.

---

## Phase 1: Setup — issue #1 scaffold (Shared Infrastructure)

**Purpose**: buildable, CI-green project foundation. Replaces the stub CI.

- [ ] T001 Create `pyproject.toml` (package `wav2tidal`, console-script
  entry `wav2tidal=wav2tidal.cli:main`, Paths autogen per haskell/cabal-
  analogue N/A; ruff+black config), version 0.1.0.
- [ ] T002 Create `flake.nix` — Python dev shell with the R1/R4/R6 verified
  nixpkgs deps (`soundfile librosa soxr numpy scipy transformers`,
  `soundcard sounddevice resampy`, `just ruff black pytest`), plus a
  SEPARATE optional `training` shell exposing
  `torchWithRocm.override { gpuTargets = ["gfx1151"]; }` + `peft trl
  outlines` (kept out of the default shell so CI never pulls ROCm). Pin
  nixpkgs to the system rev.
- [ ] T003 [P] Create `justfile` with recipes: `ci` (ruff check + black
  --check + pytest), `format`, `format-check`, `test`, `fetch-models`,
  `smoke-gpu`, `doctor`, `pipeline`, `build-docs`, `serve-docs`,
  `deploy-docs`.
- [ ] T004 [P] Create the package skeleton (empty `__init__.py` +
  module stubs) under `src/wav2tidal/{core/{dsp,descriptor,pattern,render},
  io,train,pipeline}/` and `src/wav2tidal/cli.py` with an argparse
  subcommand shell matching contracts/cli.md (each subcommand prints
  "not implemented" and exits 2).
- [ ] T005 [P] Replace the stub `.github/workflows/ci.yml` with the real
  build-gate CI (runs-on: nixos, `Build Gate` job builds the default
  devshell inputDerivation; `test`/`lint` jobs `needs: build-gate` run
  `nix develop -c just ci`). Keep the job name `Build Gate` (branch
  protection depends on it).
- [ ] T006 [P] MkDocs skeleton: `mkdocs.yml` (Material) + `docs/index.md`
  (project overview, links to quickstart) + `.github/workflows/deploy-
  docs.yml` per new-repository skill (build --strict on PR, gh-deploy on
  main via `paolino/dev-assets?dir=mkdocs`).
- [ ] T007 [P] Create the FR-018 GPU smoke test `tests/smoke/smoke_gpu.py`
  (the R2 two-step: torch sees gfx1151 + one bf16 LoRA/seq2seq step) wired
  to `just smoke-gpu`; prints PASS/FAIL with the fix on FAIL. Not in CI.
- [ ] T008 Verify locally: `nix develop -c just ci` is green (empty pytest
  OK), `nix flake check` passes. Push; confirm CI `Build Gate` green on PR #2.

**Checkpoint**: repo builds, real CI green, `wav2tidal --help` lists all
stages. Foundational work can begin.

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: blocks all user stories.

- [ ] T009 Author `grammar/pattern_subset.lark` — the versioned EBNF of the
  supported mini-notation + combinator subset (R5 list), with a `version`
  header and documented complexity bounds (max events/cycle, max nesting).
  This is the single source of truth for generator+validator+decoder (FR-008).
- [ ] T010 [P] `src/wav2tidal/core/config.py` — typed config loading from
  committed YAML (dataclasses per stage), with `seed` and shared
  `TARGET_SR` constant (couples R4↔R1). Pure; no IO beyond reading a path
  at the edge.
- [ ] T011 [P] `src/wav2tidal/core/descriptor/types.py` — `StyleDescriptor`
  (embedding + handcrafted blocks + `embedder_id`/`sr_used`) and
  `similarity()` (block-weighted cosine) + `nearest()` (data-model.md).
  Pure numpy.
- [ ] T012 [P] `src/wav2tidal/io/storage.py` — the on-disk layout helpers
  (banks/, profile/, datasets/<id>/, checkpoints/<id>/, sessions/<id>/),
  artifact `(config, seed)` embedding, corpus manifest (path/size/mtime/
  hash) for incremental ingest (FR-006). Confirm all paths are gitignored.
- [ ] T013 [P] Tests: `tests/unit/test_descriptor.py` (similarity is
  symmetric, self-similarity=1, cross-`embedder_id` comparison rejected)
  and `tests/property/test_config_roundtrip.py` (config serialize/parse
  identity).

**Checkpoint**: grammar + config + descriptor core + storage exist and are
tested. Stories can start.

---

## Phase 3: User Story 1 — Ingest corpus into playable banks (P1) 🎯 MVP

**Goal**: raw WAVs → beat-sliced SuperDirt banks + style profile with
nearest-neighbour query. Zero ML. Standalone value (SC-001, SC-002).

**Independent Test**: `wav2tidal ingest` on a real corpus → play a bank by
hand in stock Tidal (`d1 $ s "bank"`) → `wav2tidal profile --query`
ranks similar tracks above dissimilar (SC-002).

### Tests for US1 (write first, must fail)
- [ ] T014 [P] [US1] `tests/unit/test_dsp_slicing.py` — 120-BPM click-track
  fixture → `beat_track` tempo≈120 & even beat spacing; silence fixture →
  no slices; determinism (same input → same boundaries) (R4).
- [ ] T015 [P] [US1] `tests/unit/test_dsp_features.py` — 440 Hz sine →
  chroma peak at A, centroid≈440; clipped/corrupt fixtures flagged; pooled
  vectors fixed-length regardless of clip length (R4).
- [ ] T016 [P] [US1] `tests/property/test_ingest_idempotent.py` — re-ingest
  unchanged corpus = no-op; adding a file processes only it; bank names +
  slice ids stable (FR-006).

### Implementation for US1
- [ ] T017 [P] [US1] `src/wav2tidal/io/wav.py` — soundfile read/write,
  mono downmix, `soxr_hq` resample to `TARGET_SR`; raise/handle
  `LibsndfileError` at the edge; silence(RMS)/clip(|y|≥1-eps) detection.
- [ ] T018 [P] [US1] `src/wav2tidal/core/dsp/slice.py` — pure
  `slice_boundaries(y,sr,hop,strategy)`: onset_strength → beat_track →
  onset_detect(backtrack) → fallback (tempo subdivisions / effects.split).
- [ ] T019 [P] [US1] `src/wav2tidal/core/dsp/features.py` — pure
  `slice_features` + `track_descriptors`: mfcc/chroma/spectral pooled;
  tempo(+derived confidence proxy via tempogram/PLP); hand-rolled
  Krumhansl–Schmuckler key + strength (R4 — librosa has neither built-in).
- [ ] T020 [US1] `src/wav2tidal/io/banks.py` — write slices to
  `banks/<bank>/NN_*.wav` in the SuperDirt layout (basename=name, alpha
  order=`:n`), stable naming across re-ingest (R5).
- [ ] T021 [US1] `src/wav2tidal/io/embedder.py` — CLAP load
  (`laion/larger_clap_music` @ pinned SHA, `local_files_only`), CPU embed
  → 512-d; `just fetch-models` populates the local HF cache (R1).
- [ ] T022 [US1] `src/wav2tidal/pipeline/ingest.py` — orchestrate: manifest
  diff → slice → bank → embed → assemble descriptors → build profile
  (parquet + NN index); run report of produced banks + skipped files.
- [ ] T023 [US1] `src/wav2tidal/pipeline/profile.py` + wire `ingest` &
  `profile` subcommands in `cli.py` (contracts/cli.md).
- [ ] T024 [US1] `configs/ingest.yaml` — documented defaults (target_sr,
  hop, slice strategy, thresholds, bank naming).

**Checkpoint**: US1 fully functional — playable banks + working similarity.
MVP shippable.

---

## Phase 4: User Story 2 — One-shot style-matched generation (P2)

**Goal**: target audio → N validated candidate patterns ranked by
render-similarity. Exercises generator→dataset→train(ByT5)→generate→validate.

**Independent Test**: with a trained checkpoint, `wav2tidal generate
--target X --n 8` → 8 valid patterns, higher-ranked measure closer to X
(SC-004); ≥95% validity (SC-003).

### Tests for US2 (write first, must fail)
- [ ] T025 [P] [US2] `tests/unit/test_pattern_validate.py` — grammar
  membership, unknown-bank rejection, complexity bounds; invalid never
  passes (FR-009/010) using `grammar/pattern_subset.lark`.
- [ ] T026 [P] [US2] `tests/property/test_generator_valid.py` — every
  seeded `generate_pattern` output validates; determinism from seed (FR-011).
- [ ] T027 [P] [US2] `tests/property/test_render_deterministic.py` —
  `schedule_events`+`render` deterministic given (pattern,banks,seed);
  render timing matches a reference schedule within FR-013 tolerance.
- [ ] T028 [P] [US2] `tests/property/test_dataset_reproducible.py` — same
  (config,seed) → identical dataset (SC-008).

### Implementation for US2
- [ ] T029 [P] [US2] `src/wav2tidal/core/pattern/validate.py` — pure
  validator parsing against the lark grammar + bank-ref + bounds checks.
- [ ] T030 [P] [US2] `src/wav2tidal/core/pattern/generate.py` +
  `mutate.py` — pure seeded procedural generator (diversity distributions)
  and mutation ops over the subset (also used by US3 evolution).
- [ ] T031 [P] [US2] `src/wav2tidal/core/render/schedule.py` — pure
  `schedule_events(pattern,cps,n_cycles)` → events. **Shared timing core**
  the live path (US3) must reuse (FR-013).
- [ ] T032 [US2] `src/wav2tidal/core/render/mixdown.py` — pure
  `render(events,banks,sr)` slice mixdown (playback rate, gain, pan)
  (FR-012). Depends on T031.
- [ ] T033 [US2] `src/wav2tidal/pipeline/dataset.py` — seeded, resumable
  (descriptor-of-render → pattern) pair synthesis; embed config+seed
  (FR-014). Depends on T030,T032,T021.
- [ ] T034 [US2] `src/wav2tidal/train/decode.py` — `outlines` grammar-
  constrained decoding bound to the lark subset (the ≥95%-valid mechanism,
  R3). Pin outlines version.
- [ ] T035 [US2] `src/wav2tidal/train/byt5.py` — ByT5-small seq2seq
  fine-tune (descriptor-text→pattern), bf16 on torch-ROCm training shell;
  seeded/deterministic where feasible; writes checkpoint + config (R3).
  Refuses to run unless `smoke-gpu` passed.
- [ ] T036 [US2] `src/wav2tidal/pipeline/evaluate.py` — held-out EvalReport:
  validity rate, style-match distribution (render-vs-source similarity),
  vs-previous (FR-017).
- [ ] T037 [US2] `src/wav2tidal/pipeline/generate.py` + wire `dataset`,
  `smoke-gpu`, `train`, `eval`, `generate` subcommands; malformed outputs
  discarded+retried within budget (FR-019).
- [ ] T038 [US2] `configs/{dataset,train}.yaml` — documented defaults.

**Checkpoint**: US1+US2 both work; one-shot generation produces
style-plausible patterns.

---

## Phase 5: User Story 3 — Live evolving session (P3)

**Goal**: continuous playback through SuperDirt; evolve a population toward
a target scored on **live-captured** audio (user decision); cycle-boundary
swaps; user controls.

**Independent Test**: `wav2tidal live --target X` → continuous audio 30 min,
no dropouts, swaps on boundaries (SC-005); similarity trends up over 10 min
(SC-006); controls effect within one cycle (SC-007).

### Tests for US3 (write first, must fail)
- [ ] T039 [P] [US3] `tests/unit/test_evolution.py` — pure population step
  (select/mutate/regenerate) improves or holds best; patience → keep-best +
  notify (FR-024); all candidates validated before playing.
- [ ] T040 [P] [US3] `tests/unit/test_tidal_format.py` — GHCi block framing
  (`:{ … :}`), `jumpIn'` used for boundary swaps not bare `d1` (R5).

### Implementation for US3
- [ ] T041 [P] [US3] `src/wav2tidal/io/tidal.py` — GHCi stdin driver
  (BootTidal, `:{ }:` framing, `jumpIn'`/`xfade` swaps, `setcps`, `hush`),
  cycle/tempo via `streamGetNow`/Link, stdout/stderr error scrape + restart
  (R5).
- [ ] T042 [P] [US3] `src/wav2tidal/io/capture.py` — PipeWire null-sink
  routing + `pw-record --target …monitor -n <samples>` windowed capture →
  numpy, resample to embedder rate (R6). Authoritative scoring signal;
  offline render as fallback/cross-check.
- [ ] T043 [US3] `src/wav2tidal/core/pattern/evolution.py` — pure
  population/candidate model + step (selection by measured similarity,
  mutation via T030, validation via T029) (FR-022/024).
- [ ] T044 [US3] `src/wav2tidal/pipeline/live.py` — the live loop: schedule
  swaps on cycle boundaries off the audio thread, capture→score→evolve,
  user controls (pause/freeze/retarget/stop) applied at next boundary,
  session log (FR-020/021/023/025). Refuses to start evolution if capture
  path missing (edge case).
- [ ] T045 [US3] `src/wav2tidal/pipeline/doctor.py` + `live` & `doctor`
  subcommands — preflight GHCi/SuperDirt/PipeWire/model cache (R5/R6).
- [ ] T046 [US3] `configs/live.yaml` — population size, mutation rates,
  patience, cps.

**Checkpoint**: all three stories independently functional; the full
listen→evolve loop runs live.

---

## Phase 6: User Story 4 — Reproducible retraining (P4, cross-cutting)

**Goal**: any stage re-runs from committed (config, seed) to identical
artifacts; newcomer reaches a live session from docs alone.

**Independent Test**: wipe derived artifacts, `just pipeline`, compare
checksums + eval metrics to a recorded run (SC-008); execute quickstart.md
as a script (SC-009).

- [ ] T047 [P] [US4] `tests/property/test_pipeline_reproducible.py` — end-
  to-end (small corpus fixture) ingest→dataset→train(CPU tiny)→eval twice →
  identical dataset + eval metrics within tolerance (SC-008).
- [ ] T048 [US4] `just pipeline` recipe chaining all stages from configs;
  `just repro-check` comparing artifact checksums/metrics to a recorded
  baseline.
- [ ] T049 [US4] Docs: make quickstart.md executable — a `just quickstart-
  check` that runs each documented step against a tiny bundled corpus
  fixture and asserts the verification points (SC-009/FR-028).

**Checkpoint**: pipeline reproducible and doc-verified.

---

## Phase 7: Polish & Cross-Cutting

- [ ] T050 [P] Flesh out `docs/` — per-stage pages (inputs/outputs/verify)
  mirroring quickstart; architecture page linking constitution + research.
- [ ] T051 [P] Ruff/black/type pass across `src/`; ensure `just ci` green.
- [ ] T052 Update PR #2 body to final merged state; ensure issue #1 closed
  by the scaffold; wiki logbook entry.
- [ ] T053 Run `just quickstart-check` + `just repro-check` end to end on
  the box; record the baseline artifacts' checksums in `docs/`.

---

## Dependencies & Execution Order

- **Phase 1 (Setup)**: start immediately; T001-T007 mostly [P]; T008 gates.
- **Phase 2 (Foundational)**: after Phase 1; T009 (grammar) blocks US2/US3
  pattern work; T010-T012 [P].
- **US1 (P3-phase)**: after Phase 2. MVP. No dependency on other stories.
- **US2**: after Phase 2. Needs grammar (T009), descriptor (T011),
  embedder (T021 from US1) — implement US1 first or lift T021 earlier.
- **US3**: after US2 (reuses generator/mutate T030, render T031/32,
  validator T029, evolution). Live scoring needs capture (T042).
- **US4**: after the stages it reproduces exist (ingest+dataset+train).
- **Polish**: last.

### Within a story: tests (fail first) → pure core → io → pipeline → cli → config.

### Parallel opportunities
- Phase 1: T001-T007 largely parallel (distinct files).
- Phase 2: T010/T011/T012/T013 parallel after T009.
- Each story's `[P]` tests and pure-core modules (distinct files) parallel.

## Implementation Strategy
- **MVP = Phase 1 + Phase 2 + US1** (playable banks + similarity, no ML).
  Stop and validate against SC-001/SC-002 before touching training.
- Then US2 (one-shot), US3 (live), US4 (repro) incrementally; each is a
  demoable increment that doesn't break the previous.
- Gate all training work behind `just smoke-gpu` (FR-018) — do not invest
  in US2 training until the gfx1151 torch build is proven on the box.

## Notes
- Constitution: `core/` imports no IO; it is the entire unit/property test
  surface (CPU-only). GPU/audio only in `tests/smoke/` + `doctor`.
- Commit per task or logical group; one worktree (this one) for the feature.
- The pattern-subset grammar (T009) is load-bearing — generator, validator,
  and decoder must all derive from it, never fork.
