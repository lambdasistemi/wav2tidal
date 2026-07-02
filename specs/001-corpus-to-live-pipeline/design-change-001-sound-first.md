# Design Change 001 — Sound-first: real-time control of SuperDirt synths

**Date**: 2026-07-02
**Status**: Accepted
**Affects**: spec.md (scope, FR-008/012/015/019/021), plan.md, research.md
(adds R7), tasks.md (reshapes US2). Supersedes the slice-only framing of
User Story 2 and 3.

## Context

The original v1 framed the instrument as arranging **corpus sample slices**
rhythmically, with timbre coming from the recordings themselves and a
small control set (gain/speed/pan). The user has clarified the actual
priority: **sound/timbre first** — the agent should **control SuperDirt
synths and their full parameter/effect vocabulary in real time**, not just
retrigger slices.

Scope decisions (user, 2026-07-02):
- **Sound palette = all three**: SuperDirt `Super*` synths, corpus samples
  through live effects, and custom SynthDefs.
- **Global effects (reverb/delay/orbit) are in scope now**, not deferred.
- **Broad synth set** (most Super* synths, including percussive/noise ones)
  and the full continuous-parameter vocabulary.
- **The trained model stays central**: it maps a target sound/descriptor →
  synth+parameter configuration; training data comes from rendered
  (config → audio) pairs.

## What the research established (R7, verified on the box 2026-07-02)

Full findings + citations in `research.md` §R7 and
`reference_superdirt_nrt` memory. Load-bearing verdict:

- **Tier 1 — a source synth + per-event effects** (supersaw, superhoover,
  … + cutoff/resonance/hcutoff/bandf/shape/crush/coarse/vowel/envelope):
  **headless, deterministic NRT rendering WORKS** (`Score.recordNRT` →
  `scsynth -N … _ out.wav 44100 WAV int16`, no audio device; determinism
  via `RandSeed.ir`). Empirically proven: two runs → byte-identical WAV.
- **Tier 2 — the global orbit FX graph** (dirt_reverb/dirt_delay + the
  `source → dryBus → globalFX → monitor` routing): SuperDirt ships **no
  NRT support**; the effects use Rand/LFNoise. Faithfully reproducing the
  wet global-FX signal requires **real-time capture** of a booted
  SuperDirt.
- **Live real-time control** is `/ctrl name value` to **Tidal on
  127.0.0.1:6010** (not SuperDirt): run a pattern parameterised with
  `cF "name" default`, stream `/ctrl` values, Tidal forwards to SuperDirt
  on 57120. This is the agent's continuous-modulation channel.
- **nixpkgs**: supercollider 3.13.1 packaged; SuperDirt is a bespoke
  derivation on the box (quark v1.7.3), not `nixpkgs#superdirt` — pin it.

## Decision

Because global FX are in scope, **the offline deterministic NRT path
cannot render the full sonic chain**. Therefore:

1. **Dataset generation uses real-time capture through a booted
   SuperDirt** — the *same* sound engine and capture path the live agent
   uses (PipeWire null-sink monitor + `pw-record`, per R6). One sound
   engine serves both training-data generation and live scoring.
   - Deterministic NRT (tier 1) is retained as a **fast offline path** for
     the synth+per-event-FX subset and for unit-testing the scheduler, but
     it is no longer the sole/primary renderer.
2. **The pure-numpy slice mixer (US2a) is demoted** to the sample-only fast
   path and to a test/reference renderer. It is not wasted — the scheduler,
   grammar, generator, and validator it accompanies are reused — but it no
   longer generates the synth dataset.
3. **The action space (grammar) expands** from `{slice, gain/speed/pan}` to
   `{source ∈ (Super* synth | sample bank | custom synthdef), note/n,
   per-event FX params (continuous), global FX sends}` — a versioned
   grammar v2 built from the R7 param table.
4. **Determinism (SC-008) relaxes** for the synth/FX dataset: renders are
   seeded where the synthdef allows (`RandSeed.ir`), and otherwise
   reproducible only within a documented audio-similarity tolerance.
   Deterministic reproducibility remains a hard requirement for the pattern
   *generator* (config emission from a seed) and for tier-1 NRT renders.
5. **Scoring stays timbre-focused** (CLAP + spectral/MFCC on captured
   audio), which the research confirms these embeddings handle well.

## Consequences

- **New load-bearing unknown to prove before implementing US2's dataset
  gen**: real-time capture through a booted SuperDirt end to end
  (sclang-with-superdirt boot → drive a pattern → PipeWire monitor capture
  → numpy). This is the audio-path analogue of the (now-passed) GPU smoke
  gate. A `doctor`/`smoke-audio` check will gate it.
- **Dataset generation is wall-clock-bound** (each render plays in real
  time), so dataset sizes and SC-010's 24h budget must be re-derived. NRT
  tier-1 remains available for fast bulk generation of the deterministic
  subset.
- **The model target text changes** from a slice pattern to a synth+param
  configuration string — still string→string, so ByT5 seq2seq (chosen,
  and its GPU substrate now proven) remains the right tool; the grammar it
  emits is grammar v2.
- **FR deltas** (to be folded into spec.md):
  - FR-008 (pattern subset) → grammar v2 with synths + params + FX.
  - FR-012 (offline render) → split: tier-1 NRT (deterministic, subset) +
    real-time SuperDirt capture (full chain).
  - FR-013 (offline≈live) → now trivially satisfied for the RT-capture
    path (it *is* live); retained as a tier-1-NRT-vs-live tolerance check.
  - FR-015 (model) → target = synth+param config; unchanged tool (ByT5).
  - FR-021 (live capture authoritative) → reinforced; now also the dataset
    signal.
  - New FR: pin the SuperDirt derivation + a `smoke-audio` gate for the
    real-time capture path.

## Reshaped US2 task plan (supersedes tasks.md US2 for the synth path)

> **Execution note (2026-07-02).** The renderers were built *before*
> grammar v2 (each renderer proof was the riskier unknown), so the
> executed US2-synth-N numbering in issues/PRs differs from the original
> draft below: renderers became -1/-2 and grammar v2 became -3. The list
> below is updated to the executed order, with status and issue links.

- **US2a (MERGED, PR #6)** — pattern engine (grammar, generator, validator,
  scheduler, numpy mixer). Reused as scaffolding; grammar → v2 next.
- **GPU gate (MERGED, PR #7)** — ByT5 training on gfx1151 proven.
- **US2-synth-1 — NRT renderer + audio smoke gate (DONE — issue #9,
  PR #10)**: a real `supersaw` renders headless + deterministic via
  `Score.recordNRT` (closed the tier-1 NEEDS-HARDWARE-TEST); `just
  smoke-audio` gates it. The SuperDirt/sclang flake pinning from this
  step's scope is still pending → **issue #23**.
- **US2-synth-2 — RT capture renderer (DONE — issue #11, PR #12)**:
  real-time capture of the full chain incl. global FX through a booted
  SuperDirt (`/dirt/play` → orbit bus record → PipeWire null sink).
- **US2-synth-3 — grammar v2 + generator + validator (DONE — issue #13,
  PR #20)**: synths + params + FX action space from the R7 table
  (`contracts/params-v2.md`, `core/pattern/params.py`); grammar v2.0.0
  governs the full Tidal line; seeded valid-by-construction configs;
  config → renderer params mapping (`core/pattern/dirt.py`). Follow-up:
  chain per-event `dirt_*` FX into the NRT score → **issue #24**.
- **US2-synth-4 — dataset (NEXT — issue #21)**: (captured-audio
  descriptor → synth+param config) pairs via both renderers; seeded
  generator, tolerance-based reproducibility.
- **US2-synth-5 — ByT5 Seq2SeqTrainer + eval + generate (issue #22,
  blocked by #21)** (GPU training shell, gated by `smoke-gpu`).

US3 (live evolution) then reuses the same SuperDirt engine, grammar v2,
capture, and scoring; the action space is the continuous param vector
streamed over `/ctrl`.

## Alternatives considered

- **Tier-1-only (defer global FX)** — cleaner/deterministic, rejected by
  the user in favour of full sonic richness now.
- **Rebuild the DirtOrbit graph by hand in NRT** — the research judged this
  error-prone and non-deterministic anyway; rejected.
- **Pure-numpy synth emulation** — would diverge from real SuperDirt DSP;
  rejected except as a possible future differentiable proxy.
