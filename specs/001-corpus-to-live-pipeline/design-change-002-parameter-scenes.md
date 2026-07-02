# Design Change 002 — Parameter scenes: compose in parameter space

**Date**: 2026-07-02
**Status**: Accepted
**Builds on**: [design-change-001](./design-change-001-sound-first.md)
(sound-first). **Affects**: FR-008/011/012/014/015 (grammar, generator,
renderers, dataset, model target), tasks.md (reshapes the remaining US2
step), issue #22. Event patterns (grammar v2) are retained as a layer,
not superseded.

## Context

After hearing the grammar-v2 action space rendered ([PR #26](https://github.com/lambdasistemi/wav2tidal/pull/26)
demo reel), the user's verdict (2026-07-02): quality decent, **but the
composition model is wrong-side-up**. Event lines make Tidal score *when
sounds trigger*; the user wants to compose **at the synth level**: a few
sustained voices, with Tidal scoring **the parameters** — moving cutoff,
resonance, detune, room… along shapes over time. Warm, evolving texture
("drone fashion / modulation of the sound"), not samples stacked on a
grid.

Decisions (user, 2026-07-02):

- **Hybrid**: parameter *scenes* (3–4 drone voices + per-param
  trajectories) become the primary composition mode; event patterns
  (drums, sample hits) remain a layer that can play alongside. Both stay
  in the model's vocabulary.
- **Reshape before training**: no ByT5 run on the event-only target;
  train once on the space actually wanted (#22 retargeted).
- **Trajectory shapes v1**: constant, ramp/line, sine LFO, seeded random
  walk, step sequence.
- Loudness is a render-stage concern (normalization/gain staging), fixed
  in the renderer, not the design.

## What a scene is

```
scene   := voices [1..4] + layer? + duration
voice   := source (Super* synth | custom def) + note + static params
traj    := (voice, param) -> shape       one per modulated param
shape   := const v
         | ramp v0 v1                    linear over the scene
         | sine center depth rate        LFO (rate Hz, depth in param units)
         | walk center depth rate seed   seeded random walk (S&H interp)
         | steps v1 v2 ... (per cycle)   Tidal-like value pattern
layer   := an optional grammar-v2 event line (rhythmic material)
```

Trajectory values are bounded by the same table (`core/pattern/params.py`)
that bounds static params: `lo/hi` clamp the shape's range, the lin/log
tag tells the shape generator how to interpolate (log params sweep in
log space). The table is reused as-is — grammar v2's semantic layer *is*
the modulation-target vocabulary.

The concrete text form of a scene is grammar v3's job (US2-scene-1); it
extends `grammar/pattern_subset.lark` versioned, keeping the v2 `line`
and `mini` rules valid (the layer reuses them verbatim).

## Why the architecture already supports this

- **Live**: design-change-001 already fixed the continuous channel —
  `/ctrl name value` → Tidal:6010 (`cF`-parameterised pattern) →
  SuperDirt:57120. A trajectory is exactly a stream on that channel.
  US3's "action space is the continuous param vector" line anticipated
  this change.
- **NRT**: a voice is one long-sustain `s_new`; a trajectory is timed
  `n_set` rows in the score — **fully deterministic** automation. The
  Super* defs' args are control-rate (settable while running) and the
  per-event effect synths are explicitly modulatable (core-synths.scd
  marks cutoff/resonance/… `\kr`, "can be modulated").
- **RT**: the batch script (PR #26) already schedules timed messages
  through one booted SuperDirt; parameter ticks (`n_set` at 20–50 Hz on
  the voice nodes) are the same mechanism. Global-FX params modulate the
  orbit's long-running effect synths the same way (subject to the known
  delay caveat, R7 addendum).
- **Dataset**: `config_dataset` (PR #26) is renderer-injected; scenes
  slot in as a third config kind with the same routing fidelity rule.

## Consequences

- **Grammar v3** (versioned, supersedes v2 as the model's emission
  language; v2 lines remain a sub-language used by the `layer`).
- **Generator/validator** extend to scenes: seeded, valid-by-construction
  voices + trajectories (bounded by the table), bounded voice/trajectory
  counts.
- **Renderers** gain parameter automation: NRT score rows (deterministic)
  and RT ticks; plus output **normalization** (the loudness fix) applied
  uniformly so descriptors are level-comparable.
- **Descriptors** likely need movement-aware features (e.g. centroid
  trajectory statistics) — a static brightness bucket can't distinguish a
  filter sweep from a fixed timbre. Scoped in US2-scene-3.
- **Model target (#22)** becomes the hybrid corpus: scene texts (+ event
  lines as layers). Still string→string; ByT5 unchanged as the tool.
- **Routing fidelity rule extends**: a scene with global-FX trajectories
  → RT; synth-only voices with bare-def param trajectories → NRT
  (deterministic); the layer routes as in v2.

## Reshaped remaining plan (supersedes the US2-synth-5-next step)

- **US2-scene-1 — grammar v3 + generator + validator** (scenes,
  trajectories, hybrid layer).
- **US2-scene-2 — renderer parameter automation** (NRT `n_set` rows, RT
  ticks, normalization/gain staging).
- **US2-scene-3 — scene dataset** (both renderers; movement-aware
  descriptor extension).
- **US2-synth-5 / #22 — ByT5 Seq2SeqTrainer** on the hybrid corpus
  (unchanged tooling, retargeted data; now blocked by US2-scene-3).

US3 then drives the *same* trajectories live over `/ctrl`, and evolution
mutates trajectory shapes/rates/depths — a smoother search space than
event-line edits, which suits the warm-drift aesthetic.

## Alternatives considered

- **Train the event model first as a shakedown** — rejected by the user;
  costs a GPU cycle on a target that isn't the goal.
- **Scenes only (full pivot)** — rejected; rhythmic material stays
  available as a layer.
- **Free-form modulation (arbitrary curves)** — deferred; the five-shape
  vocabulary keeps the space learnable and mutable, and `/ctrl` can carry
  richer curves later without a grammar change.
