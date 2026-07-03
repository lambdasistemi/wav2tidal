# Roadmap — improvements we will try

## The north star

Play the machine a piece you know — *Take Five* — and get back a
synthetic reinterpretation you would still **recognize**: its harmony,
its melody, its pulse, re-clothed in this instrument's sounds. And it
should *feel alive*: **full** (a many-voiced body of sound, not one thin
line) and **coherent** (the voices belong together — one creature, "a
crazy orchestra, a crazy monster that is playing"), perpetually *trying
to catch up* with what it hears — continuous pursuit, never a one-shot
answer.

Today the system matches sound *character*; recognizability of the
*tune* needs melodic/harmonic tracking over time (in tasting notes
**and** in scoring), which is a future chapter, planned after the live
loop first plays. Every design choice below is ultimately judged against
these tests.

Ordered roughly by expected leverage. Items with issue links are
committed work; the rest are candidates we consider worth trying, each
with the reasoning and the risk stated.

## Recently landed

- **Global FX in NRT**
  ([#40](https://github.com/lambdasistemi/wav2tidal/issues/40)) — the
  scene graph owns its reverb/delay/monitor, so they render as ordinary
  seeded synthdefs in the NRT score: 6 s of audio in ~0.6 s wall,
  byte-deterministic, within 7.5% centroid / 2% RMS of an RT render (the
  FR-013 scene tolerance). Only sample layers still require real-time
  capture; the RT fleet remains the live path and validation reference.
- **Render-phase overlap**
  ([PR #39](https://github.com/lambdasistemi/wav2tidal/pull/39)) — the
  CPU-bound NRT pool now runs concurrently with the wall-clock-bound RT
  fleets.

## In flight

- **Big-corpus retrain** — regenerate the training corpus at 10× (4,000
  pairs) on the render fleet and retrain. The first model's failure mode
  (unterminated generations) is exactly what more data teaches away.
  Running as of 2026-07-03; numbers will land on
  [#36](https://github.com/lambdasistemi/wav2tidal/issues/36) as the new
  baseline.

## Committed next

- **Grammar-constrained decoding**
  ([#36](https://github.com/lambdasistemi/wav2tidal/issues/36)) — the
  designed ≥95%-validity gate (FR-015): constrain generation to the lark
  grammar so every emission is a member of the language *by
  construction*, termination included. Plan: evaluate `outlines` CFG
  against grammar v3, fall back to `lm-format-enforcer` or a hand-rolled
  byte-level automaton walker. Constrained decoding makes validity
  structural; the big corpus makes it statistical — we want both.
- **US3: the live evolutionary loop** — the destination. Boot SuperDirt
  as the instrument; the model proposes a scene for a target sound;
  PipeWire captures what actually plays; CLAP+cosine scores it against
  the target; `mutate_scene` walks the parameter space; trajectories
  stream over `/ctrl` → Tidal `cF` → SuperDirt. Design session first
  (ADR, like every story): scoring cadence, evolution population vs
  single-lineage, human-in-the-loop overrides, and what "converged"
  means musically. `repair` is already the send-safety layer.
- **Pin SuperDirt + sclang into the flake**
  ([#23](https://github.com/lambdasistemi/wav2tidal/issues/23)) — the
  audio path still depends on two env vars pointing at a bespoke store
  path; reproducibility says pin it.
- **Event-line NRT FX chains**
  ([#24](https://github.com/lambdasistemi/wav2tidal/issues/24) residue) —
  scenes already chain per-voice FX in NRT; the v2 event-line path still
  renders bare defs. Closing this widens the deterministic fast path.
- **Upstream SuperDirt report** — the pause/resume bug class (paused
  global-effect nodes never sound again after resume; two confirmed
  instances: orbit delay, monitor). Evidence is written up in the R7
  addenda; needs maintainer-eyes review before submission.

## Candidates worth trying

- **Loudness normalization beyond peak** — peak-normalize to −1 dBFS is
  honest but crude; LUFS-based leveling would make descriptors more
  comparable across dense vs sparse material. Cheap to try, measurable
  effect on descriptor stability.
- **Richer movement descriptors** — `motion` covers the centroid trend;
  candidates: loudness-envelope shape, modulation-rate buckets
  (slow drift vs audible wobble), stereo-width motion. Each new field
  must stay bucketed and learnable — the descriptor is a language, not a
  feature dump.
- **MERT embedder (opt-in)** — better music-style fidelity than CLAP in
  the literature, but CC-BY-NC weights; already designed as an explicit
  opt-in flag, never the default.
- **Evolution operators beyond point mutation** — crossover between
  scenes (exchange voices), trajectory-phase nudges, and
  temperature-controlled mutation size; the scene space was designed to
  make these smooth.
- **Vowel formants in the scene chain** — `dirt_vowel` needs the Vowel
  quark's formant tables inlined into our synth graphs; currently the
  one renderable gap in the per-voice FX vocabulary.
- **Custom SynthDefs in the palette** — the grammar and validator
  already admit them; what's missing is a user workflow (drop an `.scd`,
  declare its params) and range metadata for the table.
- **Fleet autoscaling** — pick `fleet_size`/`max_workers` from CPU count
  and measured per-job cost instead of config knobs.
- **Descriptor-conditioned sampling at eval** — the model currently
  decodes greedily; nucleus sampling under grammar constraints would
  trade exactness for diversity, which the evolutionary loop may prefer
  as a proposal distribution.

## Standing constraints (what we will *not* change lightly)

- One grammar, three consumers — no private dialects.
- The render engine is the live engine (one-engine principle).
- CI never needs GPU or audio hardware; local smoke gates guard those.
- Every corpus artifact carries its reproducibility contract; renders
  that cannot be faithful are refused, not fudged.
