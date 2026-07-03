# US3 design — the live pursuit loop

**Date**: 2026-07-03
**Status**: Accepted (design session with the user)
**Builds on**: design-change-001 (sound-first), design-change-002
(parameter scenes), and the north-star statements recorded in
`docs/roadmap.md` (Take Five recognizability; fullness/coherence/
perpetual pursuit; the DJ-set operating model).

## The picture

A DJ (or, first, a WAV file standing in for one) feeds music to the
machine. The machine listens in windows, privately auditions candidate
scenes *faster than real time*, plays only winners through a live
SuperDirt, and keeps pursuing — always a little behind, carrying the
input's drama in its own voice. The audience hears **only the machine's
output, delayed**.

## Decisions (user, 2026-07-03)

1. **Shadow audition.** Candidate mutations are rendered offline (NRT,
   ~10× real time, issue #40) and scored privately; only the winning
   scene plays. The audience never hears dead ends.
2. **Free swapping.** Whatever scores best plays next, even if it is a
   jump cut. Coherence lives *within* a scene (shared key/tempo/motion
   across its voices), not in cross-generation continuity. (Crossfades
   or drift-limits can be revisited later by ear.)
3. **File-in, file-out first.** Milestone 1 is a *session replay*: input
   mix WAV → the machine's delayed reinterpretation as a WAV, exercising
   the full loop with no audio-device work. Live capture (PipeWire) and
   live playback come after the loop is proven musical.

## The loop (one generation ≈ one analysis window)

```text
input stream ──window(≈4 s, hop 2 s)──> analysis
    analysis: descriptor text (tempo/density/key/brightness/motion/energy)
              + CLAP∥handcrafted embedding      (the target)
propose:  if score stagnates or the input jumps → ByT5(descriptor) + repair
          else → K mutations of the current scene (mutate_scene)
audition: render all K via NRT scene renders (parallel, seeded)
          score = cosine(candidate embedding, target embedding)
play:     winner → live SuperDirt as a scene graph (same machinery as
          the RT renderer: known nodes, trajectory ticks, own monitor)
verify:   capture what actually played; compare to the shadow prediction
          (FR-021 kept: judging *what played* uses real capture; the
          shadow only screens candidates — A/B tolerance ~7% centroid)
repeat.
```

Timing budget (DJ-set model: delay is a feature): window 4 s, hop 2 s;
K ≈ 8 candidates × ~0.5 s NRT ≈ 2–4 s of shadow work per generation on
the thread pool — the loop sustains a generation every 2–4 s, output
delayed one-to-two windows behind the booth.

## What must never be lost (from the north star)

- **Tempo continuity**: the input's tempo estimate maps to the scene's
  `cps`; layer patterns and `steps` trajectories inherit it.
- **The energy arc**: input loudness/density per window steers ensemble
  gain/voice-count/trajectory depth — a drop in the booth becomes a drop
  in the machine. (Adds an `energy` term next to `motion` in analysis.)
- **Fullness**: proposals default to multi-voice scenes; the mutation
  pool must not collapse the ensemble to one thin voice.

## Components → tickets

- **US3-1 — input analysis stream** (pure + thin IO): windowed
  descriptor + embedding over a WAV (later: a capture stream); tempo and
  energy tracks. Pure functions over arrays; CI-tested on synthetic
  fixtures.
- **US3-2 — live scene player**: a persistent SuperDirt session manager
  (`io/live.py`): boot once (fleet-style ports), spawn/replace scene
  graphs with known nodes, stream trajectory ticks, swap scenes on
  command; also an offline "player" that concatenates winner renders for
  the file-out milestone.
- **US3-3 — the pursuit engine** (pure policy + orchestration):
  propose/audition/select loop over the analysis stream; ByT5 proposal
  with `repair` as send-safety; stagnation/jump detection; session log
  (every generation: target descriptor, candidates, scores, winner).
- **US3-4 — session replay CLI**: `wav2tidal replay --input mix.wav`
  → `out.wav` + session log; the listening test we iterate by ear.

Live input/output (device capture, real-time playback) is US3-5,
deliberately after the replay milestone proves musicality.

## Out of scope here (future chapters, recorded)

- Melody/harmony tracking for Take Five-grade recognizability (north
  star; needs chroma/chord sequences in analysis *and* scoring).
- Grammar-constrained decoding for proposals (#36) — the loop uses
  greedy + repair meanwhile.
- Crossfade/drift continuity policies (revisit by ear after replay).
