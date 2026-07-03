# wav2tidal

Learn a *sound* from local WAV recordings and play it live through
[TidalCycles](https://tidalcycles.org)/[SuperDirt](https://github.com/musikinformatik/SuperDirt)
— composing **in parameter space**: sustained synth voices whose
parameters move along shapes over time, with rhythmic sample layers on
top.

Everything runs locally and offline on one machine (an AMD Strix Halo
box: gfx1151 GPU for training, PipeWire for audio). The corpus never
leaves disk.

## What it does

1. **Ingest** — beat-slice a WAV corpus into SuperDirt-loadable sample
   banks and compute a style profile (CLAP + handcrafted descriptors).
2. **Compose-able language** — a versioned grammar of *configs*: event
   lines (`d1 $ s "bd(3,8)" …`) and parameter **scenes** (voices +
   modulation trajectories). Everything the system plays or learns is a
   sentence of this language.
3. **Render** — turn configs into audio through the *same SuperDirt
   engine used live*: a pure numpy mixer for plain sample lines, headless
   deterministic SuperCollider NRT for synth material, and real-time
   capture of a booted SuperDirt (full FX chain) for everything else — in
   parallel, across a fleet of instances.
4. **Learn** — describe each rendering in a compact bucketed text and
   fine-tune a byte-level seq2seq model (ByT5) to invert it:
   *description of a sound → config that produces it*.
5. **Play (next)** — a live evolutionary agent: the model proposes, the
   engine plays, the capture is scored against the target sound, and
   mutation walks the config toward it.

Start with the **[Architecture](architecture.md)** overview; the
**[Roadmap](roadmap.md)** lists what we will try next.

## Status

**User Story 2 is complete** (2026-07-03): the full corpus → dataset →
trained-model pipeline runs end to end on the box. Design history and
verified research live under
[`specs/001-corpus-to-live-pipeline/`](https://github.com/lambdasistemi/wav2tidal/tree/main/specs/001-corpus-to-live-pipeline)
— the two design-change ADRs
([sound-first](https://github.com/lambdasistemi/wav2tidal/blob/main/specs/001-corpus-to-live-pipeline/design-change-001-sound-first.md),
[parameter scenes](https://github.com/lambdasistemi/wav2tidal/blob/main/specs/001-corpus-to-live-pipeline/design-change-002-parameter-scenes.md))
are the best short reads on *why* the system is shaped this way.

## Development

```bash
nix develop        # CPU dev shell (pure core, tests, CI parity)
just ci            # format + lint + tests, mirrors GitHub CI exactly
just smoke-audio   # hardware gate: NRT determinism + RT capture (local only)
just smoke-gpu     # hardware gate: ROCm + a real bf16 training step
just train         # ByT5 fine-tune (inside `nix develop .#training`)
```

The default shell is CPU-only; CI never touches ROCm or SuperCollider.
GPU and audio behaviour are guarded by the two local smoke gates.
