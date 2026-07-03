# Descriptors, matching, and the model

## Descriptors: the shared language about sound

Every rendered clip is summarized as a compact, bucketed text — the
model's input surface:

```text
tempo=122 density=mid key=C#m brightness=4/5 motion=rising
```

Buckets are deliberate: a learnable, stable vocabulary beats raw floats
for a text model. `motion` (rising / falling / steady / wobbly, from the
spectral-centroid trend and its wobble) is what makes *movement*
describable — without it, a filter sweep and a static timbre of equal
average brightness are indistinguishable, and the model could never
learn to reach for a `ramp`.

## Matching sounds: two mechanisms

**Sound ↔ sound — the score.** Each clip becomes a vector:

- a **CLAP** embedding (`laion/larger_clap_music`, Apache-2.0, CPU):
  512 dimensions in a space trained on (audio, caption) pairs — two
  sounds are close when they would be *described* similarly. This
  carries timbre and texture.
- a **handcrafted block**: pooled MFCCs, chroma, tempo/onset statistics
  — rhythm and key, which CLAP encodes weakly (verified).

Similarity is the cosine between vectors. Ear-verified in US1: querying
the corpus with a track ranks genuinely similar tracks above dissimilar
ones. The honest limitation: cosine-over-CLAP measures "the same kind of
sound", not "the identical sound" — for steering a live instrument
toward a *character*, that is a feature.

**Sound ↔ config — the model.** No direct audio comparison: the dataset
describes each render, ByT5 learns description → config, and at play
time the target sound goes through the same describer. Live (US3), the
two compose: the model proposes, the engine plays, capture-and-cosine
judges, evolution refines.

## Training

- **Model**: `google/byt5-small` (300 M params, byte-level — pattern
  punctuation like `~ [ ] # :` can never be mangled by a tokenizer),
  pinned by commit, cached project-locally.
- **Data**: (descriptor → config-text) pairs from the hybrid dataset —
  parameter scenes and event lines, each carrying its renderer and kind,
  under an explicit reproducibility contract.
- **Loop**: a compact seeded manual loop — full fine-tune, **fp32 master
  weights + bf16 autocast** on the gfx1151 GPU. The full-bf16 cast that
  seems natural silently stalls AdamW (updates below one bf16 ulp are
  lost); the symptom is a loss plateau *without* overfitting and greedy
  repetition collapse. Fixing precisely this took grammar validity from
  5% to 80% on the first corpus.
- **Determinism**: seeded end to end — identical losses to four decimal
  places across runs.
- **Loss floor**: config bytes are dominated by 3-significant-figure
  parameter values the descriptor genuinely underdetermines, so
  cross-entropy cannot approach zero. Validity, not loss, is the gate.

## Evaluation: validity is the metric

Held-out descriptors are decoded greedily and scored by the *same*
grammar and validator as everything else:

| metric | meaning |
|---|---|
| `grammar_valid` | parses as a sentence of the language |
| `validator_valid` | semantically valid as-is (inventory, ranges, bounds) |
| `repaired_valid` | valid after the minimal repair pass |
| `exact_match` | byte-identical to the reference (diagnostic only) |

First-corpus baseline (400 pairs, 40 epochs): **80% grammar-valid, 80%
repaired-valid**; the residual 20% are unterminated generations — a
termination problem, not a grammar one. The designed route to ≥95% is
grammar-constrained decoding plus a larger corpus (see the
[Roadmap](roadmap.md)).
