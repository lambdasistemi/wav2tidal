# The config language

Everything the system generates, validates, renders, learns, or plays is
a sentence of one versioned grammar. This is the project's central
contract: **one language, three consumers** — the seeded generator emits
it, the validator accepts exactly it, and the model is trained (and will
eventually be constrained) to produce it. No consumer gets a private
dialect.

## Three layers, one file

[`grammar/pattern_subset.lark`](https://github.com/lambdasistemi/wav2tidal/blob/main/grammar/pattern_subset.lark)
(v3) has three start rules, grown over three design generations:

**`mini` — mini-notation** (v1): the Tidal rhythm subset — sequences,
rests, groups, `*`/`/` modifiers, Euclidean rhythms, stacks:

```text
bd:3 ~ [sn sn, hh*2] bd(3,8)
```

**`line` — event lines** (v2, sound-first): a full config: sources
(Super\* synths, sample banks, custom SynthDefs — which names exist is
validator inventory, not syntax) plus `# param value` controls from the
verified SuperDirt vocabulary:

```text
d1 $ s "supersaw supersaw:7 ~" # note 7 # cutoff 1200 # resonance 0.3 # room 0.4
```

**`scene` — parameter scenes** (v3, the primary mode): compose in
parameter space:

```text
scene voice supersaw # note -12 # lfo 0.5
      mod cutoff sine 800 600 0.25
      mod resonance ramp 0.2 0.6
      voice superhammond:3 # vibrato 0.5
      mod room walk 0.4 0.3 0.5 7
      layer d1 $ s "bd(3,8)" # gain 0.9
```

1–4 sustained voices; each `mod` moves one param along a shape; an
optional `layer` reuses the v2 line for rhythmic material. Shape arity
is *syntactic* (a two-argument `sine` does not parse); scene duration is
deliberately **not** in the text — wall-clock belongs to the renderer
and the live session, never to the model.

## Shapes

| shape | args | meaning |
|---|---|---|
| `ramp` | `v0 v1` | linear over the scene (log-space for log params) |
| `sine` | `center depth rate` | LFO, rate in Hz (0.02–8) |
| `walk` | `center depth rate seed` | seeded random walk, reproducible |
| `steps` | `v1 v2 …` | values spread over one cycle, held, repeating |

## Semantics: the param table

Syntax says *what can be written*; the table
([`core/pattern/params.py`](https://github.com/lambdasistemi/wav2tidal/blob/main/src/wav2tidal/core/pattern/params.py),
human-readable form with provenance in
[`contracts/params-v2.md`](https://github.com/lambdasistemi/wav2tidal/blob/main/specs/001-corpus-to-live-pipeline/contracts/params-v2.md))
says *what it means*:

- every param's range, scale (linear/log), kind (continuous / integer /
  choice), and scope (per-event vs global orbit FX);
- per-synth applicability with range overrides for all 32 Super\* synths
  (read from the SuperDirt sources, not guessed);
- one event namespace, as in SuperDirt: `resonance` reaching a supersaw
  is the same param that configures the chained low-pass filter;
- what is *modulatable*: continuous/log params that SuperDirt actually
  reads while a note runs (`gain` is explicitly excluded — `dirt_gate`
  declares it `\ir`).

A unit test pins the grammar's vocabulary to the table's, so the two
sources of truth cannot drift.

## Valid by construction, valid by repair

- The **generator** samples only applicable params inside effective
  ranges and only well-formed shapes: 300 random scenes + 300 chained
  mutations validate in the property tests, by construction.
- The **validator** re-checks everything (membership, inventory, ranges,
  bounds) — model output and hand-written configs get no leniency.
- **Repair**
  ([`core/pattern/repair.py`](https://github.com/lambdasistemi/wav2tidal/blob/main/src/wav2tidal/core/pattern/repair.py))
  coerces *near*-valid texts into the space: truncate to the voice
  bound, drop duplicate/inapplicable mods, clamp values. It exists
  because a generative model's failures are overwhelmingly termination
  artifacts, and because the live loop needs a guarantee that whatever
  it sends is safe.

## Mutation is the search space

`mutate_scene` makes one small validity-preserving move: resample or
reshape a trajectory, nudge a note, tweak a static param. These
operators are deliberately *smooth* — the US3 evolutionary agent walks
this space, and a warm drone should drift, not teleport.
