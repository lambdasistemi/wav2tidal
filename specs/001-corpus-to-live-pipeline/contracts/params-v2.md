# Grammar v2 — synth + params + FX action space (R7 param table)

**Source of truth**: syntax in [`grammar/pattern_subset.lark`](../../../grammar/pattern_subset.lark)
(v2.0.0), semantics in [`core/pattern/params.py`](../../../src/wav2tidal/core/pattern/params.py).
A unit test (`tests/unit/test_params.py`) keeps the two vocabularies in sync.
This document is the human-readable record with provenance.

**Provenance**: *defaults* and *hard clips* are read from the SuperDirt
quark v1.7.3 sources on the box (`synths/core-synths.scd`,
`synths/core-modules.scd`, `synths/core-synths-global.scd`,
`library/default-synths-extra.scd`, `used-parameters.scd`) — see research
§R7. *Sampling ranges* are our chosen action space: musically sensible,
inside every hard clip, and bounded so the generator is
valid-by-construction (FR-010) and the model's target space is learnable.

## Config text form

One config = one full Tidal line (grammar start rule `line`):

```
d1 $ s "supersaw supersaw:7 ~" # note 7 # cutoff 1200 # resonance 0.3 # room 0.4
```

- **Sources** in the mini-notation are `NAME(:INT)` events: a Super* synth,
  an ingested sample bank, or a custom SynthDef — which names exist is a
  validator inventory (`validate.Sources`), not syntax. The `:INT` selector
  is the sample index on banks and the `n` knob on synths (bounded ≤ 24).
- **Controls** (`# param value`) apply to every event of the line —
  "per-event" means event-scope (each `/dirt/play` carries them, and each
  event gets its own FX synth chain), as opposed to the orbit-level global
  sends. Per-control patterns (`# cutoff "400 800"`) stay deferred.
- v1 lines (`d1 $ s "bd sn" # gain 1 # speed 1 # pan 0.5`) remain members
  of the language; the mini-notation subset is unchanged from v1.

## Core event params (any source)

| param | kind | range | notes (source facts in *italics*) |
|---|---|---|---|
| note | cont | -24..24 | semitones; *SuperDirt freq = midicps(note + 60) at default octave 5* |
| n | int | 0..24 | drum-synth pitch knob / sample index (usually via `:n`) |
| gain | cont | 0.5..1.3 | *dirt_gate: amp × (gain min 2)⁴* |
| pan | cont | 0..1 | *DirtPan* |
| speed | log | 0.25..4 | sample rate factor; *synths fold it into DirtFreqScale* |
| accelerate | cont | -2..2 | pitch glide over the event |
| attack | cont | 0..0.5 s | *activates dirt_envelope (with release)* |
| release | cont | 0.02..2 s | |

`sustain` is deliberately **not** in the action space: event duration comes
from the scheduler slot (the renderer mapping injects it).

## Per-event FX params

Each activates its `dirt_*` effect synth for that event when set
(*activation condition in core-modules.scd*).

| param | kind | range | effect | source facts |
|---|---|---|---|---|
| cutoff | log | 60..12000 Hz | dirt_lpf | *clip 20..SR/2*; resonance rides along |
| resonance | cont | 0..0.8 | dirt_lpf rq | *linexp 0→1 to rq 1→0.001*; also a direct synth arg (see overrides) |
| hcutoff | log | 60..12000 Hz | dirt_hpf | *clip 20..SR/2* |
| hresonance | cont | 0..0.8 | dirt_hpf | |
| bandf | log | 60..12000 Hz | dirt_bpf | *clip 20..SR/2* |
| bandq | log | 1..50 | dirt_bpf | *floored at 1, default 10* |
| shape | cont | 0..0.95 | dirt_shape waveshaper | *clamped < 1 (division by zero)* |
| crush | cont | 1..16 | dirt_crush | *bit depth: round(0.5^(crush−1))* |
| coarse | int | 1..32 | dirt_coarse | *activates only when > 1 (1 = full rate)* |
| vowel | choice | a e i o u | dirt_vowel formants | *fixed per event; blends with resonance* |

Not exposed in v2 (documented, not silently missing): `hold`, `curve`,
`tilt`/`plat` (grain envelope), `psrate`/`psdisp` (pitch shift),
`tremolorate`/`tremolodepth`, `phaserrate`/`phaserdepth`, `timescale`
(stretch sample player), `cut`/`orbit`/`channel` routing.

## Global FX sends (orbit scope — RT path only, R7 tier 2)

| param | kind | range | source facts |
|---|---|---|---|
| room | cont | 0..1 | reverb feed level (*"size is not room size, just the feed level"*); activates dirt_reverb |
| size | cont | 0..1 | *linexp to loop depth 0.01..0.98* |
| delaytime | cont | 0.02..1 s | *hard clip 0..4 s*; sampling range keeps echoes audible in short renders |
| delayfeedback | cont | 0..0.9 | *comb variant clips 0..0.99* |

No NRT support (SuperDirt global FX use Rand/LFNoise) — these only sound
through `rt_render`. Not exposed: `dry`, `delaySend`/`delayAmp`, `lock`,
`leslie`/`lrate`/`lsize`.

## Synth-specific params

Default specs (per-synth overrides below): voice 0..1 · semitone 0..24 ·
lfo 0..4 · pitch1 0.25..8 (log) · pitch2 1..4 · pitch3 1..6 · rate
0.25..4 (log) · decay 0..1 · detune 0..2 · slide -4..4 · velocity 0..1.2 ·
muffle 0..2 · stereo 0..1 · modamp 0..2 · modfreq 2..12 · vibrato 0..1 ·
vrate 1..10 · perc 0..1.2 · percf {2,3} · lfofreq 0.1..10 (log) ·
lfodepth 0..0.5

One event namespace: `resonance` reaching a supersaw is the same event
param that configures dirt_lpf; a synth's range override applies to any
param it lists, and a value must fit **every** listing synth in the line
(`params.effective_range`).

The broad palette (`library/default-synths-extra.scd`), params beyond
core+FX, with `(lo..hi)` where overridden:

| synth | params |
|---|---|
| supermandolin | detune (0..3) |
| supergong | voice (0..4), decay (0..2) |
| superpiano | velocity, detune (0..1), muffle, stereo |
| superhex | rate |
| superkick | pitch1 (0.25..4), decay (0.25..2) |
| super808 | rate (0.25..4), voice (0..2) |
| superhat | — |
| supersnare | decay (0.25..2) |
| superclap | rate (0.25..4), pitch1 (0.25..4) |
| supersiren | — |
| supersquare | voice (0.05..0.95 — *width 0/1 is silent*), semitone, resonance, lfo, pitch1, rate, decay |
| supersaw | voice, semitone, resonance, lfo, pitch1, rate, decay |
| superpwm | voice, semitone, resonance, lfo, pitch1, rate, decay |
| supercomparator | voice (0..5), resonance, lfo, pitch1, rate, decay |
| superchip | slide, rate (0.25..4), pitch2, pitch3, voice |
| supernoise | voice, slide (0..4), pitch1, rate (0.25..4), resonance (0..1) |
| superfork | — |
| superhammond | voice (0..9 — *drawbar presets*), vibrato, vrate, perc, percf, decay |
| supervibe | decay, velocity (0..1.5), modamp, modfreq, detune (0..1) |
| superhoover | slide, decay |
| superzow | slide (0.25..4), detune (0..3), decay |
| superstatic | — |
| supergrind | detune (0..10 — *Hz here*), voice, rate (0.25..4) |
| superprimes | detune (0..1), voice (0..2), rate (0.25..4) |
| superwavemechanics | detune (0..1.5 — *source min's at 1.5*), voice, resonance (0..1) |
| supertron | voice, detune (0..5) |
| superreese | voice (0..2), detune |
| superfm | voice (0..5 — *presets; 0 = user-defined*), lfofreq, lfodepth |
| soskick | pitch1 (0..2000 — *mod freq Hz*), voice (0..4), pitch2 (0..1 — *noise amp*) |
| soshats | pitch1 (50..1000), resonance (0..1) |
| sostoms | voice (0..2) |
| sossnare | voice (0..2), semitone (0.1..2 — *a frequency ratio here*), pitch1 (500..4000 Hz), resonance (0..1) |

`superfm`'s full 6-operator matrix (`amp1..6`, `ratio1..6`, `mod11..66`,
`eglevel*`, `egrate*`) is out of the v2 action space — presets + pitch LFO
only. Custom SynthDefs get core + event-FX params only (their own args are
unknown to the table).

## Renderer mapping (`core/pattern/dirt.py`)

- **RT** (`rt_params` → `io/superdirt.py:rt_render`): everything passes
  through one `/dirt/play` — SuperDirt's event routing dispatches the FX
  modules and the orbit picks up the global sends.
- **NRT tier-1** (`nrt_params` → `nrt_render`): the score plays the bare
  source synthdef, so only its own args survive: `sustain` (injected from
  the scheduled slot), `freq` (= midicps(60 + note)), `n`, `pan`, `speed`,
  `accelerate`, and the synth's listed params. Event-FX and global params
  are dropped — chaining `dirt_*` effect synths into the NRT score is
  future renderer work, not part of this contract.
