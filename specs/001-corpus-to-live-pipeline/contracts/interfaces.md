# Contract: internal interfaces & external system boundaries

The pure-core function shapes (constitution I) and the exact external
protocol strings verified in research (R5/R6). Signatures are conceptual;
implementation binds types. These are the seams tests target directly.

## Pure core (deterministic, numpy in/out — unit-tested with synthetic fixtures)

```
# Ingestion (R4)
read_wav(path)                         -> (y: f32[], sr)        # IMPURE edge
normalize(y, sr, target_sr)            -> f32[]                 # pure
slice_boundaries(y, sr, hop, strategy) -> times[]              # pure, beat/onset/fallback
slice_features(clip, sr)               -> descriptor_block      # pure, pooled
track_descriptors(y, sr)               -> {tempo, tempo_conf, key, key_strength, …}

# Embedding / similarity (R1) — model load is IMPURE; the math is pure
embed(audio, sr, embedder)             -> f32[512]              # CLAP, CPU
assemble_descriptor(embedding, blocks) -> StyleDescriptor
similarity(a: StyleDescriptor, b)      -> float                 # cosine, pure
nearest(query, profile_index, k)       -> [(id, score)]

# Pattern subset (R3/R5) — the single grammar artifact (FR-008)
validate(pattern_text, subset, banks)  -> {valid, reason?, events, depth}  # pure
generate_pattern(rng, subset, banks, diversity) -> pattern_text            # pure, seeded
mutate(pattern_text, rng, subset)      -> pattern_text          # pure, seeded

# Offline render (R4/R6) — the scoring signal, shares timing core with live
schedule_events(pattern_text, cps, n_cycles) -> [Event]        # pure — SHARED with live (FR-013)
render(events, banks, sr)              -> f32[]                 # pure mixdown (rate/gain/pan)
```

Determinism: every `pure` function is a mathematical function of its
inputs given pinned library versions; seeded functions take an explicit
`rng`. No wall-clock, no global state. `schedule_events` is the shared
timing core that FR-013 requires the live path to reuse.

## External boundary — TidalCycles via GHCi stdin (R5, VERIFIED)

Write blocks to the ghci child's stdin, each wrapped:
```
:{
<one Tidal statement>
:}
```
- Play a bank:            `d1 $ s "mybank"`
- Indexed + fx:          `d1 $ s "mybank:3" # gain 1.1 # speed 1 # pan 0.5`
- **On-cycle swap**:      `jumpIn' 1 $ s "mybank(3,8)"`   (NOT bare `d1` — immediate)
- Tempo:                  `setcps 0.5`
- Silence:                `hush`
- Cycle/tempo read-back:  `streamGetNow tidal >>= print` (scrape stdout) or share Ableton Link.
Robustness: no structured error channel — parse stdout/stderr, restart
interpreter on a bad prompt.

## External boundary — SuperDirt banks on disk (R5, VERIFIED)

```
banks/<bankname>/00_kick.wav   # basename = sample name; alpha sort = :0
             /01_snare.wav   # :1  … out-of-range :n wraps
```
Registered once in `superdirt_startup.scd`:
`~dirt.loadSoundFiles("<abs>/banks/*");`
WAV format: 16/24-bit PCM, mono/stereo, any rate (UNDOCUMENTED upstream —
validate empirically at ingest).

## External boundary — direct OSC to SuperDirt (R5, escape hatch only)

`127.0.0.1:57120`, address `/dirt/play`, named key/value pairs in a
timestamped bundle (add `oLatency` look-ahead):
`["s","mybank","n",3,"speed",1.0,"gain",1.0,"pan",0.5,"orbit",0,"cps",0.5]`

## External boundary — Tidal OSC control (R5, VERIFIED)

`127.0.0.1:6010`: `/setcps <f>`, `/mute <i>`, `/hush`, `/ctrl <name> <val>`.
Real-time parameter automation only — cannot define patterns.

## External boundary — audio capture (R6, VERIFIED locally)

Route scsynth → dedicated null sink, capture its monitor:
```
pactl load-module module-null-sink sink_name=tidal_out
export SC_JACK_DEFAULT_OUTPUTS="tidal_out:playback_FL,tidal_out:playback_FR"
pw-record --target tidal_out.monitor --rate 48000 --channels 2 \
          --format f32 -n <round(n_cycles/cps*rate)> win.wav
```
Used as the **verification channel** (does live ≈ offline render, FR-013),
not the authoritative evolution score (that comes from `render`, R6).
