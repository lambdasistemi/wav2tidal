# Research: Corpus-to-Live Pipeline (wav2tidal v1)

**Date**: 2026-07-02
**Method**: Six parallel research tasks, each required to verify claims
against primary sources (nixpkgs source + `nix eval` on the target box,
HuggingFace model cards/licenses, TidalCycles/SuperDirt source, PipeWire
docs + local `pw-record`/`pw-link`, librosa docs). Every decision below
carries a verification status. **NEEDS-HARDWARE-TEST** items are the ones
the FR-018 smoke-test gate exists to close before they are trusted.

Target machine ("sesimbra"): AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU
(RDNA 3.5, ROCm arch **gfx1151**), 128 GB unified RAM (~124 GB GTT),
NixOS. Memory-bandwidth-bound (~215–256 GB/s), not compute-bound.

---

## R1 — Audio style embeddings & similarity measure

**Decision**: Primary embedder **LAION-CLAP `laion/larger_clap_music`
(Apache-2.0)** via HF `transformers` `ClapModel`, run **on CPU**. Style
descriptor = CLAP 512-d vector **concatenated** with a hand-crafted block
(pooled MFCC/chroma/tempo/spectral) that injects explicit rhythm/key,
which CLAP encodes weakly. Similarity = cosine over the descriptor.
MERT-v1-95M is a documented **opt-in** upgrade for style fidelity but is
**CC-BY-NC-4.0** (non-commercial) — gated behind an explicit flag.

**Rationale (verified 2026-07-02)**:
- `laion/larger_clap_music` = **Apache-2.0**, 512-d embedding, **48 kHz**
  input, `get_audio_features()` returns a fixed vector (internal
  frame pooling). VERIFIED (model card + transformers `configuration_clap.py`).
- MERT-v1-95M (768-d, 24 kHz) / MERT-v1-330M (1024-d, 24 kHz) =
  **CC-BY-NC-4.0**. MuQ (better still) = code MIT / **weights CC-BY-NC-4.0**.
  VERIFIED (model cards / GitHub).
- MS-CLAP = MS-PL (permissive) but needs a non-transformers package;
  little style gain over LAION. VERIFIED.
- Run on CPU: clips are short, CLAP is small, and CPU removes gfx1151
  ROCm fragility from the hot path. GPU reserved for training only.

**Rejected**: MERT/MuQ as *primary* (NC license); PANNs/Wav2CLIP (sound-
event, not style); OpenL3 (older TF stack); Essentia features (AGPL +
not packaged, see R6).

**Offline pinning**: pin exact commit SHA in `from_pretrained(...,
revision=SHA)`; project-local `HF_HOME`; enforce `HF_HUB_OFFLINE=1` +
`TRANSFORMERS_OFFLINE=1` / `local_files_only=True` at runtime (no
phone-home). Wrap the model blob as a Nix fixed-output derivation.

**Verification**: licenses/dims/rates VERIFIED per-checkpoint. "CLAP
similarity leans semantic not producer-style" — reasoned; mitigated by
the hand-crafted block and the MERT opt-in. Style-quality-sufficiency is
gated by **SC-002** (user ear check) before any training.

**Open risks**: NC weights (keep CLAP default); sample-rate discipline
(48 kHz CLAP vs 24 kHz MERT — never mix descriptors across rates);
style-vs-content mismatch.

---

## R2 — PyTorch + ROCm on gfx1151 (training only)

**Decision**: **nixpkgs `python3Packages.torchWithRocm` narrowed to
gfx1151**, plain **LoRA in bf16 (NOT QLoRA / NOT bitsandbytes)**.
Fallback: AMD/TheRock gfx1151-specific PyTorch wheels in an FHS env
(`buildFHSEnv`/`nix-ld`).

**Rationale (verified by `nix eval` on the box + source read)**:
- `python3Packages.torchWithRocm.version` → **2.11.0**; `rocmPackages`
  pin is **7.2.2** (prior memory said 7.2.3 — corrected). VERIFIED.
- nixpkgs `clr` default `gpuTargets` includes `"1151" # Strix Halo`;
  `rocblas`/`hipblaslt`/`miopen` all list gfx1151. Build knob:
  `torchWithRocm.override { gpuTargets = [ "gfx1151" ]; }` (exported as
  `PYTORCH_ROCM_ARCH`). VERIFIED (source lines cited in task output).
- ROCm 7.0 enabled gfx1151; 7.2 = "Preview" (PyTorch/Linux). Native
  build needs **no `HSA_OVERRIDE_GFX_VERSION`** (that legacy override can
  *hurt* a native stack). VERIFIED (ROCm notes + tinycomputers.io
  2026-02-18 working run).
- bitsandbytes prebuilt archs exclude gfx1151; QLoRA unnecessary at
  128 GB RAM for a ≤1.5 B model. VERIFIED (bnb releases / issue #1608).

**Rejected**: QLoRA/bitsandbytes (broken arch, unneeded); HSA override as
default; generic `whl/rocm7.x` wheels (gfx1151 not in official matrix);
kyuz0 podman image (kept as tertiary turnkey fallback).

**Verification**: attribute existence, versions, gfx1151 targets, no-
override → VERIFIED. **`torchWithRocm 2.11.0` actually compiling+running
on gfx1151, and the PEFT/TRL LoRA step → NEEDS-HARDWARE-TEST** (no long
builds run; this is precisely FR-018's smoke test).

**Open risks**: unverified/long source build (fall back to wheels+FHS);
"Preview" rough edges; ROCm version drift (pin everything); wheels need
FHS on NixOS.

**Smoke test** (FR-018): (a) `torch.cuda.is_available()` True +
`torch.version.hip` set + a 4096² matmul on `cuda`; (b) one LoRA
`backward()`+`step()` on Qwen2.5-0.5B in bf16, `attn_implementation=
"eager"`, no bitsandbytes. Full code in the R2 task output.

---

## R3 — Model: descriptor → pattern text

**Decision (with a challenged assumption — see below)**:
- **Baseline / data generator**: a **procedural pattern generator** (pure,
  seeded, 100 % valid). Required infrastructure regardless — it *is* the
  synthetic-dataset generator (FR-011).
- **Learned model — CHOSEN (user, 2026-07-02): `byt5-small` seq2seq
  (Apache-2.0)**, a tokenizer-free byte-level string→string model — the
  architecturally correct fit for descriptor-text → mini-notation that
  never mangles pattern punctuation (`~ * [ ] < > / .`). Full fine-tune in
  bf16 (~300 M params, comfortable on 128 GB; LoRA optional). Trains on
  the torch-ROCm/gfx1151 substrate (R2); FR-018 smoke test still gates it.
- **Learned model — REJECTED for v1: LoRA on a tiny instruct LLM**
  (`Qwen2.5-0.5B-Instruct`/`SmolLM2-360M-Instruct`, Apache-2.0). Its only
  edge — free-form natural-language descriptor understanding — is out of
  v1 scope (targets are audio, not text). Kept as a documented future
  option if text prompting is added.
- **Output validity: grammar-constrained decoding** via `outlines`
  (Lark/EBNF CFG) with `lm-format-enforcer` as fallback. This — not the
  model — is what delivers the ≥95 %-valid gate (realistically ~100 %
  syntactic validity; musical acceptability is a separate eval).
- **Descriptor encoding**: flat **text template with bucketed numeric
  fields** (`tempo=128 density=hi key=Am brightness=3/5 …`). Do NOT feed
  raw float embeddings to a text model; if embedding conditioning is
  needed, use nearest-neighbour corpus-descriptor text or k-means token
  codes.

**Challenged assumption (surfaced to user)**: the dataset is synthetic,
so training a model to reproduce the generator `f` is partly circular;
its value is *generalization/variation beyond `f`*. The procedural
generator must exist either way. Whether the learned layer is ByT5
(recommended) or a LoRA'd LLM (user's stated interest) is a genuine fork.

**Licenses VERIFIED**: Qwen2.5-0.5B/1.5B, SmolLM2-360M/1.7B, byt5-small,
t5-small = Apache-2.0; Phi-3.5-mini = MIT (oversized at 3.8 B);
**Llama-3.2-1B = restrictive custom license (naming/attribution/700 M MAU)
→ rejected**.

**Toolchain VERIFIED**: `trl` `SFTTrainer` ↔ `peft` LoRA integration;
AMD publishes an official ROCm TRL fine-tune tutorial. `outlines` CFG is
transformers-native + offline but flagged "experimental" (LALR(1) — keep
the grammar unambiguous). Determinism knobs: `set_seed`,
`full_determinism`, `torch.use_deterministic_algorithms`,
`CUBLAS_WORKSPACE_CONFIG`.

**Open risks / NEEDS-HARDWARE-TEST**: deterministic *training* on ROCm
gfx1151 (inference determinism is easy: greedy + grammar; fallback = CPU
training at 360 M–0.5 B, feasible on this box); pin the `outlines`
version (API churn).

---

## R4 — Audio slicing & feature extraction (ingestion)

**Decision**: `soundfile` (WAV IO) + `librosa` 0.11.0 (DSP/beats/features)
+ `soxr` (resample) — **all present in nixpkgs on the box** (VERIFIED by
`nix eval`). **Essentia is NOT packaged** (VERIFIED) → deferred.

**Slicing (deterministic, two-tier + fallback)**: compute
`onset_strength` once → primary beat boundaries via `beat_track(units=
'time')` → onset-refined cuts via `onset_detect(backtrack=True)` →
fallback for beat-ambiguous material: fixed subdivisions of estimated
tempo, or energy segmentation via `effects.split(top_db=…)`. Never emit
silent slices.

**Feature → descriptor map (VERIFIED signatures)**: tempo=`beat_track`;
rhythmic density=onset rate (events/s); tonal/key=`chroma_cqt` +
hand-rolled Krumhansl–Schmuckler (librosa has **no** key fn); spectral=
`spectral_centroid/bandwidth/rolloff/contrast/flatness`; timbre=`mfcc`
(20). Per-frame features → fixed-length per-clip vectors via **mean/std
pooling**.

**Normalization/validation**: mono downmix; `resample(res_type=
'soxr_hq')` to a single `TARGET_SR` **config constant shared with R1**
(CLAP 48 kHz / MERT 24 kHz — pick per active embedder to avoid double
resample); silence via RMS; clip via `|y|≥1-eps` fraction; corrupt via
`soundfile.LibsndfileError` caught at the IO edge only.

**Purity**: all DSP is pure numpy `array→features`, deterministic with
fixed params + pinned versions (no RNG in this path; librosa's only RNG
is in `decompose`, unused). Test fixtures generated in-memory: 120-BPM
click track (beat/tempo), 440 Hz sine (chroma/key/centroid),
zeros/saturated (silence/clip), random bytes (corrupt).

**Open risks (VERIFIED)**: **librosa exposes no tempo confidence** →
derive a proxy (tempogram/PLP peak strength); **key must be hand-rolled**
(K–S, moderate accuracy — report with correlation strength); essentia
packaging is a real effort item if higher accuracy is later needed.

---

## R5 — Driving TidalCycles & SuperDirt

**Decision**: control Tidal by **writing statements to a `ghci` process's
stdin**, bootstrapped with `BootTidal.hs`, each block wrapped in GHCi
`:{ … :}` markers — exactly what the editor plugins do. This preserves
the full pattern engine + mini-notation. Supplement with Tidal's built-in
**OSC control listener on 127.0.0.1:6010** (`/ctrl /setcps /mute /hush`)
for real-time parameter automation. For cycle/tempo awareness: query
`streamGetNow`/`streamGetCPS` in GHCi, or (cleaner) share Tidal's
**Ableton Link** clock.

**Cycle-boundary swaps (VERIFIED from source)**: default `d1 $ …` =
`streamReplace` = **immediate**, not quantized. On-cycle swaps use
`jumpIn' 1 $ …` / `xfade` (`Transition.hs` aligns to `nextSam now`).
This directly implements FR-020 / SC-005 / SC-007.

**Bank layout (VERIFIED from `DirtSoundLibrary.sc`)**: one folder per
bank; **folder basename = sample name**; files **sorted alphabetically,
0-indexed** = the `:n` selector (out-of-range wraps). Registered via
`~dirt.loadSoundFiles("/path/banks/*")` in `superdirt_startup.scd`.

**Direct OSC to SuperDirt (57120, `/dirt/play`, named key/value pairs in
a timestamped bundle)** is a documented escape hatch, kept only for
one-shot use — rejected as primary (would reimplement the pattern engine).

**Mini-notation subset (VERIFIED against docs)** — the "pattern subset"
(FR-008): sequence, `~` rest, `[]` group, `.` group-shorthand, `*`/`/`
speed, `<>` alternation, `,` stack, `!` replicate, `_`/`@` elongate, `?`
probability, `(3,8)` euclid, `{}` polymeter; combinators `s n sound # speed
gain pan orbit every fast slow rev jux chunk`; swap verbs `d1..d16
jumpIn' xfade once setcps hush`.

**Open risks**: GHCi has no structured error channel (scrape stdout/
stderr, may need interpreter restart — a real robustness task); default
`d1` is not boundary-aligned (must use `jumpIn'`/`xfade` + shared clock or
drift); `oLatency` look-ahead (0.05–0.2 s) must be honored; **WAV format
requirements are UNDOCUMENTED** (validate empirically — safe assumption:
16/24-bit PCM, mono/stereo, any rate); SuperDirt moved to Codeberg
(pin `codeberg.org/musikinformatik/SuperDirt`).

---

## R6 — Capturing the agent's own audio (live session)

**Decision (USER, 2026-07-02): live capture is authoritative.** The
evolution loop scores the **live-captured** audio of the agent's own
output — the research team recommended scoring the offline render
instead (FR-013 makes them equivalent, and capture adds xruns/latency/
device-naming fragility while weakening reproducibility), but the user
chose fidelity to "listens to its own output" over the shortcut. The
offline render (below) is retained as the dataset-generation signal
(FR-012) and as a fallback/cross-check when capture is unavailable or to
validate that offline≈live (FR-013). The capture path below is therefore
on the hot path, not just telemetry — its robustness (xruns, cycle
alignment, device naming) is a first-class implementation concern.

**Capture path (VERIFIED locally on PipeWire 1.6.2)**: route scsynth to a
**dedicated PipeWire null sink** (isolation by construction) via
`SC_JACK_DEFAULT_OUTPUTS` or `pw-link`, then capture its `.monitor` with
**`pw-record --target=<sink>.monitor -n <sample_count>`** → WAV →
`soundfile`. `sample_count = round(n_cycles / cps * rate)`. In-process
alternative: `soundcard` (libpulse, `include_loopback=True`).

**nixpkgs reality (VERIFIED by `nix eval`)**: `sounddevice` 0.5.3,
`soundcard` 0.4.5, `soundfile`, `resampy` present; **`JACK-Client` and
`pipewire-python` NOT packaged** → JACK path rejected for v1; call
`pw-record` directly.

**Rejected**: JACK-Client (not packaged, callback ring-buffer complexity);
sounddevice/PortAudio for monitor selection (ALSA backend can't reliably
see `.monitor` — flagged, reasoned not proven); live capture as the
authoritative score (redundant per FR-013).

**Open risks**: device/port-name instability (fixed `sink_name`, resolve
ports at runtime via `pw-dump`); the whole render-scoring recommendation
rests on FR-013 holding — the capture verification channel must measure
and gate on it before render-based scores are trusted.

---

## Cross-cutting decisions

- **Language**: Python (per user); pure DSP/pattern/render core + impure
  IO shell (constitution I).
- **GPU only for training.** Embedding (R1) and all DSP (R4) run on CPU;
  scoring uses the CPU offline renderer (R6). gfx1151/ROCm risk is
  confined to the one training stage, gated by the FR-018 smoke test.
- **Shared `TARGET_SR` constant** couples R4 ↔ R1; **shared event-timing
  core** couples the offline renderer (R4/R6) ↔ live playback (R5) per
  FR-013.
- **Everything offline after one-time setup**: HF model SHAs pinned as Nix
  FODs; `*_OFFLINE=1` at runtime.

## Spec-affecting findings (feed back into spec per workflow rule)

1. **FR-021 RESOLVED (user)**: live capture stays authoritative; offline
   render is dataset signal + fallback/cross-check. Capture robustness is
   on the hot path. FR-021 stands as written — no spec edit.
2. **Model fork (FR-015)**: research recommends **ByT5 seq2seq** over the
   originally-stated **LoRA'd LLM**. Needs a user decision (surfaced
   separately). The procedural generator is required either way.
3. **New explicit sub-requirement**: a versioned **pattern-subset EBNF
   grammar** artifact (FR-008) is the lynchpin shared by validator (FR-009),
   generator (FR-011), and constrained decoding (R3) — call it out in the
   data model and contracts.
