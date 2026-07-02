# Data Model: Corpus-to-Live Pipeline (wav2tidal v1)

Derived from the spec's Key Entities and the research decisions. This is
the conceptual/persistence model, not a class diagram. Types are stated
abstractly (a later implementation task binds them to concrete
dataclasses/serialization). **Local-only** everywhere (FR-007).

Conventions: `Id` = stable content-addressed or deterministic string id.
All persisted artifacts embed the `(config, seed)` that produced them
(constitution III, FR-014/FR-016).

---

## Storage layout (all gitignored — FR-007)

```
corpus/                       # user-provided WAVs (input, never written)
banks/                        # SuperDirt-loadable sample banks (output)
  <bankname>/00_*.wav …       # basename = sample name; alpha sort = :n index
profile/                      # style profile (descriptors + NN index)
  slices.parquet  tracks.parquet  index.faiss|.npy
datasets/<dataset-id>/        # synthetic training data + embedded config
checkpoints/<run-id>/         # model + training config + eval report
sessions/<session-id>/        # live-session logs
.hf_cache/                    # pinned pretrained models (Nix FOD)
```

---

## Entities

### Corpus
Input directory tree of WAVs. Not persisted by us. Tracked only by a
manifest (path, size, mtime, content hash) enabling incremental ingest
(FR-006).

### Track
One source recording.
- `id` (hash of content), `source_path`, `sample_rate_in`, `channels_in`,
  `duration_s`, `status` (ok | skipped), `skip_reason?`
- `descriptor: StyleDescriptor` (track-level, pooled over its slices)
- `tempo_bpm`, `tempo_confidence` (derived proxy — R4), `key`,
  `key_strength`
- Relationships: has many `Slice`.

### Slice
Contiguous fragment cut at a musical boundary (FR-002).
- `id` (stable: `<track_id>:<start_sample>:<end_sample>`), `track_id`,
  `bank`, `index_in_bank` (the `:n` selector), `start_s`, `end_s`
- `wav_path` (in `banks/<bank>/`), `descriptor: StyleDescriptor`
- Invariants: non-silent (RMS above threshold); `end_s > start_s`;
  `wav_path` exists and is loadable.

### SampleBank
Named collection of slices = the vocabulary for patterns.
- `name` (== folder basename), `slice_ids` (ordered → defines `:n`),
  `count`
- Invariant: on-disk file order (alphabetical) matches `slice_ids` order
  (R5); names stable across incremental re-ingest (FR-006).

### StyleDescriptor
The uniform representation of "how audio sounds" (FR-004/FR-005), used for
corpus audio, rendered clips, and captured audio alike.
- `embedding: float[512]` (CLAP; or MERT dims if opt-in) — L2-normalized
- `handcrafted: {mfcc_mean/std, chroma_mean/std, tempo_bpm,
  onset_rate, spectral_centroid/bandwidth/rolloff/contrast/flatness
  mean/std}` (R4), each block independently normalized
- `sr_used` (48 k CLAP / 24 k MERT — never compare across `sr_used`)
- `embedder_id` (model + pinned SHA) — descriptors are only comparable
  within the same `embedder_id`
- Operation: `similarity(a, b) -> float` = cosine over the assembled,
  block-weighted vector. `nearest(query, k)` over the profile index.

### StyleProfile
Persisted descriptors + similarity structure for a corpus (FR-004/FR-005).
- per-slice + per-track descriptors; tempo/key statistics; nearest-
  neighbour index; `embedder_id`, `target_sr`, `config`, `corpus_manifest`.

### PatternSubset (grammar artifact — FR-008)
Versioned, machine-checkable definition of the supported mini-notation +
combinator subset (R3/R5). Single source of truth shared by generator,
validator, and constrained decoder.
- `version`, `ebnf` (Lark/EBNF grammar text), `allowed_combinators`,
  `bounds` (max events/cycle, max nesting depth).
- Invariant: generator, validator, and decoder are all derived from /
  checked against THIS artifact — no divergent copies.

### Pattern
A piece of TidalCycles code within the PatternSubset (FR-008).
- `text` (mini-notation string, e.g. `d1 $ s "mybank(3,8)" # gain 1.1`)
- `source` (sampled | model | mutation), `parent_id?`, `mutation?`
- `validation: {valid: bool, reason?, events_per_cycle, nesting_depth}`
- `bank_refs` (must all exist — FR-009)
- Invariant: only `valid==true` patterns are rendered / played / trained
  on (FR-010).

### RenderedClip
Deterministic offline audio of a pattern (FR-012), the scoring signal (R6).
- `pattern_id`, `audio: float[]` (or path), `sr`, `n_cycles`, `cps`
- `descriptor: StyleDescriptor` (computed from the audio)
- Invariant: deterministic given `(pattern, banks, config, seed)`;
  timing semantics match live playback within FR-013 tolerance.

### Dataset (FR-014)
Seeded (descriptor → pattern) pairs.
- `id`, `pairs[]` = `{descriptor, pattern_text}`, `config`, `seed`,
  `banks_ref`, `size`
- Invariant: regenerating from `(config, seed)` yields an identical
  dataset (SC-008); config+seed embedded in the artifact.

### ModelCheckpoint (FR-015/FR-017)
- `run_id`, `weights_path`, `base_model_id`, `adapter?` (LoRA) or
  `seq2seq`, `dataset_ref`, `train_config`, `seed`
- `eval_report: EvalReport`
- Invariant: reproducible metrics from `(config, seed, dataset)` within
  tolerance (FR-016/SC-008).

### EvalReport (FR-017)
- `validity_rate` (SC-003), `style_match_score_distribution` (similarity
  of generated-pattern render to descriptor's source audio),
  `vs_previous_run`, `held_out_size`.

### Target
Desired style (FR-019/FR-021).
- `source_audio_paths[]`, `descriptor: StyleDescriptor`.

### Session (FR-025)
One live run.
- `id`, `target`, `config`, `seed`, `start_ts`, `end_ts?`
- `population: Population`, `event_log[]` (patterns played, scores,
  controls issued, timestamps), `capture_health` (R6 verification channel)
- Invariant: log is sufficient to reconstruct the run.

### Population / Candidate (FR-022)
- Population: `members: Candidate[]`, `generation`, `best_id`,
  `no_improve_count` (patience — FR-024)
- Candidate: `pattern_id`, `score` (vs target, from RenderedClip),
  `lineage` (`parent_id`, `mutation`).
- Invariant: every candidate's pattern is validated before it can play.

---

## Key relationships

```
Corpus 1─* Track 1─* Slice *─1 SampleBank
Track/Slice ─ has ─ StyleDescriptor ─ indexed-in ─ StyleProfile
PatternSubset ─ constrains ─ Pattern ─ renders-to ─ RenderedClip ─ has ─ StyleDescriptor
Dataset *─ pairs (StyleDescriptor, Pattern) ─ trains ─ ModelCheckpoint ─ has ─ EvalReport
Target ─ scored-against ─ RenderedClip(of Candidate) within ─ Session/Population
```

## State transitions

**Ingest (idempotent/incremental — FR-006)**: `new/changed Track →
sliced → banked → descriptors computed → profile updated`. Unchanged
tracks: no-op.

**Pattern lifecycle**: `generated(sampled|model|mutation) → validated →
{rejected+logged | rendered → scored}` (FR-009/FR-010).

**Session/evolution (FR-022/FR-024)**: `init population → [render+score
candidates → select → mutate/regenerate → validate → swap best on cycle
boundary]* → (no_improve ≥ patience) → hold best + notify`. User controls
(pause/freeze/retarget/stop) apply at the next cycle boundary (FR-023).
