# Feature Specification: Corpus-to-Live Pipeline (wav2tidal v1)

**Feature Branch**: `001-corpus-to-live-pipeline`

**Created**: 2026-07-02

**Status**: Draft — amended by [design-change-001-sound-first.md](./design-change-001-sound-first.md)

> **Amendment (2026-07-02): sound-first.** The instrument's priority shifted
> to **real-time control of SuperDirt synths and their full FX/parameter
> vocabulary** (not only arranging sample slices). Global effects are in
> scope; the learned model targets synth+param configurations; dataset
> generation uses real-time capture through a booted SuperDirt. See the
> design-change ADR for the FR deltas (FR-008/012/013/015/021) and the
> reshaped User Story 2/3. Requirements below read against that ADR where
> they conflict.

## Overview

wav2tidal turns a personal collection of music recordings (WAV files) into
a live-performance instrument. It learns the *style* of the collection —
rhythm and groove, timbre and sound palette, harmony, and overall genre
feel — and produces TidalCycles patterns that play in that style, built
from slices of the collection itself. In its final form it runs as a live
agent: it plays continuously through the user's live-coding audio setup,
listens to its own output, measures how close it sounds to a chosen
target style, and evolves its patterns toward that target.

The system runs entirely on the user's own machine. No audio, derived
data, or model ever leaves local disk: the corpus may contain copyrighted
material and the machine-learning stack must work offline.

All four style dimensions matter and are in scope: rhythm/groove, timbre/
sound palette, harmony/melody, and holistic genre feel. Timbre similarity
is achieved primarily by construction — generated patterns play slices of
the corpus itself, not third-party sample libraries.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ingest a corpus into playable sample banks (Priority: P1)

The user points the tool at a directory of WAV files. The tool slices
each recording at musically meaningful boundaries, organizes the slices
into named sample banks that their live-coding sampler can load directly,
and produces a style profile of the collection (tempo distribution,
rhythmic density, tonal content, spectral character, and a pairwise
style-similarity measure). The user can immediately hand-play the banks
in a normal TidalCycles session and browse the style profile.

**Why this priority**: Standalone value with zero machine learning — the
user gets organized, beat-sliced, performance-ready sample banks from raw
recordings, which is a useful livecoding tool by itself. Every later
stage builds on these banks and this profile, so its quality gates the
whole system. It is also the stage where the style-similarity measure is
validated by ear before anything depends on it.

**Independent Test**: Run ingestion on a real corpus; load the banks in a
stock TidalCycles + sampler session and play them by hand; open the style
profile report and spot-check that similar-sounding tracks are reported
as similar.

**Acceptance Scenarios**:

1. **Given** a directory containing readable WAV files, **When** the user
   runs ingestion, **Then** the tool produces sample banks in the layout
   the live-coding sampler loads without manual rearrangement, and each
   bank's slices come from identified musical boundaries (not arbitrary
   fixed-length cuts).
2. **Given** a completed ingestion, **When** the user opens the style
   profile, **Then** every track and every slice has descriptors (tempo,
   rhythmic density, tonal content, spectral character) and any two
   fragments can be compared with a numeric style-similarity score.
3. **Given** a corpus where the user knows two tracks sound alike and a
   third sounds different, **When** the user queries nearest neighbours
   of the first track, **Then** the alike track ranks above the different
   one.
4. **Given** a directory containing some unreadable or non-audio files,
   **When** ingestion runs, **Then** the run completes, skipped files are
   listed with reasons, and no partial/corrupt bank is produced.
5. **Given** an already-ingested corpus with new files added, **When**
   the user re-runs ingestion, **Then** only new material is processed
   and existing banks and profiles remain valid (stable names, no
   duplicate slices).

---

### User Story 2 - One-shot style-matched pattern generation (Priority: P2)

The user selects a target — one or more audio files, or a track from the
corpus — and asks for patterns. The system returns several candidate
TidalCycles patterns, built over the ingested sample banks, ranked by
predicted closeness to the target's style. The user pastes any candidate
into their live session and it plays, sounding recognizably related to
the target.

**Why this priority**: This is the first point where the system generates
music, and it exercises the whole learned pipeline (dataset synthesis,
training, generation, validation) without the complexity of live audio
capture. If one-shot generation does not produce style-plausible
patterns, the live agent cannot either.

**Independent Test**: With an ingested corpus and a trained model, request
candidates for ten different targets; verify every candidate is valid,
plays sound, and that the user judges the majority as "in the style" in
a blind check against random patterns over the same banks.

**Acceptance Scenarios**:

1. **Given** an ingested corpus and a trained model, **When** the user
   requests candidates for a target audio file, **Then** the system
   returns the requested number of distinct patterns, each of which
   passes validation and references only banks that exist.
2. **Given** returned candidates, **When** the user auditions them,
   **Then** candidates are ordered such that higher-ranked ones measure
   closer to the target under the style-similarity measure.
3. **Given** a target very unlike anything in the corpus, **When** the
   user requests candidates, **Then** the system still returns valid
   playable patterns and reports the (low) similarity scores honestly
   rather than failing or overclaiming.
4. **Given** a model that emits malformed output for some prompt,
   **When** generation runs, **Then** malformed outputs are discarded
   and logged, never returned to the user, and the system retries up to
   a configured budget.

---

### User Story 3 - Live evolving session (Priority: P3)

The user starts a live session with a chosen target style. The agent
plays patterns continuously through the normal live performance audio
chain, captures its own output, scores it against the target, and evolves
a population of patterns — mutating, discarding, and re-generating —
so the music drifts audibly toward the target style over the course of
minutes. The user can steer at any time: change the target, freeze the
current pattern, pause evolution while sound continues, or stop.

**Why this priority**: The end goal of the project, but it depends on
everything in stories 1–2 and adds live audio capture, scheduling, and
interaction. It must come last.

**Independent Test**: Start a session against a target; observe that
sound is continuous (no dropouts or dead air), that the measured
similarity trend over 10 minutes is upward, and that every user control
takes effect at the next musical boundary.

**Acceptance Scenarios**:

1. **Given** a running session, **When** the agent swaps patterns,
   **Then** swaps happen only on cycle boundaries and audio output is
   never interrupted, glitched, or silenced by the agent's own
   computation.
2. **Given** a running session, **When** 10 minutes elapse, **Then** the
   logged similarity-to-target of the currently playing pattern is higher
   than that of the first pattern played.
3. **Given** a running session, **When** the user issues a control
   (pause evolution / freeze pattern / change target / stop), **Then**
   the control is acknowledged immediately and takes musical effect at
   the next cycle boundary.
4. **Given** an evolution that stops improving, **When** the configured
   patience is exhausted, **Then** the agent keeps playing the best
   pattern found, notifies the user, and does not thrash.
5. **Given** a session, **When** it ends, **Then** a session log exists
   on disk with every pattern played, its scores, timestamps, and the
   seeds/settings needed to analyse the run.

---

### User Story 4 - Reproducible retraining (Priority: P4)

The user (or a future automated agent) re-runs any stage — dataset
synthesis, training, evaluation — from committed configuration files and
seeds, on the same corpus, and gets the same artifacts and the same
evaluation numbers. A newcomer with no context can go from a fresh clone
plus a corpus directory to a trained model by following documented steps.

**Why this priority**: This is what makes the repo maintainable by future
agents with less context. It has no standalone musical value, hence last,
but it is a hard requirement, not a nice-to-have.

**Independent Test**: Delete all derived artifacts, re-run the documented
pipeline end to end from configs, and compare artifact checksums and
evaluation metrics against the recorded previous run.

**Acceptance Scenarios**:

1. **Given** a config and seed, **When** dataset synthesis runs twice,
   **Then** the two datasets are identical.
2. **Given** a config, seed, and dataset, **When** training runs, **Then**
   the evaluation report (validity rate, held-out style-match score) is
   produced automatically and archived with the run.
3. **Given** only the repository, its documentation, and a corpus
   directory, **When** a newcomer follows the documented steps, **Then**
   they reach a playing live session without information that exists only
   in someone's head or an old conversation.

---

### Edge Cases

- **Beat-ambiguous material** (rubato, ambient, no percussive onsets):
  slicing must fall back to a documented strategy (e.g. fixed musical
  divisions of the estimated tempo, or energy-based segmentation) and the
  style profile must flag low-confidence tempo estimates rather than
  reporting false precision.
- **Tiny corpus** (one file) and **large corpus** (hundreds of files,
  hours of audio): ingestion must work at both extremes; performance
  targets are defined for the large case, correctness for both.
- **Silent, near-silent, clipped, or corrupt WAVs**: skipped with a
  reported reason; silence never becomes a "slice".
- **Mono vs stereo vs high sample rates**: all common WAV shapes are
  normalized to one documented internal format; the report states what
  was converted.
- **Model emits invalid or dangerous output** (syntax errors, unknown
  banks, unbounded event density that would overload the sampler):
  validation rejects it before it is rendered or sent live; rejection
  rate is tracked as a model-quality metric.
- **Pattern complexity explosion**: a validated pattern still has bounded
  events-per-cycle and bounded nesting depth; bounds are configuration,
  not convention.
- **Audio capture unavailable or wrong device** (live session): the
  session refuses to start evolution (no scoring signal) but can still
  play; the error names the missing capture path.
- **No improvement / oscillating scores in evolution**: bounded patience,
  keep-best behaviour, user notification (see story 3, scenario 4).
- **Loudness confounds**: similarity scoring must not be dominated by
  level differences between the captured output and the target; levels
  are normalized before comparison.
- **Training hardware absent or unsupported**: training refuses to start
  with a clear message; everything except training and live scoring
  throughput must work on CPU so tests and CI never need the GPU.
- **Target outside corpus style**: the system reports low similarity
  honestly and still behaves (story 2, scenario 3).

## Requirements *(mandatory)*

### Functional Requirements

**Ingestion & style profile**

- **FR-001**: The system MUST accept a directory tree of WAV files as a
  corpus and process all readable audio in it, skipping and reporting
  (file, reason) for anything unreadable, non-audio, silent, or corrupt.
- **FR-002**: The system MUST slice recordings at musically meaningful
  boundaries (beats/onsets), with a documented fallback for material
  where no reliable beat is found, and MUST never emit silent slices.
- **FR-003**: The system MUST organize slices into named sample banks in
  the exact directory layout the live-coding sampler loads natively, so
  banks are playable in a stock TidalCycles session with no manual
  rearrangement.
- **FR-004**: The system MUST compute descriptors for every slice and
  every track covering at minimum: tempo (with confidence), rhythmic
  density, tonal content (key/pitch-class distribution), and spectral
  character; and MUST persist them in the style profile.
- **FR-005**: The system MUST provide a numeric style-similarity measure
  between any two audio fragments (slice, track, rendered pattern, or
  captured live audio) capturing holistic stylistic closeness, and MUST
  expose a nearest-neighbour query over the corpus.
- **FR-006**: Ingestion MUST be idempotent and incremental: re-running on
  an unchanged corpus is a no-op; adding files processes only the new
  material; bank names and existing slice identities remain stable.
- **FR-007**: Corpus audio and everything derived from it (slices, banks,
  profiles, datasets, model weights, captured audio, logs containing
  audio) MUST remain on local disk: never committed to version control,
  never uploaded, never embedded in published artifacts.

**Pattern language & validation**

- **FR-008**: Generated music MUST be expressed as TidalCycles code
  restricted to a documented subset of the pattern language (the
  "pattern subset"); the subset definition is a versioned artifact of the
  repo, and everything downstream (sampler, validator, renderer, model
  training) supports exactly that subset.
- **FR-009**: The system MUST validate every pattern — whether sampled,
  model-generated, or mutated — before it is rendered or sent to a live
  session: syntactic membership in the pattern subset, references only
  to existing sample banks, and complexity within configured bounds
  (events per cycle, nesting depth).
- **FR-010**: Invalid patterns MUST never reach audio output or training
  data; they are logged with the reason, and the invalid-rate per source
  (model, mutation) is tracked and reported.

**Synthetic dataset**

- **FR-011**: The system MUST generate syntactically valid random
  patterns over the ingested banks with configurable diversity
  (distribution over tempo, density, bank usage, transformation types),
  deterministically from a seed.
- **FR-012**: The system MUST render any valid pattern to audio offline,
  deterministically, without any live audio infrastructure running, by
  mixing the referenced slices according to the pattern's event schedule
  (including at minimum: playback rate, gain, and pan transformations).
- **FR-013**: The offline renderer and the live playback path MUST share
  the same event-timing semantics; any audible deviation between an
  offline render and live playback of the same pattern is a defect. A
  documented tolerance defines "same".
- **FR-014**: The system MUST produce a training dataset of (style
  descriptor of rendered audio → pattern text) pairs, with configurable
  size, seeded and resumable, and MUST record the exact configuration
  and seed inside the dataset artifact.

**Model training & evaluation**

- **FR-015**: The system MUST train a generative model that maps a style
  descriptor to pattern text, entirely on the local machine, using the
  synthetic dataset; no cloud services or external APIs.
- **FR-016**: Training MUST be reproducible from (config, seed, dataset):
  re-running produces the same evaluation metrics within a documented
  tolerance.
- **FR-017**: Every training run MUST automatically produce an evaluation
  report on held-out data containing at minimum: pattern validity rate,
  style-match score distribution (similarity between the descriptor's
  source audio and a render of the generated pattern), and comparison
  against the previous run.
- **FR-018**: Training-hardware feasibility MUST be gated by a documented
  smoke test that a newcomer can run to verify their machine before
  attempting a full run; unsupported hardware fails fast with a clear
  message.

**Generation & live session**

- **FR-019**: Given target audio, the system MUST return N (configurable)
  distinct validated candidate patterns ranked by measured similarity of
  their offline renders to the target; malformed model outputs are
  discarded and retried within a configured budget.
- **FR-020**: The system MUST run a live session that plays patterns
  through the user's live-coding audio chain continuously, swapping
  patterns only on cycle boundaries, with the agent's computation never
  interrupting, glitching, or silencing audio output.
- **FR-021**: The live session MUST capture the system's own audio output
  and score it against the target style using the same similarity
  measure as everywhere else, with level normalization so loudness does
  not dominate the score.
- **FR-022**: The live session MUST evolve a population of patterns
  toward the target: selection by measured similarity, mutation and
  regeneration of candidates, all candidate patterns validated before
  playing; evolution parameters (population size, mutation rates,
  patience) are configuration.
- **FR-023**: The user MUST be able to, at any time during a session:
  pause/resume evolution (sound continues), freeze the current pattern,
  replace the target, and stop; controls acknowledge immediately and
  take musical effect at the next cycle boundary.
- **FR-024**: When evolution stops improving for a configured patience,
  the session MUST keep playing the best-so-far pattern and notify the
  user rather than degrade or thrash.
- **FR-025**: Every session MUST write a log sufficient to reconstruct
  the run: patterns played, scores, targets, seeds, timestamps, and
  configuration.

**Operability & reproducibility**

- **FR-026**: Every stage (ingest, profile query, dataset synthesis,
  training, evaluation, one-shot generation, live session) MUST be
  invocable as a command with a configuration file; no stage requires
  editing source code to use.
- **FR-027**: The full pipeline MUST work offline end to end after
  initial setup (model downloads for pretrained components happen at
  setup time and are cached locally).
- **FR-028**: The repository documentation MUST let a newcomer go from
  fresh clone + corpus directory to a playing live session using only
  committed documents (no context from conversations); each stage
  documents its inputs, outputs, and how to verify it worked.

### Key Entities

- **Corpus**: A user-owned directory tree of WAV recordings; the sole
  audio input to the system. Local-only, potentially copyrighted.
- **Slice**: A contiguous audio fragment cut from a corpus recording at
  musical boundaries; has a stable identity, source reference, and
  descriptors.
- **Sample Bank**: A named, ordered collection of slices in the layout
  the live sampler loads; the vocabulary from which all patterns are
  built.
- **Style Profile**: The persisted collection of descriptors and
  similarity structure for a corpus: per-slice and per-track descriptors,
  tempo/key statistics, nearest-neighbour structure.
- **Style Descriptor**: The compact representation of "how a piece of
  audio sounds" used to condition generation and to compare audio; the
  same kind of descriptor is used for corpus audio, rendered patterns,
  and captured live audio.
- **Pattern**: A piece of TidalCycles code within the pattern subset,
  referencing sample banks; carries metadata (source: sampled/model/
  mutation; validation status; scores).
- **Pattern Subset Definition**: The versioned document + machine-checkable
  grammar of exactly which pattern-language constructs the system
  supports end to end.
- **Rendered Clip**: Deterministic offline audio of a pattern; used for
  dataset pairs and candidate ranking.
- **Dataset**: A seeded, versioned collection of (style descriptor →
  pattern) pairs with its generating configuration embedded.
- **Model Checkpoint**: A locally-stored trained model artifact with its
  training config, dataset reference, and evaluation report.
- **Target**: The style the user wants: derived from chosen audio
  file(s); represented as a style descriptor.
- **Session**: One live run: target(s), population state over time,
  patterns played, scores, controls issued, seeds, and logs.
- **Population / Candidate**: The evolving set of patterns in a session
  and its members, each with measured scores and lineage (parent,
  mutation applied).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Ingesting a 100-track corpus (~8 hours of audio) completes
  in under 30 minutes on the target workstation, and the resulting banks
  load and play in a stock live-coding session with zero manual file
  operations.
- **SC-002**: In a corpus-similarity spot check of 20 query tracks, the
  user judges the top-5 nearest neighbours as "plausibly same style" for
  at least 80% of queries (validated before any model training begins).
- **SC-003**: At least 95% of model-generated patterns pass validation
  after training (invalid outputs discarded within retry budget count
  against this rate).
- **SC-004**: In a blind audition of 10 targets, one-shot candidates are
  judged by the user as "recognizably in the target's style" for at
  least 6 targets, and always judged more style-appropriate than random
  valid patterns over the same banks.
- **SC-005**: A live session produces continuous audio for 30 minutes
  with zero dropouts or agent-caused silences, and pattern swaps land on
  cycle boundaries every time.
- **SC-006**: Over the first 10 minutes of a live session, the measured
  similarity of the playing pattern to the target improves relative to
  the first pattern played, in at least 8 of 10 sessions.
- **SC-007**: Session controls take musical effect within one cycle of
  being issued, 100% of the time.
- **SC-008**: Re-running dataset synthesis or training from the same
  (config, seed) reproduces identical datasets and evaluation metrics
  within the documented tolerance, verified by an automated check.
- **SC-009**: A newcomer following only committed documentation reaches
  each milestone (banks playable, model trained, live session running)
  without needing any undocumented knowledge — verified by executing the
  docs as a script wherever possible.
- **SC-010**: Full pipeline (ingest → dataset → train → evaluate) on the
  reference corpus completes within 24 hours on the target workstation.

## Assumptions

- Single user, single Linux workstation (the target machine is the
  user's Strix Halo workstation); no multi-user, no server deployment,
  no macOS/Windows support in v1.
- A working live-coding environment (TidalCycles + SuperCollider/
  SuperDirt + a PipeWire audio stack) is installed and functional; this
  project drives it but does not install or manage it.
- The corpus is WAV only in v1; other formats are converted by the user
  beforehand. Typical corpus size 10–500 tracks.
- The user is the sole judge of musical quality; success criteria that
  need listening (SC-002, SC-004) are evaluated by the user, not
  external listeners.
- Pretrained audio-understanding components may be downloaded once at
  setup time; training any audio encoder from scratch is out of scope
  (constitution).
- Style targets are audio files (from the corpus or not); text prompts
  ("play something jazzy") are out of scope for v1.
- Harmony/melody similarity is captured at the level of tonal descriptors
  and holistic similarity, not note-accurate transcription; polyphonic
  transcription is out of scope.
- Melody/harmony *generation* happens through slice selection and
  pattern-level transformation (e.g. playback-rate pitching), not through
  synthesizer note streams, in v1.
- The live agent performs alone in v1; co-performing alongside a human
  livecoder in the same session (shared orbits, turn-taking) is future
  work.

## Out of Scope (v1)

- Text-to-style prompting; any natural-language interface.
- Training audio encoders; any cloud training or inference.
- Non-WAV input formats; streaming input; microphone input as target.
- Synthesizer/MIDI note-level generation (slice-based only).
- Multi-machine or multi-user operation; any network service.
- Automatic mixing/mastering of the output beyond level handling needed
  for fair similarity scoring.

## Dependencies

- Feature 001 builds on the repository scaffold (issue #1 / PR #2):
  dev-shell, CI, package skeleton, and the constitution's gates —
  notably the training-hardware smoke test (FR-018, constitution
  "Domain Constraints").
- The live session depends on the user's existing TidalCycles +
  SuperDirt + PipeWire installation being functional (assumption above);
  a documented manual checklist verifies this before first use.
