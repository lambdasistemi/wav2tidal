# Contract: CLI stage commands

Every stage is a subcommand invoked with a config file; no stage needs
source edits (FR-026). All commands are offline after setup (FR-027).
Config files are committed; artifacts are gitignored (FR-007). Exit code
0 = success; non-zero = failure with a diagnostic on stderr. `--seed`
overrides the config seed (reproducibility, FR-016).

Command surface (name binding decided at implementation time; shape is
the contract):

```
wav2tidal ingest      --config ingest.yaml   [--corpus DIR] [--seed N]
wav2tidal profile     --query TRACK|FILE  [--k 5]        # nearest neighbours
wav2tidal dataset     --config dataset.yaml [--seed N] [--resume]
wav2tidal smoke-gpu   --config train.yaml                # FR-018 gate
wav2tidal train       --config train.yaml   [--seed N]
wav2tidal eval        --checkpoint RUN_ID                # FR-017 report
wav2tidal generate    --target FILE… --n 8 [--checkpoint RUN_ID]
wav2tidal live        --config live.yaml --target FILE…  [--seed N]
wav2tidal doctor                                          # env preflight (R5/R6)
```

## ingest (FR-001..006)
- **In**: corpus dir; config (target_sr, hop_length, slice strategy,
  thresholds, bank naming).
- **Out**: `banks/`, `profile/`; a run report listing produced banks +
  skipped files with reasons.
- **Guarantees**: idempotent/incremental; no silent slices; banks load in
  stock SuperDirt with zero manual ops (SC-001).
- **Errors**: unreadable/silent/corrupt files skipped+reported, never
  fatal; a partial run leaves no corrupt bank.

## profile
- **In**: a track id or audio file; `k`.
- **Out**: ranked nearest neighbours with similarity scores; the
  descriptor of the query.
- **Guarantees**: comparison only within one `embedder_id`/`sr_used`.

## dataset (FR-011, FR-014)
- **In**: banks/profile; config (size, diversity distributions, seed).
- **Out**: `datasets/<id>/` with pairs + embedded `(config, seed)`.
- **Guarantees**: deterministic from `(config, seed)` (SC-008); resumable;
  every pattern valid (FR-011).

## smoke-gpu (FR-018)
- **Out**: PASS/FAIL report: torch sees gfx1151; one LoRA `backward+step`
  in bf16 completes. FAIL is a clear, actionable message (not a stack
  trace) — gates `train`.

## train (FR-015..017)
- **In**: dataset, config (base model, LoRA/seq2seq params, seed).
- **Out**: `checkpoints/<run-id>/` with weights + config + `EvalReport`.
- **Guarantees**: reproducible metrics within tolerance (FR-016);
  eval auto-produced and archived; refuses to start if `smoke-gpu` was
  not passed for this environment.

## generate (FR-019)
- **In**: target audio; `n`; checkpoint.
- **Out**: `n` distinct **validated** patterns ranked by render-similarity
  to target; malformed model outputs discarded+retried within budget,
  never returned.
- **Guarantees**: every returned pattern passes validation and references
  only existing banks; low-similarity targets reported honestly.

## live (FR-020..025)
- **In**: target audio; config (population size, mutation rates, patience,
  cps); a running Tidal+SuperDirt+PipeWire stack (checked by `doctor`).
- **Out**: continuous audio; a `sessions/<id>/` log (FR-025).
- **Guarantees**: swaps on cycle boundaries only; agent computation never
  interrupts audio (SC-005); controls take effect within one cycle
  (SC-007); keep-best on patience exhaustion (FR-024).
- **Precondition**: refuses to *start evolution* if the capture
  verification path is unavailable (edge case), but can still play.

## doctor (R5/R6 preflight)
- Checks: GHCi+BootTidal reachable; SuperDirt banks loaded; PipeWire null
  sink + monitor present; `pw-record` works; embedder model cached.
- **Out**: a checklist with pass/fail per item and the fix for each fail.
  This is the executable form of the "manual checklist" in spec
  Dependencies.
