# Quickstart: wav2tidal v1

Fresh clone + a corpus directory → a playing live session, using only
committed docs (FR-028/SC-009). Each step says how to verify it worked.
Target: NixOS box with a working TidalCycles + SuperDirt + PipeWire stack.

> All derived data (banks, profiles, datasets, checkpoints, captured
> audio) stays on local disk and is gitignored. The corpus never leaves
> the machine (FR-007).

## 0. Environment

```
nix develop            # provides python env, ffmpeg, just, etc.
wav2tidal doctor       # preflight: GHCi/BootTidal, SuperDirt banks, PipeWire null sink, model cache
```
**Verify**: `doctor` prints all-green; any red line names the exact fix.

One-time model fetch (only step needing network), pinned by SHA:
```
just fetch-models      # downloads laion/larger_clap_music @ pinned SHA into ./.hf_cache
```
Thereafter `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` — no phone-home.

## 1. Ingest a corpus  (User Story 1 — standalone value)

```
wav2tidal ingest --config configs/ingest.yaml --corpus /path/to/wavs
```
**Verify**:
- `banks/` has one folder per bank; play by hand in a stock Tidal session:
  `d1 $ s "mybank"` makes sound (SC-001, no manual file moves).
- `wav2tidal profile --query <a-track> --k 5` ranks audibly-similar tracks
  above dissimilar ones (SC-002 — do this ear check before training).
- The run report lists skipped files with reasons; no silent slices.

## 2. Check the GPU  (FR-018 gate — do this before training)

```
wav2tidal smoke-gpu --config configs/train.yaml
```
**Verify**: PASS = torch sees gfx1151 and one bf16 LoRA step completed.
FAIL prints the fix (e.g. add user to `render`/`video`, or switch to the
TheRock-wheels FHS fallback). Do not run `train` until this passes.

## 3. Synthesize the dataset  (FR-011/FR-014)

```
wav2tidal dataset --config configs/dataset.yaml --seed 0
```
**Verify**: `datasets/<id>/` exists with the config+seed embedded; running
it twice yields byte-identical output (SC-008).

## 4. Train + evaluate  (User Story 2 backbone)

```
wav2tidal train --config configs/train.yaml --seed 0
wav2tidal eval  --checkpoint <run-id>
```
**Verify**: `checkpoints/<run-id>/` holds weights + an `EvalReport` with
validity rate (≥95 %, SC-003), style-match distribution, and a comparison
to the previous run.

## 5. One-shot generation  (User Story 2)

```
wav2tidal generate --target /path/to/target.wav --n 8 --checkpoint <run-id>
```
**Verify**: 8 distinct patterns, each validates and plays; higher-ranked
ones measure closer to the target (SC-004). Paste one into your Tidal
session — it sounds recognizably related to the target.

## 6. Live evolving session  (User Story 3 — the goal)

```
wav2tidal live --config configs/live.yaml --target /path/to/target.wav
```
Controls (take effect on the next cycle boundary, SC-007): pause/resume
evolution, freeze pattern, retarget, stop.
**Verify**:
- Continuous audio, no dropouts, swaps land on cycle boundaries (SC-005).
- Over 10 min the playing pattern's similarity-to-target trends up (SC-006).
- On patience exhaustion it holds the best pattern and notifies (FR-024).
- `sessions/<id>/` log reconstructs the run (FR-025).

## Reproduce a full run  (User Story 4)

```
rm -rf banks profile datasets checkpoints      # keep corpus/ and configs/
just pipeline                                  # ingest → dataset → train → eval
```
**Verify**: artifact checksums and eval metrics match the recorded prior
run within tolerance (SC-008/SC-010).

---

Config files under `configs/` are the committed, editable knobs for every
stage (FR-026). Nothing here requires editing source code.
