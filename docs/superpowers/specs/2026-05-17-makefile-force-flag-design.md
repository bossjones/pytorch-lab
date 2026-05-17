# Makefile FORCE flag + dataset idempotency — design

Date: 2026-05-17
Status: Approved, implemented

## Problem

`make data-setup` was not idempotent. Two destructive behaviors:

1. `download-dataset` ran an unguarded `curl ... > twitter_facebook_tiktok.zip`,
   re-downloading 74 MB on every invocation and overwriting any manually
   downloaded/verified zip.
2. `unzip-dataset` ran `rm -fv` on the zip after extraction, consuming a
   manually placed copy.

Re-running `make data-setup` therefore wasted bandwidth and destroyed a
user-verified artifact. The dataset is correct and no script needed altering
for *correctness* — this is purely an idempotency + safety change.

## Goals

- Make the classification dataset targets idempotent, keyed on the existence
  of the extracted base directory `scratch/datasets/twitter_facebook_tiktok/`.
- Provide an opt-in `FORCE` flag (off by default) that overrides every guard.
- Stop deleting the zip after extraction; add an explicit target to remove it.
- Unify `FORCE` so it also forces the localization assets fetch.

## Non-goals

- Changing `contrib/setup-dataset-scratch-env.sh` (its unused
  `scratch/datasets/scratch/` template tree is harmless and unrelated).
- Changing `contrib/data_doctor.py` (read-only, already correct).
- Refactoring the working `data-setup` orchestration — it inherits
  idempotency for free once its sub-targets are guarded.

## Approach

Chosen: **inline shell guards in the Makefile**. No new files, idiomatic Make,
smallest footprint. Alternatives considered and rejected: a bash helper script
(`contrib/setup-classification-dataset.sh`) — extra file/indirection for ~15
lines of logic; folding the classification dataset into
`fetch_screencropnet_assets.py` — unifies the idempotency engine but
re-architects two working targets (scope creep).

### Design decisions

| Decision | Choice |
|---|---|
| Sentinel for "already done" | `scratch/datasets/twitter_facebook_tiktok/` exists |
| Guarded operations | both download and extract |
| `FORCE` default | unset (off) |
| `FORCE` mechanism | Make variable `FORCE ?=`; `make … FORCE=1` or `FORCE=1 make …` |
| Zip after extraction | retained by default; `make clean-dataset-zip` removes on demand |
| `FORCE` scope | unified — also passes `--force` to `fetch_screencropnet_assets.py` |
| Extract flag | `unzip -o` (no interactive overwrite prompt on re-extract) |

### download-dataset guard order

1. `FORCE` set → `[force]` re-download (overwrite zip).
2. extracted dir exists → `[skip]` (zip not even needed).
3. zip present → `[skip]` (reuse manually downloaded/verified zip — the
   originating user scenario).
4. else → `curl` as before.

### unzip-dataset guard

1. extracted dir exists and no `FORCE` → `[skip]`.
2. `FORCE` set → `rm -rf` the extracted dir, then extract.
3. `unzip -o` into `scratch/datasets`.
4. `[keep]` message; zip is **not** deleted.

### Variables added

```makefile
FORCE ?=
DATASET_DIR := ./scratch/datasets/twitter_facebook_tiktok
DATASET_ZIP := ./scratch/datasets/twitter_facebook_tiktok.zip
DATASET_URL := https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1
```

Command-line and environment variables propagate through the `$(MAKE)`
sub-invocations in `data-setup` automatically (via MAKEFLAGS).

## Verification

1. Verified zip present, no extracted dir: `make download-dataset` →
   `[skip] zip already present`, zip byte-identical (no curl).
2. `make unzip-dataset` → extracts; zip still present afterward.
3. Re-run `make data-setup` → all `[skip]`, fast, no network.
4. `make data-setup FORCE=1` → `[force]` re-download + `rm -rf` re-extract +
   localization re-fetched with `--force`.
5. `make clean-dataset-zip` → removes only the zip, leaves the dataset dir.
6. `make data-doctor` → all ✓, exit 0.
7. `make help` → lists `clean-dataset-zip`.

## Files changed

- `Makefile` — dataset section (variables, `download-dataset`,
  `unzip-dataset`, new `clean-dataset-zip`, `download-localization-dataset`,
  `fetch-assets`).
