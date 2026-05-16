# Plan: Post-Migration Hardening — Entrypoint Testability, Lint Debt, Checkpoint Coverage & Contrib Bootstrap

> Status: blueprint (not yet implemented). Follow-up to the conda→uv migration
> (commits `7048f88`, `6c49536`). Implement in the dependency order **1 → 3 → 2 → 4**.

## Task Description

The conda→uv migration left four documented limitations. This spec turns them
into an executable blueprint:

1. **Testability** — `screennet/main.py`, `screencropnet/main.py`, and
   `going_modular/going_modular/train.py` run work at import (global excepthook
   installs; `train.py` trains a model on import), so they're excluded from the
   test suite.
2. **Lint debt** — 789 ruff errors, almost entirely pre-existing legacy code.
3. **Checkpoint coverage** — the 4 `weights_only=False` resume sites in
   `main_worker()` are correct but unexercised by any test.
4. **Contrib bootstrap** — there is no scripted way to fetch the *localization*
   dataset/weights for `screencropnet` (only the classification dataset is in the
   Makefile); standalone `contrib/*.py` scripts can't be run via `uv run`.

## Objective

When implemented: the two `main.py` CLIs and `train.py` import with **zero side
effects** and are test-covered; `uv run ruff check .` is clean (0 errors, or only
enumerated justified `# noqa`s); the checkpoint resume path is unit-tested via an
extracted helper; and `contrib/` has an idempotent PEP 723 asset-fetch script plus
PEP 723 headers on all standalone scripts, with the dataset/weights provenance
(recovered from the un-committed reference notebooks) documented in `ai_docs/`.

## Problem Statement

The biggest lint category (**E402, 201 errors**) is a *symptom* of the same
module-scope side effects that block testing. Auto-fixing 446 ruff issues in
untested 2000+ line files is unsafe. Therefore the work must be **sequenced**:
make entrypoints import-safe first (gains import-smoke + `--help` coverage), then
the lint cleanup and checkpoint refactor are safe to perform under that coverage.

## Solution Approach

Four phases, executed in dependency order **1 → 3 → 2 → 4**, each TDD-driven
(repo norm: `superpowers:test-driven-development`, RED→GREEN) and committed in
small, bisectable units. The full suite (`uv run pytest -q`) + import-smoke must
stay green after every commit.

## Relevant Files

- `screennet/main.py` (2187 lines) — side effects at L44 `better_exceptions.hook()`,
  L46 `console=Console()`, L131 `torch.set_num_threads(1)`; `def main()` @1203,
  `if __name__` @2169; argparse @1010–1198; `torch.load` @1451/1455 (resume,
  `main_worker()`), @1932/1946 (inference, state_dict).
- `screencropnet/main.py` (2569 lines) — side effects at L39
  `install(show_locals=True)`, L47 `better_exceptions.hook()`, L49 `Console()`,
  L179 `torch.set_num_threads(1)`; `def main()` @1513, `if __name__` @2549;
  argparse @1299–1508; `torch.load` @1754/1758 (resume), @2240/2255 (inference).
- `going_modular/going_modular/train.py` (63 lines) — **no `main()`/guard**;
  L23–63 load data + train + save at module scope.
- `tests/test_imports.py` — `IMPORTABLE` list currently excludes the 3 modules
  above; extend after Phase 1.
- `tests/conftest.py` — reuse `_deterministic` autouse fixture; add no-network norm.
- `pyproject.toml` — `[tool.ruff] select=["E","F","I","UP","B"]`,
  `[tool.pytest.ini_options]`, `[tool.pyright]` (keep; tighten later = out of scope).
- `contrib/is-mps-available.py` (needs `torch`), `contrib/does-matplotlib-work.py`
  (needs `matplotlib`), `contrib/collect_env.py` (stdlib only) — add PEP 723.
- `contrib/setup-dataset-scratch-env.sh` (classification dirs only),
  root `Makefile` (`download-dataset`/`get-best-model`), `CLAUDE.md`,
  `.claude/rules/python-scripts.md` (already mandates PEP 723 — align to it),
  `.gitignore` (ignore `contrib/*.ipynb`).

### New Files
- `tests/test_cli_smoke.py` — subprocess `--help` exits 0 for both CLIs.
- `tests/test_checkpoint.py` — `load_checkpoint()` helper behavior.
- `tests/test_entrypoint_import_safety.py` — importing the 3 modules mutates
  nothing (excepthook unchanged; `train.py` does not train).
- `contrib/fetch_screencropnet_assets.py` — PEP 723 uv script (idempotent
  dataset/weights/sample fetch).
- `ai_docs/screencropnet-assets.md` — asset provenance table (survives the
  un-committed notebooks).

## Asset Provenance (recovered from contrib/*.ipynb — to record in ai_docs)

| Asset | Source URL | Destination | Used by |
|---|---|---|---|
| Localization dataset | `https://www.dropbox.com/s/w5rzn8b1s0p9d2n/twitter_screenshots_localization_dataset.zip?dl=1` | `scratch/datasets/twitter_screenshots_localization_dataset/` (images nested one dir deeper; labels `labels_pascal_temp.csv`, pascal-voc cols `img_path,xmin,ymin,xmax,ymax`) | screencropnet |
| Inference checkpoint (canonical) | `https://www.dropbox.com/s/9903r4jy02rmuzh/ScreenCropNetV1_378_epochs.pth?dl=1` | `screencropnet/models/ScreenCropNetV1_378_epochs.pth` | screencropnet predict/evaluate (matches existing Makefile) |
| Alt checkpoint | `https://www.dropbox.com/s/m5mvmlv28x7xdty/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth?dl=1` | `screencropnet/models/` | transfer-learning |
| Alt checkpoint | `https://www.dropbox.com/s/rzkwy02hz2j3ath/screencropnet_best_model.pt?dl=1` | `screencropnet/models/` | reference |
| Sample image | `https://www.dropbox.com/s/livf8f0dwd6wnlr/IMG_6324.PNG?dl=1` | `scratch/IMG_6324.PNG` | predict smoke |
| Classification dataset (already scripted) | `https://www.dropbox.com/s/8w1jkcvdzmh7khh/twitter_facebook_tiktok.zip?dl=1` | `scratch/datasets/` | screennet (leave as-is) |

## Implementation Phases

### Phase 1: Foundation — entrypoint import-safety (TDD)
Make the 3 modules side-effect-free on import; gain test coverage.

### Phase 2: Core — staged, test-gated lint cleanup
Eliminate the 789 ruff errors safely, now that Phase 1 gives coverage.

### Phase 3: Core — extract & unit-test `load_checkpoint()`
Cover the `weights_only=False` resume path with a real helper.

### Phase 4: Integration & Polish — contrib bootstrap + PEP 723
Scripted asset provisioning + standalone-runnable contrib scripts + docs.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom. Run `uv run pytest -q`
after every numbered task; commit per task.

### 1. RED: entrypoint import-safety tests
- Add `tests/test_entrypoint_import_safety.py`:
  - `test_importing_screennet_main_keeps_excepthook`: snapshot `sys.excepthook`,
    `import screennet.main`, assert unchanged. (RED: module-scope hook install.)
  - same for `screencropnet.main` (also asserts rich traceback not installed).
  - `test_importing_going_modular_train_does_not_train`: `mocker.patch`
    `going_modular.engine.train`; `import going_modular.train`; assert not called
    and no model file written. (RED: trains on import / FileNotFoundError.)
- Verify all RED for the expected reasons.

### 2. GREEN: refactor entrypoints
- `screennet/main.py` / `screencropnet/main.py`: add
  `def _install_exception_hooks() -> None:` containing `better_exceptions.hook()`
  (+ `rich.traceback.install(show_locals=True)` for screencropnet); call it as the
  first statement of `def main()`. Move `torch.set_num_threads(1)` into `main()`.
  Remove the now-orphaned module-scope statements. Leave `console = Console()`
  module-scope (construction is side-effect-free; functions reference it globally).
- `going_modular/going_modular/train.py`: wrap L23–63 in `def main() -> None:`;
  add `if __name__ == "__main__":\n    main()`.
- Verify the Phase-1 tests pass (GREEN). Full suite green.

### 3. Extend coverage for the now-safe modules
- Add `screennet.main`, `screencropnet.main`, `going_modular.train` to
  `IMPORTABLE` in `tests/test_imports.py`.
- Add `tests/test_cli_smoke.py`: `subprocess.run([sys.executable,
  "screennet/main.py", "--help"])` exit 0 and "usage:" in stdout; same for
  screencropnet. (Use the project interpreter via `uv run`-equivalent.)
- Full suite green; commit Phase 1.

### 4. RED: `load_checkpoint()` helper test
- Add `tests/test_checkpoint.py` parametrized over `screennet.main`,
  `screencropnet.main`:
  - Build `ckpt = {"epoch": 7, "best_acc1": torch.tensor(91.2),
    "state_dict": TinyVGG(...).state_dict(), "optimizer": {"lr": 1e-3}}`;
    `torch.save` to `tmp_path`.
  - `mod.load_checkpoint(path)` → dict, `["epoch"] == 7`, state_dict keys present.
    (RED: `AttributeError`, no such function.)
  - `with pytest.raises(Exception): torch.load(path, weights_only=True)` —
    documents *why* resume needs `weights_only=False`.
  - `mocker`-patched `torch.cuda.is_available→False`: passing `gpu=0` still
    returns via plain `torch.load` (fixes the latent "gpu set, no CUDA →
    `checkpoint` undefined" bug).

### 5. GREEN: extract `load_checkpoint()`
- In each `main.py`, add near the other `load_*` functions:
  ```python
  def load_checkpoint(resume_path: str, gpu: int | None = None) -> dict:
      """Load a full training checkpoint (trusted local file, weights_only=False)."""
      if gpu is not None and torch.cuda.is_available():
          return torch.load(resume_path, map_location=f"cuda:{gpu}",
                             weights_only=False)
      return torch.load(resume_path, weights_only=False)
  ```
- Replace the inline `if args.gpu is None: ... elif torch.cuda.is_available(): ...`
  block in `main_worker()` with `checkpoint = load_checkpoint(args.resume,
  args.gpu)`. Behavior-preserving + fixes the latent undefined-var bug.
- Tests GREEN; full suite green; commit Phase 3.

### 6. Lint C1 — safe autofix by category
- For each group, run `uv run ruff check --select <CODES> --fix .`, then
  `uv run pytest -q` + import-smoke, then commit:
  1. `I001` (71)  2. `F401` (156)  3. `UP006,UP032,UP035,UP015,UP045,UP004,UP007,
     UP008,UP010,UP034,UP009,UP024` (~178)  4. `E401` (2), `B033` (1), `F541` (1).
- Never use `--unsafe-fixes` blindly.

### 7. Lint C2 — E402 (201)
- With Phase 1 side-effects gone, consolidate module-scope imports to the top of
  each file; `uv run ruff check --select E402 --fix .` then manual residue.
  (Function-local imports are not E402.) Suite + import-smoke green; commit.

### 8. Lint C3 — manual-review categories (no blind autofix)
- Triage and fix with review, committing in small groups:
  - **Real-bug candidates**: `F821` (1, undefined name — must fix),
    `F811` (36, redefined-while-unused — dedupe; verify intended),
    `B023` (9, loop-var closure), `B006`/`B008` (mutable/call default args).
  - **Mechanical**: `B007` (18 → rename to `_`), `E711` (4 → `is None`),
    `E721` (2 → `isinstance`), `E741` (1 rename), `UP031`/`UP030` (printf→format),
    `B018`/`B028`/`F403`/`F405`.
- Policy: **no file-level `# ruff: noqa`**; only targeted
  `# noqa: <CODE>  # <reason>` where a pattern is genuinely intentional —
  enumerate every such line in the PR description.
- Note: `screencropnet/test_predict.py` (135 errors) is a non-pytest manual
  script (excluded by `testpaths`); clean it but flag relocating it to
  `examples/` as a follow-up (out of scope here).
- `uv run ruff check .` → 0; commit Phase 2.

### 9. Phase 4 — contrib asset bootstrap (PEP 723)
- Create `contrib/fetch_screencropnet_assets.py` with shebang
  `#!/usr/bin/env -S uv run --script` and PEP 723 header:
  ```python
  # /// script
  # requires-python = ">=3.12"
  # dependencies = ["requests"]
  # ///
  ```
  Flags `--dataset/--weights/--sample/--all`; idempotent (skip if target exists);
  download + unzip per the Asset Provenance table; print resolved paths incl. the
  nested image dir and `labels_pascal_temp.csv`. (Mac-only standard PyPI — **no
  `[tool.uv.index]` needed**; include the alternative-index syntax only as a
  reference comment per https://docs.astral.sh/uv/guides/scripts/.)

### 10. Phase 4 — PEP 723 on standalone contrib scripts
- Add PEP 723 header + `#!/usr/bin/env -S uv run --script` to
  `contrib/is-mps-available.py` (`dependencies=["torch"]`),
  `contrib/does-matplotlib-work.py` (`dependencies=["matplotlib"]`),
  `contrib/collect_env.py` (`dependencies=[]`, stdlib-only). Conform to
  `.claude/rules/python-scripts.md`.

### 11. Phase 4 — wire-up & docs
- `Makefile`: add `download-localization-dataset` and `fetch-assets` targets
  calling `uv run contrib/fetch_screencropnet_assets.py …`; keep classification
  `download-dataset` unchanged.
- Write `ai_docs/screencropnet-assets.md` (the provenance table).
- `CLAUDE.md`: document the new asset commands + provenance pointer.
- `.gitignore`: add `contrib/*.ipynb` (reference notebooks stay local).

### 12. Final validation
- Run the full Validation Commands block; confirm all green; update this spec's
  "Acceptance Criteria" checkboxes in the PR.

## Testing Strategy
- **TDD** for Phases 1 & 3 (genuine RED→GREEN): excepthook-stability,
  train-not-run-on-import, `load_checkpoint`. Mock heavy I/O with `pytest-mock`;
  no datasets/training; CPU-only; seeded (reuse `conftest._deterministic`).
- **Regression gate** for Phase 2: existing 40 tests + import-smoke + new
  CLI-smoke must stay green after *every* category commit (bisectable).
- **CLI smoke** via `subprocess` (`--help` exit 0) — cheap, no model load.
- Contrib script: a test that `fetch_screencropnet_assets.py --help` runs and
  that the URL/destination map matches the documented provenance (no network in
  CI — mock `requests.get`/assert idempotent skip when target exists).

## Acceptance Criteria
- [ ] `uv run ruff check .` → **0 errors** (or only enumerated, justified
  `# noqa: CODE` lines listed in the PR).
- [ ] `python -c "import sys,screennet.main"` / `screencropnet.main` /
  `going_modular.train` leave `sys.excepthook` unchanged and write no files /
  train nothing on import.
- [ ] `tests/test_imports.py` includes all 3 formerly-excluded modules; new
  `test_cli_smoke.py`, `test_checkpoint.py`, `test_entrypoint_import_safety.py`
  pass; the original 40 tests still pass.
- [ ] `load_checkpoint()` exists in both `main.py`, is called from
  `main_worker()`, and the latent "gpu set + no CUDA" undefined-var bug is gone.
- [ ] `uv run contrib/is-mps-available.py`, `contrib/does-matplotlib-work.py`,
  `contrib/collect_env.py`, and `contrib/fetch_screencropnet_assets.py --help`
  all run standalone via their PEP 723 headers.
- [ ] Asset provenance documented in `ai_docs/screencropnet-assets.md`;
  `contrib/*.ipynb` git-ignored.

## Validation Commands
- `uv run ruff check .` — expect `All checks passed!` (or documented noqa only)
- `uv run ruff format --check .` — formatting clean
- `uv run pytest -q` — entire suite green (40 existing + new)
- `uv run pyright tests/` — 0 errors
- `uv run python -c "import sys; e=sys.excepthook; import screencropnet.main; assert sys.excepthook is e, 'excepthook mutated on import'"`
- `uv run python screennet/main.py --help` and
  `uv run python screencropnet/main.py --help` — exit 0, print usage
- `uv run python -c "import going_modular.train"` — returns immediately, trains nothing
- `uv run contrib/is-mps-available.py` — runs via PEP 723 (`tensor(... mps ...)`)
- `uv run contrib/fetch_screencropnet_assets.py --help` — runs via PEP 723

## Notes
- **Sequencing is load-bearing**: 1 → 3 → 2 → 4. Lint-fixing the two `main.py`
  files before they're import-safe & test-covered risks silent behavior changes
  in untested code.
- No new project deps needed (`requests` already a dep; contrib script declares
  its own via PEP 723). If anything is added it must be via `uv add` (never
  `uv pip install`) — repo hard rule.
- The `.ipynb` reference notebooks are intentionally **not** committed; their
  knowledge is preserved in `ai_docs/screencropnet-assets.md`.
- Out of scope (future): deduping the two identical `devices.py`; relocating
  `screencropnet/test_predict.py` to `examples/`; tightening pyright to strict;
  real end-to-end training runs.
