# uv Projects — Reference

Source: https://docs.astral.sh/uv/concepts/projects/ and
https://docs.astral.sh/uv/concepts/projects/dependencies/

## Mental model

A uv *project* is a directory with a `pyproject.toml`. uv manages a single virtual
environment at `.venv/` and a lockfile `uv.lock`. You never activate the venv or `pip
install` manually — uv keeps `pyproject.toml`, `uv.lock`, and `.venv/` in sync.

| File | Role |
|---|---|
| `pyproject.toml` | Declared dependencies + tool config (ruff, pyright, pytest). Hand/`uv add`-edited. |
| `uv.lock` | Fully resolved, cross-platform, reproducible lock. Committed. Never hand-edited. |
| `.venv/` | The environment uv creates/syncs. Git-ignored. |
| `.python-version` | Pins the interpreter uv provisions for the project. |

## Core commands

```bash
uv init --python 3.12          # create pyproject.toml + .python-version
uv add torch torchvision       # add runtime deps -> [project.dependencies], re-lock, sync
uv add "numpy>=2"              # with a version constraint
uv add --dev pytest pytest-mock # dev deps -> [dependency-groups].dev
uv add --group lint ruff       # arbitrary named group
uv remove fastai               # remove a dep (uv remove pytest --dev for groups)
uv lock                        # resolve -> uv.lock (no install)
uv sync                        # make .venv exactly match uv.lock
uv run pytest -q               # run a command inside the project env (auto-syncs)
uv run python contrib/x.py     # run a script in the env
```

**Rule for this project:** only ever use `uv add` / `uv add --dev` to introduce
dependencies. Never `uv pip install` — it bypasses the lockfile and breaks
reproducibility.

## Dependency tables produced

```toml
[project]
name = "pytorch-lab"
requires-python = ">=3.12"
dependencies = ["torch>=2.0", "torchvision"]

[dependency-groups]
dev = ["pytest", "pytest-mock", "ruff", "pyright"]
```

`uv add --dev` writes the modern `[dependency-groups].dev` table (PEP 735), not the
legacy `[tool.uv] dev-dependencies`. The `dev` group is installed by default on
`uv sync`; exclude with `uv sync --no-dev`.

## The root package is editable automatically

If `pyproject.toml` declares the project as a package (a build backend + discoverable
packages), uv installs the project itself **editable** into `.venv`. That is exactly
how we eliminate the `sys.path.append` hacks: declare `going_modular`, `screennet`,
and `screencropnet` as packages so `import going_modular.going_modular.engine` etc.
just works from anywhere.

## Why single-project (not a workspace) here

uv workspaces give each member its own `pyproject.toml` sharing one lock — useful when
members are independently versioned/published. `pytorch-lab` is a personal lab with
two scripts sharing one helper package and one Python/torch stack. A single root
project = one venv, one lock, one place to run `uv run pytest`. Workspace overhead
buys nothing here. (Documented decision.)

## PyTorch / macOS note

Default PyPI `uv add torch torchvision` ships the standard wheels, which include the
**MPS** backend on Apple Silicon. No `[tool.uv.sources]` / custom index is needed for
a Mac-only (MPS + CPU) setup. See `uv-pytorch-mac.md`.
