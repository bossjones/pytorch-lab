# Development Workflow

This guide covers the day-to-day development tasks: adding dependencies, running the
full quality suite, and using the Invoke CLI.

**What you will be able to do after this guide:**

- Add runtime and dev dependencies using `uv add`
- Run formatting, linting, and type-checking
- Run the pytest test suite
- Use the `inv` CLI for common tasks
- Understand where all tool configuration lives

**Prerequisites:** Complete [Getting Started](getting-started.md) first.

---

## 1. Adding dependencies

> **Warning:** Never run `uv pip install` or `pip install` directly. All dependency
> changes must go through `uv add` so that `uv.lock` stays in sync and the lockfile
> reflects the true state of the environment.

### Add a runtime dependency

```bash
uv add some-package
```

This adds the package to `[project.dependencies]` in `pyproject.toml`, re-resolves
`uv.lock`, and installs it into `.venv`.

### Add a development-only dependency

```bash
uv add --dev some-dev-package
```

This adds the package to `[dependency-groups] dev` in `pyproject.toml`. Dev
dependencies are installed in the local environment but not required when others install
the project as a library.

### Re-resolve the lockfile without installing

```bash
make lock
# equivalent to: uv lock
```

### Re-install from the lockfile (e.g. after pulling changes)

```bash
make setup
# equivalent to: uv sync
```

---

## 2. Formatting

The project uses **ruff format** as its formatter (replacing black and isort). Line
length is 88, configured in `pyproject.toml` under `[tool.ruff]`.

### Format all files from the repo root

```bash
make format
# equivalent to: uv run ruff format .
```

### Format from inside a subproject

```bash
cd screennet   # or screencropnet
make format
```

Both subproject Makefiles run `uv run ruff format .` and pick up the root
`pyproject.toml` configuration.

---

## 3. Linting

**ruff check** replaces pylint, autoflake, and flake8. Enabled rule sets are `E`, `F`,
`I`, `UP`, and `B` (see `[tool.ruff.lint]` in `pyproject.toml`).

### Lint all files from the repo root

```bash
make lint
# equivalent to: uv run ruff check .
```

### Auto-fix lint issues where possible

```bash
uv run ruff check . --fix
```

### Lint from inside a subproject

```bash
cd screennet   # or screencropnet
make lint
```

> **Note:** `tasks/` and `*.ipynb` files are excluded from linting via
> `extend-exclude` in `pyproject.toml`. This is intentional — those paths are
> legacy or generated files.

---

## 4. Type-checking

**pyright** in `basic` mode is the type checker (replacing mypy). Configuration is in
`pyproject.toml` under `[tool.pyright]`.

### Type-check the entire project from the repo root

```bash
make typecheck
# equivalent to: uv run pyright
```

### Type-check a subproject's entry point

```bash
cd screennet
make typecheck
# equivalent to: uv run pyright main.py

cd screencropnet
make typecheck
# equivalent to: uv run pyright main.py
```

> **Note:** `reportMissingImports` is disabled globally because some torch internals
> are not fully typed. `reportPrivateImportUsage` is also disabled for torch
> re-exports.

---

## 5. Running tests

Tests live in `tests/` and use **pytest** with **pytest-mock**. The
`pyproject.toml` options `--capture=no --disable-warnings` are set globally.

### Run all tests from the repo root

```bash
make test
# equivalent to: uv run pytest
```

### Run a specific test file

```bash
uv run pytest tests/test_something.py
```

### Run a specific test by name

```bash
uv run pytest tests/test_something.py::test_function_name
```

---

## 6. Run the full quality suite

`make check` runs linting, type-checking, and tests in sequence:

```bash
make check
# equivalent to: ruff check . && pyright && pytest
```

Run this before opening a pull request to catch issues early.

---

## 7. Invoke CLI tasks

The `inv` command (from the `invoke` package) provides additional convenience tasks.
All tasks are defined in `tasks/local.py` and configured via `invoke.yaml`.

List all available tasks:

```bash
inv -l
```

### Common local tasks

```bash
inv local.clean     # Remove all .pyc files and __pycache__ directories
inv local.sync      # uv sync (same as make setup)
inv local.lock      # uv lock (same as make lock)
inv local.jupyter   # Start a Jupyter notebook server (uv run jupyter notebook)
inv local.ipython   # Start an IPython shell (uv run ipython)
inv local.env_works # Smoke-check MPS + matplotlib (same as make env-works)
```

> **Note:** Invoke tasks honor the environment variables set in `invoke.yaml` under
> `local.env`: `BETTER_EXCEPTIONS=1`, `DEBUG=True`, `LOG_LEVEL=INFO`, `TESTING=True`.

---

## 8. Configuration reference

All tool configuration is centralized in the root `pyproject.toml`:

| Tool | Section | Key settings |
|---|---|---|
| ruff (format + lint) | `[tool.ruff]` | `line-length = 88`, `target-version = "py312"` |
| ruff lint rules | `[tool.ruff.lint]` | `select = ["E","F","I","UP","B"]` |
| pyright | `[tool.pyright]` | `typeCheckingMode = "basic"`, `pythonVersion = "3.12"` |
| pytest | `[tool.pytest.ini_options]` | `testpaths = ["tests"]`, `--capture=no --disable-warnings` |
| package layout | `[tool.setuptools]` | `going_modular` mapped to `going_modular/going_modular/` |

---

## Next steps

- See [troubleshooting.md](troubleshooting.md) for common error patterns
- Return to [README.md](README.md) for a full guide index
