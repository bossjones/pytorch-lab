# tasks — API Reference

Package path: `tasks/`

The `tasks/` package provides [Invoke](https://www.pyinvoke.org/) task collections for local development and CI quality checks, plus supporting utilities.

Tasks are invoked from the repo root via `inv <namespace>.<task>` (e.g., `inv local.clean`, `inv ci.pytest`). Run `inv -l` to list all available tasks.

---

## `tasks.local`

**Module docstring:** _Local dev tasks — uv-based. Replaces the old pip-tools/conda bootstrap scaffolding._

All functions in this module are Invoke `@task` decorators and accept an Invoke `Context` object (`ctx`) as their first argument.

### `clean`

```python
@task
def clean(ctx) -> None:
```

**Docstring:** _Remove .pyc / __pycache__ artifacts._

Runs: `find . -name '*.pyc' -delete; find . -name '__pycache__' -type d -exec rm -rf {} +`

---

### `sync`

```python
@task
def sync(ctx) -> None:
```

**Docstring:** _Create/update the uv-managed virtualenv from uv.lock._

Runs: `uv sync`

---

### `lock`

```python
@task
def lock(ctx) -> None:
```

**Docstring:** _Re-resolve dependencies into uv.lock._

Runs: `uv lock`

---

### `ipython`

```python
@task
def ipython(ctx) -> None:
```

**Docstring:** _Start an IPython shell inside the project env._

Runs: `uv run ipython`

---

### `jupyter`

```python
@task
def jupyter(ctx) -> None:
```

**Docstring:** _Start a Jupyter notebook server inside the project env._

Runs: `uv run jupyter notebook`

---

### `env_works`

```python
@task
def env_works(ctx) -> None:
```

**Docstring:** _Smoke-check the environment (MPS + matplotlib)._

Runs `contrib/is-mps-available.py` then `contrib/does-matplotlib-work.py` via `uv run`.

---

## `tasks.ci`

**Module docstring:** _CI / quality tasks — uv + ruff + pyright + pytest. Replaces the old conda/pip-tools/black/isort/mypy/pylint scaffolding._

### `clean`

```python
@task
def clean(ctx) -> None:
```

**Docstring:** _Remove compiled python artifacts and coverage data._

Runs: deletes `*.pyc`, `*.pyo`, `__pycache__` directories, and `.coverage`.

---

### `lint`

```python
@task
def lint(ctx, fix: bool = False) -> None:
```

**Docstring:** _ruff check (lint). Pass --fix to autofix._

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fix` | `bool` | `False` | If `True`, passes `--fix` to ruff |

Runs: `uv run ruff check [--fix] .`

---

### `format`

```python
@task
def format(ctx, check: bool = False) -> None:
```

**Docstring:** _ruff format. Pass --check for CI verification._

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check` | `bool` | `False` | If `True`, passes `--check` (verification mode, no writes) |

Runs: `uv run ruff format [--check] .`

---

### `typecheck`

```python
@task
def typecheck(ctx) -> None:
```

**Docstring:** _pyright static type check._

Runs: `uv run pyright` (exit code warnings are not fatal due to `warn=True`).

---

### `pytest`

```python
@task
def pytest(ctx, k=None, verbose: bool = False) -> None:
```

**Docstring:** _Run the test suite. `inv ci.pytest -k iou` to filter._

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `str \| None` | `None` | Test filter expression (passed to pytest `-k`) |
| `verbose` | `bool` | `False` | If `True`, adds `-vv` flag |

Runs: `uv run pytest [-vv] [-k <k>]`

---

### `check`

```python
@task(pre=[lint, typecheck, pytest])
def check(ctx) -> None:
```

**Docstring:** _Run lint + typecheck + tests (the full gate)._

Pre-tasks: `lint`, `typecheck`, `pytest` (run in sequence). Prints a completion message after all pre-tasks succeed.

---

## `tasks.utils`

**Module docstring:** _supporting task functions_

Miscellaneous utilities used by Invoke tasks and internal tooling.

### Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `COLOR_WARNING` | `str` | `"red"` | Rich/colorama color for warnings |
| `COLOR_DANGER` | `str` | `"red"` | Rich/colorama color for danger |
| `COLOR_SUCCESS` | `str` | `"green"` | Rich/colorama color for success |
| `COLOR_CAUTION` | `str` | `"yellow"` | Rich/colorama color for caution |
| `COLOR_STABLE` | `str` | `"blue"` | Rich/colorama color for stable |
| `ENV_WHITELIST` | `list[str]` | See source | Environment variable names allowed through `get_compose_env` |

### `_check_exe`

```python
def _check_exe(exe: str) -> None:
```

_No docstring._ Raises `invoke.Exit` if the executable `exe` is not found on `PATH`. (Inferred from implementation.)

**Raises:** `invoke.Exit`

---

### `is_venv`

```python
def is_venv() -> bool:
```

Checks whether the current Python process is running inside a virtual environment.

**Returns:** `bool`

---

### `get_compose_env`

```python
def get_compose_env(c, loc: str = "docker", name=None) -> dict | str:
```

_No docstring._ Builds a filtered environment variable dict from `invoke.yaml` config, merging in OS environment overrides. Only keys in `ENV_WHITELIST` are retained. If `name` is given, returns just the value for that key. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c` | `invoke.Context` | — | Invoke context |
| `loc` | `str` | `"docker"` | Config section to read from `invoke.yaml` |
| `name` | `str \| None` | `None` | If set, return only this key's value |

**Returns:** `dict | str` — Full filtered env dict, or a single value when `name` is given.

---

### `confirm`

```python
def confirm() -> bool:
```

Prompts the user to enter Y or N interactively.

**Returns:** `bool` — `True` if the user answered Y.

---

### `ProcessException`

```python
class ProcessException(Exception):
    pass
```

_No docstring._ Raised by `pquery` when a subprocess returns a non-zero exit code. (Inferred from implementation.)

---

### `Console`

```python
class Console:
    quiet: bool = False

    @classmethod
    def message(cls, str_format, *args) -> None:
```

_No docstring._ Simple wrapper for `print` with `stdout.flush()`. Setting `Console.quiet = True` suppresses all output. (Inferred from implementation.)

#### `message`

```python
@classmethod
def message(cls, str_format, *args) -> None:
```

Prints a formatted message to stdout (unless `quiet` is set).

---

### `pquery`

```python
def pquery(command, stdin=None, **kwargs) -> str:
```

_No docstring._ Runs a command as a subprocess and returns its stdout. Raises `ProcessException` on non-zero exit. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `list` | Command and arguments |
| `stdin` | bytes | Standard input bytes |

**Returns:** `str` — Decoded stdout.

**Raises:** `ProcessException` if return code is non-zero.

---

### `cd`

```python
@contextlib.contextmanager
def cd(newdir) -> Generator:
```

_No docstring._ Context manager that temporarily changes the working directory to `newdir`. (Inferred from implementation.)

**Yields:** Nothing; restores the original directory on exit.

---

### `scm`

```python
def scm(dir=None) -> str | None:
```

_No docstring._ Detects the source control manager for a directory by checking for `.git` or `.hg` subdirectories. (Inferred from implementation.)

**Returns:** `"git"`, `"hg"`, or `None`.

---

### `remove`

```python
def remove(path) -> None:
```

_No docstring._ Recursively removes a directory tree, handling read-only files. (Inferred from implementation.)

---

### `move_f`

```python
def move_f(src, dst) -> None:
```

_No docstring._ Moves `src` to `dst` using `shutil.move`. (Inferred from implementation.)

---

### `copy_f`

```python
def copy_f(src, dst) -> None:
```

_No docstring._ Copies a directory tree from `src` to `dst` using `shutil.copytree`. (Inferred from implementation.)

---

### `git_clone`

```python
def git_clone(repo_url: str, dest: str, sha: str = "master") -> None:
```

_No docstring._ Clones `repo_url` into `dest` and checks out `sha`, but only if `dest` does not already exist or is not a git repo. (Inferred from implementation.)

---

### `whoami`

```python
def whoami() -> bytes:
```

_No docstring._ Returns the output of `who`. (Inferred from implementation.)

---

### `environ_append`

```python
def environ_append(key: str, value: str, separator: str = " ", force: bool = False) -> None:
```

_No docstring._ Appends `value` to the environment variable `key`, separated by `separator`. (Inferred from implementation.)

---

### `environ_prepend`

```python
def environ_prepend(key: str, value: str, separator: str = " ", force: bool = False) -> None:
```

_No docstring._ Prepends `value` to the environment variable `key`. (Inferred from implementation.)

---

### `environ_remove`

```python
def environ_remove(key: str, value: str, separator: str = ":", force: bool = False) -> None:
```

_No docstring._ Removes `value` from the environment variable `key` (colon-separated by default). (Inferred from implementation.)

---

### `environ_set`

```python
def environ_set(key: str, value: str) -> None:
```

_No docstring._ Sets `os.environ[key] = value`. (Inferred from implementation.)

---

### `environ_get`

```python
def environ_get(key: str) -> str | None:
```

_No docstring._ Returns `os.environ.get(key)`. (Inferred from implementation.)

---

### `path_append`

```python
def path_append(value: str) -> None:
```

_No docstring._ Appends `value` to `PATH` if the path exists. (Inferred from implementation.)

---

### `path_prepend`

```python
def path_prepend(value: str, force: bool = False) -> None:
```

_No docstring._ Prepends `value` to `PATH` if the path exists. (Inferred from implementation.)

---

### `mkdir_p`

```python
def mkdir_p(path: str) -> None:
```

_No docstring._ Creates a directory and all parents, silently succeeding if it already exists. (Inferred from implementation.)

---

### `dump_env_var`

```python
def dump_env_var(var: str) -> None:
```

_No docstring._ Prints the value of environment variable `var` (or `<EMPTY>` if unset). (Inferred from implementation.)

---

## `tasks.constants`

_No module docstring._

### Constants

| Name | Type | Value | Description |
|------|------|-------|-------------|
| `ROOT_DIR` | `str` | `os.path.dirname(__file__)` | Absolute path to the `tasks/` directory |
| `PROJECT_BIN_DIR` | `str` | `ROOT_DIR + "/bin"` | Path to the project `bin/` directory |
| `DATA_DIR` | `str` | `ROOT_DIR + "/var"` | Path to the project data directory |
| `SCRIPT_DIR` | `str` | `ROOT_DIR + "/script"` | Path to the project script directory |

---

## `tasks.ml_logger`

_No module docstring._ Provides a Loguru-backed logging setup with `InterceptHandler` (redirects standard `logging` calls into Loguru) and `LoopDetector` (detects repetitive WARNING/ERROR log lines).

### Pydantic Models

#### `LoggerPatch`

```python
class LoggerPatch(BaseModel):
    name: str
    level: str
```

_No docstring._ Simple model representing a logger name and level string for patch operations. (Inferred from implementation.)

---

#### `LoggerModel`

```python
class LoggerModel(BaseModel):
    name: str
    level: Optional[int]
    children: Optional[List[Any]] = None
```

_No docstring._ Recursive model representing a logger node in the logging hierarchy tree. (Inferred from implementation.)

---

### `format_record`

```python
def format_record(record: Dict[str, Any]) -> str:
```

Custom Loguru format function. Appends a formatted `payload` field (pretty-printed via `pformat`) to the log line when present.

| Parameter | Type | Description |
|-----------|------|-------------|
| `record` | `Dict[str, Any]` | Loguru log record |

**Returns:** `str` — Format string to use for this record.

---

### `InterceptHandler`

```python
class InterceptHandler(logging.Handler):
    loglevel_mapping: dict = {...}

    def emit(self, record: logging.LogRecord) -> None:
```

**Class docstring:** _Intercept all logging calls (with standard logging) into our Loguru Sink._

Bridges the standard `logging` module to Loguru. Install via `logging.basicConfig(handlers=[InterceptHandler()], level=0)`.

#### `emit`

```python
def emit(self, record: logging.LogRecord) -> None:
```

Translates a standard `LogRecord` to a Loguru log call, preserving caller stack depth.

---

### `LoopDetector`

```python
class LoopDetector(logging.Filter):
    LINE_HISTORY_SIZE: int = 50
    LINE_REPETITION_THRESHOLD: int = 5
```

_No class docstring._ Log filter that detects repeating WARNING/ERROR messages. Suppresses further printing of lines that exceed `LINE_REPETITION_THRESHOLD` occurrences within the last `LINE_HISTORY_SIZE` entries. (Inferred from implementation.)

#### `filter`

```python
def filter(self, record: logging.LogRecord) -> bool:
```

Always returns `True` (allows the record through), but tracks repetitions and prints suppression notices for frequently repeated lines.

---

### `get_logger`

```python
def get_logger(
    name: str,
    provider: Optional[str] = None,
    level: int = logging.INFO,
    logger: logging.Logger = logger,
) -> logging.Logger:
```

_No docstring._ Configures a Loguru logger with colorized stdout output and installs `InterceptHandler` into `logging.basicConfig`. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Logger name |
| `provider` | `str \| None` | `None` | Provider label (unused in body; reserved for future use) |
| `level` | `int` | `logging.INFO` | Minimum log level |
| `logger` | `logging.Logger` | Loguru `logger` | Logger instance to configure |

**Returns:** `logging.Logger` — Configured Loguru logger.

---

### `intercept_all_loggers`

```python
def intercept_all_loggers(level: int = logging.DEBUG) -> None:
```

_No docstring._ Installs `InterceptHandler` at `basicConfig` level and clears uvicorn handlers. (Inferred from implementation.)

---

### `get_caller_stack_name`

```python
def get_caller_stack_name(depth: int = 1) -> str:
```

_No docstring._ Returns the function name of the caller at the given stack depth. (Inferred from implementation.)

**Returns:** `str`

---

### `get_caller_stack_and_association`

```python
def get_caller_stack_and_association(depth: int = 1) -> str:
```

_No docstring._ Returns the qualified name of the function object at the given stack depth by correlating it with `gc.get_referrers`. (Inferred from implementation.)

**Returns:** `str` — Qualified name or `"<Module>"`.

---

### `log_caller`

```python
def log_caller() -> str:
```

_No docstring._ Returns a `"<function_name>"` string for the immediate caller. (Inferred from implementation.)

**Returns:** `str`

---

### `get_lm_from_tree`

```python
def get_lm_from_tree(loggertree: LoggerModel, find_me: str) -> LoggerModel:
```

_No docstring._ Depth-first search of a `LoggerModel` tree for a node named `find_me`. (Inferred from implementation.)

**Returns:** `LoggerModel | None`

---

### `generate_tree`

```python
def generate_tree() -> LoggerModel:
```

_No docstring._ Builds a `LoggerModel` tree representing the current `logging` hierarchy. (Inferred from implementation.)

**Returns:** `LoggerModel` — Root node.

---

## `tasks.symbols`

_No module docstring._

### `is_supported`

```python
def is_supported() -> bool:
```

Checks whether the operating system supports Unicode log symbols (non-Windows returns `True`).

**Returns:** `bool`

---

### `LogSymbols`

```python
class LogSymbols(Enum):
    INFO: str    # blue ℹ (or ¡ on Windows)
    SUCCESS: str # green ✔ (or v on Windows)
    WARNING: str # yellow ⚠ (or !! on Windows)
    ERROR: str   # red ✖ (or × on Windows)
```

**Class docstring:** _LogSymbol enum class._

Colored Unicode symbols for different log levels, compatible with colorama. Uses Unicode on non-Windows, ASCII fallbacks on Windows.

**Usage example:**
```python
from tasks.symbols import LogSymbols
print(LogSymbols.SUCCESS.value)  # ✔
print(LogSymbols.ERROR.value)    # ✖
```
