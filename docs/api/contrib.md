# contrib — API Reference

Directory: `contrib/`

The `contrib/` directory contains standalone PEP 723 scripts — each carries its own inline dependency metadata and can be run directly with `uv run`. They are not importable as library modules.

---

## `contrib/fetch_screencropnet_assets.py`

**Module docstring:** _Idempotent fetcher for screencropnet localization assets. Provenance recovered from the (un-committed) reference notebooks and recorded in `ai_docs/screencropnet-assets.md`. Destinations live under `scratch/` (git-ignored) and `screencropnet/models/`. Re-running is safe: any asset whose target already exists is skipped._

**PEP 723 header:**
```
requires-python = ">=3.12"
dependencies = ["requests"]
```

**Usage:**
```bash
uv run contrib/fetch_screencropnet_assets.py --all
uv run contrib/fetch_screencropnet_assets.py --dataset --sample
uv run contrib/fetch_screencropnet_assets.py --weights --force
```

---

### Module-level constants

#### `REPO_ROOT`

```python
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
```

Absolute path to the repository root. Used as the base for all `dest` paths.

---

#### `ASSETS`

```python
ASSETS: dict[str, Asset] = {
    "dataset": Asset(...),
    "weights": Asset(...),
    "weights-collab": Asset(...),
    "weights-best": Asset(...),
    "sample": Asset(...),
}
```

Registry of all downloadable assets. Keys are stable string identifiers.

| Key | Kind | Description |
|-----|------|-------------|
| `"dataset"` | `"zip"` | Localization dataset → `scratch/datasets/twitter_screenshots_localization_dataset/` |
| `"weights"` | `"file"` | Primary checkpoint (378 epochs) → `screencropnet/models/ScreenCropNetV1_378_epochs.pth` |
| `"weights-collab"` | `"file"` | Colab-trained checkpoint → `screencropnet/models/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth` |
| `"weights-best"` | `"file"` | Best-model checkpoint → `screencropnet/models/screencropnet_best_model.pt` |
| `"sample"` | `"file"` | Sample image → `scratch/IMG_6324.PNG` |

#### `WEIGHTS_KEYS`

```python
WEIGHTS_KEYS: list[str] = ["weights", "weights-collab", "weights-best"]
```

Convenience list: all three weight asset keys fetched when `--weights` is passed.

---

### `Asset`

```python
@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    dest: str
    kind: str           # "file" | "zip"
    unpack_into: str | None = None
```

**Class docstring:** _A downloadable asset. `dest` is repo-root-relative. For `kind == "zip"` the archive is unpacked into `unpack_into` (repo-root-relative dir) and `dest` is the extracted directory used for the idempotency check._

Immutable dataclass (`frozen=True`) describing a single downloadable asset.

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Stable string identifier matching the `ASSETS` key |
| `url` | `str` | Remote URL (Dropbox direct-download link) |
| `dest` | `str` | Repo-root-relative destination path (file or directory) |
| `kind` | `str` | `"file"` — download directly; `"zip"` — download and extract |
| `unpack_into` | `str \| None` | For `kind == "zip"`: repo-root-relative extraction directory |

---

### `_download`

```python
def _download(url: str, target: Path) -> None:
```

_No docstring._ Downloads `url` to `target` using streaming requests with a 120-second timeout. Creates parent directories if needed. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Remote URL |
| `target` | `pathlib.Path` | Local destination file |

**Returns:** `None`

**Raises:** `requests.HTTPError` if the response status is not 2xx.

---

### `fetch`

```python
def fetch(asset: Asset, *, force: bool = False) -> Path:
```

**Docstring:** _Fetch `asset` idempotently; return the resolved repo-root path._

Downloads or skips an asset depending on whether its `dest` already exists. For `"zip"` assets: downloads the archive to a temporary location, extracts it to `unpack_into`, then deletes the archive.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `asset` | `Asset` | — | Asset descriptor |
| `force` | `bool` | `False` | Re-download even if `dest` already exists |

**Returns:** `pathlib.Path` — Resolved absolute path to the asset destination.

**Raises:** `requests.HTTPError` on HTTP errors during download.

---

### `_selected_keys`

```python
def _selected_keys(args: argparse.Namespace) -> list[str]:
```

_No docstring._ Translates parsed CLI flags to a list of asset keys. `--all` selects every asset; individual flags select their corresponding keys. `--weights` expands to all three weight keys. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `args` | `argparse.Namespace` | Parsed arguments with boolean attributes `all`, `dataset`, `weights`, `sample` |

**Returns:** `list[str]` — Ordered list of asset keys to fetch.

---

### `build_parser`

```python
def build_parser() -> argparse.ArgumentParser:
```

_No docstring._ Constructs and returns the argument parser for the script. (Inferred from implementation.)

**Returns:** `argparse.ArgumentParser`

**CLI flags:**

| Flag | Action | Description |
|------|--------|-------------|
| `--dataset` | `store_true` | Fetch the localization dataset zip |
| `--weights` | `store_true` | Fetch the inference checkpoint and two alternates |
| `--sample` | `store_true` | Fetch the sample image `IMG_6324.PNG` |
| `--all` | `store_true` | Fetch every asset |
| `--force` | `store_true` | Re-download even if the target exists |

---

### `main`

```python
def main(argv: list[str] | None = None) -> int:
```

_No docstring._ Script entry point. Parses `argv` (defaults to `sys.argv`), resolves which asset keys to fetch, calls `fetch` for each, and prints dataset path information when the dataset is fetched. Returns `1` if no assets were selected. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `argv` | `list[str] \| None` | `None` | Argument list; defaults to `sys.argv[1:]` |

**Returns:** `int` — Exit code: `0` on success, `1` if no assets were selected.
