# helper_functions — API Reference

File: `helper_functions.py` (repo root)

**Module docstring:** _A series of helper functions used throughout the course. If a function gets defined once and could be used over and over, it'll go in here._

This is a standalone module imported by both `screennet/main.py` and `screencropnet/main.py`. It provides plotting, accuracy calculation, timing, and download utilities.

---

## Functions

### `walk_through_dir`

```python
def walk_through_dir(dir_path) -> None:
```

Walks through `dir_path` and prints the number of subdirectories and files in each directory level.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dir_path` | `str` | — | Target directory path |

**Returns:** `None`. Side-effect: prints directory statistics to stdout.

---

### `plot_decision_boundary`

```python
def plot_decision_boundary(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> None:
```

Plots the decision boundary of a model on a 2-D feature space. Moves all data to CPU before plotting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Trained classification model |
| `X` | `torch.Tensor` | — | 2-D feature matrix of shape `(N, 2)` |
| `y` | `torch.Tensor` | — | Ground-truth labels of shape `(N,)` |

**Returns:** `None`. Produces a matplotlib `contourf` + `scatter` plot.

> For binary problems the model output is passed through `torch.sigmoid`; for multi-class problems through `torch.softmax`.

---

### `plot_predictions`

```python
def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None,
) -> None:
```

Plots linear training data, test data, and optionally predictions on a scatter plot.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data` | array-like | — | Training feature values |
| `train_labels` | array-like | — | Training label values |
| `test_data` | array-like | — | Test feature values |
| `test_labels` | array-like | — | Test label values |
| `predictions` | array-like | `None` | Optional predicted values to overlay in red |

**Returns:** `None`. Displays a matplotlib scatter plot.

---

### `accuracy_fn`

```python
def accuracy_fn(y_true, y_pred) -> float:
```

Calculates percentage accuracy between truth labels and predictions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_true` | `torch.Tensor` | — | Ground-truth labels |
| `y_pred` | `torch.Tensor` | — | Predicted labels |

**Returns:** `float` — Accuracy as a percentage (e.g., `78.45`).

---

### `print_train_time`

```python
def print_train_time(start, end, device=None) -> float:
```

Prints and returns the elapsed training time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `float` | — | Start timestamp (timeit format recommended) |
| `end` | `float` | — | End timestamp |
| `device` | any | `None` | Device label for the printed message |

**Returns:** `float` — Elapsed time in seconds.

---

### `plot_loss_curves`

```python
def plot_loss_curves(results, to_disk: bool = False) -> None:
```

Plots training and test loss/accuracy curves from a results dictionary.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `dict` | — | Dict with keys `train_loss`, `train_acc`, `test_loss`, `test_acc`, each a list of per-epoch floats |
| `to_disk` | `bool` | `False` | If `True`, saves the figure to `model-loss-curves.png` |

**Returns:** `None`.

---

### `pred_and_plot_image`

```python
def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list[str] = None,
    transform=None,
    device: torch.device | None = None,
) -> None:
```

Loads an image via `torchvision.io.read_image`, normalises pixels to `[0, 1]`, runs inference, and displays a matplotlib figure with the prediction label and probability.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Trained image classification model |
| `image_path` | `str` | — | Path to the target image |
| `class_names` | `list[str]` | `None` | Class labels; if `None` the raw label index is shown |
| `transform` | callable | `None` | Optional torchvision transform applied before inference |
| `device` | `torch.device \| None` | `None` | Device; defaults to CUDA if available, else CPU |

**Returns:** `None`. Displays a matplotlib figure.

---

### `set_seeds`

```python
def set_seeds(seed: int = 42) -> None:
```

Sets random seeds for `torch` (CPU) and `torch.cuda` (GPU) operations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed value |

**Returns:** `None`

---

### `download_data`

```python
def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
```

Downloads a zipped dataset from `source` URL, extracts it under `data/<destination>/`, and optionally removes the zip archive.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | — | URL of the zipped dataset |
| `destination` | `str` | — | Target subdirectory name under `data/` |
| `remove_source` | `bool` | `True` | Whether to delete the zip after extraction |

**Returns:** `pathlib.Path` — Path to the extracted data directory (`data/<destination>`).

> Skips the download silently if the destination directory already exists.
