# going_modular — API Reference

Package path: `going_modular/going_modular/`
Installed name: `going_modular` (editable install via `pyproject.toml`).

The package provides reusable training primitives used by both `screennet/` and
`screencropnet/` main entry-points.

---

## `going_modular.data_setup`

**Module docstring:** _Contains functionality for creating PyTorch DataLoaders for image classification data._

Module-level constant: `NUM_WORKERS = os.cpu_count()`

### `display_ascii_text`

```python
def display_ascii_text(txt: str, font: str = "stop") -> None:
```

_No docstring._ Renders `txt` as ASCII art using `pyfiglet` and prints it with magenta styling via `rich`. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `txt` | `str` | — | Text to render |
| `font` | `str` | `"stop"` | pyfiglet font name |

**Returns:** `None`

---

### `create_dataloaders`

```python
def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
```

Creates training and testing DataLoaders from image folder directories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_dir` | `str` | — | Path to the training directory |
| `test_dir` | `str` | — | Path to the testing directory |
| `transform` | `transforms.Compose` | — | torchvision transforms applied to both splits |
| `batch_size` | `int` | — | Number of samples per batch |
| `num_workers` | `int` | `os.cpu_count()` | Workers per DataLoader |
| `pin_memory` | `bool` | `False` | Enable pinned memory for faster GPU transfer |

**Returns:** `tuple[DataLoader, DataLoader, list[str]]` — `(train_dataloader, test_dataloader, class_names)`. `class_names` is derived from the subdirectory names in `train_dir`.

---

## `going_modular.engine`

**Module docstring:** _Contains functions for training and testing a PyTorch model._

### `calculate_IoU`

```python
def calculate_IoU(bb1, bb2) -> float:
```

_No docstring._ Calculates Intersection over Union (IoU) of two axis-aligned bounding boxes. Returns `0.0` when there is no overlap. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bb1` | sequence of 4 numbers | — | `(xmin, ymin, xmax, ymax)` for box 1 |
| `bb2` | sequence of 4 numbers | — | `(xmin, ymin, xmax, ymax)` for box 2 |

**Returns:** `float` — IoU value in `[0.0, 1.0]`.

---

### `display_ascii_text`

```python
def display_ascii_text(txt: str, font: str = "stop") -> None:
```

_No docstring._ Identical helper to `data_setup.display_ascii_text`; defined locally in this module. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `txt` | `str` | — | Text to render |
| `font` | `str` | `"stop"` | pyfiglet font name |

**Returns:** `None`

---

### `train_step`

```python
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
```

Trains a PyTorch model for a single epoch. Enables the PyTorch profiler writing traces to `./runs/profiler`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Model to train |
| `dataloader` | `DataLoader` | — | Training data loader |
| `loss_fn` | `torch.nn.Module` | — | Loss function to minimize |
| `optimizer` | `torch.optim.Optimizer` | — | Optimizer |
| `device` | `torch.device` | — | Target compute device |

**Returns:** `tuple[float, float]` — `(train_loss, train_accuracy)`, both averaged over all batches.

---

### `test_step`

```python
def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
```

Tests a PyTorch model for a single epoch in `torch.inference_mode()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Model to evaluate |
| `dataloader` | `DataLoader` | — | Testing data loader |
| `loss_fn` | `torch.nn.Module` | — | Loss function |
| `device` | `torch.device` | — | Target compute device |

**Returns:** `tuple[float, float]` — `(test_loss, test_accuracy)`, both averaged over all batches.

---

### `train`

```python
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: SummaryWriter,
) -> dict[str, list]:
```

Full training loop: calls `train_step` and `test_step` for each epoch, logs metrics to TensorBoard if `writer` is not `None`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Model to train and test |
| `train_dataloader` | `DataLoader` | — | Training data |
| `test_dataloader` | `DataLoader` | — | Test/validation data |
| `optimizer` | `torch.optim.Optimizer` | — | Optimizer |
| `loss_fn` | `torch.nn.Module` | — | Loss function |
| `epochs` | `int` | — | Number of epochs |
| `device` | `torch.device` | — | Target compute device |
| `writer` | `SummaryWriter` | — | TensorBoard writer; pass `None` to disable logging |

**Returns:** `dict[str, list]` with keys `train_loss`, `train_acc`, `test_loss`, `test_acc`, each mapping to a list of per-epoch float values.

---

### `train_localization_fn`

```python
def train_localization_fn(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
```

_No docstring._ Single-epoch training function for bounding-box localization models. Expects the model's `forward()` to accept `(images, gt_bboxes)` and return `(bboxes, loss)`. Enables the PyTorch profiler. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Localization model |
| `dataloader` | `DataLoader` | — | Training data yielding `(images, gt_bboxes)` |
| `optimizer` | `torch.optim.Optimizer` | — | Optimizer |
| `device` | `torch.device` | — | Target compute device |

**Returns:** `float` — Mean training loss over all batches.

---

### `eval_localization_fn`

```python
def eval_localization_fn(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
```

_No docstring._ Single-epoch evaluation function for localization models under `torch.inference_mode()`. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Localization model |
| `dataloader` | `DataLoader` | — | Validation data |
| `device` | `torch.device` | — | Target compute device |

**Returns:** `float` — Mean validation loss over all batches.

---

### `train_localization`

```python
def train_localization(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    validloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    writer: SummaryWriter = None,
) -> None:
```

_No docstring._ Full training loop for localization models. Saves the best checkpoint to `screencropnet_best_model.pt` whenever `valid_loss` improves. Optionally logs to TensorBoard. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Localization model |
| `trainloader` | `DataLoader` | — | Training data |
| `validloader` | `DataLoader` | — | Validation data |
| `optimizer` | `torch.optim.Optimizer` | — | Optimizer |
| `epochs` | `int` | — | Number of epochs |
| `device` | `torch.device` | — | Target compute device |
| `writer` | `SummaryWriter` | `None` | TensorBoard writer; `None` disables logging |

**Returns:** `None`. Side-effect: writes `screencropnet_best_model.pt` to the working directory.

---

## `going_modular.model_builder`

**Module docstring:** _Contains PyTorch model code to instantiate a TinyVGG model._

### `TinyVGG`

```python
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
```

Replicates the TinyVGG architecture from the [CNN Explainer](https://poloclub.github.io/cnn-explainer/) website.

Architecture: two convolutional blocks (each: Conv2d → ReLU → Conv2d → ReLU → MaxPool2d) followed by a Flatten + Linear classifier.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_shape` | `int` | — | Number of input channels (e.g. `3` for RGB) |
| `hidden_units` | `int` | — | Number of feature maps in each convolutional layer |
| `output_shape` | `int` | — | Number of output classes |

**Methods:**

#### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

_No docstring._ Standard forward pass through both conv blocks then the classifier. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor of shape `(N, input_shape, H, W)` |

**Returns:** `torch.Tensor` — Logits of shape `(N, output_shape)`.

> **Note:** The classifier's `in_features` is hard-coded as `hidden_units * 13 * 13`, which assumes a `64 x 64` input image after the two pooling operations.

---

## `going_modular.utils`

**Module docstring:** _Contains various utility functions for PyTorch model training and saving._

### `save_model`

```python
def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
```

Saves the model's `state_dict()` to `target_dir / model_name`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Model whose weights to save |
| `target_dir` | `str` | — | Directory to save into (created if absent) |
| `model_name` | `str` | — | Filename; must end with `.pth` or `.pt` |

**Returns:** `None`

**Raises:** `AssertionError` if `model_name` does not end with `.pth` or `.pt`.

---

## `going_modular.predictions`

**Module docstring:** _Utility functions to make predictions._

Module-level constant: `device = "cuda" if torch.cuda.is_available() else "cpu"`

### `pred_and_plot_image`

```python
def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: list[str],
    image_path: str,
    image_size: tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = device,
) -> None:
```

Loads an image, runs inference with a trained model, and displays the result via matplotlib.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Trained (or untrained) model |
| `class_names` | `list[str]` | — | Class labels indexed by model output |
| `image_path` | `str` | — | Path to the image file |
| `image_size` | `tuple[int, int]` | `(224, 224)` | Resize target if no custom transform is provided |
| `transform` | `torchvision.transforms` | `None` | Custom transform; defaults to ImageNet normalization |
| `device` | `torch.device` | module-level `device` | Device for inference |

**Returns:** `None`. Displays a matplotlib figure with predicted label and probability as the title.

---

## `going_modular.train`

**Module docstring:** _Trains a PyTorch image classification model using device-agnostic code._

This module is an executable script. It is not intended to be imported as a library; all logic lives inside `main()`.

### `main`

```python
def main() -> None:
```

_No docstring._ Script entry point: sets hyperparameters (5 epochs, batch 32, 10 hidden units, lr 0.001), creates pizza/steak/sushi DataLoaders, instantiates a `TinyVGG`, trains it via `engine.train`, and saves the result to `models/05_going_modular_script_mode_tinyvgg_model.pth`. (Inferred from implementation.)

**Returns:** `None`
