# screennet — API Reference

Package path: `screennet/`

`screennet` is the image-classification sub-project. It classifies screenshots as `twitter`, `facebook`, or `tiktok` using a pretrained EfficientNet-B0 backbone (via `torchvision`). The entry point is `screennet/main.py`, which is driven by an `argparse` CLI and is also importable as a module.

---

## `screennet.devices`

_No module docstring._ Handles device selection with CUDA → MPS → CPU priority. Implementation is identical to `screencropnet.devices`; the only difference is the import `from screennet import errors`.

For the full API of every function in this module see `screencropnet.devices` — the signatures are byte-for-byte identical. The functions present are:

| Function / constant | Brief description |
|---------------------|-------------------|
| `has_mps() -> bool` | Check MPS availability with a live tensor probe |
| `extract_device_id(args, name) -> str \| None` | Scan an arg list for a flag and return the next token |
| `get_optimal_device(args: argparse.Namespace) -> torch.device` | CUDA → MPS → CPU priority selection |
| `torch_gc() -> None` | Empty CUDA cache |
| `enable_tf32() -> None` | Enable TF32 on CUDA matmul/cuDNN |
| `randn(seed: int, shape: int) -> torch.Tensor` | Seeded random tensor with MPS workaround |
| `randn_without_seed(shape: int) -> torch.Tensor` | Unseeded random tensor with MPS workaround |
| `autocast(disable=False, precision="autocast")` | Mixed-precision context manager |
| `mps_contiguous(input_tensor, device) -> torch.Tensor` | Ensure contiguous layout on MPS |
| `mps_contiguous_to(input_tensor, device) -> torch.Tensor` | `mps_contiguous` then `.to(device)` |
| `mps_check() -> None` | Diagnostic: print MPS availability and exercise the device |
| `seed_everything(seed: int) -> None` | Set seeds for random, numpy, torch, cuDNN |

Module-level constants: `cpu = torch.device("cpu")`, `device = None`, `dtype = torch.float16`, `dtype_vae = torch.float16`.

---

## `screennet.errors`

_No module docstring._ Identical to `screencropnet.errors`.

### `run`

```python
def run(code, task: str) -> None:
```

_No docstring._ Calls `code()`, catching any exception and printing it to stderr. Used as a safe-import guard around `enable_tf32()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `code` | `callable` | Zero-argument callable |
| `task` | `str` | Label for the error message |

**Returns:** `None`

---

## `screennet.main`

_No module docstring._ Entry-point module for ScreenNet training, inference, evaluation, and utility operations. Driven by a module-level `argparse.ArgumentParser`. Also importable as a library — all public functions are accessible after import.

> **Warning:** `argparse.ArgumentParser` and several module-level side-effects (assertion guards on `torch`, `torchvision`, `mlxtend` versions) execute at import time.

### CLI Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `data` (positional) | `str` | `~/Downloads/datasets` | Root dataset directory |
| `-a` / `--arch` | `str` | `efficientnet_b0` | Model architecture (any `torchvision.models` name) |
| `--model-weights` | `str` | `EfficientNet_B0_Weights` | Pretrained weights class name |
| `-j` / `--workers` | `int` | `4` | DataLoader workers |
| `--epochs` | `int` | `5` | Training epochs |
| `--start-epoch` | `int` | `0` | Resume epoch offset |
| `-b` / `--batch-size` | `int` | `32` | Mini-batch size |
| `--lr` | `float` | `0.001` | Initial learning rate |
| `-p` / `--print-freq` | `int` | `10` | Log print frequency |
| `--resume` | `str` | `""` | Checkpoint path to resume from |
| `--predict` | `str` | `""` | Image path / directory for prediction |
| `--results` | `str` | `""` | CSV file path for prediction output |
| `--weights` | `str` | `""` | Saved weights path to load |
| `-e` / `--evaluate` | flag | — | Evaluate on validation set |
| `--test` | flag | — | Run test-set predictions |
| `--info` | flag | — | Print environment and dataset info, then exit |
| `--download-and-predict` | `str` | `""` | URL to download and predict |
| `--pretrained` | flag | `True` | Use pretrained model |
| `--interactive` | flag | `False` | Enable matplotlib interactive mode |
| `--debug` | flag | `False` | Enable extra logging |
| `--to-disk` | flag | `False` | Write output files to disk |
| `--summary` | flag | — | Print model summary |
| `--worst-first` | flag | — | Sort CSV by lowest prediction confidence |
| `--world-size` | `int` | `-1` | Nodes for distributed training |
| `--rank` | `int` | `-1` | Node rank for distributed training |
| `--dist-url` | `str` | `tcp://224.66.41.62:23456` | Distributed init URL |
| `--dist-backend` | `str` | `nccl` | Distributed backend |
| `--seed` | `int` | `42` | Random seed |
| `--gpu` | `int` | `None` | Specific GPU id |
| `--multiprocessing-distributed` | flag | — | Launch N processes per node |
| `--dummy` | flag | — | Use fake data for benchmarking |

---

### Public Functions

#### `_install_exception_hooks`

```python
def _install_exception_hooks() -> None:
```

Install `better_exceptions` pretty-traceback hooks. Called from `main()`, not at import. (Inferred from implementation.)

**Returns:** `None`

---

#### `get_pil_image_channels`

```python
def get_pil_image_channels(image_path: str) -> int:
```

Opens an image with Pillow and returns the number of channels.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | `str` | Path to image file |

**Returns:** `int` — Channel count (e.g. `3` for RGB, `4` for RGBA).

---

#### `convert_pil_image_to_rgb_channels`

```python
def convert_pil_image_to_rgb_channels(image_path: str) -> Image:
```

_No docstring._ Opens an image and converts it to RGB if it has fewer than 4 channels; returns as-is for 4-channel images. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | `str` | Path to image file |

**Returns:** `PIL.Image.Image`

---

#### `convert_pil_image_to_torch_tensor`

```python
def convert_pil_image_to_torch_tensor(pil_image: Image) -> torch.Tensor:
```

Converts a PIL image to a PyTorch tensor.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pil_image` | `PIL.Image.Image` | Input image |

**Returns:** `torch.Tensor`

---

#### `convert_tensor_to_pil_image`

```python
def convert_tensor_to_pil_image(tensor_image: torch.Tensor) -> Image:
```

Converts a PyTorch tensor to a PIL image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor_image` | `torch.Tensor` | Input tensor |

**Returns:** `PIL.Image.Image`

---

#### `predict_from_dir`

```python
def predict_from_dir(
    path_to_image_from_cli: str,
    model: torch.nn.Module,
    transforms: torchvision.transforms,
    class_names: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
```

Wrapper: iterates all images in a directory and calls `predict_from_file` for each.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path_to_image_from_cli` | `str` | Directory path |
| `model` | `torch.nn.Module` | Trained model |
| `transforms` | `torchvision.transforms` | Image transforms |
| `class_names` | `list[str]` | Class labels |
| `device` | `torch.device` | Inference device |
| `args` | `argparse.Namespace` | Parsed CLI args (used for `--to-disk`, `--results`) |

**Returns:** `None`

---

#### `predict_from_file`

```python
def predict_from_file(
    path_to_image_from_cli: str,
    model: torch.nn.Module,
    transforms: torchvision.transforms,
    class_names: list[str],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
```

Runs prediction on a single image file, prints metadata, and optionally writes results to CSV and plots to disk.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path_to_image_from_cli` | `str` | File path |
| `model` | `torch.nn.Module` | Trained model |
| `transforms` | `torchvision.transforms` | Image transforms |
| `class_names` | `list[str]` | Class labels |
| `device` | `torch.device` | Inference device |
| `args` | `argparse.Namespace` | Parsed CLI args |

**Returns:** `None`

---

#### `is_file`

```python
def is_file(path: str) -> bool:
```

Checks whether `path` points to a file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to check |

**Returns:** `bool`

---

#### `is_directory`

```python
def is_directory(path: str) -> bool:
```

Checks whether `path` points to a directory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to check |

**Returns:** `bool`

---

#### `tilda`

```python
def tilda(obj) -> str | list:
```

_No docstring._ Expands `~` in a path string or list of path strings to the user's home directory. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `obj` | `str \| list` | Path string or list of path strings |

**Returns:** `str | list` — Expanded path(s); non-string values are returned unchanged.

---

#### `fix_path`

```python
def fix_path(path: str) -> str | list:
```

_No docstring._ Resolves a path by expanding `~` and checking home-directory variants. Exits the process if the path cannot be resolved. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| list` | Path or list of paths |

**Returns:** `str | list` — Resolved path(s).

---

#### `from_pil_image_to_plt_display`

```python
def from_pil_image_to_plt_display(
    img: Image,
    pred_dicts: list[dict],
    to_disk: bool = True,
    interactive: bool = True,
    fname: str = "plot.png",
) -> None:
```

Renders a PIL image as a matplotlib figure annotated with prediction class, probability, and timing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img` | `PIL.Image.Image` | — | Image to display |
| `pred_dicts` | `list[dict]` | — | List of prediction dicts; uses `pred_dicts[0]` |
| `to_disk` | `bool` | `True` | Save figure to `fname` |
| `interactive` | `bool` | `True` | Enable `plt.ion()` |
| `fname` | `str` | `"plot.png"` | Output filename |

**Returns:** `None`

---

#### `create_writer`

```python
def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str = None,
) -> SummaryWriter:
```

Creates a `torch.utils.tensorboard.SummaryWriter` saving to `runs/<YYYY-MM-DD>/<experiment_name>/<model_name>[/<extra>]/`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | `str` | — | Experiment label |
| `model_name` | `str` | — | Model label |
| `extra` | `str` | `None` | Optional extra subdirectory segment |

**Returns:** `torch.utils.tensorboard.writer.SummaryWriter`

---

#### `download_and_predict`

```python
def download_and_predict(
    url: str,
    model: torch.nn.Module,
    data_path: pathlib.PosixPath,
    class_names: list[str],
    device: torch.device = None,
) -> None:
```

_No docstring._ Downloads an image from `url` into `data_path`, then calls `pred_and_plot_image`. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | — | Remote image URL |
| `model` | `torch.nn.Module` | — | Trained model |
| `data_path` | `pathlib.PosixPath` | — | Directory for the downloaded image |
| `class_names` | `list[str]` | — | Class labels |
| `device` | `torch.device` | `None` | Inference device |

**Returns:** `None`

---

#### `show_confusion_matrix_helper`

```python
def show_confusion_matrix_helper(
    cmat: np.ndarray,
    class_names: list[str],
    to_disk: bool = True,
    fname: str = "plot.png",
) -> None:
```

_No docstring._ Renders a confusion matrix using `mlxtend.plotting.plot_confusion_matrix` with a log-norm colormap. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmat` | `np.ndarray` | — | Confusion matrix array |
| `class_names` | `list[str]` | — | Class labels for axes |
| `to_disk` | `bool` | `True` | Save to `fname` |
| `fname` | `str` | `"plot.png"` | Output filename |

**Returns:** `None`

---

#### `compute_accuracy`

```python
def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> torch.Tensor:
```

_No docstring._ Evaluates accuracy of `model` on `data_loader` under `torch.no_grad()`. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Model to evaluate |
| `data_loader` | `DataLoader` | Data to evaluate on |
| `device` | `str` | Device string |

**Returns:** `torch.Tensor` — Accuracy as a percentage.

---

#### `compute_epoch_loss`

```python
def compute_epoch_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> torch.Tensor:
```

_No docstring._ Computes mean cross-entropy loss over all batches under `torch.no_grad()`. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Model to evaluate |
| `data_loader` | `DataLoader` | Data to evaluate on |
| `device` | `str` | Device string |

**Returns:** `torch.Tensor` — Mean loss.

---

#### `compute_confusion_matrix`

```python
def compute_confusion_matrix(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device,
) -> np.ndarray:
```

_No docstring._ Computes a confusion matrix by running inference over `data_loader` and comparing predicted vs. true labels. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Model |
| `data_loader` | `DataLoader` | Data |
| `device` | device-like | Inference device |

**Returns:** `np.ndarray` — Square confusion matrix of shape `(n_classes, n_classes)`.

---

#### `run_confusion_matrix`

```python
def run_confusion_matrix(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
) -> None:
```

_No docstring._ Wrapper: calls `compute_confusion_matrix` then `show_confusion_matrix_helper`. (Inferred from implementation.)

**Returns:** `None`

---

#### `run_validate`

```python
def run_validate(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> None:
```

_No docstring._ Evaluates `model` on `test_dataloader` via `engine.test_step`, prints loss and accuracy, and reports elapsed time. (Inferred from implementation.)

**Returns:** `None`

---

#### `run_train`

```python
def run_train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    batch_size: int,
) -> None:
```

_No docstring._ Runs the full training loop via `engine.train`, times it, saves the model, writes a benchmark CSV, and plots loss curves. (Inferred from implementation.)

**Returns:** `None`

---

#### `write_training_results_to_csv`

```python
def write_training_results_to_csv(
    MACHINE,
    device,
    dataset_name="",
    num_epochs="",
    batch_size="",
    image_size="",
    train_data="",
    test_data="",
    total_train_time="",
    model="",
) -> None:
```

_No docstring._ Writes a one-row benchmark CSV to `results/<machine>_<device>_<dataset_name>_image_size.csv`. (Inferred from implementation.)

**Returns:** `None`

---

#### `write_predict_results_to_csv`

```python
def write_predict_results_to_csv(
    pred_dicts: list[dict],
    args: argparse.Namespace,
) -> None:
```

_No docstring._ Appends or creates a CSV file at `args.results` with prediction results. (Inferred from implementation.)

**Returns:** `None`

---

#### `df_to_table`

```python
def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = True,
    index_name: str | None = None,
) -> Table:
```

Converts a Pandas DataFrame to a `rich.Table` instance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pandas_dataframe` | `pd.DataFrame` | — | Source DataFrame |
| `rich_table` | `rich.table.Table` | — | Target rich Table to populate |
| `show_index` | `bool` | `True` | Add a row-index column |
| `index_name` | `str \| None` | `None` | Header for the index column |

**Returns:** `rich.table.Table` — The populated table.

---

#### `console_print_table`

```python
def console_print_table(results_df: pd.DataFrame) -> None:
```

_No docstring._ Renders a DataFrame as a styled rich table to the module-level `console`. (Inferred from implementation.)

**Returns:** `None`

---

#### `csv_to_df`

```python
def csv_to_df(path: str) -> pd.DataFrame:
```

_No docstring._ Reads a CSV file into a DataFrame. (Inferred from implementation.)

**Returns:** `pd.DataFrame`

---

#### `inspect_csv_results`

```python
def inspect_csv_results() -> pd.DataFrame:
```

_No docstring._ Reads and concatenates all `*.csv` files under `results/`, prints them as a table, and returns the combined DataFrame. (Inferred from implementation.)

**Returns:** `pd.DataFrame`

---

#### `walk_through_dir`

```python
def walk_through_dir(dir_path) -> None:
```

Walks through `dir_path` and prints subdirectory and file counts at each level.

**Returns:** `None`

---

#### `create_effnetb0_model`

```python
def create_effnetb0_model(
    device: str,
    class_names: list[str],
    args: argparse.Namespace,
) -> torch.nn.Module:
```

Creates a pretrained EfficientNet-B0 model, freezes the feature extractor, replaces the classifier with a dropout + linear head for `len(class_names)` outputs, and attaches `model.name = "effnetb0"`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | `str` | Target device string |
| `class_names` | `list[str]` | Class labels (determines output units) |
| `args` | `argparse.Namespace` | Provides `args.model_weights` and `args.seed` |

**Returns:** `torch.nn.Module` — Configured EfficientNet-B0.

---

#### `get_model_summary`

```python
def get_model_summary(
    model: torch.nn.Module,
    input_size: tuple = (32, 3, 224, 224),
    verbose: int = 0,
    col_names: list[str] | None = None,
    col_width: int = 20,
    row_settings: list[str] | None = None,
) -> None:
```

_No docstring._ Prints a `torchinfo.summary` for the model. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Model to summarise |
| `input_size` | `tuple` | `(32, 3, 224, 224)` | `(batch, C, H, W)` input shape |
| `verbose` | `int` | `0` | torchinfo verbosity |
| `col_names` | `list[str] \| None` | `["input_size","output_size","num_params","trainable"]` | Columns to display |
| `col_width` | `int` | `20` | Column width |
| `row_settings` | `list[str] \| None` | `["var_names"]` | Row display settings |

**Returns:** `None`

---

#### `pred_and_plot_image`

```python
def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list[str],
    image_size: tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = None,
    y_preds: list[torch.Tensor] | None = None,
    y_pred_tensor: torch.Tensor = None,
) -> None:
```

_No docstring._ Runs inference on a single image and plots the result. Appends the prediction probability to `y_preds` (accumulator list). Saves the figure to `prediction-<model.name>-<stem>.png`. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | — | Trained model |
| `image_path` | `str` | — | Image file path |
| `class_names` | `list[str]` | — | Class labels |
| `image_size` | `tuple[int, int]` | `(224, 224)` | Unused when `transform` is provided |
| `transform` | `torchvision.transforms` | `None` | Required transform (None causes a runtime error) |
| `device` | `torch.device` | `None` | Inference device |
| `y_preds` | `list[torch.Tensor] \| None` | `None` | Accumulator list for softmax probabilities |
| `y_pred_tensor` | `torch.Tensor` | `None` | Unused parameter (reserved) |

**Returns:** `None`

---

#### `run_save_model_for_inference`

```python
def run_save_model_for_inference(model: torch.nn.Module) -> tuple[pathlib.PosixPath]:
```

Saves model weights to `models/ScreenNetV1.pth` and returns the path.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Model to save |

**Returns:** `tuple[pathlib.PosixPath]` (effectively a single `Path`).

---

#### `run_get_model_for_inference`

```python
def run_get_model_for_inference(
    model: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
    path_to_model: pathlib.PosixPath,
    args: argparse.Namespace,
) -> torch.nn.Module:
```

Wrapper: loads model weights from `path_to_model` via `load_model_for_inference`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `torch.nn.Module` | Uninitialised model (architecture template) |
| `device` | `torch.device` | Target device |
| `class_names` | `list[str]` | Class labels |
| `path_to_model` | `pathlib.PosixPath` | Path to `.pth` file |
| `args` | `argparse.Namespace` | Parsed CLI args |

**Returns:** `torch.nn.Module` — Loaded model in eval mode.

---

#### `save_model_to_disk`

```python
def save_model_to_disk(my_model_name: str, model: torch.nn.Module) -> pathlib.Path:
```

_No docstring._ Saves `model.state_dict()` to `models/<my_model_name>.pth`. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `my_model_name` | `str` | Base filename (without extension) |
| `model` | `torch.nn.Module` | Model to save |

**Returns:** `pathlib.Path` — Full path to the saved file.

---

#### `load_model_for_inference`

```python
def load_model_for_inference(
    save_path: str,
    device: str,
    class_names: list[str],
    args: argparse.Namespace,
) -> nn.Module:
```

_No docstring._ Creates a fresh EfficientNet-B0 via `create_effnetb0_model`, loads state dict from `save_path` with `weights_only=True`, and sets eval mode. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `save_path` | `str` | Path to `.pth` file |
| `device` | `str` | Target device |
| `class_names` | `list[str]` | Class labels |
| `args` | `argparse.Namespace` | Parsed CLI args |

**Returns:** `nn.Module` — Loaded model in eval mode.

---

#### `load_model_from_disk`

```python
def load_model_from_disk(save_path: str, empty_model: nn.Module) -> nn.Module:
```

_No docstring._ Loads a state dict into `empty_model` from `save_path` with `weights_only=True` and sets eval mode. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `save_path` | `str` | Path to `.pth` file |
| `empty_model` | `nn.Module` | Pre-built model with matching architecture |

**Returns:** `nn.Module` — Loaded model in eval mode.

---

#### `plot_image_with_predicted_label`

```python
def plot_image_with_predicted_label(
    to_disk: bool = True,
    img: Image = None,
    target_image_pred_label: torch.Tensor = None,
    target_image_pred_probs: torch.Tensor = None,
    class_names: list[str] = None,
    fname: str = "plot.png",
) -> None:
```

_No docstring._ Plots a PIL image with predicted class label and probability as the title. Saves with `plt.imsave` if `to_disk=True`. (Inferred from implementation.)

**Returns:** `None`

---

#### `validate_seed`

```python
def validate_seed(seed: int) -> None:
```

Sets random seeds via `devices.seed_everything` and performs a quick numerical validation to confirm MPS randomness matches CPU. (Docstring notes reproduce warning about MPS `from_numpy` shared memory.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | `int` | Seed value |

**Returns:** `None`

---

#### `info`

```python
def info(args, dataset_root_dir="") -> None:
```

_No docstring._ Prints watermark (package versions), MPS check, seed validation, directory walk, then calls `sys.exit(0)`. (Inferred from implementation.)

**Returns:** Does not return — calls `sys.exit(0)`.

---

#### `load_checkpoint`

```python
def load_checkpoint(resume_path: str, gpu: int | None = None) -> dict:
```

Load a full training checkpoint (uses `weights_only=False` — trusted local file).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resume_path` | `str` | — | Path to the checkpoint file |
| `gpu` | `int \| None` | `None` | GPU id; if set, maps the checkpoint to that device |

**Returns:** `dict` — Checkpoint dictionary (typically contains `epoch`, `best_acc1`, `state_dict`, `optimizer`).

---

#### `get_model_named_params`

```python
def get_model_named_params(model: torch.nn.Module) -> None:
```

_No docstring._ Prints each parameter name and its `requires_grad` flag. (Inferred from implementation.)

**Returns:** `None`

---

#### `print_train_time`

```python
def print_train_time(start, end, device=None, machine=None) -> float:
```

Prints and returns elapsed training time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `float` | — | Start timestamp |
| `end` | `float` | — | End timestamp |
| `device` | any | `None` | Device label for the print message |
| `machine` | any | `None` | Machine label for the print message |

**Returns:** `float` — Elapsed seconds (rounded to 3 decimal places).

---

#### `pred_and_store`

```python
def pred_and_store(
    paths: list[pathlib.Path],
    model: torch.nn.Module,
    transform: torchvision.transforms,
    class_names: list[str],
    device: torch.device = "",
) -> list[dict]:
```

_No docstring._ Runs inference on each image path and returns a list of prediction dictionaries. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `paths` | `list[pathlib.Path]` | — | Image paths |
| `model` | `torch.nn.Module` | — | Trained model |
| `transform` | `torchvision.transforms` | — | Image transform |
| `class_names` | `list[str]` | — | Class labels |
| `device` | `torch.device` | `""` | Inference device |

**Returns:** `list[dict]` — Each dict contains `image_path`, `class_name`, `pred_prob`, `pred_class`, `time_for_pred`, `correct`.

---

#### `accuracy`

```python
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
```

Computes top-k accuracy for the specified values of k.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output` | `torch.Tensor` | — | Model logits `(N, num_classes)` |
| `target` | `torch.Tensor` | — | True labels `(N,)` |
| `topk` | `tuple` | `(1,)` | Values of k to compute accuracy for |

**Returns:** `list` — One tensor per k, each containing the top-k accuracy percentage.

---

#### `get_random_images_from_dataset`

```python
def get_random_images_from_dataset(
    model: torch.nn.Module,
    test_dir: pathlib.PosixPath,
    class_names: list[str],
    num_images_to_plot: int = 3,
    device: torch.device = None,
    y_preds: list[torch.Tensor] | None = None,
    y_pred_tensor: torch.Tensor = None,
) -> None:
```

_No docstring._ Randomly samples images from `test_dir` and calls `pred_and_plot_image` on each. (Inferred from implementation.)

**Returns:** `None`

---

#### `get_random_perdictions_and_plots`

```python
def get_random_perdictions_and_plots(
    best_model: nn.Module,
    test_dir: pathlib.PosixPath = "",
    class_names: list[str] = None,
    transform: torchvision.transforms = None,
    device: torch.device = None,
) -> None:
```

_No docstring._ Samples 3 random images from `test_dir` and runs `pred_and_plot_image` on each. (Inferred from implementation.)

**Returns:** `None`

---

#### `save_checkpoint`

```python
def save_checkpoint(state, is_best: bool, filename: str = "checkpoint.pth.tar") -> None:
```

_No docstring._ Saves a training state dict to `filename`; copies to `model_best.pth.tar` if `is_best`. (Inferred from implementation.)

**Returns:** `None`

---

#### `setup_workspace`

```python
def setup_workspace(
    data_path: pathlib.PosixPath,
    image_path: pathlib.PosixPath,
) -> None:
```

_No docstring._ Creates the dataset directory if missing and downloads the twitter/facebook/tiktok zip from Dropbox, then unzips it. (Inferred from implementation.)

**Returns:** `None`

---

### Classes

#### `Summary`

```python
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
```

_No docstring._ Enum controlling how `AverageMeter.summary()` formats its output. (Inferred from implementation.)

---

#### `AverageMeter`

```python
class AverageMeter:
    def __init__(
        self,
        name: str,
        fmt: str = ":f",
        summary_type: Summary = Summary.AVERAGE,
    ) -> None:
```

**Class docstring:** _Computes and stores the average and current value._

Used to track running metrics (loss, accuracy) during training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Metric name |
| `fmt` | `str` | `":f"` | Python format string suffix for display |
| `summary_type` | `Summary` | `Summary.AVERAGE` | Which aggregate to show in `summary()` |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset()` | `-> None` | Resets val/avg/sum/count to 0 |
| `update(val, n=1)` | `-> None` | Accumulate value `val` weighted by `n` |
| `all_reduce()` | `-> None` | Distributed: sum across processes via `dist.all_reduce` |
| `__str__()` | `-> str` | Formatted current and average values |
| `summary()` | `-> str` | Formatted summary based on `summary_type` |

---

#### `ProgressMeter`

```python
class ProgressMeter:
    def __init__(
        self,
        num_batches: int,
        meters: list,
        prefix: str = "",
    ) -> None:
```

_No docstring._ Formats and prints training progress using a collection of `AverageMeter` instances. (Inferred from implementation.)

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `display(batch)` | `-> None` | Print current batch progress |
| `display_summary()` | `-> None` | Print final epoch summary |

---

#### `main`

```python
def main() -> None:
```

_No docstring._ CLI entry point. Parses arguments, sets up distributed training if configured, loads or builds the EfficientNet-B0 model, creates DataLoaders, and dispatches to train/evaluate/predict/test modes. (Inferred from implementation.)

**Returns:** `None`

---

#### `main_worker`

```python
def main_worker(gpu: int, ngpus_per_node: int, args: argparse.Namespace) -> None:
```

_No docstring._ Core training worker function, called by `main()` directly or via `torch.multiprocessing.spawn`. Sets up distributed process groups, creates the model, DataLoaders, loss, and optimizer, then dispatches to the appropriate mode. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpu` | `int` | GPU id for this process |
| `ngpus_per_node` | `int` | Total GPUs on this node |
| `args` | `argparse.Namespace` | Parsed CLI arguments |

**Returns:** `None`
