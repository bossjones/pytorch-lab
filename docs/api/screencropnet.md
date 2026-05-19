# screencropnet — API Reference

Package path: `screencropnet/`

`screencropnet` is the bounding-box regression sub-project. It uses an EfficientNet-B0 backbone (via `timm`) to predict `[xmin, ymin, xmax, ymax]` coordinates on iPhone screenshot images, trained with MSELoss.

---

## `screencropnet.arch`

_No module docstring._

### `ObjLocModel`

```python
class ObjLocModel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
```

_No class docstring._ Object localization model wrapping an EfficientNet-B0 backbone from `timm`. The backbone's classification head is replaced with a 4-output linear layer for bounding-box regression. (Inferred from implementation.)

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained` | `bool` | `True` | Whether to load ImageNet-pretrained weights from `timm` |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `backbone` | `timm` model | EfficientNet-B0 with `num_classes=4` |

#### `forward`

```python
def forward(
    self,
    images: torch.Tensor,
    gt_bboxes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
```

_No docstring._ Forward pass. If `gt_bboxes` is provided, computes MSELoss against predicted boxes and returns both. Otherwise returns raw logits only. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | `torch.Tensor` | — | Batch of images `(N, C, H, W)` |
| `gt_bboxes` | `torch.Tensor \| None` | `None` | Ground-truth boxes `(N, 4)` in `[xmin, ymin, xmax, ymax]` format |

**Returns:**
- When `gt_bboxes` is not `None`: `tuple[torch.Tensor, torch.Tensor]` — `(bboxes_logits, loss)`
- When `gt_bboxes` is `None`: `torch.Tensor` — `bboxes_logits` of shape `(N, 4)`

Module-level constant `MODEL_NAMES`: sorted list of callable lowercase names from `torchvision.models.__dict__`.

---

## `screencropnet.data_set`

_No module docstring._

### `ObjLocDataset`

```python
class ObjLocDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        root_dir: str = "",
    ) -> None:
```

**Class docstring:** _Localization Dataset._

PyTorch `Dataset` backed by a Pandas DataFrame containing image paths and Pascal VOC bounding-box annotations. Reads images with OpenCV (`BGR → RGB`), applies optional albumentations transforms (which must handle bbox re-labelling), and returns normalised CHW tensors.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | — | DataFrame with columns `img_path`, `xmin`, `ymin`, `xmax`, `ymax` |
| `transform` | callable | `None` | albumentations `Compose` transform; must include `bbox_params` |
| `root_dir` | `str` | `""` | Prefix prepended to every `img_path` value |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Annotation dataframe |
| `transform` | callable | Optional albumentations transform |
| `root_dir` | `str` | Image directory prefix |

#### `__len__`

```python
def __len__(self) -> int:
```

_No docstring._ Returns the number of samples in the dataset. (Inferred from implementation.)

**Returns:** `int`

#### `__getitem__`

```python
def __getitem__(self, idx) -> tuple[TensorCHW, torch.Tensor]:
```

_No docstring._ Returns a `(image_tensor, bbox_tensor)` pair. The image tensor is shape `(3, H, W)` normalised to `[0, 1]`; the bbox tensor is shape `(4,)`. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `idx` | `int` | Sample index |

**Returns:** `tuple[TensorCHW, torch.Tensor]`

---

## `screencropnet.helpers`

_No module docstring._

### `find_intersection`

```python
def find_intersection(set_1, set_2) -> torch.Tensor:
```

Finds the intersection area of every box combination between two sets of boundary-coordinate boxes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `set_1` | `torch.Tensor` shape `(n1, 4)` | First set of boxes in `[xmin, ymin, xmax, ymax]` format |
| `set_2` | `torch.Tensor` shape `(n2, 4)` | Second set of boxes |

**Returns:** `torch.Tensor` shape `(n1, n2)` — Pairwise intersection areas.

---

### `find_jaccard_overlap`

```python
def find_jaccard_overlap(set_1, set_2) -> torch.Tensor:
```

Finds the Jaccard Overlap (IoU) of every box combination between two sets.

| Parameter | Type | Description |
|-----------|------|-------------|
| `set_1` | `torch.Tensor` shape `(n1, 4)` | First set of boxes |
| `set_2` | `torch.Tensor` shape `(n2, 4)` | Second set of boxes |

**Returns:** `torch.Tensor` shape `(n1, n2)` — Pairwise IoU values.

---

### `iou_width_height`

```python
def iou_width_height(boxes1, boxes2) -> torch.Tensor:
```

IoU computed from width-and-height only (no spatial position).

| Parameter | Type | Description |
|-----------|------|-------------|
| `boxes1` | `torch.Tensor` | Width and height of first boxes `(..., 2)` |
| `boxes2` | `torch.Tensor` | Width and height of second boxes `(..., 2)` |

**Returns:** `torch.Tensor` — Element-wise IoU.

---

### `intersection_over_union`

```python
def intersection_over_union(
    boxes_preds,
    boxes_labels,
    box_format: str = "corners",
) -> torch.Tensor:
```

Calculates IoU for batches of predicted and ground-truth boxes. Supports both `"midpoint"` (`x, y, w, h`) and `"corners"` (`x1, y1, x2, y2`) formats.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boxes_preds` | `torch.Tensor` shape `(N, 4)` | — | Predicted bounding boxes |
| `boxes_labels` | `torch.Tensor` shape `(N, 4)` | — | Ground-truth bounding boxes |
| `box_format` | `str` | `"corners"` | `"midpoint"` or `"corners"` |

**Returns:** `torch.Tensor` — Element-wise IoU with a small epsilon `1e-6` added to the denominator.

---

### `non_max_suppression`

```python
def non_max_suppression(
    bboxes: list,
    iou_threshold: float,
    threshold: float,
    box_format: str = "corners",
) -> list:
```

Applies Non-Maximum Suppression to a list of bounding boxes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bboxes` | `list` | — | List of `[class_pred, prob_score, x1, y1, x2, y2]` entries |
| `iou_threshold` | `float` | — | IoU threshold above which a box is suppressed |
| `threshold` | `float` | — | Confidence score threshold; boxes below are discarded before NMS |
| `box_format` | `str` | `"corners"` | `"midpoint"` or `"corners"` |

**Returns:** `list` — Filtered boxes after NMS.

**Raises:** `AssertionError` if `bboxes` is not a `list`.

> A 10-second time limit is enforced; a warning is printed and NMS exits early if exceeded.

---

### `mean_average_precision`

```python
def mean_average_precision(
    pred_boxes: list,
    true_boxes: list,
    iou_threshold: float = 0.5,
    box_format: str = "midpoint",
    num_classes: int = 20,
) -> float:
```

Computes mean Average Precision (mAP) across all classes at a given IoU threshold.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_boxes` | `list` | — | Predictions: each entry `[train_idx, class_pred, prob_score, x1, y1, x2, y2]` |
| `true_boxes` | `list` | — | Ground truths in the same format |
| `iou_threshold` | `float` | `0.5` | IoU threshold for a true-positive |
| `box_format` | `str` | `"midpoint"` | `"midpoint"` or `"corners"` |
| `num_classes` | `int` | `20` | Total number of object classes |

**Returns:** `float` — Mean AP value across all classes with at least one ground-truth box.

---

### `TqdmUpTo`

```python
class TqdmUpTo(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize=None) -> None:
```

_No docstring._ `tqdm` subclass compatible with `urllib.request.urlretrieve` progress callbacks. (Inferred from implementation.)

#### `update_to`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `b` | `int` | `1` | Number of blocks transferred so far |
| `bsize` | `int` | `1` | Block size in bytes |
| `tsize` | `int \| None` | `None` | Total size in bytes; sets `self.total` when provided |

**Returns:** `None`

---

### `download_url`

```python
def download_url(url: str, filepath: str) -> None:
```

_No docstring._ Downloads a file from `url` to `filepath`, creating any missing parent directories. Skips silently if `filepath` already exists. Shows a `tqdm` progress bar during download. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Remote URL |
| `filepath` | `str` | Local destination path |

**Returns:** `None`

---

### `extract_archive`

```python
def extract_archive(filepath: str) -> None:
```

_No docstring._ Unpacks an archive (any format supported by `shutil.unpack_archive`) into the same directory as `filepath`. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Path to the archive file |

**Returns:** `None`

---

## `screencropnet.ml_types`

_No module docstring._

Type aliases defined with `typing.NewType`. Used as documentation markers only — they carry no runtime enforcement.

| Alias | Base type | Semantic meaning |
|-------|-----------|-----------------|
| `ImageNdarrayBGR` | `np.ndarray` | Image array in BGR channel order (OpenCV default) |
| `ImageNdarrayHWC` | `np.ndarray` | Image array in height × width × channels layout |
| `TensorCHW` | `torch.Tensor` | PyTorch tensor in channels × height × width layout |

---

## `screencropnet.devices`

_No module docstring._ Handles device selection with CUDA → MPS → CPU priority. Mirrors `screennet.devices` almost exactly; the only difference is that it imports from `screencropnet.errors`.

Module-level constants (set at import time):

| Name | Value |
|------|-------|
| `cpu` | `torch.device("cpu")` |
| `device` | `None` (set after `get_optimal_device` is called) |
| `dtype` | `torch.float16` |
| `dtype_vae` | `torch.float16` |

The following aliases are also set to `None` at import: `device_interrogate`, `device_gfpgan`, `device_swinir`, `device_esrgan`, `device_scunet`, `device_codeformer`.

`enable_tf32()` is called via `errors.run` at module import time.

### `has_mps`

```python
def has_mps() -> bool:
```

_No docstring._ Returns `True` if MPS (Apple Silicon GPU) is available and a small tensor can be moved to it. Compatible with both stable and nightly PyTorch builds. (Inferred from implementation.)

**Returns:** `bool`

---

### `extract_device_id`

```python
def extract_device_id(args, name) -> str | None:
```

_No docstring._ Scans a list `args` for `name` and returns the element immediately following it. Returns `None` if not found. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `args` | `list` | Argument list (e.g. `sys.argv`) |
| `name` | `str` | Flag name to search for |

**Returns:** `str | None`

---

### `get_optimal_device`

```python
def get_optimal_device(args: argparse.Namespace) -> torch.device:
```

_No docstring._ Returns the best available device: CUDA (respecting `args.gpu`), then MPS, then CPU. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `args` | `argparse.Namespace` | Parsed arguments; uses `args.gpu` (`int | None`) |

**Returns:** `torch.device`

---

### `torch_gc`

```python
def torch_gc() -> None:
```

_No docstring._ Empties the CUDA cache and collects inter-process memory if CUDA is available. No-op on MPS/CPU. (Inferred from implementation.)

**Returns:** `None`

---

### `enable_tf32`

```python
def enable_tf32() -> None:
```

_No docstring._ Enables TF32 precision on CUDA matmul and cuDNN when a CUDA device is available. (Inferred from implementation.)

**Returns:** `None`

---

### `randn`

```python
def randn(seed: int, shape: int) -> torch.Tensor:
```

_No docstring._ Generates a random tensor with a given seed, applying an MPS workaround (generate on CPU then transfer) for PyTorch MPS randomness bug. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | `int` | Manual seed |
| `shape` | `int` | Tensor shape (passed directly to `torch.randn`) |

**Returns:** `torch.Tensor`

---

### `randn_without_seed`

```python
def randn_without_seed(shape: int) -> torch.Tensor:
```

_No docstring._ Like `randn` but without setting a manual seed. Applies the same MPS workaround. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `int` | Tensor shape |

**Returns:** `torch.Tensor`

---

### `autocast`

```python
def autocast(disable: bool = False, precision: str = "autocast") -> contextlib.AbstractContextManager:
```

Returns a context manager for mixed-precision training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable` | `bool` | `False` | If `True`, returns a no-op context manager |
| `precision` | `str` | `"autocast"` | `"full"` forces float32 (no-op); `"autocast"` uses CUDA autocast |

**Returns:** `contextlib.AbstractContextManager` — Either `torch.autocast("cuda")` or `contextlib.nullcontext()`.

---

### `mps_contiguous`

```python
def mps_contiguous(input_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
```

Returns a contiguous tensor, applying `.contiguous()` only when on MPS (workaround for PyTorch MPS memory layout issue).

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_tensor` | `torch.Tensor` | Input tensor |
| `device` | `torch.device` | Current device |

**Returns:** `torch.Tensor`

---

### `mps_contiguous_to`

```python
def mps_contiguous_to(input_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
```

_No docstring._ Calls `mps_contiguous` then moves the result to `device`. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_tensor` | `torch.Tensor` | Input tensor |
| `device` | `torch.device` | Target device |

**Returns:** `torch.Tensor`

---

### `mps_check`

```python
def mps_check() -> None:
```

_No docstring._ Diagnostic function: checks MPS availability and prints status messages. Also exercises the MPS device with a small tensor operation if available. (Inferred from implementation.)

**Returns:** `None`

---

### `seed_everything`

```python
def seed_everything(seed: int) -> None:
```

_No docstring._ Sets random seeds for `random`, `numpy`, `torch` (CPU and CUDA) and enables cuDNN deterministic mode. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | `int` | Seed value |

**Returns:** `None`

---

## `screencropnet.errors`

_No module docstring._

### `run`

```python
def run(code, task: str) -> None:
```

_No docstring._ Calls `code()` and silently catches any exception, printing the exception type and full traceback to stderr. Used to guard side-effectful initialisation (e.g. `enable_tf32`) at module import time. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `code` | `callable` | Zero-argument callable to execute |
| `task` | `str` | Label printed before the exception type on stderr |

**Returns:** `None`

---

## `screencropnet.image_utils`

_No module docstring._

### `opencv_read_and_convert_image`

```python
def opencv_read_and_convert_image(path: str, cvt=cv2.COLOR_BGR2RGB) -> np.ndarray:
```

_No docstring._ Reads an image from disk with OpenCV and converts its color space. (Inferred from implementation.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | File path |
| `cvt` | `int` | `cv2.COLOR_BGR2RGB` | OpenCV color conversion code |

**Returns:** `np.ndarray` — Converted image array.

---

### `convert_image_numpy_array_to_tensor`

```python
def convert_image_numpy_array_to_tensor(img: np.ndarray) -> torch.Tensor:
```

_No docstring._ Converts an HWC numpy array to a CHW PyTorch tensor normalised to `[0, 1]`. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `img` | `np.ndarray` | Image in HWC layout, pixel values `[0, 255]` |

**Returns:** `torch.Tensor` — Shape `(C, H, W)`, values `[0.0, 1.0]`.

---

### `safe_read_image`

```python
def safe_read_image(path: str) -> torch.Tensor:
```

_No docstring._ Convenience wrapper: reads an image with `opencv_read_and_convert_image` and converts it to a normalised CHW tensor. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | File path |

**Returns:** `torch.Tensor` — Shape `(C, H, W)`, values `[0.0, 1.0]`.

---

### `load_and_transform_image_for_prediction`

```python
def load_and_transform_image_for_prediction(
    path: str,
    transform: A.Compose | NoneType = None,
    img_size: int = 140,
) -> tuple[torch.Tensor, A.Compose]:
```

_No docstring._ Reads an image and builds a resize-only albumentations transform for inference. Note: when `transform` is `None` the returned `img_transform` is still the resize-only `A.Compose`, not `None`. (Inferred from implementation; behavior when `transform` is not `None` is arguably a bug — the parameter is checked but the passed transform is ignored.)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Image file path |
| `transform` | `A.Compose \| None` | `None` | Currently unused (see note) |
| `img_size` | `int` | `140` | Square resize target |

**Returns:** `tuple[torch.Tensor, A.Compose]` — `(img_tensor, img_transform)`

---

## `screencropnet.pascal_to_csv`

_No module docstring._

> **Note:** This script calls `main()` unconditionally at the module level (`main()` is the last statement). It is not safe to import as a library — doing so will attempt to read Pascal VOC XML files from the hard-coded path `/Users/malcolm/Downloads/pascal_temp/annotations`.

### `xml_to_csv`

```python
def xml_to_csv(path: str) -> pd.DataFrame:
```

_No docstring._ Parses all Pascal VOC XML annotation files in `path` and returns a consolidated DataFrame. (Inferred from implementation.)

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Directory containing `.xml` annotation files |

**Returns:** `pd.DataFrame` with columns `img_path`, `xmin`, `ymin`, `xmax`, `ymax`, `width`, `height`, `label`.

---

### `main`

```python
def main() -> None:
```

_No docstring._ Converts Pascal VOC XML annotations for the `pascal_temp` split to a CSV file `labels2_pascal_temp.csv`. Hard-coded base path: `/Users/malcolm/Downloads`. Called unconditionally at module level. (Inferred from implementation.)

**Returns:** `None`
