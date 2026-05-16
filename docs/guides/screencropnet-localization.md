# ScreenCropNet: Bounding-Box Localization

ScreenCropNet trains a regression model that predicts a bounding box around the
"content area" of a screenshot. The model uses an EfficientNet-B0 backbone (from
`timm`) and is trained with MSELoss on Pascal VOC format annotations.

**What you will be able to do after this guide:**

- Fetch the localization dataset and pre-trained weights
- Train the model from scratch (40 or 378 epochs)
- Resume training from a checkpoint (transfer learning)
- Evaluate the model on the test split
- Run inference on a single image and save a bounding-box overlay
- Autocrop a screenshot to the predicted content region

**Prerequisites:** Complete [Getting Started](getting-started.md) first. You need a
working `uv` environment and `PYTORCH_ENABLE_MPS_FALLBACK=1` set.

**Time estimate:** Asset download ~5 min; 40-epoch training ~20-40 min on M-series Mac.

---

## 1. Fetch all assets

The localization dataset, model weights, and a sample image are hosted on Dropbox. Use
the idempotent fetcher from the **repo root**:

```bash
make fetch-assets
```

This runs `uv run contrib/fetch_screencropnet_assets.py --all` and downloads:

| Asset | Destination |
|---|---|
| Localization dataset (zip, ~300 MB) | `scratch/datasets/twitter_screenshots_localization_dataset/` |
| Canonical weights (378 epochs) | `screencropnet/models/ScreenCropNetV1_378_epochs.pth` |
| Collab weights (40 epochs) | `screencropnet/models/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth` |
| Best reference checkpoint | `screencropnet/models/screencropnet_best_model.pt` |
| Sample image | `scratch/IMG_6324.PNG` |

> **Note:** The fetcher is idempotent. Any asset whose destination already exists is
> skipped. Use `--force` to re-download:
> `uv run contrib/fetch_screencropnet_assets.py --all --force`

To download only the dataset (without weights):

```bash
make download-localization-dataset
```

To download only weights and the sample image:

```bash
uv run contrib/fetch_screencropnet_assets.py --weights --sample
```

---

## Dataset layout

After `make fetch-assets` completes, the dataset is structured as:

```
scratch/datasets/twitter_screenshots_localization_dataset/
└── twitter_screenshots_localization_dataset/   # JPEG/PNG images
    labels_pascal_temp.csv                       # Pascal VOC annotations
```

`labels_pascal_temp.csv` columns: `img_path,xmin,ymin,xmax,ymax`.

---

## 2. Alternatively — download only the weights

If you just want to run inference or evaluate without training, the `screencropnet/`
Makefile also provides a direct weight download:

```bash
cd screencropnet
make get-best-model
```

This downloads `ScreenCropNetV1_378_epochs.pth` directly to `screencropnet/models/`.

---

## 3. Train from scratch (40 epochs)

Switch to the `screencropnet/` subproject:

```bash
cd screencropnet
make train
```

This runs:

```bash
uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 40
```

What each flag does:

| Flag | Effect |
|---|---|
| `--seed 42` | Fix random seed for reproducibility |
| `--epochs 40` | Train for 40 epochs |
| `--to-disk` | Save checkpoint and plots to `models/` |
| `--debug` | Enable extra logging |
| `--interactive` | Set matplotlib to interactive mode |

Training uses MSELoss on the four bounding-box coordinates
(`xmin, ymin, xmax, ymax`). Device priority is CUDA > MPS > CPU.

Expected last-epoch output (values approximate):

```
Epoch 40/40 | train_loss: 0.0042 | test_loss: 0.0051
Saved model to: models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth
```

---

## 4. Train for 378 epochs (full run)

For maximum accuracy, train for 378 epochs. This takes significantly longer:

```bash
cd screencropnet
make train_100
```

This runs:

```bash
uv run python main.py --seed 42 --to-disk --debug --interactive --epochs 378
```

> **Warning:** On an M-series Mac without CUDA, this run can take several hours. Use
> the 40-epoch checkpoint for experimentation and reserve the 378-epoch run for final
> model production.

---

## 5. Resume training from a checkpoint (transfer learning)

To continue training from the 40-epoch checkpoint:

```bash
cd screencropnet
make transfer_learning
```

This runs:

```bash
uv run python main.py \
  --seed 42 \
  --to-disk \
  --debug \
  --interactive \
  --epochs 40 \
  --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth
```

The `--weights` flag loads the saved checkpoint before training begins. This is useful
for:

- Resuming an interrupted run
- Fine-tuning with a different learning rate
- Starting from the pre-trained Dropbox weights and adapting to new data

To run a single epoch from a checkpoint (quick sanity check):

```bash
cd screencropnet
uv run python main.py \
  --seed 42 \
  --debug \
  --epochs 1 \
  --weights ./models/ScreenCropNetV1_ObjLocModelV1_pascalVOC_40_epochs.pth
```

---

## 6. Evaluate the model

Run the model over the test split and compute metrics:

```bash
cd screencropnet
make evaluate
```

This runs:

```bash
uv run python main.py \
  --to-disk \
  --debug \
  --test \
  --epochs 5 \
  --weights ./models/ScreenCropNetV1_378_epochs.pth
```

The `--test` flag activates the test-set evaluation path. Results are written to disk
when `--to-disk` is set.

---

## 7. View training curves in TensorBoard

Open a second terminal while training is in progress (or after it completes):

```bash
cd screencropnet
make tensorboard
```

This runs:

```bash
uv run tensorboard --logdir runs/
```

Open `http://localhost:6006` in your browser. Refresh to pick up new events.

---

## 8. Predict — bounding-box overlay on a single image

Run inference on one image and save an overlay image showing the predicted box:

```bash
cd screencropnet
make predict
```

This runs:

```bash
uv run python main.py \
  --predict "~/Downloads/IMG_6400.PNG" \
  --weights ./models/ScreenCropNetV1_378_epochs.pth \
  --to-disk
```

To predict on any image of your choice:

```bash
cd screencropnet
uv run python main.py \
  --predict /path/to/your/screenshot.png \
  --weights ./models/ScreenCropNetV1_378_epochs.pth \
  --to-disk
```

The `--to-disk` flag saves the overlay image alongside the input.

---

## 9. Autocrop — crop to the predicted content region

Autocrop uses the predicted bounding box to crop and return just the content area:

```bash
cd screencropnet
make autocrop
```

This runs:

```bash
uv run python main.py \
  --autocrop "~/Downloads/2021-10-20_12-44-46_000.png" \
  --weights ./models/ScreenCropNetV1_378_epochs.pth \
  --to-disk
```

To autocrop with a dark-mode resize applied first:

```bash
cd screencropnet
uv run python main.py \
  --resize "darkmode" \
  --autocrop "~/Downloads/2021-10-20_12-44-46_000.png" \
  --weights ./models/ScreenCropNetV1_378_epochs.pth \
  --to-disk
```

---

## 10. Code quality for screencropnet

From inside `screencropnet/`:

```bash
make format      # uv run ruff format .
make lint        # uv run ruff check .
make typecheck   # uv run pyright main.py
```

---

## Key CLI flags reference

All flags are passed to `screencropnet/main.py`:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--epochs N` | int | 5 | Number of training or evaluation epochs |
| `--batch-size N` | int | 32 | Mini-batch size |
| `--lr RATE` | float | 0.001 | Initial learning rate |
| `--seed N` | int | — | Random seed for reproducibility |
| `--weights PATH` | str | — | Load saved `.pth` or `.pt` checkpoint |
| `--predict PATH` | str | — | Image path for inference with bbox overlay |
| `--autocrop PATH` | str | — | Image path to crop to predicted bbox |
| `--resize MODE` | str | — | Pre-processing resize mode (e.g. `darkmode`) |
| `--test` | flag | off | Evaluate on the test split |
| `--to-disk` | flag | off | Save checkpoints and output images to disk |
| `--debug` | flag | off | Enable extra logging |
| `--interactive` | flag | off | Enable matplotlib interactive mode |
| `--summary` | flag | off | Print model architecture summary |

---

## Next steps

- Train the classifier counterpart: [screennet-classification.md](screennet-classification.md)
- Add a dependency or run the full quality suite: [development-workflow.md](development-workflow.md)
- Something not working? See [troubleshooting.md](troubleshooting.md)
