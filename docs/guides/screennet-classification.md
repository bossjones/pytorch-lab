# ScreenNet: Image Classification

ScreenNet classifies screenshots into three classes — **twitter**, **facebook**, and
**tiktok** — using an EfficientNet-B0 backbone fine-tuned on a custom dataset.

**What you will be able to do after this guide:**

- Fetch the classification dataset
- Train the model for 10 epochs
- View live training curves in TensorBoard
- Run inference on a single image
- Rank the test set by prediction error (worst-first)

**Prerequisites:** Complete [Getting Started](getting-started.md) first. You need a
working `uv` environment and `PYTORCH_ENABLE_MPS_FALLBACK=1` set.

**Time estimate:** Dataset download ~2 min; training ~5-15 min on M-series Mac.

---

## 1. Fetch the classification dataset

All commands in this guide are run from the **repo root** unless stated otherwise.

```bash
make download-dataset
```

This downloads `twitter_facebook_tiktok.zip` from Dropbox into
`scratch/datasets/twitter_facebook_tiktok.zip` and lists its contents. Then unzip it:

```bash
make unzip-dataset
```

This extracts the archive to `scratch/datasets/` and removes the zip file. After
extraction the layout is:

```
scratch/datasets/twitter_facebook_tiktok/
├── train/
│   ├── facebook/
│   ├── tiktok/
│   └── twitter/
└── test/
    ├── facebook/
    ├── tiktok/
    └── twitter/
```

> **Note:** `scratch/` is gitignored. Datasets are never committed to the repository.

---

## 2. Train the model

Switch to the `screennet/` subproject and run training:

```bash
cd screennet
make train
```

This runs:

```bash
uv run python main.py --to-disk --debug --interactive --epochs 10
```

What each flag does:

| Flag | Effect |
|---|---|
| `--epochs 10` | Train for 10 epochs |
| `--to-disk` | Save the trained model checkpoint and plots to disk |
| `--debug` | Enable extra logging during training |
| `--interactive` | Set matplotlib to interactive mode |

The model is EfficientNet-B0 with pretrained ImageNet weights, fine-tuned on the
three-class dataset. Training uses the device priority CUDA > MPS > CPU.

Expected console output (last few lines):

```
Epoch 10/10 | train_loss: 0.21 | train_acc: 0.93 | test_loss: 0.19 | test_acc: 0.94
Training complete. Time: 312.4s
Saving model to: models/ScreenNet_efficientnet_b0_basic_10_epochs.pth
```

The checkpoint is written to `screennet/models/` and TensorBoard event files to
`screennet/runs/`.

> **Warning:** Do not interrupt training with Ctrl+C unless you want to discard the
> current epoch. The model is only saved at the end of training.

---

## 3. View training curves in TensorBoard

While training is running (or after it completes), open a second terminal and:

```bash
cd screennet
make tensorboard
```

This runs:

```bash
uv run tensorboard --logdir runs/
```

Open `http://localhost:6006` in your browser to see loss and accuracy curves for each
epoch. Refresh the page to pick up new data while training is in progress.

---

## 4. Run prediction on a single image

Once you have a trained checkpoint, classify any image:

```bash
cd screennet
make predict
```

This runs inference on a hardcoded demo path:

```bash
uv run python main.py \
  --predict "~/Downloads/2020-11-25_10-47-32_867.jpeg" \
  --weights ./models/ScreenNetV1.pth \
  --interactive \
  --to-disk
```

To classify your own image, call `main.py` directly with the path you want:

```bash
cd screennet
uv run python main.py \
  --predict /path/to/your/screenshot.png \
  --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth \
  --to-disk
```

Replace the `--weights` path with whatever checkpoint was saved in step 2.

Expected output:

```
Predicting on: /path/to/your/screenshot.png
Predicted class: twitter  (confidence: 0.98)
```

---

## 5. Rank the test set by prediction error (worst-first)

To find which test images the model gets most wrong:

```bash
cd screennet
make worst-first
```

This runs:

```bash
uv run python main.py \
  --to-disk \
  --debug \
  --predict ~/dev/bossjones/pytorch-lab/scratch/datasets/twitter_facebook_tiktok/test \
  --weights ./models/ScreenNet_efficientnet_b0_basic_10_epochs.pth \
  --worst-first \
  --results ./worst-first.csv
```

> **Note:** The `--predict` flag accepts a directory as well as a single file. When
> passed a directory, the model runs inference on every image inside it.

The `--worst-first` flag ranks results by loss descending. Results are written to
`screennet/worst-first.csv` with columns for image path, predicted class, true class,
and loss. This is useful for understanding failure modes and deciding which images need
better augmentation or labeling.

---

## 6. Code quality for screennet

From inside `screennet/`:

```bash
make format      # uv run ruff format .
make lint        # uv run ruff check .
make typecheck   # uv run pyright main.py
```

These use the configuration in the root `pyproject.toml` (`[tool.ruff]` and
`[tool.pyright]`). Line length is 88.

---

## Key CLI flags reference

All flags are passed directly to `screennet/main.py`:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--epochs N` | int | 5 | Number of training epochs |
| `--batch-size N` | int | 32 | Mini-batch size |
| `--lr RATE` | float | 0.001 | Initial learning rate |
| `--weights PATH` | str | — | Path to a saved `.pth` checkpoint to load |
| `--predict PATH` | str | — | Image or directory to run inference on |
| `--worst-first` | flag | off | Sort results CSV by highest loss first |
| `--results PATH` | str | — | Write prediction results to this CSV path |
| `--to-disk` | flag | off | Save model checkpoints and plots to disk |
| `--debug` | flag | off | Enable extra logging |
| `--interactive` | flag | off | Enable matplotlib interactive mode |
| `--summary` | flag | off | Print model architecture summary |

---

## Next steps

- Explore bounding-box regression with [screencropnet-localization.md](screencropnet-localization.md)
- Add a dependency or run the full quality suite: [development-workflow.md](development-workflow.md)
