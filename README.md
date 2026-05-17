# pytorch-lab

A lab to experiment with Daniel Bourke's pytorch class https://github.com/mrdbourke/pytorch-deep-learning

Two production-style model subprojects share a common set of training utilities:

| Subproject | Task | Model |
|---|---|---|
| `screennet/` | Image classification (twitter / facebook / tiktok) | EfficientNet-B0 |
| `screencropnet/` | Screenshot localization (bounding-box regression) | EfficientNet-B0 backbone |

> `screencropnet`'s IoU / NMS / mAP evaluation helpers are adapted from the
> Kaggle [*YOLOv3 for Pascal VOC*](https://www.kaggle.com/code/dqhdqmcttdqx/yolov3-for-pascal-voc/notebook)
> notebook; the model itself is single-box bounding-box regression, not a full
> YOLO network.

---

## Environment setup

```bash
uv sync          # create / update .venv from uv.lock
make env-works   # smoke-test: verify MPS and matplotlib work
```

---

## Data setup

### Classification dataset (screennet)

```bash
make download-dataset   # download twitter/facebook/tiktok zip from Dropbox
make unzip-dataset      # extract into scratch/datasets/
```

### Localization dataset + assets (screencropnet)

```bash
make fetch-assets                    # dataset + all checkpoints + sample image (recommended)
make download-localization-dataset   # dataset only
```

To download just the pretrained checkpoint instead of the full asset bundle:

```bash
cd screencropnet
make get-best-model   # downloads ScreenCropNetV1_378_epochs.pth into models/
```

---

## Training

### screennet — image classification

```bash
cd screennet
make train   # train EfficientNet-B0 for 10 epochs
```

### screencropnet — bounding-box regression

```bash
cd screencropnet
make train             # train ObjLocModel for 40 epochs
make train_100         # train for 378 epochs (full run)
make transfer_learning # resume training from an existing checkpoint
```

Monitor training in TensorBoard (either subproject):

```bash
make tensorboard   # launch TensorBoard at runs/
```

---

## Prediction / inference

### screennet

```bash
cd screennet
make predict       # classify a single image and save result to disk
make worst-first   # rank every test-set image by highest loss → worst-first.csv
```

### screencropnet

```bash
cd screencropnet
make predict         # predict bounding box for a single image and save result
make autocrop        # crop a screenshot using the predicted bbox
make autocrop-resize # crop + resize to a target dark-mode resolution
make evaluate        # evaluate the model on the test set (runs 5 epochs with --test)
```

---

## Quality

Run from the repo root:

```bash
make test          # run pytest
make test-cov      # pytest + terminal coverage report
make test-cov-html # pytest + generate htmlcov/ HTML report
make open-cov      # open htmlcov/index.html in the browser
make lint          # ruff check
make format        # ruff format
make typecheck     # pyright
make check         # lint + typecheck + test
```

---

## Useful utilities

```bash
make clean     # remove .pyc / __pycache__ / .coverage / htmlcov/
make jupyter   # start Jupyter notebook server
make ipython   # launch IPython REPL
make help      # print all available make targets with descriptions
```
