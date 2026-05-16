# ScreenCropNet Asset Provenance — Reference

Source: recovered from the **un-committed** reference notebooks
`contrib/ScreenCropNet Predict Test.ipynb` and
`contrib/ScreenCropNet Training.ipynb` (intentionally not tracked — see
`.gitignore`). This file is the durable record so the knowledge survives the
notebooks staying local.

Fetch everything with:

```bash
make fetch-assets                    # all assets
make download-localization-dataset   # dataset only
uv run contrib/fetch_screencropnet_assets.py --weights --sample
```

The fetcher (`contrib/fetch_screencropnet_assets.py`) is **idempotent**: any
asset whose target already exists is skipped (use `--force` to re-download).
Destinations under `scratch/` are git-ignored.

## Assets

| Key | Source URL | Destination (repo-root-relative) | Used by |
|---|---|---|---|
| `dataset` | `https://www.dropbox.com/s/w5rzn8b1s0p9d2n/twitter_screenshots_localization_dataset.zip?dl=1` | `scratch/datasets/twitter_screenshots_localization_dataset/` | screencropnet train/eval |
| `weights` (canonical) | `https://www.dropbox.com/s/9903r4jy02rmuzh/ScreenCropNetV1_378_epochs.pth?dl=1` | `screencropnet/models/ScreenCropNetV1_378_epochs.pth` | screencropnet predict/evaluate |
| `weights-collab` | `https://www.dropbox.com/s/m5mvmlv28x7xdty/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth?dl=1` | `screencropnet/models/collab_ScreenCropNetV1_ObjLocModelV1_basic_40_epochs.pth` | transfer-learning |
| `weights-best` | `https://www.dropbox.com/s/rzkwy02hz2j3ath/screencropnet_best_model.pt?dl=1` | `screencropnet/models/screencropnet_best_model.pt` | reference |
| `sample` | `https://www.dropbox.com/s/livf8f0dwd6wnlr/IMG_6324.PNG?dl=1` | `scratch/IMG_6324.PNG` | predict smoke |

The canonical `weights` entry matches the existing
`screencropnet/Makefile` `get-best-model` target. `make fetch-assets`
(`--all`) pulls the dataset, all three checkpoints, and the sample image.

## Dataset layout

After unzip, the images sit **one directory deeper**:

```
scratch/datasets/twitter_screenshots_localization_dataset/
  twitter_screenshots_localization_dataset/   # images
  labels_pascal_temp.csv                       # Pascal VOC labels
```

`labels_pascal_temp.csv` columns: `img_path,xmin,ymin,xmax,ymax`.

## Related

- Classification dataset (`twitter_facebook_tiktok.zip`) is a separate,
  already-scripted asset — see root `Makefile` `download-dataset`. Left
  unchanged.
