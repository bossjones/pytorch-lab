# API Reference — pytorch-lab

Python module, function, and class reference for the pytorch-lab codebase.

This is **not** an HTTP/OpenAPI reference — the project has no web endpoints. Everything documented here is Python-callable.

---

## Index

| File | Description |
|------|-------------|
| [going_modular.md](./going_modular.md) | Reusable training primitives: DataLoader creation, train/test loops (classification + localization), TinyVGG model, model save/load, and prediction utilities. |
| [helper_functions.md](./helper_functions.md) | Standalone plotting and utility helpers shared across notebooks and both sub-projects: decision boundary plots, accuracy calculation, data download, and seed management. |
| [screencropnet.md](./screencropnet.md) | Bounding-box regression sub-project: `ObjLocModel` (EfficientNet-B0 backbone), `ObjLocDataset`, IoU / NMS / mAP helpers, device selection, image I/O utilities, and Pascal VOC CSV converter. |
| [screennet.md](./screennet.md) | Image classification sub-project: EfficientNet-B0 fine-tuning for twitter / facebook / tiktok screenshots, CLI entry point (`main`), training orchestration, prediction helpers, confusion matrix tools, and checkpoint management. |
| [tasks.md](./tasks.md) | Invoke task collections for local development (`local.*`) and CI quality checks (`ci.*`), plus supporting utilities, constants, a Loguru-based logger, and colored log symbols. |
| [contrib.md](./contrib.md) | Standalone PEP 723 scripts: idempotent asset fetcher (`fetch_screencropnet_assets.py`) that downloads the localization dataset, model checkpoints, and sample images from Dropbox. |

---

## Quick navigation by concern

### Training a classification model
- Data loading: [`going_modular.data_setup.create_dataloaders`](./going_modular.md#create_dataloaders)
- Train loop: [`going_modular.engine.train`](./going_modular.md#train)
- Single-epoch steps: [`going_modular.engine.train_step`](./going_modular.md#train_step), [`going_modular.engine.test_step`](./going_modular.md#test_step)
- Model architecture: [`going_modular.model_builder.TinyVGG`](./going_modular.md#tinyvgg)
- Saving weights: [`going_modular.utils.save_model`](./going_modular.md#save_model)

### Training a localization model
- Dataset: [`screencropnet.data_set.ObjLocDataset`](./screencropnet.md#objlocdataset)
- Model: [`screencropnet.arch.ObjLocModel`](./screencropnet.md#objlocmodel)
- Train loop: [`going_modular.engine.train_localization`](./going_modular.md#train_localization)

### Running inference
- Classification: [`screennet.main.pred_and_store`](./screennet.md#pred_and_store), [`screennet.main.pred_and_plot_image`](./screennet.md#pred_and_plot_image)
- General plotting helper: [`going_modular.predictions.pred_and_plot_image`](./going_modular.md#pred_and_plot_image), [`helper_functions.pred_and_plot_image`](./helper_functions.md#pred_and_plot_image)

### Device management
- screencropnet: [`screencropnet.devices.get_optimal_device`](./screencropnet.md#get_optimal_device)
- screennet: [`screennet.devices.get_optimal_device`](./screennet.md#screennetdevices) (identical API)

### Object detection utilities
- IoU: [`screencropnet.helpers.intersection_over_union`](./screencropnet.md#intersection_over_union), [`screencropnet.helpers.find_jaccard_overlap`](./screencropnet.md#find_jaccard_overlap)
- NMS: [`screencropnet.helpers.non_max_suppression`](./screencropnet.md#non_max_suppression)
- mAP: [`screencropnet.helpers.mean_average_precision`](./screencropnet.md#mean_average_precision)

### Developer operations
- Environment sync: [`tasks.local.sync`](./tasks.md#sync)
- Full quality gate: [`tasks.ci.check`](./tasks.md#check)
- Fetch model assets: [`contrib/fetch_screencropnet_assets.py`](./contrib.md)

---

## Conventions used in this reference

- **_No docstring._** — the function or class has no Python docstring in the source.
- **(Inferred from implementation.)** — behavior described from reading the code, not from a docstring; treat with appropriate caution.
- Type annotations are taken directly from the source; where absent, types are inferred and noted.
- Default values are exactly as written in the source code.
