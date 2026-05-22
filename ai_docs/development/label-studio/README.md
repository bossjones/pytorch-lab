# Label Studio Docs — ScreenCropNet Annotation

Source: <https://labelstud.io/guide/>

Label Studio is the annotation UI for producing `screencropnet` (bounding-box
regression) and `screennet` (classification) training data. It is installed as
an isolated `uv` tool — its pinned `requests`/`pillow` versions conflict with
the project venv — and launched with `make label-studio`.

## Contents

| Doc | What it covers |
|---|---|
| [setup.md](setup.md) | Install rationale, run commands, local file serving, creating a project |
| [labeling-configs.md](labeling-configs.md) | Labeling-interface XML — `RectangleLabels` for bounding boxes/YOLO, `Choices` for classification — plus data import and the Data Manager |
| [screencropnet-workflow.md](screencropnet-workflow.md) | End-to-end: annotate twitter screenshots → export Pascal VOC XML → convert to the `img_path,xmin,ymin,xmax,ymax` CSV that `ObjLocDataset` reads |

## Start here

1. Read [setup.md](setup.md), then run:

   ```bash
   make label-studio-local        # UI on http://localhost:8080, serves scratch/datasets/
   ```

2. Building screencropnet bounding-box labels? Follow
   [screencropnet-workflow.md](screencropnet-workflow.md) start to finish.
3. Need a different task (multi-class detection, classification)? See the XML
   templates in [labeling-configs.md](labeling-configs.md).

## Related

- `ai_docs/screencropnet-assets.md` — dataset layout and label CSV column spec.
