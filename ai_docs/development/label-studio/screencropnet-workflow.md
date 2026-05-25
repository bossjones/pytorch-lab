# Label Studio → ScreenCropNet Workflow — Reference

Source: <https://labelstud.io/guide/export>, <https://labelstud.io/guide/tasks>

End-to-end recipe for producing `screencropnet` training data: draw one
bounding box around the relevant content region of each twitter screenshot,
then convert the export into the Pascal VOC CSV that `ObjLocDataset`
(`screencropnet/data_set.py`) reads.

## 1. Project setup

1. Run `make label-studio-local` (serves `scratch/datasets/` as local files).
2. Create a project `screencropnet-bbox`.
3. Labeling Interface — paste the bounding-box config (one `Content` label):

   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="Content" background="green"/>
     </RectangleLabels>
   </View>
   ```

4. Add a **Local files** source storage pointing at
   `twitter_screenshots_localization_dataset/` inside the document root, then
   **Sync** to import the `.PNG` screenshots.

## 2. Annotate

For each screenshot draw **one** rectangle around the content you want
screencropnet to crop to (the tweet body). `ObjLocModel` is a single-box
regressor — exactly one box per image. Use the Data Manager (`Order` by ID) to
label in a stable order.

## 3. Export — coordinate systems matter

See <https://labelstud.io/guide/export>. Project → **Export** → pick a format:

| Format | Coordinates | Notes |
|---|---|---|
| JSON / JSON-MIN | **percent** `x,y,w,h` of image size | corner-anchored |
| CSV | **percent** | flat columns per `from_name`/`to_name` |
| COCO | **pixels** `[x,y,w,h]` | corner + size |
| **Pascal VOC XML** | **pixels** `xmin,ymin,xmax,ymax` | one XML per image |
| YOLO | **normalized [0,1]** `x_c,y_c,w,h` | center-anchored, `.txt` per image |

screencropnet's CSV needs **pixel corner** coords (`xmin,ymin,xmax,ymax`), so
**Pascal VOC XML** is the right export — no coordinate math needed.

## 4. Convert Pascal VOC XML → screencropnet CSV

`screencropnet/pascal_to_csv.py` already does this conversion. It globs `*.xml`
in a directory and writes a CSV with columns
`img_path,xmin,ymin,xmax,ymax,width,height,label` — the format `ObjLocDataset`
consumes.

Caveats before running it:

- **Hardcoded path.** `pascal_to_csv.py:54` reads from
  `/Users/malcolm/Downloads/pascal_temp/annotations` and writes
  `labels2_pascal_temp.csv`. Edit `image_path` (and the `datasets` list) to
  point at your unzipped Pascal VOC export. Parameterizing this via `argparse`
  is a tracked follow-up.
- **`img_path` rewriting.** The XML `<filename>` becomes the CSV `img_path`.
  `ObjLocDataset.__getitem__` builds the path as `root_dir + row.img_path`, so
  `img_path` must be relative to the `root_dir` passed to the dataset (e.g.
  `train_images/00000_twitter.PNG`). If Label Studio wrote bare filenames or
  `/data/local-files/?d=...` URLs, fix the column after conversion.

```bash
# after editing the path in pascal_to_csv.py
cd screencropnet && uv run python pascal_to_csv.py
```

Verify the result has the expected header and pixel-corner rows:

```text
img_path,xmin,ymin,xmax,ymax,width,height,label
train_images/00000_twitter.PNG,30,391,1161,752,1179,2556,twitter
```

`ObjLocDataset` only requires `img_path,xmin,ymin,xmax,ymax`; the extra
`width,height,label` columns are harmless.

## YOLO alternative

For YOLO training (`screencropnet/helpers.py` carries YOLO-derived eval utils),
export **YOLO** instead: Label Studio writes one `.txt` per image with
normalized `class x_center y_center width height` lines plus a `classes.txt`.
That output is consumed directly by YOLO trainers — no conversion script
needed. Use the Pascal VOC → CSV path only for the `ObjLocModel` regression
model.

## Related

- `setup.md` — install, run, local file serving.
- `labeling-configs.md` — the labeling-interface XML.
- `ai_docs/screencropnet-assets.md` — dataset layout and the
  `labels_pascal_temp.csv` column spec.
