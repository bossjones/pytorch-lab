# Label Studio Labeling Configs — Reference

Source: <https://labelstud.io/guide/setup>, <https://labelstud.io/templates>,
<https://labelstud.io/guide/tasks>, <https://labelstud.io/guide/manage_data>

The labeling interface for a project is an **XML labeling config**. It uses
three kinds of tags:

- **Object tags** — declare the data shown (`Image`, `Text`, ...). The
  `value="$key"` attribute binds to a key in each task's `data` dict.
- **Control tags** — declare what you annotate (`RectangleLabels`, `Choices`,
  ...). The `toName` attribute links a control to an object tag's `name`.
- **Visual tags** — layout only (`View`, `Header`, ...).

Edit it under Project → Settings → **Labeling Interface** (Code view).

## Bounding boxes (object detection / YOLO)

Template: <https://labelstud.io/templates/image_bbox>

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Content" background="green"/>
  </RectangleLabels>
</View>
```

- `Image value="$image"` — each task's `data.image` holds the image URL/path.
- `RectangleLabels` — the box-drawing control; `toName="image"` ties it to the
  `Image` tag.
- `Label` — one per class. Add more `<Label>` lines for multi-class detection
  (YOLO training); a single label is enough for screencropnet's one-box task.

## Image classification

Template: <https://labelstud.io/templates/image_classification>

```xml
<View>
  <Image name="image" value="$image"/>
  <Choices name="choice" toName="image">
    <Choice value="twitter"/>
    <Choice value="facebook"/>
    <Choice value="tiktok"/>
  </Choices>
</View>
```

Add `choice="multiple"` to the `Choices` tag to allow more than one label per
image (the default is single-select). The three choices above match
`screennet`'s classes.

## Importing data

See <https://labelstud.io/guide/tasks>. A task is JSON whose `data` keys map to
the `$value` references in the config:

```json
[
  {"data": {"image": "https://example.com/00000_twitter.PNG"}},
  {"data": {"image": "/data/local-files/?d=twitter_screenshots_localization_dataset/00001_twitter.PNG"}}
]
```

Import routes:

- **UI** — Data Manager → **Import**: drag files, paste a list of URLs, or
  upload a CSV/JSON of tasks.
- **Local files** — with local file serving on (see `setup.md`), add a Local
  files source storage; Label Studio generates `/data/local-files/?d=...`
  references automatically.
- **Cloud storage / API** — S3/GCS/Azure source storage, or the import API
  endpoint.

Supported image types: `.bmp .gif .jpg .png .svg .webp`. Keep roughly 100k or
fewer tasks per project for good performance.

## Data Manager

See <https://labelstud.io/guide/manage_data>. The Data Manager lists every
task:

- **Order** dropdown — sort by created date, ID, prediction score, etc.;
  toggle ascending/descending.
- **Filters** — narrow by annotation status, ID range, agreement, and more.
- **Tabs** — save filtered/sorted views to split work across annotators.
- **Label / Label Tasks As Displayed** — open tasks in the current view order.

## Related

- `setup.md` — install, run, local file serving, project creation.
- `screencropnet-workflow.md` — export the annotations to screencropnet's CSV.
