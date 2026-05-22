# Label Studio Setup — Reference

Source: <https://labelstud.io/guide/install.html>,
<https://labelstud.io/guide/quick_start>

Label Studio is the annotation UI used to draw bounding boxes and assign
classification labels for the `screencropnet` / `screennet` datasets.

## Why it is an isolated `uv` tool, not a project dependency

Label Studio is a Django app that pins old `requests` (`<2.33`) and `pillow`
(`9.0.0`) releases. Those conflict with this project's `requests>=2.34.2` and
`pillow>=12.2.0` pins, so `uv add label-studio` **cannot** resolve and it
cannot live in the project venv. It is installed in its own isolated
environment instead:

```bash
uv tool install label-studio      # one-time, isolated venv
uv tool list                      # shows: label-studio v1.x.x
```

`uvx` is shorthand for `uv tool run` — it runs the tool from that isolated
environment. The Makefile targets below use `uvx label-studio`.

## Running it

| Command | What it does |
|---|---|
| `make label-studio` | annotation UI on <http://localhost:8080> |
| `make label-studio-local` | same, serving `scratch/datasets/` as local files |
| `cd screencropnet && make label-studio` | same UI, from the subproject |
| `cd screencropnet && make label-studio-local` | serves `../scratch/datasets/` |
| `cd screencropnet && make label-studio-docker` | the legacy Docker workflow |

First run: open <http://localhost:8080> and create a local account (email +
password). The account is stored in the local SQLite DB — there is no network
signup.

State lives in `~/.local/share/label-studio/` (SQLite DB + uploaded media) by
default; override with the `LABEL_STUDIO_BASE_DATA_DIR` environment variable.

## Local file serving

By default Label Studio only shows media you upload through the UI or that is
reachable by URL. To annotate the on-disk screenshots **without uploading
copies**, enable local file serving:

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/abs/path/to/scratch/datasets \
uvx label-studio
```

- `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` — turns the feature on.
- `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` — the only directory tree Label
  Studio may read from. Point it at a parent of your images.

`make label-studio-local` sets both for you (document root = `scratch/datasets/`).
Inside the app, add the files as a **Local files** source storage (Project →
Settings → Cloud Storage → Add Source Storage → Local files) pointing at a
subdirectory of the document root.

## Creating a project

1. Click **Create Project** (upper right).
2. **Project Name** tab — name it (e.g. `screencropnet-bbox`); only the name
   is required.
3. **Data Import** tab — upload images now, or skip and use local-files
   storage later.
4. **Labeling Setup** tab — pick a template (see `labeling-configs.md`) and
   customize the labels.
5. Click **Save**.

Post-creation options live under **Settings** (Data Manager, upper-right):
annotation instructions, task sampling order (sequential vs random), and more.

## Related

- `labeling-configs.md` — labeling-interface XML for bbox and classification.
- `screencropnet-workflow.md` — annotate twitter screenshots → screencropnet CSV.
