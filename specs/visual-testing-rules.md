# Plan: Add a `.claude/rules/visual-testing.md` conventions file

## Context

`ai_docs/html/visual-testing-patterns.html` was recently produced as a field-notes
reference on testing plots, images, and tensors with pytest (pytest-mpl, Axes
data-extraction, OpenCV/SSIM/imagehash diffing, pytest-mock, `torch.testing`).
The repo already encodes pure-unit / mock / tensor-tolerance conventions in
`ai_docs/pytorch-testing-notes.md` and an autouse `_deterministic` fixture in
`tests/conftest.py`. The genuine gap is **visual/plot/image** testing: matplotlib
plotting code in `helper_functions.py` and `screennet/main.py` is untested, and no
visual-testing plugin is installed. Without a short, enforced convention, future
visual tests will drift toward flaky byte-for-byte comparisons.

This plan creates one concise `.claude/rules/` file that turns the HTML field
notes into actionable, repo-specific rules — consistent in style with the
existing rules files — without adding dependencies or test code yet.

## Task Description

Author `.claude/rules/visual-testing.md`: a visual/image-focused testing
conventions file derived from `ai_docs/html/visual-testing-patterns.html`,
following the repo's existing `.claude/rules/*` template (`paths:` frontmatter,
`audit-protocol.md`-style "rules → Correct → BAD → Why This Matters"). It must
complement — not duplicate — `ai_docs/pytorch-testing-notes.md`, and reference
both the notes and the HTML doc as the detailed catalog.

Task type: **chore** (documentation/conventions). Complexity: **simple**.

## Objective

A single new file, `.claude/rules/visual-testing.md`, exists, follows the
established rules-file format, is markdown-clean per
`.claude/rules/documentation.md`, introduces **no** dependency or `pyproject.toml`
changes, and gives any contributor unambiguous guidance for testing plots/images/
tensors in this repo.

## Problem Statement

The HTML reference is long-form and generic ("for the Astronauts workshop"). It is
not discoverable as a rule, not scoped to this repo's CPU-only/offline/
deterministic philosophy, and not in the lightweight directive format the team's
other `.claude/rules/` files use. Contributors adding the first matplotlib/image
test have no concise "do this, not that" to follow, risking flaky pixel-equality
tests in CI.

## Solution Approach

Distill the HTML doc into ~6 numbered, imperative rules scoped to test files via
`paths:` frontmatter. Keep the visual/image emphasis (data-extraction-first,
pytest-mpl with Agg + tolerance, the cv2/SSIM/imagehash decision, inspectable
failure artifacts) and cross-link `ai_docs/pytorch-testing-notes.md` for the
mock/tensor conventions rather than restating them. State explicitly that
`pytest-mpl`/`scikit-image`/`imagehash`/`opencv-python` are not yet deps and must
be added via `uv add --dev` only when first needed. Mirror `audit-protocol.md`'s
structure (intro → numbered list → `## Correct` → `## BAD` → `## Why This
Matters` → bold `**Remember:**`).

## Relevant Files

Use these files to complete the task:

- `ai_docs/html/visual-testing-patterns.html` — **source material**; the 5 sections
  (pytest-mpl, Axes data-extraction, cv2/SSIM/imagehash, pytest-mock,
  `torch.testing.assert_close`) and the decision table drive the rule content.
- `.claude/rules/audit-protocol.md` — **structural template**: intro sentence,
  numbered do/don't list, `## Correct`, `## BAD`, `## Why This Matters`, closing
  `**Remember:**` line.
- `.claude/rules/python-scripts.md` — exact **frontmatter format**:
  `paths: <comma-separated globs>` on line 2 between `---` fences.
- `.claude/rules/documentation.md` — markdown conventions the new file must obey
  (ATX headers, one blank line around headers, fenced blocks with language, no
  trailing whitespace, single trailing newline).
- `ai_docs/pytorch-testing-notes.md` — **cross-reference target**; the new file
  must point here for mock/tensor/CPU-only conventions instead of duplicating
  them.
- `tests/conftest.py` — cite the existing autouse `_deterministic` fixture
  (`torch.manual_seed(0)`) the rules should rely on.

### New Files

- `.claude/rules/visual-testing.md` — the deliverable (full proposed content
  below).
- `specs/visual-testing-rules.md` — this spec, per the `/plan` convention
  (`PLAN_OUTPUT_DIRECTORY: specs/`).

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom.

### 1. Save the spec to `specs/`

- Write this plan to `specs/visual-testing-rules.md` (per the `/plan`
  command's `PLAN_OUTPUT_DIRECTORY`). Pure documentation, no code impact.

### 2. Create `.claude/rules/visual-testing.md`

- Create the file with exactly the content in **Proposed File Content** below.
- Frontmatter `paths:` must be a single comma-separated line, matching the
  `python-scripts.md` format.
- Use ATX headers with one blank line before/after, fenced code blocks with
  `python` language tags, no trailing whitespace, end with a single newline
  (per `documentation.md`).

### 3. Verify scope & cross-links

- Confirm the file references `ai_docs/pytorch-testing-notes.md` and
  `ai_docs/html/visual-testing-patterns.html`, and does **not** restate the
  mock/tensor conventions already in the notes.
- Confirm it adds **no** `pyproject.toml`/`uv.lock` changes and explicitly defers
  tool installation to `uv add --dev`.

### 4. Validate

- Run the validation commands in the section below.

## Proposed File Content

The implementation must write `.claude/rules/visual-testing.md` with this exact
content:

````md
---
paths: tests/**/*.py, **/test_*.py, **/conftest.py
---

# Visual & Image Testing Conventions

Plots, images, and tensors don't fit `assertEqual` — compare them by numeric or
perceptual tolerance, never byte-for-byte. These rules extend (don't replace)
`ai_docs/pytorch-testing-notes.md`; the full pattern catalog with copy-paste
code lives in `ai_docs/html/visual-testing-patterns.html`.

When a test produces a plot, image, or tensor:

1. Test the data, not the pixels, first - Interrogate the matplotlib `Axes`
   (`ax.patches` heights, `line.get_xdata()`, tick labels) with
   `np.testing.assert_array_equal`. Most plot bugs are math bugs in disguise.
2. Use pixel/baseline tests only for "does it look right" - Reach for
   `pytest-mpl` (`@pytest.mark.mpl_image_compare`) with an explicit
   `tolerance=`, commit baselines under `tests/baseline/`, and force the Agg
   backend. Never generate baselines on macOS for Linux CI.
3. Pick the image-diff tool by intent - `cv2.absdiff().mean() < tol` for exact
   output from the same pipeline; `skimage`'s `structural_similarity`
   (SSIM ≥ ~0.95) for anything lossy (JPEG/resize); `imagehash.phash` Hamming
   distance ≤ 5 for format/size-agnostic "same picture". Never assert exact
   equality on a lossy image.
4. Keep visual tests deterministic and offline - Rely on the autouse
   `_deterministic` fixture (`tests/conftest.py`, seeds `torch.manual_seed(0)`),
   set `matplotlib.use("Agg")`, and `mocker.patch` disk/network/model I/O. No
   dataset downloads, no training loops.
5. Make failures inspectable - On mismatch, write `actual.png` and an amplified
   `diff.png` (×10) into `tmp_path`; gate baseline overwrites behind a
   deliberate `--update-baselines` option, never silently.
6. Compare tensors with tolerance - Use
   `torch.testing.assert_close(actual, expected, rtol=…, atol=…)`; exact `==`
   only for shapes/ints. `pytest-pytorch` is for contributing to PyTorch
   itself, not for testing code that merely uses it.

Visual-testing tools (`pytest-mpl`, `scikit-image`, `imagehash`,
`opencv-python`) are not yet project dependencies. Add them only when a test
first needs one, via `uv add --dev <pkg>` — never `uv pip install`.

## Correct

```python
def test_revenue_bars_match_input():
    """Math-first: assert the plotted heights, not the pixels."""
    fig, ax = plt.subplots()
    ax.bar(["q1", "q2", "q3"], [1.2, 1.8, 2.6])
    heights = [b.get_height() for b in ax.patches]
    assert heights == [1.2, 1.8, 2.6]


def test_thumbnail_perceptually_matches():
    """Lossy output → SSIM with an explicit threshold, not byte equality."""
    score, _ = ssim(expected_gray, actual_gray, full=True)
    assert score >= 0.95, f"SSIM {score:.3f} below 0.95"
```

## BAD

```python
def test_chart():
    fig = make_chart()
    assert fig == load_baseline()          # figures aren't ==-comparable


def test_thumbnail():
    actual = Image.open("build/out.jpg").tobytes()
    expected = Image.open("tests/fixtures/out.jpg").tobytes()
    assert actual == expected              # exact bytes on a lossy JPEG: flaky
```

## Why This Matters

Byte-for-byte assertions on visual output fail for reasons that have nothing to
do with correctness — font stacks, anti-aliasing, JPEG round-trips, matplotlib
versions. They produce flaky CI and train the team to ignore failures.
Tolerance- and data-based assertions test what the code actually promises, stay
stable across machines, and turn a red build into a real question: *is the new
image correct?*

**Remember:** Assert the math when you can, perceptual tolerance when you must,
exact pixels only when the pipeline is deterministic.
````

## Acceptance Criteria

- `.claude/rules/visual-testing.md` exists with the exact content above.
- Frontmatter is a single `paths:` line between `---` fences, matching
  `python-scripts.md`'s format.
- Structure mirrors `audit-protocol.md`: intro → numbered list → `## Correct` →
  `## BAD` → `## Why This Matters` → `**Remember:**`.
- Scope is visual/image-focused and cross-references
  `ai_docs/pytorch-testing-notes.md` and the HTML doc; it does not duplicate the
  notes' mock/tensor sections.
- No changes to `pyproject.toml`, `uv.lock`, `tests/`, or any source file.
- `specs/visual-testing-rules.md` exists (this spec).
- Markdown obeys `.claude/rules/documentation.md` (ATX headers, blank lines
  around headers, fenced blocks with language tags, no trailing whitespace,
  single trailing newline).

## Validation Commands

Execute these to validate the task is complete:

- `test -f .claude/rules/visual-testing.md && echo OK` — file was created.
- `test -f specs/visual-testing-rules.md && echo OK` — spec copy saved.
- `head -3 .claude/rules/visual-testing.md` — confirm `---` / `paths:` / `---`
  frontmatter shape matches `.claude/rules/python-scripts.md`.
- `grep -n "pytorch-testing-notes.md\|visual-testing-patterns.html" .claude/rules/visual-testing.md`
  — both cross-references present.
- `git status --porcelain` — only `.claude/rules/visual-testing.md` and
  `specs/visual-testing-rules.md` are added; `pyproject.toml`/`uv.lock`/`tests/`
  untouched.
- `git diff --stat -- pyproject.toml uv.lock` — empty (no dependency changes).
- If `rumdl` is available: `rumdl .claude/rules/visual-testing.md` — markdown
  lint clean (matches `documentation.md` linting note).

## Notes

- No new libraries are required by this task; the rules file intentionally only
  *documents* `pytest-mpl`/`scikit-image`/`imagehash`/`opencv-python` and defers
  any `uv add --dev` to a future change that adds an actual visual test.
- Style decisions confirmed with the user: **visual/image-focused** scope,
  **docs-only** (no dependency changes), filename **`visual-testing.md`**.
- `.claude/rules/` files are discovered by convention (referenced where needed,
  e.g. from commands), not auto-loaded; no index/registration step is required.
- The unicode arrows/symbols (→, ×, ≥, ≤, …) in the proposed content are
  intentional and match the field-notes tone; keep them verbatim.
