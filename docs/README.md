# pytorch-lab Documentation

Generated documentation for the PyTorch experimentation lab — a `uv`-managed,
Mac-only (Apple Silicon MPS + CPU) project containing two standalone model
subprojects (`screennet/` classification, `screencropnet/` bbox localization)
and shared training utilities (`going_modular/`).

## Sections

| Section | Description |
|---------|-------------|
| [Architecture](architecture/README.md) | System overview, component diagram, training-pipeline and device-selection Mermaid diagrams, component-responsibility tables. |
| [API Reference](api/README.md) | Module/function/class reference for `going_modular`, `helper_functions`, `screencropnet`, `screennet`, `tasks`, and `contrib` — extracted from signatures, type hints, and docstrings. |
| [Guides](guides/README.md) | Step-by-step usage: getting started, screennet classification, screencropnet localization, development workflow, and troubleshooting. |
| [Docstring Coverage](coverage/report.md) | AST-based audit of public functions/classes missing docstrings, with per-directory and per-file rollups and prioritized gaps. |

## Quick Start

New to the project? Start with [Guides → Getting Started](guides/getting-started.md).

Want the big picture? See [Architecture](architecture/README.md).

Looking up a function? See the [API Reference](api/README.md).

## Maintenance

This documentation is generated from the codebase. When source code changes
significantly, regenerate the affected section so docs stay synchronized. The
docstring coverage report ([coverage/report.md](coverage/report.md)) highlights
where inline documentation is weakest and should be improved first.
