#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Docstring-coverage audit — regenerates docs/coverage_docs/report.md.

Scans the first-party Python source with the ``ast`` module and reports, per
file and per directory, how many public functions, methods, and classes carry
a docstring. A symbol is public when its name does not start with ``_``, with
``__init__`` always counted as public. Class bodies are walked so methods are
counted; defs nested inside function bodies are not.

Usage:
    uv run contrib/docstring_coverage.py
    uv run contrib/docstring_coverage.py --output docs/coverage_docs/report.md
"""

from __future__ import annotations

import argparse
import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories scanned (relative to the repo root) and standalone in-scope files.
SCOPE_DIRS = (
    "going_modular/going_modular",
    "screencropnet",
    "screennet",
    "tasks",
    "contrib",
)
SCOPE_FILES = ("helper_functions.py",)

# Path components that exclude a file even within a scope directory.
EXCLUDED_PARTS = frozenset({"__pycache__"})

# Per-directory table row order.
DIRECTORY_ORDER = (
    "going_modular",
    "screencropnet",
    "screennet",
    "tasks",
    "contrib",
    "root",
)

# Prose wrap width — comfortably under the .rumdl.toml MD013 line-length limit.
WRAP_WIDTH = 88


@dataclass
class Symbol:
    """An undocumented public symbol: qualified name, line number, and kind."""

    qualname: str
    lineno: int
    kind: str


@dataclass
class FileStats:
    """Docstring-coverage counts for a single scanned file."""

    pub_funcs: int
    doc_funcs: int
    pub_classes: int
    doc_classes: int
    has_module_docstring: bool
    gaps: list[Symbol]


def scan_file(path: Path) -> FileStats:
    """Scan one Python file and return its docstring-coverage counts.

    Module-level functions/classes and class methods are counted; a symbol is
    public unless its name starts with ``_`` (``__init__`` excepted). Defs
    nested in function bodies are ignored. Undocumented public symbols are
    collected in :attr:`FileStats.gaps`.
    """
    tree = ast.parse(path.read_text())
    pub_funcs = doc_funcs = pub_classes = doc_classes = 0
    gaps: list[Symbol] = []

    def is_public(name: str) -> bool:
        return not name.startswith("_") or name == "__init__"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not is_public(node.name):
                continue
            pub_funcs += 1
            if ast.get_docstring(node):
                doc_funcs += 1
            else:
                gaps.append(Symbol(node.name, node.lineno, "function"))
        elif isinstance(node, ast.ClassDef):
            if not is_public(node.name):
                continue
            pub_classes += 1
            if ast.get_docstring(node):
                doc_classes += 1
            else:
                gaps.append(Symbol(node.name, node.lineno, "class"))
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not is_public(item.name):
                    continue
                pub_funcs += 1
                qualname = f"{node.name}.{item.name}"
                if ast.get_docstring(item):
                    doc_funcs += 1
                else:
                    gaps.append(Symbol(qualname, item.lineno, "function"))

    return FileStats(
        pub_funcs=pub_funcs,
        doc_funcs=doc_funcs,
        pub_classes=pub_classes,
        doc_classes=doc_classes,
        has_module_docstring=ast.get_docstring(tree) is not None,
        gaps=gaps,
    )


def directory_bucket(rel_path: str) -> str:
    """Return the per-directory table label for a repo-relative file path.

    Repo-root files map to ``"root"``; nested files map to their top-level
    directory (``going_modular/going_modular/x.py`` -> ``going_modular``).
    """
    parts = Path(rel_path).parts
    return "root" if len(parts) == 1 else parts[0]


def iter_scope_files(repo_root: Path) -> list[Path]:
    """Return the sorted list of in-scope Python files under `repo_root`.

    Scans :data:`SCOPE_DIRS` recursively plus the standalone
    :data:`SCOPE_FILES`, skipping any path containing an excluded component
    (see :data:`EXCLUDED_PARTS`). Only ``*.py`` files — notebooks are skipped.
    """
    found: list[Path] = []
    for rel_file in SCOPE_FILES:
        candidate = repo_root / rel_file
        if candidate.is_file():
            found.append(candidate)
    for rel_dir in SCOPE_DIRS:
        for path in (repo_root / rel_dir).rglob("*.py"):
            parts = path.relative_to(repo_root).parts
            if EXCLUDED_PARTS.isdisjoint(parts):
                found.append(path)
    return sorted(found)


def _wrap(text: str, initial: str = "", subsequent: str = "") -> list[str]:
    """Wrap prose to the report's line width, returning a list of lines.

    Words (backtick code spans, file paths) are never split, so a rare long
    token may still overrun; tables and code blocks are exempt from the
    line-length check via ``.rumdl.toml``.
    """
    return textwrap.wrap(
        text,
        width=WRAP_WIDTH,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _pct(num: int, den: int, *, na_when_zero: bool = False) -> str:
    """Format a coverage percentage; a zero denominator yields 'n/a' or 100%."""
    if den == 0:
        return "n/a" if na_when_zero else "100.0%"
    return f"{100 * num / den:.1f}%"


def render_report(file_stats: dict[str, FileStats]) -> str:
    """Render the full Markdown docstring-coverage report from per-file stats."""
    tot_pf = sum(s.pub_funcs for s in file_stats.values())
    tot_df = sum(s.doc_funcs for s in file_stats.values())
    tot_pc = sum(s.pub_classes for s in file_stats.values())
    tot_dc = sum(s.doc_classes for s in file_stats.values())

    lines: list[str] = [
        "# Docstring Coverage Audit",
        "",
        *_wrap(
            "First-party Python source only. Generated by "
            "`contrib/docstring_coverage.py` (AST scan via "
            "`ast.get_docstring`). Run `make docstring-audit` to regenerate. "
            "See Methodology below.",
            "> ",
            "> ",
        ),
        "",
        "## Summary",
        "",
        f"- **Public functions/methods documented:** {tot_df}/{tot_pf} = "
        f"**{_pct(tot_df, tot_pf)}**",
        f"- **Public classes documented:** {tot_dc}/{tot_pc} = "
        f"**{_pct(tot_dc, tot_pc)}**",
        f"- **Combined public symbols:** {tot_df + tot_dc}/{tot_pf + tot_pc} = "
        f"**{_pct(tot_df + tot_dc, tot_pf + tot_pc)}**",
        "",
        f"Total in-scope files scanned: {len(file_stats)}.",
        "",
        "## Per-directory coverage",
        "",
        "| Directory | Functions | Documented | % | Classes | Documented | % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for bucket in DIRECTORY_ORDER:
        group = [s for p, s in file_stats.items() if directory_bucket(p) == bucket]
        pf = sum(s.pub_funcs for s in group)
        df = sum(s.doc_funcs for s in group)
        pc = sum(s.pub_classes for s in group)
        dc = sum(s.doc_classes for s in group)
        lines.append(
            f"| `{bucket}` | {pf} | {df} | {_pct(df, pf)} | "
            f"{pc} | {dc} | {_pct(dc, pc)} |"
        )
    lines.append(
        f"| **TOTAL** | **{tot_pf}** | **{tot_df}** | **{_pct(tot_df, tot_pf)}** | "
        f"**{tot_pc}** | **{tot_dc}** | **{_pct(tot_dc, tot_pc)}** |"
    )

    lines += [
        "",
        "## Per-file coverage (worst first)",
        "",
        *_wrap(
            "Sorted ascending by function coverage %. Files with no public "
            "functions are listed last. `mod?` = module-level docstring "
            "present."
        ),
        "",
        "| File | Pub funcs | Doc | Func % | Pub cls | Doc | mod? |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]

    def file_sort_key(item: tuple[str, FileStats]) -> tuple[bool, float, int, str]:
        path, s = item
        no_funcs = s.pub_funcs == 0
        fpct = 0.0 if no_funcs else 100 * s.doc_funcs / s.pub_funcs
        return (no_funcs, fpct, -s.pub_funcs, path)

    for path, s in sorted(file_stats.items(), key=file_sort_key):
        lines.append(
            f"| `{path}` | {s.pub_funcs} | {s.doc_funcs} | "
            f"{_pct(s.doc_funcs, s.pub_funcs, na_when_zero=True)} | "
            f"{s.pub_classes} | {s.doc_classes} | "
            f"{'yes' if s.has_module_docstring else 'no'} |"
        )

    lines += [
        "",
        "## Top gaps (undocumented public symbols)",
        "",
        *_wrap(
            "Every in-scope file with undocumented public symbols, most gaps "
            "first. Format: `file:line` — qualified name (kind)."
        ),
    ]
    gap_files = sorted(
        ((p, s) for p, s in file_stats.items() if s.gaps),
        key=lambda item: (-len(item[1].gaps), item[0]),
    )
    total_gaps = 0
    for path, s in gap_files:
        lines += ["", f"### `{path}`", ""]
        for sym in sorted(s.gaps, key=lambda g: g.lineno):
            total_gaps += 1
            lines.append(f"- `{path}:{sym.lineno}` — `{sym.qualname}` ({sym.kind})")
    lines += [
        "",
        "Total undocumented public symbols across all in-scope files: "
        f"**{total_gaps}**.",
        "",
        "## Methodology",
        "",
        *_wrap(
            "**Technique:** Static analysis via Python's `ast` module. "
            "`contrib/docstring_coverage.py` parses each in-scope file and "
            "uses `ast.get_docstring()` to decide whether a module, function, "
            "method, or class carries a docstring.",
            "- ",
            "  ",
        ),
        *_wrap(
            "**Visibility:** Symbols whose name starts with `_` are treated "
            "as private and excluded from the public counts, **except "
            "`__init__`**, which is always counted as public.",
            "- ",
            "  ",
        ),
        *_wrap(
            "**Classes & methods:** Class bodies are walked so methods are "
            "counted; nested defs inside function bodies (local helpers) are "
            "not counted.",
            "- ",
            "  ",
        ),
        *_wrap(
            "**Scope:** `going_modular/going_modular/`, "
            "`helper_functions.py`, `screencropnet/`, `screennet/`, `tasks/`, "
            "`contrib/`. **Excluded:** `tests/`, `.claude/`, "
            "`lint-configs-python/`, `going_modular/pytorch_ipynb/`, all "
            "`*.ipynb` notebooks, `scratch/`, and `.venv/`.",
            "- ",
            "  ",
        ),
        *_wrap(
            "**Regeneration:** run `make docstring-audit` (or "
            "`uv run contrib/docstring_coverage.py`).",
            "- ",
            "  ",
        ),
        "- **Numbers** are exact AST counts, not estimates.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="docstring_coverage.py",
        description="Audit docstring coverage and regenerate the coverage report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "coverage_docs" / "report.md",
        help="report file to write (default: docs/coverage_docs/report.md)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Scans the source tree and writes the coverage report."""
    args = build_parser().parse_args(argv)
    stats = {
        str(path.relative_to(REPO_ROOT)): scan_file(path)
        for path in iter_scope_files(REPO_ROOT)
    }
    args.output.write_text(render_report(stats))
    print(f"wrote {args.output} ({len(stats)} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
