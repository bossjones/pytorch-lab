"""CI / quality tasks — uv + ruff + pyright + pytest.

Replaces the old conda/pip-tools/black/isort/mypy/pylint scaffolding.
Usage: ``inv -l`` to list, e.g. ``inv ci.pytest``, ``inv ci.lint``.
"""

from invoke import task


@task
def clean(ctx):
    """Remove compiled python artifacts and coverage data."""
    ctx.run(
        "find . -name '*.pyc' -delete; "
        "find . -name '*.pyo' -delete; "
        "find . -name '__pycache__' -type d -exec rm -rf {} +; "
        "rm -f .coverage",
        warn=True,
    )


@task
def lint(ctx, fix=False):
    """ruff check (lint). Pass --fix to autofix."""
    ctx.run(f"uv run ruff check {'--fix ' if fix else ''}.", pty=True)


@task
def format(ctx, check=False):
    """ruff format. Pass --check for CI verification."""
    ctx.run(f"uv run ruff format {'--check ' if check else ''}.", pty=True)


@task
def typecheck(ctx):
    """pyright static type check."""
    ctx.run("uv run pyright", pty=True, warn=True)


@task
def pytest(ctx, k=None, verbose=False):
    """Run the test suite. ``inv ci.pytest -k iou`` to filter."""
    cmd = "uv run pytest"
    if verbose:
        cmd += " -vv"
    if k:
        cmd += f" -k {k!r}"
    ctx.run(cmd, pty=True)


@task(pre=[lint, typecheck, pytest])
def check(ctx):
    """Run lint + typecheck + tests (the full gate)."""
    print("ci.check: ruff + pyright + pytest complete")
