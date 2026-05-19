"""Local dev tasks — uv-based.

Replaces the old pip-tools/conda bootstrap scaffolding.
Usage: ``inv -l`` to list, e.g. ``inv local.clean``, ``inv local.jupyter``.
"""

from invoke import task


@task
def clean(ctx):
    """Remove .pyc / __pycache__ artifacts."""
    ctx.run(
        "find . -name '*.pyc' -delete; "
        "find . -name '__pycache__' -type d -exec rm -rf {} +",
        warn=True,
    )


@task
def sync(ctx):
    """Create/update the uv-managed virtualenv from uv.lock."""
    ctx.run("uv sync", pty=True)


@task
def lock(ctx):
    """Re-resolve dependencies into uv.lock."""
    ctx.run("uv lock", pty=True)


@task
def ipython(ctx):
    """Start an IPython shell inside the project env."""
    ctx.run("uv run ipython", pty=True)


@task
def jupyter(ctx):
    """Start a Jupyter notebook server inside the project env."""
    ctx.run("uv run jupyter notebook", pty=True)


@task
def env_works(ctx):
    """Smoke-check the environment (MPS + matplotlib)."""
    ctx.run("uv run python ./contrib/is-mps-available.py", pty=True)
    ctx.run("uv run python ./contrib/does-matplotlib-work.py", pty=True)
