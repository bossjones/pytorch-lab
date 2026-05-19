"""errors.run must swallow any exception and write a descriptive
line + traceback to stderr, never to stdout, and never re-raise."""

from __future__ import annotations

import importlib

import pytest

ERRORS_MODULES = ["screennet.errors", "screencropnet.errors"]


@pytest.fixture(params=ERRORS_MODULES)
def errors_mod(request):
    return importlib.import_module(request.param)


def test_run_executes_the_callable(errors_mod, capsys) -> None:
    called = []
    errors_mod.run(lambda: called.append(1), "task")
    assert called == [1]
    captured = capsys.readouterr()
    assert captured.err == ""


def test_run_swallows_exception_and_writes_to_stderr(errors_mod, capsys) -> None:
    def boom():
        raise ValueError("boom")

    errors_mod.run(boom, "mytask")  # must not raise
    captured = capsys.readouterr()
    assert "mytask: ValueError" in captured.err
    assert "ValueError: boom" in captured.err
