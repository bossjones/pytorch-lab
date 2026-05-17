"""screennet.errors.run must swallow any exception and write a descriptive
line + traceback to stderr, never to stdout, and never re-raise."""

from __future__ import annotations

from screennet import errors


def test_run_executes_the_callable(capsys) -> None:
    called = []
    errors.run(lambda: called.append(1), "task")
    assert called == [1]
    captured = capsys.readouterr()
    assert captured.err == ""


def test_run_swallows_exception_and_writes_to_stderr(capsys) -> None:
    def boom():
        raise ValueError("boom")

    errors.run(boom, "mytask")  # must not raise
    captured = capsys.readouterr()
    assert "mytask: ValueError" in captured.err
    assert "ValueError: boom" in captured.err
