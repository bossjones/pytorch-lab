"""Shared pytest fixtures.

Conventions: CPU-only, deterministic seeds, mock heavy I/O.
See ai_docs/pytorch-testing-notes.md.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed every test so tensor-valued assertions are stable."""
    torch.manual_seed(0)


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")
