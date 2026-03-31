"""Shared fixtures for AMFS tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_amfs_root(tmp_path: Path) -> Path:
    """Provide a temporary directory for filesystem adapter tests."""
    root = tmp_path / ".amfs"
    root.mkdir()
    return root
