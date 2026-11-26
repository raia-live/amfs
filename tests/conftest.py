"""Shared fixtures for AMFS tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

#: Where a process is pointed at a real backend. The HTTP server and the MCP
#: server both prefer ``AMFS_HTTP_URL`` over any local store, so a test run from
#: a shell that still has these exported talks to whatever they name.
_BACKEND_ENV = ("AMFS_HTTP_URL", "AMFS_API_KEY", "AMFS_POSTGRES_DSN")


@pytest.fixture(autouse=True)
def _no_inherited_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit and integration tests never inherit a backend from the shell.

    A test run of the HTTP server, in a shell with ``AMFS_HTTP_URL`` and
    ``AMFS_API_KEY`` left over from other work, writes its synthetic entries
    to the hosted account they name, and there is no entry delete to take
    them back out. A test that wants one of these sets it itself
    (``monkeypatch.setenv``), which still works: this runs first.

    ``tests/e2e`` is exempt. Those tests exist to reach a hosted endpoint and
    skip themselves when the variables are absent.
    """
    if "e2e" in request.node.path.parts:
        return
    for name in _BACKEND_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def tmp_amfs_root(tmp_path: Path) -> Path:
    """Provide a temporary directory for filesystem adapter tests."""
    root = tmp_path / ".amfs"
    root.mkdir()
    return root
