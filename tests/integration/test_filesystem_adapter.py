"""Filesystem adapter tests — runs the full adapter contract suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from amfs_filesystem.adapter import FilesystemAdapter

from tests.integration.adapter_contract import AdapterContractTests


class TestFilesystemAdapter(AdapterContractTests):
    """Run all contract tests against the FilesystemAdapter."""

    @pytest.fixture
    def adapter(self, tmp_amfs_root: Path) -> FilesystemAdapter:
        a = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        yield a  # type: ignore[misc]
        a.close()
