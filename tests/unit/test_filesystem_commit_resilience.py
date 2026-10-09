"""Tests for filesystem adapter commit resilience and error handling."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

if sys.platform == "win32" and "fcntl" not in sys.modules:
    sys.modules["fcntl"] = MagicMock()

import pytest
from amfs_core.models import Commit
from amfs_filesystem.adapter import FilesystemAdapter


def test_save_and_get_commit_atomic(tmp_path: Path) -> None:
    adapter = FilesystemAdapter(root=tmp_path)
    commit = Commit(id=str(uuid4()), branch="main", message="test commit", author_agent_id="test-agent")
    adapter.save_commit(commit)

    loaded = adapter.get_commit(commit.id)
    assert loaded is not None
    assert loaded.id == commit.id
    assert loaded.message == "test commit"
    assert loaded.author_agent_id == "test-agent"


def test_get_commit_handles_corrupt_json(tmp_path: Path) -> None:
    adapter = FilesystemAdapter(root=tmp_path)
    commits_dir = tmp_path / "_commits"
    commits_dir.mkdir(parents=True, exist_ok=True)

    corrupt_id = "corrupt-commit-1"
    (commits_dir / f"{corrupt_id}.json").write_text("{malformed-json: True,,,", encoding="utf-8")

    # Must return None rather than raising json.JSONDecodeError or crashing
    assert adapter.get_commit(corrupt_id) is None


def test_get_commit_handles_empty_file(tmp_path: Path) -> None:
    adapter = FilesystemAdapter(root=tmp_path)
    commits_dir = tmp_path / "_commits"
    commits_dir.mkdir(parents=True, exist_ok=True)

    empty_id = "empty-commit-2"
    (commits_dir / f"{empty_id}.json").write_text("", encoding="utf-8")

    # Must return None rather than raising exception
    assert adapter.get_commit(empty_id) is None


def test_get_commit_handles_schema_mismatch(tmp_path: Path) -> None:
    adapter = FilesystemAdapter(root=tmp_path)
    commits_dir = tmp_path / "_commits"
    commits_dir.mkdir(parents=True, exist_ok=True)

    invalid_id = "invalid-schema-3"
    (commits_dir / f"{invalid_id}.json").write_text('{"random_field": 123}', encoding="utf-8")

    # Must return None rather than raising ValidationError
    assert adapter.get_commit(invalid_id) is None
