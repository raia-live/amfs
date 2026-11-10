"""A failed write must not take the current version of a key with it.

``write`` moved the current version to superseded before it wrote the new
one. A write that failed after that step left the key with no current file:
the previous value was still on disk, but ``read`` returned ``None``. The new
version now goes in first, and the old one is superseded only once it is
safely in place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from amfs_core.models import MemoryEntry, Provenance
from amfs_filesystem import adapter as fs_adapter
from amfs_filesystem.adapter import FilesystemAdapter


def _entry(value: str) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/checkout",
        key="retry",
        value=value,
        provenance=Provenance(
            agent_id="ag", session_id="s", written_at=datetime.now(UTC)
        ),
        confidence=1.0,
    )


@pytest.fixture
def adapter(tmp_path: Path) -> FilesystemAdapter:
    return FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")


def _disk_fills_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the next serialisation fail, as a full disk would mid-write."""

    def _fail(*args: object, **kwargs: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(fs_adapter.json, "dumps", _fail)


def test_a_failed_write_leaves_the_previous_version_current(
    adapter: FilesystemAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter.write(_entry("old"))
    _disk_fills_up(monkeypatch)

    with pytest.raises(OSError):
        adapter.write(_entry("new"))

    current = adapter.read("svc/checkout", "retry")
    assert current is not None
    assert current.value == "old"


def test_a_failed_batch_leaves_the_previous_version_current(
    adapter: FilesystemAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter.write(_entry("old"))
    _disk_fills_up(monkeypatch)

    with pytest.raises(OSError):
        adapter.write_batch([_entry("new")])

    current = adapter.read("svc/checkout", "retry")
    assert current is not None
    assert current.value == "old"


def test_a_good_write_leaves_one_current_file(adapter: FilesystemAdapter) -> None:
    adapter.write(_entry("old"))
    adapter.write(_entry("new"))

    current = adapter.list("svc/checkout")
    assert [(e.version, e.value) for e in current] == [(2, "new")]


def test_a_write_clears_a_current_file_left_by_an_interrupted_write(
    adapter: FilesystemAdapter,
) -> None:
    adapter.write(_entry("old"))
    # What a write that stopped after writing v002 but before superseding v001
    # leaves on disk: two current files.
    key_dir = adapter._layout.key_dir("svc/checkout", "retry")
    (key_dir / "v002_current.json").write_bytes(
        (key_dir / "v001_current.json").read_bytes()
    )

    adapter.write(_entry("newest"))

    current = adapter.list("svc/checkout")
    assert [(e.version, e.value) for e in current] == [(3, "newest")]
