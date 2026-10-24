"""A batch may hold more than one write to the same key.

``write_batch`` took a lock per entry, so two entries for one key tried to
take the same lock twice. The second acquire waited on the first from inside
the same process and raised ``LockTimeoutError`` after five seconds, and the
whole batch was lost. The writes are applied in order, so the second one
should simply become the next version.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from amfs_core.models import MemoryEntry, Provenance
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


def test_two_writes_to_one_key_in_a_batch_are_both_applied(tmp_path: Path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")

    written = adapter.write_batch([_entry("first"), _entry("second")])

    assert [e.version for e in written] == [1, 2]
    current = adapter.read("svc/checkout", "retry")
    assert current is not None
    assert current.value == "second"
    history = adapter.list("svc/checkout", include_superseded=True)
    assert sorted((e.version, e.value) for e in history) == [(1, "first"), (2, "second")]
