"""Read-time TTL enforcement: expired entries must not be returned by read().

Complements test_lifecycle.py (which covers the background archive sweep). Here
we assert that read() hides an expired entry immediately, without waiting for a
sweep, so a stale/expired fact can never be surfaced to an agent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from amfs_core.models import MemoryEntry, Provenance
from amfs_filesystem.adapter import FilesystemAdapter


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _entry(key: str, ttl_at: datetime | None) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/mod",
        key=key,
        value="secret-token",
        provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
        confidence=1.0,
        ttl_at=ttl_at,
    )


def test_read_hides_expired_entry(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    adapter.write(_entry("expired", ttl_at=_now() - timedelta(hours=1)))

    assert adapter.read("svc/mod", "expired") is None


def test_read_returns_future_ttl_entry(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    adapter.write(_entry("future", ttl_at=_now() + timedelta(hours=1)))

    got = adapter.read("svc/mod", "future")
    assert got is not None and got.key == "future"


def test_read_returns_entry_without_ttl(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    adapter.write(_entry("no-ttl", ttl_at=None))

    got = adapter.read("svc/mod", "no-ttl")
    assert got is not None and got.key == "no-ttl"


def test_is_expired_handles_naive_datetime() -> None:
    naive_past = datetime.utcnow() - timedelta(hours=1)
    entry = MemoryEntry(
        entity_path="svc/mod",
        key="k",
        value="v",
        provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
        ttl_at=naive_past,
    )
    assert entry.is_expired() is True
