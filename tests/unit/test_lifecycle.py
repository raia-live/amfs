"""Unit tests for LifecycleManager (TTL sweep)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import MemoryEntry, OutcomeRecord, Provenance


# ---------------------------------------------------------------------------
# Mock adapter (same pattern as test_engine but tracks writes separately)
# ---------------------------------------------------------------------------

class MockAdapter(AdapterABC):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[MemoryEntry]] = {}
        self.writes: list[MemoryEntry] = []

    def read(
        self, entity_path: str, key: str, *, min_confidence: float = 0.0
    ) -> MemoryEntry | None:
        versions = self._store.get((entity_path, key))
        if not versions:
            return None
        current = versions[-1]
        return current if current.confidence >= min_confidence else None

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        k = (entry.entity_path, entry.key)
        if k not in self._store:
            self._store[k] = []
        self._store[k].append(entry)
        self.writes.append(entry)
        return entry

    def list(
        self, entity_path: str | None = None, *, include_superseded: bool = False
    ) -> list[MemoryEntry]:
        result: list[MemoryEntry] = []
        for (ep, _), versions in self._store.items():
            if entity_path is not None and ep != entity_path:
                continue
            if include_superseded:
                result.extend(versions)
            else:
                result.append(versions[-1])
        return result

    def watch(
        self, entity_path: str, callback: Callable[[MemoryEntry], None]
    ) -> WatchHandle:
        return WatchHandle(lambda: None)

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        return []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_entry(
    entity_path: str = "svc",
    key: str = "k",
    ttl_at: datetime | None = None,
    confidence: float = 1.0,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value="data",
        provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
        confidence=confidence,
        ttl_at=ttl_at,
    )


class TestLifecycleManager:
    def test_sweep_archives_expired(self) -> None:
        adapter = MockAdapter()
        # Entry expired 1 hour ago
        expired = _make_entry(ttl_at=_now() - timedelta(hours=1))
        adapter.write(expired)

        mgr = LifecycleManager(adapter)
        archived = mgr.sweep()

        assert len(archived) == 1
        assert archived[0].confidence == 0.0
        assert archived[0].ttl_at is None

    def test_sweep_ignores_non_expired(self) -> None:
        adapter = MockAdapter()
        future = _make_entry(ttl_at=_now() + timedelta(hours=1))
        adapter.write(future)

        mgr = LifecycleManager(adapter)
        archived = mgr.sweep()

        assert len(archived) == 0

    def test_sweep_ignores_no_ttl(self) -> None:
        adapter = MockAdapter()
        no_ttl = _make_entry(ttl_at=None)
        adapter.write(no_ttl)

        mgr = LifecycleManager(adapter)
        archived = mgr.sweep()

        assert len(archived) == 0

    def test_sweep_mixed_entries(self) -> None:
        adapter = MockAdapter()
        adapter.write(_make_entry(key="expired", ttl_at=_now() - timedelta(minutes=5)))
        adapter.write(_make_entry(key="future", ttl_at=_now() + timedelta(hours=1)))
        adapter.write(_make_entry(key="no-ttl"))

        mgr = LifecycleManager(adapter)
        archived = mgr.sweep()

        assert len(archived) == 1
        assert archived[0].key == "expired"

    def test_background_thread_starts_and_stops(self) -> None:
        adapter = MockAdapter()
        mgr = LifecycleManager(adapter, interval=0.1)

        assert not mgr.running
        mgr.start()
        assert mgr.running

        mgr.stop()
        assert not mgr.running

    def test_background_thread_runs_sweep(self) -> None:
        adapter = MockAdapter()
        adapter.write(_make_entry(ttl_at=_now() - timedelta(hours=1)))

        mgr = LifecycleManager(adapter, interval=0.1)
        mgr.start()
        # Wait for at least one sweep
        time.sleep(0.3)
        mgr.stop()

        # The sweep should have written an archived version
        assert len(adapter.writes) >= 2  # original + at least one archived version


class TestSweepAsksTheAdapterForExpiredRows:
    """``sweep()`` reads ``list_expired`` where the adapter has it and lists
    the store only where it does not. Over HTTP the listing was the whole
    tenant once per client process per interval; the query is what a hosted
    server sweeps with."""

    def test_uses_list_expired_and_does_not_list(self) -> None:
        class Querying(MockAdapter):
            listed = 0

            def list_expired(self, *, now, branch="main"):
                assert now.tzinfo is not None
                return [e for e in super().list() if e.ttl_at is not None and e.ttl_at <= now]

            def list(self, *args, **kwargs):
                Querying.listed += 1
                return super().list(*args, **kwargs)

        adapter = Querying()
        adapter.write(_make_entry(key="expired", ttl_at=_now() - timedelta(hours=1)))
        adapter.write(_make_entry(key="live", ttl_at=_now() + timedelta(hours=1)))
        adapter.write(_make_entry(key="forever", ttl_at=None))
        Querying.listed = 0

        archived = LifecycleManager(adapter).sweep()

        assert [e.key for e in archived] == ["expired"]
        assert archived[0].confidence == 0.0 and archived[0].ttl_at is None
        assert Querying.listed == 0

    def test_falls_back_to_listing_without_list_expired(self) -> None:
        adapter = MockAdapter()
        adapter.write(_make_entry(key="expired", ttl_at=_now() - timedelta(hours=1)))
        adapter.write(_make_entry(key="live", ttl_at=_now() + timedelta(hours=1)))

        assert [e.key for e in LifecycleManager(adapter).sweep()] == ["expired"]
