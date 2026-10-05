"""A store-wide consolidation pass is bounded; a fleet runs it once.

``ConsolidationStrategy.run`` used to call ``adapter.list(branch=...)`` — every
current entry of the tenant, deserialised before a single rule ran. On a large
tenant that one statement exceeded the statement timeout, held its transaction
past the idle-in-transaction limit while Python chewed the rows, and sat on the
embedded worker's two-connection pool while the catch-up scan waited for one.
The pass now walks the store entity by entity through ``list_scopes`` and an
indexed ``search`` per entity, stops at an entity or time ceiling, and skips
adapters without the aggregate back to the old listing.

In the worker, catch-up and consolidation share a non-blocking scan lock, the
consolidation pass is taken under a fleet-wide advisory lock so one instance
runs it per interval, and both clocks advance on failure so a broken pass is
retried on the next interval rather than the next 5-second tick.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("amfs_cortex", reason="amfs_cortex not installed")

from amfs_core.models import MemoryEntry, MemoryType, Provenance
from amfs_cortex.consolidator import ConsolidationStrategy


def _entry(key: str, *, entity_path: str, age_days: float = 0.0,
           memory_type: MemoryType = MemoryType.FACT, confidence: float = 1.0,
           agent_id: str = "agent-a") -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path, key=key, value=f"value-{key}",
        provenance=Provenance(agent_id=agent_id, session_id="s",
                              written_at=datetime.now(timezone.utc) - timedelta(days=age_days)),
        confidence=confidence, memory_type=memory_type, tier=1,
    )


def _store(by_entity: dict[str, list[MemoryEntry]]) -> MagicMock:
    adapter = MagicMock()
    adapter.list_scopes.return_value = (set(by_entity), {"agent-a"})
    adapter.search.side_effect = lambda q, branch="main": list(by_entity.get(q.entity_path, []))
    adapter.list.side_effect = AssertionError("the whole-store listing is what this replaces")
    adapter.write.side_effect = lambda e: e
    adapter.list_branches.return_value = []
    return adapter


# ── ConsolidationStrategy.run ───────────────────────────────────────────────


def test_pass_reads_entity_by_entity_and_never_lists_the_store() -> None:
    adapter = _store({
        "acme/billing": [_entry("stale", entity_path="acme/billing", age_days=60)],
        "acme/support": [_entry("fresh", entity_path="acme/support")],
    })

    report = ConsolidationStrategy(adapter).run(branch="main")

    adapter.list_scopes.assert_called_once_with(branch="main")
    adapter.list.assert_not_called()
    assert [c.args[0].entity_path for c in adapter.search.call_args_list] == ["acme/billing", "acme/support"]
    assert all(c.args[0].limit == 1000 and c.kwargs == {"branch": "main"} for c in adapter.search.call_args_list)
    assert report.entity_path == "*"
    assert report.auto_archived == 1  # the stale billing entry, as the listing pass would archive it
    assert report.compression_ratio == 2.0


def test_tier_a_rules_still_apply_within_an_entity() -> None:
    """Superseded beliefs are found per key within an entity, so the per-entity read loses nothing."""
    adapter = _store({
        "svc": [
            _entry("k1", entity_path="svc", confidence=0.95),
            _entry("k1", entity_path="svc", confidence=0.6, memory_type=MemoryType.BELIEF, agent_id="agent-b"),
        ],
    })
    report = ConsolidationStrategy(adapter).run()
    assert report.auto_archived >= 1
    assert adapter.write.call_args_list[0].args[0].tier == 3


def test_pass_stops_at_the_entity_ceiling_and_leaves_the_rest() -> None:
    adapter = _store({f"e{i:03d}": [_entry("k", entity_path=f"e{i:03d}")] for i in range(10)})

    report = ConsolidationStrategy(adapter, max_entities=4).run()

    assert adapter.search.call_count == 4
    # Sorted order: the same first four every time until they are done, the next pass continues.
    assert [c.args[0].entity_path for c in adapter.search.call_args_list] == ["e000", "e001", "e002", "e003"]
    assert report.entity_path == "*"


def test_pass_stops_when_the_time_budget_is_spent() -> None:
    by_entity = {f"e{i}": [_entry("k", entity_path=f"e{i}")] for i in range(5)}
    adapter = _store(by_entity)
    slow = adapter.search.side_effect

    def slow_search(q, branch="main"):
        time.sleep(0.03)
        return slow(q, branch=branch)

    adapter.search.side_effect = slow_search
    ConsolidationStrategy(adapter, time_budget_s=0.05).run()
    assert 1 <= adapter.search.call_count < 5


def test_empty_store_reports_nothing_without_reading_entries() -> None:
    adapter = _store({})
    report = ConsolidationStrategy(adapter).run()
    adapter.search.assert_not_called()
    assert report.auto_archived == 0 and report.compression_ratio == 1.0


def test_adapter_without_the_aggregate_takes_the_listing_path() -> None:
    class Plain:
        def __init__(self):
            self.listed = 0

        def list(self, branch="main"):
            self.listed += 1
            return [_entry("stale", entity_path="svc", age_days=60)]

        def search(self, q, branch="main"):
            return self.list(branch)

        def write(self, e):
            return e

        def log_event(self, e):
            return None

        def list_branches(self):
            return []

        def create_branch(self, b):
            return b

    adapter = Plain()
    report = ConsolidationStrategy(adapter).run()
    assert adapter.listed >= 1
    assert report.auto_archived == 1


def test_tier_b_detection_reuses_the_entries_in_hand() -> None:
    """Three agents converging on a key produce a proposal from the one search, not a second one."""
    adapter = _store({
        "svc": [_entry("k1", entity_path="svc", agent_id=a, confidence=0.9) for a in ("a", "b", "c")],
    })
    report = ConsolidationStrategy(adapter).run()
    assert adapter.search.call_count == 1
    assert report.proposals_created >= 1
    adapter.create_branch.assert_called()


# ── CortexWorker ────────────────────────────────────────────────────────────


def _worker(adapter=None, **kw):
    from amfs_cortex.compiler import DigestCompiler
    from amfs_cortex.worker import CortexWorker

    adapter = adapter or MagicMock()
    compiler = DigestCompiler(adapter=adapter, namespace="ns")
    return CortexWorker(dsn="postgresql://fake", compiler=compiler, use_advisory_lock=False, **kw)


def test_consolidation_skips_while_a_catchup_scan_holds_the_scan_lock() -> None:
    worker = _worker()
    worker._acquire_fleet_lock = lambda name: MagicMock()  # would win the fleet lock
    worker._run_consolidation_locked = MagicMock()

    assert worker._scan_lock.acquire(blocking=False)
    try:
        worker._run_consolidation()
    finally:
        worker._scan_lock.release()

    worker._run_consolidation_locked.assert_not_called()
    assert worker._last_consolidation == 0.0  # the tick retries once the scan is over


def test_catchup_skips_while_consolidation_holds_the_scan_lock() -> None:
    worker = _worker()
    worker._catchup_missing_digests_locked = MagicMock()
    with worker._scan_lock:
        worker._catchup_missing_digests()
    worker._catchup_missing_digests_locked.assert_not_called()


def test_consolidation_skips_when_another_instance_holds_the_fleet_lock() -> None:
    worker = _worker()
    worker._acquire_fleet_lock = lambda name: None
    worker._run_consolidation_locked = MagicMock()

    before = time.monotonic()
    worker._run_consolidation()

    worker._run_consolidation_locked.assert_not_called()
    assert worker._last_consolidation >= before  # waits a full interval, not 5 s
    assert worker.activity_log[-1]["type"] == "consolidation_skipped"
    assert not worker._scan_lock.locked()


def test_consolidation_runs_under_the_fleet_lock_and_releases_it() -> None:
    adapter = _store({"svc": [_entry("stale", entity_path="svc", age_days=60)]})
    worker = _worker(adapter, tenant_provider=lambda: ["t1", "t2"])
    lock_conn = MagicMock()
    names: list[str] = []

    def take(name):
        names.append(name)
        return lock_conn

    worker._acquire_fleet_lock = take
    worker._run_consolidation()

    assert names == ["amfs_cortex_consolidation"]
    lock_conn.close.assert_called_once()  # closing the session releases the advisory lock
    assert worker._consolidation_runs == 1
    assert worker.activity_log[-1]["type"] == "consolidation_run"
    assert worker.activity_log[-1]["tenants_scanned"] == 2
    assert worker.activity_log[-1]["auto_archived"] == 2
    assert not worker._scan_lock.locked()


def test_one_tenants_failure_does_not_cost_the_others_their_pass() -> None:
    calls: list[str | None] = []

    def tenant_search(q, branch="main"):
        from amfs_postgres.tenant_context import get_request_tenant_account_id
        tid = get_request_tenant_account_id()
        calls.append(tid)
        if tid == "bad":
            raise TimeoutError("canceling statement due to statement timeout")
        return [_entry("stale", entity_path="svc", age_days=60)]

    adapter = MagicMock()
    adapter.list_scopes.return_value = ({"svc"}, set())
    adapter.search.side_effect = tenant_search
    adapter.write.side_effect = lambda e: e
    adapter.list_branches.return_value = []

    worker = _worker(adapter, tenant_provider=lambda: ["bad", "good"])
    worker._acquire_fleet_lock = lambda name: MagicMock()
    worker._run_consolidation()

    assert calls == ["bad", "good"]
    assert worker.activity_log[-1]["type"] == "consolidation_run"
    assert worker.activity_log[-1]["auto_archived"] == 1


def test_failed_passes_advance_their_clocks() -> None:
    """Neither clock used to move on failure, so a broken pass re-ran every 5-second tick."""
    worker = _worker()
    worker._acquire_fleet_lock = lambda name: MagicMock()
    worker._compiler._adapter.list_scopes.side_effect = RuntimeError("pool exhausted")
    worker._tenant_provider = lambda: ["t1"]

    before = time.monotonic()
    worker._run_consolidation()
    assert worker._last_consolidation >= before

    def boom():
        raise RuntimeError("pool exhausted")

    worker._tenant_provider = boom
    worker._catchup_missing_digests()
    assert worker._last_catchup >= before


def test_maybe_consolidate_runs_the_pass_on_its_own_thread_once() -> None:
    worker = _worker(consolidation_interval_s=1.0)
    started = threading.Event()
    release = threading.Event()

    def pass_(self=worker):
        started.set()
        release.wait(2.0)

    worker._run_consolidation = pass_
    worker._maybe_consolidate()
    assert started.wait(1.0)
    first = worker._consolidation_thread
    worker._maybe_consolidate()  # the previous pass is still running: no second thread
    assert worker._consolidation_thread is first
    release.set()
    first.join(1.0)
    assert first.name == "cortex-consolidation"


def test_disabled_interval_never_consolidates() -> None:
    worker = _worker(consolidation_interval_s=0)
    worker._run_consolidation = MagicMock()
    worker._maybe_consolidate()
    worker._run_consolidation.assert_not_called()
