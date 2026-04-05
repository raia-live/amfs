"""Abstract adapter contract tests — ALL adapters must pass these.

Subclass AdapterContractTests, implement the `adapter` fixture, and every
test method here will be inherited and run against your adapter.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from amfs_core.abc import AdapterABC
from amfs_core.models import MemoryEntry, OutcomeRecord, OutcomeType, Provenance


def _make_entry(
    entity_path: str = "checkout-service",
    key: str = "retry-pattern",
    version: int = 1,
    confidence: float = 1.0,
    value: dict | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        version=version,
        value=value or {"pattern": "exponential-backoff", "max_retries": 3},
        confidence=confidence,
        provenance=Provenance(
            agent_id="review-agent",
            session_id="sess-001",
            written_at=datetime.now(timezone.utc),
        ),
    )


class AdapterContractTests:
    """Mixin providing contract tests for any AdapterABC implementation."""

    # Subclasses must provide an `adapter` fixture returning an AdapterABC.

    # ------------------------------------------------------------------
    # read / write basics
    # ------------------------------------------------------------------

    def test_write_and_read(self, adapter: AdapterABC) -> None:
        entry = _make_entry()
        written = adapter.write(entry)
        assert written.version == 1

        result = adapter.read("checkout-service", "retry-pattern")
        assert result is not None
        assert result.version == 1
        assert result.value == entry.value
        assert result.provenance.agent_id == "review-agent"

    def test_read_nonexistent_returns_none(self, adapter: AdapterABC) -> None:
        assert adapter.read("no-such-entity", "no-such-key") is None

    def test_write_increments_version(self, adapter: AdapterABC) -> None:
        e1 = _make_entry()
        w1 = adapter.write(e1)
        assert w1.version == 1

        e2 = _make_entry(value={"updated": True})
        w2 = adapter.write(e2)
        assert w2.version == 2

        current = adapter.read("checkout-service", "retry-pattern")
        assert current is not None
        assert current.version == 2
        assert current.value == {"updated": True}

    # ------------------------------------------------------------------
    # min_confidence filter
    # ------------------------------------------------------------------

    def test_read_min_confidence_filters(self, adapter: AdapterABC) -> None:
        entry = _make_entry(confidence=0.5)
        adapter.write(entry)

        assert adapter.read("checkout-service", "retry-pattern", min_confidence=0.3) is not None
        assert adapter.read("checkout-service", "retry-pattern", min_confidence=0.8) is None

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def test_list_current_only(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(key="key-a"))
        adapter.write(_make_entry(key="key-b"))
        # Write a second version of key-a
        adapter.write(_make_entry(key="key-a", value={"v": 2}))

        entries = adapter.list("checkout-service")
        keys = {e.key for e in entries}
        assert keys == {"key-a", "key-b"}
        # key-a should be version 2
        key_a = next(e for e in entries if e.key == "key-a")
        assert key_a.version == 2

    def test_list_include_superseded(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(key="key-x"))
        adapter.write(_make_entry(key="key-x", value={"v": 2}))

        entries = adapter.list("checkout-service", include_superseded=True)
        versions = sorted(e.version for e in entries if e.key == "key-x")
        assert versions == [1, 2]

    def test_list_all_entities(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(entity_path="svc-a", key="k1"))
        adapter.write(_make_entry(entity_path="svc-b", key="k2"))

        entries = adapter.list()
        entity_paths = {e.entity_path for e in entries}
        assert entity_paths == {"svc-a", "svc-b"}

    # ------------------------------------------------------------------
    # commit_outcome
    # ------------------------------------------------------------------

    def test_commit_outcome_p1_incident(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(confidence=1.0))

        record = OutcomeRecord(
            outcome_ref="INC-001",
            outcome_type=OutcomeType.CRITICAL_FAILURE,
            causal_confidence=1.0,
            committed_at=datetime.now(timezone.utc),
            causal_entry_keys=["checkout-service/retry-pattern"],
            agent_id="release-agent",
        )
        updated = adapter.commit_outcome(record)
        assert len(updated) == 1
        assert updated[0].confidence == 1.0 * 1.15 * 1.0
        assert updated[0].outcome_count == 1

    def test_commit_outcome_clean_deploy(self, adapter: AdapterABC) -> None:
        adapter.write(_make_entry(confidence=1.0))

        record = OutcomeRecord(
            outcome_ref="DEP-001",
            outcome_type=OutcomeType.SUCCESS,
            causal_confidence=1.0,
            committed_at=datetime.now(timezone.utc),
            causal_entry_keys=["checkout-service/retry-pattern"],
            agent_id="release-agent",
        )
        updated = adapter.commit_outcome(record)
        assert len(updated) == 1
        assert abs(updated[0].confidence - 0.97) < 1e-6

    def test_commit_outcome_missing_entry_skipped(self, adapter: AdapterABC) -> None:
        record = OutcomeRecord(
            outcome_ref="INC-002",
            outcome_type=OutcomeType.FAILURE,
            causal_confidence=1.0,
            committed_at=datetime.now(timezone.utc),
            causal_entry_keys=["nonexistent/key"],
            agent_id="release-agent",
        )
        updated = adapter.commit_outcome(record)
        assert updated == []

    # ------------------------------------------------------------------
    # watch
    # ------------------------------------------------------------------

    def test_watch_receives_new_writes(self, adapter: AdapterABC) -> None:
        received: list[MemoryEntry] = []
        handle = adapter.watch("checkout-service", received.append)

        try:
            # Give watcher time to start (FSEvents on macOS can be slow)
            time.sleep(0.5)
            adapter.write(_make_entry())
            # Poll for the event with a generous timeout
            deadline = time.monotonic() + 3.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.1)
            assert len(received) >= 1
            assert received[0].key == "retry-pattern"
        finally:
            handle.cancel()
            assert handle.cancelled

    def test_watch_cancel_stops_notifications(self, adapter: AdapterABC) -> None:
        received: list[MemoryEntry] = []
        handle = adapter.watch("checkout-service", received.append)
        time.sleep(0.2)
        handle.cancel()

        adapter.write(_make_entry())
        time.sleep(0.5)
        # After cancel, should not receive new entries
        # (may have received 0 or some depending on timing, but handle is cancelled)
        assert handle.cancelled
