"""Unit tests for OutcomeBackPropagator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    MemoryEntry,
    OutcomeRecord,
    OutcomeType,
    Provenance,
)
from amfs_core.outcome import OutcomeBackPropagator


# ---------------------------------------------------------------------------
# Mock adapter that implements commit_outcome properly
# ---------------------------------------------------------------------------

class MockAdapter(AdapterABC):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[MemoryEntry]] = {}

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
        return entry

    def list(
        self, entity_path: str | None = None, *, include_superseded: bool = False
    ) -> list[MemoryEntry]:
        result: list[MemoryEntry] = []
        for (ep, _), versions in self._store.items():
            if entity_path is not None and ep != entity_path:
                continue
            result.append(versions[-1])
        return result

    def watch(
        self, entity_path: str, callback: Callable[[MemoryEntry], None]
    ) -> WatchHandle:
        return WatchHandle(lambda: None)

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        multiplier = OUTCOME_MULTIPLIERS[record.outcome_type]
        updated: list[MemoryEntry] = []
        for spec in record.causal_entry_keys:
            parts = spec.rsplit("/", 1)
            if len(parts) != 2:
                continue
            ep, key = parts
            current = self.read(ep, key)
            if current is None:
                continue
            new_conf = current.confidence * multiplier * record.causal_confidence
            new_entry = current.model_copy(
                update={
                    "confidence": new_conf,
                    "outcome_count": current.outcome_count + 1,
                    "version": current.version + 1,
                }
            )
            self.write(new_entry)
            updated.append(new_entry)
        return updated


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_entry(adapter: MockAdapter, ep: str = "svc", key: str = "k", conf: float = 1.0) -> None:
    adapter.write(
        MemoryEntry(
            entity_path=ep,
            key=key,
            value="data",
            provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
            confidence=conf,
        )
    )


class TestOutcomeBackPropagator:
    def test_propagate_p1(self) -> None:
        adapter = MockAdapter()
        _seed_entry(adapter, "svc", "k", conf=1.0)

        prop = OutcomeBackPropagator(adapter)
        record = prop.make_record(
            "INC-001", OutcomeType.P1_INCIDENT, ["svc/k"], "release-agent"
        )
        updated = prop.propagate(record)

        assert len(updated) == 1
        assert updated[0].confidence == 1.0 * 1.15
        assert updated[0].outcome_count == 1

    def test_propagate_clean_deploy(self) -> None:
        adapter = MockAdapter()
        _seed_entry(adapter, "svc", "k", conf=1.0)

        prop = OutcomeBackPropagator(adapter)
        record = prop.make_record(
            "DEP-001", OutcomeType.CLEAN_DEPLOY, ["svc/k"], "release-agent"
        )
        updated = prop.propagate(record)

        assert len(updated) == 1
        assert abs(updated[0].confidence - 0.97) < 1e-6

    def test_propagate_with_causal_confidence(self) -> None:
        adapter = MockAdapter()
        _seed_entry(adapter, "svc", "k", conf=1.0)

        prop = OutcomeBackPropagator(adapter)
        record = prop.make_record(
            "INC-002",
            OutcomeType.P2_INCIDENT,
            ["svc/k"],
            "release-agent",
            causal_confidence=0.8,
        )
        updated = prop.propagate(record)

        expected = 1.0 * 1.10 * 0.8
        assert abs(updated[0].confidence - expected) < 1e-6

    def test_propagate_multiple_entries(self) -> None:
        adapter = MockAdapter()
        _seed_entry(adapter, "svc-a", "k1")
        _seed_entry(adapter, "svc-b", "k2")

        prop = OutcomeBackPropagator(adapter)
        record = prop.make_record(
            "INC-003",
            OutcomeType.REGRESSION,
            ["svc-a/k1", "svc-b/k2"],
            "release-agent",
        )
        updated = prop.propagate(record)

        assert len(updated) == 2
        for entry in updated:
            assert abs(entry.confidence - 1.08) < 1e-6

    def test_propagate_missing_entry_skipped(self) -> None:
        adapter = MockAdapter()
        prop = OutcomeBackPropagator(adapter)
        record = prop.make_record(
            "INC-004", OutcomeType.P1_INCIDENT, ["nope/nope"], "agent"
        )
        updated = prop.propagate(record)
        assert updated == []

    def test_propagate_batch(self) -> None:
        adapter = MockAdapter()
        _seed_entry(adapter, "svc", "k")

        prop = OutcomeBackPropagator(adapter)
        records = [
            prop.make_record("INC-A", OutcomeType.P1_INCIDENT, ["svc/k"], "agent"),
            prop.make_record("INC-B", OutcomeType.P2_INCIDENT, ["svc/k"], "agent"),
        ]
        all_updated = prop.propagate_batch(records)
        assert len(all_updated) == 2
        # After P1: 1.0 * 1.15 = 1.15
        # After P2: 1.15 * 1.10 = 1.265
        assert abs(all_updated[1].confidence - 1.265) < 1e-6

    def test_compute_new_confidence(self) -> None:
        result = OutcomeBackPropagator.compute_new_confidence(
            1.0, OutcomeType.P1_INCIDENT, 0.9
        )
        assert abs(result - (1.0 * 1.15 * 0.9)) < 1e-6

    def test_make_record(self) -> None:
        record = OutcomeBackPropagator.make_record(
            "INC-X", OutcomeType.REGRESSION, ["svc/k"], "agent"
        )
        assert record.outcome_ref == "INC-X"
        assert record.outcome_type == OutcomeType.REGRESSION
        assert record.causal_confidence == 1.0
