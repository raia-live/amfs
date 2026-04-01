"""Unit tests for AMFS core models."""

from __future__ import annotations

from datetime import datetime, timezone

from amfs_core.models import (
    AMFSConfig,
    LayerConfig,
    MemoryEntry,
    MemoryType,
    OutcomeRecord,
    OutcomeType,
    OUTCOME_MULTIPLIERS,
    Provenance,
    ProvenanceTier,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestProvenance:
    def test_creation(self) -> None:
        p = Provenance(agent_id="a1", session_id="s1", written_at=_now())
        assert p.agent_id == "a1"
        assert p.pattern_refs == []

    def test_with_pattern_refs(self) -> None:
        p = Provenance(
            agent_id="a1",
            session_id="s1",
            written_at=_now(),
            pattern_refs=["retry-logic", "timeout-constants"],
        )
        assert len(p.pattern_refs) == 2


class TestMemoryEntry:
    def test_defaults(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value={"x": 1},
            provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
        )
        assert entry.version == 1
        assert entry.confidence == 1.0
        assert entry.outcome_count == 0
        assert entry.ttl_at is None
        assert entry.amfs_version == "0.2.0"
        assert entry.memory_type == MemoryType.FACT

    def test_memory_type_belief(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="hypothesis",
            provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
            memory_type=MemoryType.BELIEF,
        )
        assert entry.memory_type == MemoryType.BELIEF

    def test_serialization_roundtrip(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value={"nested": {"data": [1, 2, 3]}},
            provenance=Provenance(agent_id="a", session_id="s", written_at=_now()),
            memory_type=MemoryType.EXPERIENCE,
        )
        data = entry.model_dump(mode="json")
        restored = MemoryEntry.model_validate(data)
        assert restored.value == entry.value
        assert restored.entity_path == entry.entity_path
        assert restored.memory_type == MemoryType.EXPERIENCE

    def test_provenance_tier_production_validated(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="x",
            provenance=Provenance(agent_id="agent/review", session_id="s", written_at=_now()),
            outcome_count=2,
        )
        assert entry.provenance_tier == ProvenanceTier.PRODUCTION_VALIDATED

    def test_provenance_tier_production_observed(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="x",
            provenance=Provenance(agent_id="prod/deploy", session_id="s", written_at=_now()),
            outcome_count=0,
        )
        assert entry.provenance_tier == ProvenanceTier.PRODUCTION_OBSERVED

    def test_provenance_tier_development(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="x",
            provenance=Provenance(agent_id="dev/test-runner", session_id="s", written_at=_now()),
        )
        assert entry.provenance_tier == ProvenanceTier.DEVELOPMENT

    def test_provenance_tier_manual(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="x",
            provenance=Provenance(agent_id="manual/seed", session_id="s", written_at=_now()),
        )
        assert entry.provenance_tier == ProvenanceTier.MANUAL


class TestOutcomeRecord:
    def test_creation(self) -> None:
        record = OutcomeRecord(
            outcome_ref="INC-001",
            outcome_type=OutcomeType.P1_INCIDENT,
            committed_at=_now(),
            causal_entry_keys=["svc/key"],
            agent_id="release-agent",
        )
        assert record.causal_confidence == 1.0

    def test_all_outcome_types_have_multipliers(self) -> None:
        for ot in OutcomeType:
            assert ot in OUTCOME_MULTIPLIERS


class TestAMFSConfig:
    def test_creation(self) -> None:
        cfg = AMFSConfig(
            namespace="prod",
            layers={
                "primary": LayerConfig(
                    adapter="filesystem",
                    options={"root": "/data/.amfs"},
                )
            },
        )
        assert cfg.namespace == "prod"
        assert cfg.layers["primary"].adapter == "filesystem"
