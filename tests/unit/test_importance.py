"""Unit tests for the importance evaluator hook (HMO Track 4)."""

from __future__ import annotations

from datetime import datetime, timezone

from amfs_core.importance import ImportanceEvaluator, NoOpEvaluator
from amfs_core.models import MemoryEntry, Provenance


class TestNoOpEvaluator:
    def test_returns_none_score(self) -> None:
        evaluator = NoOpEvaluator()
        score, dims = evaluator.evaluate(
            "some value",
            entity_path="svc",
            key="k",
        )
        assert score is None
        assert dims == {}

    def test_is_importanceevaluator(self) -> None:
        evaluator = NoOpEvaluator()
        assert isinstance(evaluator, ImportanceEvaluator)


class TestImportanceFieldOnEntry:
    def test_default_none(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        assert entry.importance_score is None
        assert entry.importance_dimensions is None

    def test_set_score_and_dimensions(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
            importance_score=0.85,
            importance_dimensions={
                "behavioral_alignment": 9.0,
                "reasoning_utility": 7.0,
                "contextual_persistence": 8.5,
            },
        )
        assert entry.importance_score == 0.85
        assert len(entry.importance_dimensions) == 3

    def test_serialization_roundtrip(self) -> None:
        entry = MemoryEntry(
            entity_path="svc",
            key="k",
            value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
            importance_score=0.7,
            importance_dimensions={"behavioral_alignment": 8.0},
        )
        data = entry.model_dump(mode="json")
        restored = MemoryEntry.model_validate(data)
        assert restored.importance_score == 0.7
        assert restored.importance_dimensions == {"behavioral_alignment": 8.0}


class TestImportanceInSDK:
    def test_write_with_evaluator(self, tmp_path) -> None:
        from amfs.memory import AgentMemory
        from amfs_filesystem.adapter import FilesystemAdapter

        class MockEvaluator(ImportanceEvaluator):
            def evaluate(self, value, *, entity_path, key, context=None):
                return 0.75, {"dim1": 7.5}

        adapter = FilesystemAdapter(root=tmp_path / "store")
        mem = AgentMemory(
            adapter=adapter,
            agent_id="test-agent",
            session_id="s1",
            importance_evaluator=MockEvaluator(),
        )

        mem.write("svc", "k1", "important memory")
        entry = adapter.read("svc", "k1")
        assert entry.importance_score == 0.75
        assert entry.importance_dimensions == {"dim1": 7.5}

    def test_write_without_evaluator(self, tmp_path) -> None:
        from amfs.memory import AgentMemory
        from amfs_filesystem.adapter import FilesystemAdapter

        adapter = FilesystemAdapter(root=tmp_path / "store")
        mem = AgentMemory(
            adapter=adapter,
            agent_id="test-agent",
            session_id="s1",
        )

        mem.write("svc", "k1", "normal memory")
        entry = adapter.read("svc", "k1")
        assert entry.importance_score is None

    def test_evaluator_failure_doesnt_block_write(self, tmp_path) -> None:
        from amfs.memory import AgentMemory
        from amfs_filesystem.adapter import FilesystemAdapter

        class FailingEvaluator(ImportanceEvaluator):
            def evaluate(self, value, *, entity_path, key, context=None):
                raise RuntimeError("LLM unavailable")

        adapter = FilesystemAdapter(root=tmp_path / "store")
        mem = AgentMemory(
            adapter=adapter,
            agent_id="test-agent",
            session_id="s1",
            importance_evaluator=FailingEvaluator(),
        )

        mem.write("svc", "k1", "should still write")
        entry = adapter.read("svc", "k1")
        assert entry is not None
        assert entry.importance_score is None
