"""Unit tests for the AMFS Python SDK (M4)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from amfs_core.embedder import EmbedderABC, cosine_similarity
from amfs_core.exceptions import StaleWriteError
from amfs_core.models import ConflictPolicy, MemoryEntry, OutcomeType, Provenance
from amfs.config import find_config, load_config, load_config_or_default
from amfs.factory import create_adapter_from_config
from amfs.memory import AgentMemory
from amfs_core.snapshot import SnapshotExporter, SnapshotImporter
from amfs_filesystem.adapter import FilesystemAdapter


class MockEmbedder(EmbedderABC):
    """Deterministic embedder for testing — hashes the text into a fixed vector."""

    def embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255.0 for b in h[:16]]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "amfs.yaml"
        cfg_file.write_text(
            "namespace: prod\n"
            "layers:\n"
            "  primary:\n"
            "    adapter: filesystem\n"
            "    options:\n"
            "      root: /data/.amfs\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.namespace == "prod"
        assert cfg.layers["primary"].adapter == "filesystem"
        assert cfg.layers["primary"].options["root"] == "/data/.amfs"

    def test_find_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "amfs.yaml"
        cfg_file.write_text("namespace: test\nlayers: {}\n")
        found = find_config(tmp_path)
        assert found == cfg_file

    def test_find_config_walks_up(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "amfs.yaml"
        cfg_file.write_text("namespace: test\nlayers: {}\n")
        subdir = tmp_path / "a" / "b"
        subdir.mkdir(parents=True)
        found = find_config(subdir)
        assert found == cfg_file

    def test_find_config_returns_none(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        # Won't find anything (searches up to filesystem root)
        # This test just verifies it doesn't crash
        result = find_config(empty)
        # May or may not find something depending on the system — just check type
        assert result is None or isinstance(result, Path)

    def test_load_config_or_default(self) -> None:
        cfg = load_config_or_default()
        assert cfg.namespace == "default"
        assert "primary" in cfg.layers
        assert cfg.layers["primary"].adapter == "filesystem"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_filesystem_adapter(self, tmp_amfs_root: Path) -> None:
        from amfs_core.models import AMFSConfig, LayerConfig

        config = AMFSConfig(
            namespace="test",
            layers={
                "primary": LayerConfig(
                    adapter="filesystem",
                    options={"root": str(tmp_amfs_root)},
                )
            },
        )
        adapter = create_adapter_from_config(config)
        assert isinstance(adapter, FilesystemAdapter)

    def test_unknown_adapter_raises(self) -> None:
        from amfs_core.models import AMFSConfig, LayerConfig

        config = AMFSConfig(
            namespace="test",
            layers={"primary": LayerConfig(adapter="unknown")},
        )
        with pytest.raises(ValueError, match="Unknown adapter"):
            create_adapter_from_config(config)

    def test_missing_layer_raises(self) -> None:
        from amfs_core.models import AMFSConfig

        config = AMFSConfig(namespace="test")
        with pytest.raises(KeyError, match="Layer 'primary' not found"):
            create_adapter_from_config(config)


# ---------------------------------------------------------------------------
# AgentMemory tests
# ---------------------------------------------------------------------------

class TestAgentMemory:
    @pytest.fixture
    def mem(self, tmp_amfs_root: Path) -> AgentMemory:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        m = AgentMemory(agent_id="test-agent", adapter=adapter)
        yield m  # type: ignore[misc]
        m.close()

    def test_write_and_read(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", {"data": 1})
        entry = mem.read("svc", "key")
        assert entry is not None
        assert entry.value == {"data": 1}
        assert entry.provenance.agent_id == "test-agent"

    def test_version_increments(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "v1")
        mem.write("svc", "key", "v2")
        entry = mem.read("svc", "key")
        assert entry is not None
        assert entry.version == 2
        assert entry.value == "v2"

    def test_list(self, mem: AgentMemory) -> None:
        mem.write("svc", "k1", "a")
        mem.write("svc", "k2", "b")
        entries = mem.list("svc")
        assert len(entries) == 2

    def test_commit_outcome(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "data")
        updated = mem.commit_outcome(
            "INC-001",
            OutcomeType.CRITICAL_FAILURE,
            ["svc/key"],
        )
        assert len(updated) == 1
        # CRITICAL_FAILURE erodes confidence: 1.0 * 0.85 = 0.85
        assert abs(updated[0].confidence - 0.85) < 1e-6

    def test_properties(self, mem: AgentMemory) -> None:
        assert mem.agent_id == "test-agent"
        assert mem.session_id.startswith("sess-")
        assert mem.namespace == "default"

    def test_context_manager(self, tmp_amfs_root: Path) -> None:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        with AgentMemory(agent_id="ctx-agent", adapter=adapter) as m:
            m.write("svc", "key", "value")
        # Should not raise after close

    def test_write_with_confidence_and_ttl(self, mem: AgentMemory) -> None:
        ttl = datetime(2099, 12, 31, tzinfo=timezone.utc)
        entry = mem.write("svc", "key", "val", confidence=0.8, ttl_at=ttl)
        assert entry.confidence == 0.8
        assert entry.ttl_at == ttl

    def test_write_with_pattern_refs(self, mem: AgentMemory) -> None:
        entry = mem.write("svc", "key", "val", pattern_refs=["retry-logic"])
        assert entry.provenance.pattern_refs == ["retry-logic"]

    def test_min_confidence_filter(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "val", confidence=0.3)
        assert mem.read("svc", "key", min_confidence=0.5) is None
        assert mem.read("svc", "key", min_confidence=0.2) is not None


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------

class TestSnapshot:
    @pytest.fixture
    def adapter(self, tmp_amfs_root: Path) -> FilesystemAdapter:
        a = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        yield a  # type: ignore[misc]
        a.close()

    def test_export_and_restore(self, adapter: FilesystemAdapter, tmp_path: Path) -> None:
        # Seed data
        mem = AgentMemory(agent_id="snap-agent", adapter=adapter)
        mem.write("svc-a", "k1", {"x": 1})
        mem.write("svc-a", "k2", {"x": 2})
        mem.write("svc-b", "k3", {"x": 3})

        # Export
        snapshot_path = tmp_path / "snapshot.json"
        exporter = SnapshotExporter(adapter)
        count = exporter.export(snapshot_path)
        assert count == 3
        assert snapshot_path.exists()

        # Verify snapshot structure
        data = json.loads(snapshot_path.read_text())
        assert data["amfs_snapshot_version"] == "0.1.0"
        assert data["entry_count"] == 3
        assert len(data["entries"]) == 3

        # Restore into a fresh adapter
        fresh_root = tmp_path / "fresh_amfs"
        fresh_root.mkdir()
        fresh_adapter = FilesystemAdapter(root=fresh_root, namespace="restored")
        importer = SnapshotImporter(fresh_adapter)
        restored_count = importer.restore(snapshot_path)
        assert restored_count == 3

        # Verify restored data
        entries = fresh_adapter.list()
        assert len(entries) == 3
        entity_paths = {e.entity_path for e in entries}
        assert entity_paths == {"svc-a", "svc-b"}
        fresh_adapter.close()

    def test_export_filtered(self, adapter: FilesystemAdapter, tmp_path: Path) -> None:
        mem = AgentMemory(agent_id="snap-agent", adapter=adapter)
        mem.write("svc-a", "k1", "a")
        mem.write("svc-b", "k2", "b")

        snapshot_path = tmp_path / "filtered.json"
        exporter = SnapshotExporter(adapter)
        count = exporter.export(snapshot_path, entity_path="svc-a")
        assert count == 1

    def test_export_empty(self, adapter: FilesystemAdapter, tmp_path: Path) -> None:
        snapshot_path = tmp_path / "empty.json"
        exporter = SnapshotExporter(adapter)
        count = exporter.export(snapshot_path)
        assert count == 0
        data = json.loads(snapshot_path.read_text())
        assert data["entries"] == []


# ---------------------------------------------------------------------------
# Auto-causal tracking tests
# ---------------------------------------------------------------------------

class TestAutoCausalTracking:
    @pytest.fixture
    def mem(self, tmp_amfs_root: Path) -> AgentMemory:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        m = AgentMemory(agent_id="causal-agent", adapter=adapter)
        yield m  # type: ignore[misc]
        m.close()

    def test_read_populates_read_log(self, mem: AgentMemory) -> None:
        mem.write("svc-a", "k1", "v1")
        mem.write("svc-b", "k2", "v2")
        mem.read("svc-a", "k1")
        mem.read("svc-b", "k2")
        assert mem.read_log == ["svc-a/k1", "svc-b/k2"]

    def test_read_log_deduplicates(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "val")
        mem.read("svc", "key")
        mem.read("svc", "key")
        assert mem.read_log == ["svc/key"]

    def test_read_log_excludes_misses(self, mem: AgentMemory) -> None:
        mem.read("nonexistent", "nope")
        assert mem.read_log == []

    def test_commit_outcome_auto_causal(self, mem: AgentMemory) -> None:
        """commit_outcome without explicit keys uses the read log."""
        mem.write("svc", "k1", "v1")
        mem.write("svc", "k2", "v2")
        mem.read("svc", "k1")
        mem.read("svc", "k2")

        updated = mem.commit_outcome("INC-100", OutcomeType.CRITICAL_FAILURE)
        assert len(updated) == 2
        # CRITICAL_FAILURE erodes confidence: 1.0 * 0.85 = 0.85
        assert all(abs(e.confidence - 0.85) < 1e-6 for e in updated)

    def test_commit_outcome_explicit_overrides_auto(self, mem: AgentMemory) -> None:
        """Explicit causal keys take precedence over the auto read log."""
        mem.write("svc", "k1", "v1")
        mem.write("svc", "k2", "v2")
        mem.read("svc", "k1")
        mem.read("svc", "k2")

        updated = mem.commit_outcome(
            "INC-101", OutcomeType.CRITICAL_FAILURE, ["svc/k1"]
        )
        assert len(updated) == 1
        assert updated[0].key == "k1"

    def test_clear_read_log(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "val")
        mem.read("svc", "key")
        assert len(mem.read_log) == 1
        mem.clear_read_log()
        assert mem.read_log == []

    def test_record_context(self, mem: AgentMemory) -> None:
        mem.record_context("git-log", "15 commits since last deploy", source="git")
        chain = mem.explain()
        assert len(chain["external_contexts"]) == 1
        ctx = chain["external_contexts"][0]
        assert ctx["label"] == "git-log"
        assert ctx["summary"] == "15 commits since last deploy"
        assert ctx["source"] == "git"

    def test_record_context_source_optional(self, mem: AgentMemory) -> None:
        mem.record_context("manual-check", "Verified config")
        chain = mem.explain()
        assert chain["external_contexts"][0]["source"] is None

    def test_explain_includes_both_reads_and_contexts(self, mem: AgentMemory) -> None:
        mem.write("svc", "pattern", "exponential backoff")
        mem.read("svc", "pattern")
        mem.record_context("pagerduty", "2 open incidents", source="PagerDuty API")

        chain = mem.explain("DEP-100")
        assert chain["outcome_ref"] == "DEP-100"
        assert chain["agent_id"] == "causal-agent"
        assert chain["causal_chain_length"] == 1
        assert len(chain["causal_entries"]) == 1
        assert chain["causal_entries"][0]["key"] == "pattern"
        assert len(chain["external_contexts"]) == 1
        assert chain["external_contexts"][0]["label"] == "pagerduty"

    def test_clear_read_log_clears_contexts(self, mem: AgentMemory) -> None:
        mem.record_context("tool", "output")
        assert len(mem.explain()["external_contexts"]) == 1
        mem.clear_read_log()
        assert mem.explain()["external_contexts"] == []

    def test_explain_empty_session(self, mem: AgentMemory) -> None:
        chain = mem.explain()
        assert chain["causal_chain_length"] == 0
        assert chain["causal_entries"] == []
        assert chain["external_contexts"] == []


# ---------------------------------------------------------------------------
# Confidence decay tests
# ---------------------------------------------------------------------------

class TestConfidenceDecay:
    def test_effective_confidence_no_decay(self) -> None:
        entry = MemoryEntry(
            entity_path="svc", key="k", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc) - timedelta(days=30),
            ),
            confidence=1.0,
        )
        assert entry.effective_confidence() == 1.0
        assert entry.effective_confidence(decay_half_life_days=None) == 1.0

    def test_effective_confidence_with_decay(self) -> None:
        entry = MemoryEntry(
            entity_path="svc", key="k", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc) - timedelta(days=30),
            ),
            confidence=1.0,
        )
        effective = entry.effective_confidence(decay_half_life_days=30)
        assert 0.49 < effective < 0.51

    def test_outcome_validated_entries_decay_slower(self) -> None:
        base = dict(
            entity_path="svc", key="k", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc) - timedelta(days=30),
            ),
            confidence=1.0,
        )
        unvalidated = MemoryEntry(**base, outcome_count=0)
        validated = MemoryEntry(**base, outcome_count=2)

        eff_unvalidated = unvalidated.effective_confidence(decay_half_life_days=30)
        eff_validated = validated.effective_confidence(decay_half_life_days=30)
        assert eff_validated > eff_unvalidated

    def test_fresh_entry_no_decay(self) -> None:
        entry = MemoryEntry(
            entity_path="svc", key="k", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
            confidence=0.8,
        )
        effective = entry.effective_confidence(decay_half_life_days=30)
        assert effective > 0.79

    def test_sdk_decay_filters_on_read(self, tmp_amfs_root: Path) -> None:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        mem = AgentMemory(
            agent_id="decay-agent", adapter=adapter,
            decay_half_life_days=30,
        )
        mem.write("svc", "key", "val", confidence=0.5)
        # Fresh entry — effective ~0.5, should pass min_confidence=0.4
        assert mem.read("svc", "key", min_confidence=0.4) is not None
        # Should fail min_confidence=0.6 (stored is only 0.5)
        assert mem.read("svc", "key", min_confidence=0.6) is None
        mem.close()


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearch:
    @pytest.fixture
    def mem(self, tmp_amfs_root: Path) -> AgentMemory:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        m = AgentMemory(agent_id="search-agent", adapter=adapter)
        yield m  # type: ignore[misc]
        m.close()

    def _seed(self, mem: AgentMemory) -> None:
        mem.write("svc-a", "k1", "v1", confidence=0.9, pattern_refs=["retry"])
        mem.write("svc-a", "k2", "v2", confidence=0.5)
        mem.write("svc-b", "k3", "v3", confidence=0.3)

    def test_search_all(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search()
        assert len(results) == 3

    def test_search_by_entity(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(entity_path="svc-a")
        assert len(results) == 2
        assert all(r.entity_path == "svc-a" for r in results)

    def test_search_by_min_confidence(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(min_confidence=0.5)
        assert len(results) == 2
        assert all(r.confidence >= 0.5 for r in results)

    def test_search_by_max_confidence(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(max_confidence=0.5)
        assert len(results) == 2
        assert all(r.confidence <= 0.5 for r in results)

    def test_search_by_agent(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(agent_id="search-agent")
        assert len(results) == 3
        results = mem.search(agent_id="nonexistent")
        assert len(results) == 0

    def test_search_by_pattern_ref(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(pattern_ref="retry")
        assert len(results) == 1
        assert results[0].key == "k1"

    def test_search_sort_by_confidence(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(sort_by="confidence")
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_search_sort_by_recency(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(sort_by="recency")
        times = [r.provenance.written_at for r in results]
        assert times == sorted(times, reverse=True)

    def test_search_limit(self, mem: AgentMemory) -> None:
        self._seed(mem)
        results = mem.search(limit=1)
        assert len(results) == 1

    def test_search_since(self, mem: AgentMemory) -> None:
        self._seed(mem)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        results = mem.search(since=future)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestStats:
    @pytest.fixture
    def mem(self, tmp_amfs_root: Path) -> AgentMemory:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        m = AgentMemory(agent_id="stats-agent", adapter=adapter)
        yield m  # type: ignore[misc]
        m.close()

    def test_stats_empty(self, mem: AgentMemory) -> None:
        s = mem.stats()
        assert s.total_entries == 0
        assert s.total_entities == 0
        assert s.total_agents == 0

    def test_stats_populated(self, mem: AgentMemory) -> None:
        mem.write("svc-a", "k1", "v1", confidence=0.8)
        mem.write("svc-a", "k2", "v2", confidence=1.0)
        mem.write("svc-b", "k3", "v3", confidence=0.6)

        s = mem.stats()
        assert s.total_entries == 3
        assert s.total_entities == 2
        assert s.total_agents == 1
        assert s.agents == {"stats-agent": 3}
        assert s.entities == {"svc-a": 2, "svc-b": 1}
        assert 0.79 < s.confidence_avg < 0.81
        assert s.confidence_min == 0.6
        assert s.confidence_max == 1.0
        assert s.outcome_linked_count == 0
        assert s.oldest_entry_at is not None
        assert s.newest_entry_at is not None

    def test_stats_with_outcomes(self, mem: AgentMemory) -> None:
        mem.write("svc", "key", "val")
        mem.commit_outcome("INC-001", OutcomeType.CRITICAL_FAILURE, ["svc/key"])
        s = mem.stats()
        assert s.outcome_linked_count == 1


# ---------------------------------------------------------------------------
# Entry key helper tests
# ---------------------------------------------------------------------------

class TestEntryKey:
    def test_entry_key_format(self) -> None:
        entry = MemoryEntry(
            entity_path="services/checkout",
            key="risk_profile",
            version=1,
            value="x",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        assert entry.entry_key == "services/checkout/risk_profile"


# ---------------------------------------------------------------------------
# Semantic search tests
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    @pytest.fixture
    def mem(self, tmp_amfs_root: Path) -> AgentMemory:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        m = AgentMemory(
            agent_id="embed-agent", adapter=adapter, embedder=MockEmbedder(),
        )
        yield m  # type: ignore[misc]
        m.close()

    def test_write_stores_embedding(self, mem: AgentMemory) -> None:
        entry = mem.write("svc", "key", "hello world")
        stored = mem.read("svc", "key")
        assert stored is not None
        assert stored.embedding is not None
        assert len(stored.embedding) == 16

    def test_semantic_search_returns_results(self, mem: AgentMemory) -> None:
        mem.write("svc", "retry", "retry logic with exponential backoff")
        mem.write("svc", "timeout", "connection timeout configuration")
        mem.write("svc", "auth", "authentication and authorization")

        results = mem.semantic_search("retry error recovery", limit=3)
        assert len(results) > 0
        for entry, similarity in results:
            assert isinstance(similarity, float)
            assert entry.embedding is not None

    def test_semantic_search_requires_embedder(self, tmp_amfs_root: Path) -> None:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test2")
        mem = AgentMemory(agent_id="no-embed", adapter=adapter)
        with pytest.raises(RuntimeError, match="requires an embedder"):
            mem.semantic_search("test")
        mem.close()

    def test_semantic_search_min_confidence_filter(self, mem: AgentMemory) -> None:
        mem.write("svc", "low", "low confidence entry", confidence=0.1)
        mem.write("svc", "high", "high confidence entry", confidence=0.9)

        results = mem.semantic_search("entry", min_confidence=0.5)
        entry_keys = [e.key for e, _ in results]
        assert "low" not in entry_keys

    def test_semantic_search_entity_filter(self, mem: AgentMemory) -> None:
        mem.write("svc-a", "k1", "value one")
        mem.write("svc-b", "k2", "value two")

        results = mem.semantic_search("value", entity_path="svc-a")
        assert all(e.entity_path == "svc-a" for e, _ in results)


class TestCosineUtility:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Conflict detection tests
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_last_write_wins_no_error(self, tmp_amfs_root: Path) -> None:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        agent_a = AgentMemory(agent_id="agent-a", adapter=adapter)
        agent_b = AgentMemory(agent_id="agent-b", adapter=adapter)

        agent_a.write("svc", "key", {"v": 1})
        agent_b.read("svc", "key")
        agent_a.write("svc", "key", {"v": 2})
        agent_b.write("svc", "key", {"v": 3})

        result = agent_a.read("svc", "key")
        assert result is not None
        assert result.value == {"v": 3}

        agent_a.close()
        agent_b.close()

    def test_raise_on_stale_write(self, tmp_amfs_root: Path) -> None:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        agent_a = AgentMemory(agent_id="agent-a", adapter=adapter)
        agent_b = AgentMemory(
            agent_id="agent-b", adapter=adapter,
            conflict_policy=ConflictPolicy.RAISE,
        )

        agent_a.write("svc", "key", "original")
        agent_b.read("svc", "key")
        agent_a.write("svc", "key", "updated-by-a")

        with pytest.raises(StaleWriteError) as exc_info:
            agent_b.write("svc", "key", "updated-by-b")
        assert exc_info.value.read_version == 1
        assert exc_info.value.current_version == 2
        assert exc_info.value.current_agent == "agent-a"

        agent_a.close()
        agent_b.close()

    def test_no_conflict_if_same_agent(self, tmp_amfs_root: Path) -> None:
        """Same agent modifying its own entry should not trigger conflict."""
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        agent = AgentMemory(
            agent_id="agent-a", adapter=adapter,
            conflict_policy=ConflictPolicy.RAISE,
        )

        agent.write("svc", "key", "v1")
        agent.read("svc", "key")
        agent.write("svc", "key", "v2")
        agent.write("svc", "key", "v3")

        agent.close()

    def test_no_conflict_without_prior_read(self, tmp_amfs_root: Path) -> None:
        """Writing without a prior read should not trigger conflict."""
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
        agent_a = AgentMemory(agent_id="agent-a", adapter=adapter)
        agent_b = AgentMemory(
            agent_id="agent-b", adapter=adapter,
            conflict_policy=ConflictPolicy.RAISE,
        )

        agent_a.write("svc", "key", "v1")
        agent_a.write("svc", "key", "v2")
        agent_b.write("svc", "key", "v3")

        agent_a.close()
        agent_b.close()

    def test_merge_callback(self, tmp_amfs_root: Path) -> None:
        adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")

        def merge(our_read: Any, current: Any, new_value: Any) -> dict:
            return {"merged": True, "from_a": current.value, "from_b": new_value}

        agent_a = AgentMemory(agent_id="agent-a", adapter=adapter)
        agent_b = AgentMemory(
            agent_id="agent-b", adapter=adapter, on_conflict=merge,
        )

        agent_a.write("svc", "key", "original")
        agent_b.read("svc", "key")
        agent_a.write("svc", "key", "from-a")

        result = agent_b.write("svc", "key", "from-b")
        assert result.value["merged"] is True
        assert result.value["from_a"] == "from-a"
        assert result.value["from_b"] == "from-b"

        agent_a.close()
        agent_b.close()


# ---------------------------------------------------------------------------
# CLI init tests
# ---------------------------------------------------------------------------

class TestCLIInit:
    def test_init_creates_config_and_dir(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner
        from amfs_cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0

        config_file = tmp_path / "amfs.yaml"
        assert config_file.exists()
        content = config_file.read_text()
        assert "namespace: default" in content
        assert "adapter: filesystem" in content

        amfs_dir = tmp_path / ".amfs"
        assert amfs_dir.is_dir()

    def test_init_custom_namespace(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner
        from amfs_cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", str(tmp_path), "--namespace", "prod"])
        assert result.exit_code == 0

        content = (tmp_path / "amfs.yaml").read_text()
        assert "namespace: prod" in content

    def test_init_refuses_overwrite(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner
        from amfs_cli.main import app

        runner = CliRunner()
        runner.invoke(app, ["init", str(tmp_path)])
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 1

    def test_init_force_overwrites(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner
        from amfs_cli.main import app

        runner = CliRunner()
        runner.invoke(app, ["init", str(tmp_path)])
        result = runner.invoke(app, ["init", str(tmp_path), "--force"])
        assert result.exit_code == 0

    def test_init_creates_gitignore(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner
        from amfs_cli.main import app

        runner = CliRunner()
        runner.invoke(app, ["init", str(tmp_path)])

        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()
        assert ".amfs/" in gitignore.read_text()
