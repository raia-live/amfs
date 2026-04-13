"""Unit tests for CoWEngine, CausalTagger, and ReadTracker."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.engine import CausalTagger, CoWEngine, ReadTracker
from amfs_core.models import MemoryEntry, OutcomeRecord, Provenance


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class MockAdapter(AdapterABC):
    """In-memory adapter for unit testing the engine layer."""

    def __init__(self) -> None:
        # keyed by (entity_path, key) → list of versions (newest last)
        self._store: dict[tuple[str, str], list[MemoryEntry]] = {}

    def read(
        self, entity_path: str, key: str, *, min_confidence: float = 0.0, branch: str = "main",
    ) -> MemoryEntry | None:
        versions = self._store.get((entity_path, key))
        if not versions:
            return None
        current = versions[-1]
        if current.confidence < min_confidence:
            return None
        return current

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        k = (entry.entity_path, entry.key)
        if k not in self._store:
            self._store[k] = []
        self._store[k].append(entry)
        return entry

    def list(
        self, entity_path: str | None = None, *, include_superseded: bool = False, branch: str = "main",
    ) -> list[MemoryEntry]:
        result: list[MemoryEntry] = []
        for (ep, _key), versions in self._store.items():
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


# ---------------------------------------------------------------------------
# CausalTagger tests
# ---------------------------------------------------------------------------

class TestCausalTagger:
    def test_tag_creates_provenance(self) -> None:
        tagger = CausalTagger(agent_id="test-agent", session_id="sess-1")
        prov = tagger.tag()
        assert prov.agent_id == "test-agent"
        assert prov.session_id == "sess-1"
        assert prov.written_at <= datetime.now(timezone.utc)
        assert prov.pattern_refs == []

    def test_tag_with_pattern_refs(self) -> None:
        tagger = CausalTagger(agent_id="test-agent")
        prov = tagger.tag(pattern_refs=["retry-logic", "timeout"])
        assert prov.pattern_refs == ["retry-logic", "timeout"]

    def test_auto_session_id(self) -> None:
        tagger = CausalTagger(agent_id="a")
        assert tagger.session_id.startswith("sess-")


# ---------------------------------------------------------------------------
# CoWEngine tests
# ---------------------------------------------------------------------------

class TestCoWEngine:
    def _make_engine(self) -> tuple[CoWEngine, MockAdapter]:
        adapter = MockAdapter()
        tagger = CausalTagger(agent_id="engine-test", session_id="sess-e")
        engine = CoWEngine(adapter, tagger)
        return engine, adapter

    def test_write_first_version(self) -> None:
        engine, adapter = self._make_engine()
        written = engine.write("svc-a", "key-1", {"data": 42})
        assert written.version == 1
        assert written.entity_path == "svc-a"
        assert written.key == "key-1"
        assert written.value == {"data": 42}
        assert written.provenance.agent_id == "engine-test"

    def test_write_increments_version(self) -> None:
        engine, _ = self._make_engine()
        engine.write("svc-a", "key-1", {"v": 1})
        w2 = engine.write("svc-a", "key-1", {"v": 2})
        assert w2.version == 2

    def test_write_preserves_outcome_count(self) -> None:
        engine, adapter = self._make_engine()
        # Manually insert an entry with outcome_count > 0
        entry = MemoryEntry(
            entity_path="svc-a",
            key="key-1",
            version=1,
            value={"x": 1},
            provenance=Provenance(
                agent_id="other", session_id="s", written_at=datetime.now(timezone.utc)
            ),
            outcome_count=3,
        )
        adapter.write(entry)

        w2 = engine.write("svc-a", "key-1", {"x": 2})
        assert w2.version == 2
        assert w2.outcome_count == 3

    def test_write_with_confidence(self) -> None:
        engine, _ = self._make_engine()
        written = engine.write("svc-a", "key-1", "val", confidence=0.75)
        assert written.confidence == 0.75

    def test_write_with_ttl(self) -> None:
        engine, _ = self._make_engine()
        ttl = datetime(2099, 1, 1, tzinfo=timezone.utc)
        written = engine.write("svc-a", "key-1", "val", ttl_at=ttl)
        assert written.ttl_at == ttl

    def test_write_sets_content_hash(self) -> None:
        from amfs_core.hashing import content_hash

        engine, _ = self._make_engine()
        written = engine.write("svc-a", "key-1", {"data": 42})
        assert written.content_hash is not None
        assert written.content_hash == content_hash({"data": 42})

    def test_write_sets_integrity_chain(self) -> None:
        engine, _ = self._make_engine()
        w1 = engine.write("svc-a", "key-1", "first")
        assert w1.integrity_chain is not None

        w2 = engine.write("svc-a", "key-1", "second")
        assert w2.integrity_chain is not None
        assert w2.integrity_chain != w1.integrity_chain

    def test_read_delegates(self) -> None:
        engine, _ = self._make_engine()
        engine.write("svc-a", "key-1", {"hello": "world"})
        result = engine.read("svc-a", "key-1")
        assert result is not None
        assert result.value == {"hello": "world"}

    def test_read_nonexistent(self) -> None:
        engine, _ = self._make_engine()
        assert engine.read("nope", "nope") is None

    def test_list_delegates(self) -> None:
        engine, _ = self._make_engine()
        engine.write("svc-a", "k1", 1)
        engine.write("svc-b", "k2", 2)
        entries = engine.list()
        assert len(entries) == 2

    def test_list_filters_entity(self) -> None:
        engine, _ = self._make_engine()
        engine.write("svc-a", "k1", 1)
        engine.write("svc-b", "k2", 2)
        entries = engine.list("svc-a")
        assert len(entries) == 1
        assert entries[0].entity_path == "svc-a"


# ---------------------------------------------------------------------------
# ReadTracker tests
# ---------------------------------------------------------------------------

class TestReadTracker:
    def test_record_and_causal_keys(self) -> None:
        tracker = ReadTracker()
        entry = MemoryEntry(
            entity_path="svc", key="k1", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        tracker.record(entry)
        assert tracker.causal_keys == ["svc/k1"]
        assert tracker.read_count == 1

    def test_deduplication(self) -> None:
        tracker = ReadTracker()
        entry = MemoryEntry(
            entity_path="svc", key="k1", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        tracker.record(entry)
        tracker.record(entry)
        assert tracker.read_count == 1

    def test_clear(self) -> None:
        tracker = ReadTracker()
        entry = MemoryEntry(
            entity_path="svc", key="k1", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        tracker.record(entry)
        tracker.clear()
        assert tracker.read_count == 0
        assert tracker.causal_keys == []

    def test_contains(self) -> None:
        tracker = ReadTracker()
        entry = MemoryEntry(
            entity_path="svc", key="k1", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        tracker.record(entry)
        assert tracker.contains("svc/k1")
        assert not tracker.contains("svc/other")


class TestReadTrackerExternalContexts:
    def test_record_context_adds_to_list(self) -> None:
        tracker = ReadTracker()
        tracker.record_context("git-log", "15 commits since last deploy", source="git")
        assert len(tracker.external_contexts) == 1
        ctx = tracker.external_contexts[0]
        assert ctx["label"] == "git-log"
        assert ctx["summary"] == "15 commits since last deploy"
        assert ctx["source"] == "git"
        assert "recorded_at" in ctx

    def test_record_context_source_optional(self) -> None:
        tracker = ReadTracker()
        tracker.record_context("manual-check", "Verified deployment config")
        ctx = tracker.external_contexts[0]
        assert ctx["source"] is None

    def test_record_context_preserves_order(self) -> None:
        tracker = ReadTracker()
        tracker.record_context("step-1", "first")
        tracker.record_context("step-2", "second")
        tracker.record_context("step-3", "third")
        labels = [c["label"] for c in tracker.external_contexts]
        assert labels == ["step-1", "step-2", "step-3"]

    def test_external_contexts_returns_copy(self) -> None:
        tracker = ReadTracker()
        tracker.record_context("a", "b")
        contexts = tracker.external_contexts
        contexts.append({"label": "injected"})
        assert len(tracker.external_contexts) == 1

    def test_clear_removes_external_contexts(self) -> None:
        tracker = ReadTracker()
        entry = MemoryEntry(
            entity_path="svc", key="k1", version=1, value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        tracker.record(entry)
        tracker.record_context("tool", "output")
        assert tracker.read_count == 1
        assert len(tracker.external_contexts) == 1

        tracker.clear()
        assert tracker.read_count == 0
        assert tracker.causal_keys == []
        assert tracker.external_contexts == []


class TestCoWEngineWithReadTracker:
    def test_read_auto_tracks(self) -> None:
        adapter = MockAdapter()
        tagger = CausalTagger(agent_id="t", session_id="s")
        tracker = ReadTracker()
        engine = CoWEngine(adapter, tagger, tracker)

        engine.write("svc", "key", "val")
        engine.read("svc", "key")

        assert tracker.causal_keys == ["svc/key"]

    def test_read_miss_not_tracked(self) -> None:
        adapter = MockAdapter()
        tagger = CausalTagger(agent_id="t", session_id="s")
        tracker = ReadTracker()
        engine = CoWEngine(adapter, tagger, tracker)

        engine.read("nonexistent", "nope")
        assert tracker.read_count == 0

    def test_engine_without_tracker_still_works(self) -> None:
        adapter = MockAdapter()
        tagger = CausalTagger(agent_id="t", session_id="s")
        engine = CoWEngine(adapter, tagger)
        engine.write("svc", "key", "val")
        result = engine.read("svc", "key")
        assert result is not None
