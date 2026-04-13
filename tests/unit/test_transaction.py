"""Tests for TransactionBuffer and atomic commit workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.engine import CausalTagger
from amfs_core.models import Commit, MemoryEntry, OutcomeRecord, Provenance
from amfs_core.transaction import TransactionBuffer


class MockCommitAdapter(AdapterABC):
    """In-memory adapter supporting batch writes and commit storage."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[MemoryEntry]] = {}
        self._commits: dict[str, Commit] = {}

    def read(
        self, entity_path: str, key: str, *, min_confidence: float = 0.0, branch: str = "main",
    ) -> MemoryEntry | None:
        versions = self._store.get((entity_path, key))
        if not versions:
            return None
        return versions[-1]

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        k = (entry.entity_path, entry.key)
        self._store.setdefault(k, []).append(entry)
        return entry

    def list(self, entity_path=None, *, include_superseded=False):
        entries = []
        for versions in self._store.values():
            if entity_path and versions[0].entity_path != entity_path:
                continue
            if include_superseded:
                entries.extend(versions)
            elif versions:
                entries.append(versions[-1])
        return entries

    def search(self, query):
        return []

    def stats(self):
        from amfs_core.models import MemoryStats
        return MemoryStats(
            total_entries=sum(len(v) for v in self._store.values()),
            total_entities=0, total_agents=0,
        )

    def watch(self, entity_path, callback):
        pass

    def commit_outcome(self, record):
        return []

    def write_batch(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        return [self.write(e) for e in entries]

    def save_commit(self, commit: Commit) -> None:
        self._commits[commit.id] = commit

    def get_commit(self, commit_id: str) -> Commit | None:
        return self._commits.get(commit_id)

    def list_commits(self, *, branch="main", limit=50, namespace="default") -> list[Commit]:
        commits = [c for c in self._commits.values() if c.branch == branch]
        commits.sort(key=lambda c: c.created_at, reverse=True)
        return commits[:limit]


def _make_tagger() -> CausalTagger:
    return CausalTagger(agent_id="test-agent", session_id="test-session")


class TestTransactionBuffer:
    def test_empty_flush_raises(self):
        buf = TransactionBuffer(agent_id="a", session_id="s")
        adapter = MockCommitAdapter()
        tagger = _make_tagger()
        try:
            buf.flush(adapter, tagger)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_single_write(self):
        buf = TransactionBuffer(agent_id="a", session_id="s")
        buf.set_message("test commit")
        buf.write("repo/svc", "k1", {"data": 1})
        adapter = MockCommitAdapter()
        commit, entries = buf.flush(adapter, _make_tagger())

        assert commit.message == "test commit"
        assert commit.author_agent_id == "a"
        assert len(commit.entries) == 1
        assert len(entries) == 1
        assert entries[0].entity_path == "repo/svc"
        assert entries[0].key == "k1"
        assert entries[0].version == 1
        assert entries[0].commit_id == commit.id

    def test_multi_key_commit(self):
        buf = TransactionBuffer(agent_id="a", session_id="s")
        buf.write("repo/svc", "k1", "v1")
        buf.write("repo/svc", "k2", "v2")
        buf.write("repo/other", "k3", "v3")
        adapter = MockCommitAdapter()
        commit, entries = buf.flush(adapter, _make_tagger())

        assert len(commit.entries) == 3
        assert len(entries) == 3
        assert commit.tree_hash is not None
        keys = {e.key for e in entries}
        assert keys == {"k1", "k2", "k3"}
        assert all(e.commit_id == commit.id for e in entries)

    def test_version_increment_across_commits(self):
        adapter = MockCommitAdapter()
        tagger = _make_tagger()

        buf1 = TransactionBuffer(agent_id="a", session_id="s")
        buf1.write("repo/svc", "k1", "first")
        _, entries1 = buf1.flush(adapter, tagger)
        assert entries1[0].version == 1

        buf2 = TransactionBuffer(agent_id="a", session_id="s")
        buf2.write("repo/svc", "k1", "second")
        _, entries2 = buf2.flush(adapter, tagger)
        assert entries2[0].version == 2

    def test_content_hashes_set(self):
        buf = TransactionBuffer(agent_id="a", session_id="s")
        buf.write("repo/svc", "k1", {"x": 42})
        adapter = MockCommitAdapter()
        commit, entries = buf.flush(adapter, _make_tagger())

        assert entries[0].content_hash is not None
        assert entries[0].integrity_chain is not None
        assert commit.entries[0].content_hash == entries[0].content_hash

    def test_pending_count(self):
        buf = TransactionBuffer(agent_id="a", session_id="s")
        assert buf.pending_count == 0
        buf.write("repo/svc", "k1", "v1")
        assert buf.pending_count == 1
        buf.write("repo/svc", "k2", "v2")
        assert buf.pending_count == 2

    def test_flush_clears_pending(self):
        buf = TransactionBuffer(agent_id="a", session_id="s")
        buf.write("repo/svc", "k1", "v1")
        adapter = MockCommitAdapter()
        buf.flush(adapter, _make_tagger())
        assert buf.pending_count == 0

    def test_commit_stored_in_adapter(self):
        buf = TransactionBuffer(agent_id="a", session_id="s")
        buf.write("repo/svc", "k1", "v1")
        adapter = MockCommitAdapter()
        commit, _ = buf.flush(adapter, _make_tagger())

        stored = adapter.get_commit(commit.id)
        assert stored is not None
        assert stored.id == commit.id

    def test_list_commits(self):
        adapter = MockCommitAdapter()
        tagger = _make_tagger()

        for i in range(3):
            buf = TransactionBuffer(agent_id="a", session_id="s")
            buf.set_message(f"commit {i}")
            buf.write("repo/svc", f"k{i}", f"v{i}")
            buf.flush(adapter, tagger)

        commits = adapter.list_commits()
        assert len(commits) == 3
