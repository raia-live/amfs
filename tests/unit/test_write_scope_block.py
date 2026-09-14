"""What a write hands back about the scope it just wrote into.

An agent writes far more than it reads. The cheapest thing that changes that is
telling it, at the moment it writes, that "six entries are already here and four
have never been read back" — a fact about its own store, with no instruction
attached, delivered as data in a tool result so it reaches every client rather
than only the ones that support hooks.

The reason this lives on the server is cost. The gateway used to ask separately,
which is a second GET /api/v1/entries and so a third billed op on every write —
enough that the feature shipped switched off and stayed off on all four services.
Computed inside the write request it is one aggregate in a request that was
already happening, so nothing new is metered and the block can simply be on.
"""

from __future__ import annotations

import asyncio
import types
from datetime import UTC, datetime

import pytest
from amfs_core.abc import AdapterABC
from amfs_core.models import MemoryEntry, Provenance


def _entry(key: str, *, recall_count: int = 0, entity_path: str = "repo/topic") -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        version=1,
        value=f"value of {key}",
        provenance=Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC)),
        confidence=1.0,
        recall_count=recall_count,
    )


class _ListOnlyAdapter(AdapterABC):
    """An adapter with no query language, exercising the Python form."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self._entries = entries

    def list(self, entity_path=None, *, include_superseded=False, branch="main"):
        if entity_path is None:
            return list(self._entries)
        return [e for e in self._entries if e.entity_path == entity_path]

    # The rest of the ABC is unused here.
    def read(self, *a, **k): return None
    def write(self, entry): return entry
    def search(self, query, **k): return []
    def history(self, *a, **k): return []
    def watch(self, *a, **k): return None
    def commit_outcome(self, *a, **k): return None


class TestScopeCounts:
    def test_counts_neighbours_and_excludes_the_entry_just_written(self):
        a = _ListOnlyAdapter([
            _entry("just-written"), _entry("sibling-a"), _entry("sibling-b"),
        ])
        got = a.scope_counts("repo/topic", exclude_key="just-written")
        assert got["existing_entries"] == 2, "the write is not its own neighbour"
        assert "just-written" not in got["keys"]

    def test_never_read_counts_entries_at_recall_count_zero(self):
        a = _ListOnlyAdapter([
            _entry("read-once", recall_count=1),
            _entry("never-a"), _entry("never-b"), _entry("never-c"),
        ])
        got = a.scope_counts("repo/topic")
        assert got["existing_entries"] == 4
        assert got["never_read"] == 3, (
            "a stored recall_count of 0 is the whole signal: written, never read"
        )

    def test_key_sample_is_bounded(self):
        """A scope with hundreds of entries must not dominate the reply."""
        a = _ListOnlyAdapter([_entry(f"k{i:03d}") for i in range(300)])
        got = a.scope_counts("repo/topic", key_limit=8)
        assert got["existing_entries"] == 300
        assert len(got["keys"]) == 8

    def test_a_scope_holding_only_this_write_reports_nothing_to_report(self):
        a = _ListOnlyAdapter([_entry("only-me")])
        got = a.scope_counts("repo/topic", exclude_key="only-me")
        assert got["existing_entries"] == 0
        assert got["never_read"] == 0
        assert got["keys"] == []

    def test_other_entity_paths_are_not_counted(self):
        a = _ListOnlyAdapter([
            _entry("here"), _entry("elsewhere", entity_path="repo/other"),
        ])
        assert a.scope_counts("repo/topic")["existing_entries"] == 1


server = pytest.importorskip("amfs_http.server")


class _Adapter:
    def __init__(self, counts):
        self._counts = counts
        self.calls: list[tuple] = []

    def scope_counts(self, entity_path, *, exclude_key=None, branch="main", key_limit=8):
        self.calls.append((entity_path, exclude_key, branch))
        return self._counts

    def list(self, entity_path=None, *, branch="main", **k):
        return []


def _request():
    return types.SimpleNamespace(state=types.SimpleNamespace(visibility_filter=None))


class TestScopeBlockOnTheWritePath:
    def _block(self, monkeypatch, counts):
        adapter = _Adapter(counts)
        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(
            server, "_get_memory",
            lambda: types.SimpleNamespace(_adapter=adapter, namespace="default"),
        )
        out = asyncio.run(server._scope_block("repo/topic", "just-written"))
        return out, adapter

    def test_returns_the_counts_for_the_scope(self, monkeypatch):
        out, adapter = self._block(monkeypatch, {
            "entity_path": "repo/topic", "existing_entries": 6,
            "never_read": 4, "keys": ["a", "b"],
        })
        assert out["existing_entries"] == 6
        assert out["never_read"] == 4
        assert adapter.calls == [("repo/topic", "just-written", "main")], (
            "the entry just written must be excluded from its own neighbours"
        )

    def test_says_nothing_when_the_scope_is_empty(self, monkeypatch):
        """An empty block would read as a finding; absence is the honest answer."""
        out, _ = self._block(monkeypatch, {
            "entity_path": "repo/topic", "existing_entries": 0,
            "never_read": 0, "keys": [],
        })
        assert out is None

    def test_a_failure_costs_the_footnote_not_the_write(self, monkeypatch):
        class _Broken:
            def scope_counts(self, *a, **k):
                raise RuntimeError("db down")

        monkeypatch.setattr(server, "_async_adapter", None, raising=False)
        monkeypatch.setattr(
            server, "_get_memory",
            lambda: types.SimpleNamespace(_adapter=_Broken(), namespace="default"),
        )
        assert asyncio.run(server._scope_block("repo/topic", "k")) is None

    def test_awaits_an_async_adapter(self, monkeypatch):
        """The hosted path is async; the same helper has to serve both."""
        class _Async:
            async def scope_counts(self, entity_path, *, exclude_key=None,
                                   branch="main", key_limit=8):
                return {"entity_path": entity_path, "existing_entries": 2,
                        "never_read": 1, "keys": ["x"]}

        monkeypatch.setattr(server, "_async_adapter", _Async(), raising=False)
        out = asyncio.run(server._scope_block("repo/topic", "k"))
        assert out["existing_entries"] == 2
