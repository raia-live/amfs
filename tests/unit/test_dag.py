"""Tests for DAG traversal utilities."""

from __future__ import annotations

from datetime import datetime, timezone

from amfs_core.dag import ancestors, commit_range, find_common_ancestor
from amfs_core.models import Commit


def _make_commit(
    commit_id: str,
    parent_ids: list[str] | None = None,
) -> Commit:
    return Commit(
        id=commit_id,
        message=f"commit {commit_id}",
        author_agent_id="test",
        parent_ids=parent_ids or [],
        branch="main",
        created_at=datetime.now(timezone.utc),
    )


class TestFindCommonAncestor:
    def test_same_commit(self):
        store: dict[str, Commit] = {}
        assert find_common_ancestor("a", "a", store.get) == "a"

    def test_linear_chain(self):
        # c1 <- c2 <- c3
        store = {
            "c1": _make_commit("c1"),
            "c2": _make_commit("c2", ["c1"]),
            "c3": _make_commit("c3", ["c2"]),
        }
        assert find_common_ancestor("c3", "c2", store.get) == "c2"
        assert find_common_ancestor("c3", "c1", store.get) == "c1"

    def test_diamond_merge(self):
        #   c1
        #  /  \
        # c2  c3
        #  \  /
        #   c4
        store = {
            "c1": _make_commit("c1"),
            "c2": _make_commit("c2", ["c1"]),
            "c3": _make_commit("c3", ["c1"]),
            "c4": _make_commit("c4", ["c2", "c3"]),
        }
        assert find_common_ancestor("c2", "c3", store.get) == "c1"
        assert find_common_ancestor("c4", "c3", store.get) == "c3"

    def test_no_common_ancestor(self):
        store = {
            "a1": _make_commit("a1"),
            "b1": _make_commit("b1"),
        }
        assert find_common_ancestor("a1", "b1", store.get) is None

    def test_branched_from_same_point(self):
        # root <- a1, root <- b1
        store = {
            "root": _make_commit("root"),
            "a1": _make_commit("a1", ["root"]),
            "b1": _make_commit("b1", ["root"]),
        }
        assert find_common_ancestor("a1", "b1", store.get) == "root"


class TestCommitRange:
    def test_linear_range(self):
        store = {
            "c1": _make_commit("c1"),
            "c2": _make_commit("c2", ["c1"]),
            "c3": _make_commit("c3", ["c2"]),
        }
        result = commit_range("c3", "c1", store.get)
        ids = [c.id for c in result]
        assert "c3" in ids
        assert "c2" in ids
        assert "c1" not in ids

    def test_range_with_limit(self):
        store = {
            "c1": _make_commit("c1"),
            "c2": _make_commit("c2", ["c1"]),
            "c3": _make_commit("c3", ["c2"]),
        }
        result = commit_range("c3", None, store.get, limit=2)
        assert len(result) == 2

    def test_full_range_no_end(self):
        store = {
            "c1": _make_commit("c1"),
            "c2": _make_commit("c2", ["c1"]),
            "c3": _make_commit("c3", ["c2"]),
        }
        result = commit_range("c3", None, store.get)
        assert len(result) == 3


class TestAncestors:
    def test_no_parents(self):
        store = {"c1": _make_commit("c1")}
        result = ancestors("c1", store.get)
        assert result == set()

    def test_linear_ancestors(self):
        store = {
            "c1": _make_commit("c1"),
            "c2": _make_commit("c2", ["c1"]),
            "c3": _make_commit("c3", ["c2"]),
        }
        result = ancestors("c3", store.get)
        assert result == {"c1", "c2"}

    def test_diamond_ancestors(self):
        store = {
            "c1": _make_commit("c1"),
            "c2": _make_commit("c2", ["c1"]),
            "c3": _make_commit("c3", ["c1"]),
            "c4": _make_commit("c4", ["c2", "c3"]),
        }
        result = ancestors("c4", store.get)
        assert result == {"c1", "c2", "c3"}
