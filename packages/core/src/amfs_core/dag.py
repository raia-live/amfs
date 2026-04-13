"""DAG traversal utilities for commit ancestry and range queries."""

from __future__ import annotations

from collections import deque
from typing import Callable

from amfs_core.models import Commit


def find_common_ancestor(
    commit_a_id: str,
    commit_b_id: str,
    get_commit: Callable[[str], Commit | None],
) -> str | None:
    """Find the most recent common ancestor of two commits.

    Uses a BFS approach: expand both commit lineages in parallel and
    return the first commit ID that appears in both sets.

    ``get_commit`` should return a Commit for a given ID or None.
    """
    if commit_a_id == commit_b_id:
        return commit_a_id

    visited_a: set[str] = set()
    visited_b: set[str] = set()
    queue_a: deque[str] = deque([commit_a_id])
    queue_b: deque[str] = deque([commit_b_id])

    while queue_a or queue_b:
        if queue_a:
            current = queue_a.popleft()
            if current in visited_b:
                return current
            if current not in visited_a:
                visited_a.add(current)
                commit = get_commit(current)
                if commit:
                    for parent_id in commit.parent_ids:
                        if parent_id not in visited_a:
                            queue_a.append(parent_id)

        if queue_b:
            current = queue_b.popleft()
            if current in visited_a:
                return current
            if current not in visited_b:
                visited_b.add(current)
                commit = get_commit(current)
                if commit:
                    for parent_id in commit.parent_ids:
                        if parent_id not in visited_b:
                            queue_b.append(parent_id)

    return None


def commit_range(
    start_id: str,
    end_id: str | None,
    get_commit: Callable[[str], Commit | None],
    *,
    limit: int = 100,
) -> list[Commit]:
    """Walk backwards from start_id to end_id (exclusive), returning commits in order.

    If end_id is None, returns up to ``limit`` commits from start_id back.
    """
    result: list[Commit] = []
    queue: deque[str] = deque([start_id])
    visited: set[str] = set()

    while queue and len(result) < limit:
        current_id = queue.popleft()
        if current_id in visited:
            continue
        if end_id and current_id == end_id:
            continue
        visited.add(current_id)

        commit = get_commit(current_id)
        if commit is None:
            continue
        result.append(commit)

        for parent_id in commit.parent_ids:
            if parent_id not in visited:
                queue.append(parent_id)

    return result


def ancestors(
    commit_id: str,
    get_commit: Callable[[str], Commit | None],
    *,
    limit: int = 1000,
) -> set[str]:
    """Return the set of all ancestor commit IDs (not including commit_id itself)."""
    result: set[str] = set()
    queue: deque[str] = deque()

    commit = get_commit(commit_id)
    if commit:
        for parent_id in commit.parent_ids:
            queue.append(parent_id)

    while queue and len(result) < limit:
        current = queue.popleft()
        if current in result:
            continue
        result.add(current)
        c = get_commit(current)
        if c:
            for pid in c.parent_ids:
                if pid not in result:
                    queue.append(pid)

    return result
