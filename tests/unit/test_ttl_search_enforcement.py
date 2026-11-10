"""Read-time TTL enforcement for search and agent listings.

Complements test_ttl_read_enforcement.py (which covers read()).
Expired entries whose ``ttl_at`` has passed must not be surfaced to callers
through search(), semantic_search(), or list_entries_for_agent(), ensuring
expired secrets and stale hypotheses are never leaked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from amfs_core.default_embedder import HashEmbedder
from amfs_core.models import MemoryEntry, Provenance, SearchQuery, SemanticQuery
from amfs_filesystem.adapter import FilesystemAdapter


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _entry(
    key: str,
    ttl_at: datetime | None,
    *,
    agent_id: str = "agent-a",
    entity_path: str = "svc/mod",
    value: str = "token-val",
) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=value,
        provenance=Provenance(agent_id=agent_id, session_id="sess-1", written_at=_now()),
        confidence=1.0,
        ttl_at=ttl_at,
    )


def test_search_filters_expired_entries(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    past = _now() - timedelta(hours=1)
    future = _now() + timedelta(hours=1)

    adapter.write(_entry("expired-secret", ttl_at=past, value="expired-val"))
    adapter.write(_entry("active-secret", ttl_at=future, value="active-val"))
    adapter.write(_entry("permanent-fact", ttl_at=None, value="fact-val"))

    results = adapter.search(SearchQuery(entity_path="svc/mod"))
    keys = [r.key for r in results]

    assert "expired-secret" not in keys
    assert "active-secret" in keys
    assert "permanent-fact" in keys


def test_search_text_query_filters_expired_entries(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    past = _now() - timedelta(minutes=30)
    future = _now() + timedelta(minutes=30)

    adapter.write(_entry("auth-temp", ttl_at=past, value="auth-token-123"))
    adapter.write(_entry("auth-live", ttl_at=future, value="auth-token-456"))

    results = adapter.search(SearchQuery(query="auth-token"))
    keys = [r.key for r in results]

    assert "auth-temp" not in keys
    assert "auth-live" in keys


def test_list_entries_for_agent_filters_expired_entries(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    past = _now() - timedelta(hours=2)
    future = _now() + timedelta(hours=2)

    adapter.write(_entry("stale-task", ttl_at=past, agent_id="agent-worker"))
    adapter.write(_entry("current-task", ttl_at=future, agent_id="agent-worker"))
    adapter.write(_entry("fixed-task", ttl_at=None, agent_id="agent-worker"))

    agent_entries = adapter.list_entries_for_agent("agent-worker")
    keys = [e.key for e in agent_entries]

    assert "stale-task" not in keys
    assert "current-task" in keys
    assert "fixed-task" in keys


def test_semantic_search_filters_expired_entries(tmp_path) -> None:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    embedder = HashEmbedder(dim=8)
    past = _now() - timedelta(hours=1)
    future = _now() + timedelta(hours=1)

    e_expired = _entry("expired-mem", ttl_at=past, value="deploy config")
    e_active = _entry("active-mem", ttl_at=future, value="deploy config")

    e_expired = e_expired.model_copy(update={"embedding": embedder.embed("deploy config")})
    e_active = e_active.model_copy(update={"embedding": embedder.embed("deploy config")})

    adapter.write(e_expired)
    adapter.write(e_active)

    query = SemanticQuery(text="deploy config", entity_path="svc/mod", min_similarity=0.0)
    results = adapter.semantic_search(query, embedder)
    keys = [entry.key for entry, _ in results]

    assert "expired-mem" not in keys
    assert "active-mem" in keys
