"""Integration tests for write-time embeddings on the Postgres adapter.

Requires a running Postgres with the pgvector extension available. Set
AMFS_TEST_PG_DSN to enable; skipped otherwise (so unit CI stays green).

Example:
    AMFS_TEST_PG_DSN=postgresql://localhost/amfs_test \
        pytest tests/integration/test_postgres_embeddings.py

These validate the unvalidated-in-unit-CI paths added for configurable-dimension
embeddings: PostgresAdapter(embedder=..., embedding_dim=...),
ensure_embedding_column(), write-time embedding, backfill_embeddings(), and that
semantic_search returns the stored vectors.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from amfs_core.models import MemoryEntry, Provenance, SemanticQuery

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(
    PG_DSN is None,
    reason="AMFS_TEST_PG_DSN not set — skipping Postgres embedding tests",
)

_DIM = 8


class FixedEmbedder:
    """Deterministic embedder (no model download): hashes tokens into _DIM bins."""

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        for tok in str(text).lower().split():
            vec[hash(tok) % _DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_value(self, value) -> list[float]:
        return self.embed(value if isinstance(value, str) else str(value))


def _entry(key: str, value: str) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/db", key=key, version=1, value=value,
        provenance=Provenance(agent_id="a", session_id="s",
                              written_at=datetime.now(timezone.utc)),
        confidence=0.9,
    )


@pytest.fixture
def adapter():
    import psycopg
    from amfs_postgres.adapter import PostgresAdapter

    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("DROP TABLE IF EXISTS amfs_outcomes CASCADE")
    conn.execute("DROP TABLE IF EXISTS amfs_memory_entries CASCADE")
    conn.close()

    a = PostgresAdapter(dsn=PG_DSN, namespace="emb-test", auto_schema=True,
                        embedder=FixedEmbedder(), embedding_dim=_DIM)
    a.ensure_embedding_column()
    yield a
    a.close()


def test_write_time_embedding_is_persisted(adapter):
    adapter.write(_entry("db-billing", "billing runs on postgres"))
    results = adapter.semantic_search(
        SemanticQuery(text="billing runs on postgres", entity_path="svc/db", limit=5),
        FixedEmbedder(),
    )
    assert results, "expected a semantic hit from a write-time embedding"
    top_entry, sim = results[0]
    assert top_entry.key == "db-billing"
    assert sim > 0.5


def test_backfill_embeddings_fills_missing(adapter):
    # Write without an embedder to leave embedding NULL, then backfill.
    from amfs_postgres.adapter import PostgresAdapter
    no_embed = PostgresAdapter(dsn=PG_DSN, namespace="emb-test", auto_schema=False)
    no_embed.write(_entry("db-analytics", "analytics runs on clickhouse"))
    no_embed.close()

    filled = adapter.backfill_embeddings()
    assert filled >= 1
    results = adapter.semantic_search(
        SemanticQuery(text="analytics runs on clickhouse", entity_path="svc/db", limit=5),
        FixedEmbedder(),
    )
    assert any(e.key == "db-analytics" for e, _ in results)


def test_semantic_search_reaches_beneath_a_scope_only_when_asked(adapter):
    """A served model trained on ``svc/*`` retrieves with its prefix as the
    path and no request names a deeper one: an exact match on ``svc`` finds
    nothing, and ``include_descendants`` finds ``svc/db`` and ``svc/db/replica``
    but never the look-alike ``svcx``. Both adapters, the same rule as
    ``search``."""
    import asyncio

    from amfs_postgres.async_adapter import AsyncPostgresAdapter

    def at(path: str, key: str, value: str) -> MemoryEntry:
        return _entry(key, value).model_copy(update={"entity_path": path})

    adapter.write(at("svc/db", "db-billing", "billing runs on postgres"))
    adapter.write(at("svc/db/replica", "replica", "billing replica runs on postgres too"))
    adapter.write(at("svcx", "lookalike", "billing runs on postgres elsewhere"))
    q = SemanticQuery(text="billing runs on postgres", entity_path="svc", limit=10)
    wide = q.model_copy(update={"include_descendants": True})

    assert adapter.semantic_search(q, FixedEmbedder()) == []
    keys = sorted(e.entity_path for e, _ in adapter.semantic_search(wide, FixedEmbedder()))
    assert keys == ["svc/db", "svc/db/replica"]

    async def go():
        a = AsyncPostgresAdapter(dsn=PG_DSN, namespace="emb-test")
        await a.open()
        try:
            exact = await a.semantic_search(q, FixedEmbedder())
            beneath = await a.semantic_search(wide, FixedEmbedder())
            return exact, beneath
        finally:
            await a.close()

    exact, beneath = asyncio.run(go())
    assert exact == []
    assert sorted(e.entity_path for e, _ in beneath) == ["svc/db", "svc/db/replica"]
