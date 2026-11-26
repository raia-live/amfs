"""``visible_to_agent`` in the statements, against a real Postgres.

``tests/unit/test_retrieve_private_entries.py`` holds the rule: an entry
written ``shared=False`` is handed to its author and to nobody else. What a
unit test cannot hold is the reason the rule is in the query and not applied
to its results. The statements end in a ``LIMIT``. A store where one agent
keeps many private entries, all of them nearer the query or higher in
confidence than the shared one, fills the page with rows the reader may not
have; dropped afterwards, the page is empty and the entry the reader could
have had was never fetched.

So each test here writes more private rows than the page holds, ranks them
ahead of the one shared row, and asks for a page: lexical and semantic, on
the sync adapter and the async one.

Requires a running Postgres with pgvector. Set AMFS_TEST_PG_DSN to enable;
skipped otherwise. The fixture drops and recreates the entries table, so point
it at a scratch database.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest
from amfs_core.models import MemoryEntry, Provenance, SearchQuery, SemanticQuery

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(
    PG_DSN is None,
    reason="AMFS_TEST_PG_DSN not set — skipping Postgres visible_to_agent tests",
)

NAMESPACE = "visible-to-agent-test"
_DIM = 8
PAGE = 5
PRIVATE_ROWS = 40
QUERY = "billing deploy order"


class FixedEmbedder:
    """Deterministic embedder (no model download)."""

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        for tok in str(text).lower().split():
            vec[sum(tok.encode()) % _DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_value(self, value) -> list[float]:
        return self.embed(value if isinstance(value, str) else str(value))


def _entry(key: str, *, by: str, shared: bool, value: str, confidence: float) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/billing", key=key, version=1, value=value, shared=shared,
        provenance=Provenance(agent_id=by, session_id="s", written_at=datetime.now(UTC)),
        confidence=confidence,
    )


@pytest.fixture
def adapter():
    import psycopg
    from amfs_postgres.adapter import PostgresAdapter

    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("DROP TABLE IF EXISTS amfs_outcomes CASCADE")
    conn.execute("DROP TABLE IF EXISTS amfs_memory_entries CASCADE")
    conn.close()

    a = PostgresAdapter(dsn=PG_DSN, namespace=NAMESPACE, auto_schema=True,
                        embedder=FixedEmbedder(), embedding_dim=_DIM)
    a.ensure_embedding_column()
    # The simulator's private rows carry the query's exact words at the
    # highest confidence, so they lead both orderings. The one shared row is
    # further from the query and less trusted, and so comes last.
    for i in range(PRIVATE_ROWS):
        a.write(_entry(f"scratch-{i:02d}", by="simulator", shared=False,
                       value=QUERY, confidence=0.99))
    a.write(_entry("runbook", by="simulator", shared=True,
                   value=f"{QUERY} after the migration window", confidence=0.5))
    a.write(_entry("readers-own", by="reader", shared=False,
                   value=f"{QUERY} notes to self", confidence=0.4))
    yield a
    a.close()


def _run_async(fn):
    from amfs_postgres.async_adapter import AsyncPostgresAdapter

    async def go():
        a = AsyncPostgresAdapter(dsn=PG_DSN, namespace=NAMESPACE)
        await a.open()
        try:
            return await fn(a)
        finally:
            await a.close()

    return asyncio.run(go())


def _lexical(reader: str | None) -> SearchQuery:
    return SearchQuery(query=QUERY, entity_path="svc/billing", limit=PAGE,
                       sort_by="confidence", visible_to_agent=reader)


def _semantic(reader: str | None) -> SemanticQuery:
    return SemanticQuery(text=QUERY, entity_path="svc/billing", limit=PAGE,
                         visible_to_agent=reader)


def _both(adapter, lexical, semantic) -> dict[str, list[str]]:
    """The keys each of the four statements returns for one pair of queries."""
    async def go(a):
        return (
            [e.key for e in await a.search(lexical)],
            [e.key for e, _ in await a.semantic_search(semantic, FixedEmbedder())],
        )

    async_lexical, async_semantic = _run_async(go)
    return {
        "sync lexical": [e.key for e in adapter.search(lexical)],
        "sync semantic": [e.key for e, _ in adapter.semantic_search(semantic, FixedEmbedder())],
        "async lexical": async_lexical,
        "async semantic": async_semantic,
    }


def test_without_the_rule_the_page_is_all_private_rows(adapter):
    """The premise, shown and not assumed: unasked, every statement fills its
    page from the forty private rows and the shared one is not on it. This is
    also the statement a caller that names no agent is still sent."""
    for name, keys in _both(adapter, _lexical(None), _semantic(None)).items():
        assert len(keys) == PAGE, name
        assert all(k.startswith("scratch-") for k in keys), (name, keys)


def test_another_agent_gets_the_shared_row_and_its_own(adapter):
    """Filtered after the limit this page would be empty. Filtered in the
    statement it holds everything the reader may have."""
    for name, keys in _both(adapter, _lexical("reader"), _semantic("reader")).items():
        assert sorted(keys) == ["readers-own", "runbook"], (name, keys)


def test_the_author_gets_its_own_private_rows(adapter):
    for name, keys in _both(adapter, _lexical("simulator"), _semantic("simulator")).items():
        assert len(keys) == PAGE, name
        assert all(k.startswith("scratch-") for k in keys), (name, keys)
        assert "readers-own" not in keys, name


def test_an_agent_with_nothing_of_its_own_gets_only_what_is_shared(adapter):
    for name, keys in _both(adapter, _lexical("stranger"), _semantic("stranger")).items():
        assert keys == ["runbook"], (name, keys)


def test_the_agent_is_a_parameter_and_not_text_in_the_statement(adapter):
    """An agent id is a caller's string. One built to close the predicate and
    widen it has to arrive as a value."""
    hostile = "x' OR shared IS NOT NULL OR agent_id = 'simulator"
    for name, keys in _both(adapter, _lexical(hostile), _semantic(hostile)).items():
        assert keys == ["runbook"], (name, keys)
