""""This path and below" as a range, against a real Postgres.

``amfs_core.scope`` has two SQL forms of one rule: the ``LIKE`` every caller
gets, and a byte-order range a caller gets by asking for its own account's
rows (``own_account_only``). The unit tests hold both to the Python form with
emulators. This file asks the database itself, because the claims the change
rests on are claims about Postgres:

  - the range selects exactly the rows the ``LIKE`` selects, under a collation
    that is not C, for paths with pattern characters, a backslash, upper case
    and non-ASCII in them;
  - an index over ``entity_path text_pattern_ops`` led by the account is what
    the planner uses for one account's rows beneath a path that many accounts
    have;
  - ``own_account_only`` narrows to the calling account's rows, on both
    adapters, and a search that does not ask is sent the statement it was
    always sent.

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
from amfs_core.scope import covers, descendants_range_sql, descendants_sql

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(
    PG_DSN is None,
    reason="AMFS_TEST_PG_DSN not set — skipping Postgres descendants-range tests",
)

NAMESPACE = "range-test"
_DIM = 8

ACCOUNT_A = "aaaaaaaa-0000-4000-8000-000000000001"
ACCOUNT_B = "bbbbbbbb-0000-4000-8000-000000000002"

# Scopes and paths chosen for what goes wrong with each: the near miss
# (``a/bc``), the characters a LIKE reads as wildcards, the LIKE escape
# character, upper case (which a linguistic collation sorts beside lower case,
# and bytes do not), non-ASCII, and the two bounds of the range themselves.
SCOPES = ["a", "a/b", "amfs", "amfs-internal", "a_b", "a%b", "a\\b", "café", "A", "a/b/"]
PATHS = [
    "a", "a/b", "a/bc", "a/b/c", "a/b/c/d", "a/b.", "a/b0", "a0", "a.",
    "amfs", "amfs-internal", "amfs-internal/deploy",
    "a_b", "a_b/c", "axb", "axb/c",
    "a%b", "a%b/c", "anything/c",
    "a\\b", "a\\b/c", "a\\x/c",
    "A", "A/b", "B/b", "b/b",
    "café", "café/menu", "caféine", "caféine/menu", "cafe/menu",
    "a/b/é", "a/b/Z", "a/b/~", "x/y",
]


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


def _entry(path: str, key: str, value: str = "billing runs on postgres") -> MemoryEntry:
    return MemoryEntry(
        entity_path=path, key=key, version=1, value=value,
        provenance=Provenance(agent_id="a", session_id="s",
                              written_at=datetime.now(UTC)),
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

    a = PostgresAdapter(dsn=PG_DSN, namespace=NAMESPACE, auto_schema=True,
                        embedder=FixedEmbedder(), embedding_dim=_DIM)
    a.ensure_embedding_column()
    yield a
    a.close()


@pytest.fixture
def sql():
    import psycopg

    conn = psycopg.connect(PG_DSN, autocommit=True)
    yield conn
    conn.close()


@pytest.fixture
def account():
    """Put an account on the request the way the HTTP middleware does."""
    from amfs_postgres import tenant_context

    def use(account_id: str | None) -> None:
        tenant_context.set_tls_tenant_account_id(account_id)

    yield use
    tenant_context.clear_tls_tenant_account_id()


def _like_escape(value: str) -> str:
    """The escaping the replaced LIKE form applied to a scope."""
    return value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


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


class TestTheRangeIsTheSameRowsAsTheLike:
    def test_the_database_is_not_one_where_any_form_would_do(self, sql):
        """Under a C collation a LIKE prefix and a plain comparison already
        agree with byte order. The claim is worth testing where they do not."""
        collation = sql.execute(
            "SELECT datcollate FROM pg_database WHERE datname = current_database()"
        ).fetchone()[0]
        if collation in ("C", "POSIX"):
            pytest.skip("database collation is C: run against a linguistic collation")

    def test_every_scope_selects_what_the_like_selected_and_what_covers_says(self, adapter, sql):
        for i, path in enumerate(PATHS):
            adapter.write(_entry(path, f"k{i}"))

        for scope in SCOPES:
            root = scope.rstrip("/")
            like_clause, like_params = descendants_sql("entity_path", scope)
            assert like_params == [root, _like_escape(root) + "/%"]
            was = sorted(
                r[0] for r in sql.execute(
                    "SELECT entity_path FROM amfs_memory_entries "
                    f"WHERE namespace = %s AND {like_clause}",
                    (NAMESPACE, *like_params),
                ).fetchall()
            )
            clause, params = descendants_range_sql("entity_path", scope)
            now = sorted(
                r[0] for r in sql.execute(
                    "SELECT entity_path FROM amfs_memory_entries "
                    f"WHERE namespace = %s AND {clause}",
                    (NAMESPACE, *params),
                ).fetchall()
            )
            expected = sorted(p for p in PATHS if covers(scope, p))
            assert now == was, f"range and LIKE disagree for scope {scope!r}"
            assert now == expected, f"range and covers() disagree for scope {scope!r}"

    def test_both_adapters_find_the_same_rows_whichever_form_they_send(
        self, adapter, sql, account
    ):
        """Every path, owned by one account. Unasked the adapters send the
        LIKE; asked, with that account on the request, they send the range
        and name the account. Both must reach what ``covers`` says."""
        for i, path in enumerate(PATHS):
            adapter.write(_entry(path, f"k{i}"))
        sql.execute(
            "UPDATE amfs_memory_entries SET account_id = %s::uuid WHERE namespace = %s",
            (ACCOUNT_A, NAMESPACE),
        )
        account(ACCOUNT_A)

        for scope in SCOPES:
            expected = sorted(p for p in PATHS if covers(scope, p))
            for own in (False, True):
                lexical = SearchQuery(entity_path=scope, include_descendants=True,
                                      own_account_only=own, limit=500)
                semantic = SemanticQuery(text="billing runs on postgres", entity_path=scope,
                                         include_descendants=True, own_account_only=own,
                                         limit=500)

                assert sorted(e.entity_path for e in adapter.search(lexical)) == expected, scope
                assert sorted(
                    e.entity_path for e, _ in adapter.semantic_search(semantic, FixedEmbedder())
                ) == expected, scope

                async def both(a, lexical=lexical, semantic=semantic):
                    return (await a.search(lexical),
                            await a.semantic_search(semantic, FixedEmbedder()))

                found, pairs = _run_async(both)
                assert sorted(e.entity_path for e in found) == expected, scope
                assert sorted(e.entity_path for e, _ in pairs) == expected, scope


class TestTheStatementAnExistingCallerSends:
    """The range is read by a different plan than the LIKE, and where the
    statement orders by vector distance the plan decides what is found. So a
    search that does not ask must be sent, character for character, what it
    was sent before."""

    def _statements(self, adapter, monkeypatch, query, *, semantic: bool):
        sent: list[str] = []
        import psycopg

        original = psycopg.Cursor.execute

        def recording(cursor, statement, params=None, **kwargs):
            sent.append(str(statement))
            return original(cursor, statement, params, **kwargs)

        monkeypatch.setattr(psycopg.Cursor, "execute", recording)
        if semantic:
            adapter.semantic_search(query, FixedEmbedder())
        else:
            adapter.search(query)
        return [s for s in sent if "FROM amfs_memory_entries" in s and "entity_path" in s]

    @pytest.mark.parametrize("semantic", [False, True])
    def test_unasked_it_is_the_like_and_names_no_account(
        self, adapter, account, monkeypatch, semantic
    ):
        account(ACCOUNT_A)
        query = (
            SemanticQuery(text="billing", entity_path="notes", include_descendants=True)
            if semantic else SearchQuery(entity_path="notes", include_descendants=True)
        )

        statements = self._statements(adapter, monkeypatch, query, semantic=semantic)

        assert statements
        for statement in statements:
            assert "entity_path LIKE %s" in statement
            assert "~>=~" not in statement and "~<~" not in statement
            assert "account_id = %s::uuid" not in statement

    @pytest.mark.parametrize("semantic", [False, True])
    def test_asked_it_is_the_range_and_names_the_account(
        self, adapter, account, monkeypatch, semantic
    ):
        account(ACCOUNT_A)
        query = (
            SemanticQuery(text="billing", entity_path="notes", include_descendants=True,
                          own_account_only=True)
            if semantic else SearchQuery(entity_path="notes", include_descendants=True,
                                         own_account_only=True)
        )

        statements = self._statements(adapter, monkeypatch, query, semantic=semantic)

        assert statements
        for statement in statements:
            assert "entity_path ~>=~ %s AND entity_path ~<~ %s" in statement
            assert "LIKE %s" not in statement
            assert "account_id = %s::uuid" in statement

    @pytest.mark.parametrize("semantic", [False, True])
    def test_asked_with_no_account_on_the_request_it_is_the_like(
        self, adapter, account, monkeypatch, semantic
    ):
        account(None)
        query = (
            SemanticQuery(text="billing", entity_path="notes", include_descendants=True,
                          own_account_only=True)
            if semantic else SearchQuery(entity_path="notes", include_descendants=True,
                                         own_account_only=True)
        )

        statements = self._statements(adapter, monkeypatch, query, semantic=semantic)

        assert statements
        for statement in statements:
            assert "entity_path LIKE %s" in statement
            assert "~>=~" not in statement
            assert "account_id = %s::uuid" not in statement


class TestTheIndex:
    def test_the_schema_creates_it(self, adapter, sql):
        definition = sql.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'amfs_memory_entries' "
            "AND indexname = 'idx_entries_account_path_prefix'"
        ).fetchone()
        assert definition is not None
        assert "(account_id, namespace, branch, entity_path text_pattern_ops)" in definition[0]
        assert "superseded_at IS NULL" in definition[0]

    def test_one_accounts_rows_beneath_a_shared_path_are_read_through_it(self, adapter, sql):
        """Forty accounts with the same path, as a product that files every
        customer's notes under one name produces. Asking for one account's
        rows beneath it must not read the other thirty-nine's."""
        sql.execute(
            """
            INSERT INTO amfs_memory_entries
                   (namespace, entity_path, key, version, value, agent_id, session_id,
                    written_at, confidence, branch, account_id)
            SELECT %s, 'notes/default/sessions', 'k' || n, 1, to_jsonb('note ' || n), 'a', 's',
                   now(), 0.9, 'main',
                   ('00000000-0000-4000-8000-' || lpad((n %% 40)::text, 12, '0'))::uuid
              FROM generate_series(1, 8000) n
            """,
            (NAMESPACE,),
        )
        sql.execute("ANALYZE amfs_memory_entries")
        clause, params = descendants_range_sql("entity_path", "notes/default")
        plan = "\n".join(
            r[0] for r in sql.execute(
                "EXPLAIN SELECT key FROM amfs_memory_entries WHERE namespace = %s "
                f"AND branch = 'main' AND superseded_at IS NULL AND {clause} "
                "AND account_id = %s::uuid",
                (NAMESPACE, *params, "00000000-0000-4000-8000-000000000007"),
            ).fetchall()
        )
        assert "idx_entries_account_path_prefix" in plan, plan

        rows = sql.execute(
            "SELECT count(*) FROM amfs_memory_entries WHERE namespace = %s "
            f"AND branch = 'main' AND superseded_at IS NULL AND {clause} "
            "AND account_id = %s::uuid",
            (NAMESPACE, *params, "00000000-0000-4000-8000-000000000007"),
        ).fetchone()[0]
        assert rows == 200


class TestOwnAccountOnly:
    def _two_accounts(self, adapter, sql):
        adapter.write(_entry("notes/default/sessions", "mine"))
        adapter.write(_entry("notes/default/sessions", "theirs"))
        adapter.write(_entry("notes/default", "root-mine"))
        adapter.write(_entry("notes/defaults", "lookalike-mine"))
        for key, account_id in (("mine", ACCOUNT_A), ("root-mine", ACCOUNT_A),
                                ("lookalike-mine", ACCOUNT_A), ("theirs", ACCOUNT_B)):
            sql.execute(
                "UPDATE amfs_memory_entries SET account_id = %s::uuid "
                "WHERE namespace = %s AND key = %s",
                (account_id, NAMESPACE, key),
            )

    def _keys(self, adapter, account, *, own: bool, account_id: str | None):
        account(account_id)
        lexical = SearchQuery(entity_path="notes/default", include_descendants=True,
                              own_account_only=own, limit=50)
        semantic = SemanticQuery(text="billing runs on postgres", entity_path="notes/default",
                                 include_descendants=True, own_account_only=own, limit=50)

        async def both(a):
            return (await a.search(lexical), await a.semantic_search(semantic, FixedEmbedder()))

        found, pairs = _run_async(both)
        return (
            sorted(e.key for e in adapter.search(lexical)),
            sorted(e.key for e, _ in adapter.semantic_search(semantic, FixedEmbedder())),
            sorted(e.key for e in found),
            sorted(e.key for e, _ in pairs),
        )

    def test_unasked_a_search_returns_what_it_returned_before(self, adapter, sql, account):
        """This schema has no row-level rules, so everything at the path comes
        back; where there are some, they decide. Either way the flag's absence
        adds nothing to the statement."""
        self._two_accounts(adapter, sql)
        everything = ["mine", "root-mine", "theirs"]

        for result in self._keys(adapter, account, own=False, account_id=ACCOUNT_A):
            assert result == everything

    def test_asked_it_returns_the_calling_accounts_rows_only(self, adapter, sql, account):
        self._two_accounts(adapter, sql)

        for result in self._keys(adapter, account, own=True, account_id=ACCOUNT_A):
            assert result == ["mine", "root-mine"]
        for result in self._keys(adapter, account, own=True, account_id=ACCOUNT_B):
            assert result == ["theirs"]

    def test_asked_with_no_account_on_the_request_it_changes_nothing(self, adapter, sql, account):
        self._two_accounts(adapter, sql)

        for result in self._keys(adapter, account, own=True, account_id=None):
            assert result == ["mine", "root-mine", "theirs"]
