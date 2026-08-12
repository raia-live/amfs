"""The schema bootstrap every process runs when it constructs an adapter.

These exist because of a production incident on 2026-08-12. The bootstrap re-ran
the whole DDL on every container start; ALTER TABLE and CREATE INDEX ask for
ACCESS EXCLUSIVE, a waiting exclusive lock blocks every reader queued behind it,
and a stalled table stalls startup — so the platform started more containers,
each adding another exclusive lock. One orphaned SELECT was enough to wedge the
service, with 29 copies of this DDL queued behind it.

So what is asserted here is not "the schema gets created" (the adapter tests
cover that) but the two properties that stop the convoy forming: a warm database
is not touched at all, and a cold one refuses to wait for a lock.
"""

from __future__ import annotations

import os

import pytest

psycopg = pytest.importorskip("psycopg")

from amfs_postgres.adapter import PostgresAdapter  # noqa: E402

DSN = os.environ.get("AMFS_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="AMFS_TEST_PG_DSN not set")


def _tables(dsn: str) -> set[str]:
    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    return {r[0] for r in rows}


class TestAWarmDatabaseIsLeftAlone:
    def test_the_second_adapter_skips_the_ddl_entirely(self) -> None:
        first = PostgresAdapter(DSN, namespace="bootstrap-warm")
        assert "amfs_schema_state" in _tables(DSN)

        # A statement-level counter would be ideal; the reachable equivalent is
        # to make the DDL impossible to run and show that construction still
        # succeeds. Revoking CREATE on the schema means any CREATE TABLE or
        # CREATE INDEX in the bootstrap would raise, so a clean construction is
        # proof that none of them ran.
        with psycopg.connect(DSN, autocommit=True) as conn:
            user = conn.execute("SELECT current_user").fetchone()[0]
            conn.execute(f'REVOKE CREATE ON SCHEMA public FROM "{user}"')
            try:
                second = PostgresAdapter(DSN, namespace="bootstrap-warm")
                assert second is not None
            finally:
                conn.execute(f'GRANT CREATE ON SCHEMA public TO "{user}"')

        assert first is not None

    def test_a_changed_fingerprint_makes_it_apply_again(self) -> None:
        PostgresAdapter(DSN, namespace="bootstrap-refresh")
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("UPDATE amfs_schema_state SET fingerprint = 'stale'")
            PostgresAdapter(DSN, namespace="bootstrap-refresh")
            current = conn.execute(
                "SELECT fingerprint FROM amfs_schema_state"
            ).fetchall()
        # One row, and not the stale one: a build whose DDL differs from what the
        # database was last given must not be waved through.
        assert len(current) == 1
        assert current[0][0] != "stale"


class TestTheConnectionGoesBackClean:
    def test_lock_timeout_does_not_leak_into_pooled_connections(self) -> None:
        # The pool runs autocommit, so a plain SET lasts for the life of the
        # session. Leaving lock_timeout at 5s would hand it to ordinary writes
        # and turn everyday contention into failed requests.
        adapter = PostgresAdapter(DSN, namespace="bootstrap-leak")
        with adapter._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW lock_timeout")
                assert cur.fetchone()["lock_timeout"] == "0"


class TestTheStatementCeiling:
    def test_it_is_off_unless_asked_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", raising=False)
        adapter = PostgresAdapter(DSN, namespace="bootstrap-timeout-off")
        with adapter._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW statement_timeout")
                assert cur.fetchone()["statement_timeout"] == "0"

    def test_it_applies_to_every_connection_when_asked_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMFS_POSTGRES_STATEMENT_TIMEOUT", "7s")
        adapter = PostgresAdapter(DSN, namespace="bootstrap-timeout-on")
        # Twice, from separate checkouts: the point of setting it on connect
        # rather than per query is that no code path can forget it.
        for _ in range(2):
            with adapter._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SHOW statement_timeout")
                    assert cur.fetchone()["statement_timeout"] == "7s"


class TestAColdDatabaseWillNotQueue:
    def test_it_gives_up_rather_than_joining_a_lock_convoy(self) -> None:
        # A held ACCESS EXCLUSIVE lock stands in for the orphaned query from the
        # incident. The bootstrap must fail its attempts and raise, not sit in
        # the lock queue in front of every reader that arrives next.
        adapter = PostgresAdapter(DSN, namespace="bootstrap-cold")
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute("UPDATE amfs_schema_state SET fingerprint = 'stale'")

        blocker = psycopg.connect(DSN)
        try:
            blocker.execute("BEGIN")
            blocker.execute("LOCK TABLE amfs_memory_entries IN ACCESS EXCLUSIVE MODE")
            with pytest.raises(Exception) as caught:
                PostgresAdapter(DSN, namespace="bootstrap-cold")
            assert "lock timeout" in str(caught.value).lower()
        finally:
            blocker.rollback()
            blocker.close()
        assert adapter is not None
