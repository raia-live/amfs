"""Postgres adapter tests — runs the full adapter contract suite.

Requires a running Postgres instance. Set AMFS_TEST_PG_DSN to enable.
Example: AMFS_TEST_PG_DSN=postgresql://localhost/amfs_test pytest tests/integration/test_postgres_adapter.py
"""

from __future__ import annotations

import os

import pytest

from tests.integration.adapter_contract import AdapterContractTests, _make_entry

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(
    PG_DSN is None,
    reason="AMFS_TEST_PG_DSN not set — skipping Postgres tests",
)


@pytest.fixture
def adapter():
    """Create a PostgresAdapter with a fresh schema for each test."""
    from amfs_postgres.adapter import PostgresAdapter
    import psycopg

    # Clean tables before each test
    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("DROP TABLE IF EXISTS amfs_outcomes CASCADE")
    conn.execute("DROP TABLE IF EXISTS amfs_memory_entries CASCADE")
    # Traces and events too, so the pagination and partitioning tests below
    # start from an empty, freshly bootstrapped (partitioned) trace table.
    conn.execute("DROP TABLE IF EXISTS amfs_decision_traces CASCADE")
    conn.execute("DROP TABLE IF EXISTS amfs_decision_traces_legacy CASCADE")
    conn.execute("DROP TABLE IF EXISTS amfs_events CASCADE")
    conn.close()

    a = PostgresAdapter(dsn=PG_DSN, namespace="test", auto_schema=True)
    yield a
    a.close()


class TestPostgresAdapter(AdapterContractTests):
    """Run all contract tests against the PostgresAdapter."""

    pass


def test_captured_text_survives_a_round_trip(adapter) -> None:
    """``task_input``/``response_text`` must come back out of the database.

    The columns were written by ``save_trace`` but omitted from the SELECT lists in
    ``get_trace`` and ``list_traces``, and ``_row_to_trace`` reads them with
    ``row.get``, so every read returned ``None`` without raising. Behaviour cloning
    just found nothing eligible and reported zero examples.

    Asserted through both read paths, since each had its own column list and only
    fixing one would leave half the bug in place.
    """
    from amfs_core.models import DecisionTrace

    saved = adapter.save_trace(
        DecisionTrace(
            agent_id="capture-agent",
            session_id="capture-session",
            outcome_ref="round-trip-1",
            outcome_type="success",
            task_input="restart the payments worker in staging",
            response_text="scaled the deployment to zero and back",
        )
    )

    fetched = adapter.get_trace(saved.id)
    assert fetched is not None
    assert fetched.task_input == "restart the payments worker in staging"
    assert fetched.response_text == "scaled the deployment to zero and back"

    listed = [t for t in adapter.list_traces(agent_id="capture-agent")]
    assert listed, "the trace should be listable"
    assert listed[0].task_input == "restart the payments worker in staging"
    assert listed[0].response_text == "scaled the deployment to zero and back"


def test_a_trace_without_captured_text_round_trips_as_none(adapter) -> None:
    """The columns are nullable and absence must stay absence, not empty string."""
    from amfs_core.models import DecisionTrace

    saved = adapter.save_trace(
        DecisionTrace(
            agent_id="capture-agent",
            session_id="no-capture",
            outcome_ref="round-trip-2",
            outcome_type="success",
        )
    )

    fetched = adapter.get_trace(saved.id)
    assert fetched is not None
    assert fetched.task_input is None
    assert fetched.response_text is None
    assert fetched.tool_calls == []


def test_recorded_actions_survive_a_round_trip(adapter) -> None:
    """The same omission, on the column that completes the training pair.

    ``tool_calls`` is written by ``save_trace`` and has to appear in both SELECT
    lists to come back. ``_row_to_trace`` reads it with ``row.get``, so a missing
    column yields an empty list rather than an error — the exact shape that made
    the captured-text bug invisible, and the reason this is asserted through
    ``get_trace`` and ``list_traces`` separately.
    """
    from amfs_core.models import DecisionTrace, ToolCall

    saved = adapter.save_trace(
        DecisionTrace(
            agent_id="action-agent",
            session_id="action-session",
            outcome_ref="round-trip-3",
            outcome_type="success",
            task_input="checkout is erroring after the deploy",
            tool_calls=[
                ToolCall(
                    tool_name="deploy_rollback",
                    arguments={"service": "checkout", "to_version": "v41"},
                    result_summary="rolled back",
                    duration_ms=1430,
                )
            ],
        )
    )

    fetched = adapter.get_trace(saved.id)
    assert fetched is not None
    assert [t.tool_name for t in fetched.tool_calls] == ["deploy_rollback"]
    assert fetched.tool_calls[0].arguments == {
        "service": "checkout",
        "to_version": "v41",
    }
    assert fetched.tool_calls[0].duration_ms == 1430

    listed = [t for t in adapter.list_traces(agent_id="action-agent")]
    assert listed, "the trace should be listable"
    assert [t.tool_name for t in listed[0].tool_calls] == ["deploy_rollback"]


def test_a_trace_carrying_session_metadata_is_saved(adapter) -> None:
    """The model, not a dict, is what actually reaches ``save_trace``.

    ``DecisionTrace.session_metadata`` is typed, so ``POST /api/v1/traces``
    validating a request body hands the adapter a ``SessionMetadata`` — which
    ``json.dumps`` refused to encode, failing the save with a 500 and dropping
    the trace. Every other test here left the field unset, so the whole hosted
    write path was uncovered while the column looked exercised.

    Constructed as a model on purpose: passing a dict would be validated into
    one by ``DecisionTrace`` anyway, and asserting on the round trip is what
    proves the value survived rather than being written as an empty object.
    """
    from amfs_core.models import DecisionTrace, SessionMetadata

    saved = adapter.save_trace(
        DecisionTrace(
            agent_id="metadata-agent",
            session_id="metadata-session",
            outcome_ref="round-trip-4",
            outcome_type="success",
            session_metadata=SessionMetadata(
                model="claude-4-opus",
                client_name="cursor",
                platform="cursor",
                tools_available=["Shell", "Read"],
            ),
        )
    )

    fetched = adapter.get_trace(saved.id)
    assert fetched is not None
    assert fetched.session_metadata is not None
    assert fetched.session_metadata.model == "claude-4-opus"
    assert fetched.session_metadata.client_name == "cursor"
    assert fetched.session_metadata.tools_available == ["Shell", "Read"]


def test_a_trace_without_session_metadata_still_saves(adapter) -> None:
    """The empty case the fix must not regress: no metadata is not an error."""
    from amfs_core.models import DecisionTrace

    saved = adapter.save_trace(
        DecisionTrace(
            agent_id="metadata-agent",
            session_id="metadata-session",
            outcome_ref="round-trip-5",
            outcome_type="success",
        )
    )

    fetched = adapter.get_trace(saved.id)
    assert fetched is not None
    assert fetched.session_metadata is None or (
        fetched.session_metadata.model is None
    )


# ──────────────────────────────────────────────────────────────────────
# Keyset pagination, SQL counts, and the causal_entries containment filter
# ──────────────────────────────────────────────────────────────────────


def _saved_traces(adapter, n: int, *, agent_id: str = "page-agent", entity_path: str = "svc/api"):
    """Persist *n* traces a minute apart, oldest first; returns them newest first."""
    from datetime import UTC, datetime, timedelta

    from amfs_core.models import DecisionTrace, TraceEntry

    base = datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        out.append(
            adapter.save_trace(
                DecisionTrace(
                    agent_id=agent_id,
                    session_id="s",
                    outcome_ref=f"PAGE-{i}",
                    outcome_type="success" if i % 2 == 0 else "failure",
                    causal_entries=[
                        TraceEntry(entity_path=entity_path, key=f"k{i}", version=1, confidence=1.0)
                    ],
                    created_at=base + timedelta(minutes=i),
                )
            )
        )
    return list(reversed(out))


def test_list_traces_pages_by_cursor_without_gaps_or_repeats(adapter) -> None:
    from amfs_core.pagination import page_from_overfetch

    expected = [t.outcome_ref for t in _saved_traces(adapter, 11)]

    seen, cursor, pages = [], None, 0
    while True:
        rows = adapter.list_traces(agent_id="page-agent", limit=4 + 1, cursor=cursor)
        page = page_from_overfetch(
            rows, limit=4, timestamp=lambda t: t.created_at, tiebreak=lambda t: t.id
        )
        pages += 1
        seen.extend(t.outcome_ref for t in page.items)
        if not page.has_more:
            assert page.next_cursor is None
            break
        cursor = page.next_cursor
    assert pages == 3
    assert seen == expected

    # offset is still honoured when no cursor is given
    rows = adapter.list_traces(agent_id="page-agent", limit=3, offset=4)
    assert [t.outcome_ref for t in rows] == expected[4:7]


def test_list_traces_since_until_and_count(adapter) -> None:
    from datetime import UTC, datetime, timedelta

    _saved_traces(adapter, 10)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    since, until = base + timedelta(minutes=2), base + timedelta(minutes=6)

    rows = adapter.list_traces(agent_id="page-agent", since=since, until=until, limit=100)
    assert [t.outcome_ref for t in rows] == ["PAGE-5", "PAGE-4", "PAGE-3", "PAGE-2"]

    assert adapter.count_traces(agent_id="page-agent") == 10
    assert adapter.count_traces(agent_id="page-agent", since=since, until=until) == 4
    assert adapter.count_traces(agent_id="page-agent", outcome_type="success") == 5
    assert adapter.count_traces(agent_id="nobody") == 0


def test_entity_path_filter_uses_containment_and_matches_the_old_scan(adapter, pg_conn) -> None:
    """The ``@>`` rewrite must return exactly what ``jsonb_array_elements`` did."""
    _saved_traces(adapter, 6, entity_path="svc/api")
    _saved_traces(adapter, 3, agent_id="other-agent", entity_path="svc/other")

    new = adapter.list_traces(entity_path="svc/api", limit=100)
    assert len(new) == 6
    assert all(any(c.entity_path == "svc/api" for c in t.causal_entries) for t in new)

    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM amfs_decision_traces
            WHERE namespace = %s AND EXISTS (
                SELECT 1 FROM jsonb_array_elements(causal_entries) ce
                WHERE ce->>'entity_path' = %s
            )
            ORDER BY created_at DESC, id DESC
            """,
            (adapter._namespace, "svc/api"),
        )
        old_ids = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_traces_causal_entries_gin'"
        )
        assert cur.fetchone() is not None, "GIN index on causal_entries was not created"
    assert [t.id for t in new] == [str(i) for i in old_ids]

    assert adapter.count_traces(entity_path="svc/api") == 6
    assert adapter.list_traces(entity_path="svc/nope", limit=100) == []


def test_trace_read_counts_aggregates_in_sql(adapter) -> None:
    from amfs_core.abc import AdapterABC

    _saved_traces(adapter, 5)
    counts = adapter.trace_read_counts("page-agent")
    assert counts == {"svc/api": {f"k{i}": 1 for i in range(5)}}
    # identical to the base-class scan it replaces
    assert counts == AdapterABC.trace_read_counts(adapter, "page-agent")
    assert adapter.trace_read_counts("nobody") == {}


def test_count_outcomes_matches_rows_not_a_capped_list(adapter, monkeypatch) -> None:
    from amfs_core.models import OutcomeRecord, OutcomeType

    adapter.write(_make_entry(key="counted"))
    for i in range(7):
        adapter.commit_outcome(
            OutcomeRecord(
                outcome_ref=f"CNT-{i}",
                outcome_type=OutcomeType.SUCCESS,
                committed_at=_make_entry().provenance.written_at,
                causal_entry_keys=["checkout-service/counted"],
                agent_id="review-agent",
            )
        )
    # A scan ceiling smaller than the row count must not affect a SQL count.
    monkeypatch.setenv("AMFS_MAX_SCAN_ROWS", "3")
    assert adapter.count_outcomes() == 7
    assert len(adapter.list_outcomes(limit=3)) == 3
    assert [o.outcome_ref for o in adapter.list_outcomes(outcome_ref="CNT-4")] == ["CNT-4"]


def test_list_entries_for_agent_pages_and_filters_in_sql(adapter) -> None:
    from datetime import UTC, datetime, timedelta

    from amfs_core.abc import AdapterABC
    from amfs_core.models import MemoryEntry, Provenance
    from amfs_core.pagination import entry_tiebreak, page_from_overfetch

    base = datetime(2026, 2, 1, tzinfo=UTC)

    def write(i, agent="act-agent", entity="svc/api"):
        adapter.write(
            MemoryEntry(
                entity_path=entity,
                key=f"k{i:02d}",
                value={"i": i},
                provenance=Provenance(
                    agent_id=agent, session_id="s", written_at=base + timedelta(minutes=i)
                ),
            )
        )

    for i in range(9):
        write(i)
    write(50, agent="someone-else")
    write(60, entity="_system/hidden")

    seen, cursor = [], None
    while True:
        rows = adapter.list_entries_for_agent("act-agent", limit=4 + 1, cursor=cursor)
        page = page_from_overfetch(
            rows, limit=4, timestamp=lambda e: e.provenance.written_at, tiebreak=entry_tiebreak
        )
        seen.extend(e.key for e in page.items)
        if not page.has_more:
            break
        cursor = page.next_cursor
    assert seen == [f"k{i:02d}" for i in reversed(range(9))]

    window = adapter.list_entries_for_agent(
        "act-agent", since=base + timedelta(minutes=2), until=base + timedelta(minutes=5), limit=50
    )
    assert [e.key for e in window] == ["k04", "k03", "k02"]

    # same answer as the in-memory base implementation
    base_rows = AdapterABC.list_entries_for_agent(adapter, "act-agent", limit=6)
    sql_rows = adapter.list_entries_for_agent("act-agent", limit=6)
    assert [(e.entity_path, e.key) for e in sql_rows] == [
        (e.entity_path, e.key) for e in base_rows
    ]


def test_list_events_pages_by_cursor(adapter) -> None:
    from amfs_core.models import Event, EventType
    from amfs_core.pagination import page_from_overfetch

    for i in range(7):
        adapter.log_event(
            Event(
                agent_id="evt-agent", namespace=adapter._namespace,
                event_type=EventType.READ, summary=f"r{i}",
            )
        )
    seen, cursor = [], None
    while True:
        rows = adapter.list_events("evt-agent", adapter._namespace, limit=3 + 1, cursor=cursor)
        page = page_from_overfetch(
            rows, limit=3, timestamp=lambda e: e.created_at, tiebreak=lambda e: e.id
        )
        seen.extend(e.summary for e in page.items)
        if not page.has_more:
            break
        cursor = page.next_cursor
    assert sorted(seen) == [f"r{i}" for i in range(7)]
    assert len(seen) == 7


# ──────────────────────────────────────────────────────────────────────
# Partitioning and retention
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def pg_conn():
    import psycopg

    conn = psycopg.connect(PG_DSN, autocommit=True)
    yield conn
    conn.close()


def _relkind(conn, name: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT relkind FROM pg_class WHERE relname = %s", (name,))
        row = cur.fetchone()
    return row[0] if row else None


def _partitions(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            WHERE i.inhparent = 'amfs_decision_traces'::regclass ORDER BY 1
            """
        )
        return [r[0] for r in cur.fetchall()]


def test_fresh_install_creates_a_partitioned_trace_table(adapter, pg_conn) -> None:
    from datetime import UTC, datetime

    from amfs_postgres.trace_partitions import partition_name

    assert _relkind(pg_conn, "amfs_decision_traces") == "p"
    parts = _partitions(pg_conn)
    assert "amfs_decision_traces_default" in parts
    now = datetime.now(UTC)
    assert partition_name(now.year, now.month) in parts  # ensure_trace_partitions ran at init
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname FROM pg_index x
            JOIN pg_attribute a ON a.attrelid = x.indrelid AND a.attnum = ANY (x.indkey)
            WHERE x.indrelid = 'amfs_decision_traces'::regclass AND x.indisprimary
            ORDER BY a.attname
            """
        )
        assert [r[0] for r in cur.fetchall()] == ["created_at", "id"]


def test_get_trace_by_id_works_on_the_partitioned_table(adapter) -> None:
    traces = _saved_traces(adapter, 3)
    for t in traces:
        got = adapter.get_trace(t.id)
        assert got is not None and got.outcome_ref == t.outcome_ref
    assert adapter.get_trace("00000000-0000-0000-0000-000000000000") is None


def test_flat_table_is_migrated_in_place_with_rows_and_policies_kept(adapter, pg_conn) -> None:
    """The upgrade path: a database from before partitioning, with traces in it."""
    from amfs_postgres.adapter import PostgresAdapter

    saved = _saved_traces(adapter, 5)
    adapter.close()

    with pg_conn.cursor() as cur:
        # Rebuild the pre-partitioning shape: a flat table with PRIMARY KEY (id),
        # holding the same rows, an RLS policy and an extra index.
        cur.execute("CREATE TABLE flat_traces (LIKE amfs_decision_traces INCLUDING DEFAULTS)")
        cur.execute("INSERT INTO flat_traces SELECT * FROM amfs_decision_traces")
        cur.execute("DROP TABLE amfs_decision_traces CASCADE")
        cur.execute("ALTER TABLE flat_traces RENAME TO amfs_decision_traces")
        cur.execute("ALTER TABLE amfs_decision_traces ADD PRIMARY KEY (id)")
        cur.execute("CREATE INDEX idx_traces_test_extra ON amfs_decision_traces (outcome_ref)")
        cur.execute(
            "CREATE POLICY traces_test_policy ON amfs_decision_traces "
            "USING (namespace = current_setting('amfs.test_ns', true))"
        )
        cur.execute("ALTER TABLE amfs_decision_traces ENABLE ROW LEVEL SECURITY")
        # A stored fingerprint would let the fast path skip the migration.
        cur.execute("DELETE FROM amfs_schema_state")
    assert _relkind(pg_conn, "amfs_decision_traces") == "r"

    migrated = PostgresAdapter(dsn=PG_DSN, namespace="test", auto_schema=True)
    try:
        assert _relkind(pg_conn, "amfs_decision_traces") == "p"
        assert _relkind(pg_conn, "amfs_decision_traces_legacy") is None
        parts = _partitions(pg_conn)
        assert "amfs_decision_traces_default" in parts
        assert "amfs_decision_traces_y2026m01" in parts  # the month the rows live in

        # every row survived, is reachable by id, and lives in its month
        assert migrated.count_traces(agent_id="page-agent") == 5
        for t in saved:
            assert migrated.get_trace(t.id) is not None
        with pg_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM amfs_decision_traces_default")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM amfs_decision_traces_y2026m01")
            assert cur.fetchone()[0] == 5
            cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'idx_traces_test_extra'")
            assert cur.fetchone() is not None
            cur.execute(
                "SELECT 1 FROM pg_policies WHERE tablename = 'amfs_decision_traces' "
                "AND policyname = 'traces_test_policy'"
            )
            assert cur.fetchone() is not None
            cur.execute(
                "SELECT relrowsecurity FROM pg_class WHERE relname = 'amfs_decision_traces'"
            )
            assert cur.fetchone()[0] is True
            # the new primary key includes the partition key
            cur.execute(
                """
                SELECT count(*) FROM pg_index x
                JOIN pg_attribute a ON a.attrelid = x.indrelid AND a.attnum = ANY (x.indkey)
                WHERE x.indrelid = 'amfs_decision_traces'::regclass AND x.indisprimary
                """
            )
            assert cur.fetchone()[0] == 2

        # a second start is a no-op, not a second migration
        again = PostgresAdapter(dsn=PG_DSN, namespace="test", auto_schema=True)
        again.close()
        assert migrated.count_traces(agent_id="page-agent") == 5
    finally:
        migrated.close()


def test_ensure_trace_partitions_creates_months_ahead_and_drains_default(adapter, pg_conn) -> None:
    from datetime import UTC, datetime, timedelta

    from amfs_core.models import DecisionTrace
    from amfs_postgres.trace_partitions import partition_name

    now = datetime.now(UTC)
    created = adapter.ensure_trace_partitions(months_ahead=4)
    parts = _partitions(pg_conn)
    m = now.replace(day=1)
    for _ in range(5):
        assert partition_name(m.year, m.month) in parts
        m = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
    # the two extra months are new; this month and the next two existed from init
    assert len(created) == 2

    # A backdated trace lands in DEFAULT; the next upkeep gives its month a
    # partition and moves the row out.
    old = adapter.save_trace(
        DecisionTrace(
            agent_id="old-agent",
            session_id="s",
            outcome_ref="OLD-1",
            outcome_type="success",
            created_at=datetime(2019, 6, 15, tzinfo=UTC),
        )
    )
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM amfs_decision_traces_default")
        assert cur.fetchone()[0] == 1
    assert adapter.ensure_trace_partitions() == ["amfs_decision_traces_y2019m06"]
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM amfs_decision_traces_default")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM amfs_decision_traces_y2019m06")
        assert cur.fetchone()[0] == 1
    assert adapter.get_trace(old.id) is not None
    assert adapter.ensure_trace_partitions() == []


# ──────────────────────────────────────────────────────────────────────
# Row-level security on the partitions
#
# A partitioned parent's policies are applied to rows reached through the parent.
# A query naming a partition directly is checked against that partition's own
# policies, and a partition starts with none — so upkeep has to put them there.
#
# The isolation key here is ``agent_id`` rather than the ``account_id`` the hosted
# deployment scopes on, because that column belongs to the tenant package and is
# not in this schema. That is the right shape for the test regardless: the code
# under test copies whatever policies the parent carries without reading them, so
# a test that proved it only for one predicate would be proving less than it looks.
# ──────────────────────────────────────────────────────────────────────

PROBE_ROLE = "amfs_rls_probe"


def _make_parent_tenant_scoped(conn) -> None:
    """Do to the parent what a deployment with row-level security does to it."""
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE amfs_decision_traces ENABLE ROW LEVEL SECURITY")
        cur.execute("ALTER TABLE amfs_decision_traces FORCE ROW LEVEL SECURITY")
        cur.execute(
            "CREATE POLICY agent_isolation ON amfs_decision_traces "
            "USING (agent_id = NULLIF(current_setting('amfs.test_agent', true), ''))"
        )
        cur.execute(
            f"DO $$ BEGIN IF NOT EXISTS ("
            f"  SELECT 1 FROM pg_roles WHERE rolname = '{PROBE_ROLE}'"
            f") THEN CREATE ROLE {PROBE_ROLE} NOSUPERUSER NOBYPASSRLS; END IF; END $$"
        )


def _grant_to_probe(conn, tables: list[str]) -> None:
    with conn.cursor() as cur:
        for t in tables:
            cur.execute(f"GRANT SELECT ON {t} TO {PROBE_ROLE}")


def _read_as_probe(conn, table: str, agent: str) -> list[str]:
    """SELECT from *table* as an unprivileged role with *agent* in scope."""
    with conn.cursor() as cur:
        cur.execute(f"SET ROLE {PROBE_ROLE}")
        try:
            cur.execute("SELECT set_config('amfs.test_agent', %s, false)", (agent,))
            cur.execute(f"SELECT agent_id FROM {table} ORDER BY agent_id")
            return [r[0] for r in cur.fetchall()]
        finally:
            cur.execute("RESET ROLE")


def _rls_flags(conn, name: str) -> tuple[bool, bool]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = %s",
            (name,),
        )
        row = cur.fetchone()
    return (row[0], row[1]) if row else (False, False)


def _policy_names(conn, name: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT policyname FROM pg_policies WHERE tablename = %s ORDER BY policyname",
            (name,),
        )
        return [r[0] for r in cur.fetchall()]


def _save_two_agents(adapter):
    from amfs_core.models import DecisionTrace

    for agent in ("agent-a", "agent-b"):
        adapter.save_trace(
            DecisionTrace(
                agent_id=agent, session_id="s", outcome_ref=f"REF-{agent}",
                outcome_type="success",
            )
        )


def test_postgres_does_not_give_a_partition_its_parents_row_security(adapter, pg_conn) -> None:
    """The behaviour the sync exists for, stated as the fact it is.

    If a future Postgres inherits policies down to partitions, this test fails and
    ``sync_partition_rls`` becomes unnecessary — which is worth being told about
    rather than left to keep running.
    """
    _make_parent_tenant_scoped(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE amfs_decision_traces_y2031m01 PARTITION OF amfs_decision_traces "
            "FOR VALUES FROM ('2031-01-01') TO ('2031-02-01')"
        )

    assert _rls_flags(pg_conn, "amfs_decision_traces") == (True, True)
    assert _rls_flags(pg_conn, "amfs_decision_traces_y2031m01") == (False, False)
    assert _policy_names(pg_conn, "amfs_decision_traces_y2031m01") == []


def test_a_partition_without_the_sync_is_readable_with_no_agent_in_scope(adapter, pg_conn) -> None:
    """The gap itself: the parent filters, the partition named directly does not."""
    _save_two_agents(adapter)
    _make_parent_tenant_scoped(pg_conn)
    from amfs_postgres.trace_partitions import partition_name
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    this_month = partition_name(now.year, now.month)
    _grant_to_probe(pg_conn, ["amfs_decision_traces", this_month])

    assert _read_as_probe(pg_conn, "amfs_decision_traces", "agent-a") == ["agent-a"]
    # Same rows, same role, same session — reached by name instead of through the parent.
    assert _read_as_probe(pg_conn, this_month, "agent-a") == ["agent-a", "agent-b"]


def test_upkeep_scopes_a_partition_read_directly(adapter, pg_conn) -> None:
    """The property that matters: after upkeep, by-name is scoped like the parent."""
    _save_two_agents(adapter)
    _make_parent_tenant_scoped(pg_conn)
    from amfs_postgres.trace_partitions import partition_name
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    this_month = partition_name(now.year, now.month)

    adapter.ensure_trace_partitions()
    _grant_to_probe(pg_conn, ["amfs_decision_traces", this_month])

    assert _read_as_probe(pg_conn, this_month, "agent-a") == ["agent-a"]
    assert _read_as_probe(pg_conn, this_month, "agent-b") == ["agent-b"]


def test_upkeep_does_not_leave_a_partition_denying_everything(adapter, pg_conn) -> None:
    """Enabling row security without copying the policies would read as an empty month.

    The failure this guards against is quieter than the one it replaces: no error,
    no refusal, just zero rows where there are two.
    """
    _save_two_agents(adapter)
    _make_parent_tenant_scoped(pg_conn)
    from amfs_postgres.trace_partitions import partition_name
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    this_month = partition_name(now.year, now.month)

    adapter.ensure_trace_partitions()
    assert _rls_flags(pg_conn, this_month) == (True, False)
    assert _policy_names(pg_conn, this_month) == ["agent_isolation"]
    _grant_to_probe(pg_conn, [this_month])
    assert _read_as_probe(pg_conn, this_month, "agent-a") != []


def test_upkeep_covers_every_partition_not_only_the_new_ones(adapter, pg_conn) -> None:
    """Convergence, not stamping at creation.

    The partitions for this month and the next two exist from init, before any
    policy did. A fix applied only where a partition is created would leave exactly
    those three — the ones production is reading — as the gap.
    """
    from amfs_postgres.trace_partitions import sync_partition_rls

    _make_parent_tenant_scoped(pg_conn)
    before = _partitions(pg_conn)
    assert len(before) >= 3

    adapter.ensure_trace_partitions()
    for name in before:
        assert _policy_names(pg_conn, name) == ["agent_isolation"], name
        assert _rls_flags(pg_conn, name) == (True, False), name

    # Idempotent: nothing left to change on a second pass.
    with pg_conn.cursor() as cur:
        assert sync_partition_rls(cur) == []


def test_upkeep_leaves_an_install_with_no_row_security_alone(adapter, pg_conn) -> None:
    """An install whose parent carries nothing must not have security switched on for it.

    Enabling it here would take an OSS deployment's own traces away from it, since
    there would be no policy to let them back in.
    """
    _save_two_agents(adapter)
    assert _rls_flags(pg_conn, "amfs_decision_traces") == (False, False)

    adapter.ensure_trace_partitions()
    for name in _partitions(pg_conn):
        assert _rls_flags(pg_conn, name) == (False, False), name
    assert adapter.count_traces() == 2


def test_maintenance_can_still_strip_payloads_on_a_scoped_partition(adapter, pg_conn) -> None:
    """Retention names a partition on a checkout that deliberately carries no tenant.

    ``PostgresAdapter._maintenance_connection`` blanks the tenant settings, so a
    forced partition answers the payload-strip ``UPDATE`` with ``UPDATE 0`` — no
    error, no refusal, a retention job reporting success having changed nothing,
    and months stuck in the default partition never drained. Enabled-but-unforced
    exempts the owner, which is the role that runs maintenance.

    The owner is what makes this test mean anything, so it hands the partition to
    the probe role and becomes it. Run as the superuser the suite connects as, it
    would pass whatever the flags said.
    """
    _save_two_agents(adapter)
    _make_parent_tenant_scoped(pg_conn)
    from amfs_postgres.trace_partitions import partition_name
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    this_month = partition_name(now.year, now.month)
    adapter.ensure_trace_partitions()

    with pg_conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {this_month} OWNER TO {PROBE_ROLE}")
        cur.execute("SELECT set_config('amfs.test_agent', '', false)")
        cur.execute(f"SET ROLE {PROBE_ROLE}")
        try:
            cur.execute(f"UPDATE {this_month} SET task_input = NULL")
            stripped = cur.rowcount
        finally:
            cur.execute("RESET ROLE")
            cur.execute(f"ALTER TABLE {this_month} OWNER TO CURRENT_USER")

    assert stripped == 2, (
        "the payload strip matched no rows — a forced partition blinds retention "
        "instead of refusing it"
    )


def test_partitions_are_not_forced_so_the_owner_stays_exempt(adapter, pg_conn) -> None:
    """The parent is forced and that is deliberately not passed down.

    Stated as its own assertion because the consequence of forcing is invisible:
    nothing raises, retention just stops doing anything. Covering the owner as
    well needs maintenance to have a sanctioned way through first — a role with
    BYPASSRLS for the strip and drain statements — and only then forcing these.
    """
    _make_parent_tenant_scoped(pg_conn)
    adapter.ensure_trace_partitions()

    assert _rls_flags(pg_conn, "amfs_decision_traces") == (True, True)
    for name in _partitions(pg_conn):
        assert _rls_flags(pg_conn, name) == (True, False), name


def test_a_restrictive_policy_is_copied_as_restrictive(adapter, pg_conn) -> None:
    """The catalogue is replayed, not paraphrased.

    A restrictive policy copied as permissive would widen what it was written to
    narrow, which is the kind of mistake a hand-written copy of the predicate makes.
    """
    from amfs_postgres.trace_partitions import partition_name
    from datetime import UTC, datetime

    _make_parent_tenant_scoped(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute(
            "CREATE POLICY no_placeholders ON amfs_decision_traces AS RESTRICTIVE "
            "FOR SELECT USING (outcome_ref <> 'PLACEHOLDER')"
        )
    adapter.ensure_trace_partitions()

    now = datetime.now(UTC)
    this_month = partition_name(now.year, now.month)
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT permissive, cmd FROM pg_policies "
            "WHERE tablename = %s AND policyname = 'no_placeholders'",
            (this_month,),
        )
        assert cur.fetchone() == ("RESTRICTIVE", "SELECT")


def test_apply_trace_retention_strips_payloads_and_drops_old_partitions(adapter, pg_conn) -> None:
    from datetime import UTC, datetime, timedelta

    from amfs_core.models import DecisionTrace, ToolCall

    now = datetime.now(UTC)

    def save(ref, age_days):
        return adapter.save_trace(
            DecisionTrace(
                agent_id="ret-agent",
                session_id="s",
                outcome_ref=ref,
                outcome_type="success",
                task_input="prompt " + ref,
                response_text="answer " + ref,
                tool_calls=[ToolCall(tool_name="Shell", arguments={"cmd": "ls"})],
                created_at=now - timedelta(days=age_days),
            )
        )

    fresh = save("FRESH", 1)
    cold = save("COLD", 45)
    ancient = save("ANCIENT", 400)
    adapter.ensure_trace_partitions()  # gives the backdated rows their months

    # Payload stripping only: nothing dropped, metadata kept.
    result = adapter.apply_trace_retention(hot_days=30)
    assert result["dropped_partitions"] == []
    assert result["stripped_rows"] == 2

    f = adapter.get_trace(fresh.id)
    assert f.task_input == "prompt FRESH" and len(f.tool_calls) == 1
    for t in (adapter.get_trace(cold.id), adapter.get_trace(ancient.id)):
        assert t is not None
        assert t.task_input is None and t.response_text is None
        assert t.tool_calls == []
        assert t.outcome_ref in ("COLD", "ANCIENT")  # metadata stays

    # Running again strips nothing new.
    assert adapter.apply_trace_retention(hot_days=30)["stripped_rows"] == 0

    # Dropping: only partitions wholly older than the cutoff go.
    ancient_month = (now - timedelta(days=400)).strftime("amfs_decision_traces_y%Ym%m")
    result = adapter.apply_trace_retention(hot_days=30, drop_after_days=365)
    assert ancient_month in result["dropped_partitions"]
    assert adapter.get_trace(ancient.id) is None
    assert adapter.get_trace(cold.id) is not None
    assert adapter.get_trace(fresh.id) is not None
    assert ancient_month not in _partitions(pg_conn)


def test_retention_cli_dry_run_reports_without_changing_anything(adapter, pg_conn) -> None:
    from datetime import UTC, datetime, timedelta

    from amfs_core.models import DecisionTrace
    from amfs_postgres import retention

    adapter.save_trace(
        DecisionTrace(
            agent_id="cli-agent",
            session_id="s",
            outcome_ref="CLI-1",
            outcome_type="success",
            task_input="keep me",
            created_at=datetime.now(UTC) - timedelta(days=90),
        )
    )
    adapter.ensure_trace_partitions()
    code = retention.main(["--dsn", PG_DSN, "--namespace", "test", "--hot-days", "30", "--dry-run"])
    assert code == 0
    rows = adapter.list_traces(agent_id="cli-agent", limit=10)
    assert rows[0].task_input == "keep me"


# ----------------------------------------------------------------------
# Outcome propagation carries the row forward
#
# The trigger used to name the columns a new version should carry, so any
# column added after that list was written took its DEFAULT on every outcome.
# Nothing failed; the entry simply came back different. These cover the fields
# the adapter contract cannot reach because write() does not accept them.
# ----------------------------------------------------------------------


def _commit_success_on(adapter, entity_path="checkout-service", key="retry-pattern"):
    import uuid
    from datetime import UTC, datetime

    from amfs_core.models import OutcomeRecord, OutcomeType

    return adapter.commit_outcome(
        OutcomeRecord(
            outcome_ref=f"DEP-{uuid.uuid4()}",
            outcome_type=OutcomeType.SUCCESS,
            causal_confidence=1.0,
            committed_at=datetime.now(UTC),
            causal_entry_keys=[f"{entity_path}/{key}"],
            agent_id="release-agent",
        )
    )


def test_outcome_preserves_recall_count_and_tier(adapter) -> None:
    """Both are maintained by the store, so write() cannot set them.

    recall_count is the counter the product reports back to users as evidence
    that memory is being reused, and it was reset to 0 by every outcome.
    """
    import psycopg

    adapter.write(_make_entry(confidence=0.9))
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(
            "UPDATE amfs_memory_entries SET recall_count = 7, tier = 1"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL"
        )

    _commit_success_on(adapter)

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        row = conn.execute(
            "SELECT recall_count, tier FROM amfs_memory_entries"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL"
        ).fetchone()
    assert row[0] == 7, "an outcome erased the entry's reuse history"
    assert row[1] == 1, "an outcome moved the entry to a different tier"


def test_outcome_carries_forward_a_column_the_trigger_never_heard_of(adapter) -> None:
    """The regression guard for the whole class, not three instances of it.

    A deployment that extends this table gets columns the trigger's author
    never saw. This adds one, sets it, and asserts the value survives an
    outcome — which is only true because the INSERT copies the row.
    """
    import psycopg

    adapter.write(_make_entry(confidence=0.9))
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(
            "ALTER TABLE amfs_memory_entries"
            " ADD COLUMN IF NOT EXISTS zz_downstream TEXT DEFAULT 'default-value'"
        )
        conn.execute(
            "UPDATE amfs_memory_entries SET zz_downstream = 'set-by-deployment'"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL"
        )

    _commit_success_on(adapter)

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        row = conn.execute(
            "SELECT zz_downstream FROM amfs_memory_entries"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL"
        ).fetchone()
    assert row[0] == "set-by-deployment", (
        "a column the trigger does not name was reset to its default, which is "
        "how recall_count, shared, tier, branch and embedding were all lost"
    )


def test_outcome_preserves_the_embedding(adapter) -> None:
    """A NULL embedding drops the live version out of vector search.

    Also the one column whose jsonb round-trip cannot be assumed, pgvector
    being an extension type, so this asserts it rather than trusting it.
    """
    import psycopg

    adapter.write(_make_entry(confidence=0.9))
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        if not conn.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        ).fetchone():
            pytest.skip("pgvector not installed in this database")
        # The adapter bootstrap does not create this column — migration 002
        # does, and the fixture drops the table — so add it the way 002 would.
        conn.execute(
            "ALTER TABLE amfs_memory_entries"
            " ADD COLUMN IF NOT EXISTS embedding vector(384)"
        )
        vector = "[" + ",".join(["0.5"] * 384) + "]"
        conn.execute(
            "UPDATE amfs_memory_entries SET embedding = %s::vector"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL",
            (vector,),
        )

    _commit_success_on(adapter)

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        row = conn.execute(
            "SELECT embedding::text FROM amfs_memory_entries"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL"
        ).fetchone()
    assert row[0] == vector, "the embedding was lost, so the entry left vector search"


def test_an_outcome_without_an_account_leaves_other_accounts_alone(adapter) -> None:
    """The account clause has to hold when the outcome carries no account.

    Nothing in this adapter sets account_id on the outcome row — deployments
    rely on a column default — so an outcome written by a path that never
    established one is the realistic case, not a contrived one. Written as
    `account_id = NEW.account_id OR NEW.account_id IS NULL`, that outcome
    matched entries in EVERY account and reinforced all of them.
    """
    import uuid as _uuid

    import psycopg

    adapter.write(_make_entry(confidence=0.9))
    other_account = str(_uuid.uuid4())
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(
            "UPDATE amfs_memory_entries SET account_id = %s::uuid"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL",
            (other_account,),
        )

    # The outcome carries no account, as this adapter always writes it.
    _commit_success_on(adapter)

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT version, confidence FROM amfs_memory_entries"
            " WHERE key = 'retry-pattern' ORDER BY version"
        ).fetchall()
    assert len(rows) == 1, (
        "an outcome with no account reinforced an entry belonging to an "
        "account, creating a new version across the tenancy boundary"
    )
    assert abs(float(rows[0][1]) - 0.9) < 1e-6, "the other account's entry was modified"


def test_an_outcome_still_reinforces_when_neither_side_has_an_account(adapter) -> None:
    """The single-account case, which is every self-hosted install.

    IS NOT DISTINCT FROM is what keeps this working while still rejecting the
    case above; plain equality would make NULL = NULL unknown and propagation
    would silently never fire for anyone.
    """
    import psycopg

    adapter.write(_make_entry(confidence=0.9))
    _commit_success_on(adapter)

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        row = conn.execute(
            "SELECT confidence FROM amfs_memory_entries"
            " WHERE key = 'retry-pattern' AND superseded_at IS NULL"
        ).fetchone()
    assert abs(float(row[0]) - 0.927) < 1e-6, (
        "propagation stopped firing for entries with no account, which is all "
        "of them in a single-account install"
    )
