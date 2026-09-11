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
