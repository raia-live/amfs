"""Postgres adapter tests — runs the full adapter contract suite.

Requires a running Postgres instance. Set AMFS_TEST_PG_DSN to enable.
Example: AMFS_TEST_PG_DSN=postgresql://localhost/amfs_test pytest tests/integration/test_postgres_adapter.py
"""

from __future__ import annotations

import os

import pytest

from tests.integration.adapter_contract import AdapterContractTests

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
