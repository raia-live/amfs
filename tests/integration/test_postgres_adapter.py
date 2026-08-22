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
