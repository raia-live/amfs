"""Upgrade and rollback safety of the step that knows an entry's role.

Migration 012 gained a tenth argument on ``amfs_apply_outcome_step`` (the keys
that were only in view) and two columns on ``amfs_outcomes``. A live database
is upgraded while outcomes are being committed, and may be rolled back to the
previous release, so:

* the nine-argument step is still there and means "nothing in view";
* a table that does not have the two columns still takes outcomes;
* applying the file again changes nothing.

The arithmetic itself is covered against the Python model in
``test_postgres_evidence_parity.py``.

Requires a running Postgres instance. Set AMFS_TEST_PG_DSN to enable.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from amfs_core import evidence as ev
from amfs_core.models import AttemptRecord, MemoryEntry, OutcomeRecord, OutcomeType, Provenance

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(PG_DSN is None, reason="AMFS_TEST_PG_DSN not set")

SCOPE = "svc/mod"


def _adapter(ns: str):
    from amfs_postgres.adapter import PostgresAdapter

    return PostgresAdapter(dsn=PG_DSN, namespace=ns, auto_schema=True)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    a = _adapter(f"roles-{uuid.uuid4().hex[:8]}")
    yield a
    a.close()


def _entry(key: str, conf: float = 1.0) -> MemoryEntry:
    return MemoryEntry(
        entity_path=SCOPE,
        key=key,
        value={"rule": key},
        confidence=conf,
        provenance=Provenance(agent_id="writer", session_id="s0", written_at=datetime.now(UTC)),
    )


def _record(outcome: OutcomeType, keys: list[str], **extra) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_ref=f"t-{uuid.uuid4().hex[:6]}",
        outcome_type=outcome,
        committed_at=datetime.now(UTC),
        causal_entry_keys=[f"{SCOPE}/{k}" for k in keys],
        agent_id="agent-1",
        **extra,
    )


def _step(cur, namespace: str, key: str, *, in_view: str | None) -> None:
    args = [namespace, None, [f"{SCOPE}/{key}"], "failure", 1.0, "evidence", "{}", "agent-1", False]
    sql = (
        "SELECT amfs_apply_outcome_step(%s::text, %s::uuid, %s::text[], %s::text, "
        "%s::numeric, %s::text, %s::jsonb, %s::text, %s::boolean"
    )
    if in_view == "omitted":
        cur.execute(sql + ")", args)
    else:
        cur.execute(sql + ", %s::text[])", [*args, [f"{SCOPE}/{in_view}"] if in_view else None])


def test_both_steps_exist_and_the_nine_argument_one_has_nothing_in_view(adapter) -> None:
    """What the previous release's trigger calls. It must exist at every
    moment of an upgrade and after a rollback, and weigh a failure in full."""
    for key in ("old-call", "null-view", "in-view"):
        adapter.write(_entry(key))
    with adapter._pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT array_agg(pronargs ORDER BY pronargs) AS n FROM pg_proc "
            "WHERE proname = 'amfs_apply_outcome_step'"
        )
        assert {9, 10} <= set(cur.fetchone()["n"])
        _step(cur, adapter._namespace, "old-call", in_view="omitted")
        _step(cur, adapter._namespace, "null-view", in_view=None)
        _step(cur, adapter._namespace, "in-view", in_view="in-view")
    old, null, soft = (adapter.read(SCOPE, k) for k in ("old-call", "null-view", "in-view"))
    assert old.confidence == pytest.approx(1 / 3, abs=5e-4)
    assert old.discredited_at is not None
    assert (null.confidence, null.failure_count, null.discredited_at is None) == (
        old.confidence,
        old.failure_count,
        old.discredited_at is None,
    )
    assert soft.confidence == pytest.approx(ev.DISCREDIT_THRESHOLD, abs=1e-9)
    assert soft.failure_count == 1 and soft.discredited_at is None


def test_the_terminal_roles_are_stored_and_applied(adapter) -> None:
    for key in ("named", "seen", "skipped"):
        adapter.write(_entry(key))
    record = _record(
        OutcomeType.FAILURE,
        ["named", "seen"],
        in_view_entry_keys=[f"{SCOPE}/seen"],
        not_followed_entry_keys=[f"{SCOPE}/skipped"],
    )
    adapter.commit_outcome(record)
    named, seen, skipped = (adapter.read(SCOPE, k) for k in ("named", "seen", "skipped"))
    # Two cited keys split the failure: 2 / (2 + 2*1) for the named one.
    assert named.confidence == pytest.approx(0.5, abs=5e-4)
    assert seen.confidence == pytest.approx(ev.DISCREDIT_THRESHOLD, abs=1e-9)
    assert seen.discredited_at is None and seen.failure_count == 1
    assert skipped.success_count == 1 and skipped.failure_count == 0
    assert list(skipped.validators or []) == []
    kept = adapter.list_outcomes(outcome_ref=record.outcome_ref)[0]
    assert kept.in_view_entry_keys == [f"{SCOPE}/seen"]
    assert kept.not_followed_entry_keys == [f"{SCOPE}/skipped"]
    assert kept.causal_entry_keys == [f"{SCOPE}/named", f"{SCOPE}/seen"]


def test_a_table_without_the_role_columns_still_takes_outcomes(adapter) -> None:
    """The columns are added by ``ALTER TABLE``, which can lose a lock race
    while the functions after it are replaced. The trigger must then read
    "no roles" off the row instead of failing every outcome insert."""
    with adapter._pool.connection() as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE amfs_outcomes DROP COLUMN IF EXISTS in_view_entry_keys")
        cur.execute("ALTER TABLE amfs_outcomes DROP COLUMN IF EXISTS not_followed_entry_keys")
    try:
        from amfs_postgres.adapter import PostgresAdapter

        bare = PostgresAdapter(dsn=PG_DSN, namespace=adapter._namespace, auto_schema=False)
        try:
            assert bare._has_entry_role_cols is False
            bare.write(_entry("seen"))
            bare.write(_entry("tried"))
            record = _record(
                OutcomeType.FAILURE,
                ["seen"],
                in_view_entry_keys=[f"{SCOPE}/seen"],
                attempts=[
                    AttemptRecord(
                        attempt=1,
                        outcome_type=OutcomeType.FAILURE,
                        causal_entry_keys=[f"{SCOPE}/tried"],
                        in_view_entry_keys=[f"{SCOPE}/tried"],
                    )
                ],
            )
            bare.commit_outcome(record)
            seen, tried = bare.read(SCOPE, "seen"), bare.read(SCOPE, "tried")
            # No column to carry the terminal step's role: charged in full, as
            # the previous release would have. The attempt carries its own.
            assert seen.confidence == pytest.approx(1 / 3, abs=5e-4)
            assert seen.discredited_at is not None
            assert tried.confidence == pytest.approx(ev.DISCREDIT_THRESHOLD, abs=1e-9)
            assert tried.discredited_at is None
            assert bare.list_outcomes(outcome_ref=record.outcome_ref)[0].in_view_entry_keys == []
        finally:
            bare.close()
    finally:
        with adapter._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS in_view_entry_keys TEXT[]"
            )
            cur.execute(
                "ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS not_followed_entry_keys TEXT[]"
            )


def test_applying_the_file_again_changes_nothing(adapter) -> None:
    import amfs_postgres

    sql = (
        Path(amfs_postgres.__file__).parent / "migrations" / "012_evidence_from_model.sql"
    ).read_text(encoding="utf-8")

    def overloads(cur) -> list[int]:
        cur.execute(
            "SELECT array_agg(pronargs ORDER BY pronargs) AS n FROM pg_proc "
            "WHERE proname = 'amfs_apply_outcome_step'"
        )
        return list(cur.fetchone()["n"])

    with adapter._pool.connection() as conn, conn.cursor() as cur:
        before = overloads(cur)
        cur.execute(sql)
        cur.execute(sql)
        assert overloads(cur) == before
        assert 9 in before and 10 in before and 8 not in before
    adapter.write(_entry("seen"))
    adapter.commit_outcome(
        _record(OutcomeType.FAILURE, ["seen"], in_view_entry_keys=[f"{SCOPE}/seen"])
    )
    assert adapter.read(SCOPE, "seen").discredited_at is None
