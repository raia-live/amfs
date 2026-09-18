"""Migration 009 on a real Postgres: the outcome row carries the action record,
the trigger applies first-strike tolerance and maintains validators to the same
numbers as ``amfs_core.evidence``, and the priors queries scope by entity.

Requires a running Postgres with pgvector. Set AMFS_TEST_PG_DSN to enable.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from amfs_core import evidence as ev
from amfs_core.models import MemoryEntry, OutcomeRecord, OutcomeType, Provenance

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(PG_DSN is None, reason="AMFS_TEST_PG_DSN not set")


class _Embedder:
    """Deterministic 384-dim embedder: one hot per distinct text, so identical
    situations are identical vectors and different ones are orthogonal."""

    dim = 384

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        vec[hash(text.strip().lower()) % self.dim] = 1.0
        return vec

    def embed_value(self, value):
        return self.embed(str(value))

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def adapter(monkeypatch):
    import psycopg
    from amfs_postgres.adapter import PostgresAdapter

    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    # The embedding columns are shared across namespaces and fixed in dimension once
    # provisioned; other integration tests provision them at 8. Start from a clean pair of
    # tables so this file's 384 is what both columns get (same pattern as
    # test_postgres_embeddings.py).
    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("DROP TABLE IF EXISTS amfs_outcomes CASCADE")
    conn.execute("DROP TABLE IF EXISTS amfs_memory_entries CASCADE")
    conn.close()
    ns = f"ap-{uuid.uuid4().hex[:8]}"
    a = PostgresAdapter(dsn=PG_DSN, namespace=ns, auto_schema=True, embedder=_Embedder())
    a.ensure_embedding_column(384)
    a.ensure_outcome_embedding_column()
    yield a
    a.close()


def _entry(key: str, conf: float = 0.7, path: str = "acme/support") -> MemoryEntry:
    return MemoryEntry(
        entity_path=path, key=key, value={"rule": key}, confidence=conf,
        provenance=Provenance(agent_id="a1", session_id="s", written_at=datetime.now(UTC)),
    )


def _record(outcome, keys, *, agent="a1", actions=None, paths=None, situation=None, task_input=None):
    return OutcomeRecord(
        outcome_ref=f"t-{uuid.uuid4().hex[:6]}", outcome_type=outcome, causal_confidence=1.0,
        committed_at=datetime.now(UTC), causal_entry_keys=keys, agent_id=agent,
        actions_taken=actions or [], entity_paths=paths or [], situation=situation,
        task_input=task_input,
    )


def test_columns_exist_and_the_outcome_row_carries_the_action_record(adapter) -> None:
    assert adapter._has_action_cols and adapter._has_validators_col and adapter._has_outcome_embedding_col
    adapter.write(_entry("rule"))
    rec = _record(
        OutcomeType.SUCCESS, ["acme/support/rule"],
        actions=[{"index": 1, "action_key": "resolve:resend_email", "tool_name": "resolve", "success": True, "attempt": None}],
        paths=["acme/support"], situation="card declined", task_input="customer says card declined",
    )
    adapter.commit_outcome(rec)
    rows = adapter.list_outcomes(entity_path="acme/support")
    assert len(rows) == 1
    assert rows[0].actions_taken[0]["action_key"] == "resolve:resend_email"
    assert rows[0].entity_paths == ["acme/support"] and rows[0].situation == "card declined"


def test_trigger_first_strike_and_two_strike_discredit_match_python(adapter) -> None:
    adapter.write(_entry("rule"))
    seq = [OutcomeType.SUCCESS] * 4 + [OutcomeType.FAILURE, OutcomeType.FAILURE]
    py = adapter.read("acme/support", "rule")
    for o in seq:
        rec = _record(o, ["acme/support/rule"])
        adapter.commit_outcome(rec)
        py, _ = ev.apply_record_to_entry(py, rec)
        pg = adapter.read("acme/support", "rule")
        assert pg.confidence == pytest.approx(py.confidence, abs=1e-3), o
        assert (pg.discredited_at is None) == (py.discredited_at is None), o
    # 4 wins then one loss stayed out of the discredit zone; the second loss did not.
    hist = adapter.list_outcomes(entity_path="acme/support")
    assert len(hist) == 6
    assert pg.discredited_at is not None and pg.confidence < ev.DISCREDIT_THRESHOLD


def test_trigger_maintains_validators(adapter) -> None:
    adapter.write(_entry("rule"))
    for agent in ("a1", "a2", "a2", "a3"):
        adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["acme/support/rule"], agent=agent))
    assert adapter.read("acme/support", "rule").validators == ["a1", "a2", "a3"]
    adapter.commit_outcome(_record(OutcomeType.FAILURE, ["acme/support/rule"], agent="a4"))
    assert adapter.read("acme/support", "rule").validators == ["a1", "a2", "a3"]
    adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["acme/support/rule"], agent="a1"))
    assert adapter.read("acme/support", "rule").validators == ["a2", "a3", "a1"]
    # Capped at ten, oldest dropped.
    for i in range(12):
        adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["acme/support/rule"], agent=f"v{i}"))
    vals = adapter.read("acme/support", "rule").validators
    assert len(vals) == ev.MAX_VALIDATORS and vals[-1] == "v11"
    # A restatement of the same claim keeps them; a new claim starts clean.
    adapter.write(_entry("rule"))
    assert adapter.read("acme/support", "rule").validators == vals
    adapter.write(_entry("rule").model_copy(update={"value": {"rule": "something else"}}))
    assert adapter.read("acme/support", "rule").validators == []


def test_similar_outcomes_and_action_stats_scope_by_entity(adapter) -> None:
    emb = _Embedder()
    win = {"index": 0, "action_key": "resolve:resend_email", "tool_name": "resolve", "success": True, "attempt": None}
    lose = {"index": 0, "action_key": "resolve:update_payment_method", "tool_name": "resolve", "success": False, "attempt": 1}
    for agent in ("a1", "a2"):
        adapter.commit_outcome(_record(OutcomeType.SUCCESS, [], agent=agent, actions=[lose, win],
                                       paths=["acme/support"], situation="card declined"))
    adapter.commit_outcome(_record(OutcomeType.FAILURE, [], agent="a3", actions=[lose],
                                   paths=["acme/billing"], situation="card declined"))
    adapter.commit_outcome(_record(OutcomeType.SUCCESS, [], agent="a1", actions=[win],
                                   paths=["acme/support"], situation="shipping delayed"))

    similar = adapter.similar_outcomes("acme/support", emb.embed("card declined"), k=10, min_similarity=0.9)
    assert len(similar) == 2 and all(r["similarity"] > 0.99 for r in similar)
    assert {r["agent_id"] for r in similar} == {"a1", "a2"}
    # The billing outcome is about another entity; the shipping one is another task.
    stats = adapter.action_stats("acme/support")
    assert len(stats) == 3
    assert not adapter.similar_outcomes("acme/ops", emb.embed("card declined"))

    from amfs_core.actions import aggregate_priors

    pr = aggregate_priors(similar, candidate_actions=["resolve:resend_email", "resolve:update_payment_method", "resolve:refund"])
    by = {t["action_key"]: t for t in pr["tried"]}
    assert by["resolve:resend_email"]["won"] == 2 and by["resolve:update_payment_method"]["lost"] == 2
    assert pr["untried"] == ["resolve:refund"]


def test_migration_is_idempotent_on_a_bootstrapped_database(adapter) -> None:
    """Re-applying 008+009 (a restart) changes nothing and keeps one step overload."""
    adapter._apply_schema_if_needed() if hasattr(adapter, "_apply_schema_if_needed") else None
    with adapter._pool.connection() as conn, conn.cursor() as cur:
        from amfs_postgres.adapter import _ACTION_PRIORS_SQL, _OUTCOME_EVIDENCE_SQL

        cur.execute(_OUTCOME_EVIDENCE_SQL)
        cur.execute(_ACTION_PRIORS_SQL)
        cur.execute("SELECT count(*) AS n FROM pg_proc WHERE proname = 'amfs_apply_outcome_step'")
        assert cur.fetchone()["n"] == 1
        cur.execute("SELECT pronargs FROM pg_proc WHERE proname = 'amfs_apply_outcome_step'")
        assert cur.fetchone()["pronargs"] == 8


def test_evidence_near_counts_outcomes_not_credits(adapter) -> None:
    """One task that failed on a rule and then succeeded with the same rule
    credits it twice — a failure from the attempt, a success from the terminal
    outcome — but is one nearby task. ``n`` must say one, or a single task
    would clear LOCAL_EVIDENCE_MIN_N by itself and override the pooled record."""
    from amfs_core.models import AttemptRecord

    emb = _Embedder()
    adapter.write(_entry("rule"))
    key = "acme/support/rule"
    rec = _record(OutcomeType.SUCCESS, [key], paths=["acme/support"], situation="card declined")
    rec = rec.model_copy(update={"attempts": [
        AttemptRecord(attempt=1, outcome_type=OutcomeType.FAILURE, causal_entry_keys=[key], action_indices=[])
    ]})
    adapter.commit_outcome(rec)

    near = adapter.evidence_near([key], emb.embed("card declined"))
    assert near[key]["n"] == 1, near
    assert near[key]["success"] == pytest.approx(1.0, abs=1e-6)
    assert near[key]["failure"] == pytest.approx(1.0, abs=1e-6)

    # A second task like it is the second outcome.
    adapter.commit_outcome(_record(OutcomeType.SUCCESS, [key], paths=["acme/support"], situation="card declined"))
    near = adapter.evidence_near([key], emb.embed("card declined"))
    assert near[key]["n"] == 2 and near[key]["success"] == pytest.approx(2.0, abs=1e-6)
    # A task about something else is orthogonal here and does not count.
    adapter.commit_outcome(_record(OutcomeType.FAILURE, [key], paths=["acme/support"], situation="shipping delayed"))
    near = adapter.evidence_near([key], emb.embed("card declined"), min_similarity=0.5)
    assert near[key]["n"] == 2 and near[key]["failure"] == pytest.approx(1.0, abs=1e-6)
