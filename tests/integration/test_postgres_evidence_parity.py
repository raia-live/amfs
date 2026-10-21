"""t2-11e: property-based parity between the SQL evidence functions (migrations
008/009/010, as re-rendered by 012) and ``amfs_core.evidence``.

Random outcome sequences — every known type, causal
confidence over its range, one to three cited keys with the credit split,
failed attempts before the terminal outcome, two agents for the validator
list, a ``verified_by`` on some outcomes for the verification factor — are
applied through the adapter (the trigger) and through the Python model, and
the two records must agree on every evidence field after every step. Credit
by uplift is checked both ways: the uplift the trigger refreshed on each
cited row must equal ``amfs_core.evidence.uplift_counts`` over the same
outcomes, and the Python model is then handed that uplift for the next step,
as the adapter's own rows carry it. The seeds are fixed so a disagreement is
reproducible by seed; raise ``AMFS_PARITY_CASES`` to run more.

Requires a running Postgres instance. Set AMFS_TEST_PG_DSN to enable.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import UTC, datetime

import pytest
from amfs_core import evidence as ev
from amfs_core.models import AttemptRecord, MemoryEntry, OutcomeRecord, OutcomeType, Provenance

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(PG_DSN is None, reason="AMFS_TEST_PG_DSN not set")

CASES = int(os.environ.get("AMFS_PARITY_CASES", "40"))
STEPS = 8
KEYS = ("rule-a", "rule-b", "rule-c")
#: Every known type. An unknown one cannot reach the trigger through the
#: adapter — ``OutcomeRecord`` rejects it — so "unknown is not evidence" is
#: covered by the model's own unit tests, not here.
TYPES = [t.value for t in OutcomeType]
#: The stored columns are NUMERIC(.,4) (confidence 6,4; evidence masses 10,4),
#: so each SQL step rounds to four places where Python carries the float. Under
#: the 0.8 decay the rounding compounds to at most 0.5e-4 / (1 - 0.8) = 2.5e-4
#: on a mass and less on the posterior; anything over this bound is arithmetic,
#: not storage. The discredit flag, the label and the counts compare exactly.
TOL = 5e-4


@pytest.fixture
def adapter():
    from amfs_postgres.adapter import PostgresAdapter

    ns = f"parity-{uuid.uuid4().hex[:8]}"
    a = PostgresAdapter(dsn=PG_DSN, namespace=ns, auto_schema=True)
    yield a
    a.close()


def _entry(key: str, conf: float) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/mod",
        key=key,
        value={"rule": key},
        confidence=conf,
        provenance=Provenance(agent_id="writer", session_id="s0", written_at=datetime.now(UTC)),
    )


def _random_record(rng: random.Random) -> OutcomeRecord:
    cited = rng.sample(KEYS, k=rng.randint(1, len(KEYS)))
    attempts = []
    for i in range(rng.choice((0, 0, 1, 2))):
        attempts.append(
            AttemptRecord(
                attempt=i + 1,
                outcome_type=rng.choice(("minor_failure", "failure", "regression")),
                causal_entry_keys=[f"svc/mod/{k}" for k in rng.sample(KEYS, k=rng.randint(1, 2))],
            )
        )
    verified_by = rng.choice((None, None, "ci", "monitor:latency", "reviewer"))
    return OutcomeRecord(
        outcome_ref=f"t-{uuid.uuid4().hex[:6]}",
        outcome_type=rng.choice(TYPES),
        causal_confidence=round(rng.choice((0.0, 0.25, 0.5, 0.75, 1.0, rng.random())), 4),
        committed_at=datetime.now(UTC),
        causal_entry_keys=[f"svc/mod/{k}" for k in cited],
        agent_id=rng.choice(("agent-1", "agent-2")),
        attempts=attempts,
        session_metadata={"attributes": {"verified_by": verified_by}} if verified_by else {},
    )


def _with_uplift(entry: MemoryEntry, got: MemoryEntry) -> MemoryEntry:
    """The Python model reads credit off the entry; the adapter maintains it."""
    return entry.model_copy(update={"uplift": got.uplift, "uplift_sessions": got.uplift_sessions})


def _assert_same(got: MemoryEntry, want: MemoryEntry, *, seed: int, step: int, key: str) -> None:
    where = f"seed={seed} step={step} key={key}"
    assert got.confidence == pytest.approx(want.confidence, abs=TOL), where
    assert got.evidence_success == pytest.approx(want.evidence_success, abs=TOL), where
    assert got.evidence_failure == pytest.approx(want.evidence_failure, abs=TOL), where
    assert (got.success_count, got.failure_count) == (want.success_count, want.failure_count), where
    assert got.outcome_count == want.outcome_count, where
    assert (got.prior_confidence or 0.0) == pytest.approx(want.prior_confidence or 0.0, abs=TOL), (
        where
    )
    assert got.last_outcome == want.last_outcome, where
    assert (got.discredited_at is None) == (want.discredited_at is None), where
    assert got.evidence_status == want.evidence_status, where
    assert list(got.validators or []) == list(want.validators or []), where


@pytest.mark.parametrize("seed", range(CASES))
def test_sql_and_python_agree_on_random_outcome_sequences(adapter, monkeypatch, seed: int) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    rng = random.Random(seed)
    want: dict[str, MemoryEntry] = {}
    for key in KEYS:
        want[key] = adapter.write(_entry(key, round(rng.uniform(0.05, 0.95), 3)))
    history: list[OutcomeRecord] = []
    for step in range(STEPS):
        record = _random_record(rng)
        adapter.commit_outcome(record)
        history.append(record)
        cited = {k for _ep, k in ev.cited_entries(record)}
        reference = ev.uplift_counts(history, "svc/mod", KEYS)
        for key in KEYS:
            want[key], _ = ev.apply_record_to_entry(want[key], record, model="evidence")
            got = adapter.read("svc/mod", key)
            assert got is not None
            _assert_same(got, want[key], seed=seed, step=step, key=key)
            if key in cited:
                # The trigger refreshed this row's uplift from the window that
                # now includes the outcome just committed.
                uplift, sessions = reference[key]
                where = f"seed={seed} step={step} key={key}"
                assert got.uplift_sessions == sessions, where
                if uplift is None:
                    assert got.uplift is None, where
                else:
                    assert got.uplift == pytest.approx(uplift, abs=1e-9), where
            want[key] = _with_uplift(want[key], got)


def test_multiplicative_model_agrees_too(adapter, monkeypatch) -> None:
    """The legacy model is kept behind AMFS_OUTCOME_MODEL=multiplicative; the
    multipliers in SQL come from the same table as Python's."""
    monkeypatch.setenv(ev.OUTCOME_MODEL_ENV, "multiplicative")
    rng = random.Random(7)
    want = {key: adapter.write(_entry(key, 0.7)) for key in KEYS}
    for step in range(STEPS):
        record = _random_record(rng)
        adapter.commit_outcome(record)
        for key in KEYS:
            want[key], _ = ev.apply_record_to_entry(want[key], record, model="multiplicative")
            got = adapter.read("svc/mod", key)
            want[key] = _with_uplift(want[key], got)
            assert got.confidence == pytest.approx(want[key].confidence, abs=TOL), (step, key)
            assert (got.success_count, got.failure_count) == (
                want[key].success_count,
                want[key].failure_count,
            )
