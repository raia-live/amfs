"""The Postgres outcome trigger lands on the same numbers as ``amfs_core.evidence``.

Requires a running Postgres instance. Set AMFS_TEST_PG_DSN to enable.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from amfs_core import evidence as ev
from amfs_core.models import AttemptRecord, MemoryEntry, OutcomeRecord, OutcomeType, Provenance

PG_DSN = os.environ.get("AMFS_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(PG_DSN is None, reason="AMFS_TEST_PG_DSN not set")


@pytest.fixture
def adapter():
    from amfs_postgres.adapter import PostgresAdapter

    ns = f"ev-{uuid.uuid4().hex[:8]}"
    a = PostgresAdapter(dsn=PG_DSN, namespace=ns, auto_schema=True)
    yield a
    a.close()


def _entry(key: str, conf: float) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/mod",
        key=key,
        value={"rule": key},
        confidence=conf,
        provenance=Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC)),
    )


def _record(outcome: OutcomeType, keys: list[str], attempts=None, causal=1.0) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_ref=f"task-{uuid.uuid4().hex[:6]}",
        outcome_type=outcome,
        causal_confidence=causal,
        committed_at=datetime.now(UTC),
        causal_entry_keys=keys,
        agent_id="a",
        attempts=attempts or [],
    )


def _expect(entry: MemoryEntry, record: OutcomeRecord) -> MemoryEntry:
    new, _ = ev.apply_record_to_entry(entry, record, model="evidence")
    return new


def test_single_failure_matches_python(adapter, monkeypatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    written = adapter.write(_entry("rule", 0.7))
    record = _record(OutcomeType.FAILURE, ["svc/mod/rule"])
    updated = adapter.commit_outcome(record)
    assert len(updated) == 1
    got = updated[0]
    want = _expect(written, record)
    assert got.version == 2
    assert got.confidence == pytest.approx(want.confidence, abs=1e-4)
    assert got.evidence_failure == pytest.approx(want.evidence_failure, abs=1e-4)
    assert got.failure_count == 1 and got.success_count == 0
    assert got.prior_confidence == pytest.approx(0.7, abs=1e-4)
    assert got.last_outcome == "failure"
    assert got.discredited_at is not None
    assert got.evidence_status == "discredited"


def test_attempt_then_success_credits_each_side(adapter, monkeypatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    stale = adapter.write(_entry("stale", 0.9))
    good = adapter.write(_entry("good", 0.7))
    record = _record(
        OutcomeType.SUCCESS,
        ["svc/mod/good"],
        attempts=[
            AttemptRecord(
                attempt=1,
                outcome_type=OutcomeType.FAILURE,
                causal_entry_keys=["svc/mod/stale"],
                action_indices=[0],
            )
        ],
    )
    updated = {e.key: e for e in adapter.commit_outcome(record)}
    assert set(updated) == {"stale", "good"}
    assert updated["stale"].confidence == pytest.approx(_expect(stale, record).confidence, abs=1e-4)
    assert updated["stale"].discredited_at is not None
    assert updated["good"].confidence == pytest.approx(_expect(good, record).confidence, abs=1e-4)
    assert updated["good"].success_count == 1 and updated["good"].failure_count == 0

    # The outcome row carries the attempts back out.
    outcomes = adapter.list_outcomes(outcome_ref=record.outcome_ref)
    assert len(outcomes) == 1
    assert [a.causal_entry_keys for a in outcomes[0].attempts] == [["svc/mod/stale"]]


def test_row_copy_preserves_embedding_and_recall(adapter, monkeypatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("keep", 0.8))
    adapter.increment_recall_count("svc/mod", "keep")
    adapter.increment_recall_count("svc/mod", "keep")
    adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["svc/mod/keep"]))
    after = adapter.read("svc/mod", "keep")
    assert after.version == 2
    assert after.recall_count == 2


def test_long_validated_rule_survives_one_failure_not_two(adapter, monkeypatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("rule", 0.7))
    for _ in range(8):
        adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["svc/mod/rule"]))
    assert adapter.read("svc/mod", "rule").confidence > 0.9
    adapter.commit_outcome(_record(OutcomeType.FAILURE, ["svc/mod/rule"]))
    first = adapter.read("svc/mod", "rule")
    assert first.discredited_at is None and first.evidence_status == "contested"
    adapter.commit_outcome(_record(OutcomeType.FAILURE, ["svc/mod/rule"]))
    second = adapter.read("svc/mod", "rule")
    assert second.discredited_at is not None
    assert second.confidence < ev.DISCREDIT_THRESHOLD


def test_multiplicative_flag_reaches_the_trigger(adapter, monkeypatch) -> None:
    monkeypatch.setenv(ev.OUTCOME_MODEL_ENV, "multiplicative")
    adapter.write(_entry("legacy", 0.9))
    adapter.commit_outcome(_record(OutcomeType.FAILURE, ["svc/mod/legacy"]))
    got = adapter.read("svc/mod", "legacy")
    assert got.confidence == pytest.approx(0.81, abs=1e-4)
    assert got.failure_count == 1


def test_fresh_write_resets_evidence(adapter, monkeypatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("rule", 0.7))
    adapter.commit_outcome(_record(OutcomeType.FAILURE, ["svc/mod/rule"]))
    assert adapter.read("svc/mod", "rule").discredited_at is not None
    # The author rewrites the rule: a new claim, tested from scratch.
    adapter.write(_entry("rule", 0.7))
    fresh = adapter.read("svc/mod", "rule")
    assert fresh.version == 3
    assert fresh.discredited_at is None
    assert fresh.failure_count == 0
    assert fresh.evidence_status == "untested"
