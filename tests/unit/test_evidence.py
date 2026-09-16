"""The evidence model: what an outcome does to an entry's confidence.

The numbers pinned here are the contract between ``amfs_core.evidence`` (used
by the filesystem and S3 adapters) and ``migrations/008_outcome_evidence.sql``
(the Postgres trigger). ``tests/integration/test_postgres_outcome_evidence.py``
asserts the trigger lands on the same values.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from amfs_core import evidence as ev
from amfs_core.models import AttemptRecord, MemoryEntry, OutcomeRecord, OutcomeType, Provenance


def _entry(conf: float, **kw) -> MemoryEntry:
    return MemoryEntry(
        entity_path="svc/mod",
        key="rule",
        value="x",
        confidence=conf,
        provenance=Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC)),
        **kw,
    )


def _record(
    outcome: OutcomeType, keys: list[str], attempts: list[AttemptRecord] | None = None
) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_ref="task-1",
        outcome_type=outcome,
        committed_at=datetime.now(UTC),
        causal_entry_keys=keys,
        agent_id="a",
        attempts=attempts or [],
    )


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


class TestSingleOutcome:
    def test_first_failure_on_fresh_entry_drops_below_gate(self) -> None:
        # w = 2.0 * 1.0 / 1 * (1 + |0 - 0.7|) = 3.4
        # conf = (2*0.7 + 0) / (2 + 0 + 3.4) = 1.4 / 5.4
        upd = ev.apply_outcome(_entry(0.7), OutcomeType.FAILURE)
        assert upd.confidence == pytest.approx(1.4 / 5.4, abs=1e-6)
        assert upd.confidence < ev.DISCREDIT_THRESHOLD
        assert upd.discredited is True
        assert upd.failure_count == 1 and upd.success_count == 0
        assert upd.prior_confidence == pytest.approx(0.7)

    def test_first_success_on_fresh_entry_lifts(self) -> None:
        # w = 1.0 * (1 + |1 - 0.7|) = 1.3 ; conf = (1.4 + 1.3) / (2 + 1.3)
        upd = ev.apply_outcome(_entry(0.7), OutcomeType.SUCCESS)
        assert upd.confidence == pytest.approx(2.7 / 3.3, abs=1e-6)
        assert upd.discredited is False
        assert upd.success_count == 1

    def test_surprise_scales_with_prior_trust(self) -> None:
        low = ev.evidence_weight(OutcomeType.FAILURE, current_confidence=0.5)
        high = ev.evidence_weight(OutcomeType.FAILURE, current_confidence=0.95)
        assert high == pytest.approx(2.0 * 1.95) and low == pytest.approx(2.0 * 1.5)
        assert high > low

    def test_credit_split_divides_weight(self) -> None:
        one = ev.evidence_weight(OutcomeType.SUCCESS, current_confidence=0.7, n_causal=1)
        eight = ev.evidence_weight(OutcomeType.SUCCESS, current_confidence=0.7, n_causal=8)
        assert eight == pytest.approx(one / 8)

    def test_severity_ordering(self) -> None:
        assert (
            ev.severity(OutcomeType.SUCCESS)
            < ev.severity(OutcomeType.MINOR_FAILURE)
            < ev.severity(OutcomeType.FAILURE)
            < ev.severity(OutcomeType.CRITICAL_FAILURE)
        )

    def test_success_never_sets_discredited(self) -> None:
        # Deeply discredited entry; one success does not clear it but must not
        # re-stamp it either.
        e = _entry(
            0.2,
            evidence_failure=6.0,
            prior_confidence=0.7,
            failure_count=3,
            discredited_at=datetime.now(UTC),
        )
        upd = ev.apply_outcome(e, OutcomeType.SUCCESS)
        assert upd.confidence < ev.DISCREDIT_THRESHOLD
        assert upd.discredited is True  # unchanged, still below threshold


class TestRegimeChangeTimescale:
    """A long-validated rule stops working: how many failures until it is gated?"""

    def _validated(self) -> MemoryEntry:
        e = _entry(0.7)
        for _ in range(10):
            upd = ev.apply_outcome(e, OutcomeType.SUCCESS)
            e = e.model_copy(update=upd.as_entry_update(datetime.now(UTC)))
        assert e.confidence > 0.9
        return e

    def test_one_failure_contests_two_discredit(self) -> None:
        e = self._validated()
        first = ev.apply_outcome(e, OutcomeType.FAILURE)
        assert ev.DISCREDIT_THRESHOLD <= first.confidence < 0.7
        assert first.discredited is False
        e = e.model_copy(update=first.as_entry_update(datetime.now(UTC)))
        second = ev.apply_outcome(e, OutcomeType.FAILURE)
        assert second.confidence < ev.DISCREDIT_THRESHOLD
        assert second.discredited is True

    def test_recovery_clears_discredit(self) -> None:
        e = _entry(0.7)
        e = e.model_copy(
            update=ev.apply_outcome(e, OutcomeType.FAILURE).as_entry_update(datetime.now(UTC))
        )
        assert e.discredited_at is not None
        for _ in range(3):
            e = e.model_copy(
                update=ev.apply_outcome(e, OutcomeType.SUCCESS).as_entry_update(datetime.now(UTC))
            )
        assert e.confidence >= ev.DISCREDIT_THRESHOLD
        assert e.discredited_at is None
        assert e.evidence_status == "contested"

    def test_generic_entry_saturates_not_at_one(self) -> None:
        # Cited in every session alongside seven others: it cannot climb to 1.0
        # the way *1.03 forever did.
        e = _entry(0.7)
        for _ in range(50):
            upd = ev.apply_outcome(e, OutcomeType.SUCCESS, n_causal=8)
            e = e.model_copy(update=upd.as_entry_update(datetime.now(UTC)))
        assert 0.7 < e.confidence < 0.85


class TestRecordApplication:
    def test_attempt_failure_then_success_splits_credit(self) -> None:
        stale = _entry(0.9).model_copy(update={"key": "stale"})
        good = _entry(0.7).model_copy(update={"key": "good"})
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
        new_stale, steps_s = ev.apply_record_to_entry(stale, record)
        new_good, steps_g = ev.apply_record_to_entry(good, record)
        assert len(steps_s) == 1 and steps_s[0].last_outcome == "failure"
        assert (
            new_stale.confidence < ev.DISCREDIT_THRESHOLD and new_stale.discredited_at is not None
        )
        assert len(steps_g) == 1 and steps_g[0].last_outcome == "success"
        assert new_good.confidence > 0.7
        assert new_stale.outcome_count == 1 and new_good.outcome_count == 1

    def test_entry_in_both_attempt_and_final_gets_both_steps(self) -> None:
        e = _entry(0.7)
        record = _record(
            OutcomeType.SUCCESS,
            ["svc/mod/rule"],
            attempts=[AttemptRecord(attempt=1, causal_entry_keys=["svc/mod/rule"])],
        )
        new, steps = ev.apply_record_to_entry(e, record)
        assert [s.last_outcome for s in steps] == ["minor_failure", "success"]
        assert new.failure_count == 1 and new.success_count == 1
        assert new.evidence_status == "contested"

    def test_cited_entries_covers_attempts(self) -> None:
        record = _record(
            OutcomeType.SUCCESS,
            ["a/b/c"],
            attempts=[AttemptRecord(attempt=1, causal_entry_keys=["a/b/d", "bad"])],
        )
        assert ev.cited_entries(record) == [("a/b", "d"), ("a/b", "c")]

    def test_multiplicative_flag_restores_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ev.OUTCOME_MODEL_ENV, "multiplicative")
        new, _ = ev.apply_record_to_entry(
            _entry(0.9), _record(OutcomeType.FAILURE, ["svc/mod/rule"])
        )
        assert new.confidence == pytest.approx(0.81)
        assert new.failure_count == 1


class TestContrastLesson:
    def test_fail_then_succeed_produces_lesson(self) -> None:
        record = _record(
            OutcomeType.SUCCESS,
            ["svc/mod/good"],
            attempts=[
                AttemptRecord(
                    attempt=1, causal_entry_keys=["svc/mod/stale"], summary="tried restart"
                )
            ],
        )
        lesson = ev.contrast_lesson(record)
        assert lesson is not None
        assert lesson["avoid"] == ["svc/mod/stale"]
        assert lesson["resolved_with"] == ["svc/mod/good"]
        assert lesson["attempt_summaries"] == ["tried restart"]
        assert ev.is_synthetic_key(ev.contrast_lesson_key(record.outcome_ref))

    def test_no_lesson_without_failed_attempts_or_on_failure(self) -> None:
        assert ev.contrast_lesson(_record(OutcomeType.SUCCESS, ["a/b/c"])) is None
        rec = _record(
            OutcomeType.FAILURE,
            ["a/b/c"],
            attempts=[AttemptRecord(attempt=1, causal_entry_keys=["a/b/d"])],
        )
        assert ev.contrast_lesson(rec) is None


class TestStatus:
    def test_vocabulary(self) -> None:
        assert _entry(0.7).evidence_status == "untested"
        assert _entry(0.7, success_count=2).evidence_status == "validated"
        assert _entry(0.7, success_count=2, failure_count=1).evidence_status == "contested"
        assert (
            _entry(0.3, failure_count=1, discredited_at=datetime.now(UTC)).evidence_status
            == "discredited"
        )
        # Pre-split-count history: direction read off the confidence left behind.
        assert _entry(0.9, outcome_count=3).evidence_status == "validated"
        assert _entry(0.3, outcome_count=3).evidence_status == "contested"
