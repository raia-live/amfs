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

    def test_lesson_carries_the_actions_and_the_task_when_the_record_has_them(self) -> None:
        """What failed and what worked, as action keys, plus a task excerpt:
        the fields a near-identical task needs to act on the pair. Optional —
        a record without ``actions_taken`` yields the same lesson minus them."""
        record = _record(
            OutcomeType.SUCCESS,
            ["svc/mod/good"],
            attempts=[AttemptRecord(attempt=1, causal_entry_keys=["svc/mod/stale"])],
        ).model_copy(update={
            "actions_taken": [
                {"index": 0, "action_key": "resolve:a", "success": False, "attempt": 1},
                {"index": 2, "action_key": "resolve:b", "success": True, "attempt": None},
            ],
            "situation": "card declined for an annual plan",
            "task_input": "x" * 500,
        })
        lesson = ev.contrast_lesson(record)
        assert lesson["failed_actions"] == ["resolve:a"]
        assert lesson["resolved_action"] == "resolve:b"
        assert lesson["task_excerpt"] == "card declined for an annual plan"
        bare = ev.contrast_lesson(_record(
            OutcomeType.SUCCESS, ["svc/mod/good"],
            attempts=[AttemptRecord(attempt=1, causal_entry_keys=["svc/mod/stale"])],
        ))
        assert bare["failed_actions"] == [] and bare["resolved_action"] is None
        assert bare["task_excerpt"] is None
        assert bare["avoid"] == lesson["avoid"] and bare["resolved_with"] == lesson["resolved_with"]

    def test_replacements_read_the_shared_shape_only(self) -> None:
        """``replaced_by`` comes from ``{avoid, resolved_with}`` and nothing
        else, so a lesson written by the repair loop (no action fields) and one
        written here resolve alike; non-lessons are skipped."""
        def _lesson(key: str, value) -> MemoryEntry:
            return MemoryEntry(
                entity_path="svc/mod", key=key, value=value, confidence=0.8,
                provenance=Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC)),
            )
        rich = _lesson(ev.contrast_lesson_key("o1"), {
            "avoid": ["svc/mod/stale"], "resolved_with": ["svc/mod/good"],
            "failed_actions": ["resolve:a"], "resolved_action": "resolve:b",
        })
        pointer = _lesson(ev.contrast_lesson_key("fix-9"), {
            "avoid": ["svc/mod/stale", "svc/mod/older"], "resolved_with": ["svc/mod/corrected"],
        })
        plain = _lesson("fix-good", "a rule")
        out = ev.replacements_from_lessons([plain, rich, pointer])
        assert out == {
            "svc/mod/stale": ["svc/mod/good", "svc/mod/corrected"],
            "svc/mod/older": ["svc/mod/corrected"],
        }


class TestLocalRecord:
    """The query-conditioned record ``evidence_near`` returns, read two ways."""

    def _local(self, *recent: bool, n: int | None = None) -> dict:
        s = sum(1.0 for r in recent if r)
        f = sum(1.0 for r in recent if not r)
        return {
            "success": s, "failure": f, "n": n if n is not None else len(recent),
            "best_similarity": 0.95,
            "recent": [{"success": r, "similarity": 0.95, "committed_at": None} for r in recent],
        }

    def test_locally_discredited_needs_two_consecutive_recent_failures(self) -> None:
        assert ev.locally_discredited(self._local(False, False)) is True
        assert ev.locally_discredited(self._local(False, False, True, True, True)) is True
        assert ev.locally_discredited(self._local(False)) is False, "one failure may be the agent's"
        assert ev.locally_discredited(self._local(True, False, False)) is False, "a later success clears it"
        assert ev.locally_discredited(self._local(False, True, False)) is False
        assert ev.locally_discredited(None) is False

    def test_a_record_without_the_order_never_discredits(self) -> None:
        # An adapter predating ``recent``: the sums cannot say which came last.
        assert ev.locally_discredited({"success": 0.0, "failure": 5.0, "n": 5}) is False

    def test_one_nearby_outcome_enters_the_blend_at_a_third(self) -> None:
        pooled = 0.9
        blended, w = ev.blend_local_evidence(pooled, self._local(False))
        assert w == pytest.approx(1 / 3)
        assert blended < pooled and blended > 0.0, "moves the rank, does not flip the sign"
        # Two nearby outcomes weigh what they always did.
        _, w2 = ev.blend_local_evidence(pooled, self._local(False, False))
        assert w2 == pytest.approx(0.5)
        assert ev.blend_local_evidence(pooled, None) == (pooled, 0.0)
        assert ev.blend_local_evidence(pooled, {"n": 0}) == (pooled, 0.0)

    def test_rescue_still_needs_two(self) -> None:
        assert ev.locally_valid(self._local(True)) is False
        assert ev.locally_valid(self._local(True, True)) is True


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


class TestReviewFindings:
    """Two model-side findings from the first review of this branch."""

    def test_multiplicative_success_below_the_gate_keeps_the_flag(self) -> None:
        # Python used to clear ``discredited`` on any success under the legacy
        # model while the SQL step kept it until the posterior cleared 0.5, so
        # filesystem and Postgres disagreed on what retrieval excluded.
        e = _entry(0.30, discredited_at=datetime.now(UTC), success_count=0, failure_count=3)
        upd = ev.apply_outcome_multiplicative(e, OutcomeType.SUCCESS)
        assert upd.confidence < ev.DISCREDIT_THRESHOLD
        assert upd.discredited is True

    def test_multiplicative_success_over_the_gate_clears_the_flag(self) -> None:
        e = _entry(0.49, discredited_at=datetime.now(UTC))
        upd = ev.apply_outcome_multiplicative(e, OutcomeType.SUCCESS)
        assert upd.confidence >= ev.DISCREDIT_THRESHOLD
        assert upd.discredited is False

    def test_an_unknown_outcome_type_is_not_evidence(self) -> None:
        assert not ev.is_known("something_new")
        e = _entry(0.9)
        rec = _record(OutcomeType.SUCCESS, ["svc/mod/rule"])
        rec = rec.model_copy(update={"outcome_type": "something_new"})
        new, updates = ev.apply_record_to_entry(e, rec, model="evidence")
        assert updates == []
        assert new.confidence == pytest.approx(0.9)
        assert (new.success_count, new.failure_count) == (0, 0)
