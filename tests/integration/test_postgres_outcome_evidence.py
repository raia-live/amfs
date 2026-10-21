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
    # Not discredited, and — under the posterior label rule — still worth
    # acting on: 8/9 with a first strike weighed without surprise.
    assert first.discredited_at is None and first.evidence_status == "validated"
    assert first.confidence > 0.6
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
    # The author rewrites the rule: a new claim, tested from scratch. (Restating
    # the same claim would keep the record — see inherit_evidence.)
    e = _entry("rule", 0.7)
    e.value = {"rule": "something else"}
    adapter.write(e)
    fresh = adapter.read("svc/mod", "rule")
    assert fresh.version == 3
    assert fresh.discredited_at is None
    assert fresh.failure_count == 0
    assert fresh.evidence_status == "untested"


def _versions(adapter, key: str) -> dict[str, int]:
    return {f"svc/mod/{key}": adapter.read("svc/mod", key).version}


def test_a_rewritten_claim_is_not_charged_for_the_old_one(adapter, monkeypatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("rule", 0.7))
    read = _versions(adapter, "rule")  # the agent acted on v1
    # Reflection rewrites the lesson before the commit lands.
    e = _entry("rule", 0.7)
    e.value = {"rule": "corrected"}
    adapter.write(e)
    rec = _record(OutcomeType.FAILURE, ["svc/mod/rule"])
    rec.causal_entry_versions = read
    adapter.commit_outcome(rec)

    live = adapter.read("svc/mod", "rule")
    assert live.value == {"rule": "corrected"}
    assert live.failure_count == 0
    assert live.evidence_status == "untested"
    assert live.confidence == pytest.approx(0.7)


def test_a_restated_claim_takes_the_credit_across_versions(adapter, monkeypatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("rule", 0.7))
    read = _versions(adapter, "rule")
    adapter.write(_entry("rule", 0.7))  # same claim, new version
    rec = _record(OutcomeType.SUCCESS, ["svc/mod/rule"])
    rec.causal_entry_versions = read
    adapter.commit_outcome(rec)
    live = adapter.read("svc/mod", "rule")
    assert live.success_count == 1
    assert live.evidence_status == "validated"


def test_an_attempt_charges_the_claim_it_tried_and_the_success_goes_to_the_fix(
    adapter, monkeypatch
) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("rule", 0.7))
    tried = _versions(adapter, "rule")
    e = _entry("rule", 0.7)
    e.value = {"rule": "corrected"}
    adapter.write(e)
    retried = _versions(adapter, "rule")
    rec = _record(
        OutcomeType.SUCCESS,
        ["svc/mod/rule"],
        attempts=[
            AttemptRecord(
                attempt=1,
                outcome_type=OutcomeType.MINOR_FAILURE,
                causal_entry_keys=["svc/mod/rule"],
                causal_entry_versions=tried,
            )
        ],
    )
    rec.causal_entry_versions = retried
    adapter.commit_outcome(rec)
    live = adapter.read("svc/mod", "rule")
    assert live.value == {"rule": "corrected"}
    assert (live.success_count, live.failure_count) == (1, 0)


# ── Verification and credit by uplift (012) ─────────────────────────────


def _verified(record: OutcomeRecord, by: str) -> OutcomeRecord:
    record.session_metadata = {"attributes": {"verified_by": by}}
    return record


def test_a_verified_outcome_weighs_more_than_the_agents_own_word(adapter, monkeypatch) -> None:
    """Two identical entries, two identical failures — one declared by the
    agent, one decided by CI. The verified one moves its entry further, by
    VERIFIED_EVIDENCE_WEIGHT, and the SQL agrees with the Python model on both."""
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    told = adapter.write(_entry("told", 0.8))
    proven = adapter.write(_entry("proven", 0.8))
    r_told = _record(OutcomeType.FAILURE, ["svc/mod/told"])
    r_proven = _verified(_record(OutcomeType.FAILURE, ["svc/mod/proven"]), "ci")
    adapter.commit_outcome(r_told)
    adapter.commit_outcome(r_proven)
    got_told = adapter.read("svc/mod", "told")
    got_proven = adapter.read("svc/mod", "proven")
    assert got_proven.evidence_failure == pytest.approx(
        got_told.evidence_failure * ev.VERIFIED_EVIDENCE_WEIGHT, abs=1e-4
    )
    assert got_proven.confidence < got_told.confidence
    assert got_told.evidence_failure == pytest.approx(
        _expect(told, r_told).evidence_failure, abs=1e-4
    )
    assert got_proven.evidence_failure == pytest.approx(
        _expect(proven, r_proven).evidence_failure, abs=1e-4
    )
    with adapter._pool.connection() as conn:
        rows = conn.execute(
            "SELECT outcome_ref, verified_by FROM amfs_outcomes WHERE namespace = %s",
            (adapter._namespace,),
        ).fetchall()
    assert {r["outcome_ref"]: r["verified_by"] for r in rows} == {
        r_told.outcome_ref: None,
        r_proven.outcome_ref: "ci",
    }


def test_credit_follows_uplift(adapter, monkeypatch) -> None:
    """Sessions that read ``good`` succeed; sessions that read ``bad`` fail.
    Once the entity's record says so, a success citing both hands ``good`` its
    full share and ``bad`` less, and a failure citing both charges ``bad`` in
    full and ``good`` less. Uniform credit would move the two identically."""
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("good", 0.6))
    adapter.write(_entry("bad", 0.6))
    history = []
    for _ in range(6):
        history.append(_record(OutcomeType.SUCCESS, ["svc/mod/good"]))
        history.append(_record(OutcomeType.FAILURE, ["svc/mod/bad"]))
    for rec in history:
        adapter.commit_outcome(rec)
    good = adapter.read("svc/mod", "good")
    bad = adapter.read("svc/mod", "bad")
    assert good.uplift is not None and good.uplift > 0
    assert bad.uplift is not None and bad.uplift < 0
    # Each row carries the uplift as of the last outcome that cited it: ``good``
    # was last cited by the eleventh outcome (six reads, five unread so far),
    # ``bad`` by the twelfth. The stored value is the reference definition
    # over the outcomes committed by then.
    assert (good.uplift_sessions, bad.uplift_sessions) == (5, 6)
    at_good = ev.uplift_counts(history[:11], "svc/mod", ["good"])["good"]
    at_bad = ev.uplift_counts(history, "svc/mod", ["bad"])["bad"]
    assert (good.uplift, good.uplift_sessions) == (pytest.approx(at_good[0], abs=1e-9), at_good[1])
    assert (bad.uplift, bad.uplift_sessions) == (pytest.approx(at_bad[0], abs=1e-9), at_bad[1])

    both = _record(OutcomeType.SUCCESS, ["svc/mod/good", "svc/mod/bad"])
    adapter.commit_outcome(both)
    good2 = adapter.read("svc/mod", "good")
    bad2 = adapter.read("svc/mod", "bad")
    gained_good = good2.evidence_success - good.evidence_success * ev.EVIDENCE_DECAY
    gained_bad = bad2.evidence_success - bad.evidence_success * ev.EVIDENCE_DECAY
    # Per unit of surprise (``bad`` sits lower, so a success on it is the more
    # surprising), ``bad`` received less of the split than ``good``...
    surprise_good = 1.0 + (1.0 - good.confidence)
    surprise_bad = 1.0 + (1.0 - bad.confidence)
    assert gained_bad / surprise_bad < gained_good / surprise_good
    # ...by exactly the credit the model computes from the uplift it carried:
    # severity 1 * causal 1 * credit / 2 keys * surprise.
    expect_good = ev.uplift_credit(good.uplift, good.uplift_sessions, success=True)
    expect_bad = ev.uplift_credit(bad.uplift, bad.uplift_sessions, success=True)
    assert expect_good == 1.0 and ev.UPLIFT_FLOOR <= expect_bad < 1.0
    assert gained_good == pytest.approx(expect_good / 2 * surprise_good, abs=1e-3)
    assert gained_bad == pytest.approx(expect_bad / 2 * surprise_bad, abs=1e-3)
    # And the SQL agrees with the Python model step for step, each row's own
    # confidence and uplift going into its weight.
    want_good = _expect(good, both)
    want_bad = _expect(bad, both)
    assert good2.evidence_success == pytest.approx(want_good.evidence_success, abs=1e-4)
    assert bad2.evidence_success == pytest.approx(want_bad.evidence_success, abs=1e-4)


def test_uplift_is_refreshed_only_for_the_keys_an_outcome_cites(adapter, monkeypatch) -> None:
    """Credit is read at citation time, so the refresh is per cited key. An
    entry on the entity no outcome has cited yet keeps the uniform split."""
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("cited", 0.6))
    adapter.write(_entry("bystander", 0.6))
    adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["svc/mod/cited"]))
    adapter.commit_outcome(_record(OutcomeType.FAILURE, ["svc/mod/other"]))
    adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["svc/mod/cited"]))
    cited = adapter.read("svc/mod", "cited")
    bystander = adapter.read("svc/mod", "bystander")
    assert cited.uplift == pytest.approx(1.0) and cited.uplift_sessions == 1
    assert bystander.uplift is None and bystander.uplift_sessions == 0
    assert bystander.version == 1


# ── The learning series (GET /api/v1/learning reads this) ───────────────


def test_learning_series_returns_the_sessions_in_order_and_the_shifts_among_cited_entries(
    adapter, monkeypatch
) -> None:
    from datetime import timedelta

    from amfs_core.learning import learning_curve

    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("rule", 0.9))
    adapter.write(_entry("other", 0.9))
    shifted_at = None
    for outcome in [OutcomeType.SUCCESS] * 4 + [OutcomeType.FAILURE] * 2 + [OutcomeType.SUCCESS] * 3:
        rec = _record(outcome, ["svc/mod/rule"])
        if outcome is OutcomeType.FAILURE:
            rec.session_metadata = {"attributes": {"verified_by": "ci"}}
        adapter.commit_outcome(rec)
        if outcome is OutcomeType.FAILURE:
            # The second CI-graded failure discredits the rule: that moment is
            # the shift, and it stays in the version history after the three
            # successes that follow lift the flag again.
            shifted_at = adapter.read("svc/mod", "rule").discredited_at or shifted_at
    assert shifted_at is not None
    # Another agent's outcome, and one on another entity, for the scoping.
    rec = _record(OutcomeType.FAILURE, ["svc/mod/other"])
    rec.agent_id = "b"
    adapter.commit_outcome(rec)
    rec = _record(OutcomeType.SUCCESS, ["elsewhere/thing/k"])
    adapter.commit_outcome(rec)
    assert adapter.read("svc/mod", "rule").discredited_at is None

    since = datetime.now(UTC) - timedelta(days=1)
    series = adapter.learning_series(since=since, agent="a")
    assert [s["outcome_type"] for s in series["sessions"]] == (
        ["success"] * 4 + ["failure"] * 2 + ["success"] * 3 + ["success"]
    )
    assert [s["verified_by"] for s in series["sessions"]].count("ci") == 2
    ats = [s["at"] for s in series["sessions"]]
    assert ats == sorted(ats)
    assert [(s["entity_path"], s["key"]) for s in series["shifts"]] == [("svc/mod", "rule")]
    assert series["shifts"][0]["at"] == shifted_at

    # The curve reads straight off it.
    curve = learning_curve(series["sessions"], shifts=series["shifts"], window=4)
    assert curve["sessions"] == 10
    assert curve["recovery"]["shifts"] == 1
    assert curve["verified_share"] == 0.2

    # Scoping: the entity narrows; the visibility scope bounds; agent narrows on top.
    on_entity = adapter.learning_series(since=since, entity_path="svc/mod")
    assert len(on_entity["sessions"]) == 10  # a's nine on the rule plus b's one on other
    assert len(adapter.learning_series(since=since, entity_path="elsewhere/thing")["sessions"]) == 1
    empty = {"sessions": [], "shifts": [], "truncated": False}
    assert adapter.learning_series(since=since, visible_agents=set()) == empty
    assert adapter.learning_series(since=since, agent="a", visible_agents={"b"}) == empty
    only_b = adapter.learning_series(since=since, visible_agents={"b"})
    assert [s["agent_id"] for s in only_b["sessions"]] == ["b"]
    # b's one failure discredited ``other``; b never cited the rule.
    assert [(s["entity_path"], s["key"]) for s in only_b["shifts"]] == [("svc/mod", "other")]
    assert series["truncated"] is False


def test_learning_series_keeps_the_newest_sessions_under_the_cap(adapter, monkeypatch) -> None:
    """A busy account's curve ends at its current sessions. The cap keeps the
    newest rows, still returned oldest first, and the shifts are bounded by
    the oldest session kept so none is reported without its sessions."""
    from datetime import timedelta

    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("early", 0.9))
    adapter.write(_entry("late", 0.9))
    # Two failures on ``early`` discredit it, then six successes on ``late``.
    for _ in range(2):
        adapter.commit_outcome(_record(OutcomeType.FAILURE, ["svc/mod/early"]))
    early_shift = adapter.read("svc/mod", "early").discredited_at
    assert early_shift is not None
    for _ in range(6):
        adapter.commit_outcome(_record(OutcomeType.SUCCESS, ["svc/mod/late"]))

    since = datetime.now(UTC) - timedelta(days=1)
    whole = adapter.learning_series(since=since)
    assert [s["outcome_type"] for s in whole["sessions"]] == ["failure"] * 2 + ["success"] * 6
    assert whole["truncated"] is False
    assert [s["key"] for s in whole["shifts"]] == ["early"]

    capped = adapter.learning_series(since=since, limit=4)
    assert [s["outcome_type"] for s in capped["sessions"]] == ["success"] * 4
    ats = [s["at"] for s in capped["sessions"]]
    assert ats == sorted(ats)
    assert capped["truncated"] is True
    # The discredit of ``early`` happened before the oldest session kept.
    assert capped["shifts"] == []
    # Exactly the cap is a complete series: nothing dropped, the shift kept.
    exact = adapter.learning_series(since=since, limit=8)
    assert len(exact["sessions"]) == 8
    assert exact["truncated"] is False
    assert [s["key"] for s in exact["shifts"]] == ["early"]


def test_learning_series_counts_entries_cited_only_in_failed_attempts(adapter, monkeypatch) -> None:
    """An entry the agent relied on in an attempt that failed, and dropped
    before the terminal step, is still one it relied on: its discredit is a
    shift in this agent's series, as it is a read in the uplift refresh."""
    from datetime import timedelta

    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("stale", 0.9))
    adapter.write(_entry("fresh", 0.9))
    for _ in range(2):
        adapter.commit_outcome(
            _record(
                OutcomeType.SUCCESS,
                ["svc/mod/fresh"],
                attempts=[
                    AttemptRecord(
                        attempt=1,
                        outcome_type=OutcomeType.FAILURE,
                        causal_entry_keys=["svc/mod/stale"],
                        summary="followed stale; did not work",
                    )
                ],
            )
        )
    stale = adapter.read("svc/mod", "stale")
    assert stale.discredited_at is not None
    since = datetime.now(UTC) - timedelta(days=1)
    series = adapter.learning_series(since=since, agent="a")
    assert [s["key"] for s in series["shifts"]] == ["stale"]
    assert series["shifts"][0]["at"] == stale.discredited_at


# ── 013: verified_by onto the outcomes that predate the column ──────────


def test_verified_by_backfill_copies_from_the_trace_once(adapter, monkeypatch) -> None:
    """An outcome committed before 012 has NULL ``verified_by`` while its
    sealed trace carries the attribute. The backfill copies it across for
    those rows only, leaves a never-verified outcome NULL, records the pass,
    and does nothing on the next application."""
    from amfs_postgres.adapter import _VERIFIED_BY_BACKFILL_SQL

    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    adapter.write(_entry("rule", 0.9))
    graded = _record(OutcomeType.SUCCESS, ["svc/mod/rule"])
    graded.session_metadata = {"attributes": {"verified_by": "driver-monitor"}}
    adapter.commit_outcome(graded)
    own_word = _record(OutcomeType.SUCCESS, ["svc/mod/rule"])
    adapter.commit_outcome(own_word)

    with adapter._pool.connection() as conn, conn.cursor() as cur:
        # The trace the SDK seals beside the outcome, with the attribute.
        cur.execute(
            """INSERT INTO amfs_decision_traces
                   (namespace, agent_id, session_id, outcome_ref, outcome_type, session_metadata)
               VALUES (%s, 'a', 'sess-1', %s, 'success', %s::jsonb)""",
            (adapter._namespace, graded.outcome_ref, '{"attributes": {"verified_by": "driver-monitor"}}'),
        )
        # Pre-012 state: the column exists but nothing was written to it.
        cur.execute(
            "UPDATE amfs_outcomes SET verified_by = NULL WHERE namespace = %s", (adapter._namespace,)
        )
        cur.execute("DELETE FROM amfs_backfills WHERE name = '013_outcome_verified_by'")

        cur.execute(_VERIFIED_BY_BACKFILL_SQL)
        cur.execute(
            "SELECT outcome_ref, verified_by FROM amfs_outcomes WHERE namespace = %s ORDER BY id",
            (adapter._namespace,),
        )
        rows = {r["outcome_ref"]: r["verified_by"] for r in cur.fetchall()}
        assert rows == {graded.outcome_ref: "driver-monitor", own_word.outcome_ref: None}
        cur.execute("SELECT rows_written FROM amfs_backfills WHERE name = '013_outcome_verified_by'")
        row = cur.fetchone()
        assert row is not None and row["rows_written"] >= 1

        # Marked done: a second application changes nothing, even after the
        # column is cleared again — the pass is one-time by design.
        cur.execute(
            "UPDATE amfs_outcomes SET verified_by = NULL WHERE namespace = %s", (adapter._namespace,)
        )
        cur.execute(_VERIFIED_BY_BACKFILL_SQL)
        cur.execute(
            "SELECT count(*) AS n FROM amfs_outcomes WHERE namespace = %s AND verified_by IS NOT NULL",
            (adapter._namespace,),
        )
        assert cur.fetchone()["n"] == 0
