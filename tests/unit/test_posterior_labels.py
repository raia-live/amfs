"""Posterior evidence: labels judge the record, one slip does not discredit a
proven rule, and validators name who stands behind a claim.

Pins the numbers the SQL step (migration 009) has to reproduce; the Postgres
integration test compares against the same Python model.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core import labels
from amfs_core.models import MemoryEntry, OutcomeType, Provenance
from amfs_cortex.briefing import BriefingService
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    monkeypatch.delenv(labels.LABELS_ENV, raising=False)


def _entry(conf: float = 0.7) -> MemoryEntry:
    return MemoryEntry(
        entity_path="acme/support", key="rule", value="do the thing",
        provenance=Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC)),
        confidence=conf,
    )


def _run(seq: list[str], conf: float = 0.7) -> list[MemoryEntry]:
    e = _entry(conf)
    out = []
    for o in seq:
        upd = ev.apply_outcome(e, o)
        e = e.model_copy(update={**upd.as_entry_update(datetime.now(UTC)),
                                 "outcome_count": e.outcome_count + 1})
        out.append(e)
    return out


# ── first strike ──────────────────────────────────────────────────────────


def test_one_failure_on_a_proven_rule_leaves_it_contested_not_discredited() -> None:
    hist = _run(["success"] * 4 + ["failure"])
    last = hist[-1]
    assert last.discredited_at is None
    assert 0.55 <= last.confidence <= 0.7, last.confidence
    assert last.evidence_status == "validated"  # 4/5 with p above the bar


def test_the_second_failure_still_discredits_it() -> None:
    hist = _run(["success"] * 5 + ["failure"] * 2)
    assert hist[-2].discredited_at is None
    assert hist[-1].discredited_at is not None
    assert hist[-1].confidence < ev.DISCREDIT_THRESHOLD


def test_a_fresh_claim_is_still_discredited_by_its_first_failure() -> None:
    hist = _run(["failure"])
    assert hist[-1].discredited_at is not None
    assert hist[-1].confidence < 0.3


def test_first_strike_needs_a_clean_record_of_enough_wins() -> None:
    assert ev.first_strike("failure", success_count=3, failure_count=0)
    assert not ev.first_strike("failure", success_count=2, failure_count=0)
    assert not ev.first_strike("failure", success_count=5, failure_count=1)
    assert not ev.first_strike("success", success_count=5, failure_count=0)
    with_cap = ev.evidence_weight("failure", current_confidence=0.9, success_count=4)
    without = ev.evidence_weight("failure", current_confidence=0.9)
    assert with_cap == pytest.approx(ev.severity("failure"))
    assert without > with_cap


def test_pinned_numbers_for_the_sql_parity_test() -> None:
    """The trajectory migration 009 has to reproduce, to four decimals."""
    hist = _run(["success"] * 4 + ["failure", "failure"])
    confs = [round(e.confidence, 4) for e in hist]
    assert confs == [0.8182, 0.8579, 0.878, 0.8901, 0.6157, 0.3995]


# ── labels ────────────────────────────────────────────────────────────────


def test_label_judges_the_record_not_the_last_event() -> None:
    hist = _run(["success"] * 4 + ["failure"])
    assert hist[-1].evidence_status == "validated"
    p, n = hist[-1].posterior
    assert n == 5 and p >= labels.VALIDATED_MIN_P - 0.15  # decayed masses: near but not at the bar


def test_thin_mixed_record_is_contested() -> None:
    hist = _run(["success", "failure"])
    assert hist[-1].evidence_status in ("contested", "discredited")


def test_strict_rule_is_available(monkeypatch) -> None:
    monkeypatch.setenv(labels.LABELS_ENV, "strict")
    hist = _run(["success"] * 6 + ["minor_failure"])
    assert hist[-1].discredited_at is None
    assert hist[-1].evidence_status == "contested"


def test_posterior_is_neutral_for_an_untested_entry() -> None:
    assert _entry(0.95).posterior == (0.5, 0)
    assert _entry(0.95).evidence_status == "untested"


# ── validators ────────────────────────────────────────────────────────────


def test_validators_after() -> None:
    assert ev.validators_after([], "a", True) == ["a"]
    assert ev.validators_after(["a"], "b", True) == ["a", "b"]
    assert ev.validators_after(["a", "b"], "a", True) == ["b", "a"]  # moved to the end
    assert ev.validators_after(["a"], "b", False) == ["a"]  # failures do not add
    assert ev.validators_after([str(i) for i in range(12)], "x", True) == [str(i) for i in range(3, 12)] + ["x"]


@pytest.fixture
def world(tmp_path):
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mems = {a: AgentMemory(agent_id=a, adapter=adapter) for a in ("a1", "a2", "a3")}
    mems["a1"].write("acme/support", "rule", "restart the ingest worker", confidence=0.8)
    for m in mems.values():
        m._read_tracker.clear()
    return adapter, mems


def test_validators_accumulate_across_agents_and_survive_a_restatement(world) -> None:
    adapter, mems = world
    for i, a in enumerate(("a1", "a2", "a2", "a3")):
        mems[a].read("acme/support", "rule")
        mems[a].commit_outcome(f"t{i}", OutcomeType.SUCCESS)
    e = adapter.read("acme/support", "rule")
    assert e.validators == ["a1", "a2", "a3"]
    # A failure by a4 does not remove anyone.
    m4 = AgentMemory(agent_id="a4", adapter=adapter)
    m4.read("acme/support", "rule")
    m4.commit_outcome("t9", OutcomeType.MINOR_FAILURE)
    e = adapter.read("acme/support", "rule")
    assert e.validators == ["a1", "a2", "a3"]
    # Restating the same claim keeps the validators with the record.
    mems["a1"].write("acme/support", "rule", "restart the ingest worker", confidence=0.8)
    e = adapter.read("acme/support", "rule")
    assert e.validators == ["a1", "a2", "a3"] and e.success_count == 4


def test_briefing_rows_carry_posterior_and_validators(world) -> None:
    adapter, mems = world
    for i, a in enumerate(("a1", "a2")):
        mems[a].read("acme/support", "rule")
        mems[a].commit_outcome(f"t{i}", OutcomeType.SUCCESS)
    adapter.list_digests = lambda **kw: []  # type: ignore[attr-defined]
    adapter.list_branches = lambda **kw: []  # type: ignore[attr-defined]
    svc = BriefingService(adapter=adapter, namespace="test")
    lead = svc.briefing(entity_path="acme/support")[0]
    row = lead.summary["hot_context"][0]
    assert row["validators"] == 2
    assert row["posterior"]["n"] == 2 and row["posterior"]["p"] > 0.5


def test_briefing_tried_here_and_since(world) -> None:
    adapter, mems = world
    now = datetime.now(UTC)
    rows = [
        {"actions_taken": [{"action_key": "resolve:a", "success": False}], "committed_at": now, "agent_id": "a1"},
        {"actions_taken": [{"action_key": "resolve:a", "success": False}], "committed_at": now, "agent_id": "a2"},
        {"actions_taken": [{"action_key": "resolve:b", "success": True}], "committed_at": now, "agent_id": "a1"},
    ]
    adapter.action_stats = lambda entity_path, **kw: list(rows)  # type: ignore[attr-defined]
    adapter.list_digests = lambda **kw: []  # type: ignore[attr-defined]
    adapter.list_branches = lambda **kw: []  # type: ignore[attr-defined]
    svc = BriefingService(adapter=adapter, namespace="test")
    lead = svc.briefing(entity_path="acme/support", compact=True)[0]
    tried = {t["action"]: t for t in lead.summary["tried_here"]}
    assert tried["resolve:a"]["n"] == 2 and tried["resolve:a"]["agents"] == 2
    assert lead.summary["explore"]["avoid"] == ["resolve:a"]
    assert lead.summary["explore"]["thin_evidence"] == ["resolve:b"]
    # since: nothing changed after "now + 1h" -> the list sections are empty.
    from datetime import timedelta

    later = svc.briefing(entity_path="acme/support", since=now + timedelta(hours=1))[0]
    assert later.summary["hot_context"] == [] and later.summary["tried_here"] == []
    assert later.summary["since"]
