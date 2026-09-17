"""A restated claim keeps its outcome record; a changed claim starts untested.

Found while closing the loop end to end with a reflecting agent: its end-of-task
note rewrote the same key with the same text every episode, and every rewrite
opened a fresh version with an empty record. A lesson validated many times
looked untested the moment its author repeated it; a lesson discredited an
hour ago came back clean. The record belongs to the claim, not the row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import MemoryEntry, OutcomeType, Provenance
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def mem(tmp_amfs_root: Path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
    return AgentMemory(agent_id="support-agent", adapter=adapter)


def _validate(mem: AgentMemory, key: str, times: int) -> None:
    for n in range(times):
        mem._read_tracker.clear()
        mem.read("acme/support", key)
        mem.commit_outcome(f"ok-{key}-{n}", OutcomeType.SUCCESS)


def _discredit(mem: AgentMemory, key: str) -> None:
    mem._read_tracker.clear()
    mem.read("acme/support", key)
    mem.commit_outcome(f"bad-{key}", OutcomeType.FAILURE)


class TestUnchangedClaim:
    def test_a_validated_lesson_restated_stays_validated(self, mem: AgentMemory) -> None:
        mem.write("acme/support", "fix", "rotate the key", confidence=0.8)
        _validate(mem, "fix", 3)
        before = mem.read("acme/support", "fix")
        assert before.evidence_status == "validated" and before.success_count == 3

        again = mem.write("acme/support", "fix", "rotate the key", confidence=0.8)

        assert again.version == before.version + 1
        assert again.success_count == 3
        assert again.evidence_status == "validated"
        assert again.confidence == pytest.approx(before.confidence)
        assert again.last_outcome == before.last_outcome

    def test_a_discredited_lesson_restated_stays_discredited(self, mem: AgentMemory) -> None:
        mem.write("acme/support", "fix", "restart the worker", confidence=0.8)
        _discredit(mem, "fix")
        before = mem.read("acme/support", "fix")
        assert before.evidence_status == "discredited"

        again = mem.write("acme/support", "fix", "restart the worker", confidence=0.9)

        assert again.evidence_status == "discredited"
        assert again.failure_count == before.failure_count
        assert again.discredited_at == before.discredited_at
        # The writer's restated prior does not outrank the posterior.
        assert again.confidence == pytest.approx(before.confidence)

    def test_json_round_trip_counts_as_the_same_claim(self, mem: AgentMemory) -> None:
        mem.write("acme/support", "fix", {"b": 1, "a": [1, 2]}, confidence=0.8)
        _validate(mem, "fix", 2)
        again = mem.write("acme/support", "fix", {"a": [1, 2], "b": 1}, confidence=0.8)
        assert again.success_count == 2

    def test_an_untested_entry_restated_takes_the_new_prior(self, mem: AgentMemory) -> None:
        mem.write("acme/support", "fix", "rotate the key", confidence=0.6)
        again = mem.write("acme/support", "fix", "rotate the key", confidence=0.9)
        assert again.confidence == 0.9
        assert again.evidence_status == "untested"


class TestChangedClaim:
    def test_a_new_claim_under_the_same_key_starts_untested(self, mem: AgentMemory) -> None:
        mem.write("acme/support", "fix", "restart the worker", confidence=0.8)
        _discredit(mem, "fix")

        changed = mem.write("acme/support", "fix", "rotate the key instead", confidence=0.8)

        assert changed.evidence_status == "untested"
        assert changed.failure_count == 0
        assert changed.discredited_at is None
        assert changed.confidence == 0.8

    def test_history_shows_the_change_of_mind(self, mem: AgentMemory) -> None:
        mem.write("acme/support", "fix", "restart the worker", confidence=0.8)
        _discredit(mem, "fix")
        mem.write("acme/support", "fix", "rotate the key instead", confidence=0.8)
        statuses = [v.evidence_status for v in mem.history("acme/support", "fix")]
        assert statuses[-1] == "untested"
        assert "discredited" in statuses


def test_the_helper_alone() -> None:
    prov = Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC))
    current = MemoryEntry(
        entity_path="e", key="k", value="same", provenance=prov, confidence=0.93,
        success_count=4, failure_count=1, evidence_success=3.5, evidence_failure=0.8,
        last_outcome=OutcomeType.SUCCESS,
    )
    fresh = MemoryEntry(entity_path="e", key="k", value="same", provenance=prov, confidence=0.7,
                        version=2)
    kept = ev.inherit_evidence(fresh, current)
    assert kept.version == 2
    assert (kept.success_count, kept.failure_count) == (4, 1)
    assert kept.evidence_success == 3.5 and kept.confidence == 0.93

    other = ev.inherit_evidence(fresh.model_copy(update={"value": "different"}), current)
    assert other.success_count == 0 and other.confidence == 0.7
    assert ev.inherit_evidence(fresh, None) is fresh
