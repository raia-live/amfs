"""An outcome credits the claim the agent read, not whatever the key says now.

Found with a reflecting agent: it read ``fix`` (v1, "restart the worker"),
tried it, failed, found another route, wrote the better lesson under the same
key (v2, "clear the queue first") and then committed. The failure it had just
earned landed on v2 — the correction was born discredited, and the stale
lesson it replaced kept its record. Now the record carries the version read
per key; a step whose key was rewritten with a different claim in between is
skipped. Restatements and propagation-opened versions still count: the claim
is the same, so the credit belongs to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import OutcomeType
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def mem(tmp_amfs_root: Path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
    return AgentMemory(agent_id="ops-agent", adapter=adapter)


EP = "acme/ops"


def test_the_record_carries_the_version_read(mem: AgentMemory) -> None:
    mem.write(EP, "fix", "restart the worker", confidence=0.8)
    mem._read_tracker.clear()
    read = mem.read(EP, "fix")
    seen: list = []
    real = mem.adapter.commit_outcome
    mem.adapter.commit_outcome = lambda rec: (seen.append(rec), real(rec))[1]  # type: ignore[method-assign]
    mem.commit_outcome("t1", OutcomeType.SUCCESS)

    assert seen[0].causal_entry_versions == {f"{EP}/fix": read.version}


def test_a_rewritten_key_does_not_inherit_the_old_claims_failure(mem: AgentMemory) -> None:
    mem.write(EP, "fix", "restart the worker", confidence=0.8)
    mem._read_tracker.clear()
    mem.read(EP, "fix")  # acted on v1
    corrected = mem.write(EP, "fix", "clear the queue first", confidence=0.8)  # v2
    mem.commit_outcome("t1", OutcomeType.FAILURE)  # about v1

    live = mem.read(EP, "fix")
    assert live.value == "clear the queue first"
    assert live.version == corrected.version
    assert live.failure_count == 0
    assert live.evidence_status == "untested"
    assert live.confidence == pytest.approx(0.8)


def test_a_failed_attempt_stays_with_the_claim_it_tried(mem: AgentMemory) -> None:
    mem.write(EP, "fix", "restart the worker", confidence=0.8)
    mem._read_tracker.clear()
    mem.read(EP, "fix")
    mem.record_attempt(outcome_type=OutcomeType.MINOR_FAILURE, summary="restart did nothing")
    # Reflection rewrites the lesson before the terminal commit.
    mem.write(EP, "fix", "clear the queue first", confidence=0.8)
    mem.read(EP, "fix")  # the retry reads the corrected claim
    mem.commit_outcome("t1", OutcomeType.SUCCESS)

    live = mem.read(EP, "fix")
    assert live.value == "clear the queue first"
    # The attempt's failure was about v1; the success was earned on v2.
    assert (live.success_count, live.failure_count) == (1, 0)
    assert live.evidence_status == "validated"


def test_a_restated_claim_still_takes_the_credit(mem: AgentMemory) -> None:
    mem.write(EP, "fix", "restart the worker", confidence=0.8)
    mem._read_tracker.clear()
    mem.read(EP, "fix")
    mem.write(EP, "fix", "restart the worker", confidence=0.8)  # same words, new version
    mem.commit_outcome("t1", OutcomeType.SUCCESS)

    live = mem.read(EP, "fix")
    assert live.success_count == 1
    assert live.evidence_status == "validated"


def test_versions_opened_by_propagation_do_not_block_later_outcomes(mem: AgentMemory) -> None:
    mem.write(EP, "fix", "restart the worker", confidence=0.8)
    for n in range(3):
        mem._read_tracker.clear()
        mem.read(EP, "fix")
        mem.commit_outcome(f"t{n}", OutcomeType.SUCCESS)
    live = mem.read(EP, "fix")
    assert live.success_count == 3


def test_a_record_without_versions_applies_as_before(mem: AgentMemory) -> None:
    mem.write(EP, "fix", "restart the worker", confidence=0.8)
    mem.write(EP, "fix", "clear the queue first", confidence=0.8)
    mem._read_tracker.clear()
    mem.commit_outcome(
        "t1", OutcomeType.FAILURE, causal_entry_keys=[f"{EP}/fix"], causal_entry_versions={}
    )
    live = mem.read(EP, "fix")
    assert live.failure_count == 1


def test_claim_still_held_helper() -> None:
    from types import SimpleNamespace

    live = SimpleNamespace(entity_path=EP, key="fix", version=3, value="b")
    assert ev.claim_still_held(live, None, lambda *_: None)
    assert ev.claim_still_held(live, 3, lambda *_: None)
    assert ev.claim_still_held(live, 1, None)
    assert ev.claim_still_held(live, 1, lambda *_: None)  # version not found
    same = SimpleNamespace(value="b")
    other = SimpleNamespace(value="a")
    assert ev.claim_still_held(live, 1, lambda *_: same)
    assert not ev.claim_still_held(live, 1, lambda *_: other)
