"""A replay's outcome is the repair loop's verdict on a fix, not evidence.

A Tier 1 replay (``attributes.origin = repair``) or a Tier 2 customer replay
(``attributes.replay_delivery_id``) re-runs a failed case with memory on the
fix's branch. The hosted loop already grades such a run by the tier that asked
for it and by nothing else — it does not vote in a canary or count as exposure.
Its outcome was still applied to the entries it read, and the fix under test
first among them: on clbench (2026-09-30, fleet 10) two Tier 2 replays that
*passed* each carried two failed attempts, each attempt handed the correction a
failure, and 65 seconds after it was written the correction was discredited —
before its canary's first session. The same runs wrote 425
``lesson-contrast-replay-*`` entries onto ``main`` naming corrections under
test as the thing to avoid. These tests pin the rule end to end over the
filesystem adapter: SDK attribute bag → ``commit_outcome`` → propagator →
no evidence, no lesson, trace still built.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import OutcomeType
from amfs_core.outcome import OutcomeBackPropagator
from amfs_filesystem.adapter import FilesystemAdapter

SCOPE = "acme/support"


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def mem(tmp_amfs_root: Path) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
    m = AgentMemory(agent_id="support-agent", adapter=adapter)
    m.write(SCOPE, "fix-restart", "restart the worker", confidence=0.9)
    m.write(SCOPE, "correction", "rotate the API key instead", confidence=0.45)
    m._read_tracker.clear()
    return m


def _fail_then_succeed(mem: AgentMemory, ref: str) -> list:
    """The shape that discredited the corrections: the correction is read,
    the first attempt fails, the session ends in success."""
    mem.read(SCOPE, "correction")
    mem.record_action("rotate_key", {"service": "ingest"}, result="still stuck", success=False)
    mem.record_attempt(summary="first try did nothing")
    mem.read(SCOPE, "fix-restart")
    mem.record_action("restart_worker", {"service": "ingest"}, result="queue drained")
    return mem.commit_outcome(ref, OutcomeType.SUCCESS, task_input="ingest queue stuck")


class TestPredicate:
    def _record(self, attrs):
        meta = {"attributes": attrs} if attrs is not None else None
        return OutcomeBackPropagator.make_record(
            "r", OutcomeType.SUCCESS, [f"{SCOPE}/correction"], "a", session_metadata=meta,
        )

    def test_tier2_delivery_and_tier1_origin_are_replays(self) -> None:
        assert ev.is_replay_outcome(self._record({"replay_delivery_id": "d-1"}))
        assert ev.is_replay_outcome(self._record({"origin": "repair"}))

    def test_served_sessions_are_not(self) -> None:
        assert not ev.is_replay_outcome(self._record(None))
        assert not ev.is_replay_outcome(self._record({}))
        assert not ev.is_replay_outcome(self._record({"customer": "acme"}))
        # The labelled copy of a served inference is the served decision with
        # the customer's outcome on it, not a test the loop ran.
        assert not ev.is_replay_outcome(self._record({"origin": "served"}))

    def test_attributes_that_are_not_a_mapping_are_not_a_replay(self) -> None:
        rec = self._record(None).model_copy(update={"session_metadata": {"attributes": "x"}})
        assert not ev.is_replay_outcome(rec)


class TestSdk:
    def test_a_served_session_still_learns(self, mem: AgentMemory) -> None:
        updated = {e.key: e for e in _fail_then_succeed(mem, "ticket-1")}
        assert updated["correction"].failure_count == 1
        assert updated["correction"].evidence_status == "discredited"
        assert mem.read(SCOPE, ev.contrast_lesson_key("ticket-1")) is not None

    def test_a_replay_leaves_no_evidence_and_no_lesson(self, mem: AgentMemory) -> None:
        mem.set_session_attributes({"replay_delivery_id": "d-1", "fix_id": "fix-1"})
        updated = _fail_then_succeed(mem, "replay:fix-1:case-1")
        assert updated == []
        correction = mem.read(SCOPE, "correction")
        assert correction.failure_count == 0 and correction.success_count == 0
        assert correction.confidence == 0.45 and correction.discredited_at is None
        assert correction.version == 1  # no evidence version was written
        restart = mem.read(SCOPE, "fix-restart")
        assert restart.success_count == 0 and restart.confidence == 0.9
        assert restart.version == 1
        assert mem.read(SCOPE, ev.contrast_lesson_key("replay:fix-1:case-1")) is None

    def test_the_replay_trace_is_still_built(self, mem: AgentMemory) -> None:
        mem.set_session_attributes({"replay_delivery_id": "d-1"})
        _fail_then_succeed(mem, "replay:fix-1:case-1")
        trace = mem._last_trace
        assert trace is not None and trace.outcome_ref == "replay:fix-1:case-1"
        meta = trace.session_metadata.model_dump()
        assert meta["attributes"]["replay_delivery_id"] == "d-1"
        assert len(meta["attempts"]) == 1
        assert [c.key for c in trace.causal_entries] == ["fix-restart"]

    def test_a_tier1_replay_is_the_same(self, mem: AgentMemory) -> None:
        mem.set_session_attributes({"origin": "repair"})
        assert _fail_then_succeed(mem, "replay:fix-1:case-2") == []
        assert mem.read(SCOPE, "correction").failure_count == 0

    def test_the_bag_is_per_session(self, mem: AgentMemory) -> None:
        mem.set_session_attributes({"replay_delivery_id": "d-1"})
        _fail_then_succeed(mem, "replay:fix-1:case-1")
        mem._read_tracker.clear()
        updated = {e.key: e for e in _fail_then_succeed(mem, "ticket-2")}
        assert updated["correction"].failure_count == 1


class TestPropagatorForwardsToARemoteLearner:
    def test_the_http_adapter_still_receives_the_record(self) -> None:
        """The client's adapter learns remotely: the server applies this rule
        on its own handle and seals the client's trace from the call, so the
        record must reach it."""
        sent: list = []

        class Remote:
            remote_learning = True

            def commit_outcome(self, record):
                sent.append(record)
                return []

        prop = OutcomeBackPropagator(Remote())  # type: ignore[arg-type]
        rec = prop.make_record(
            "replay:f:c", OutcomeType.SUCCESS, [f"{SCOPE}/correction"], "a",
            session_metadata={"attributes": {"replay_delivery_id": "d-1"}},
        )
        assert prop.propagate(rec) == []
        assert sent == [rec]

    def test_a_local_learner_does_not(self) -> None:
        sent: list = []

        class Local:
            def commit_outcome(self, record):
                sent.append(record)
                return []

        prop = OutcomeBackPropagator(Local())  # type: ignore[arg-type]
        rec = prop.make_record(
            "replay:f:c", OutcomeType.SUCCESS, [f"{SCOPE}/correction"], "a",
            session_metadata={"attributes": {"replay_delivery_id": "d-1"}},
        )
        assert prop.propagate(rec) == []
        assert sent == []
