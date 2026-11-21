"""Which entry a failure is charged to: named, in view, or not followed.

An outcome used to charge every entry in the session's read window alike, so
an entry that merely came back from a search was discredited by a failure it
had no part in. Three roles replace that:

* **named** — the agent put the entry in ``causal_entry_keys``. Charged in
  full, exactly as before.
* **in view** — the entry was in the read window and the agent named nothing.
  The first failure lowers it to the discredit threshold and stops there; any
  later failure is weighed in full.
* **not followed** — the agent says the step failed because it did *not* do
  what the entry says. Credited, never charged.

And a failure never lands on text written after the attempt was closed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import (
    AttemptRecord,
    MemoryEntry,
    OutcomeRecord,
    OutcomeType,
    Provenance,
)
from amfs_filesystem.adapter import FilesystemAdapter

SCOPE = "acme/support"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def adapter(tmp_amfs_root: Path) -> FilesystemAdapter:
    return FilesystemAdapter(root=tmp_amfs_root, namespace="test")


@pytest.fixture
def mem(adapter: FilesystemAdapter) -> AgentMemory:
    m = AgentMemory(agent_id="support-agent", adapter=adapter)
    m.write(SCOPE, "rule", "rotate the API key", confidence=1.0)
    m.write(SCOPE, "other", "restart the worker", confidence=0.9)
    m._read_tracker.clear()
    return m


def _entry(key: str, confidence: float) -> MemoryEntry:
    return MemoryEntry(
        entity_path=SCOPE,
        key=key,
        value="v",
        confidence=confidence,
        provenance=Provenance(agent_id="seed", session_id="s", written_at=datetime.now(UTC)),
    )


def _fresh(mem: AgentMemory, key: str = "rule") -> MemoryEntry:
    mem._read_tracker.clear()
    entry = mem._adapter.read(SCOPE, key)
    assert entry is not None
    return entry


# ── In view ──────────────────────────────────────────────────────────────


class TestInView:
    def test_the_first_failure_stops_at_the_threshold(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        mem.commit_outcome("t-1", OutcomeType.FAILURE)
        entry = _fresh(mem)
        assert entry.confidence == pytest.approx(ev.DISCREDIT_THRESHOLD)
        assert entry.failure_count == 1
        assert entry.discredited_at is None
        assert entry.evidence_status == "contested"

    def test_naming_the_entry_charges_it_in_full_as_before(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        mem.commit_outcome("t-1", OutcomeType.FAILURE, causal_entry_keys=[f"{SCOPE}/rule"])
        entry = _fresh(mem)
        # 2 / (2 + 2*2): one failure of severity 2 at full causal weight.
        assert entry.confidence == pytest.approx(1 / 3, abs=1e-4)
        assert entry.discredited_at is not None
        assert entry.evidence_status == "discredited"

    def test_a_second_failure_in_view_discredits(self, mem: AgentMemory) -> None:
        for n in range(2):
            mem._read_tracker.clear()
            mem.read(SCOPE, "rule")
            mem.commit_outcome(f"t-{n}", OutcomeType.FAILURE)
        entry = _fresh(mem)
        assert entry.failure_count == 2
        assert entry.confidence < ev.DISCREDIT_THRESHOLD
        assert entry.discredited_at is not None

    def test_the_worst_outcome_is_held_the_first_time_too(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        mem.commit_outcome("t-1", OutcomeType.CRITICAL_FAILURE)
        entry = _fresh(mem)
        assert entry.confidence == pytest.approx(ev.DISCREDIT_THRESHOLD)
        assert entry.discredited_at is None

    def test_a_failed_attempt_holds_what_was_only_in_view(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        attempt = mem.record_attempt(summary="tried it, no luck")
        assert attempt.causal_entry_keys == [f"{SCOPE}/rule"]
        assert attempt.in_view_entry_keys == [f"{SCOPE}/rule"]
        mem.read(SCOPE, "other")
        updated = {e.key: e for e in mem.commit_outcome("t-1", OutcomeType.SUCCESS)}
        assert updated["rule"].confidence == pytest.approx(ev.DISCREDIT_THRESHOLD)
        assert updated["rule"].discredited_at is None
        assert updated["other"].success_count == 1

    def test_an_attempt_that_names_its_entry_marks_nothing_in_view(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        attempt = mem.record_attempt(
            outcome_type=OutcomeType.FAILURE, causal_entry_keys=[f"{SCOPE}/rule"]
        )
        assert attempt.in_view_entry_keys == []
        mem.commit_outcome("t-1", OutcomeType.SUCCESS)
        assert _fresh(mem).discredited_at is not None

    def test_an_entry_already_under_the_threshold_is_not_moved_or_discredited(
        self, mem: AgentMemory
    ) -> None:
        mem.write(SCOPE, "weak", "an untested correction", confidence=0.45)
        mem._read_tracker.clear()
        mem.read(SCOPE, "weak")
        mem.commit_outcome("t-1", OutcomeType.FAILURE)
        entry = _fresh(mem, "weak")
        assert entry.confidence == pytest.approx(0.45)
        assert entry.failure_count == 1 and entry.discredited_at is None
        mem.read(SCOPE, "weak")
        mem.commit_outcome("t-2", OutcomeType.FAILURE)
        assert _fresh(mem, "weak").discredited_at is not None

    def test_a_long_validated_entry_is_treated_as_before(self, adapter: FilesystemAdapter) -> None:
        """Three wins already buy an entry its first failure; being in view
        adds nothing to that, and takes nothing away."""
        results = {}
        for label, name_it in (("in-view", False), ("named", True)):
            m = AgentMemory(agent_id="a", adapter=adapter)
            m.write(SCOPE, label, "the rule", confidence=0.8)
            for n in range(3):
                m._read_tracker.clear()
                m.read(SCOPE, label)
                m.commit_outcome(f"ok-{label}-{n}", OutcomeType.SUCCESS)
            m._read_tracker.clear()
            m.read(SCOPE, label)
            m.commit_outcome(
                f"bad-{label}",
                OutcomeType.FAILURE,
                causal_entry_keys=[f"{SCOPE}/{label}"] if name_it else None,
            )
            e = adapter.read(SCOPE, label)
            results[label] = (round(e.confidence, 6), e.failure_count, e.discredited_at is None)
        assert results["in-view"] == results["named"]

    def test_a_success_is_not_changed_by_being_in_view(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "other")
        in_view = mem.commit_outcome("t-1", OutcomeType.SUCCESS)[0]
        mem.write(SCOPE, "twin", "restart the worker", confidence=0.9)
        mem._read_tracker.clear()
        mem.read(SCOPE, "twin")
        named = mem.commit_outcome(
            "t-2", OutcomeType.SUCCESS, causal_entry_keys=[f"{SCOPE}/twin"]
        )[0]
        assert in_view.confidence == pytest.approx(named.confidence)
        assert in_view.success_count == named.success_count == 1


class TestTheArithmetic:
    @pytest.mark.parametrize("prior", [0.3, 0.5, 0.55, 0.7, 0.9, 1.0])
    @pytest.mark.parametrize("wins", [0, 1, 2])
    @pytest.mark.parametrize(
        "outcome",
        [OutcomeType.MINOR_FAILURE, OutcomeType.FAILURE, OutcomeType.CRITICAL_FAILURE],
    )
    def test_a_first_failure_in_view_never_discredits_and_never_raises(
        self, prior: float, wins: int, outcome: OutcomeType
    ) -> None:
        entry = _entry("k", prior)
        for _ in range(wins):
            upd = ev.apply_outcome(entry, OutcomeType.SUCCESS)
            entry = entry.model_copy(update=upd.as_entry_update(entry.provenance.written_at))
        before = entry.confidence
        full = ev.apply_outcome(entry, outcome)
        soft = ev.apply_outcome(entry, outcome, in_view=True)
        assert not soft.discredited
        assert soft.confidence <= before + 1e-9
        assert soft.confidence >= full.confidence - 1e-9
        assert soft.confidence >= min(ev.DISCREDIT_THRESHOLD, full.confidence) - 1e-9
        if full.confidence >= ev.DISCREDIT_THRESHOLD:
            # A failure that would not have discredited is not softened at all.
            assert soft.confidence == pytest.approx(full.confidence)
            assert soft.evidence_failure == pytest.approx(full.evidence_failure)

    def test_the_cap_is_the_weight_that_lands_on_the_threshold(self) -> None:
        room = ev.in_view_room(0.9, 0.0, 0.0)
        assert ev.posterior(0.9, 0.0, room) == pytest.approx(ev.DISCREDIT_THRESHOLD)
        assert ev.in_view_room(0.4, 0.0, 0.0) == 0.0

    def test_only_a_first_failure_is_softened(self) -> None:
        assert ev.in_view_first_failure(OutcomeType.FAILURE, 0, True)
        assert not ev.in_view_first_failure(OutcomeType.FAILURE, 1, True)
        assert not ev.in_view_first_failure(OutcomeType.FAILURE, 0, False)
        assert not ev.in_view_first_failure(OutcomeType.SUCCESS, 0, True)


# ── Not followed ─────────────────────────────────────────────────────────


class TestNotFollowed:
    def test_a_failure_that_ignored_the_entry_confirms_it(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        mem.commit_outcome(
            "t-1", OutcomeType.FAILURE, not_followed_entry_keys=[f"{SCOPE}/rule"]
        )
        entry = _fresh(mem)
        assert entry.success_count == 1 and entry.failure_count == 0
        assert entry.confidence == pytest.approx(1.0)
        assert entry.discredited_at is None
        # Nobody succeeded by following it: the agent is not one of its validators.
        assert "support-agent" not in (entry.validators or [])

    def test_it_is_never_charged_even_when_also_listed_as_causal(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        mem.commit_outcome(
            "t-1",
            OutcomeType.FAILURE,
            causal_entry_keys=[f"{SCOPE}/rule"],
            not_followed_entry_keys=[f"{SCOPE}/rule"],
        )
        entry = _fresh(mem)
        assert entry.failure_count == 0 and entry.success_count == 1

    def test_on_a_failed_attempt(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "rule")
        mem.read(SCOPE, "other")
        attempt = mem.record_attempt(
            outcome_type=OutcomeType.FAILURE, not_followed_entry_keys=[f"{SCOPE}/rule"]
        )
        assert attempt.not_followed_entry_keys == [f"{SCOPE}/rule"]
        assert f"{SCOPE}/rule" not in attempt.causal_entry_keys
        updated = {e.key: e for e in mem.commit_outcome("t-1", OutcomeType.SUCCESS)}
        assert updated["rule"].success_count == 1 and updated["rule"].failure_count == 0
        # What was in view beside it is still held, not discredited.
        assert updated["other"].failure_count == 1
        assert updated["other"].discredited_at is None

    def test_a_success_ignores_the_list(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "other")
        mem.commit_outcome(
            "t-1", OutcomeType.SUCCESS, not_followed_entry_keys=[f"{SCOPE}/rule"]
        )
        entry = _fresh(mem)
        assert entry.success_count == 0 and entry.failure_count == 0
        assert entry.outcome_count == 0

    def test_the_credit_is_split_and_weighs_as_unverified(self) -> None:
        entry = _entry("k", 0.6)
        other = _entry("k2", 0.6)
        record = OutcomeRecord(
            outcome_ref="t-1",
            outcome_type=OutcomeType.FAILURE,
            causal_entry_keys=[],
            not_followed_entry_keys=[entry.entry_key, other.entry_key],
            agent_id="a",
            committed_at=NOW,
            verified_by="owner@example.com",
        )
        after, updates = ev.apply_record_to_entry(entry, record)
        assert len(updates) == 1
        alone = ev.apply_outcome(entry, OutcomeType.SUCCESS, n_causal=2, verified=False)
        assert after.confidence == pytest.approx(alone.confidence)
        assert after.validators in (None, [])

    def test_the_record_keeps_the_roles_apart(self) -> None:
        record = OutcomeRecord(
            outcome_ref="t-1",
            outcome_type=OutcomeType.FAILURE,
            causal_entry_keys=["a/b/k1", "a/b/k2"],
            in_view_entry_keys=["a/b/k1", "a/b/k2", "a/b/elsewhere"],
            not_followed_entry_keys=["a/b/k2", "a/b/k2", ""],
            agent_id="a",
            committed_at=NOW,
        )
        assert record.causal_entry_keys == ["a/b/k1"]
        assert record.in_view_entry_keys == ["a/b/k1"]
        assert record.not_followed_entry_keys == ["a/b/k2"]
        attempt = AttemptRecord(
            attempt=1,
            causal_entry_keys=["a/b/k1"],
            not_followed_entry_keys=["a/b/k1"],
            in_view_entry_keys=["a/b/k1"],
        )
        assert attempt.causal_entry_keys == [] and attempt.in_view_entry_keys == []

    def test_a_record_from_a_client_that_sends_neither_list_is_charged_in_full(self) -> None:
        """Every record written before these lists existed, and every client
        that has not been upgraded: nothing in view, nothing softened."""
        entry = _entry("k", 1.0)
        record = OutcomeRecord.model_validate(
            {
                "outcome_ref": "t-1",
                "outcome_type": "failure",
                "causal_entry_keys": [entry.entry_key],
                "agent_id": "a",
                "committed_at": NOW.isoformat(),
            }
        )
        assert record.in_view_entry_keys == [] and record.not_followed_entry_keys == []
        after, _ = ev.apply_record_to_entry(entry, record)
        assert after.confidence == pytest.approx(1 / 3, abs=1e-4)
        assert after.discredited_at is not None

    def test_the_keys_an_outcome_touches_include_the_confirmed_ones(self) -> None:
        record = OutcomeRecord(
            outcome_ref="t-1",
            outcome_type=OutcomeType.SUCCESS,
            causal_entry_keys=["a/b/final"],
            agent_id="a",
            committed_at=NOW,
            attempts=[
                AttemptRecord(
                    attempt=1,
                    outcome_type=OutcomeType.FAILURE,
                    causal_entry_keys=["a/b/tried"],
                    not_followed_entry_keys=["a/b/skipped"],
                )
            ],
        )
        assert set(ev.outcome_entry_keys(record)) == {"a/b/final", "a/b/tried", "a/b/skipped"}
        charges = ev.outcome_charges(record)
        assert [(c.outcome_type, list(c.keys), c.confirms) for c in charges] == [
            (OutcomeType.FAILURE, ["a/b/tried"], False),
            (OutcomeType.SUCCESS, ["a/b/skipped"], True),
            (OutcomeType.SUCCESS, ["a/b/final"], False),
        ]


# ── Text written after the attempt ───────────────────────────────────────


class TestTextWrittenAfterTheAttempt:
    def test_a_correction_written_after_the_failed_attempt_is_not_charged(
        self, mem: AgentMemory
    ) -> None:
        """The agent finds the wrong entry without reading it, marks the
        attempt failed against it, writes the correction, then commits. The
        failure belongs to the text that was there, not to the correction."""
        mem.record_attempt(outcome_type=OutcomeType.FAILURE, causal_entry_keys=[f"{SCOPE}/other"])
        mem.write(SCOPE, "other", "rotate the key, then restart", confidence=0.95)
        mem.commit_outcome("t-1", OutcomeType.SUCCESS)
        entry = _fresh(mem, "other")
        assert entry.value == "rotate the key, then restart"
        assert entry.failure_count == 0
        assert entry.confidence == pytest.approx(0.95)
        assert entry.discredited_at is None

    def test_an_entry_created_after_the_attempt_leaves_it(self, mem: AgentMemory) -> None:
        mem.record_attempt(outcome_type=OutcomeType.FAILURE, causal_entry_keys=[f"{SCOPE}/new"])
        mem.write(SCOPE, "new", "what actually works", confidence=0.8)
        mem.commit_outcome("t-1", OutcomeType.SUCCESS)
        entry = _fresh(mem, "new")
        assert entry.failure_count == 0 and entry.discredited_at is None

    def test_an_unchanged_restatement_still_takes_the_failure(self, mem: AgentMemory) -> None:
        """Writing the same claim again is not a correction."""
        mem.record_attempt(outcome_type=OutcomeType.FAILURE, causal_entry_keys=[f"{SCOPE}/other"])
        mem.write(SCOPE, "other", "restart the worker", confidence=0.9)
        mem.commit_outcome("t-1", OutcomeType.SUCCESS)
        assert _fresh(mem, "other").failure_count == 1

    def test_a_version_the_attempt_did_read_is_left_as_recorded(self, mem: AgentMemory) -> None:
        mem.read(SCOPE, "other")
        attempt = mem.record_attempt(summary="restart did nothing")
        assert attempt.causal_entry_versions == {f"{SCOPE}/other": 1}
        mem.write(SCOPE, "other", "rotate the key, then restart", confidence=0.95)
        mem.commit_outcome("t-1", OutcomeType.SUCCESS)
        entry = _fresh(mem, "other")
        assert entry.failure_count == 0 and entry.confidence == pytest.approx(0.95)


# ── MCP ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mcp(tmp_amfs_root: Path):
    """The stdio server's tools over a store of their own (as test_mcp_server does)."""
    import amfs_mcp.server as srv

    srv._adapter = FilesystemAdapter(root=tmp_amfs_root, namespace="test")
    srv._active_identity = "test-mcp-agent"
    srv._memories = {}
    srv._last_activity = 0.0
    yield srv
    for m in srv._memories.values():
        m.close()
    srv._memories = {}
    srv._adapter = None
    srv._active_identity = None


class TestMcp:
    def test_the_attempt_says_what_was_only_in_view(self, mcp) -> None:
        mcp.amfs_write("svc", "key", "value")
        mcp.amfs_read("svc", "key")
        reply = json.loads(mcp.amfs_record_attempt(summary="tried it"))
        assert reply["in_view_entry_keys"] == ["svc/key"]
        assert "hint" in reply and "not_followed_entry_keys" in reply["hint"]

    def test_naming_the_entry_on_the_commit_charges_it_in_full(self, mcp) -> None:
        mcp.amfs_write("svc", "key", "value")
        mcp.amfs_read("svc", "key")
        reply = json.loads(
            mcp.amfs_commit_outcome("INC-1", "critical_failure", causal_entry_keys=["svc/key"])
        )
        assert reply["entries"][0]["confidence"] == pytest.approx(0.25)

    def test_not_followed_reaches_the_record(self, mcp) -> None:
        mcp.amfs_write("svc", "key", "value")
        mcp.amfs_read("svc", "key")
        attempt = json.loads(
            mcp.amfs_record_attempt(summary="skipped the rule", not_followed_entry_keys=["svc/key"])
        )
        assert attempt["not_followed_entry_keys"] == ["svc/key"]
        reply = json.loads(mcp.amfs_commit_outcome("INC-2", "success"))
        applied = reply["attempts_applied"]
        assert applied[0]["credited_not_followed"] == ["svc/key"]
        assert applied[0]["charged"] == []
        entry = json.loads(mcp.amfs_read("svc", "key"))
        assert entry["success_count"] == 1 and entry["failure_count"] == 0
