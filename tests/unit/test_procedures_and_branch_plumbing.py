"""Procedures as a memory type, corrective-key marking, and branch plumbing.

Three things the recursive-learning loop needs from the OSS core:

* ``MemoryType.PROCEDURE`` — how to do a task. Decays slowest, gets its own
  ``procedures`` section on the lead briefing digest, is accepted by the
  write paths, and the quality report says when one has no steps.
* ``is_corrective_key`` — the repair loop's ``risk-*`` / ``correction-*``
  writes are knowledge agents must read, so they are *not* synthetic; they
  are excluded from training prompts through ``is_training_excluded_key``.
* The active branch reaches the server on every read surface: the SDK's
  ``retrieve`` and ``briefing`` forward it to a branch-aware adapter, the
  HTTP adapter sends it on ``GET /api/v1/briefing``, and ``AMFS_BRANCH``
  starts an ``AgentMemory`` on a branch without a code change.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import (
    MEMORY_TYPE_DECAY_MULTIPLIERS,
    Digest,
    DigestType,
    MemoryEntry,
    MemoryType,
    OutcomeType,
    Provenance,
    procedure_issues,
)
from amfs_core.quality import HeuristicQualityEvaluator
from amfs_cortex.briefing import BriefingService
from amfs_filesystem.adapter import FilesystemAdapter

PROCEDURE = {
    "goal": "Rotate the ingest API key without dropping traffic",
    "preconditions": ["both keys accepted by the gateway"],
    "steps": ["issue new key", "deploy readers with both", "revoke old key"],
    "on_failure": "re-enable the old key; it stays valid for 24h",
    "verify": "no 401s in the gateway log for 10 minutes",
}


# ---------------------------------------------------------------------------
# The type itself
# ---------------------------------------------------------------------------


class TestProcedureType:
    def test_is_a_memory_type_and_decays_slowest(self) -> None:
        assert MemoryType("procedure") is MemoryType.PROCEDURE
        assert MEMORY_TYPE_DECAY_MULTIPLIERS[MemoryType.PROCEDURE] == max(
            MEMORY_TYPE_DECAY_MULTIPLIERS.values()
        )

    def test_effective_confidence_outlives_a_fact(self) -> None:
        written = datetime.now(UTC) - timedelta(days=60)
        prov = Provenance(agent_id="a", session_id="s", written_at=written)
        fact = MemoryEntry(entity_path="e", key="f", value="x", provenance=prov, confidence=0.9)
        proc = fact.model_copy(update={"memory_type": MemoryType.PROCEDURE})
        assert proc.effective_confidence(decay_half_life_days=30) > fact.effective_confidence(
            decay_half_life_days=30
        )

    @pytest.mark.parametrize(
        "value, expected",
        [
            (PROCEDURE, []),
            ({"goal": "x", "steps": ["a"]}, []),
            ({"goal": "x", "steps": [{"action": "call:rotate"}]}, []),
            ({"steps": ["a"]}, ["missing_goal"]),
            ({"goal": "x"}, ["missing_steps"]),
            ({"goal": "x", "steps": []}, ["missing_steps"]),
            ({"goal": "x", "steps": [{"tool": "no action key"}]}, ["malformed_step"]),
            # A blank action is no more a step than a blank string is.
            ({"goal": "x", "steps": [{"action": "  "}]}, ["malformed_step"]),
            ({"goal": "x", "steps": [{"action": ""}]}, ["malformed_step"]),
            ({"goal": "x", "steps": ["   "]}, ["malformed_step"]),
            ("1. issue new key\n2. deploy readers\n3. revoke old key", []),
            ("- issue new key\n- revoke old key", []),
            ("restart the worker", ["not_structured"]),
            (42, ["not_structured"]),
        ],
    )
    def test_procedure_issues(self, value: Any, expected: list[str]) -> None:
        assert procedure_issues(value) == expected

    def test_quality_flags_a_procedure_without_steps(self) -> None:
        report = HeuristicQualityEvaluator().evaluate(
            {"goal": "rotate the key, which is a long enough goal to pass the length check"},
            entity_path="svc",
            key="procedure-rotate",
            memory_type="procedure",
        )
        codes = {i.type for i in report.issues}
        assert "procedure_incomplete" in codes
        assert "unstructured" not in codes
        assert report.score < 0.8

    def test_quality_accepts_a_complete_procedure(self) -> None:
        report = HeuristicQualityEvaluator().evaluate(
            PROCEDURE, entity_path="svc", key="procedure-rotate", memory_type="procedure",
        )
        assert not any(i.type == "procedure_incomplete" for i in report.issues)
        assert report.score >= 0.8

    def test_quality_is_untouched_for_other_types(self) -> None:
        report = HeuristicQualityEvaluator().evaluate(
            {"goal": "x"}, entity_path="svc", key="k", memory_type="fact",
        )
        assert not any(i.type == "procedure_incomplete" for i in report.issues)


# ---------------------------------------------------------------------------
# Corrective keys
# ---------------------------------------------------------------------------


class TestCorrectiveKeys:
    def test_corrective_is_not_synthetic(self) -> None:
        for key in ("risk-stale-policy", "correction-fix-restart"):
            assert ev.is_corrective_key(key)
            assert not ev.is_synthetic_key(key)
            assert ev.is_training_excluded_key(key)

    def test_synthetic_is_training_excluded_too(self) -> None:
        key = ev.contrast_lesson_key("ticket-1")
        assert ev.is_synthetic_key(key)
        assert not ev.is_corrective_key(key)
        assert ev.is_training_excluded_key(key)

    def test_ordinary_keys_are_neither(self) -> None:
        for key in ("fix-restart", "pattern-retry", "procedure-rotate", "decision-x"):
            assert not ev.is_corrective_key(key)
            assert not ev.is_training_excluded_key(key)

    def test_briefing_keeps_corrective_entries_in_hot_context(self, world) -> None:
        mem, service, _ = world
        mem.write("acme/support", "risk-stale-restart", "restart no longer clears the queue "
                  "because the worker now drains on shutdown", confidence=0.8,
                  memory_type=MemoryType.BELIEF)
        lead = _lead(service.briefing(entity_path="acme/support"))
        assert "risk-stale-restart" in {h["key"] for h in lead.summary["hot_context"]}


# ---------------------------------------------------------------------------
# Briefing section
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)
    monkeypatch.delenv("AMFS_BRANCH", raising=False)


@pytest.fixture
def world(tmp_path):
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="ops-agent", adapter=adapter)
    mem.write("acme/support", "fix-restart", "restart the ingest worker", confidence=0.85)
    mem.write("acme/support", "fix-rotate", "rotate the ingest API key", confidence=0.7)
    mem._read_tracker.clear()
    digests: list[Digest] = []
    adapter.list_digests = lambda **kw: list(digests)  # type: ignore[attr-defined]
    adapter.list_branches = lambda **kw: []  # type: ignore[attr-defined]
    service = BriefingService(adapter=adapter, namespace="test")
    return mem, service, digests


def _lead(digests: list[Digest]) -> Digest:
    return next(
        d for d in digests if d.digest_type == DigestType.ENTITY and d.scope == "acme/support"
    )


class TestBriefingProcedures:
    def test_no_section_without_procedures(self, world) -> None:
        _, service, _ = world
        assert "procedures" not in _lead(service.briefing(entity_path="acme/support")).summary

    def test_section_lists_procedures_validated_first(self, world) -> None:
        mem, service, _ = world
        mem.write("acme/support", "procedure-rotate", PROCEDURE, confidence=0.8,
                  memory_type=MemoryType.PROCEDURE)
        mem.write("acme/support", "procedure-scale",
                  "1. add consumers\n2. watch lag\n3. remove when lag < 1s",
                  confidence=0.9, memory_type=MemoryType.PROCEDURE)
        mem._read_tracker.clear()
        mem.read("acme/support", "procedure-rotate")
        mem.commit_outcome("ok-1", OutcomeType.SUCCESS)

        lead = _lead(service.briefing(entity_path="acme/support"))
        rows = lead.summary["procedures"]
        assert [r["key"] for r in rows] == ["procedure-rotate", "procedure-scale"]
        assert rows[0]["success_count"] == 1
        assert rows[0]["evidence_status"] == "validated"
        assert rows[0]["goal"] == PROCEDURE["goal"]
        assert rows[1]["goal"] is None
        assert rows[1]["value_preview"].startswith("1. add consumers")
        assert rows[0]["written_at"]
        # Facts stay out of it.
        assert "fix-restart" not in {r["key"] for r in rows}

    def test_procedures_do_not_take_hot_context_slots(self, world) -> None:
        """The section exists so methods stop competing with facts for the
        three hot-context rows. A procedure that outranks every fact on
        priority must still leave those rows to the facts — and not cost one
        of them a slot by being fetched and then dropped."""
        mem, service, _ = world
        for i in range(3):
            mem.write("acme/support", f"procedure-{i}", {**PROCEDURE, "goal": f"method {i}"},
                      confidence=0.99, memory_type=MemoryType.PROCEDURE)
        mem.write("acme/support", "fix-scale", "add consumers when lag grows", confidence=0.6)
        mem._read_tracker.clear()
        for i in range(3):
            mem.read("acme/support", f"procedure-{i}")
        mem.commit_outcome("ok-2", OutcomeType.SUCCESS)

        lead = _lead(service.briefing(entity_path="acme/support"))
        hot = [h["key"] for h in lead.summary["hot_context"]]
        assert not any(k.startswith("procedure-") for k in hot)
        assert set(hot) == {"fix-restart", "fix-rotate", "fix-scale"}
        assert {r["key"] for r in lead.summary["procedures"]} == {
            "procedure-0", "procedure-1", "procedure-2",
        }

    def test_discredited_procedure_is_not_repeated(self, world) -> None:
        mem, service, _ = world
        mem.write("acme/support", "procedure-rotate", PROCEDURE, confidence=0.7,
                  memory_type=MemoryType.PROCEDURE)
        mem._read_tracker.clear()
        mem.read("acme/support", "procedure-rotate")
        mem.commit_outcome("bad-1", OutcomeType.FAILURE)

        lead = _lead(service.briefing(entity_path="acme/support"))
        assert "procedures" not in lead.summary
        assert [d["key"] for d in lead.summary["discredited"]] == ["procedure-rotate"]

    def test_compact_keeps_the_section(self, world) -> None:
        mem, service, _ = world
        mem.write("acme/support", "procedure-rotate", PROCEDURE, confidence=0.8,
                  memory_type=MemoryType.PROCEDURE)
        digests = service.briefing(entity_path="acme/support", compact=True)
        assert len(digests) == 1
        assert [r["key"] for r in digests[0].summary["procedures"]] == ["procedure-rotate"]

    def test_since_trims_the_section(self, world) -> None:
        mem, service, _ = world
        mem.write("acme/support", "procedure-rotate", PROCEDURE, confidence=0.8,
                  memory_type=MemoryType.PROCEDURE)
        later = datetime.now(UTC) + timedelta(minutes=1)
        lead = _lead(service.briefing(entity_path="acme/support", since=later))
        assert lead.summary["procedures"] == []


# ---------------------------------------------------------------------------
# Branch plumbing
# ---------------------------------------------------------------------------


class _BranchAwareAdapter(FilesystemAdapter):
    """A filesystem adapter that records the branch every search / retrieve /
    briefing call named, the way the Postgres and HTTP adapters take one."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.calls: list[tuple[str, str | None]] = []

    def search(self, query, branch: str | None = None, **kw):  # type: ignore[override]
        self.calls.append(("search", branch))
        return super().search(query)

    def retrieve(self, query: str, **kw: Any):
        self.calls.append(("retrieve", kw.get("branch")))
        return []

    def briefing(self, **kw: Any):
        self.calls.append(("briefing", kw.get("branch")))
        return []


class TestBranchPlumbing:
    def test_retrieve_forwards_the_active_branch(self, tmp_path) -> None:
        adapter = _BranchAwareAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        assert mem.branch == "repair/fix-1"
        mem.retrieve("how do I rotate the key")
        assert ("retrieve", "repair/fix-1") in adapter.calls

    def test_retrieve_does_not_send_main(self, tmp_path) -> None:
        adapter = _BranchAwareAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter)
        mem.retrieve("anything")
        assert ("retrieve", None) in adapter.calls

    def test_retrieve_call_argument_wins_over_the_active_branch(self, tmp_path) -> None:
        adapter = _BranchAwareAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        mem.retrieve("anything", branch="canary/fix-1")
        assert ("retrieve", "canary/fix-1") in adapter.calls

    def test_local_search_fallback_forwards_the_branch(self, tmp_path) -> None:
        adapter = _BranchAwareAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        mem.search(query="rotate")
        assert ("search", "repair/fix-1") in adapter.calls

    def test_search_survives_an_adapter_without_the_keyword(self, tmp_path) -> None:
        adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        mem.write("svc", "k", "the value of k, long enough to matter here")
        assert mem.search(query="value") is not None  # no TypeError

    def test_briefing_forwards_the_active_branch(self, tmp_path) -> None:
        adapter = _BranchAwareAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        mem.briefing(entity_path="svc")
        assert ("briefing", "repair/fix-1") in adapter.calls
        mem.briefing(entity_path="svc", branch="main")
        assert ("briefing", None) in adapter.calls

    def test_briefing_sheds_only_the_keyword_an_old_adapter_rejects(self, tmp_path) -> None:
        """An adapter that predates ``branch`` but knows ``compact`` / ``since``
        must still be asked for the compact delta — not for a full briefing
        of everything, which is what dropping every optional keyword at the
        first TypeError produced."""
        calls: list[dict[str, Any]] = []

        class _PreBranchAdapter(FilesystemAdapter):
            def briefing(self, *, entity_path=None, agent_id=None, limit=10,
                         compact=False, since=None, credit_reuse=False):
                calls.append({"compact": compact, "since": since, "credit_reuse": credit_reuse})
                return []

        adapter = _PreBranchAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        moment = datetime.now(UTC)
        mem.briefing(entity_path="svc", compact=True, since=moment, credit_reuse=True)
        assert calls == [{"compact": True, "since": moment, "credit_reuse": True}]

    def test_briefing_survives_an_adapter_that_knows_none_of_the_options(self, tmp_path) -> None:
        calls: list[dict[str, Any]] = []

        class _OldAdapter(FilesystemAdapter):
            def briefing(self, *, entity_path=None, agent_id=None, limit=10):
                calls.append({"entity_path": entity_path, "limit": limit})
                return []

        adapter = _OldAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter, branch="repair/fix-1")
        mem.briefing(entity_path="svc", compact=True, since=datetime.now(UTC))
        assert calls == [{"entity_path": "svc", "limit": 10}]

    def test_briefing_reraises_an_adapter_s_own_type_error(self, tmp_path) -> None:
        class _BrokenAdapter(FilesystemAdapter):
            def briefing(self, **kw: Any):
                raise TypeError("digest scoring got a str where a datetime was expected")

        adapter = _BrokenAdapter(root=tmp_path / ".amfs", namespace="test")
        mem = AgentMemory(agent_id="a", adapter=adapter)
        with pytest.raises(TypeError, match="digest scoring"):
            mem.briefing(entity_path="svc")

    def test_env_starts_the_memory_on_a_branch(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("AMFS_BRANCH", "canary/fix-7")
        adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
        assert AgentMemory(agent_id="a", adapter=adapter).branch == "canary/fix-7"
        assert AgentMemory(agent_id="a", adapter=adapter, branch="x").branch == "x"
        monkeypatch.setenv("AMFS_BRANCH", "   ")
        assert AgentMemory(agent_id="a", adapter=adapter).branch == "main"


class TestHttpAdapterBriefingBranch:
    def test_branch_is_a_query_param_only_off_main(self) -> None:
        from tests.unit.test_http_adapter import _make_adapter

        adapter, calls = _make_adapter({"/api/v1/briefing": {"digests": [], "total": 0}})
        adapter.briefing(entity_path="svc", branch="repair/fix-1")
        adapter.briefing(entity_path="svc", branch="main")
        adapter.briefing(entity_path="svc")
        assert calls[0]["params"]["branch"] == "repair/fix-1"
        assert "branch" not in calls[1]["params"]
        assert "branch" not in calls[2]["params"]


class TestServerBriefingBranch:
    def test_get_briefing_passes_branch_to_memory(self, monkeypatch) -> None:
        import asyncio

        from amfs_http import server

        seen: dict[str, Any] = {}

        class _Mem:
            def briefing(self, **kw: Any) -> list:
                seen.update(kw)
                return []

        monkeypatch.setattr(server, "_get_memory", lambda: _Mem())
        monkeypatch.setattr(server, "_get_visibility_filter", lambda request: None)
        out = asyncio.run(
            server.get_briefing(
                request=None, entity_path="svc", agent_id="a", limit=5,
                credit_reuse=False, compact=True, since=None, branch="repair/fix-1",
                response=None, _auth=None,
            )
        )
        assert out["total"] == 0
        assert seen["branch"] == "repair/fix-1"
        assert seen["compact"] is True

    def test_get_briefing_omits_branch_when_not_asked(self, monkeypatch) -> None:
        import asyncio

        from amfs_http import server

        seen: dict[str, Any] = {}

        class _Mem:
            def briefing(self, **kw: Any) -> list:
                seen.update(kw)
                return []

        monkeypatch.setattr(server, "_get_memory", lambda: _Mem())
        monkeypatch.setattr(server, "_get_visibility_filter", lambda request: None)
        asyncio.run(
            server.get_briefing(
                request=None, entity_path="svc", agent_id=None, limit=10,
                credit_reuse=False, compact=False, since=None, branch=None,
                response=None, _auth=None,
            )
        )
        assert "branch" not in seen
