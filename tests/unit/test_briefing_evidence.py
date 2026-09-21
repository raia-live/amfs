"""The briefing tells the agent what the outcome record says about a scope.

``validated`` / ``discredited`` / ``regime_shift`` sections on the lead entity
digest, hot context that omits discredited entries, and the compact mode.
Runs the Cortex ``BriefingService`` over the filesystem adapter with a stub
``list_digests`` so no Postgres is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from amfs import AgentMemory
from amfs_core import evidence as ev
from amfs_core.models import Digest, DigestType, OutcomeType
from amfs_cortex.briefing import BriefingService
from amfs_filesystem.adapter import FilesystemAdapter


@pytest.fixture(autouse=True)
def _evidence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ev.OUTCOME_MODEL_ENV, raising=False)


@pytest.fixture
def world(tmp_path):
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    mem = AgentMemory(agent_id="ops-agent", adapter=adapter)
    mem.write("acme/support", "fix-restart", "restart the ingest worker", confidence=0.85)
    mem.write("acme/support", "fix-rotate", "rotate the ingest API key", confidence=0.7)
    mem.write("acme/support", "fix-scale", "scale the ingest consumers", confidence=0.7)
    mem._read_tracker.clear()

    digests: list[Digest] = []
    adapter.list_digests = lambda **kw: list(digests)  # type: ignore[attr-defined]
    adapter.list_branches = lambda **kw: []  # type: ignore[attr-defined]
    service = BriefingService(adapter=adapter, namespace="test")
    return mem, service, digests


def _outcome(mem: AgentMemory, key: str, outcome: OutcomeType, ref: str) -> None:
    mem.read("acme/support", key)
    mem.commit_outcome(ref, outcome)


def _lead(digests: list[Digest]) -> Digest:
    return next(
        d for d in digests if d.digest_type == DigestType.ENTITY and d.scope == "acme/support"
    )


def test_sections_reflect_the_record(world) -> None:
    mem, service, _ = world
    for i in range(3):
        _outcome(mem, "fix-rotate", OutcomeType.SUCCESS, f"ok-{i}")
    _outcome(mem, "fix-scale", OutcomeType.FAILURE, "bad-1")

    lead = _lead(service.briefing(entity_path="acme/support"))
    validated = [v["key"] for v in lead.summary["validated"]]
    discredited = lead.summary["discredited"]
    assert validated == ["fix-rotate"]
    assert lead.summary["validated"][0]["success_count"] == 3
    assert [d["key"] for d in discredited] == ["fix-scale"]
    assert discredited[0]["failure_count"] == 1
    assert discredited[0]["value_preview"].startswith("scale the ingest")
    assert "regime_shift" not in lead.summary
    # Hot context carries the evidence vocabulary and omits the gated entry.
    hot = {h["key"]: h for h in lead.summary["hot_context"]}
    assert "fix-scale" not in hot
    assert hot["fix-rotate"]["evidence_status"] == "validated"
    assert hot["fix-restart"]["evidence_status"] == "untested"


def test_discredited_entry_shows_what_replaced_it(world) -> None:
    mem, service, _ = world
    mem.read("acme/support", "fix-restart")
    mem.record_attempt(summary="restart did nothing")
    mem.read("acme/support", "fix-rotate")
    mem.commit_outcome("ticket-1", OutcomeType.SUCCESS)

    lead = _lead(service.briefing(entity_path="acme/support"))
    disc = {d["key"]: d for d in lead.summary["discredited"]}
    assert disc["fix-restart"]["replaced_by"] == ["acme/support/fix-rotate"]
    # The synthetic lesson itself is not listed as knowledge to act on.
    assert all(not ev.is_synthetic_key(v["key"]) for v in lead.summary["validated"])
    assert all(not ev.is_synthetic_key(h["key"]) for h in lead.summary["hot_context"])


def test_regime_shift_when_a_long_validated_rule_starts_failing(world) -> None:
    mem, service, _ = world
    for i in range(6):
        _outcome(mem, "fix-restart", OutcomeType.SUCCESS, f"ok-{i}")
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "bad-1")
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "bad-2")

    lead = _lead(service.briefing(entity_path="acme/support"))
    shift = lead.summary["regime_shift"]
    assert shift["suspected"] is True
    assert [e["key"] for e in shift["entries"]] == ["fix-restart"]
    assert shift["entries"][0]["success_count"] == 6
    assert "changed" in shift["message"]


def test_one_failure_on_a_long_validated_rule_is_a_first_strike_not_a_shift(world) -> None:
    """The label forgives one failure against a long run (first-strike
    tolerance keeps it ``validated``); the regime-shift section has to agree,
    or the briefing would tell agents the world changed on the same failure it
    tells them to keep acting through."""
    mem, service, _ = world
    for i in range(6):
        _outcome(mem, "fix-restart", OutcomeType.SUCCESS, f"ok-{i}")
    _outcome(mem, "fix-restart", OutcomeType.FAILURE, "bad-1")

    lead = _lead(service.briefing(entity_path="acme/support"))
    assert "regime_shift" not in lead.summary
    hot = {h["key"]: h for h in lead.summary["hot_context"]}
    assert hot["fix-restart"]["evidence_status"] == "validated"


def test_compact_returns_only_the_lead_with_evidence(world) -> None:
    mem, service, digests = world
    digests.append(
        Digest(
            digest_type=DigestType.ENTITY,
            scope="acme/support",
            summary={
                "narrative": "x" * 1000,
                "agents": ["ops-agent"],
                "risks": ["r"],
                "who_to_ask": [],
            },
            entry_count=3,
            source_agents=["ops-agent"],
            compiled_at=datetime.now(UTC),
            namespace="test",
            branch="main",
        )
    )
    digests.append(
        Digest(
            digest_type=DigestType.AGENT_BRIEF,
            scope="ops-agent",
            summary={"entities_written": ["acme/support"]},
            entry_count=3,
            source_agents=["ops-agent"],
            compiled_at=datetime.now(UTC),
            namespace="test",
            branch="main",
        )
    )
    _outcome(mem, "fix-rotate", OutcomeType.SUCCESS, "ok")

    full = service.briefing(entity_path="acme/support", agent_id="ops-agent")
    assert len(full) == 2
    compact = service.briefing(entity_path="acme/support", agent_id="ops-agent", compact=True)
    assert len(compact) == 1
    lead = compact[0]
    assert set(lead.summary) <= {
        "narrative",
        "hot_context",
        "validated",
        "discredited",
        "regime_shift",
        "guidance_strength",
    }
    assert len(lead.summary["narrative"]) < 1000
    assert [v["key"] for v in lead.summary["validated"]] == ["fix-rotate"]
    # A validated entry in scope is what makes guidance worth acting on.
    assert lead.summary["guidance_strength"] == "strong"
    assert "risks" not in lead.summary and "who_to_ask" not in lead.summary
