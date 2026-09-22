"""The SDK's three seams — ``Run.begin`` / ``on_tool_result`` / ``complete`` —
and the ``Guidance`` they hand back, over the filesystem adapter.

What these pin: guidance is built from the briefing and a retrieve and
rendered with the shared renderer; its ``strength`` follows the evidence; the
served ``guidance_id`` and the outcome's provenance (``verified_by``,
``evidence_*``) reach the sealed trace as session attributes; a procedure
whose environment preconditions the run contradicts is set apart; and a failed
tool result yields a hint while a successful one is only recorded.
"""

from __future__ import annotations

import pytest
from amfs import AgentMemory, Guidance, Run
from amfs.memory import GUIDANCE_COUNT_ATTRIBUTE, GUIDANCE_ID_ATTRIBUTE, provenance_attributes
from amfs_core.models import MemoryType, OutcomeType
from amfs_core.render import ContextEntry, guidance_id
from amfs_filesystem.adapter import FilesystemAdapter

PROCEDURE = {
    "goal": "Recover a failed dependency install",
    "preconditions": {"runtime": "python3.12"},
    "steps": [
        {"action": "read the resolver error", "action_key": "shell:pip_install"},
        {"action": "pin the conflicting package", "action_key": "edit:requirements"},
        "re-run the install",
    ],
    "on_failure": "clear the pip cache and retry once",
    "verify": "pip check reports no broken requirements",
}


@pytest.fixture
def mem(tmp_path):
    adapter = FilesystemAdapter(root=tmp_path / ".amfs", namespace="test")
    # The filesystem adapter compiles no digests; the briefing service needs
    # the two lookups to exist to build the standalone lead digest.
    adapter.list_digests = lambda **kw: []  # type: ignore[attr-defined]
    adapter.list_branches = lambda **kw: []  # type: ignore[attr-defined]
    m = AgentMemory(agent_id="ci-bot", adapter=adapter)
    m.write("acme/ci", "fix-pin", "pin urllib3<2 when requests fails to import", confidence=0.8)
    m.write("acme/ci", "procedure-install", PROCEDURE, confidence=0.8,
            memory_type=MemoryType.PROCEDURE)
    m._read_tracker.clear()
    return m


def _attributes(mem: AgentMemory) -> dict:
    trace = mem._last_trace
    meta = trace.session_metadata.model_dump() if trace.session_metadata else {}
    return dict(meta.get("attributes") or {})


class TestGuidance:
    def test_guidance_id_names_branch_entries_and_versions(self) -> None:
        a = [ContextEntry("acme/ci", "fix-pin", "x", version=1)]
        b = [ContextEntry("acme/ci", "fix-pin", "x", version=2)]
        assert guidance_id(a) == guidance_id(a)
        assert guidance_id(a) != guidance_id(b)
        assert guidance_id(a) != guidance_id(a, branch="repair/abc")
        assert len(guidance_id(a)) == 16

    def test_build_from_nothing_is_empty_and_none(self) -> None:
        g = Guidance.build()
        assert g.is_empty and g.strength == "none" and not g.should_inject()

    def test_build_renders_procedures_ahead_of_context_and_takes_server_strength(self) -> None:
        class Hit:
            def __init__(self, entry):
                self.entry, self.breakdown = entry, {}

        class E:
            def __init__(self, key, value, mt, status):
                self.entity_path, self.key, self.value = "acme/ci", key, value
                self.memory_type, self.evidence_status = mt, status
                self.confidence, self.success_count, self.failure_count, self.version = 0.8, 0, 0, 3

        hits = [Hit(E("fix-pin", "pin urllib3", "fact", "untested")),
                Hit(E("procedure-install", PROCEDURE, "procedure", "validated"))]
        g = Guidance.build(hits=hits, meta={"guidance_strength": "strong", "priors": None,
                                            "recommendation": {"mode": "act", "why": "validated"}})
        assert g.strength == "strong" and g.should_inject()
        assert g.text.index("Procedures:") < g.text.index("Memory context:")
        assert "1. read the resolver error" in g.text
        assert "Recommendation: act" in g.text
        assert g.mode == "act"

    def test_not_applicable_from_meta_drops_the_entry_and_says_why(self) -> None:
        class Hit:
            def __init__(self, entry):
                self.entry, self.breakdown = entry, {}

        class E:
            entity_path, key, value = "acme/ci", "procedure-install", PROCEDURE
            memory_type, evidence_status = "procedure", "validated"
            confidence, success_count, failure_count, version = 0.8, 2, 0, 1

        g = Guidance.build(hits=[Hit(E())], meta={
            "not_applicable": [{"entity_path": "acme/ci", "key": "procedure-install",
                                "why": ["runtime: wants python3.12, run has python3.9"]}],
        })
        assert g.entries == []
        assert [p["key"] for p in g.not_applicable] == ["procedure-install"]
        assert "Not for this run:" in g.text and "python3.9" in g.text


class TestRun:
    def test_begin_scopes_procedures_and_stamps_the_guidance(self, mem) -> None:
        run = Run(mem)
        g = run.begin("pip install fails on requests", entity_path="acme/ci",
                      agent_version="ci-bot@1.4", runtime="python3.12", model="gpt-4o")
        assert isinstance(g, Guidance)
        assert mem.environment() == {
            "model": "gpt-4o", "agent_version": "ci-bot@1.4", "runtime": "python3.12",
        }
        # The procedure applies here (python3.12) and is the served text.
        assert [p["key"] for p in g.procedures] == ["procedure-install"]
        assert g.not_applicable == []
        assert g.guidance_id
        assert mem.session_attributes[GUIDANCE_ID_ATTRIBUTE] == g.guidance_id
        assert mem.session_attributes[GUIDANCE_COUNT_ATTRIBUTE] == 1
        # Nothing has been validated on this entity: strength is none, and the
        # caller's default policy is not to inject.
        assert g.strength == "none"
        assert not g.should_inject()

    def test_begin_on_another_runtime_sets_the_procedure_apart(self, mem) -> None:
        run = Run(mem)
        g = run.begin("pip install fails", entity_path="acme/ci", runtime="python3.9")
        assert g.procedures == []
        [row] = g.not_applicable
        assert row["key"] == "procedure-install"
        assert row["applicability_detail"] == ["runtime: wants python3.12, run has python3.9"]
        assert "Not for this run" in g.text

    def test_strength_turns_strong_once_the_scope_is_validated(self, mem) -> None:
        mem.read("acme/ci", "procedure-install")
        mem.commit_outcome("earlier", OutcomeType.SUCCESS)
        run = Run(mem)
        g = run.begin("pip install fails", entity_path="acme/ci", runtime="python3.12")
        assert g.strength == "strong" and g.should_inject()

    def test_on_tool_result_records_and_hints_only_on_failure(self, mem) -> None:
        run = Run(mem)
        run.begin("pip install fails", entity_path="acme/ci", runtime="python3.12")
        assert run.on_tool_result("shell", {"cmd": "pytest"}, "ok", success=True) is None
        hint = run.on_tool_result(
            "shell", {"cmd": "pip install -r requirements.txt"},
            "ResolutionImpossible: requests 2.31 requires urllib3<3", success=False,
            action_key="shell:pip_install",
        )
        assert hint is not None and not hint.is_empty
        assert mem.session_attributes[GUIDANCE_COUNT_ATTRIBUTE] == 2
        run.complete(True, verified_by="CI", evidence={"run_id": 98765, "url": "https://ci/1"})
        trace = mem._last_trace
        keys = [c.action_key for c in trace.tool_calls]
        assert keys[1] == "shell:pip_install"
        assert trace.tool_calls[1].success is False
        attrs = _attributes(mem)
        assert attrs["verified_by"] == "ci"
        assert attrs["evidence_run_id"] == 98765
        assert attrs["evidence_url"] == "https://ci/1"
        assert attrs["runtime"] == "python3.12"
        assert attrs[GUIDANCE_ID_ATTRIBUTE] == hint.guidance_id
        assert trace.task_input == "pip install fails"
        assert run.completed

    def test_attempt_failed_draws_a_boundary(self, mem) -> None:
        run = Run(mem)
        run.begin("pip install fails", entity_path="acme/ci")
        run.on_tool_result("shell", {"cmd": "pip install"}, "boom", success=False,
                           guide_on_failure=False)
        run.attempt_failed("pinning did not help")
        run.on_tool_result("shell", {"cmd": "pip cache purge"}, "ok", success=True)
        run.complete(True)
        meta = mem._last_trace.session_metadata.model_dump()
        assert len(meta.get("attempts") or []) == 1
        assert meta["attempts"][0]["summary"] == "pinning did not help"

    def test_named_causal_keys_replace_the_read_window(self, mem) -> None:
        """Two entries served, one followed: the outcome reaches the one that
        was named and not the other. Bare keys are qualified with the run's
        entity; ``[]`` credits nothing."""
        mem.write("acme/ci", "fix-cache", "purge the pip cache when the wheel is corrupt", confidence=0.8)
        mem._read_tracker.clear()
        run = Run(mem)
        g = run.begin("pip install fails", entity_path="acme/ci")
        served = set(g.entry_keys)
        assert {"acme/ci/fix-pin", "acme/ci/fix-cache"} <= served
        assert g.top_key in served and not g.top_key.endswith("procedure-install")

        run.on_tool_result("shell", {"cmd": "pip install urllib3<2"}, "boom", success=False,
                           guide_on_failure=False)
        run.attempt_failed("pinning did not help", causal_entry_keys=["fix-pin"])
        run.on_tool_result("shell", {"cmd": "pip cache purge"}, "ok", success=True)
        affected = run.complete(True, causal_entry_keys=["acme/ci/fix-cache"])

        meta = mem._last_trace.session_metadata.model_dump()
        assert meta["attempts"][0]["causal_entry_keys"] == ["acme/ci/fix-pin"]
        # Both named entries were touched — the attempt's with the failure,
        # the final one with the success — and nothing else.
        assert {e.key for e in affected} == {"fix-cache", "fix-pin"}
        pin = mem.read("acme/ci", "fix-pin")
        cache = mem.read("acme/ci", "fix-cache")
        assert pin.failure_count == 1 and pin.success_count == 0
        assert cache.success_count == 1 and cache.failure_count == 0
        # The procedure shared the window and was named by neither.
        proc = mem.read("acme/ci", "procedure-install")
        assert proc.success_count == 0 and proc.failure_count == 0

    def test_empty_causal_keys_credit_nothing(self, mem) -> None:
        run = Run(mem)
        run.begin("pip install fails", entity_path="acme/ci")
        assert run.complete(True, causal_entry_keys=[]) == []
        assert mem.read("acme/ci", "fix-pin").success_count == 0

    def test_assign_branch_hook_checks_out_and_falls_back(self, mem) -> None:
        seen = []

        def assign(agent_id, unit):
            seen.append((agent_id, unit))
            return "canary/fix-1"

        run = Run(mem, assign_branch=assign)
        g = run.begin("x", entity_path="acme/ci", unit="repo-7")
        assert seen == [("ci-bot", "repo-7")]
        assert mem.branch == "canary/fix-1" and g.branch == "canary/fix-1"

        def broken(agent_id, unit):
            raise RuntimeError("no assignment service")

        mem2 = AgentMemory(agent_id="ci-bot", adapter=mem._adapter)
        g2 = Run(mem2, assign_branch=broken).begin("x", entity_path="acme/ci")
        assert g2.branch == "main"


def test_provenance_attributes_shape() -> None:
    assert provenance_attributes(None, None) == {}
    out = provenance_attributes(" Human ", {"Run ID": 12, "nested": {"a": 1}, "skip": None,
                                            "long": "x" * 300})
    assert out["verified_by"] == "human"
    assert out["evidence_run id"] == 12
    assert out["evidence_nested"] == '{"a":1}'
    assert "evidence_skip" not in out
    assert len(out["evidence_long"]) == 256 and out["evidence_long"].endswith("…")
