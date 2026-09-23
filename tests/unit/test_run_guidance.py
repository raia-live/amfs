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
from amfs.memory import (
    GUIDANCE_COUNT_ATTRIBUTE,
    GUIDANCE_ID_ATTRIBUTE,
    SITUATION_ATTRIBUTE,
    provenance_attributes,
)
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

    def test_a_pooled_neighbourhood_is_thin_whatever_the_briefing_rated_the_entity(self) -> None:
        """The briefing rates strength over the entity's whole record, and an
        entity-wide winner is the very action that won on the other class
        half the time. When the retrieve's own neighbourhood is pooled — its
        recommendation abstains — its ``thin`` stands over the briefing's
        ``strong``; a plain ``act`` neighbourhood still takes the stronger."""
        class D:
            digest_type, scope = "entity", "acme/ci"
            summary = {"guidance_strength": "strong", "tried_here": [{"action": "fix:a"}]}

        pooled_meta = {
            "guidance_strength": "thin", "priors": {"source": "similar_outcomes", "contrasts": []},
            "recommendation": {"mode": "abstain", "suggested_action": None,
                               "pooled": {"actions": ["fix:a", "fix:b"], "reversals": 3},
                               "why": "near-identical tasks here alternate"},
        }
        g = Guidance.build(digests=[D()], meta=pooled_meta, entity_path="acme/ci")
        assert g.strength == "thin" and g.mode == "abstain"
        act_meta = {"guidance_strength": "thin", "priors": {"source": "similar_outcomes"},
                    "recommendation": {"mode": "act", "suggested_action": "fix:a", "why": "won"}}
        assert Guidance.build(digests=[D()], meta=act_meta, entity_path="acme/ci").strength == "strong"
        # Silent pooling (the caller did not ask to be told) is read off the
        # local contrasts themselves; an entity-wide block is never read.
        from amfs_core.actions import aggregate_priors, pooled_classes
        from tests.unit.test_action_priors import _pooled_rows
        pr = aggregate_priors(_pooled_rows())
        assert pooled_classes(pr["contrasts"]) is not None
        quiet = {"guidance_strength": "thin", "priors": pr, "recommendation": None}
        assert Guidance.build(digests=[D()], meta=quiet, entity_path="acme/ci").strength == "thin"
        wide = {**quiet, "priors": {**pr, "source": "action_stats"}}
        assert Guidance.build(digests=[D()], meta=wide, entity_path="acme/ci").strength == "strong"

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
        # No situation was declared: none is stamped.
        assert SITUATION_ATTRIBUTE not in mem.session_attributes
        # Nothing has been validated on this entity: strength is none, and the
        # caller's default policy is not to inject.
        assert g.strength == "none"
        assert not g.should_inject()

    def test_the_environment_declared_on_begin_reaches_the_trace_metadata_itself(self, mem) -> None:
        """``model`` is a reserved trace attribute: the seal path stamps it from
        ``session_metadata.model`` and drops a caller's copy from the bag. A run
        that declared its model only through ``Run.begin(model=...)`` sealed
        with none — and the repair loop's feedback contract read 0% of runs
        carrying a model. The fields travel on the metadata itself now, the
        attribute winning over identity metadata (the reading ``environment()``
        already gave)."""
        from amfs_core.models import SessionMetadata
        mem.session_metadata = SessionMetadata(model="identity-said-gpt-4", client_name="cursor")
        run = Run(mem)
        run.begin("pip install fails", entity_path="acme/ci",
                  model="gpt-5.4-mini", agent_version="ci-bot@1.4", runtime="python3.12")
        run.complete(True, verified_by="ci")
        meta = mem._last_trace.session_metadata
        assert meta.model == "gpt-5.4-mini"
        assert meta.agent_version == "ci-bot@1.4" and meta.runtime == "python3.12"
        assert meta.client_name == "cursor"
        # The bag still carries them too, for the paths that read it there.
        assert meta.attributes["model"] == "gpt-5.4-mini"

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

    def test_a_none_key_is_nothing_to_name_not_an_entry_called_none(self, mem) -> None:
        """``cited or [guidance.top_key]`` yields ``[None]`` on empty guidance;
        that must not become ``entity/None`` and must not fall back to the
        read window either — the caller named nothing on purpose."""
        run = Run(mem)
        assert run._qualify([None, "", " fix-pin "]) == ["fix-pin"]  # no entity yet: bare key kept as is
        run.begin("pip install fails", entity_path="acme/ci")
        assert run._qualify([None, "", " fix-pin "]) == ["acme/ci/fix-pin"]
        assert run._qualify([None]) == []
        run.on_tool_result("shell", {"cmd": "pip install"}, "boom", success=False, guide_on_failure=False)
        run.attempt_failed("nothing followed", causal_entry_keys=[None])
        assert run.complete(True, causal_entry_keys=[None]) == []
        meta = mem._last_trace.session_metadata.model_dump()
        assert meta["attempts"][0]["causal_entry_keys"] == []
        assert mem.read("acme/ci", "fix-pin").success_count == 0
        assert mem.read("acme/ci", "fix-pin").failure_count == 0

    def test_entry_keys_and_top_key_name_only_what_was_rendered(self, mem) -> None:
        """A synthetic lesson can outrank every authored note and still be
        omitted by the renderer; the agent never saw it, so it is not the
        fallback blame."""
        mem.write("acme/ci", "lesson-contrast-ep9", {"kind": "contrast", "avoid": ["x"]}, confidence=0.99)
        mem._read_tracker.clear()
        g = Run(mem).begin("pip install fails", entity_path="acme/ci")
        assert any(e.key == "lesson-contrast-ep9" for e in g.entries)
        assert "acme/ci/lesson-contrast-ep9" not in g.text
        assert "acme/ci/lesson-contrast-ep9" not in g.entry_keys
        assert g.top_key == "acme/ci/fix-pin"

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


class TestLessonsAndPlan:
    def test_learn_writes_a_structured_lesson_whose_record_survives_a_rewrite(self, mem) -> None:
        """The reflection step restates its lesson every task. As prose, each
        restatement opened an untested version; as a claim on (situation,
        action, worked), the record follows the claim and the words are free."""
        from amfs_core.lessons import lesson_key, lesson_of

        run = Run(mem)
        run.begin("jest snapshot failed on a backend PR", entity_path="acme/ci")
        entry = run.learn("jest snapshot failed, backend-only PR", "fix:fix_code", True,
                          "the snapshot caught a real regression in the totals")
        assert entry.key == lesson_key("jest snapshot failed, backend-only PR")
        assert entry.key.startswith("learned-") and entry.entity_path == "acme/ci"
        lesson = lesson_of(entry.value)
        assert lesson and lesson["action"] == "fix:fix_code" and lesson["worked"] is True
        # Give the claim a record, then restate it in other words.
        mem._adapter.write(entry.model_copy(update={"success_count": 3, "outcome_count": 3,
                                                    "evidence_status": "validated"}))
        again = run.learn("jest snapshot failed, backend-only PR", "fix:fix_code", True,
                          "fixing the code is right; regenerating hides the bug")
        assert again.key == entry.key
        assert again.success_count == 3 and again.evidence_status == "validated"
        # A different verdict is a new claim, and starts untested.
        flipped = run.learn("jest snapshot failed, backend-only PR", "fix:fix_code", False,
                            "the fix stopped working after the jest upgrade")
        assert flipped.key == entry.key and flipped.success_count == 0
        assert flipped.evidence_status != "validated"

    def test_a_lesson_renders_as_its_claim_and_names_the_causal_keys(self, mem) -> None:
        run = Run(mem)
        run.begin("jest snapshot failed on a backend PR", entity_path="acme/ci")
        run.learn("jest snapshot failed, backend-only PR", "fix:fix_code", True, "real regression")
        run.learn("jest snapshot failed, UI PR", "fix:update_snapshots", True, "intended change")
        run.learn("flaky webhook timeout", "fix:rerun_job", False, "no longer flaky, a real hang")
        g = Run(mem).begin("jest snapshot failed on a backend PR", entity_path="acme/ci", limit=10)
        assert "When: jest snapshot failed, backend-only PR. fix:fix_code worked. real regression" in g.text
        keys = {lesson["action"]: lesson["key"] for lesson in g.lessons}
        assert set(keys) >= {"fix:fix_code", "fix:update_snapshots"}
        assert g.lessons_claiming("fix:fix_code") == [keys["fix:fix_code"]]
        assert g.lessons_claiming("fix:update_snapshots") == [keys["fix:update_snapshots"]]
        assert g.lessons_claiming("fix:rerun_job") == []
        if "fix:rerun_job" in keys:
            assert g.lessons_claiming("fix:rerun_job", worked=False) == [keys["fix:rerun_job"]]

    def test_a_declared_situation_is_stamped_on_the_trace(self, mem) -> None:
        """The repair loop tells same-work runs apart by it: the task text's
        first line is one customer's wording, the situation is the class."""
        Run(mem).begin("“my card keeps getting declined”", entity_path="acme/support",
                       situation="  card declined at checkout ")
        assert mem.session_attributes[SITUATION_ATTRIBUTE] == "card declined at checkout"

    def test_learn_needs_an_entity(self, mem) -> None:
        with pytest.raises(ValueError):
            Run(mem).learn("x", "fix:a", True)
        entry = Run(mem).learn("x", "fix:a", True, entity_path="acme/other")
        assert entry.entity_path == "acme/other"

    def test_plan_is_read_from_the_recommendation_or_computed_from_the_priors(self) -> None:
        priors = {"tried": [{"action_key": "fix:rerun_job", "won": 3, "lost": 1, "n": 4, "p": 0.7,
                             "last_3": ["lost", "won", "won"], "agents": 1}],
                  "untried": ["fix:edit_generated_file", "fix:regen_migrations"], "source": "similar_outcomes"}
        sent = Guidance.build(meta={"priors": priors, "recommendation": {
            "mode": "act", "suggested_action": "fix:rerun_job", "why": "", "plan": ["fix:rerun_job", "fix:regen_migrations"]}})
        assert sent.plan == ["fix:rerun_job", "fix:regen_migrations"] and sent.next_action == "fix:rerun_job"
        # An older server sends no plan: the SDK orders the priors itself.
        computed = Guidance.build(meta={"priors": priors, "recommendation": {
            "mode": "act", "suggested_action": "fix:rerun_job", "why": ""}})
        assert computed.plan[0] == "fix:rerun_job"
        assert set(computed.plan[1:]) == set(priors["untried"])
        assert "Try in this order: fix:rerun_job -> fix:" in computed.text
        assert Guidance.build().plan == [] and Guidance.build().next_action is None

    def test_no_plan_without_an_act_or_explore(self) -> None:
        """An abstain, or no recommendation at all, says the record here is not
        about this task. On the ops-queue demo a plan drawn from it anyway led
        with the untried candidates in a stable order and the agent read it as
        memory telling it what to do — on a class memory had never seen."""
        priors = {"tried": [{"action_key": "fix:bump_dependency", "won": 1, "lost": 0, "n": 1, "p": 1.0,
                             "last_3": ["won"], "agents": 1}],
                  "untried": ["fix:regen_migrations", "fix:add_audit_exception"], "source": "similar_outcomes"}
        silent = Guidance.build(meta={"priors": priors})
        assert silent.plan == [] and silent.next_action is None
        assert "Try in this order" not in silent.text and "Do not spend" not in silent.text
        abstained = Guidance.build(meta={"priors": priors, "recommendation": {"mode": "abstain", "why": "pooled"}})
        assert abstained.plan == [] and abstained.next_action is None
        assert "Try in this order" not in abstained.text

    def test_a_lesson_for_the_exact_situation_reorders_the_plan_and_the_text(self) -> None:
        """Ops-queue CI #45: the server's plan put bump_dependency second (it
        wins on the unpinned audit class the neighbourhood also holds) while
        the shown lesson for the pinned class said it had failed. Given the
        task text the guidance reads the lesson as about this task and moves
        the action to the back — in the plan, in next_action after the first
        fails, and in the rendered order. A lesson for another class does not
        touch the plan; without the task text or a situation nothing moves."""
        from amfs_core.lessons import make_lesson

        class Hit:
            def __init__(self, entry):
                self.entry, self.breakdown = entry, {}

        def entry(key, value, status="validated"):
            return type("E", (), {
                "entity_path": "acme/ci", "key": key, "value": value, "memory_type": "fact",
                "evidence_status": status, "confidence": 0.9, "success_count": 3,
                "failure_count": 0, "version": 1})()

        sit = "pip-audit: requests has ; fix version · backend-only PR · changed requirements.txt"
        task = ("PR #3864 · audit pinned. CI is red (audit pinned related change).\n"
                "pip-audit: requests==2.31.0 has CVE-2024-XXXX; fix version 2.32.0\n"
                "backend-only PR, no UI files changed. Files changed: requirements.txt")
        hits = [
            Hit(entry("learned-audit", make_lesson(sit, "fix:bump_dependency", False, "the pin is deliberate"))),
            Hit(entry("learned-jest", make_lesson("jest: snapshots failed in web/components/Checkout.test.tsx",
                                                  "fix:update_snapshots", True))),
        ]
        priors = {"tried": [{"action_key": "fix:add_audit_exception", "won": 9, "lost": 1, "n": 10, "p": 0.8,
                             "last_3": ["lost", "won", "won"], "agents": 2},
                            {"action_key": "fix:bump_dependency", "won": 1, "lost": 1, "n": 2, "p": 0.5,
                             "last_3": ["won", "lost"], "agents": 1}],
                  "untried": ["fix:fix_code", "fix:update_snapshots"], "source": "similar_outcomes"}
        rec = {"mode": "act", "suggested_action": "fix:add_audit_exception", "why": "",
               "plan": ["fix:add_audit_exception", "fix:bump_dependency", "fix:fix_code", "fix:update_snapshots"]}
        g = Guidance.build(hits=hits, meta={"priors": priors, "recommendation": rec}, task_text=task)
        assert [c["action"] for c in g.applicable_lessons] == ["fix:bump_dependency"]
        assert g.plan == ["fix:add_audit_exception", "fix:fix_code", "fix:update_snapshots", "fix:bump_dependency"]
        assert "Try in this order: fix:add_audit_exception -> fix:fix_code -> fix:update_snapshots -> fix:bump_dependency." in g.text
        # The jest lesson's action was not moved up: it is about another class.
        assert g.plan[1] != "fix:update_snapshots"
        # A declared situation is compared exactly and needs no task text.
        declared = Guidance.build(hits=hits, meta={"priors": priors, "recommendation": rec}, situation=sit.upper())
        assert declared.plan == g.plan
        # Nothing to read the lessons against: the server's order stands.
        blind = Guidance.build(hits=hits, meta={"priors": priors, "recommendation": rec})
        assert blind.applicable_lessons == [] and blind.plan == rec["plan"]
        # The retry, over the untried actions, leads with what the lesson did not bar.
        retry = Guidance.build(hits=hits, meta={"priors": priors, "recommendation": dict(rec, mode="explore")},
                               task_text=task, candidate_actions=["fix:bump_dependency", "fix:fix_code"])
        assert retry.plan == ["fix:fix_code", "fix:bump_dependency"] and retry.next_action == "fix:fix_code"

    def test_a_computed_plan_is_drawn_from_the_candidates_only(self) -> None:
        """A retry asks for guidance over the actions it has not tried. The
        situation's record still has the action that just failed as a winner;
        the plan must not hand it back (the demo's retry hint led with the
        action that had just failed, three tasks running)."""
        priors = {"tried": [{"action_key": "fix:bump_dependency", "won": 3, "lost": 1, "n": 4, "p": 0.7,
                             "last_3": ["lost", "won", "won"], "agents": 1}],
                  "untried": ["fix:regen_migrations", "fix:add_audit_exception"], "source": "similar_outcomes"}
        rec = {"mode": "explore", "suggested_action": "fix:regen_migrations", "why": ""}
        g = Guidance.build(meta={"priors": priors, "recommendation": rec},
                           candidate_actions=["fix:regen_migrations", "fix:add_audit_exception"])
        assert g.plan and "fix:bump_dependency" not in g.plan
        assert g.candidate_actions == ["fix:regen_migrations", "fix:add_audit_exception"]
        # The rendered order — what the agent actually reads — is filtered too.
        assert "Try in this order:" in g.text
        assert "fix:bump_dependency" not in g.text.split("Try in this order:")[1]
        # A plan the server sent is filtered the same way, in the plan and in the text.
        sent = Guidance.build(meta={"priors": priors, "recommendation": dict(
            rec, plan=["fix:regen_migrations", "fix:bump_dependency", "fix:add_audit_exception"])},
            candidate_actions=["fix:regen_migrations", "fix:add_audit_exception"])
        assert sent.plan == ["fix:regen_migrations", "fix:add_audit_exception"]
        assert "fix:bump_dependency" not in sent.text.split("Try in this order:")[1]
        # A suggestion outside the candidates is no next action either: a server
        # asked without candidates may suggest the winner the retry has just
        # tried; the run must not be handed it back.
        stale = Guidance.build(meta={"priors": priors, "recommendation": {
            "mode": "act", "suggested_action": "fix:bump_dependency", "why": "",
            "plan": ["fix:bump_dependency"]}}, candidate_actions=["fix:regen_migrations"])
        assert stale.plan == [] and stale.next_action is None
        assert "Try in this order" not in stale.text
        # Nor does the recommendation line name it: the mode and reason stay,
        # the barred action does not.
        assert "Recommendation: act." in stale.text and "-> fix:bump_dependency" not in stale.text
        unfiltered = Guidance.build(meta={"priors": priors, "recommendation": {
            "mode": "act", "suggested_action": "fix:bump_dependency", "why": ""}})
        assert unfiltered.next_action == "fix:bump_dependency"

    def test_begin_survives_a_memory_read_failure(self, mem) -> None:
        """One ``/search`` read timeout took a demo worker thread down with an
        unhandled exception out of ``begin``. Memory being down is not the
        agent being down: the run gets empty guidance that says why, and the
        outcome still seals."""
        def broken(*a, **kw):
            raise TimeoutError("read timed out")

        mem.retrieve = broken  # type: ignore[method-assign]
        run = Run(mem)
        g = run.begin("pip install fails", entity_path="acme/ci")
        assert g.is_empty and g.strength == "none" and not g.should_inject()
        assert g.error and "TimeoutError" in g.error
        assert run.guidances == [g]
        # A failure hint is a bonus: it comes back as None, not an exception.
        assert run.on_tool_result("shell", {"cmd": "pip install"}, "boom", success=False) is None
        run.complete(False)
        assert run.completed


def test_provenance_attributes_shape() -> None:
    assert provenance_attributes(None, None) == {}
    out = provenance_attributes(" Human ", {"Run ID": 12, "nested": {"a": 1}, "skip": None,
                                            "long": "x" * 300})
    assert out["verified_by"] == "human"
    assert out["evidence_run id"] == 12
    assert out["evidence_nested"] == '{"a":1}'
    assert "evidence_skip" not in out
    assert len(out["evidence_long"]) == 256 and out["evidence_long"].endswith("…")
