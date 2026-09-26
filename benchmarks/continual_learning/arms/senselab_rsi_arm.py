"""``senselab-rsi``: the shipped repair loop on the production SaaS, decided by the product and
observed by the harness (HYPOTHESIS_RSI.md).

What the arm adds to ``senselab``:

* **Per cell, once**: a continuous behaviour judge on the cell's agent (the frozen rubric for
  the scenario), and the agent's repair settings — policy and canary knobs — as the study
  preregisters them. Everything after that is the product's: the eval worker grades every
  sealed trace, clusters failing verdicts into behaviours, opens incidents, proposes a fix
  (auto-propose), queues its Tier 1 replay, applies the policy (start a canary, hold, ship),
  and verifies prevention afterwards.
* **Per session**: nothing new. The SDK's bound adapter carries ``X-AMFS-Agent-Id`` and
  ``X-AMFS-Session``; the hosted API puts the session in the canary or control arm of the
  agent's running canary (``amfs_tenant.canary_routing``), serves reads from the fix's branch
  for the canary arm, and stamps ``canary_fix_id`` / ``canary_arm`` / ``memory_branch`` on the
  sealed trace. The harness only records the ``X-AMFS-Canary`` header it was answered with.
* **Between episodes (the look schedule)**: what the hourly ``canary_sweep`` would do for this
  one agent, through the account's own endpoints, every ``look_every`` sessions —
  ``POST /fixes/{id}/canary/evaluate`` for the running canary, ``POST /fixes/{id}/canary`` for
  the oldest tested fix waiting on a freed slot, ``POST /fixes/{id}/recheck`` for shipped fixes
  — plus a snapshot of the agent's fixes so the episode row carries the lifecycle.
  Nothing that decides (propose, gate, ``decide()``, ship, roll back) runs here.

Variants:

``senselab-rsi``           policy ``auto_after_canary``; judges on; outcomes verified.
``senselab-rsi-nocanary``  policy ``auto_after_replay``: a passed Tier 1 test ships.
``senselab-rsi-noreplay``  policy ``review`` with the harness starting the canary as soon as a
                           fix is proposed, whatever its test says, and shipping (``approve-
                           memory``) when the canary passes. The test still runs and is
                           recorded, so what it would have blocked is measurable.
``senselab-rsi-review``    policy ``review``: nothing ships; the proposals and tests a person
                           would have read are the result (review load).
``senselab-rsi-minimal``   the agent protocol cut to ``retrieve`` + ``commit_outcome``: no
                           attempt boundaries, no recorded actions, no priors.
``senselab-rsi-selfreport`` the ``senselab-rsi`` settings with ``verified_by`` omitted on every
                           outcome (E3 self-report): same information, unverified.
``senselab-rsi-judges``    every outcome committed as the agent would report it — ``success``
                           unless it escalated — with no attempt boundaries and no
                           ``verified_by``; the judges are the only failure signal (E3 judges).

Canary knobs are read from ``CL_RSI_CANARY_PCT`` / ``CL_RSI_CANARY_MIN`` (defaults 50 / 8, the
E2 mapping of the product defaults 10 / 20) and the harness-side timeout in sessions from
``CL_RSI_CANARY_TIMEOUT`` (default 60): the product's own timeout is in days and cannot fire
inside a cell, so a canary still running after that many sessions is stopped by the harness
and recorded as timed out. The look interval is ``CL_RSI_LOOK_EVERY`` (default 5).
"""

from __future__ import annotations

import os
import time
from typing import Any

from .. import config
from .base import ArmAccounting, EpisodeSession, Outcome
from .senselab_arm import SenseLabArm, _SenseLabSession

JUDGE_ID = "cl-task-failure"
#: Frozen before any run; published with the paper. The judge names the failure so discovery
#: can cluster verdicts into behaviours; it must quote the trace.
JUDGE_PROMPT = (
    "You grade one run of an operations agent. Decide whether the run FAILED the task it was "
    "given and, if so, name the failure in one short phrase that would be the same for every "
    "run that fails the same way (for example 'stale fix: reset_password on a 2FA ticket', "
    "'wrong root cause', 'escalated a resolvable ticket'). A run failed when it ended in an "
    "unresolved task, a rejected or rolled-back action, an escalation the task did not need, "
    "or an exhausted retry budget; a run that failed one or more attempts before succeeding "
    "is a PASS with the failed attempts noted. Use only what the trace shows: the task as "
    "posed, the tool calls and their results, the memory entries the agent read, the decision "
    "summary and the outcome. Quote the tool result or summary line your verdict rests on.\n\n"
    "When the run failed, say where the wrong action came from. Compare the failing action "
    "with the memory entries under memory_read: if an entry recommends, records or endorses "
    "the action the agent took (or the class of action, for this kind of task), add a "
    "criterion named 'followed-memory' with passed=false whose evidence_quote is the exact "
    "text of that entry's value (copy it verbatim, up to one sentence). If no entry read "
    "supports the action, add 'followed-memory' with passed=true and quote the agent's own "
    "reasoning or the tool call instead. Do not grade tone or verbosity.\n\n"
    "Scenario notes: {rubric}\n\nTRACE:\n{{{{trace}}}}"
)
# ``JUDGE_PROMPT.format(rubric=...)`` yields the literal ``{{trace}}`` the judge store requires.
# The 'followed-memory' criterion is what lets the platform's lever classifier see the memory:
# a verbatim quote of an entry resolves to an ``entry`` evidence ref (amfs_eval/runner.py
# ``_resolve_evidence``), which ``classify_lever`` counts as knowledge. Without it every
# failing run cites only the tool span that carried the bad action and the classifier reads
# the failure as ``code`` (pilot 1, 2026-09-26: two template fixes, no branch, nothing to canary).

OPEN_STATUSES = ("proposed", "testing", "test_passed", "test_failed", "test_inconclusive", "canary")
SHIPPED_STATUSES = ("memory_shipped", "verified", "regressed")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _fix_row(f: dict[str, Any]) -> dict[str, Any]:
    """The part of a fix record the episode row keeps."""
    canary = f.get("canary") if isinstance(f.get("canary"), dict) else None
    test = f.get("test") if isinstance(f.get("test"), dict) else None
    manifest = f.get("manifest") if isinstance(f.get("manifest"), dict) else None
    row: dict[str, Any] = {
        "id": str(f.get("id")), "status": f.get("status"), "lever": f.get("lever"),
        "source": f.get("source"), "branch": f.get("branch"), "behavior_id": f.get("behavior_id"),
        "incident_id": f.get("incident_id"), "created_at": f.get("created_at"),
        "shipped_at": f.get("shipped_at"), "rolled_back_at": f.get("rolled_back_at"),
    }
    if manifest:
        row["manifest"] = {"predicted_fixes": len(manifest.get("predicted_fixes") or []),
                           "predicted_regressions": len(manifest.get("predicted_regressions") or []),
                           "expected_prevention": manifest.get("expected_prevention"),
                           "repairer_variant": manifest.get("repairer_variant"),
                           "model": manifest.get("model")}
    if test:
        row["test"] = {k: test.get(k) for k in ("status", "passed", "repaired", "total", "regressed",
                                                "anchors", "fidelity", "unreplayable", "model")}
    if canary:
        row["canary"] = {
            "status": canary.get("status"), "pct": canary.get("pct"), "min_outcomes": canary.get("min_outcomes"),
            "started_at": canary.get("started_at"), "decided_at": canary.get("decided_at"),
            "evaluations": canary.get("evaluations"), "recurrence_p": canary.get("recurrence_p"),
            "success_p": canary.get("success_p"), "proven_by": canary.get("proven_by"), "reason": canary.get("reason"),
            "canary": {k: (canary.get("canary") or {}).get(k) for k in ("traces", "graded", "recurred", "outcomes",
                                                                         "succeeded", "verified_outcomes")},
            "control": {k: (canary.get("control") or {}).get(k) for k in ("traces", "graded", "recurred", "outcomes",
                                                                           "succeeded", "verified_outcomes")},
        }
    return row


class _RsiSession(_SenseLabSession):
    def end(self, outcome: Outcome, *, task_input: str, response_text: str, cited_keys: list[str]) -> None:
        super().end(outcome, task_input=task_input, response_text=response_text, cited_keys=cited_keys)
        trace = getattr(self.mem, "_last_trace", None)
        trace_id = getattr(trace, "id", None) if trace is not None else None
        self.arm.sessions_sealed += 1
        self.arm.ledger.append({"episode": self.episode, "replica": self.replica, "trace_id": str(trace_id) if trace_id else None,
                                "success": outcome.success, "severity": outcome.severity,
                                "canary": self.acct.notes.get("canary"), "memory_branch": self.acct.notes.get("memory_branch")})

    def record_action(self, tool: str, arguments: dict[str, Any], result: str, success: bool) -> None:
        if not self.arm.records_actions:
            return None
        return super().record_action(tool, arguments, result, success)


class SenseLabRsiArm(SenseLabArm):
    name = "senselab-rsi"
    fleet_replicas = True                 # one repair loop per cell: fleet members are replicas
    repair_policy = "auto_after_canary"   # auto_after_replay | auto_after_canary | review
    judges_on = True
    records_actions = True
    # E3 verdict source. ``oracle``: the environment's severity, ``verified_by`` stamped.
    # ``self-report``: the same severity, unverified. ``judges``: ``success`` unless the agent
    # escalated, unverified, no attempt boundaries — the judges are the only failure signal.
    verdict_source = "oracle"
    # ``-noreplay``: the harness starts a canary for every proposed fix without waiting for the
    # Tier 1 verdict, and ships a passed canary by hand (the policy is ``review``).
    canary_regardless_of_test = False
    canary_pct = _int_env("CL_RSI_CANARY_PCT", 50)
    canary_min_outcomes = _int_env("CL_RSI_CANARY_MIN", 8)
    canary_timeout_sessions = _int_env("CL_RSI_CANARY_TIMEOUT", 60)
    look_every = _int_env("CL_RSI_LOOK_EVERY", 5)

    # -- lifecycle ------------------------------------------------------------------
    def open(self, scope: str) -> None:
        super().open(scope)
        from ..pro_client import ProClient, assert_dev

        # The study runs on the production SaaS by design (HYPOTHESIS_RSI.md): the dedicated
        # benchmark account is the only thing this key can write to. ``assert_dev`` is the
        # prototype arm's guard; it is honoured unless the owner overrode it in the env.
        assert_dev(config.AMFS_HTTP_URL, config.AMFS_PRO_URL)
        self.pro = ProClient()
        self.ledger: list[dict[str, Any]] = []       # one row per sealed session
        self.events: list[dict[str, Any]] = []       # what the harness did to the loop, and when
        self.fixes_seen: dict[str, dict[str, Any]] = {}
        self.sessions_sealed = 0
        self._sessions_at_look = 0
        self._canary_started_at: dict[str, int] = {}   # fix id -> sessions_sealed when the canary started
        self._agents_ready: set[str] = set()
        self._rubric = ""
        self._scenario_name = ""
        self._agent: str | None = None

    def configure(self, scenario) -> None:
        super().configure(scenario)
        self._rubric = getattr(scenario, "judge_rubric", "") or "none"
        self._scenario_name = getattr(scenario, "name", "")
        # The cell's shared agent identity: every replica maps onto it.
        self._agent = self.saas_agent_id(scenario.agent_for(0))
        self._ready(self._agent)

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        s = _RsiSession(self, agent_id, episode)
        if self._agent is None:
            self._agent = s.agent_id
            self._ready(self._agent)
        return s

    def outcome_for_commit(self, outcome: Outcome, session: _SenseLabSession) -> tuple[str, dict[str, Any]]:
        if self.verdict_source == "judges":
            # What an agent that reports "done" sends: success unless it gave up. The
            # environment's verdict never reaches memory; a judge has to find the failure.
            return ("failure" if outcome.escalated else "success"), {}
        extra: dict[str, Any] = {}
        if self.verdict_source == "oracle" and self.verified_by:
            extra["verified_by"] = self.verified_by
        return outcome.severity, extra

    # -- per-agent setup (once per cell) ---------------------------------------------
    def _ready(self, agent_id: str) -> None:
        if agent_id in self._agents_ready:
            return
        from ..pro_client import ProApiError

        t0 = time.perf_counter()
        try:
            if self.judges_on:
                self.pro.ensure_judge(JUDGE_ID, agent_id, "Task failure (benchmark, frozen rubric)",
                                      JUDGE_PROMPT.format(rubric=self._rubric[:1500]), mode="continuous")
            settings = self.pro.set_repair_settings(
                agent_id, repair_policy=self.repair_policy, canary_pct=self.canary_pct,
                canary_min_outcomes=self.canary_min_outcomes, canary_timeout_days=1,
            )
            contract = None
            try:
                contract = self.pro.feedback_contract(agent_id)
            except ProApiError:
                pass
            self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "kind": "agent_ready",
                                "agent_id": agent_id, "settings": {k: settings.get(k) for k in (
                                    "repair_policy", "canary_pct", "canary_min_outcomes", "canary_timeout_days",
                                    "replay_fidelity_floor")}, "contract": contract,
                                "ms": round((time.perf_counter() - t0) * 1000, 1)})
        except ProApiError as e:
            self.events.append({"t": time.time(), "kind": "agent_ready_error", "agent_id": agent_id,
                                "error": f"{e.status}: {str(e.detail)[:300]}"})
            raise
        self._agents_ready.add(agent_id)

    # -- the look schedule -----------------------------------------------------------
    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        if self._agent is None:
            return None
        acct = ArmAccounting()
        t0 = time.perf_counter()
        due = (self.sessions_sealed - self._sessions_at_look) >= self.look_every
        try:
            fixes = self.pro.list_fixes(self._agent)
            acct.ops += 1
        except Exception as e:  # noqa: BLE001
            acct.notes["fixes_error"] = f"{type(e).__name__}: {e}"[:200]
            acct.memory_ms = (time.perf_counter() - t0) * 1000
            return acct
        snapshot = [_fix_row(f) for f in fixes]
        changed = []
        for row in snapshot:
            prev = self.fixes_seen.get(row["id"])
            if prev is None or prev.get("status") != row["status"] or (prev.get("canary") or {}).get("status") != (row.get("canary") or {}).get("status"):
                changed.append(row)
                self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "episode": episode,
                                    "kind": "fix_status", "fix": row})
            self.fixes_seen[row["id"]] = row
        if due:
            self._sessions_at_look = self.sessions_sealed
            acct.ops += self._advance(fixes, episode)
        acct.memory_ms = (time.perf_counter() - t0) * 1000
        acct.notes = {
            "look": bool(due), "sessions_sealed": self.sessions_sealed,
            "fixes": {s: sum(1 for r in snapshot if r["status"] == s) for s in sorted({r["status"] for r in snapshot})},
            "changed": changed[:8],
        }
        active = next((r for r in snapshot if r["status"] == "canary"), None)
        if active:
            acct.notes["canary_active"] = {"id": active["id"], **(active.get("canary") or {})}
        return acct

    def _advance(self, fixes: list[dict[str, Any]], episode: int) -> int:
        """One agent's share of the hourly sweep, through the account's endpoints. Returns
        the number of calls made. Never raises."""
        from ..pro_client import ProApiError

        calls = 0

        def log(kind: str, fix_id: str, **data: Any) -> None:
            self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "episode": episode,
                                "kind": kind, "fix_id": fix_id, **data})

        running = [f for f in fixes if f.get("status") == "canary"]
        slot_busy = bool(running)
        for f in running:
            fid = str(f["id"])
            self._canary_started_at.setdefault(fid, self.sessions_sealed)
            age = self.sessions_sealed - self._canary_started_at[fid]
            try:
                if age >= self.canary_timeout_sessions:
                    after = self.pro.stop_canary(fid, reason=f"benchmark timeout: {age} sessions (HYPOTHESIS_RSI.md E2)")
                    log("canary_timeout", fid, age=age, status=after.get("status"))
                    slot_busy = False
                else:
                    after = self.pro.evaluate_canary(fid)
                    c = after.get("canary") or {}
                    log("canary_evaluate", fid, status=after.get("status"), canary_status=c.get("status"),
                        recurrence_p=c.get("recurrence_p"), success_p=c.get("success_p"),
                        canary_n=(c.get("canary") or {}).get("outcomes"), control_n=(c.get("control") or {}).get("outcomes"))
                    if after.get("status") != "canary":
                        slot_busy = False
                    if after.get("status") == "test_passed" and self.repair_policy == "review" and c.get("status") == "passed":
                        # -noreplay: a passed canary under ``review`` is held for a person; the
                        # harness is that person.
                        shipped = self.pro.approve_memory(fid)
                        calls += 1
                        log("ship_by_hand", fid, status=shipped.get("status"))
                calls += 1
            except ProApiError as e:
                log("canary_error", fid, error=f"{e.status}: {str(e.detail)[:200]}")
            except Exception as e:  # noqa: BLE001
                log("canary_error", fid, error=f"{type(e).__name__}: {e}"[:200])

        # A freed slot: start the oldest tested fix waiting on it (what ``_start_next_canary``
        # does in the sweep). Under ``-noreplay`` any open fix with a branch qualifies.
        if not slot_busy and self.repair_policy != "auto_after_replay":
            if self.canary_regardless_of_test:
                waiting = [f for f in fixes if f.get("status") in ("proposed", "test_passed", "test_failed", "test_inconclusive")
                           and f.get("branch") and not f.get("canary")]
            elif self.repair_policy == "auto_after_canary":
                waiting = [f for f in fixes if f.get("status") == "test_passed" and f.get("branch") and not f.get("canary")]
            else:
                waiting = []
            waiting.sort(key=lambda f: str(f.get("created_at") or ""))
            for f in waiting[:1]:
                fid = str(f["id"])
                try:
                    after = self.pro.start_canary(fid, pct=self.canary_pct, min_outcomes=self.canary_min_outcomes,
                                                  timeout_days=1)
                    calls += 1
                    self._canary_started_at[fid] = self.sessions_sealed
                    log("canary_start", fid, by="harness-sweep", status=after.get("status"),
                        test_status=(f.get("test") or {}).get("status") if isinstance(f.get("test"), dict) else f.get("status"))
                except ProApiError as e:
                    calls += 1
                    log("canary_start_refused", fid, error=f"{e.status}: {str(e.detail)[:200]}")

        # Shipped fixes: ask for a prevention re-check (the daily loop sweep's job).
        for f in fixes:
            if f.get("status") == "memory_shipped":
                fid = str(f["id"])
                try:
                    after = self.pro.recheck(fid)
                    calls += 1
                    log("recheck", fid, status=after.get("status"), prevention=(after.get("prevention") or {}) if isinstance(after.get("prevention"), dict) else None)
                except ProApiError as e:
                    calls += 1
                    log("recheck_error", fid, error=f"{e.status}: {str(e.detail)[:200]}")
        return calls

    # -- end of cell -------------------------------------------------------------------
    def report(self) -> dict[str, Any] | None:
        """The cell's fix lifecycle, verdicts, incidents and behaviours, for ``rsi_fixes.jsonl``."""
        if self._agent is None:
            return None
        from ..pro_client import ProApiError

        out: dict[str, Any] = {"agent_id": self._agent, "scenario": self._scenario_name, "policy": self.repair_policy,
                               "verdict_source": self.verdict_source, "judges_on": self.judges_on,
                               "canary": {"pct": self.canary_pct, "min_outcomes": self.canary_min_outcomes,
                                          "timeout_sessions": self.canary_timeout_sessions, "look_every": self.look_every},
                               "sessions_sealed": self.sessions_sealed, "ledger": self.ledger, "events": self.events}
        try:
            fixes = self.pro.list_fixes(self._agent)
            full = []
            for f in fixes:
                row = dict(f)
                try:
                    row["events"] = (self.pro.fix_events(str(f["id"]), limit=100) or {}).get("items")
                except ProApiError:
                    pass
                try:
                    row["tests"] = (self.pro.fix_tests(str(f["id"])) or {}).get("items")
                except ProApiError:
                    pass
                if f.get("status") in SHIPPED_STATUSES:
                    try:
                        row["prevention"] = self.pro.prevention(str(f["id"]))
                    except ProApiError:
                        pass
                full.append(row)
            out["fixes"] = full
        except Exception as e:  # noqa: BLE001
            out["fixes_error"] = f"{type(e).__name__}: {e}"[:200]
        for name, fn in (("verdicts", lambda: self.pro.verdicts(self._agent, limit=500)),
                         ("incidents", lambda: self.pro.incidents(self._agent)),
                         ("behaviors", lambda: self.pro.behaviors(self._agent)),
                         ("contract", lambda: self.pro.feedback_contract(self._agent)),
                         ("repairs_summary", lambda: self.pro.repairs_summary(self._agent))):
            try:
                out[name] = fn()
            except Exception as e:  # noqa: BLE001
                out[f"{name}_error"] = f"{type(e).__name__}: {e}"[:200]
        return out

    def close(self) -> None:
        try:
            self.pro.close()
        except Exception:  # noqa: BLE001
            pass
        super().close()


class SenseLabRsiNoCanaryArm(SenseLabRsiArm):
    name = "senselab-rsi-nocanary"
    repair_policy = "auto_after_replay"


class SenseLabRsiNoReplayArm(SenseLabRsiArm):
    name = "senselab-rsi-noreplay"
    repair_policy = "review"
    canary_regardless_of_test = True


class SenseLabRsiReviewArm(SenseLabRsiArm):
    name = "senselab-rsi-review"
    repair_policy = "review"


class SenseLabRsiMinimalArm(SenseLabRsiArm):
    name = "senselab-rsi-minimal"
    attempt_boundaries = False
    records_actions = False
    priors = False


class SenseLabRsiSelfReportArm(SenseLabRsiArm):
    name = "senselab-rsi-selfreport"
    verdict_source = "self-report"


class SenseLabRsiJudgesArm(SenseLabRsiArm):
    name = "senselab-rsi-judges"
    verdict_source = "judges"
    attempt_boundaries = False


__all__ = ["JUDGE_ID", "JUDGE_PROMPT", "SenseLabRsiArm", "SenseLabRsiJudgesArm", "SenseLabRsiMinimalArm",
           "SenseLabRsiNoCanaryArm", "SenseLabRsiNoReplayArm", "SenseLabRsiReviewArm", "SenseLabRsiSelfReportArm"]
