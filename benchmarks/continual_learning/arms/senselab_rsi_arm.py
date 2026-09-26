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
  the oldest fix in the sweep's queue waiting on a freed slot (passed, inconclusive, or never
  tested for want of a replayable case; the latter two at the product's 10 % cap — amendment
  (k)), ``POST /fixes/{id}/recheck`` for shipped fixes — plus a snapshot of the agent's fixes
  so the episode row carries the lifecycle. Nothing that decides (propose, gate, ``decide()``,
  ship, roll back) runs here.

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
    "text of that entry's value (copy it verbatim, up to one sentence) and whose span_id is "
    "null — the evidence is the memory entry, not a span; a span_id on this criterion is "
    "wrong. If no entry read supports the action, add 'followed-memory' with passed=true and "
    "quote the agent's own reasoning or the tool call instead. Do not grade tone or "
    "verbosity.\n\n"
    "Scenario notes: {rubric}\n\nTRACE:\n{{{{trace}}}}"
)
# ``JUDGE_PROMPT.format(rubric=...)`` yields the literal ``{{trace}}`` the judge store requires.
# The 'followed-memory' criterion is what lets the platform's lever classifier see the memory:
# a verbatim quote of an entry resolves to an ``entry`` evidence ref (amfs_eval/runner.py
# ``_resolve_evidence``), which ``classify_lever`` counts as knowledge. Without it every
# failing run cites only the tool span that carried the bad action and the classifier reads
# the failure as ``code`` (pilot 1, 2026-09-26: two template fixes, no branch, nothing to canary).
# ``span_id`` must be null on that criterion: ``locate()`` honours a span hint before it
# looks at the quote, so a verbatim entry quote with a span_id still resolves to the span
# (pilot 2: three followed-memory criteria, all resolved to tool spans, zero entry refs).

OPEN_STATUSES = ("proposed", "testing", "test_passed", "test_failed", "test_inconclusive", "canary")
SHIPPED_STATUSES = ("memory_shipped", "verified", "regressed")
#: The product's cap on a canary no offline test vouched for (``amfs_eval.repair.loop
#: .INCONCLUSIVE_CANARY_PCT_CAP``); the look schedule applies it as ``_canary_overrides`` does.
INCONCLUSIVE_CANARY_PCT_CAP = 10


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _canary_ready(f: dict[str, Any]) -> bool:
    """``FixStore.canary_queue``'s readiness, on the fix record as the API returns it: tested
    (passed or inconclusive), or ``proposed`` with a test behind it or with nothing to replay
    at all (no case set, no predicted fixes or regressions — an outcomes-only proposal whose
    canary is its only test)."""
    status = f.get("status")
    if status in ("test_passed", "test_inconclusive"):
        return True
    if status != "proposed":
        return False
    if f.get("tested_at"):
        return True
    manifest = f.get("manifest") if isinstance(f.get("manifest"), dict) else {}
    return (not f.get("case_set_id") and not manifest.get("predicted_fixes")
            and not manifest.get("predicted_regressions"))


def _replay_awaited(f: dict[str, Any]) -> bool:
    """Tier 2 comes before Tier 3: a fix whose customer replay is still requested is not
    ready for a canary (the canary would cancel the replay unread)."""
    rep = f.get("replay") if isinstance(f.get("replay"), dict) else None
    return bool(rep and rep.get("status") == "requested")


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
    replay = f.get("replay") if isinstance(f.get("replay"), dict) else None
    if replay:
        row["replay"] = {k: replay.get(k) for k in ("status", "total", "delivered", "repaired", "failed", "missing",
                                                    "requested_at", "decided_at", "gradings", "note")}
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


def _task_key(task_input: str) -> str:
    """How the platform matches a replayed trace to its case when ``case_id`` is absent
    (``md5(lower(btrim(task_input)))``); the same normalisation keys the episode index."""
    return str(task_input or "")[:2000].strip().lower()


class _RsiSession(_SenseLabSession):
    def end(self, outcome: Outcome, *, task_input: str, response_text: str, cited_keys: list[str]) -> None:
        super().end(outcome, task_input=task_input, response_text=response_text, cited_keys=cited_keys)
        trace = getattr(self.mem, "_last_trace", None)
        trace_id = getattr(trace, "id", None) if trace is not None else None
        self.arm.sessions_sealed += 1
        # Which episode a task came from, for the Tier 2 replay: the platform sends the
        # case's ``task_input`` (the prompt the trace was committed with), not the episode.
        self.arm.episode_by_task.setdefault(_task_key(task_input), self.episode)
        self.arm.ledger.append({"episode": self.episode, "replica": self.replica, "agent": self.agent_id,
                                "trace_id": str(trace_id) if trace_id else None, "sealed_at": time.time(),
                                "success": outcome.success, "severity": outcome.severity,
                                "canary": self.acct.notes.get("canary"), "memory_branch": self.acct.notes.get("memory_branch")})

    def record_action(self, tool: str, arguments: dict[str, Any], result: str, success: bool) -> None:
        if not self.arm.records_actions:
            return None
        return super().record_action(tool, arguments, result, success)


class _RsiReplaySession(_RsiSession):
    """A Tier 2 replay: the same session as an episode's, with memory checked out on the
    fix's branch and the grader's keys on the trace (``case_id``, ``fix_id``,
    ``replay_delivery_id``; the SDK stamps ``memory_branch`` for any branch but main). It is
    not a benchmark session: it is counted and ledgered apart, never as ``sessions_sealed``."""

    def __init__(self, arm: "SenseLabRsiArm", agent_id: str, episode: int, request: Any) -> None:
        super().__init__(arm, agent_id, episode)
        self.request = request
        self.mem.checkout(request.branch)
        try:
            self.mem.set_session_attributes({"case_id": request.case_id, "fix_id": request.fix_id,
                                             "replay_delivery_id": request.delivery_id, "replay": "tier2"})
        except Exception as e:  # noqa: BLE001
            self.acct.notes["session_attributes_error"] = str(e)[:120]

    def outcome_ref(self) -> str:
        return f"replay:{self.request.fix_id[:8]}:{self.request.case_id[:8]}"

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        # A replay tests the fix; its notes do not belong on the fix's branch (and the
        # platform refuses the write: the account's keys read a repair branch, they do
        # not write it). Counted, not written.
        self.acct.notes["replay_writes_dropped"] = int(self.acct.notes.get("replay_writes_dropped", 0)) + 1

    def end(self, outcome: Outcome, *, task_input: str, response_text: str, cited_keys: list[str]) -> None:
        _SenseLabSession.end(self, outcome, task_input=task_input, response_text=response_text, cited_keys=cited_keys)
        trace = getattr(self.mem, "_last_trace", None)
        trace_id = getattr(trace, "id", None) if trace is not None else None
        self.arm.replay_ledger.append({
            "fix_id": self.request.fix_id, "case_id": self.request.case_id, "delivery_id": self.request.delivery_id,
            "branch": self.request.branch, "episode": self.episode, "trace_id": str(trace_id) if trace_id else None,
            "sealed_at": time.time(), "success": outcome.success, "severity": outcome.severity,
            "commit_error": self.acct.notes.get("commit_error"),
        })


class _ReplayArmView:
    """What ``run_episode`` sees when it replays a case: the cell's arm, except that the
    session it opens reads the fix's branch and carries the case's keys."""

    def __init__(self, arm: "SenseLabRsiArm", request: Any) -> None:
        self._arm = arm
        self._request = request

    def __getattr__(self, name: str) -> Any:
        return getattr(self._arm, name)

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _RsiReplaySession(self._arm, agent_id, episode, self._request)


class SenseLabRsiArm(SenseLabArm):
    name = "senselab-rsi"
    auto_propose = True                   # the repair loop is the treatment: proposals on, explicitly
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
    # Pacing floor: seconds to hold the cell after each seal until the trace is graded
    # (0 = no wait). Set from pilot 3 (HYPOTHESIS_RSI.md, amendment (e)).
    verdict_wait_s = _int_env("CL_RSI_VERDICT_WAIT", 0)
    verdict_poll_s = _int_env("CL_RSI_VERDICT_POLL", 5)
    # Tier 2: serve the agent's replay endpoint from the harness (replay_hub.py) and register
    # it on the cell's agent, so a fix the Tier 1 stand-in cannot vouch for is tested by
    # re-running its failing cases here, with memory on the fix's branch.
    tier2 = os.environ.get("CL_RSI_TIER2", "0").strip().lower() in ("1", "true", "yes")
    replay_wait_hours = _int_env("CL_RSI_REPLAY_WAIT_HOURS", 1)

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
        # The veterans' identity (every replica maps onto it) and, in order of arrival,
        # every hosted identity the cell has opened — the joiners of the team-grows probe
        # (amendment (i)) have their own. Fixes, verdicts and incidents are per identity.
        self._agent: str | None = None
        self._agents: list[str] = []
        # Tier 2 state: the cell's scenario (the environment the replay runs against), the
        # model the cell runs on, which episode each committed task came from, and the record
        # of every replay run here.
        self._scenario = None
        self._llm = None
        self.episode_by_task: dict[str, int] = {}
        self.replay_ledger: list[dict[str, Any]] = []
        self._hub = None

    def configure(self, scenario) -> None:
        super().configure(scenario)
        self._rubric = getattr(scenario, "judge_rubric", "") or "none"
        self._scenario_name = getattr(scenario, "name", "")
        # Kept for Tier 2 only: a replay re-runs ``scenario.task(ep)`` through the same
        # ``scenario.step`` the episode used. The agent sees the prompt, as it did then; the
        # truth stays inside the environment.
        self._scenario = scenario
        # The cell's shared agent identity: every replica maps onto it.
        self._agent = self.saas_agent_id(scenario.agent_for(0))
        self._ready(self._agent)

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        s = _RsiSession(self, agent_id, episode)
        if self._agent is None:
            self._agent = s.agent_id
        # A joiner's first session opens its identity on the platform (judge, repair
        # settings, replay endpoint) the moment it arrives, as a new agent's would.
        self._ready(s.agent_id)
        return s

    def _ready(self, agent_id: str) -> None:
        if agent_id in self._agents_ready:
            return
        self._ready_agent(agent_id)
        if agent_id not in self._agents:
            self._agents.append(agent_id)

    def _list_fixes_all(self) -> tuple[list[dict[str, Any]], int]:
        """Every identity's fixes, and the number of calls made."""
        fixes: list[dict[str, Any]] = []
        for a in self._agents or ([self._agent] if self._agent else []):
            fixes.extend(self.pro.list_fixes(a))
        return fixes, max(1, len(self._agents))

    def outcome_for_commit(self, outcome: Outcome, session: _SenseLabSession) -> tuple[str, dict[str, Any]]:
        if self.verdict_source == "judges":
            # What an agent that reports "done" sends: success unless it gave up. The
            # environment's verdict never reaches memory; a judge has to find the failure.
            return ("failure" if outcome.escalated else "success"), {}
        extra: dict[str, Any] = {}
        if self.verdict_source == "oracle" and self.verified_by:
            extra["verified_by"] = self.verified_by
        return outcome.severity, extra

    # -- per-agent setup (once per identity) -------------------------------------------
    def _ready_agent(self, agent_id: str) -> None:
        from ..pro_client import ProApiError

        t0 = time.perf_counter()
        try:
            if self.judges_on:
                self.pro.ensure_judge(JUDGE_ID, agent_id, "Task failure (benchmark, frozen rubric)",
                                      JUDGE_PROMPT.format(rubric=self._rubric[:1500]), mode="continuous")
            fields: dict[str, Any] = dict(repair_policy=self.repair_policy, canary_pct=self.canary_pct,
                                          canary_min_outcomes=self.canary_min_outcomes, canary_timeout_days=1,
                                          auto_propose=self.auto_propose)
            if self.tier2:
                from ..replay_hub import ReplayHub

                self._hub = ReplayHub.get()
                self._hub.register(agent_id, self)
                fields.update(replay_webhook_url=self._hub.url, replay_webhook_secret=self._hub.secret,
                              replay_wait_hours=self.replay_wait_hours)
            else:
                # An agent id is fresh per cell, but be explicit: no endpoint, Tier 1 and the
                # canary are the only tests.
                fields["replay_webhook_url"] = ""
            settings = self.pro.set_repair_settings(agent_id, **fields)
            # Recorded here so the session's hold does not PUT a second time for this identity.
            self.auto_propose_state[agent_id] = ("on" if settings.get("auto_propose") is True
                                                 else "unsupported")
            contract = None
            try:
                contract = self.pro.feedback_contract(agent_id)
            except ProApiError:
                pass
            ping = None
            if self.tier2:
                try:
                    ping = self.pro.test_replay_webhook(agent_id)
                except ProApiError as e:
                    ping = {"ok": False, "error": f"{e.status}: {str(e.detail)[:200]}"}
            self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "kind": "agent_ready",
                                "agent_id": agent_id, "settings": {k: settings.get(k) for k in (
                                    "auto_propose", "repair_policy", "canary_pct", "canary_min_outcomes",
                                    "canary_timeout_days", "replay_fidelity_floor", "replay_webhook_url",
                                    "replay_wait_hours")},
                                "contract": contract, "replay_webhook_ping": ping,
                                "ms": round((time.perf_counter() - t0) * 1000, 1)})
        except ProApiError as e:
            self.events.append({"t": time.time(), "kind": "agent_ready_error", "agent_id": agent_id,
                                "error": f"{e.status}: {str(e.detail)[:300]}"})
            raise
        self._agents_ready.add(agent_id)

    # -- pacing floor ---------------------------------------------------------------------
    def _await_verdict(self, acct: ArmAccounting) -> None:
        """Hold the cell until the platform has graded the session just sealed (or
        ``verdict_wait_s`` has passed). The shared eval worker grades a trace 60–90 s after
        the seal at a customer's pace; a benchmark that seals faster than that measures the
        queue, not the loop. The wait converts grading lag into wall-clock, which is
        reported (``verdict_wait_ms``, ``verdict_lag_s``), and keeps time-to-repair in
        episodes a property of the loop (HYPOTHESIS_RSI.md, amendment (b) pacing floor)."""
        if not self.ledger or not self.ledger[-1].get("trace_id"):
            return
        trace_id = self.ledger[-1]["trace_id"]
        agent = self.ledger[-1].get("agent") or self._agent
        t0 = time.perf_counter()
        deadline = t0 + self.verdict_wait_s
        found = None
        while True:
            try:
                v = self.pro.verdicts(agent, limit=5, trace_id=trace_id)
                acct.ops += 1
                items = v.get("items") or v.get("verdicts") or []
                if items:
                    found = items[0]
                    break
            except Exception as e:  # noqa: BLE001
                acct.notes["verdict_wait_error"] = f"{type(e).__name__}: {e}"[:200]
                break
            if time.perf_counter() >= deadline:
                break
            time.sleep(self.verdict_poll_s)
        waited = time.perf_counter() - t0
        acct.notes["verdict_wait_ms"] = round(waited * 1000, 1)
        acct.notes["verdict_graded"] = found is not None
        if found is not None:
            acct.notes["verdict"] = found.get("verdict")
            sealed_at = self.ledger[-1].get("sealed_at")
            if sealed_at:
                acct.notes["verdict_lag_s"] = round(time.time() - sealed_at, 1)

    # -- the look schedule -----------------------------------------------------------
    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        if self._agent is None:
            return None
        self._llm = llm   # the cell's model; a Tier 2 replay runs on it
        acct = ArmAccounting()
        t0 = time.perf_counter()
        due = (self.sessions_sealed - self._sessions_at_look) >= self.look_every
        if self.verdict_wait_s > 0:
            self._await_verdict(acct)
        try:
            fixes, calls = self._list_fixes_all()
            acct.ops += calls
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
        # ``update``, not assignment: ``_await_verdict`` wrote the pacing-floor notes
        # (``verdict_wait_ms`` / ``verdict_graded`` / ``verdict_lag_s``) into this dict first.
        acct.notes.update({
            "look": bool(due), "sessions_sealed": self.sessions_sealed,
            "fixes": {s: sum(1 for r in snapshot if r["status"] == s) for s in sorted({r["status"] for r in snapshot})},
            "changed": changed[:8],
        })
        active = next((r for r in snapshot if r["status"] == "canary"), None)
        if active:
            acct.notes["canary_active"] = {"id": active["id"], **(active.get("canary") or {})}
        return acct

    # -- Tier 2: the agent's own replay ------------------------------------------------
    def replay(self, request: Any) -> dict[str, Any]:
        """Answer one replay request from the platform (replay_hub.py): re-run the episode the
        case came from — same task, same environment, same model — with memory on the fix's
        branch, and commit the outcome under the case's keys so the platform grades it. Runs
        on the hub's thread while the cell goes on; the replay is not a benchmark session.
        Returns a summary for the hub's delivery record."""
        from ..agent.loop import run_episode

        t0 = time.perf_counter()
        ep = self.episode_by_task.get(_task_key(request.task_input))
        if ep is None:
            # A case whose prompt this cell never committed (another replica's identity would
            # be the same agent, so this is a truncated or foreign task_input). Nothing to run;
            # the case stays ``missing`` and the replay expires on the platform's clock.
            self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "kind": "replay_unmatched",
                                "fix_id": request.fix_id, "case_id": request.case_id,
                                "task_input": str(request.task_input)[:160]})
            return {"ok": False, "code": "unmatched_task"}
        if self._scenario is None or self._llm is None:
            self.events.append({"t": time.time(), "kind": "replay_not_ready", "fix_id": request.fix_id,
                                "case_id": request.case_id, "episode": ep})
            return {"ok": False, "code": "cell_not_ready"}
        task = self._scenario.task(ep)
        self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "kind": "replay_start",
                            "fix_id": request.fix_id, "case_id": request.case_id, "branch": request.branch,
                            "episode": ep, "delivery_id": request.delivery_id})
        rec = run_episode(self._scenario, _ReplayArmView(self, request), self._llm, task,
                          seed=self._scenario.seed, retry_budget=config.STUDY.retry_budget)
        row = self.replay_ledger[-1] if self.replay_ledger and self.replay_ledger[-1].get("case_id") == request.case_id else None
        summary = {"ok": True, "episode": ep, "success": rec.success, "severity": rec.severity,
                   "attempts": rec.attempts, "escalated": rec.escalated, "answer": str(rec.final_answer)[:80],
                   "trace_id": row.get("trace_id") if row else None, "ms": round((time.perf_counter() - t0) * 1000, 1),
                   "error": rec.error}
        if row is not None:
            row.update({"attempts": rec.attempts, "escalated": rec.escalated, "answer": str(rec.final_answer)[:80],
                        "usage": rec.usage, "ms": summary["ms"], "error": rec.error})
        self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "kind": "replay_done",
                            "fix_id": request.fix_id, "case_id": request.case_id, **{k: v for k, v in summary.items() if k != "ok"}})
        return summary

    def _advance(self, fixes: list[dict[str, Any]], episode: int) -> int:
        """One agent's share of the hourly sweep, through the account's endpoints. Returns
        the number of calls made. Never raises."""
        from ..pro_client import ProApiError

        calls = 0

        def log(kind: str, fix_id: str, **data: Any) -> None:
            self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "episode": episode,
                                "kind": kind, "fix_id": fix_id, **data})

        # Tier 2: grade what the agent has replayed so far, as the hourly replay sweep would.
        # A pass makes the fix ``test_passed`` and applies the policy (the canary starts if the
        # slot is free); a fail makes it ``test_failed``; cases not back yet keep it awaited.
        for f in fixes:
            rep = f.get("replay") if isinstance(f.get("replay"), dict) else None
            if not rep or rep.get("status") != "requested":
                continue
            fid = str(f["id"])
            try:
                after = self.pro.grade_replay(fid)
                calls += 1
                r2 = after.get("replay") if isinstance(after.get("replay"), dict) else {}
                log("replay_grade", fid, status=after.get("status"), replay_status=r2.get("status"),
                    repaired=r2.get("repaired"), failed=r2.get("failed"), missing=r2.get("missing"),
                    total=r2.get("total"), note=str(r2.get("note") or "")[:200])
                if after.get("status") != f.get("status"):
                    f.update(after)   # the canary logic below sees the fix as it is now
            except ProApiError as e:
                calls += 1
                log("replay_grade_error", fid, error=f"{e.status}: {str(e.detail)[:200]}")
            except Exception as e:  # noqa: BLE001
                log("replay_grade_error", fid, error=f"{type(e).__name__}: {e}"[:200])

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

        # A freed slot: start the oldest fix in the sweep's queue (what ``_start_next_canary``
        # does hourly, through ``FixStore.canary_queue``). Amendment (k): the queue, not only
        # the passed fixes — an inconclusive fix, or one never tested because nothing was
        # replayable, runs the small canary *as* its test, at the product's cap for a change
        # no offline test vouched for (``_canary_overrides``). Before (k) the harness started
        # only ``test_passed`` fixes and the others waited for the platform's hourly sweep,
        # which on the pilots fired after the cell had ended. Under ``-noreplay`` any open
        # fix with a branch qualifies.
        if not slot_busy and self.repair_policy != "auto_after_replay":
            if self.canary_regardless_of_test:
                waiting = [f for f in fixes if f.get("status") in ("proposed", "test_passed", "test_failed", "test_inconclusive")
                           and f.get("branch") and not f.get("canary")]
            elif self.repair_policy == "auto_after_canary":
                waiting = [f for f in fixes if _canary_ready(f) and f.get("branch") and not f.get("canary")
                           and not _replay_awaited(f)]
            else:
                waiting = []
            waiting.sort(key=lambda f: (str(f.get("tested_at") or "9"), str(f.get("created_at") or "")))
            for f in waiting[:1]:
                fid = str(f["id"])
                pct = self.canary_pct if f.get("status") == "test_passed" else min(self.canary_pct, INCONCLUSIVE_CANARY_PCT_CAP)
                try:
                    after = self.pro.start_canary(fid, pct=pct, min_outcomes=self.canary_min_outcomes,
                                                  timeout_days=1)
                    calls += 1
                    self._canary_started_at[fid] = self.sessions_sealed
                    log("canary_start", fid, by="harness-sweep", status=after.get("status"), pct=pct,
                        from_status=f.get("status"),
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

        out: dict[str, Any] = {"agent_id": self._agent, "agents": list(self._agents),
                               "scenario": self._scenario_name, "policy": self.repair_policy,
                               "verdict_source": self.verdict_source, "judges_on": self.judges_on,
                               "canary": {"pct": self.canary_pct, "min_outcomes": self.canary_min_outcomes,
                                          "timeout_sessions": self.canary_timeout_sessions, "look_every": self.look_every,
                                          "inconclusive_pct_cap": INCONCLUSIVE_CANARY_PCT_CAP, "look_fills_queue": "k"},
                               "pacing": {"verdict_wait_s": self.verdict_wait_s, "verdict_poll_s": self.verdict_poll_s},
                               "tier2": {"enabled": self.tier2, "replay_wait_hours": self.replay_wait_hours,
                                         "hub": self._hub.report() if self._hub is not None else None,
                                         "replays": self.replay_ledger},
                               "sessions_sealed": self.sessions_sealed, "ledger": self.ledger, "events": self.events}
        try:
            fixes, _ = self._list_fixes_all()
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
            # Amendment (k) K1/K3: the canaries still running when the cell ended, with their
            # arms as the last evaluation left them — a fleet-1 cell's expected reading for
            # a 10 % canary, reported rather than hidden.
            out["canary_undecided_at_end"] = [
                {"id": str(f["id"]), "source": f.get("source"), "scoped": bool(f.get("judge_id") or f.get("behavior_id")),
                 "from_status": (f.get("test") or {}).get("status") if isinstance(f.get("test"), dict) else None,
                 **{k: (f.get("canary") or {}).get(k) for k in ("pct", "min_outcomes", "evaluations", "note")},
                 "canary_graded": ((f.get("canary") or {}).get("canary") or {}).get("graded"),
                 "control_graded": ((f.get("canary") or {}).get("control") or {}).get("graded")}
                for f in fixes if f.get("status") == "canary" and isinstance(f.get("canary"), dict)
            ]
        except Exception as e:  # noqa: BLE001
            out["fixes_error"] = f"{type(e).__name__}: {e}"[:200]
        def per_agent(agent: str) -> dict[str, Any]:
            got: dict[str, Any] = {}
            for name, fn in (("verdicts", lambda: self.pro.verdicts(agent, limit=500)),
                             ("incidents", lambda: self.pro.incidents(agent)),
                             ("behaviors", lambda: self.pro.behaviors(agent)),
                             ("contract", lambda: self.pro.feedback_contract(agent)),
                             ("repairs_summary", lambda: self.pro.repairs_summary(agent))):
                try:
                    got[name] = fn()
                except Exception as e:  # noqa: BLE001
                    got[f"{name}_error"] = f"{type(e).__name__}: {e}"[:200]
            return got

        # The veterans' identity keeps the report's top-level keys (what every reader of
        # ``rsi_fixes.jsonl`` expects); each joiner's record sits under ``peers``.
        out.update(per_agent(self._agent))
        peers = [a for a in self._agents if a != self._agent]
        if peers:
            out["peers"] = {a: per_agent(a) for a in peers}
        return out

    def close(self) -> None:
        if self._hub is not None:
            # A replay request for this cell after its end has nowhere to run: the hub answers
            # 404 and the platform's case stays missing until the replay expires.
            for a in self._agents or ([self._agent] if self._agent else []):
                self._hub.unregister(a)
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
    tier2 = False   # no replay of either tier: the canary is the only test


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
