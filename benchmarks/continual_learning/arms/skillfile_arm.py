"""Skill-file arm (WikiSkill-lite): experience distilled into a file the agent reads, not a
memory it searches.

Three layers, as in WikiSkill (Google, 2026):

* **raw** — every episode's record (task, actions, environment responses, verdict), append-only
  in ``raw.jsonl``. Also sealed as a SenseLab decision trace (``record_action`` +
  ``commit_outcome``; no memory reads or writes) so the arm's decisions are training data for
  the skill-file-teacher student exactly like every other arm's. ``CL_SKILLFILE_TRACES=0``
  turns that off.
* **wiki** — ``patterns.md``: a maintainer call every K episodes folds the raw records since
  the last checkpoint into the wiki (new patterns, failures and what worked instead, patterns
  the recent runs contradict marked outdated). The wiki persists whatever the gate decides.
* **skill** — ``SKILL.md``: a proposer call drafts the whole file from the wiki, the current
  skill and the impact log. The draft is **gated** on a validation replay: the same held-out
  tasks (regenerated instances of the last K episodes, different seed, same regime) are run
  once with the current skill and once with the candidate; the candidate is accepted only if
  it does at least as well and strictly better whenever the current skill is non-empty. A
  rejected skill is rolled back; ``skill-impact.md`` records every decision.

The skill reaches the model as an addendum to the system prompt (``prompt_addendum``);
``has_memory`` is False — no memory tools, nothing to search. The initial SKILL.md (v0) is
proposed from the seed knowledge every arm receives (minus the distractor notes, which go to
the wiki as ``seed.md``), so the file starts where the other arms' seeded stores start.

Every LLM call the arm makes — maintainer, proposer, both validation replays — is charged to
the arm (``ArmAccounting.extra_*``), the way ``pgvector-diy`` charges its consolidation. A
skill file is not free knowledge; that cost is part of the comparison.

Env: ``CL_SKILL_EVERY`` (checkpoint period, default 5), ``CL_SKILL_VALIDATE`` (validation tasks
per replay, default 4), ``CL_SKILL_MAINTAINER`` (model for maintainer/proposer when the cell's
own model is a single-shot student; default gpt-5.4), ``CL_SKILLFILE_TRACES`` (default 1).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .. import config
from .base import ArmAccounting, EpisodeSession, MemoryArm, MemoryHit, Outcome

SKILL_MAX_CHARS = 2400
WIKI_MAX_CHARS = 12000
_DISTRACTOR_KEY = re.compile(r"^note-\d{4}$")

MAINTAINER_PROMPT = """You maintain the knowledge wiki of an operations agent. Below are the current wiki
(patterns.md) and the raw records of the agent's most recent runs: the task, what the agent
tried, how the environment responded, and the final verdict.

Update the wiki. Add new patterns (which action resolves which kind of task, under which
conditions). Record failures and what worked instead. Where the recent runs contradict an
existing pattern, revise it or mark it OUTDATED and state the replacement — a pattern that
keeps failing is stale even if it used to work. Keep every pattern the runs still support.
Be factual and compact; no chatter. Stay under {max_chars} characters.

Output the full updated patterns.md as plain Markdown, no preamble.

== patterns.md (current) ==
{wiki}

== raw records since the last checkpoint ==
{raw}
"""

PROPOSER_PROMPT = """You write SKILL.md for an operations agent: a short procedural skill file the agent reads
at the start of every task, before it sees the task. Below are the wiki (patterns.md), the
current SKILL.md (may be empty) and the impact log of earlier skill versions (accepted or
rejected by validation, with their validation scores).

Write the full new SKILL.md: concrete decision rules ("if the request says X and the account
is Y, do Z"), most frequently applicable first, each with the conditions under which it holds
and known exceptions; then how to handle a request no rule covers, and what to do after a
failed attempt. Drop rules the wiki marks OUTDATED. Learn from the impact log: if a previous
version was rejected, do not resubmit its rules unchanged. Stay under {max_chars} characters.

Output only the file contents, plain Markdown, no preamble.

== patterns.md ==
{wiki}

== SKILL.md (current) ==
{skill}

== skill-impact.md ==
{impact}
"""

_TRACES = os.environ.get("CL_SKILLFILE_TRACES", "1") not in ("0", "false", "no")


def _slug(scope: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", scope.lower()).strip("-")


class _SkillSession(EpisodeSession):
    def __init__(self, arm: "SkillFileArm", agent_id: str, episode: int) -> None:
        super().__init__(arm, agent_id, episode)
        self.arm: SkillFileArm = arm
        self.actions: list[dict[str, Any]] = []
        self._mem = None
        self._task_input = ""

    def search(self, query: str, top_k: int) -> list[MemoryHit]:
        return []

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        return None

    def _memory(self):
        if self._mem is None:
            from amfs import AgentMemory

            from .senselab_arm import _ThrottledHttpAdapter

            self._mem = AgentMemory(agent_id=self.agent_id,
                                    adapter=_ThrottledHttpAdapter(base_url=config.AMFS_HTTP_URL,
                                                                  api_key=config.env("AMFS_API_KEY", required=True)))
            # The trace's only link to the cell (this arm reads no entries): the scope, as
            # ``attributes.entity_path``, which is what a Managed Model's entity_prefix matches.
            self._mem.set_session_attributes({"entity_path": self.arm.scope[:256], "bench_episode": self.episode,
                                              "skill_version": self.arm.version})
        return self._mem

    def record_action(self, tool: str, arguments: dict[str, Any], result: str, success: bool) -> None:
        self.actions.append({"tool": tool, "arguments": arguments, "result": result[:400], "success": success})
        if self.arm.validating or not _TRACES:
            return
        kwargs: dict[str, Any] = {}
        action = arguments.get("action") if isinstance(arguments, dict) else None
        key = f"{tool}:{action}" if isinstance(action, str) else None
        if key is not None and key in (self.candidate_actions or ()):
            kwargs["action_key"] = key
        try:
            self._memory().record_action(tool, arguments, result=result[:500], success=success, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.acct.notes["record_action_error"] = str(e)[:200]

    def end(self, outcome: Outcome, *, task_input: str, response_text: str, cited_keys: list[str]) -> None:
        if self.arm.validating:
            return  # the replay's verdict is read off run_episode's record, not recorded here
        self.arm.raw_append({"episode": self.episode, "agent_id": self.agent_id, "task": task_input,
                             "actions": self.actions, "success": outcome.success, "severity": outcome.severity,
                             "answer": response_text[:300], "skill_version": self.arm.version, "ts": time.time()})
        if not _TRACES:
            return
        from amfs_core.models import OutcomeType

        with self._timed():
            try:
                self._memory().commit_outcome(
                    f"{self.agent_id}-ep{self.episode}", OutcomeType(outcome.severity),
                    causal_entry_keys=None, task_input=task_input[:2000], response_text=response_text[:2000],
                    decision_summary=f"skill v{self.arm.version}: {outcome.summary}"[:500],
                )
                self.acct.trace_verifiable = True
            except Exception as e:  # noqa: BLE001
                self.acct.notes["commit_error"] = str(e)[:300]

    def close(self) -> None:
        if self._mem is not None:
            try:
                self._mem.close()
            except Exception:  # noqa: BLE001
                pass


class SkillFileArm(MemoryArm):
    name = "skillfile"
    has_memory = False
    learns_from_outcomes = True   # the gate reads the environment's verdicts (raw layer)

    every = int(os.environ.get("CL_SKILL_EVERY", "5"))
    n_validate = int(os.environ.get("CL_SKILL_VALIDATE", "4"))

    def open(self, scope: str) -> None:
        super().open(scope)
        self.dir = config.RESULTS_DIR / "skillfiles" / _slug(scope)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "versions").mkdir(exist_ok=True)
        self.wiki = ""
        self.skill = ""
        self.version = 0
        self.impact: list[dict[str, Any]] = []
        self.raw: list[dict[str, Any]] = []
        self._raw_checkpoint = 0
        self.validating = False
        self._candidate: str | None = None
        self._val_usage = ArmAccounting()   # LLM spend of validation replays, charged at the checkpoint
        self.seed_acct: ArmAccounting | None = None
        self._lock = threading.Lock()

    def __init__(self) -> None:
        self.scenario = None
        self.factory: Callable[[int], Any] | None = None
        self.llm = None
        self.retry_budget = config.STUDY.retry_budget

    def attach(self, scenario, factory: Callable[[int], Any], llm, *, retry_budget: int) -> None:
        """Called by the runner: the cell's scenario (for the change schedule), a factory that
        builds a same-shaped scenario for another seed (validation instances) and the cell's
        model (validation replays; maintainer/proposer when it is a frontier model)."""
        self.scenario = scenario
        self.factory = factory
        self.llm = llm
        self.retry_budget = retry_budget

    # ------------------------------------------------------------------ layers on disk
    def _write(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text)

    def raw_append(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.raw.append(rec)
            with (self.dir / "raw.jsonl").open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def _log(self, line: str) -> None:
        with (self.dir / "logs.md").open("a") as f:
            f.write(f"- {time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")

    # ------------------------------------------------------------------ the model used for maintenance
    def _maintainer(self):
        from ..agent.clients import get_llm

        if self.llm is not None and not getattr(self.llm, "decision_mode", False):
            return self.llm
        return get_llm(config.env("CL_SKILL_MAINTAINER") or "gpt-5.4")

    def _call(self, prompt: str, acct: ArmAccounting, *, max_tokens: int) -> str:
        text, usage = self._maintainer().complete_text(prompt, max_tokens=max_tokens)
        acct.extra_prompt_tokens += usage.prompt_tokens
        acct.extra_completion_tokens += usage.completion_tokens
        acct.extra_cost_usd += usage.cost_usd
        return text.strip()

    # ------------------------------------------------------------------ protocol
    def seed(self, entries, *, agent_id: str = "seed-agent") -> None:
        """Seed knowledge goes to the wiki; distractor notes to seed.md (the wiki's archive);
        SKILL.md v0 is proposed from the non-distractor part so the file starts where the other
        arms' seeded stores start."""
        core = [(k, t, c) for k, t, c in entries if not _DISTRACTOR_KEY.match(k)]
        noise = [(k, t, c) for k, t, c in entries if _DISTRACTOR_KEY.match(k)]
        self.wiki = "# patterns.md\n\n## Seed knowledge\n" + "\n".join(f"- ({k}) {t}" for k, t, _ in core) + "\n"
        self._write("patterns.md", self.wiki)
        self._write("seed.md", "\n".join(f"- ({k}) {t}" for k, t, _ in noise))
        self._write("skill-impact.md", "# skill-impact.md\n")
        acct = ArmAccounting()
        if core and self.llm is not None:
            try:
                draft = self._call(PROPOSER_PROMPT.format(max_chars=SKILL_MAX_CHARS, wiki=self.wiki, skill="(empty)",
                                                          impact="(none yet)"), acct, max_tokens=900)
                self._adopt(draft, note="v0 from seed knowledge (not validated)", acct=acct,
                            val={"candidate": None, "current": None})
            except Exception as e:  # noqa: BLE001
                self._log(f"v0 proposal failed: {e}")
        self.seed_acct = acct

    def prompt_addendum(self, agent_id: str) -> str | None:
        text = self._candidate if self.validating else self.skill
        if not text:
            return None
        return ("SKILL FILE (procedural knowledge distilled from your past runs here; treat it as your "
                "long-term knowledge and follow it when it applies):\n" + text)

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _SkillSession(self, agent_id, episode)

    def _adopt(self, text: str, *, note: str, acct: ArmAccounting, val: dict[str, Any]) -> None:
        self.version += 1
        self.skill = text[:SKILL_MAX_CHARS]
        self._write("SKILL.md", self.skill)
        self._write(f"versions/SKILL.v{self.version}.md", self.skill)
        self._impact(accepted=True, note=note, val=val)

    def _impact(self, *, accepted: bool, note: str, val: dict[str, Any]) -> None:
        row = {"version": self.version if accepted else f"rejected@v{self.version}", "accepted": accepted,
               "note": note, "val_candidate": val.get("candidate"), "val_current": val.get("current"),
               "episodes_seen": len(self.raw), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.impact.append(row)
        with (self.dir / "skill-impact.md").open("a") as f:
            f.write(f"- {'ACCEPTED' if accepted else 'REJECTED'} {row['version']}: {note} "
                    f"(validation candidate={row['val_candidate']} current={row['val_current']}, "
                    f"after {row['episodes_seen']} episodes)\n")

    def _validate(self, skill_text: str, episode: int) -> float | None:
        """Success rate of ``skill_text`` on regenerated instances of the last K episodes
        (another seed, same regime). Nothing here touches raw, the wiki or the trace store."""
        if self.factory is None or self.llm is None or self.n_validate <= 0:
            return None
        from ..agent.loop import run_episode

        val = self.factory(self.scenario.seed + 1000)
        idxs = list(range(max(0, episode - self.every + 1), episode + 1))[-self.n_validate:]
        self.validating, self._candidate = True, skill_text
        results: list[bool] = []
        try:
            for idx in idxs:
                rec = run_episode(val, self, self.llm, val.task(idx), seed=val.seed, retry_budget=self.retry_budget)
                results.append(bool(rec.success))
                self._val_usage.extra_prompt_tokens += rec.usage["prompt_tokens"]
                self._val_usage.extra_completion_tokens += rec.usage["completion_tokens"]
                self._val_usage.extra_cost_usd += rec.usage["cost_usd"]
        finally:
            self.validating, self._candidate = False, None
        return sum(results) / len(results) if results else None

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        if (episode + 1) % self.every != 0:
            return None
        if self.llm is None:
            self.llm = llm
        acct = ArmAccounting()
        t0 = time.perf_counter()
        new_raw = self.raw[self._raw_checkpoint:]
        self._raw_checkpoint = len(self.raw)
        if not new_raw:
            return None
        raw_text = "\n".join(
            f"- ep{r['episode']} [{'OK' if r['success'] else r['severity'].upper()}] task: {r['task'][:300]!r}\n"
            + "".join(f"    - {a['tool']}({json.dumps(a['arguments'])[:160]}) -> "
                      f"{'ok' if a['success'] else 'FAILED'}: {a['result'][:160]!r}\n" for a in r["actions"])
            for r in new_raw)
        # 1. wiki
        try:
            self.wiki = self._call(MAINTAINER_PROMPT.format(max_chars=WIKI_MAX_CHARS, wiki=self.wiki or "(empty)",
                                                            raw=raw_text), acct, max_tokens=2500)[:WIKI_MAX_CHARS]
            self._write("patterns.md", self.wiki)
        except Exception as e:  # noqa: BLE001
            self._log(f"ep{episode} maintainer failed: {e}")
        # 2. skill proposal
        try:
            impact = "\n".join(f"- {'accepted' if r['accepted'] else 'rejected'} {r['version']}: {r['note']} "
                               f"(candidate={r['val_candidate']} current={r['val_current']})" for r in self.impact[-8:])
            candidate = self._call(PROPOSER_PROMPT.format(max_chars=SKILL_MAX_CHARS, wiki=self.wiki,
                                                          skill=self.skill or "(empty)", impact=impact or "(none yet)"),
                                   acct, max_tokens=900)[:SKILL_MAX_CHARS]
        except Exception as e:  # noqa: BLE001
            self._log(f"ep{episode} proposer failed: {e}")
            candidate = ""
        accepted = False
        val: dict[str, Any] = {"candidate": None, "current": None}
        if candidate and candidate.strip() != self.skill.strip():
            # 3. gate: validation replay of the candidate and of the current skill on the same tasks
            v_cand = self._validate(candidate, episode)
            v_cur = self._validate(self.skill, episode)
            val = {"candidate": v_cand, "current": v_cur}
            if v_cand is not None and v_cur is not None:
                accepted = v_cand > v_cur or (v_cand == v_cur and (not self.skill or v_cand >= 1.0))
            else:
                accepted = True  # no validation possible (no factory): accept, and say so in the log
            if accepted:
                self._adopt(candidate, note=f"ep{episode} proposal", acct=acct, val=val)
            else:
                self._impact(accepted=False, note=f"ep{episode} proposal rolled back", val=val)
        # Validation replays run through run_episode on this arm with the cell's model, and the
        # v0 proposal ran at seed time; both are charged here so the arm's cost carries the gate.
        for extra in (self._val_usage, self.seed_acct):
            if extra is not None:
                acct.extra_prompt_tokens += extra.extra_prompt_tokens
                acct.extra_completion_tokens += extra.extra_completion_tokens
                acct.extra_cost_usd += extra.extra_cost_usd
        self._val_usage, self.seed_acct = ArmAccounting(), None
        acct.memory_ms = (time.perf_counter() - t0) * 1000
        acct.ops = 1
        acct.notes = {"skill_version": self.version, "accepted": accepted, "val_candidate": val["candidate"],
                      "val_current": val["current"], "raw_records": len(new_raw), "wiki_chars": len(self.wiki),
                      "skill_chars": len(self.skill)}
        self._log(f"ep{episode} checkpoint: wiki {len(self.wiki)}c, skill v{self.version} "
                  f"({'accepted' if accepted else 'kept'}), validation {val}")
        return acct

    def confidence_history(self, keys: list[str]) -> list[dict[str, Any]] | None:
        return None

    def close(self) -> None:
        self._write("skill-impact.json", json.dumps(self.impact, indent=1))
