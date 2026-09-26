"""SenseLab arm (production HTTP API), wired per its documented agent contract.

Per episode, from a FRESH connection (so the causal read set is this episode's alone):
  1. ``briefing(entity_path=scope, compact=True)``   — the lead digest with its hot context and
                                                       the outcome-derived sections (validated /
                                                       discredited / regime_shift)
  2. ``retrieve(query, include_avoid, adaptive_k,
                include_priors, candidate_actions, compact)``
                                                     — ranked recall with the evidence signal in
                                                       the score; discredited knowledge is
                                                       served as an explicit avoid list, not as
                                                       a candidate; after the hits, the action
                                                       priors (what was tried on similar tasks
                                                       here and how it went, what is untried)
                                                       and an act / explore / escalate
                                                       recommendation. ``compact`` keeps the
                                                       first two hits whole and the rest as
                                                       one-liners.
  3. ``write(...)``                                   — notes / heuristics with a type
  4. ``record_action(..., action_key=)``              — domain tool calls, sealed in the trace;
                                                       the terminal call carries its
                                                       ``<tool>:<action>`` key so the outcome is
                                                       credited to the action as well as to the
                                                       entries
  5. ``record_attempt(...)``  (local, no round trip)  — when a remembered approach failed and
                                                       the agent is about to try another, so
                                                       the failure lands on what that attempt
                                                       relied on
  6. ``commit_outcome(ref, type, causal_entry_keys, task_input, response_text)``
     — the environment's verdict, attributed to the memories the agent cited (or, if it
       cited none, to the top hit of each retrieve, which the SDK records automatically).
       Attempt boundaries travel inside the same request.

Variants:
  ``senselab``             the full protocol above
  ``senselab-episode``     steps 1-4 and 6 only: one outcome per task, attributed to the final
                           attempt's reads. Isolates what per-attempt credit assignment adds.
  ``senselab-nofeedback``  steps 1-4 only; step 6 never happens. The ablation that separates
                           "good retrieval" from "learning from outcomes".
  ``senselab-attempts``    legacy: per-attempt failures committed as separate outcomes (one
                           HTTP round trip each). Kept so the grid-v1 rows stay interpretable.
  ``senselab-nopriors``    the full protocol without action priors / recommendation / compact
                           payload (the grid-v2 wiring). The gap to ``senselab`` is what
                           action-level learning buys.
  ``senselab-lean``        the full protocol with the briefing cut to its evidence sections
                           (regime shift, discredited, validated, tried here, explore) — no
                           hot-context previews. The retrieve that follows returns the same
                           notes when they are relevant, so this measures what the previews
                           cost in tokens against what they add in accuracy.

Env: ``CL_SENSELAB_PRIORS=0`` turns priors off for every senselab arm (same as
``senselab-nopriors``); ``CL_SENSELAB_SINCE=0`` turns off incremental repeat briefings
(``since=<last briefing for this agent>``). On by default since the server's since-delta
keeps the standing warnings (discredited rows and losing actions) whatever their age, so
the delta is safe to act on and pays only for what changed.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any

from amfs import SITUATION_ATTRIBUTE, AgentMemory
from amfs_adapter_http import HttpAdapter
from amfs_core.actions import render_priors
from amfs_core.models import MemoryType, OutcomeType, RecallConfig

from .. import config
from .base import ArmAccounting, EpisodeSession, MemoryArm, MemoryHit, Outcome

#: Key of the entity's induced decision rules (``SenseLabInduceArm``). One key, versioned.
INDUCED_RULES_KEY = "decision-rules"


def fold_situation(value: Any) -> str:
    """A situation label as the harness sends it: lowercase, punctuation to spaces, one space
    between words, at most eight words. The server compares labels folded too (its
    ``_fold_situation``); folding here as well keeps what the analysis logs equal to what the
    record is keyed by, whichever server version answers."""
    text = "".join(ch if ch.isalnum() else " " for ch in str(value or "").casefold())
    words = text.split()
    return " ".join(words[:8])


#: The standing instruction every SenseLab arm with priors gives the model: how to use the
#: recommendation and the plan the search footer carries. It is the product's documented
#: integration contract (``amfs.Run``: act on ``act``, work the plan on a retry, label the kind
#: of task), applied identically to every SenseLab arm and disclosed as such. Grid v5
#: (2026-09-20) measured the model without it: on ``ci-fix`` it took the recommended action on
#: 41% of first attempts and retried an action already failed on this task 29% of the time.
PRIORS_CONTRACT = (
    "Memory search results end with a footer: what has been tried on this kind of task, how it "
    "went, a recommendation (act / explore / escalate / abstain) and an order to try actions in. "
    "Use it like this: on your first attempt, take the action the footer says to act on unless "
    "the task text plainly rules it out. If the footer says explore, the record shows the fix has "
    "changed or is unknown — take its first untried action, or your own best guess among the "
    "untried ones when it says to use your judgement. On a retry, take the next action in the "
    "footer's order that you have not tried on this task; never repeat an action that already "
    "failed on this task. Give every memory_search a `situation` label for the kind of task, "
    "reusing the labels already in use. If the footer says two kinds of task share your label, "
    "search once more with a finer label that names what distinguishes this task, and decide "
    "from the task's own details."
)

_KIND = {"fact": MemoryType.FACT, "belief": MemoryType.BELIEF, "experience": MemoryType.EXPERIENCE}
BRIEFING_MAX_CHARS = 2400


class _RateLimiter:
    """Process-global token bucket. The production API key is limited to 120 requests/min;
    the benchmark runs many SenseLab cells in parallel, so calls are paced below that
    limit here rather than discovered as 429s mid-episode. Override with AMFS_RPM."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / max(rpm, 1.0)
        self.lock = threading.Lock()
        self.next_at = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if wait > 0:
            time.sleep(wait)
            _TLS.waited = getattr(_TLS, "waited", 0.0) + wait


_TLS = threading.local()
_LIMITER = _RateLimiter(float(os.environ.get("AMFS_RPM", "100")))
_SEED_LOCK = threading.Lock()


class _PacedTimer:
    """Like the base timer, but benchmark-side pacing (limiter sleeps, 429 back-off) is
    subtracted from the arm's latency and reported separately as ``rate_limit_wait_ms``.
    Waiting on our own key's quota is a property of this benchmark, not of the product."""

    def __init__(self, acct) -> None:
        self.acct = acct

    def __enter__(self):
        self.t = time.perf_counter()
        self.w0 = getattr(_TLS, "waited", 0.0)
        return self

    def __exit__(self, *exc):
        waited = getattr(_TLS, "waited", 0.0) - self.w0
        self.acct.memory_ms += max(0.0, (time.perf_counter() - self.t) - waited) * 1000
        self.acct.ops += 1
        if waited:
            self.acct.notes["rate_limit_wait_ms"] = round(self.acct.notes.get("rate_limit_wait_ms", 0.0) + waited * 1000, 1)


class _ThrottledHttpAdapter(HttpAdapter):
    """HttpAdapter paced by the global limiter, with patient 429 handling (the SDK's own
    retry gives up after four quick attempts, which is not enough under sustained load).

    ``CL_SENSELAB_TIMEOUT`` (seconds, default the SDK's 30) sets the read timeout of the
    per-episode client. Under a capacity-capped API tier the 30 s default turned queueing
    delay into lost reflection notes; a longer timeout keeps the write and the latency is
    still measured in ``memory_ms``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "timeout" not in kwargs and os.environ.get("CL_SENSELAB_TIMEOUT"):
            import httpx

            kwargs["timeout"] = httpx.Timeout(float(os.environ["CL_SENSELAB_TIMEOUT"]), connect=10.0)
        super().__init__(*args, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        # The bound handle's identity headers (``X-AMFS-Agent-Id``, ``X-AMFS-Session``), as the
        # SDK's own ``_request`` merges them: they are what lets the hosted server attribute a
        # read to the agent and put the session in a repair canary's arm. An override that
        # dropped them would leave every benchmark session unrouted.
        identity = getattr(self, "_identity_headers", None)
        if identity:
            kwargs["headers"] = {**identity, **(kwargs.get("headers") or {})}
        for attempt in range(10):
            _LIMITER.acquire()
            resp = self._client.request(method, path, **kwargs)
            if resp.status_code == 429 and attempt < 9:
                wait = min(max(float(resp.headers.get("Retry-After", 1.0)), 0.5) * (1.5 ** attempt), 30.0)
                wait += random.uniform(0, 0.5)
                time.sleep(wait)
                _TLS.waited = getattr(_TLS, "waited", 0.0) + wait
                continue
            from amfs_adapter_http.adapter import _raise_with_detail
            _raise_with_detail(resp)
            # The routing layer's answer for this session: ``X-AMFS-Canary`` is
            # ``<fix_id>; arm=canary|control`` when a repair canary is running for the agent,
            # ``unrouted`` when one is running but the request named no session, absent when
            # none is. Kept on the bound handle (one per session) so the episode row can say
            # which arm the session drew without a second request.
            canary = resp.headers.get("X-AMFS-Canary")
            if canary is not None:
                self.last_canary = canary
            branch = resp.headers.get("X-AMFS-Memory-Branch")
            if branch is not None:
                self.last_branch = branch
            return resp.json()
        raise RuntimeError("unreachable")


def _ev(it: dict[str, Any]) -> str:
    st = it.get("evidence_status")
    w, l = int(it.get("success_count") or 0), int(it.get("failure_count") or 0)
    parts = []
    conf = it.get("confidence")
    if isinstance(conf, (int, float)):
        parts.append(f"confidence {conf:.2f}")
    if st and (st != "untested" or w or l):
        parts.append(f"{st}: {w} won / {l} failed" if (w or l) else st)
    return f" ({'; '.join(parts)})" if parts else ""


def _render_briefing(digests: list[Any], *, hot_context: bool = True) -> str | None:
    """Render what the agent reads at the top of a task. Evidence sections first — they are
    the part of a briefing that only a system with outcomes can produce. ``hot_context=False``
    leaves out the entry previews (``senselab-lean``)."""
    lines: list[str] = []
    for d in digests or []:
        summary = getattr(d, "summary", None)
        if summary is None and isinstance(d, dict):
            summary = d.get("summary")
        dtype = getattr(d, "digest_type", None) or (d.get("digest_type") if isinstance(d, dict) else "")
        if not isinstance(summary, dict):
            if summary:
                lines.append(f"- ({dtype}) {str(summary)[:400]}")
            continue
        rs = summary.get("regime_shift")
        if isinstance(rs, dict) and rs.get("suspected"):
            keys = ", ".join(str(e.get("key")) for e in (rs.get("entries") or [])[:6] if isinstance(e, dict))
            lines.append(f"WARNING — regime shift suspected: {rs.get('message', '')} Affected: {keys}.")
        disc = summary.get("discredited")
        if isinstance(disc, list) and disc:
            lines.append("Discredited by outcomes (do not act on these):")
            for it in disc[:6]:
                if not isinstance(it, dict):
                    continue
                rep = it.get("replaced_by") or []
                rep_s = f" Replaced by: {', '.join(map(str, rep[:3]))}." if rep else ""
                lines.append(f"  - [{it.get('key')}]{_ev(it)}: {str(it.get('value_preview') or '')[:120]}{rep_s}")
        val = summary.get("validated")
        if isinstance(val, list) and val:
            lines.append("Validated by outcomes: " + ", ".join(
                f"[{it.get('key')}] ({int(it.get('success_count') or 0)} won)" for it in val[:8] if isinstance(it, dict)))
        tried = summary.get("tried_here")
        if isinstance(tried, list) and tried:
            lines.append("Actions tried here: " + "; ".join(
                f"{it.get('action_key') or it.get('action')} {int(it.get('won') or 0)}/{int(it.get('n') or 0)}"
                for it in tried[:6] if isinstance(it, dict)))
        explore = summary.get("explore")
        if isinstance(explore, dict) and explore.get("suggested_action"):
            lines.append(f"Explore: try {explore['suggested_action']} first. {explore.get('why', '')}".rstrip())
        elif isinstance(explore, dict) and (explore.get("avoid") or explore.get("thin_evidence")):
            bits = []
            if explore.get("avoid"):
                bits.append("avoid " + ", ".join(map(str, explore["avoid"][:6])))
            if explore.get("thin_evidence"):
                bits.append("thin evidence on " + ", ".join(map(str, explore["thin_evidence"][:6])))
            lines.append("Explore: " + "; ".join(bits) + ".")
        for k in ("hot_context", "entries", "facts", "patterns", "key_facts", "risks"):
            if not hot_context:
                break
            items = summary.get(k)
            if isinstance(items, list) and items:
                for it in items[:8]:
                    if isinstance(it, dict):
                        key = it.get("key") or it.get("entity_path") or ""
                        val_ = it.get("value") or it.get("summary") or it.get("text") or ""
                        lines.append(f"- [{key}]{_ev(it)}: {str(val_)[:300]}")
                    else:
                        lines.append(f"- {str(it)[:300]}")
        text = summary.get("narrative") or summary.get("text") or summary.get("overview")
        if text:
            lines.append(f"- ({dtype}) {str(text)[:400]}")
    if not lines:
        return None
    out = "\n".join(lines)
    return out[:BRIEFING_MAX_CHARS]


class _SenseLabSession(EpisodeSession):
    def __init__(self, arm: "SenseLabArm", agent_id: str, episode: int) -> None:
        super().__init__(arm, agent_id, episode)
        self.arm: SenseLabArm = arm
        # Fleet as replicas (repair-loop study, HYPOTHESIS_RSI.md): the harness hands out one
        # identity per fleet member, but fixes, judges, incidents and the canary slot are
        # scoped per (account, agent_id), so a fleet that should share one repair loop runs
        # as ``n`` concurrent sessions of ONE agent. The member id stays on the trace as
        # ``replica`` so propagation across replicas is still measurable.
        self.replica = agent_id
        saas_agent = arm.saas_agent_id(agent_id)
        self.agent_id = saas_agent
        self.mem = AgentMemory(
            agent_id=saas_agent,
            adapter=_ThrottledHttpAdapter(base_url=config.AMFS_HTTP_URL,
                                          api_key=config.env("AMFS_API_KEY", required=True)),
        )
        # Stamp the cell's scope on the trace itself (``attributes.entity_path``), not only on
        # the causal entries: an episode that read nothing still belongs to the cell, and a
        # Managed Model scoped to this cell's traces by ``entity_prefix`` must see it too.
        # ``model`` / ``agent_version`` / ``runtime`` are the confounders the repair loop's
        # feedback contract requires on traces before it ships anything automatically; every
        # SenseLab arm stamps them so the control arms' traffic is provenance-identical.
        attrs: dict[str, Any] = {"entity_path": arm.scope[:256], "bench_episode": episode,
                                 "runtime": "cl-harness", "agent_version": config.HARNESS_VERSION[:64]}
        if getattr(arm, "cell_model", ""):
            attrs["model"] = str(arm.cell_model)[:64]
        if saas_agent != agent_id:
            attrs["replica"] = agent_id[:256]
        try:
            self.mem.set_session_attributes(attrs)
        except Exception as e:  # noqa: BLE001
            self.acct.notes["session_attributes_error"] = str(e)[:120]

        self._attempts_marked = 0
        self._tried_this_task: set[str] = set()
        self._footer: str | None = None
        # The kind of task the agent declared on its first search this episode, as the
        # product compares it (``fold_situation``). Sent on every retrieve of the episode and
        # stamped on the trace, so the situation's own outcome record is what the priors
        # read and what the repair loop groups failing runs by.
        self.situation: str | None = None
        # Set when the server's recommendation said the declared label pools two kinds of
        # task; lets a finer label replace it (see ``_declare_situation``).
        self._situation_pooled = False
        # Entries the briefing already named as discredited. A retrieve's avoid list names
        # them again; those rows are rendered as a reference to the briefing rather than
        # carried twice in every call of the episode.
        self._briefed_discredited: set[str] = set()

    def _timed(self):
        return _PacedTimer(self.acct)

    def briefing(self) -> str | None:
        since = self.arm.last_briefing_at(self.agent_id) if self.arm.briefing_since else None
        with self._timed():
            try:
                kwargs: dict[str, Any] = {}
                if since is not None:
                    kwargs["since"] = since
                digests = self.mem.briefing(entity_path=self.arm.scope, limit=8,
                                            compact=self.arm.compact_briefing, **kwargs)
            except Exception as e:  # noqa: BLE001
                self.acct.notes["briefing_error"] = str(e)[:200]
                return None
        if self.arm.briefing_since:
            self.arm.mark_briefed(self.agent_id)
            if since is not None:
                self.acct.notes["briefing_since"] = True
        # Hard cell-isolation guard. The server's briefing is agent-centric and may include
        # digests for other entities the same agent identity touched; keep only this cell's
        # entity digest and this cell's own agent brief. Dropped digests are counted so the
        # leak channel is measurable.
        kept, dropped = [], 0
        for d in digests or []:
            sc = getattr(d, "scope", None) or (d.get("scope") if isinstance(d, dict) else None)
            if sc in (self.arm.scope, self.agent_id) or (sc or "").startswith(self.arm.scope + "/"):
                kept.append(d)
            else:
                dropped += 1
        if dropped:
            self.acct.notes["briefing_digests_dropped"] = self.acct.notes.get("briefing_digests_dropped", 0) + dropped
        for d in kept:
            summary = getattr(d, "summary", None)
            if summary is None and isinstance(d, dict):
                summary = d.get("summary")
            disc = summary.get("discredited") if isinstance(summary, dict) else None
            for it in (disc or [])[:6]:
                if isinstance(it, dict) and it.get("key"):
                    self._briefed_discredited.add(str(it["key"]))
        text = _render_briefing(kept, hot_context=self.arm.briefing_hot_context)
        rules = self._induced_rules() if getattr(self.arm, "induces", False) else None
        if rules:
            text = rules + ("\n\n" + text if text else "")
        if text:
            self.acct.retrieved_bytes += len(text)
        return text

    def _induced_rules(self) -> str | None:
        """The entity's induced decision rules (``SenseLabInduceArm``), read by key the way
        an agent recalls a runbook it knows exists. The read is in the trace, so the
        episode's outcome lands on the rules entry and its evidence status is shown."""
        with self._timed():
            try:
                e = self.mem.read(self.arm.scope, INDUCED_RULES_KEY)
            except Exception as ex:  # noqa: BLE001
                self.acct.notes["rules_read_error"] = str(ex)[:120]
                return None
        if e is None:
            return None
        status = getattr(e, "evidence_status", None) or "untested"
        s, f = int(getattr(e, "success_count", 0) or 0), int(getattr(e, "failure_count", 0) or 0)
        self.read_keys.append(e.key)
        self.acct.notes["rules_version"] = int(getattr(e, "version", 0) or 0)
        return (f"Decision rules induced from this entity's outcome record (v{getattr(e, 'version', '?')}; "
                f"evidence: {status}, {s} succeeded / {f} failed on tasks that read them):\n{e.value}")

    def search(self, query: str, top_k: int, *, situation: str | None = None) -> list[MemoryHit]:
        cfg = RecallConfig(include_avoid=self.arm.evidence_aware, adaptive_k=self.arm.evidence_aware)
        extra: dict[str, Any] = {}
        cands: list[str] | None = None
        # Priors and the recommendation are asked for only when the task ends in a choice
        # among a fixed set of actions (the terminal tool has an ``action`` enum) — the
        # product's documented use. Grid v5 (2026-09-20) measured what asking anyway costs:
        # on concierge, whose terminal tool proposes an option with no enum, the server still
        # returned ``mode: act`` on 92% of episodes (top hit validated), and GPT-5.5 followed
        # the top hit over the task's own constraints — 42% first-attempt failures against 6%
        # for the same protocol without priors. Scenarios with an enum are unchanged.
        if self.arm.priors and self.candidate_actions:
            # On a retry the candidates are the actions not yet tried on this task: the
            # actions already tried in this episode are known to fail here, and ``explore``
            # picks one untried candidate per agent, so without this it would name the
            # same one on every attempt. What a caller with a fixed action set would do.
            cands = [c for c in self.candidate_actions if c.split(":", 1)[-1] not in self._tried_this_task]
            cands = cands or list(self.candidate_actions)
            # ``abstain``: when the record cannot say, the server says so ("no outcome for
            # this situation yet", "no hit is validated") instead of sending nothing — the
            # documented ``Run.begin`` call. Grid v5 (2026-09-20) measured the silence: 17%
            # of searched attempts came back with no recommendation at all.
            extra = {"include_priors": True, "candidate_actions": cands, "compact": True, "abstain": True}
            label = self._declare_situation(situation)
            if label:
                extra["situation"] = label
        self._footer, self.last_recommendation = None, None
        with self._timed():
            rows = self.mem.retrieve(query, entity_path=self.arm.scope,
                                     min_confidence=config.STUDY.min_confidence_gate, limit=top_k,
                                     recall_config=cfg, **extra)
        if self.arm.priors:
            meta = self.mem.last_priors or {}
            rec = meta.get("recommendation")
            priors = meta.get("priors")
            # The order may only name what the caller can still take: a retry must not be
            # told to try the action that just failed because the record has it as a winner.
            self._footer = render_priors(priors, rec, candidate_actions=cands) or None
            if isinstance(rec, dict) and rec.get("mode"):
                self.last_recommendation = {"mode": rec.get("mode"), "suggested_action": rec.get("suggested_action"),
                                            "regime_shift": bool(meta.get("regime_shift"))}
                if rec.get("plan"):
                    self.last_recommendation["plan"] = [str(a) for a in rec["plan"]][:8]
                if rec.get("untried") is not None:
                    self.last_recommendation["untried"] = len(rec["untried"])
                if rec.get("sweep"):
                    self.last_recommendation["sweep"] = True
                record = (priors or {}).get("situation_record") if isinstance(priors, dict) else None
                if isinstance(record, dict):
                    self.last_recommendation["situation_outcomes"] = int(record.get("outcomes") or 0)
                if self.situation:
                    self.last_recommendation["situation"] = self.situation
                if rec.get("pooled") and self.situation:
                    # The server read two kinds of task under this label. Remembered on the
                    # arm so the label is no longer offered for reuse, and on the session so
                    # a finer label may replace it (``_declare_situation``).
                    self.last_recommendation["pooled"] = True
                    self._situation_pooled = True
                    self.arm.note_pooled(self.situation)
                if rec.get("mixed"):
                    self.last_recommendation["mixed"] = True
                self.acct.notes[f"rec_{rec['mode']}"] = self.acct.notes.get(f"rec_{rec['mode']}", 0) + 1
            elif extra:
                self.acct.notes["rec_none"] = self.acct.notes.get("rec_none", 0) + 1
        hits = []
        for r in rows:
            e = r.entry
            avoid = bool((r.breakdown or {}).get("_avoid"))
            text = str(e.value)
            if avoid and e.key in self._briefed_discredited:
                text = "(discredited — see briefing above)"
            hits.append(MemoryHit(
                key=e.key, text=text, score=float(r.score), confidence=float(e.confidence),
                evidence_status=getattr(e, "evidence_status", None) or "untested",
                success_count=int(getattr(e, "success_count", 0) or 0),
                failure_count=int(getattr(e, "failure_count", 0) or 0), avoid=avoid))
        kept = [h for h in hits if not h.avoid]
        self.acct.retrieved_bytes += sum(len(h.text) for h in hits)
        if any(h.avoid for h in hits):
            self.acct.notes["avoid_served"] = self.acct.notes.get("avoid_served", 0) + sum(h.avoid for h in hits)
        self.read_keys.extend(h.key for h in kept[:1])
        return hits

    def _declare_situation(self, situation: str | None) -> str | None:
        """The label this episode's retrieves carry, or ``None`` when the arm does not declare
        one. The first label of the episode sticks: a retry that rephrases the kind of task
        must not open a second record for the same ticket. The label goes on the trace as
        the SDK's ``Run.begin`` puts it (``attributes.situation``), once."""
        if not self.arm.declares_situation:
            return None
        label = fold_situation(situation)
        if self.situation:
            # Sticky — except when the server has said this episode's label pools two kinds
            # of task (``recommendation.pooled``) and the agent now offers a different one:
            # the finer label is the fix the server asked for, and the outcome must land on
            # it, not on the label that pooled. Pilot 5a (2026-09-26): both ci-fix audit
            # classes were declared ``audit failure`` and the record alternated all run.
            if not (self._situation_pooled and label and label != self.situation):
                return self.situation
            self.acct.notes["situation_refined"] = self.acct.notes.get("situation_refined", 0) + 1
        if not label:
            return None
        self.situation = label
        self._situation_pooled = False
        self.arm.note_situation(label)
        try:
            self.mem.set_session_attributes({SITUATION_ATTRIBUTE: label[:200]})
        except Exception as e:  # noqa: BLE001
            self.acct.notes["situation_attribute_error"] = str(e)[:120]
        return label

    def search_footer(self) -> str | None:
        return self._footer

    def write(self, key: str, text: str, *, confidence: float = 0.7, kind: str = "experience") -> None:
        with self._timed():
            self.mem.write(self.arm.scope, key, text, confidence=confidence,
                           memory_type=_KIND.get(kind, MemoryType.EXPERIENCE))

    def record_action(self, tool: str, arguments: dict[str, Any], result: str, success: bool) -> None:
        # Local: the SDK buffers actions and ships them as ``tool_calls`` on the commit.
        # Not timed and not an op — there is no round-trip to count.
        kwargs: dict[str, Any] = {}
        action = arguments.get("action") if isinstance(arguments, dict) else None
        key = f"{tool}:{action}" if isinstance(action, str) else None
        if self.arm.priors and key is not None and key in (self.candidate_actions or ()):
            # The terminal call: name the action so the outcome is credited to it. Same key
            # the SDK would derive; explicit so a schema change cannot silently unname it.
            kwargs["action_key"] = key
        try:
            self.mem.record_action(tool, arguments, result=result[:500], success=success, **kwargs)
        except Exception as e:  # noqa: BLE001
            self.acct.notes["record_action_error"] = str(e)[:200]

    def _blame(self, cited_keys: list[str]) -> list[str] | None:
        """The entries an outcome is attributed to.

        What the agent cited (``used_memory_keys``) when it cited anything; otherwise the
        top hit of each retrieve since the last attempt boundary — the entry it most
        plausibly acted on. Never the whole read window: a retrieve returns 5-7 entries and
        the agent follows one, so blaming all of them for a failed attempt discredits
        correct lessons that merely shared the context. ``None`` when nothing identifies.
        """
        boundary = getattr(self, "_boundary", 0)
        keys = list(cited_keys) or list(dict.fromkeys(self.read_keys[boundary:]))
        return [f"{self.arm.scope}/{k}" for k in keys] or None

    def outcome_ref(self) -> str:
        """The reference the session's terminal outcome is committed under."""
        return f"{self.agent_id}-ep{self.episode}"

    def end(self, outcome: Outcome, *, task_input: str, response_text: str,
            cited_keys: list[str]) -> None:
        if not self.arm.learns_from_outcomes:
            return
        keys = self._blame(cited_keys)
        severity, extra = self.arm.outcome_for_commit(outcome, self)
        # The outcome is about this cell's entity whether or not the agent cited an entry
        # (an episode that read nothing still happened here — without the path its actions
        # never reach the entity's priors), and it is an outcome of the situation the agent
        # declared: the label on the record is what partitions the situation's own record
        # (``Run.complete`` passes both the same way).
        extra.setdefault("entity_path", self.arm.scope)
        if self.situation:
            extra.setdefault("situation", self.situation)
        with self._timed():
            try:
                affected = self.mem.commit_outcome(
                    self.outcome_ref(), OutcomeType(severity),
                    causal_entry_keys=keys, task_input=task_input[:2000],
                    response_text=response_text[:2000], decision_summary=outcome.summary[:500],
                    **extra,
                )
                self.acct.trace_verifiable = True
                self.acct.notes["affected_entries"] = len(affected)
                if self._attempts_marked:
                    self.acct.notes["attempts_in_trace"] = self._attempts_marked
                if severity != outcome.severity:
                    self.acct.notes["committed_severity"] = severity
            except Exception as e:  # noqa: BLE001
                self.acct.notes["commit_error"] = str(e)[:300]
        self._note_routing()

    def _note_routing(self) -> None:
        """Which canary arm the session's requests drew, from the server's response headers."""
        adapter = getattr(self.mem, "_adapter", None)
        canary = getattr(adapter, "last_canary", None)
        if canary:
            self.acct.notes["canary"] = str(canary)[:120]
        branch = getattr(adapter, "last_branch", None)
        if branch and branch != "main":
            self.acct.notes["memory_branch"] = str(branch)[:120]

    def attempt_failed(self, attempt: int, severity: str, cited_keys: list[str], answer: str) -> None:
        if answer:
            self._tried_this_task.add(str(answer))
        if self.arm.attempt_boundaries and self.arm.learns_from_outcomes:
            # Local bookkeeping only: the boundary and the entries it blames leave with the
            # terminal commit_outcome, inside the same trace. Zero extra round trips.
            keys = self._blame(cited_keys)
            if keys is None:
                # Nothing identifies what this attempt relied on: no search since the last
                # boundary and no citation. An attempt that names no entry teaches nothing,
                # and blaming the whole read window would charge entries the agent never
                # followed (the guilt-by-association that grid v2 measured: 25% of versions
                # discredited, pre-change first-attempt success below the no-feedback arm).
                self.acct.notes["attempt_unattributed"] = self.acct.notes.get("attempt_unattributed", 0) + 1
                return
            try:
                self.mem.record_attempt(outcome_type=OutcomeType(severity), causal_entry_keys=keys,
                                        summary=f"attempt {attempt} failed; answer={answer}"[:300])
                self._attempts_marked += 1
                self._boundary = len(self.read_keys)
            except Exception as e:  # noqa: BLE001
                self.acct.notes["record_attempt_error"] = str(e)[:200]
            return
        if not getattr(self.arm, "per_attempt_outcomes", False) or not cited_keys:
            return
        keys = [f"{self.arm.scope}/{k}" for k in cited_keys]
        with self._timed():
            try:
                affected = self.mem.commit_outcome(
                    f"{self.agent_id}-ep{self.episode}-attempt{attempt}", OutcomeType(severity),
                    causal_entry_keys=keys, decision_summary=f"attempt {attempt} failed; answer={answer}"[:500],
                )
                self.acct.notes["attempt_outcomes"] = self.acct.notes.get("attempt_outcomes", 0) + 1
                self.acct.notes["attempt_affected"] = self.acct.notes.get("attempt_affected", 0) + len(affected)
            except Exception as e:  # noqa: BLE001
                self.acct.notes["attempt_commit_error"] = str(e)[:300]

    def close(self) -> None:
        try:
            self.mem.close()
        except Exception:  # noqa: BLE001
            pass


class SenseLabArm(MemoryArm):
    """The full documented protocol: compact briefing with evidence sections, evidence-aware
    retrieve with an avoid list, attempt boundaries, one outcome per task."""

    name = "senselab"
    learns_from_outcomes = True
    compact_briefing = True      # briefing(compact=True): lead digest + evidence sections
    evidence_aware = True        # retrieve(include_avoid=True, adaptive_k=True)
    attempt_boundaries = True    # record_attempt on each failed non-final attempt
    # retrieve(include_priors=True, candidate_actions=<terminal enum>, compact=True) and
    # action_key on the terminal record_action. CL_SENSELAB_PRIORS=0 turns it off.
    priors = os.environ.get("CL_SENSELAB_PRIORS", "1") not in ("0", "false", "no")
    # retrieve(situation=<the agent's label for the kind of task>) — the priors are then the
    # situation's own outcome record (one loss on an unsolved task turns a winner; a first win
    # is acted on; the untried are swept in the order the entity's record elsewhere gives)
    # instead of the embedding neighbourhood, where a class's losses pool with its
    # neighbours' wins. CL_SENSELAB_SITUATION=0 turns it off (the ``-nosituation`` ablation).
    situations = os.environ.get("CL_SENSELAB_SITUATION", "1") not in ("0", "false", "no")
    # The standing instruction on how to use the footer (``PRIORS_CONTRACT``).
    # CL_SENSELAB_CONTRACT=0 turns it off (the ``-nocontract`` ablation).
    contract = os.environ.get("CL_SENSELAB_CONTRACT", "1") not in ("0", "false", "no")
    # briefing(since=<last briefing this agent received>) — see the module docstring.
    briefing_since = os.environ.get("CL_SENSELAB_SINCE", "1") not in ("0", "false", "no")
    # Whether the rendered briefing carries the hot-context previews (``senselab-lean`` drops them).
    briefing_hot_context = True

    # Fleet members share one SaaS agent identity (see ``_SenseLabSession``). Off by default
    # (the v7 protocol: one identity per member); ``CL_FLEET_REPLICAS=1`` turns it on for every
    # SenseLab arm of a run so control and repair arms are compared like for like; the
    # repair-loop arms force it.
    fleet_replicas = os.environ.get("CL_FLEET_REPLICAS", "0") in ("1", "true", "yes")
    # ``verified_by`` stamped on every committed outcome: the benchmark's verdict is the
    # environment's, not the agent's, and the repair loop's canary promotion requires
    # outcomes verified from outside the agent. ``None`` = omitted (the self-report condition).
    verified_by: str | None = "benchmark-oracle"
    # Set by the runner before ``open``: the agent model of the cell, stamped on traces.
    cell_model: str = ""

    def open(self, scope: str) -> None:
        super().open(scope)
        self._briefed_at: dict[str, Any] = {}
        # label -> episodes that declared it, this cell. Shown to the agent on the search
        # tool so the labels converge; the server keeps the record, this is the taxonomy.
        self._situations: dict[str, int] = {}
        self._situations_lock = threading.Lock()
        self._pooled_situations: set[str] = set()

    def configure(self, scenario) -> None:
        # The cell's agent base (the runner has already suffixed it per cell attempt, e.g.
        # ``support-agent-388584``); fleet members are ``<base>-<n>``.
        self._agent_base = getattr(scenario, "agent_base", "")
        # The situation record is asked for where the priors are: a task that ends in a
        # choice among a fixed set of actions. Free-form scenarios get no label parameter.
        from ..agent.loop import candidate_actions

        self.declares_situation = bool(self.priors and self.situations and candidate_actions(scenario))

    def note_situation(self, label: str) -> None:
        with getattr(self, "_situations_lock", threading.Lock()):
            labels = getattr(self, "_situations", None)
            if labels is not None:
                labels[label] = labels.get(label, 0) + 1

    def note_pooled(self, label: str) -> None:
        """The server said *label* pools two kinds of task with different fixes. It is no
        longer offered for reuse; the tool description names it as too coarse instead."""
        with getattr(self, "_situations_lock", threading.Lock()):
            pooled = getattr(self, "_pooled_situations", None)
            if pooled is not None:
                pooled.add(label)

    def situation_labels(self) -> list[str]:
        labels = getattr(self, "_situations", None) or {}
        pooled = getattr(self, "_pooled_situations", None) or set()
        return [k for k, _ in sorted(labels.items(), key=lambda kv: (-kv[1], kv[0])) if k not in pooled]

    def pooled_situation_labels(self) -> list[str]:
        return sorted(getattr(self, "_pooled_situations", None) or set())

    def prompt_addendum(self, agent_id: str) -> str | None:
        return PRIORS_CONTRACT if (self.priors and self.contract) else None

    def saas_agent_id(self, member_id: str) -> str:
        """The identity a fleet member acts as on the SaaS: its own, or — with fleet replicas —
        the cell's shared one (the scenario's ``agent_base``). Only the fleet member suffix is
        stripped, relative to the base: the runner's per-cell suffix is numeric too, and
        stripping any trailing ``-<digits>`` collapsed every cell of a family onto one hosted
        agent (shared behaviours, incident baseline, fixes and canary slot). Found in the RSI
        pilot, 2026-09-26."""
        if not self.fleet_replicas:
            return member_id
        base = getattr(self, "_agent_base", "") or ""
        if base and member_id.startswith(base + "-") and member_id[len(base) + 1:].isdigit():
            return base
        return member_id

    def outcome_for_commit(self, outcome: Outcome, session: "_SenseLabSession") -> tuple[str, dict[str, Any]]:
        """What ``commit_outcome`` sends: the severity and the extra keyword arguments. The
        default is the environment's verdict, verified by the benchmark oracle."""
        extra: dict[str, Any] = {}
        if self.verified_by:
            extra["verified_by"] = self.verified_by
        return outcome.severity, extra

    def last_briefing_at(self, agent_id: str):
        return getattr(self, "_briefed_at", {}).get(agent_id)

    def mark_briefed(self, agent_id: str) -> None:
        from datetime import datetime, timezone

        getattr(self, "_briefed_at", {})[agent_id] = datetime.now(timezone.utc)

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _SenseLabSession(self, agent_id, episode)

    def confidence_history(self, keys: list[str]) -> list[dict[str, Any]] | None:
        out: list[dict[str, Any]] = []
        mem = AgentMemory(agent_id="cl-analysis",
                          adapter=_ThrottledHttpAdapter(base_url=config.AMFS_HTTP_URL,
                                                        api_key=config.env("AMFS_API_KEY", required=True)))
        try:
            for key in list(dict.fromkeys(keys))[:40]:
                try:
                    versions = mem.history(self.scope, key)
                except Exception as e:  # noqa: BLE001
                    out.append({"key": key, "error": str(e)[:120]})
                    continue
                for v in versions:
                    out.append({
                        "key": key, "version": getattr(v, "version", None),
                        "confidence": round(float(getattr(v, "confidence", 0.0)), 4),
                        "evidence_status": getattr(v, "evidence_status", None),
                        "success_count": getattr(v, "success_count", 0),
                        "failure_count": getattr(v, "failure_count", 0),
                        "last_outcome": getattr(v, "last_outcome", None),
                        "written_at": str(getattr(getattr(v, "provenance", None), "written_at", "")),
                    })
        finally:
            try:
                mem.close()
            except Exception:  # noqa: BLE001
                pass
        return out

    def seed(self, entries, *, agent_id: str = "seed-agent") -> None:
        """Pre-load the store with the batch-commit endpoint (``POST /api/v1/commits``, the
        documented bulk path: one server-side transaction per chunk instead of one request
        per entry, which matters under the 120 RPM key limit). Seeding is a one-off before
        episode 1 and is timed by the runner, not charged to episodes."""
        import httpx

        adapter = _ThrottledHttpAdapter(base_url=config.AMFS_HTTP_URL,
                                        api_key=config.env("AMFS_API_KEY", required=True),
                                        timeout=httpx.Timeout(180.0, connect=10.0))
        # One cell seeds at a time: a batch commit embeds every entry server-side, and a
        # dozen cells doing that at once pushed single requests past the read timeout.
        with _SEED_LOCK:
            for i in range(0, len(entries), 20):
                chunk = entries[i:i + 20]
                for attempt in range(3):
                    try:
                        adapter.commit_batch(
                            [{"entity_path": self.scope, "key": key, "value": text, "confidence": conf,
                              "memory_type": MemoryType.FACT.value} for key, text, conf in chunk],
                            message=f"benchmark seed {i // 20 + 1}", agent_id=agent_id,
                        )
                        break
                    except (httpx.TimeoutException, httpx.TransportError):
                        if attempt == 2:
                            raise
                        time.sleep(5 * (attempt + 1))


class SenseLabEpisodeArm(SenseLabArm):
    """Ablation: everything in ``senselab`` except attempt boundaries. One outcome per task,
    attributed to what the *final* attempt read. When attempt 1 follows a stale lesson and
    fails and attempt 2 succeeds by another route, the stale lesson is not blamed here (the
    grid-v1 behaviour). The gap to ``senselab`` is what per-attempt credit assignment buys."""

    name = "senselab-episode"
    attempt_boundaries = False


class SenseLabNoFeedbackArm(SenseLabArm):
    """Ablation: the same reads, no ``commit_outcome``. Every entry stays untested, so the
    evidence sections are empty and the avoid list never fills; what remains is retrieval."""

    name = "senselab-nofeedback"
    learns_from_outcomes = False
    attempt_boundaries = False


class SenseLabAttemptsArm(SenseLabArm):
    """Legacy grid-v1 variant: each failed attempt committed as its own outcome (one HTTP
    round trip per failure, separate traces). Superseded by ``record_attempt`` in ``senselab``;
    kept only so v1 rows can be compared like for like."""

    name = "senselab-attempts"
    attempt_boundaries = False
    per_attempt_outcomes = True


class SenseLabLeanArm(SenseLabArm):
    """The full protocol with the briefing cut to its evidence sections. The runbook probe
    put the protocol's token premium over ``pgvector`` at ~1.65x, most of it the three
    hot-context previews carried on every call of the episode and then returned again by the
    retrieve. The gap to ``senselab`` is what those previews cost and what they buy."""

    name = "senselab-lean"
    briefing_hot_context = False


class SenseLabNoPriorsArm(SenseLabArm):
    """Ablation: the ``senselab`` protocol as wired for grid v2 — no action priors, no
    recommendation, full (non-compact) retrieve payload, no ``action_key``. The gap to
    ``senselab`` is what action-level learning buys on top of entry-level evidence."""

    name = "senselab-nopriors"
    priors = False
    briefing_since = False


class SenseLabNoSituationArm(SenseLabArm):
    """Ablation: ``senselab`` without the situation label — the priors are the embedding
    neighbourhood's pooled record, as every SenseLab arm ran before 2026-09-26 (grids v4–v5,
    RSI pilots 1–4). The gap to ``senselab`` is what the situation's own record buys:
    one-loss turns, first-win acting, the ordered sweep."""

    name = "senselab-nosituation"
    situations = False


class SenseLabNoContractArm(SenseLabArm):
    """Ablation: ``senselab`` without the standing instruction on how to use the footer
    (``PRIORS_CONTRACT``). The label and the record are unchanged; only the model's
    adherence is left to the footer's own wording."""

    name = "senselab-nocontract"
    contract = False


# ---------------------------------------------------------------------------
# senselab-repair: the shipped repair loop driven from the harness
# ---------------------------------------------------------------------------

REPAIR_JUDGE_ID = "cl-task-outcome"
REPAIR_JUDGE_PROMPT = (
    "You grade one run of an operations agent against the environment's own verdict. The "
    "trace's decision summary and the last tool result state whether the task succeeded "
    "(health checks green, CI green, case resolved) or failed (rolled back, still red, "
    "rejected, escalated to a human, retry budget exhausted). Return FAIL when the run "
    "ended in failure or escalation, PASS when it ended in success. Ignore how the agent "
    "reasoned; the verdict is the environment's, and the point of grading is to name the "
    "runs the repair loop should learn from. Scenario notes: {rubric}"
)

COMPOSE_PROMPT_SUFFIX = (
    "\n\nBefore drafting, read the procedures already on the entity (read_procedures). When two "
    "or more of them each cover part of the failure, compose one procedure whose steps chain "
    "them: each step names the precondition it relies on and the effect it produces, so the "
    "chain can be checked structurally — a step's preconditions must be met by the task or by "
    "an earlier step's effects. Set depends_on to the component entries (key and version) the "
    "composition rests on and evidence to the traces that support each component. Name any "
    "assumption the components disagree on rather than papering over it."
)


class _SenseLabRepairSession(_SenseLabSession):
    """The full protocol plus: after the outcome is sealed, hand the trace to the arm so the
    repair loop can grade it and, on a failure, propose / test / ship a fix between episodes."""

    def end(self, outcome: Outcome, *, task_input: str, response_text: str,
            cited_keys: list[str]) -> None:
        super().end(outcome, task_input=task_input, response_text=response_text, cited_keys=cited_keys)
        trace = getattr(self.mem, "_last_trace", None)
        trace_id = getattr(trace, "id", None) if trace is not None else None
        if trace_id:
            self.arm.pending.append({"agent_id": self.agent_id, "trace_id": str(trace_id),
                                     "episode": self.episode, "success": outcome.success})
        else:
            self.acct.notes["repair_no_trace_id"] = True


class SenseLabRepairArm(SenseLabArm):
    """``senselab`` plus the Pro repair loop, run from the harness between episodes.

    Every sealed trace is graded by one judge (the environment's verdict, not the agent's
    self-report); a failing verdict proposes a fix; the fix's Tier 1 replay runs inline
    (``POST /fixes/{id}/test?now=1``); a passed fix ships to memory (``auto_after_replay``
    policy on the cell's agents, ``approve-memory`` as the fallback when the policy did not
    apply). The shipped corrective entry or procedure is what the next episodes read.

    Requires a dev Pro deployment with the repair agent enabled (``AMFS_EVAL_REPAIR_AGENT``
    on the pro-api process). Pro calls are charged to the arm's ``extra_*`` accounting; the
    judge and repair cost is reported from what the server returns (verdict ``cost_usd``),
    so it is a lower bound where the server omits a figure.
    """

    name = "senselab-repair"
    lever_override: str | None = None       # let the classifier choose the lever
    repair_policy = "auto_after_replay"
    compose = False

    def open(self, scope: str) -> None:
        super().open(scope)
        from ..pro_client import ProClient, assert_dev

        assert_dev(config.AMFS_HTTP_URL, config.AMFS_PRO_URL)
        self.pro = ProClient()
        self.pending: list[dict[str, Any]] = []
        self._agents_ready: set[str] = set()
        self._rubric = ""
        self.repairs: list[dict[str, Any]] = []

    def configure(self, scenario) -> None:
        super().configure(scenario)
        self._rubric = getattr(scenario, "judge_rubric", "") or "none"
        if getattr(scenario, "procedural", False) and self.lever_override is None:
            self.lever_override = "procedure"

    def session(self, agent_id: str, episode: int) -> EpisodeSession:
        return _SenseLabRepairSession(self, agent_id, episode)

    def _ready(self, agent_id: str) -> None:
        if agent_id in self._agents_ready:
            return
        self.pro.ensure_judge(REPAIR_JUDGE_ID, agent_id, "Task outcome (benchmark)",
                              REPAIR_JUDGE_PROMPT.format(rubric=self._rubric[:1500]))
        self.pro.set_repair_settings(agent_id, repair_policy=self.repair_policy)
        if self.compose:
            active = self.pro.get_repair_prompt(agent_id)
            base = (active.get("active") or {}).get("prompt") if isinstance(active.get("active"), dict) else None
            base = base or active.get("prompt") or ""
            if COMPOSE_PROMPT_SUFFIX.strip() not in base:
                self.pro.set_repair_prompt(agent_id, (base + COMPOSE_PROMPT_SUFFIX).strip())
        self._agents_ready.add(agent_id)

    def _repair_one(self, item: dict[str, Any], acct: ArmAccounting) -> dict[str, Any]:
        from ..pro_client import ProApiError

        rec: dict[str, Any] = {"episode": item["episode"], "agent_id": item["agent_id"], "trace_id": item["trace_id"]}
        agent = item["agent_id"]
        self._ready(agent)
        graded = self.pro.judge(item["trace_id"], REPAIR_JUDGE_ID, agent_id=agent)
        verdicts = [v for v in graded.get("verdicts", []) if isinstance(v, dict)]
        for v in verdicts:
            acct.extra_cost_usd += float(v.get("cost_usd") or 0.0)
            acct.extra_prompt_tokens += int(v.get("input_tokens") or 0)
            acct.extra_completion_tokens += int(v.get("output_tokens") or 0)
        failing = [v for v in verdicts if v.get("verdict") == "fail"]
        rec["verdict"] = (verdicts[0].get("verdict") if verdicts else None)
        rec["judge_skipped"] = graded.get("skipped") or []
        if not failing:
            return rec
        try:
            fix = self.pro.propose(agent, verdict_id=str(failing[0]["id"]), lever_override=self.lever_override)
        except ProApiError as e:
            rec["propose_error"] = f"{e.status}: {str(e.detail)[:200]}"
            return rec
        fix_id = str(fix["id"])
        rec.update(fix_id=fix_id, lever=fix.get("lever"), proposed_status=fix.get("status"))
        try:
            fix = self.pro.test_fix(fix_id, now=True)
        except ProApiError as e:
            rec["test_error"] = f"{e.status}: {str(e.detail)[:200]}"
            return rec
        if fix.get("status") in ("proposed", "testing"):
            fix = self.pro.wait_tested(fix_id)
        rec["tested_status"] = fix.get("status")
        rec["test_result"] = (fix.get("test_result") or fix.get("last_test") or {}) if isinstance(fix, dict) else {}
        if fix.get("status") == "test_passed":
            # Policy should have shipped it; ship by hand when it did not.
            try:
                fix = self.pro.approve_memory(fix_id)
                rec["shipped_by"] = "approve-memory"
            except ProApiError as e:
                rec["approve_error"] = f"{e.status}: {str(e.detail)[:200]}"
        elif fix.get("status") in ("shipped", "live", "verified", "proven"):
            rec["shipped_by"] = "policy"
        rec["final_status"] = fix.get("status")
        return rec

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        from ..pro_client import ProApiError

        items, self.pending = list(self.pending), []
        failures = [it for it in items if not it["success"]]
        if not failures:
            return None
        acct = ArmAccounting()
        t0 = time.perf_counter()
        done: list[dict[str, Any]] = []
        for it in failures:
            try:
                rec = self._repair_one(it, acct)
            except ProApiError as e:
                rec = {"episode": it["episode"], "agent_id": it["agent_id"], "trace_id": it["trace_id"],
                       "error": f"{e.status}: {str(e.detail)[:200]}"}
            except Exception as e:  # noqa: BLE001
                rec = {"episode": it["episode"], "agent_id": it["agent_id"], "trace_id": it["trace_id"],
                       "error": f"{type(e).__name__}: {e}"[:200]}
            done.append(rec)
            acct.ops += 1
        self.repairs.extend(done)
        acct.memory_ms = (time.perf_counter() - t0) * 1000
        acct.notes = {"repairs": done}
        acct.notes["shipped_total"] = sum(1 for r in self.repairs if r.get("shipped_by"))
        acct.notes["proposed_total"] = sum(1 for r in self.repairs if r.get("fix_id"))
        return acct

    def close(self) -> None:
        try:
            self.pro.close()
        except Exception:  # noqa: BLE001
            pass


class SenseLabComposeArm(SenseLabRepairArm):
    """``senselab-repair`` with the composition prompt installed on the cell's agents: the
    repair agent reads the procedures already on the entity and composes them when each
    covers part of the failure. The prototype composer of the fleet-disjoint experiment."""

    name = "senselab-compose"
    compose = True
    lever_override = "procedure"


INDUCE_PROMPT = """You maintain the decision rules for an agent that repeatedly handles tasks of one kind on one system. Below is that agent's outcome record: for each past task, the task as it was posed, what the agent observed, the actions it attempted in order with the result of each, and the final outcome. Records are in time order, oldest first.

Induce the smallest set of rules of the form "IF <conditions on the task or observations> THEN <action>" that best predicts success on the next task of this kind. Requirements:
- Every rule is grounded in the record: give the count of successes and failures that support it.
- Newer records outrank older ones. If an action that used to succeed on some kind of task has been failing on it recently, state the current rule, say that the old one no longer holds, and mark roughly where in the record it changed.
- Where the record is insufficient, say which cases are uncertain and which actions have not been tried on them, so the agent can try those first.
- Do not invent conditions the record cannot support. Prefer conditions the agent can check from the task text or its observations.

Plain text, at most 12 rules, then a short "uncertain / untried" section. No preamble.

OUTCOME RECORD ({n} tasks):
{record}
"""


class SenseLabInduceArm(SenseLabArm):
    """The ``senselab`` protocol plus rule induction from the outcome record.

    Every ``induce_every`` episodes the arm reads the entity's decision traces — the task as
    posed, the non-memory tool results, the terminal actions attempted with each result, the
    outcome — and asks the model to induce conditional rules from them, written to one
    versioned key (``INDUCED_RULES_KEY``). Each episode's briefing then opens with the current
    rules, read by key, so the episode's outcome lands on the rules entry and its evidence
    status (validated / contested / discredited) is shown next to them.

    What this tests: grid v5 showed every memory arm — including ``senselab`` and
    ``pgvector-diy+outcomes``, whose consolidation summarises the agent's *notes* with win/loss
    tallies — plateauing at 0.55-0.65 first-attempt success on tasks whose answer is a
    conditional rule over task attributes (retention: five rules over four fields). Retrieved
    episodes and per-action priors are both marginal over the attributes; the rule has to be
    induced, and the outcome record is the one place the evidence for it is complete and not
    filtered by what the agent chose to write down. ``pgvector`` has no such record."""

    name = "senselab-induce"
    induces = True
    induce_every = int(os.environ.get("CL_INDUCE_EVERY", "10"))
    induce_window = 120     # traces read per induction (newest first from the API)

    def open(self, scope: str) -> None:
        super().open(scope)
        self._inducer = AgentMemory(
            agent_id=f"{scope.rsplit('/', 1)[-1][:40]}-inducer",
            adapter=_ThrottledHttpAdapter(base_url=config.AMFS_HTTP_URL,
                                          api_key=config.env("AMFS_API_KEY", required=True)),
        )
        self._terminal_tool: str | None = None

    def after_episode(self, episode: int, llm) -> ArmAccounting | None:
        if (episode + 1) % self.induce_every != 0:
            return None
        acct = ArmAccounting()
        t0 = time.perf_counter()
        try:
            traces = self._inducer.list_traces(entity_path=self.scope, limit=self.induce_window)
        except Exception as e:  # noqa: BLE001
            acct.notes = {"induce_error": f"list_traces: {e}"[:200]}
            return acct
        rows = []
        for t in reversed(list(traces)):       # oldest first
            d = t.model_dump(mode="json") if hasattr(t, "model_dump") else dict(getattr(t, "__dict__", {}))
            calls = d.get("tool_calls") or []
            obs, attempts = [], []
            for c in calls:
                name = str(c.get("tool_name") or "")
                if name in ("memory_search", "memory_write", "amfs_retrieve", "amfs_write") or not name:
                    continue
                args = c.get("arguments") or {}
                res = str(c.get("result_summary") or "")
                if isinstance(args, dict) and "action" in args:
                    attempts.append(f"{args.get('action')} -> {res[:160]}")
                else:
                    obs.append(f"{name}({json.dumps(args, default=str)[:60]}): {res[:160]}")
            if not attempts and d.get("response_text"):
                attempts.append(f"{str(d['response_text'])[:60]} -> {d.get('outcome_type')}")
            rows.append(
                f"- task: {str(d.get('task_input') or '')[:320]}\n"
                + (f"  observed: {' | '.join(obs)[:480]}\n" if obs else "")
                + f"  attempts: {' ; '.join(attempts)[:600]}\n"
                + f"  outcome: {d.get('outcome_type')}"
            )
        if len(rows) < 4:
            acct.notes = {"induce_skipped": f"{len(rows)} traces"}
            return acct
        try:
            text, usage = llm.complete_text(INDUCE_PROMPT.format(n=len(rows), record="\n".join(rows)), max_tokens=1100)
        except Exception as e:  # noqa: BLE001
            acct.notes = {"induce_error": f"llm: {e}"[:200]}
            return acct
        acct.extra_prompt_tokens += usage.prompt_tokens
        acct.extra_completion_tokens += usage.completion_tokens
        acct.extra_cost_usd += usage.cost_usd
        text = (text or "").strip()
        if not text:
            acct.notes = {"induce_error": "empty"}
            return acct
        try:
            self._inducer.write(self.scope, INDUCED_RULES_KEY,
                                f"Induced at episode {episode + 1} from {len(rows)} recorded outcomes.\n{text}"[:6000],
                                confidence=0.85, memory_type=MemoryType.FACT)
        except Exception as e:  # noqa: BLE001
            acct.notes = {"induce_error": f"write: {e}"[:200]}
            return acct
        acct.ops = 2
        acct.memory_ms = (time.perf_counter() - t0) * 1000
        acct.notes = {"induced_from": len(rows), "chars": len(text)}
        return acct

    def close(self) -> None:
        try:
            self._inducer.close()
        except Exception:  # noqa: BLE001
            pass
        super().close()
