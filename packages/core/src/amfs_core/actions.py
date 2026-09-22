"""Action-level learning: what happened when an agent *did* something here.

Memory entries record what someone wrote. The outcomes table records what
happened: each committed outcome carries the actions the agent took and which of
them failed (the attempts) or resolved the task (``final_action_index``). This
module turns that record into two things an agent can use before acting:

* **action priors** — for the tasks most similar to the current one, how each
  action fared: won / lost counts, a Beta posterior, the last few results, how
  many distinct agents tried it, and which candidate actions nobody has tried;
* a **recommendation** — ``act`` on a validated rule or a winning action,
  ``explore`` an untried action (assigned per agent, so a fleet spreads its
  search instead of six agents retrying the same two failures), or ``escalate``
  when everything a caller could try has already failed here.

Everything is pure and adapter-agnostic. The server pairs it with a nearest-
neighbour query over ``amfs_outcomes.task_embedding``; the benchmark's motivating
case is a support fleet that spent 24 attempts per class on the same two failing
actions and never once tried the one that worked.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .evidence import is_success
from .models import ENVIRONMENT_KEYS

#: Largest argument value that may become part of an action key. Longer strings
#: are free text (a reply, a note) and never identify an action.
ACTION_VALUE_MAX_CHARS = 40
#: Nearest outcomes considered for priors.
PRIORS_K = 20
#: Cosine similarity below which a past task is not "the same kind of task".
#: A floor, not a neighbourhood: see :func:`neighbourhood_weights`.
PRIORS_MIN_SIMILARITY = 0.75
#: How fast an outcome's weight in the priors falls with its similarity gap
#: to the nearest outcome. Calibrated on grid v3's logged task prompts under
#: the production embedder (bge-small): same-class pairs sit at 0.91-0.98,
#: cross-class pairs at 0.81-0.96, and the two overlap on any absolute
#: threshold — but within one query the nearest outcomes are the same class
#: and the cross-class ones trail by 0.05-0.11. exp(-gap/0.03) puts a task
#: 0.05 behind the best at a fifth of its weight and 0.1 behind at 4%.
PRIORS_NEIGHBOURHOOD_TAU = 0.03
#: Outcomes whose neighbourhood weight falls under this are dropped rather
#: than counted: they would inflate ``n`` while barely moving ``p``.
PRIORS_NEIGHBOURHOOD_MIN_W = 0.1
#: Per-day decay of an outcome's weight in the priors.
PRIORS_DAILY_DECAY = 0.9
#: Recommendation thresholds.
ACT_MIN_P = 0.6
ACT_MIN_N = 2
EXPLORE_MAX_P = 0.4
EXPLORE_MIN_N = 2
#: Weight of an outcome recorded under a different model / runtime / agent
#: version than the run asking. Kept well above zero: another runtime's win is
#: still evidence, it just should not outvote a same-runtime loss.
ENV_MISMATCH_WEIGHT = 0.5
#: Evidence statuses of a top hit that carry no weight of their own.
_WEAK_STATUSES = frozenset({"untested", "contested", "discredited"})
#: Consecutive newest losses after which an action's lifetime record no
#: longer makes it a winner. ``last_3`` is all the record keeps, so this is
#: its full length: a win-then-three-losses is the shape of a rule that has
#: stopped working, and grid v5 measured the alternative — ``act`` kept
#: naming an action that had won 8/8 before a change and 0/5 since, for as
#: long as it took the lifetime ratio to fall under ``ACT_MIN_P`` (a fifth
#: of the episodes of that task class never found the new fix).
RECENT_FAIL_STREAK = 3
#: Neighbourhood weight (``neighbourhood_weights``) a contrast pair needs
#: before one fail-then-succeed outcome is enough to recommend the action
#: that resolved it. ``exp(-gap/tau)`` at 0.25 is a task within ~0.04 of the
#: nearest outcome under the production embedder — the same issue phrased
#: differently, not a neighbouring class. ``ACT_MIN_N`` asks for two wins
#: before acting on a record; a contrast is the one case where one outcome
#: carries both halves of the evidence — what failed and what worked on the
#: same task — and grid v4 measured what waiting for the second costs: most
#: (store, issue) pairs saw a quirk only once or twice, and the store's
#: success on the n-th exposure of the same issue ran 0.00, 0.12, 0.23.
CONTRAST_MIN_W = 0.25
#: A validated action that has just *turned*: won at least this many times
#: on tasks like this, and lost its newest ``REGIME_TURN_STREAK`` takes. It is
#: no longer a winner, before ``RECENT_FAIL_STREAK`` would say so. This is the
#: action-level reading of :func:`amfs_core.evidence.regime_shifted`, which
#: forgives one failure against a long run of successes and not the second in
#: a row — and it survives what the entry-level rule does not: an agent that
#: rewrites its lesson after the failure ("no fix known") opens a new claim
#: with an empty record, so the entry never shows the shift, while the
#: situation's action record keeps every take. Measured on the ops-queue CI
#: demo (2026-09-22): after a change, ``rerun_job`` 3/4 with its newest take
#: lost was still ``act`` on the next task of the class, and the task after
#: that; the class cost three CI runs a task until the third loss.
REGIME_MIN_WINS = 2
REGIME_TURN_STREAK = 2
#: Two contradicting sides of a contrast record are *pooled* — two kinds of
#: task under one description — only when they sit at comparable weight. A
#: contradicting side whose heaviest contrast is under this fraction of the
#: other side's is a neighbouring class, not this one: measured on the same
#: demo, the same-class contrasts weigh 0.92-1.0 against the query and the
#: sibling class (same jest check, a backend PR instead of a UI one) 0.25-0.38.
#: Reading the sibling as a contradiction abstained on both classes when the
#: neighbourhood-weighted record named the right fix for each.
POOLED_WEIGHT_RATIO = 0.6
#: Longest plan :func:`plan_actions` returns.
PLAN_MAX = 5

_WHITESPACE = re.compile(r"\s")


def action_key(
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    explicit: str | None = None,
) -> str:
    """The identity of an action: ``<tool>:<action value>``, or the tool alone.

    The action value is the argument named ``action`` when present, otherwise the
    first argument (by name) whose value is a short string with no whitespace —
    an enum member, an id, a mode. Free text never qualifies, so a reply body or a
    note cannot split one action into a thousand keys. An explicit ``action_key``
    recorded by the agent wins over both.
    """
    if explicit:
        return str(explicit)[: ACTION_VALUE_MAX_CHARS * 2]
    tool = str(tool_name or "").strip() or "action"
    args = arguments or {}
    value = args.get("action")
    if not _short_token(value):
        value = None
        for name in sorted(args):
            if name == "used_memory_keys":
                continue
            if _short_token(args[name]):
                value = args[name]
                break
    return f"{tool}:{value}" if value is not None else tool


def _short_token(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= ACTION_VALUE_MAX_CHARS
        and not _WHITESPACE.search(value)
    )


def actions_taken(
    tool_calls: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    final_action_index: int | None,
    outcome_type: str,
) -> list[dict[str, Any]]:
    """Label the actions in a trace with what happened to them.

    The action that *ended* each failed attempt (the last of its
    ``action_indices``) is a loss with that attempt's index; the terminal action
    is a win or a loss by the outcome type. The other tool calls — lookups and
    diagnostics that preceded the decisive action — are observations, not
    decisions, and are not labeled. Defensive about indices: a trace can carry
    fewer ``tool_calls`` than the attempt boundaries expect when the client
    trimmed them.
    """
    out: list[dict[str, Any]] = []
    n = len(tool_calls)

    def _row(i: int, success: bool, attempt: int | None) -> dict[str, Any] | None:
        if not 0 <= i < n:
            return None
        call = tool_calls[i]
        key = action_key(
            str(call.get("tool_name", "")),
            call.get("arguments") or {},
            explicit=call.get("action_key"),
        )
        return {
            "index": i,
            "action_key": key,
            "tool_name": str(call.get("tool_name", "")),
            "success": bool(success),
            "attempt": attempt,
        }

    seen: set[int] = set()
    for att in attempts:
        idx = [i for i in (att.get("action_indices") or []) if isinstance(i, int) or str(i).isdigit()]
        if not idx:
            continue
        i = int(idx[-1])
        if i in seen:
            continue
        row = _row(i, False, att.get("attempt"))
        if row:
            seen.add(i)
            out.append(row)
    if final_action_index is not None and final_action_index not in seen:
        row = _row(int(final_action_index), is_success(outcome_type), None)
        if row:
            out.append(row)
    return out


def entity_paths_of(causal_entry_keys: Iterable[str], explicit: str | None = None) -> list[str]:
    """Distinct entity paths an outcome is about: the explicit one plus those of
    its causal keys (``entity/path/key`` -> ``entity/path``), in first-seen order."""
    out: dict[str, None] = {}
    if explicit:
        out[str(explicit)] = None
    for spec in causal_entry_keys or ():
        parts = str(spec).rsplit("/", 1)
        if len(parts) == 2 and parts[0]:
            out.setdefault(parts[0], None)
    return list(out)


# ── Priors ─────────────────────────────────────────────────────────────


@dataclass
class ActionPrior:
    action_key: str
    won: int = 0
    lost: int = 0
    won_w: float = 0.0
    lost_w: float = 0.0
    last_3: list[str] = field(default_factory=list)   # newest first: "won" / "lost"
    last_at: datetime | None = None
    agents: set[str] = field(default_factory=set)

    @property
    def n(self) -> int:
        return self.won + self.lost

    @property
    def p(self) -> float:
        """Beta posterior mean of success with a uniform prior, on the decayed masses."""
        return (self.won_w + 1.0) / (self.won_w + self.lost_w + 2.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_key": self.action_key,
            "won": self.won,
            "lost": self.lost,
            "p": round(self.p, 3),
            "n": self.n,
            "last_3": list(self.last_3[:3]),
            "last_at": self.last_at.isoformat() if self.last_at else None,
            "agents": len(self.agents),
            # The neighbourhood-weighted masses behind ``p``: what lets a
            # reader tell a loss on this task from a loss on a neighbour.
            "won_w": round(self.won_w, 3),
            "lost_w": round(self.lost_w, 3),
        }


def aggregate_priors(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    candidate_actions: Sequence[str] | None = None,
    now: datetime | None = None,
    daily_decay: float = PRIORS_DAILY_DECAY,
    environment: Mapping[str, Any] | None = None,
    env_mismatch_weight: float = ENV_MISMATCH_WEIGHT,
) -> dict[str, Any]:
    """Fold similar past outcomes into per-action priors.

    Each item of ``outcomes`` is a row with ``actions_taken`` (as produced by
    :func:`actions_taken`), ``committed_at`` and ``agent_id``; ``similarity`` is
    optional and multiplies the weight. Returns ``{"tried": [...], "untried":
    [...], "n_outcomes": int, "contrasts": [...]}`` with ``tried`` sorted by
    posterior descending and, within ties, by evidence.

    When *environment* is given (``{"model": ..., "runtime": ...}``, see
    ``amfs_core.models.environment_of``), a row whose ``session_metadata`` (or
    ``environment``) names a different value for any of those keys is weighted
    by *env_mismatch_weight*: what won under another runtime is evidence, but
    weaker. Rows that report no environment are unaffected, and so is every
    caller that passes none — the default output is unchanged.

    ``contrasts`` are the fail-then-succeed outcomes among the rows: one
    outcome in which an attempt ended on action ``A`` and failed, and the
    terminal action ``B`` (a different one) succeeded. Each is ``{"failed":
    [A, ...], "resolved_with": B, "weight", "task_similarity", "committed_at",
    "outcome_ref", "agent_id"}``, heaviest (nearest, then newest) first. The
    per-action ``tried`` rows already count A's loss and B's win; what they
    lose is that the two came from the *same task* — the one piece of
    evidence that says "when A fails here, B is what works", and the only
    evidence there is after a single exposure. :func:`recommend` reads it.
    """
    now = now or datetime.now(timezone.utc)
    env = {k: str(v).strip() for k, v in (environment or {}).items() if v}
    priors: dict[str, ActionPrior] = {}
    contrasts: list[dict[str, Any]] = []
    # newest first so last_3 fills in time order
    rows = sorted(outcomes, key=lambda r: _as_dt(r.get("committed_at")) or now, reverse=True)
    for row in rows:
        at = _as_dt(row.get("committed_at"))
        age_days = max(0.0, (now - at).total_seconds() / 86400.0) if at else 0.0
        w = (daily_decay ** age_days) * float(row.get("similarity", 1.0) or 1.0)
        if env:
            w *= _env_match(row, env, env_mismatch_weight)
        agent = str(row.get("agent_id") or "")
        failed_here: list[str] = []
        resolved_here: str | None = None
        for act in row.get("actions_taken") or []:
            key = str(act.get("action_key") or "")
            if not key:
                continue
            pr = priors.setdefault(key, ActionPrior(action_key=key))
            ok = bool(act.get("success"))
            if ok:
                pr.won += 1
                pr.won_w += w
            else:
                pr.lost += 1
                pr.lost_w += w
            if len(pr.last_3) < 3:
                pr.last_3.append("won" if ok else "lost")
            if at and (pr.last_at is None or at > pr.last_at):
                pr.last_at = at
            if agent:
                pr.agents.add(agent)
            if act.get("attempt") is not None and not ok:
                if key not in failed_here:
                    failed_here.append(key)
            elif act.get("attempt") is None and ok:
                resolved_here = key
        if failed_here and resolved_here and resolved_here not in failed_here:
            contrasts.append({
                "failed": failed_here,
                "resolved_with": resolved_here,
                "weight": round(float(row.get("similarity", 1.0) or 1.0), 3),
                "task_similarity": (
                    round(float(row["task_similarity"]), 3)
                    if row.get("task_similarity") is not None else None
                ),
                "committed_at": at.isoformat() if at else None,
                "outcome_ref": row.get("outcome_ref"),
                "agent_id": agent or None,
            })
    # Rows were walked newest first and the sort is stable, so within a weight
    # the newest contrast leads.
    contrasts.sort(key=lambda c: -float(c["weight"]))
    tried = sorted(priors.values(), key=lambda p: (-p.p, -p.n, p.action_key))
    tried_keys = {p.action_key for p in tried}
    untried = [a for a in (candidate_actions or []) if a and a not in tried_keys]
    return {
        "tried": [p.as_dict() for p in tried],
        "untried": untried,
        "n_outcomes": len(rows),
        "contrasts": contrasts,
    }


def recorded_environment(row: Mapping[str, Any]) -> dict[str, str]:
    """The environment an outcome row was recorded in, from wherever the
    producer put it: an ``environment`` block, the top level of
    ``session_metadata`` (``model``), its ``attributes`` (where ``Run.begin``
    and ``set_session_attributes`` stamp ``agent_version`` / ``runtime``), or a
    trace's ``attributes`` bag. Later sources fill in keys earlier ones left
    unset; the first that names a key wins."""
    out: dict[str, str] = {}
    sources: list[Any] = [row.get("environment")]
    meta = row.get("session_metadata")
    if isinstance(meta, Mapping):
        sources.append(meta)
        sources.append(meta.get("attributes"))
    sources.append(row.get("attributes"))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ENVIRONMENT_KEYS:
            if key in out:
                continue
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
    return out


def _env_match(row: Mapping[str, Any], env: Mapping[str, str], mismatch_weight: float) -> float:
    """1.0 when the row's recorded environment agrees with *env* on every key
    both report (or reports nothing); *mismatch_weight* otherwise."""
    recorded = recorded_environment(row)
    for key, want in env.items():
        have = recorded.get(key)
        if have is None:
            continue
        if have != want:
            return mismatch_weight
    return 1.0


def neighbourhood_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    tau: float = PRIORS_NEIGHBOURHOOD_TAU,
    min_weight: float = PRIORS_NEIGHBOURHOOD_MIN_W,
) -> list[dict[str, Any]]:
    """Re-weight similar outcomes relative to the nearest one.

    ``similar_outcomes`` returns rows above an absolute similarity floor, and
    under a retrieval embedder that floor admits every task on the entity:
    the priors were pooled over classes of task that share a vocabulary, and
    a "tried and failed here" read over the wrong class sent agents to
    explore actions that were winning for the class they were on (grid v3:
    an ``explore`` recommendation, when followed, succeeded 14% of the time
    against 67% when ignored). Absolute thresholds cannot fix that — the
    same-class and cross-class similarity ranges overlap — but the *gap* to
    the best match can: within one query the nearest outcomes are the same
    kind of task, and the rest trail.

    Each row's ``similarity`` becomes ``exp(-(best - sim) / tau)``, which
    :func:`aggregate_priors` multiplies into its weight; rows under
    *min_weight* are dropped so they do not count toward ``n``. Rows without a
    similarity are returned unchanged. Where classes are indistinguishable by
    task text (grid v3's diagnose: the prompt is a template and the class is
    in the diagnostic findings) the weights flatten and the priors are pooled
    as before — the caller should then pass ``situation`` so the outcome is
    embedded with what distinguished it.
    """
    sims = [float(r.get("similarity") or 0.0) for r in rows if r.get("similarity") is not None]
    if not sims:
        return [dict(r) for r in rows]
    best = max(sims)
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("similarity") is None:
            out.append(dict(r))
            continue
        gap = max(0.0, best - float(r["similarity"]))
        w = math.exp(-gap / tau) if tau > 0 else (1.0 if gap == 0.0 else 0.0)
        if w < min_weight:
            continue
        row = dict(r)
        row["task_similarity"] = float(r["similarity"])
        row["similarity"] = w
        out.append(row)
    return out


def _as_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ── Recommendation ────────────────────────────────────────────────────────


def stable_bucket(agent_id: str, n: int) -> int:
    """``agent_id`` -> ``[0, n)``, identical on every process and machine.

    Python's ``hash`` is salted per interpreter, which would hand a different
    exploration assignment to the same agent on each server instance.
    """
    if n <= 0:
        return 0
    digest = hashlib.sha1((agent_id or "").encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % n


def _won_since(prior: Mapping[str, Any], moment: datetime | None) -> bool:
    """Whether the prior's newest take came after *moment* and won.

    Without a *moment* there is nothing to be after, so the answer is no: the
    shift is then read as covering every win in the record.
    """
    if moment is None:
        return False
    last_3 = list(prior.get("last_3") or [])
    if not last_3 or last_3[0] != "won":
        return False
    raw = prior.get("last_at")
    if isinstance(raw, datetime):
        at = raw
    elif isinstance(raw, str) and raw:
        try:
            at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return False
    else:
        return False
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return at > moment


def recommend(
    priors: Mapping[str, Any] | None,
    *,
    agent_id: str = "",
    candidate_actions: Sequence[str] | None = None,
    top_hit_status: str | None = None,
    top_hit_recent_failure: bool = False,
    top_hit_shifted: bool = False,
    regime_shift: bool = False,
    regime_shift_at: datetime | None = None,
    abstain: bool = False,
    hit_statuses: Sequence[str] | None = None,
    priors_are_local: bool = True,
) -> dict[str, Any] | None:
    """Decide ``act`` / ``explore`` / ``escalate`` from priors and the top hit.

    Returns ``None`` when there is nothing to say (no priors, no validated hit,
    no regime shift), so the payload stays unchanged for callers that gain
    nothing. ``escalate`` is only ever returned when the caller supplied
    ``candidate_actions`` — without knowing what could be tried, "everything
    failed" cannot be asserted, and a false escalate costs a task a retry
    would have won.

    With ``abstain=True`` the "nothing to say" case is spelled out instead of
    returning ``None`` when the evidence is weak: no priors at all and every
    hit's evidence status (``hit_statuses``, or the top hit's alone) is
    untested, contested or discredited. The agent is told so
    (``{"mode": "abstain"}``) rather than left to read confidence into a list
    of hits nothing has confirmed. Off by default so existing payloads do not
    change.

    Everything here is meant to be read over *this kind of task*: the priors
    over the outcomes nearest the query (:func:`neighbourhood_weights`) and
    ``regime_shift`` over the entries the query is about. Grid v3 measured
    what happens otherwise — an entity-wide shift and pooled priors sent
    agents to explore past memory that was right for their task, and the
    ``explore`` they followed won 14% of the time.

    A regime shift skips the winning priors, because their wins may predate the
    change — except a winner whose latest take was *after* the shift and won
    (``regime_shift_at`` against the prior's ``last_at``, and its newest
    ``last_3`` entry). That is the replacement the shift called for, already
    found; sending the agent to explore past it would re-learn what the record
    already knows.

    A validated top hit with no recent failure is acted on even under a shift,
    unless the top hit is itself the rule that shifted (``top_hit_shifted``).
    The shift says *something* on the entity stopped working; the hit's own
    record says this did not, and the record of the thing in hand outranks a
    flag about its neighbours.

    ``explore`` needs a record to explore *from*: everything tried on tasks
    like this has failed, or a shift, and in either case at least one action
    tried. With nothing tried the "untried" list is every candidate and the
    pick is a hash of the agent's name — advice with no information in it.

    A winner is also read against its newest takes. An action that lost its
    last ``RECENT_FAIL_STREAK`` outcomes is not acted on whatever its
    lifetime ratio, and one that had won before counts as a shift at the
    action level: the explore that follows names it ("won 8/11, lost its
    last 3 — what worked here has stopped working"). Grid v5 measured the
    alternative: after a change, ``act`` kept naming the old fix for as long
    as its lifetime ratio stayed over ``ACT_MIN_P``, and an action tried once
    and lost kept "everything tried has failed" from ever becoming true —
    17 of 24 agent groups never found the new fix in 40 episodes. Firm
    failures (a low ratio over ``EXPLORE_MIN_N`` takes, or a streak) are what
    ``escalate`` asserts; for ``explore`` an action never seen to win is
    failed enough, since holding an agent on a 0/1 while candidates sit
    untried is the worse bet.

    ``priors_are_local=False`` says the priors are the entity's whole record
    (the ``action_stats`` fallback of a store without task embeddings). Such
    a record can still name a winner, and "every action tried here failed"
    still means something when it is every action; but a shift read over it
    does not send the agent exploring, since neither the shift nor the record
    is known to be about this kind of task.

    A *contrast* — one nearby outcome in which action A failed an attempt and
    action B then resolved the same task — is acted on from a single outcome
    (:func:`_act_from_contrast`), ahead of the per-action winners, when the
    winner it would displace is A itself or there is no winner. ``ACT_MIN_N``
    exists because one win may be luck; a contrast is one outcome that holds
    both the failure and the fix for the same task, and the alternative — act
    on A because its record is long — is the repeated failure grid v4
    measured. Local priors only: a contrast from the entity's whole record is
    not known to be about this kind of task.

    A validated action that lost its newest ``REGIME_TURN_STREAK`` takes has
    *turned* (:func:`turned`) and is read like one that stopped: not a winner,
    named in the explore. A *lone winner* — the only action ever seen to win
    on tasks like this, its newest take a win, every other tried action
    failed — is acted on from one win: ``ACT_MIN_N`` guards against luck, and
    when the alternatives are all known failures the one win is the best bet
    there is. Measured on the ops-queue CI demo (2026-09-22): a class whose
    fix had changed was found on one task (1/1) and, with ``act`` withheld
    and the other five actions all 0/n, the next task of the class was
    brute-forced again and missed.

    Every recommendation carries a ``plan`` (:func:`plan_actions`): the order
    to try actions in for the whole budget, never repeating what failed here.
    A single ``suggested_action`` is one attempt's worth of advice; an agent
    with three attempts wasted the other two re-trying its instinct.
    """
    tried: list[Mapping[str, Any]] = list((priors or {}).get("tried") or [])
    untried: list[str] = list((priors or {}).get("untried") or [])
    have_candidates = bool(candidate_actions)

    def _with_plan(rec: dict[str, Any] | None) -> dict[str, Any] | None:
        if rec is None:
            return None
        rec.setdefault(
            "plan",
            plan_actions(
                priors, rec, agent_id=agent_id, candidate_actions=candidate_actions,
                priors_are_local=priors_are_local,
            ),
        )
        return rec

    # A winner is read from its record *and* its recent takes: an action that
    # lost its newest RECENT_FAIL_STREAK outcomes is not one, however long it
    # won before — nor is a validated one that lost its newest
    # REGIME_TURN_STREAK (``turned``). Those with a real record behind them
    # are what a shift looks like at the action level ("what worked here has
    # stopped working"), and they are named in the explore that follows.
    winners = [
        t for t in tried
        if float(t.get("p", 0)) >= ACT_MIN_P and int(t.get("n", 0)) >= ACT_MIN_N and not _stopped_working(t)
    ]
    stopped = [t for t in tried if _stopped_working(t) and int(t.get("won", 0)) >= ACT_MIN_N]
    # Two readings of "failed". ``losers`` is the firm one — a low ratio over
    # at least EXPLORE_MIN_N takes, or a streak — and is what ``escalate``
    # asserts. ``failed_now`` also counts an action never seen to win: a 0/1
    # is weak evidence *for* the action, but it is no reason to hold an agent
    # on it when candidates are untried, and grid v5 found the explore it
    # blocked was the one that would have found the new fix.
    losers = [
        t for t in tried
        if (float(t.get("p", 1)) < EXPLORE_MAX_P and int(t.get("n", 0)) >= EXPLORE_MIN_N) or _stopped_working(t)
    ]
    failed_now = [t for t in tried if t in losers or int(t.get("won", 0)) == 0]
    all_tried_failed = bool(tried) and len(failed_now) == len(tried)
    all_tried_failed_firm = bool(tried) and len(losers) == len(tried)

    return _with_plan(_recommend_mode(
        priors, tried=tried, untried=untried, winners=winners, stopped=stopped, losers=losers,
        failed_now=failed_now, all_tried_failed=all_tried_failed,
        all_tried_failed_firm=all_tried_failed_firm, have_candidates=have_candidates,
        agent_id=agent_id, top_hit_status=top_hit_status,
        top_hit_recent_failure=top_hit_recent_failure, top_hit_shifted=top_hit_shifted,
        regime_shift=regime_shift, regime_shift_at=regime_shift_at, abstain=abstain,
        hit_statuses=hit_statuses, priors_are_local=priors_are_local,
    ))


def _recommend_mode(
    priors: Mapping[str, Any] | None,
    *,
    tried: list[Mapping[str, Any]],
    untried: list[str],
    winners: list[Mapping[str, Any]],
    stopped: list[Mapping[str, Any]],
    losers: list[Mapping[str, Any]],
    failed_now: list[Mapping[str, Any]],
    all_tried_failed: bool,
    all_tried_failed_firm: bool,
    have_candidates: bool,
    agent_id: str,
    top_hit_status: str | None,
    top_hit_recent_failure: bool,
    top_hit_shifted: bool,
    regime_shift: bool,
    regime_shift_at: datetime | None,
    abstain: bool,
    hit_statuses: Sequence[str] | None,
    priors_are_local: bool,
) -> dict[str, Any] | None:
    """The mode and suggested action of :func:`recommend`; the plan is added
    by the caller."""
    if priors_are_local:
        contrasts = list((priors or {}).get("contrasts") or [])
        # Near-identical outcomes that contradict each other over time are two
        # classes of task pooled under one description: neither the contrast
        # rule nor the per-action winners can name the fix for *this* one, and
        # ``act`` here is the other class's fix every second time. Said out
        # loud when the caller asked to be told; silent otherwise, which is
        # still better than the wrong action.
        pooled = pooled_classes(contrasts)
        if pooled is not None:
            if not abstain:
                return None
            return {
                "mode": "abstain",
                "suggested_action": None,
                "pooled": pooled,
                "why": _pooled_why(pooled),
            }
        from_contrast = _act_from_contrast(
            contrasts,
            tried,
            winners,
            regime_shift=regime_shift,
            regime_shift_at=regime_shift_at,
        )
        if from_contrast is not None:
            return from_contrast

    if winners and regime_shift:
        since = [w for w in winners if _won_since(w, regime_shift_at)]
        if since:
            best = since[0]
            return {
                "mode": "act",
                "suggested_action": best["action_key"],
                "why": f"regime shift suspected, but {best['action_key']} has won since it "
                       f"({best['won']}/{best['n']} on similar tasks here)",
            }
    if winners and not regime_shift:
        best = winners[0]
        return {
            "mode": "act",
            "suggested_action": best["action_key"],
            "why": f"{best['action_key']} won {best['won']}/{best['n']} on similar tasks here"
                   + (f" ({best['agents']} agents)" if int(best.get("agents", 0)) > 1 else ""),
        }
    lone = _lone_winner(tried, failed_now) if priors_are_local else None
    if lone is not None and (not regime_shift or _won_since(lone, regime_shift_at)):
        failed = ", ".join(f"{t['action_key']} {t['won']}/{t['n']}" for t in failed_now[:5])
        return {
            "mode": "act",
            "suggested_action": lone["action_key"],
            "why": (
                f"{lone['action_key']} is the only action that has worked on tasks like this "
                f"({lone['won']}/{lone['n']}); everything else tried here failed: {failed}"
            ),
        }
    if (
        have_candidates
        and top_hit_status == "validated"
        and not top_hit_recent_failure
        and not top_hit_shifted
        and not all_tried_failed
    ):
        # Only for a caller choosing among a fixed set of actions. Without
        # candidates there is no action to act *with*, and ``act`` on a bare
        # validated hit reads as "follow the top hit": grid v5 (2026-09-20)
        # measured it on a task whose terminal tool has no action enum —
        # ``act`` on 92% of episodes, and the agent followed the top hit over
        # the task's own constraints, 42% first-attempt failures against 6%
        # for the same protocol without the recommendation. The hit's
        # ``evidence_status`` already tells the agent it is validated.
        why = "top memory hit is validated by outcomes and has no recent failure"
        if regime_shift:
            why += "; a shift is suspected elsewhere on this entity, not in this hit's record"
        return {"mode": "act", "suggested_action": None, "why": why}
    shift_explores = regime_shift and priors_are_local
    if (all_tried_failed or shift_explores) and untried and tried:
        pick = untried[stable_bucket(agent_id, len(untried))]
        failed = ", ".join(f"{t['action_key']} {t['won']}/{t['n']}" for t in failed_now[:4])
        why = "regime shift suspected for tasks like this; " if shift_explores else ""
        if stopped:
            s = stopped[0]
            why += (f"{s['action_key']} won {s['won']}/{s['n']} on tasks like this but lost its last "
                    f"{_leading_losses(s)} — what worked here has stopped working; ")
        elif failed:
            why += f"tried and failed on similar tasks here: {failed}; "
        why += f"{len(untried)} untried — try {pick}"
        out: dict[str, Any] = {"mode": "explore", "suggested_action": pick, "untried": untried, "why": why}
        if stopped:
            out["stopped_working"] = [str(s["action_key"]) for s in stopped]
        return out
    if have_candidates and all_tried_failed_firm and not untried:
        failed = ", ".join(f"{t['action_key']} {t['won']}/{t['n']}" for t in losers[:6])
        return {
            "mode": "escalate",
            "suggested_action": None,
            "why": (
                f"every known action has failed on tasks like this: {failed}. "
                "If you have attempts left, try an action not listed; otherwise hand off"
            ),
        }
    if have_candidates and top_hit_status == "discredited" and not untried and tried:
        return {
            "mode": "escalate",
            "suggested_action": None,
            "why": "the only memory evidence is discredited and no candidate action is untried",
        }
    if abstain and not tried and not regime_shift:
        statuses = [s for s in (hit_statuses or ([top_hit_status] if top_hit_status else [])) if s]
        if statuses and all(s in _WEAK_STATUSES for s in statuses):
            counts: dict[str, int] = {}
            for s in statuses:
                counts[s] = counts.get(s, 0) + 1
            described = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
            return {
                "mode": "abstain",
                "suggested_action": None,
                "why": f"no action has been tried on similar tasks here and no hit is validated "
                       f"({described}); treat what follows as hints, not guidance",
            }
    return None


def guidance_strength(
    priors: Mapping[str, Any] | None,
    hit_statuses: Sequence[str] | None,
    *,
    regime_shift: bool = False,
    priors_are_local: bool | None = None,
) -> str:
    """How much the served context is worth acting on: ``strong`` when a hit is
    validated or an action has a winning record here; ``none`` when there is
    nothing or only untested / contested / discredited evidence; ``thin``
    otherwise (some evidence, none of it confirmed, or a regime shift in
    scope). Pure; the briefing and the SDK's ``Guidance`` carry the label.

    A winning record that :func:`pooled_classes` says is about two kinds of
    task does not make the guidance strong — it won on the other class half
    the time — but a validated hit still does: the hit's record is about the
    hit. Read only over local priors (*priors_are_local*, else
    :func:`priors_local` on the block); the entity-wide fallback always looks
    pooled and is never read for it."""
    tried = list((priors or {}).get("tried") or [])
    statuses = [s for s in (hit_statuses or []) if s]
    # The same reading of "winner" as recommend(): a record that lost its
    # newest RECENT_FAIL_STREAK takes is not one, whatever its lifetime ratio.
    winners = [
        t for t in tried
        if float(t.get("p", 0)) >= ACT_MIN_P and int(t.get("n", 0)) >= ACT_MIN_N and not _stopped_working(t)
    ]
    if regime_shift:
        return "thin" if (tried or statuses) else "none"
    local = priors_local(priors) if priors_are_local is None else priors_are_local
    if winners and local and pooled_classes(list((priors or {}).get("contrasts") or [])) is not None:
        winners = []
    if winners or "validated" in statuses:
        return "strong"
    if not tried and (not statuses or all(s in _WEAK_STATUSES for s in statuses)):
        return "none"
    return "thin"


def _recently_failing(prior: Mapping[str, Any]) -> bool:
    """The action lost its newest ``RECENT_FAIL_STREAK`` outcomes (all the record keeps)."""
    last = list(prior.get("last_3") or [])
    return len(last) >= RECENT_FAIL_STREAK and all(x == "lost" for x in last[:RECENT_FAIL_STREAK])


def turned(prior: Mapping[str, Any]) -> bool:
    """A validated action that has just stopped working: won at least
    ``REGIME_MIN_WINS`` times on tasks like this and lost its newest
    ``REGIME_TURN_STREAK`` takes. The action-level reading of
    :func:`amfs_core.evidence.regime_shifted` — one loss against a run of
    wins is forgiven, the second in a row is not — and one no rewrite of a
    lesson can erase, since it is read from the situation's outcomes."""
    last = list(prior.get("last_3") or [])
    return (
        int(prior.get("won", 0)) >= REGIME_MIN_WINS
        and len(last) >= REGIME_TURN_STREAK
        and all(x == "lost" for x in last[:REGIME_TURN_STREAK])
    )


def _stopped_working(prior: Mapping[str, Any]) -> bool:
    """Not a winner whatever the lifetime ratio: a losing streak
    (:func:`_recently_failing`) or a validated action that turned."""
    return _recently_failing(prior) or turned(prior)


def _turned_since(prior: Mapping[str, Any], contrast_weight: float) -> bool:
    """Whether B's record has turned against a contrast of *contrast_weight*:
    its newest take lost, and the weighted mass of its losses is at least
    ``POOLED_WEIGHT_RATIO`` of that weight. A record without ``lost_w`` (an
    older server) is read the old way — any newest loss turns it."""
    last = list(prior.get("last_3") or [])
    if not last or last[0] == "won":
        return False
    lost_w = prior.get("lost_w")
    if lost_w is None:
        return True
    return float(lost_w) >= POOLED_WEIGHT_RATIO * contrast_weight


def _leading_losses(prior: Mapping[str, Any]) -> int:
    n = 0
    for x in prior.get("last_3") or []:
        if x != "lost":
            break
        n += 1
    return n


def _lone_winner(
    tried: Sequence[Mapping[str, Any]], failed_now: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """The only tried action that has not failed on tasks like this, when it
    has won and its newest take is a win; else ``None``. Needs at least one
    failed alternative: a single 1/1 with nothing else tried is a hint, not
    the best of the known options."""
    standing = [t for t in tried if t not in failed_now]
    if len(standing) != 1 or len(tried) < 2:
        return None
    lone = standing[0]
    last = list(lone.get("last_3") or [])
    if int(lone.get("won", 0)) == 0 or not last or last[0] != "won":
        return None
    return lone


def plan_actions(
    priors: Mapping[str, Any] | None,
    recommendation: Mapping[str, Any] | None = None,
    *,
    agent_id: str = "",
    candidate_actions: Sequence[str] | None = None,
    priors_are_local: bool | None = None,
    limit: int = PLAN_MAX,
) -> list[str]:
    """The order to try actions in, for an agent with more than one attempt.

    ``suggested_action`` is one attempt's worth of advice. An agent with a
    budget of three that follows it and fails is then on its own, and what it
    does next is its instinct — which, on tasks like this, has already
    failed. The plan is the rest of the advice: the recommendation's action
    first; then the actions with a winning record here (not turned, not on a
    streak), best posterior first; then the ones that have won at least once
    and whose newest take won; then what the nearest contrasts resolved with;
    then the untried candidates, rotated by the agent's name so two agents on
    the same problem explore different ones first; and only then, to fill
    the budget, what has failed here. Never the same action twice. Capped at
    *limit*; empty when there is nothing to order.

    Measured on the ops-queue CI demo (2026-09-22): after a change, the three
    tasks of the affected class each cost three CI runs — the recommended
    action, then ``fix_code`` (already 0/2 on this exact situation), then a
    third guess. Two of the three would have been untried actions under this
    order, and the new fix was among five of them.
    """
    tried = list((priors or {}).get("tried") or [])
    untried = list((priors or {}).get("untried") or [])
    local = priors_local(priors) if priors_are_local is None else priors_are_local
    allowed = set(candidate_actions) if candidate_actions else None
    plan: list[str] = []

    def _add(key: Any) -> None:
        k = str(key or "")
        if k and k not in plan and (allowed is None or k in allowed) and len(plan) < limit:
            plan.append(k)

    if recommendation and recommendation.get("mode") in ("act", "explore"):
        _add(recommendation.get("suggested_action"))
    by_p = sorted(tried, key=lambda t: (-float(t.get("p", 0)), -int(t.get("n", 0))))
    losers = [
        t for t in by_p
        if (float(t.get("p", 1)) < EXPLORE_MAX_P and int(t.get("n", 0)) >= EXPLORE_MIN_N)
        or _stopped_working(t)
    ]
    failed_now = [t for t in by_p if t in losers or int(t.get("won", 0)) == 0]
    for t in by_p:
        if (
            float(t.get("p", 0)) >= ACT_MIN_P and int(t.get("n", 0)) >= ACT_MIN_N
            and not _stopped_working(t)
        ):
            _add(t.get("action_key"))
    for t in by_p:
        last = list(t.get("last_3") or [])
        if int(t.get("won", 0)) > 0 and t not in failed_now and last and last[0] == "won":
            _add(t.get("action_key"))
    if local:
        for c in (priors or {}).get("contrasts") or []:
            if float(c.get("weight") or 0.0) < CONTRAST_MIN_W:
                break
            resolved = str(c.get("resolved_with") or "")
            if resolved and all(str(t.get("action_key")) != resolved or t not in failed_now for t in by_p):
                _add(resolved)
    if untried:
        start = stable_bucket(agent_id, len(untried))
        for i in range(len(untried)):
            _add(untried[(start + i) % len(untried)])
    for t in by_p:
        if t in failed_now and not _stopped_working(t):
            _add(t.get("action_key"))
    for t in by_p:
        _add(t.get("action_key"))
    return plan


def pooled_classes(
    contrasts: Sequence[Mapping[str, Any]],
    *,
    min_weight: float = CONTRAST_MIN_W,
) -> dict[str, Any] | None:
    """Two kinds of task the query text does not tell apart, read from the
    near-identical contrasts contradicting each other *over time*.

    A contrast says "on this kind of task A failed and B resolved it". When
    another near-identical contrast says the opposite — B failed and A
    resolved it — one of two things is true: the rule flipped (a regime
    change: every old outcome says B, every new one says A), or the
    neighbourhood holds two classes of task that share a description and
    differ in the fix (the outcomes alternate: B, A, B). The order
    distinguishes them. A single reversal is read as a flip and left to the
    newest-wins rule; two or more reversals are a pooled neighbourhood, and
    no action can be recommended from it — the record is right about both
    classes and wrong about which one this is.

    Measured on the ops-queue CI demo (2026-09-21): a UI snapshot PR and a
    backend PR failing the same jest snapshot test differ by one path token,
    embed within 0.02 of each other, and ``act`` named the other class's fix
    on every third task — three first-try losses in twelve, each with the
    correct note for the exact symptom sitting in the same guidance.

    Only the rows that pit the two actions against each other are on the
    timeline — A resolving what B failed, B resolving what A failed. A's
    wins over some third action C say nothing about the A/B question, and
    counting them turned one flip plus one unrelated resolve into two
    reversals.

    Local priors only: the ``action_stats`` fallback records every outcome on
    the entity at similarity 1.0, so mixed kinds of task there always look
    near-identical and always contradict. Callers pass such a record through
    :func:`priors_local` before asking.

    Returns ``{"actions": [A, B], "sequence": [...resolved_with in time
    order...], "reversals": n, "outcome_refs": [...]}`` or ``None``.
    """
    near = [
        c for c in contrasts
        if float(c.get("weight") or 0.0) >= min_weight
        and c.get("resolved_with") and c.get("failed")
    ]
    if len(near) < 3:
        return None
    resolved_by: dict[str, list[Mapping[str, Any]]] = {}
    for c in near:
        resolved_by.setdefault(str(c["resolved_with"]), []).append(c)
    floor = datetime.min.replace(tzinfo=timezone.utc)

    def _over(rows: Sequence[Mapping[str, Any]], loser: str) -> list[Mapping[str, Any]]:
        return [c for c in rows if loser in [str(x) for x in (c.get("failed") or [])]]

    for a, a_rows in resolved_by.items():
        for b, b_rows in resolved_by.items():
            if b <= a:
                continue
            a_over_b, b_over_a = _over(a_rows, b), _over(b_rows, a)
            if not (a_over_b and b_over_a):
                continue
            # Two sides at comparable weight are one neighbourhood disagreeing
            # with itself. A side whose heaviest row sits well under the
            # other's is a neighbouring class the query merely resembles —
            # its rows are already discounted in the per-action record, and
            # the heavier side is the one about this task.
            heaviest = [max(float(c.get("weight") or 0.0) for c in rows) for rows in (a_over_b, b_over_a)]
            if min(heaviest) < POOLED_WEIGHT_RATIO * max(heaviest):
                continue
            ordered = sorted(a_over_b + b_over_a, key=lambda c: _as_dt(c.get("committed_at")) or floor)
            sequence = [str(c["resolved_with"]) for c in ordered]
            reversals = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])
            if reversals >= 2:
                return {
                    "actions": [a, b],
                    "sequence": sequence,
                    "reversals": reversals,
                    "outcome_refs": [c.get("outcome_ref") for c in ordered],
                }
    return None


def priors_local(priors: Mapping[str, Any] | None) -> bool:
    """Whether a priors block is about *this kind of task*.

    The server labels the block with its ``source``: ``similar_outcomes``
    (nearest outcomes by task embedding, re-weighted by
    :func:`neighbourhood_weights`) or ``action_stats`` (the entity's whole
    record, every row at similarity 1.0). Only the former can be read for
    contrasts or pooling; the latter can name a winner but not say what task
    it won on. A block with no label (a caller aggregating its own rows) is
    taken as local.
    """
    return (priors or {}).get("source") != "action_stats"


def _pooled_why(pooled: Mapping[str, Any]) -> str:
    a, b = pooled["actions"]
    seq = list(pooled.get("sequence") or [])
    return (
        f"outcomes on near-identical tasks here disagree: {a} resolved what {b} failed and "
        f"{b} resolved what {a} failed, alternating over time "
        f"({seq.count(a)}x {a}, {seq.count(b)}x {b}) — two kinds of task share this "
        "description; decide from what distinguishes this one, not from the action record"
    )


def _act_from_contrast(
    contrasts: Sequence[Mapping[str, Any]],
    tried: Sequence[Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
    *,
    regime_shift: bool,
    regime_shift_at: datetime | None,
) -> dict[str, Any] | None:
    """``act -> B`` from the nearest contrast pair, or ``None``.

    The pair must be near-identical to the query (``weight >=
    CONTRAST_MIN_W``); B's own record must not have turned since
    (:func:`_turned_since` — its newest take lost, and that loss weighs as
    much against the query as the contrast does: a loss on a distant
    neighbour does not turn a win on a task this near, and reading it as one
    recommended the sibling class's fix on the ops-queue demo); and under a
    regime shift the pair
    must postdate the shift, or it may itself be pre-change evidence. The
    pair yields to a per-action winner unless that winner is one of the
    actions the pair says failed: an established C that the contrast is not
    about keeps its recommendation; an established A that just failed on
    this kind of task does not.
    """
    if not contrasts:
        return None
    by_key = {str(t.get("action_key")): t for t in tried}
    for c in contrasts:
        if float(c.get("weight") or 0.0) < CONTRAST_MIN_W:
            break   # sorted heaviest first
        resolved = str(c.get("resolved_with") or "")
        failed = [str(a) for a in (c.get("failed") or [])]
        if not resolved or not failed:
            continue
        if regime_shift:
            at = _as_dt(c.get("committed_at"))
            if regime_shift_at is None or at is None or at <= (
                regime_shift_at if regime_shift_at.tzinfo
                else regime_shift_at.replace(tzinfo=timezone.utc)
            ):
                continue
        b = by_key.get(resolved)
        if b is not None and _turned_since(b, float(c.get("weight") or 0.0)):
            continue
        if winners and winners[0].get("action_key") not in failed:
            return None   # an established winner the pair is not about stands
        a_text = ", ".join(failed[:3])
        return {
            "mode": "act",
            "suggested_action": resolved,
            "why": (
                f"on a near-identical task here {a_text} failed and {resolved} resolved it"
                + (
                    f" ({b['won']}/{b['n']} overall)" if b is not None and int(b.get("n", 0)) > 1
                    else ""
                )
            ),
            "contrast": {
                "failed": failed,
                "resolved_with": resolved,
                "outcome_ref": c.get("outcome_ref"),
            },
        }
    return None


def render_priors(
    priors: Mapping[str, Any] | None,
    recommendation: Mapping[str, Any] | None,
    *,
    priors_are_local: bool | None = None,
    candidate_actions: Sequence[str] | None = None,
) -> str:
    """One compact block for an agent's context. Empty string when nothing to show.

    Contrasts — and the pooled-classes warning read from them — are shown only
    over local priors (*priors_are_local*, else :func:`priors_local` on the
    block): the entity-wide fallback is not known to be about this task.

    *candidate_actions*, when given, are the only actions the order may name:
    a retry asking over the actions it has not tried must not be told to try
    the one that just failed because the situation's record has it as a
    winner, whether the plan was computed here or sent by the server."""
    if not priors and not recommendation:
        return ""
    local = priors_local(priors) if priors_are_local is None else priors_are_local
    lines: list[str] = []
    tried = (priors or {}).get("tried") or []
    if tried:
        parts = [
            f"{t['action_key']} {t['won']}/{t['n']}"
            + ("" if int(t.get('agents', 0)) <= 1 else f" ({t['agents']} agents)")
            + (f", lost last {_leading_losses(t)}" if _stopped_working(t) and int(t.get("won", 0)) else "")
            for t in tried[:6]
        ]
        lines.append("Tried on similar tasks here: " + "; ".join(parts))
    contrasts = [
        c for c in ((priors or {}).get("contrasts") or [])
        if local and float(c.get("weight") or 0.0) >= CONTRAST_MIN_W
    ]
    pooled = pooled_classes(contrasts)
    if pooled is not None and not (recommendation or {}).get("pooled"):
        # Two "near-identical" lines that contradict each other read as a
        # coin toss; one line that names the contradiction reads as a warning.
        # (When the recommendation is the abstain that says the same, it says it.)
        lines.append(_pooled_why(pooled)[0].upper() + _pooled_why(pooled)[1:] + ".")
    for c in ([] if pooled is not None else contrasts[:2]):
        failed = ", ".join(str(a) for a in (c.get("failed") or [])[:3])
        if failed and c.get("resolved_with"):
            lines.append(
                f"On a near-identical task here {failed} failed and "
                f"{c['resolved_with']} resolved it."
            )
    allowed = set(candidate_actions) if candidate_actions else None
    untried = (priors or {}).get("untried") or []
    if allowed is not None:
        untried = [u for u in untried if u in allowed]
    if untried:
        lines.append("Not yet tried here: " + ", ".join(untried[:8]))
    if recommendation:
        mode = recommendation.get("mode")
        sug = recommendation.get("suggested_action")
        if sug and allowed is not None and sug not in allowed:
            # The suggestion is an action the run said it will not take — the
            # winner a retry has just tried, an action outside its tool. The
            # mode and the reason still stand; naming the action would hand
            # it back through the one line the plan filter below cannot reach.
            sug = None
        lines.append(f"Recommendation: {mode}" + (f" -> {sug}" if sug else "") + f". {recommendation.get('why', '')}".rstrip())
    # The imperative part. An agent with several attempts reads the record
    # above as history; what it needs is the order to try things in and what
    # not to spend an attempt on. Failed-here actions are named so the agent
    # does not re-try its instinct: on the ops-queue demo the second and third
    # attempts after a recommended action failed were actions already 0/n on
    # the exact situation.
    # Only behind an act or explore: an abstain (or no recommendation at all)
    # says the record here is not about this task — pooled classes, a
    # neighbourhood match — and an order drawn from it would be advice with
    # nothing behind it. Measured on the ops-queue demo (2026-09-22): a plan
    # rendered under no recommendation led with the untried candidates in a
    # stable order, and the agent read it as memory telling it what to do.
    mode = (recommendation or {}).get("mode")
    if mode not in ("act", "explore"):
        return "\n".join(lines)
    plan = list((recommendation or {}).get("plan") or []) or (
        plan_actions(
            priors, recommendation, priors_are_local=local, candidate_actions=candidate_actions,
        ) if priors else []
    )
    if allowed is not None:
        plan = [a for a in plan if a in allowed]
    stopped = [t for t in tried if _stopped_working(t) and int(t.get("won", 0)) > 0]
    failed = [
        t for t in tried
        if t not in stopped and int(t.get("won", 0)) == 0
    ]
    avoid = [
        f"{t['action_key']} ({t['won']}/{t['n']}, stopped working)" for t in stopped[:3]
    ] + [f"{t['action_key']} (0/{t['n']})" for t in failed[:5]]
    if avoid:
        lines.append("Do not spend an attempt on: " + "; ".join(avoid) + " — these failed on tasks like this.")
    if len(plan) > 1 or (plan and mode != "act"):
        lines.append(
            "Try in this order: " + " -> ".join(plan)
            + ". Take the first; if it fails, the next. Do not repeat an action that failed on this task."
        )
    return "\n".join(lines)


__all__ = [
    "ACTION_VALUE_MAX_CHARS",
    "CONTRAST_MIN_W",
    "PRIORS_NEIGHBOURHOOD_MIN_W",
    "PRIORS_NEIGHBOURHOOD_TAU",
    "neighbourhood_weights",
    "PRIORS_K",
    "PRIORS_MIN_SIMILARITY",
    "PRIORS_DAILY_DECAY",
    "ActionPrior",
    "action_key",
    "actions_taken",
    "aggregate_priors",
    "entity_paths_of",
    "guidance_strength",
    "plan_actions",
    "PLAN_MAX",
    "POOLED_WEIGHT_RATIO",
    "pooled_classes",
    "priors_local",
    "recommend",
    "REGIME_MIN_WINS",
    "REGIME_TURN_STREAK",
    "render_priors",
    "stable_bucket",
    "turned",
    "recorded_environment",
]
