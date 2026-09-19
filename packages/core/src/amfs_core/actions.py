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
        }


def aggregate_priors(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    candidate_actions: Sequence[str] | None = None,
    now: datetime | None = None,
    daily_decay: float = PRIORS_DAILY_DECAY,
) -> dict[str, Any]:
    """Fold similar past outcomes into per-action priors.

    Each item of ``outcomes`` is a row with ``actions_taken`` (as produced by
    :func:`actions_taken`), ``committed_at`` and ``agent_id``; ``similarity`` is
    optional and multiplies the weight. Returns ``{"tried": [...], "untried":
    [...], "n_outcomes": int, "contrasts": [...]}`` with ``tried`` sorted by
    posterior descending and, within ties, by evidence.

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
    priors: dict[str, ActionPrior] = {}
    contrasts: list[dict[str, Any]] = []
    # newest first so last_3 fills in time order
    rows = sorted(outcomes, key=lambda r: _as_dt(r.get("committed_at")) or now, reverse=True)
    for row in rows:
        at = _as_dt(row.get("committed_at"))
        age_days = max(0.0, (now - at).total_seconds() / 86400.0) if at else 0.0
        w = (daily_decay ** age_days) * float(row.get("similarity", 1.0) or 1.0)
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
    priors_are_local: bool = True,
) -> dict[str, Any] | None:
    """Decide ``act`` / ``explore`` / ``escalate`` from priors and the top hit.

    Returns ``None`` when there is nothing to say (no priors, no validated hit,
    no regime shift), so the payload stays unchanged for callers that gain
    nothing. ``escalate`` is only ever returned when the caller supplied
    ``candidate_actions`` — without knowing what could be tried, "everything
    failed" cannot be asserted, and a false escalate costs a task a retry
    would have won.

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
    """
    tried: list[Mapping[str, Any]] = list((priors or {}).get("tried") or [])
    untried: list[str] = list((priors or {}).get("untried") or [])
    have_candidates = bool(candidate_actions)

    winners = [t for t in tried if float(t.get("p", 0)) >= ACT_MIN_P and int(t.get("n", 0)) >= ACT_MIN_N]
    losers = [t for t in tried if float(t.get("p", 1)) < EXPLORE_MAX_P and int(t.get("n", 0)) >= EXPLORE_MIN_N]
    all_tried_failed = bool(tried) and len(losers) == len(tried)

    if priors_are_local:
        from_contrast = _act_from_contrast(
            list((priors or {}).get("contrasts") or []),
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
        failed = ", ".join(f"{t['action_key']} {t['won']}/{t['n']}" for t in losers[:4])
        why = "regime shift suspected for tasks like this; " if shift_explores else ""
        why += (f"tried and failed on similar tasks here: {failed}; " if failed else "")
        why += f"{len(untried)} untried — try {pick}"
        return {"mode": "explore", "suggested_action": pick, "untried": untried, "why": why}
    if have_candidates and all_tried_failed and not untried:
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
    return None


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
    CONTRAST_MIN_W``); B's own record must not have turned since (its newest
    take, if any, is a win — a B that lost more recently than it resolved
    this task is not the fix); and under a regime shift the pair must
    postdate the shift, or it may itself be pre-change evidence. The pair
    yields to a per-action winner unless that winner is one of the actions
    the pair says failed: an established C that the contrast is not about
    keeps its recommendation; an established A that just failed on this kind
    of task does not.
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
        if b is not None:
            last_3 = list(b.get("last_3") or [])
            if last_3 and last_3[0] != "won":
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


def render_priors(priors: Mapping[str, Any] | None, recommendation: Mapping[str, Any] | None) -> str:
    """One compact block for an agent's context. Empty string when nothing to show."""
    if not priors and not recommendation:
        return ""
    lines: list[str] = []
    tried = (priors or {}).get("tried") or []
    if tried:
        parts = [f"{t['action_key']} {t['won']}/{t['n']}" + ("" if int(t.get('agents', 0)) <= 1 else f" ({t['agents']} agents)")
                 for t in tried[:6]]
        lines.append("Tried on similar tasks here: " + "; ".join(parts))
    contrasts = [
        c for c in ((priors or {}).get("contrasts") or [])
        if float(c.get("weight") or 0.0) >= CONTRAST_MIN_W
    ]
    for c in contrasts[:2]:
        failed = ", ".join(str(a) for a in (c.get("failed") or [])[:3])
        if failed and c.get("resolved_with"):
            lines.append(
                f"On a near-identical task here {failed} failed and "
                f"{c['resolved_with']} resolved it."
            )
    untried = (priors or {}).get("untried") or []
    if untried:
        lines.append("Not yet tried here: " + ", ".join(untried[:8]))
    if recommendation:
        mode = recommendation.get("mode")
        sug = recommendation.get("suggested_action")
        lines.append(f"Recommendation: {mode}" + (f" -> {sug}" if sug else "") + f". {recommendation.get('why', '')}".rstrip())
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
    "recommend",
    "render_priors",
    "stable_bucket",
]
