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
PRIORS_MIN_SIMILARITY = 0.75
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
    environment: Mapping[str, Any] | None = None,
    env_mismatch_weight: float = ENV_MISMATCH_WEIGHT,
) -> dict[str, Any]:
    """Fold similar past outcomes into per-action priors.

    Each item of ``outcomes`` is a row with ``actions_taken`` (as produced by
    :func:`actions_taken`), ``committed_at`` and ``agent_id``; ``similarity`` is
    optional and multiplies the weight. Returns ``{"tried": [...], "untried":
    [...], "n_outcomes": int}`` with ``tried`` sorted by posterior descending and,
    within ties, by evidence.

    When *environment* is given (``{"model": ..., "runtime": ...}``, see
    ``amfs_core.models.environment_of``), a row whose ``session_metadata`` (or
    ``environment``) names a different value for any of those keys is weighted
    by *env_mismatch_weight*: what won under another runtime is evidence, but
    weaker. Rows that report no environment are unaffected, and so is every
    caller that passes none — the default output is unchanged.
    """
    now = now or datetime.now(timezone.utc)
    env = {k: str(v).strip() for k, v in (environment or {}).items() if v}
    priors: dict[str, ActionPrior] = {}
    # newest first so last_3 fills in time order
    rows = sorted(outcomes, key=lambda r: _as_dt(r.get("committed_at")) or now, reverse=True)
    for row in rows:
        at = _as_dt(row.get("committed_at"))
        age_days = max(0.0, (now - at).total_seconds() / 86400.0) if at else 0.0
        w = (daily_decay ** age_days) * float(row.get("similarity", 1.0) or 1.0)
        if env:
            w *= _env_match(row, env, env_mismatch_weight)
        agent = str(row.get("agent_id") or "")
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
    tried = sorted(priors.values(), key=lambda p: (-p.p, -p.n, p.action_key))
    tried_keys = {p.action_key for p in tried}
    untried = [a for a in (candidate_actions or []) if a and a not in tried_keys]
    return {
        "tried": [p.as_dict() for p in tried],
        "untried": untried,
        "n_outcomes": len(rows),
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
    regime_shift: bool = False,
    regime_shift_at: datetime | None = None,
    abstain: bool = False,
    hit_statuses: Sequence[str] | None = None,
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

    A regime shift skips the winning priors, because their wins may predate the
    change — except a winner whose latest take was *after* the shift and won
    (``regime_shift_at`` against the prior's ``last_at``, and its newest
    ``last_3`` entry). That is the replacement the shift called for, already
    found; sending the agent to explore past it would re-learn what the record
    already knows.
    """
    tried: list[Mapping[str, Any]] = list((priors or {}).get("tried") or [])
    untried: list[str] = list((priors or {}).get("untried") or [])
    have_candidates = bool(candidate_actions)

    winners = [t for t in tried if float(t.get("p", 0)) >= ACT_MIN_P and int(t.get("n", 0)) >= ACT_MIN_N]
    losers = [t for t in tried if float(t.get("p", 1)) < EXPLORE_MAX_P and int(t.get("n", 0)) >= EXPLORE_MIN_N]
    all_tried_failed = bool(tried) and len(losers) == len(tried)

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
    if top_hit_status == "validated" and not top_hit_recent_failure and not regime_shift and not all_tried_failed:
        return {
            "mode": "act",
            "suggested_action": None,
            "why": "top memory hit is validated by outcomes and has no recent failure",
        }
    if (all_tried_failed or regime_shift) and untried:
        pick = untried[stable_bucket(agent_id, len(untried))]
        failed = ", ".join(f"{t['action_key']} {t['won']}/{t['n']}" for t in losers[:4])
        why = "regime shift suspected for this entity; " if regime_shift else ""
        why += (f"tried and failed here: {failed}; " if failed else "")
        why += f"{len(untried)} untried — try {pick}"
        return {"mode": "explore", "suggested_action": pick, "untried": untried, "why": why}
    if have_candidates and all_tried_failed and not untried:
        failed = ", ".join(f"{t['action_key']} {t['won']}/{t['n']}" for t in losers[:6])
        return {
            "mode": "escalate",
            "suggested_action": None,
            "why": f"every candidate action has failed on similar tasks here: {failed}",
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
) -> str:
    """How much the served context is worth acting on: ``strong`` when a hit is
    validated or an action has a winning record here; ``none`` when there is
    nothing or only untested / contested / discredited evidence; ``thin``
    otherwise (some evidence, none of it confirmed, or a regime shift in scope).
    Pure; the briefing and the SDK's ``Guidance`` carry the label."""
    tried = list((priors or {}).get("tried") or [])
    statuses = [s for s in (hit_statuses or []) if s]
    winners = [t for t in tried if float(t.get("p", 0)) >= ACT_MIN_P and int(t.get("n", 0)) >= ACT_MIN_N]
    if regime_shift:
        return "thin" if (tried or statuses) else "none"
    if winners or "validated" in statuses:
        return "strong"
    if not tried and (not statuses or all(s in _WEAK_STATUSES for s in statuses)):
        return "none"
    return "thin"


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
    "PRIORS_K",
    "PRIORS_MIN_SIMILARITY",
    "PRIORS_DAILY_DECAY",
    "ActionPrior",
    "action_key",
    "actions_taken",
    "aggregate_priors",
    "entity_paths_of",
    "guidance_strength",
    "recommend",
    "render_priors",
    "stable_bucket",
    "recorded_environment",
]
