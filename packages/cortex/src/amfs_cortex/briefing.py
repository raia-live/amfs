"""BriefingService — reads pre-compiled digests and ranks by relevance.

Includes hot-context injection: the top priority-scored entries are
surfaced alongside digests so agents always see the most important
memories, not just the most recently written.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from amfs_core.actions import guidance_strength as _guidance_strength
from amfs_core.authority import rank_authors
from amfs_core.evidence import DISCREDIT_THRESHOLD as _DISCREDIT_THRESHOLD
from amfs_core.evidence import is_synthetic_key as _is_synthetic
from amfs_core.evidence import regime_shifted as _regime_shifted
from amfs_core.evidence import replacements_from_lessons as _replacements_from_lessons
from amfs_core.models import Digest, DigestType, MemoryEntry, SearchQuery
from amfs_core.models import preconditions_status as _preconditions_status

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter

logger = logging.getLogger(__name__)

_HOT_CONTEXT_LIMIT = 3
#: Priority rows fetched for ``hot_context``. More than the section shows,
#: because discredited, synthetic and procedure entries are dropped after the
#: fetch; without headroom each one would cost the section a slot.
_HOT_CONTEXT_SCAN_LIMIT = 12
#: Rows scanned per evidence query; the sections themselves are shorter.
_EVIDENCE_SCAN_LIMIT = 40
_EVIDENCE_SECTION_LIMIT = 5
_COMPACT_NARRATIVE_CHARS = 400
#: ``compact`` cuts each hot-context value here. A benchmark note runs 500-1500
#: characters and there are three of them on every task; the agent reads the
#: head to decide whether to act and ``amfs_read`` fetches the rest when it
#: does — and in the documented protocol a ``retrieve`` for the task follows
#: the briefing, which returns the same notes in full when they are relevant.
#: At 480 the three previews cost about a thousand characters per task, most
#: of it text the retrieve then repeated; 240 keeps the sentence that says
#: what the note is for. ``value_truncated`` marks the cut so nothing takes
#: the head for the whole.
_COMPACT_VALUE_CHARS = 240
#: ``tried_here`` rows a ``since`` delta keeps whatever their timestamp: an
#: action that has lost most of at least this many tries is a standing warning.
_STANDING_LOSS_N = 2
_STANDING_LOSS_P = 0.4
#: Digests loaded per briefing when the adapter can filter to the ones that
#: name the entity or agent asked about. Recency-ordered, so on an account
#: with more relevant digests than this the oldest are the ones not scored.
_DIGEST_SCAN_LIMIT = 200
#: Outcomes scanned for the ``tried_here`` section and rows it shows.
_ACTIONS_SCAN_LIMIT = 200
_ACTIONS_SECTION_LIMIT = 8


def _preview(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _memory_type(entry: MemoryEntry) -> str:
    mt = getattr(entry, "memory_type", None)
    return str(getattr(mt, "value", mt) or "fact")


def _procedure_goal(value: Any) -> str | None:
    """The ``goal`` of a structured procedure, or ``None`` for free text."""
    if isinstance(value, dict):
        goal = value.get("goal")
        return _preview(goal, 120) if isinstance(goal, str) and goal.strip() else None
    return None


def _hot_context_entries(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    """The priority rows that belong in ``hot_context``, in fetch order, capped
    at the section's size.

    Discredited entries are not "top priority" whatever their score says; they
    go in the discredited section with what replaced them. Synthetic lessons
    are folded into that section's ``replaced_by`` rather than shown as
    knowledge in their own right. Procedures have the ``procedures`` section:
    listing one here too would spend a slot meant for facts on a value that
    is already shown in full a few lines down.
    """
    kept = [
        e for e in entries
        if e.discredited_at is None
        and not _is_synthetic(e.key)
        and _memory_type(e) != "procedure"
    ]
    return kept[:_HOT_CONTEXT_LIMIT]

# Three authors, three keys each. This rides along on the one call every agent
# is told to make first, so it is paying for itself in context window on every
# task — enough to route, not enough to be worth skimming past.
_WHO_TO_ASK_LIMIT = 3
_WHO_TO_ASK_KEYS = 3


def _busiest_path_by_agent(
    stats: list[dict[str, Any]],
    entity_path: str,
    agent_ids: set[str],
) -> dict[str, str]:
    """The path in this subtree each agent has written to most.

    Ranking spans descendants, so a recommended author may have written
    nothing at the path that was asked about. The keys to offer them, and the
    path to address a read to, both have to come from where their work
    actually is. Ties go to the shallower path, so an agent that spread its
    work evenly is pointed at the more general topic.
    """
    prefix = entity_path.rstrip("/") + "/"
    best: dict[str, tuple[int, int, str]] = {}
    for row in stats:
        agent = str(row.get("agent_id") or "")
        if agent not in agent_ids:
            continue
        path = str(row.get("entity_path") or "")
        if path != entity_path and not path.startswith(prefix):
            continue
        rank = (int(row.get("entry_count") or 0), -path.count("/"), path)
        if agent not in best or rank > best[agent]:
            best[agent] = rank
    return {agent: rank[2] for agent, rank in best.items()}


class BriefingService:
    """Serves pre-compiled digests ranked by relevance for a given context."""

    def __init__(self, adapter: PostgresAdapter, namespace: str = "default") -> None:
        self._adapter = adapter
        self._namespace = namespace

    def briefing(
        self,
        entity_path: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        branch: str = "main",
        compact: bool = False,
        since: datetime | None = None,
        environment: Mapping[str, Any] | None = None,
    ) -> list[Digest]:
        """Get a ranked list of relevant digests for the given context.

        *environment* is the asking run's ``{"model", "agent_version",
        "runtime", "platform"}`` (``amfs_core.models.environment_of``). When
        given, each procedure in the lead digest is marked ``applicable``,
        ``not_applicable`` (an environment precondition it states is contradicted;
        such procedures move to ``procedures_not_applicable``) or ``unknown``
        (it constrains a key the run did not report). Without it every
        procedure is applicable, as before. The lead digest also carries
        ``guidance_strength`` — ``strong`` / ``thin`` / ``none`` — so an agent
        can tell a scope with validated knowledge from one with only untested
        notes before it reads either.

        Ranking (OSS, rule-based):
        1. Direct entity match (highest)
        2. Source digests for connectors with events on the same entity
        3. Agent briefs for agents that wrote to the same entity
        4. Recency-weighted
        5. Entry count weighted

        When *entity_path* is given, the lead entity digest also carries what
        the outcome record says about the scope — ``validated`` (entries every
        outcome confirmed), ``discredited`` (entries a failure gated, with what
        replaced them where known) and ``regime_shift`` (long-validated
        entries that have recently started failing, which is what a changed
        environment looks like from inside the memory).

        The lead digest also carries ``tried_here`` — what agents *did* on this
        entity and how it went, per action, from the outcome record — when the
        adapter keeps one.

        *compact* returns only that lead digest with the evidence sections and
        hot context, narrative trimmed: the shape an agent needs at the top of
        every task, at a fraction of the tokens of the full briefing.

        *since* keeps only what changed after that moment in the list sections
        (hot context, validated, discredited, tried_here): an agent that was
        briefed an hour ago pays for the delta, not the whole scope again.
        """
        if compact:
            limit = 1
        all_digests = self._list_digests(entity_path, agent_id, branch)
        if not all_digests and branch != "main":
            # Digests are compiled per branch and a short-lived branch (a repair
            # under review, a canary) rarely has its own. It reads memory as an
            # overlay on main, so main's digests describe what it sees; the
            # sections injected below are then re-read on the branch itself.
            all_digests = self._list_digests(entity_path, agent_id, "main")

        scored: list[tuple[float, Digest]] = []
        now = datetime.now(timezone.utc)

        for d in all_digests:
            score = self._score(d, entity_path, agent_id, now)
            if score > 0:
                d.staleness_ms = int((now - d.compiled_at).total_seconds() * 1000)
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        digests = [d for _, d in scored[:limit]]

        if entity_path:
            if digests:
                self._inject_hot_context(digests, entity_path, branch)
                if not compact:
                    self._inject_consolidation_notice(digests, entity_path, branch)
            else:
                self._inject_standalone_hot_context(digests, entity_path, branch)
            hit_statuses = self._inject_evidence_sections(
                digests, entity_path, branch, environment=environment
            )
            self._inject_action_sections(
                digests, entity_path, hit_statuses, environment=environment
            )
            if not compact:
                self._inject_who_to_ask(digests, entity_path, agent_id, branch)
            if compact:
                digests = self._compact(digests, entity_path)
            if since is not None:
                self._since(digests, entity_path, since)

        return digests

    def _list_digests(
        self, entity_path: str | None, agent_id: str | None, branch: str
    ) -> list[Digest]:
        """The digests worth scoring for this context.

        ``_score`` awards nothing to a digest that names neither the entity
        nor the agent asked about, so an adapter that can filter on those
        strings (Postgres) is asked for just that set, bounded. Loading every
        digest on the account was the briefing's cost under load: each row
        carries a compiled summary, and the count grows with every entity an
        agent writes to. Adapters without the filter get the plain call.
        """
        terms = [t for t in (entity_path, agent_id) if t]
        if terms:
            try:
                return self._adapter.list_digests(
                    namespace=self._namespace,
                    branch=branch,
                    relevant_to=terms,
                    limit=_DIGEST_SCAN_LIMIT,
                )
            except TypeError:
                pass
        return self._adapter.list_digests(namespace=self._namespace, branch=branch)

    def _inject_action_sections(
        self,
        digests: list[Digest],
        entity_path: str,
        hit_statuses: list[str] | None = None,
        *,
        environment: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach ``tried_here`` to the lead digest: per-action won/lost on this
        entity from the outcome record (``amfs_core.actions``), plus ``explore``
        — the actions with a thin record, where another try is information.
        Nothing is attached when the adapter keeps no outcome record.
        *hit_statuses* are the evidence statuses of the scope's entries, from
        the evidence sections, so ``guidance_strength`` can be re-rated with
        the action record included. *environment* is the caller's model /
        agent_version / runtime; outcomes recorded under another are
        down-weighted, as they are for ``retrieve`` priors."""
        hit_statuses = list(hit_statuses or [])
        lead = self._lead_digest(digests, entity_path)
        if lead is None:
            return
        stats = getattr(self._adapter, "action_stats", None)
        if not callable(stats):
            return
        try:
            rows = stats(entity_path, limit=_ACTIONS_SCAN_LIMIT)
        except Exception:
            logger.debug("action_stats failed for %s", entity_path, exc_info=True)
            return
        if not rows:
            return
        from amfs_core.actions import aggregate_priors, guidance_strength

        priors = aggregate_priors(rows, environment=environment)
        tried = priors.get("tried") or []
        if not tried:
            return
        # The evidence sections rated the scope on entries alone; a winning
        # action record is evidence too, so re-rate with the priors in hand.
        # ``action_stats`` is the entity's whole record, every row at
        # similarity 1.0: it can name a winner, but its contrasts are not
        # about this kind of task and must not be read for pooling.
        lead.summary["guidance_strength"] = guidance_strength(
            priors, hit_statuses, regime_shift=bool(lead.summary.get("regime_shift")),
            priors_are_local=False,
        )
        lead.summary["tried_here"] = [
            {
                "action": t["action_key"],
                "won": t["won"],
                "n": t["n"],
                "p": t["p"],
                "agents": t["agents"],
                "last_3": t["last_3"],
                "last_at": t["last_at"],
            }
            for t in tried[:_ACTIONS_SECTION_LIMIT]
        ]
        losing = [t for t in tried if t["n"] >= 2 and t["p"] < 0.4]
        thin = [t for t in tried if t["n"] < 2]
        if losing or thin:
            lead.summary["explore"] = {
                "avoid": [t["action_key"] for t in losing[:_ACTIONS_SECTION_LIMIT]],
                "thin_evidence": [t["action_key"] for t in thin[:_ACTIONS_SECTION_LIMIT]],
                "message": (
                    "tried_here is what agents did on this entity and how it went. "
                    "Do not repeat an action in `avoid` without a reason; prefer an "
                    "action with wins, or one not listed here at all."
                ),
            }

    @staticmethod
    def _standing(section: str, row: dict[str, Any]) -> bool:
        """A row a ``since`` delta must keep however old it is.

        The delta pays for what changed, but two kinds of row are warnings
        that stay in force: a discredited entry (with what replaced it), and
        an action that keeps losing here. An agent briefed an hour ago that
        asks for the delta would otherwise get a briefing with the "avoid"
        list missing, act on the stale fix, and fail on it again — the
        failure mode the sections exist to prevent. Both sections are capped,
        so keeping them costs a handful of rows.
        """
        if section == "discredited":
            return True
        if section == "tried_here":
            try:
                return int(row.get("n") or 0) >= _STANDING_LOSS_N and float(row.get("p") or 0.0) < _STANDING_LOSS_P
            except (TypeError, ValueError):
                return False
        return False

    @classmethod
    def _since(cls, digests: list[Digest], entity_path: str, since: datetime) -> None:
        """Trim the lead digest's list sections to what changed after ``since``,
        keeping the standing warnings (see ``_standing``)."""
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        for d in digests:
            if d.digest_type != DigestType.ENTITY or d.scope != entity_path:
                continue
            for section in ("hot_context", "validated", "discredited", "procedures", "tried_here"):
                rows = d.summary.get(section)
                if not isinstance(rows, list):
                    continue
                kept = []
                for row in rows:
                    if isinstance(row, dict) and cls._standing(section, row):
                        kept.append(row)
                        continue
                    stamps = []
                    for field in ("last_outcome_at", "discredited_at", "last_at", "written_at"):
                        raw = row.get(field)
                        if not raw:
                            continue
                        try:
                            at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        stamps.append(at if at.tzinfo else at.replace(tzinfo=timezone.utc))
                    if not stamps:
                        # No timestamp on the row: the section cannot say whether
                        # it changed, so keep it rather than hide it.
                        kept.append(row)
                        continue
                    if max(stamps) >= since:
                        kept.append(row)
                d.summary[section] = kept
            d.summary["since"] = since.isoformat()
            break

    def _search(self, query: SearchQuery, branch: str) -> list[MemoryEntry]:
        """``adapter.search`` with the branch when the adapter takes one.

        The Postgres adapter is branch-aware; the filesystem and S3 adapters
        are not and reject the keyword. Falling back keeps the briefing (hot
        context and evidence sections included) working over every adapter
        instead of silently producing a digest with no entries.
        """
        try:
            return self._adapter.search(query, branch=branch)
        except TypeError:
            return self._adapter.search(query)

    # ── Evidence sections ─────────────────────────────────────────────

    @staticmethod
    def _entry_brief(e: MemoryEntry) -> dict[str, Any]:
        """One hot-context row: what an agent needs to decide whether to act on
        an entry, including what the outcome record says about it."""
        return {
            "key": e.key,
            # Named because hot context now spans the scope: two entries can
            # share a key under different topics, and "which of these is
            # about deploys" is unanswerable from the key alone.
            "entity_path": e.entity_path,
            # Carried so a briefing can be booked as a real read: causal
            # lineage pins the version that was actually surfaced, and
            # without it the caller would have to re-read to find out.
            "version": e.version,
            "value": e.value,
            "confidence": round(e.confidence, 3),
            # Carried for the same reason as ``version``: the causal snapshot
            # a booked briefing writes has to be the one a direct read would
            # have written. Absent, ``record_surfaced`` falls back to "fact",
            # so every belief and experience surfaced by a briefing entered
            # the trace as a fact — a claim the entry never made, on the half
            # of the record a tuned model learns from. ``.value`` because
            # this dict is serialised into a digest summary.
            "memory_type": e.memory_type.value,
            "agent": e.provenance.agent_id,
            "outcome_count": e.outcome_count,
            "recall_count": e.recall_count,
            # The evidence behind the confidence number. ``evidence_status`` is
            # the word to read first: an untested 0.9 and a validated 0.9 are
            # different things to act on.
            "evidence_status": e.evidence_status,
            "success_count": e.success_count,
            "failure_count": e.failure_count,
            "last_outcome": e.last_outcome,
            "last_outcome_at": e.last_outcome_at.isoformat() if e.last_outcome_at else None,
            "written_at": e.provenance.written_at.isoformat() if e.provenance.written_at else None,
            # The record behind the label — ``p`` is the posterior success over
            # ``n`` outcomes — and how many distinct agents' successes stand
            # behind this claim. "Validated by 3 agents" outranks one agent's
            # repeated success.
            "posterior": {"p": e.posterior[0], "n": e.posterior[1]},
            "validators": len(e.validators or []),
        }

    def _lead_digest(self, digests: list[Digest], entity_path: str) -> Digest | None:
        for d in digests:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                return d
        return None

    def _inject_evidence_sections(
        self,
        digests: list[Digest],
        entity_path: str,
        branch: str,
        environment: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Attach ``validated``, ``discredited``, ``procedures``, ``regime_shift``
        and ``guidance_strength`` to the lead digest.

        Two bounded queries over the scope: the highest-priority entries (for
        what has been confirmed) and the lowest-confidence ones (for what has
        been discredited — a discredited entry is under the threshold by
        construction, so ``max_confidence`` finds them without a new column in
        the search API). Any failure leaves the digest as it was.

        Returns the evidence statuses of the entries seen, for the action
        sections to fold into ``guidance_strength``.
        """
        lead = self._lead_digest(digests, entity_path)
        if lead is None:
            return []
        try:
            top = self._search(
                SearchQuery(
                    entity_path=entity_path,
                    sort_by="priority",
                    limit=_EVIDENCE_SCAN_LIMIT,
                    include_artifacts=False,
                    include_descendants=True,
                ),
                branch=branch,
            )
            low = self._search(
                SearchQuery(
                    entity_path=entity_path,
                    max_confidence=_DISCREDIT_THRESHOLD,
                    sort_by="recency",
                    limit=_EVIDENCE_SCAN_LIMIT,
                    include_artifacts=False,
                    include_descendants=True,
                ),
                branch=branch,
            )
        except Exception:
            logger.debug("Evidence sections failed for %s", entity_path, exc_info=True)
            return []

        seen: dict[str, MemoryEntry] = {}
        for e in [*top, *low]:
            seen.setdefault(e.entry_key, e)
        entries = list(seen.values())

        validated = sorted(
            (
                e for e in entries
                if e.evidence_status == "validated"
                and e.success_count > 0
                and not _is_synthetic(e.key)
            ),
            key=lambda e: (e.success_count, e.confidence),
            reverse=True,
        )[:_EVIDENCE_SECTION_LIMIT]
        discredited = sorted(
            (e for e in entries if e.discredited_at is not None and not _is_synthetic(e.key)),
            key=lambda e: e.last_outcome_at or e.discredited_at or e.provenance.written_at,
            reverse=True,
        )[:_EVIDENCE_SECTION_LIMIT]
        # Contrast lessons name what resolved a task after a discredited entry
        # failed it; surface that beside the entry so the agent gets the
        # replacement, not just the warning.
        replacements = self._replacements_from_lessons(entries)

        shifted = [e for e in self._regime_shift(entries) if not _is_synthetic(e.key)]

        # Procedures are how to do the task, not facts about it, so they get a
        # section of their own and ``_hot_context_entries`` keeps them out of
        # the hot context. Validated first, then by confidence; discredited
        # ones are already in the section above and are not repeated here.
        procedures = sorted(
            (
                e for e in entries
                if _memory_type(e) == "procedure"
                and e.discredited_at is None
                and not _is_synthetic(e.key)
            ),
            key=lambda e: (e.success_count, e.confidence),
            reverse=True,
        )[:_EVIDENCE_SECTION_LIMIT]

        lead.summary["validated"] = [
            {
                "key": e.key,
                "entity_path": e.entity_path,
                "confidence": round(e.confidence, 3),
                "success_count": e.success_count,
                "last_outcome_at": e.last_outcome_at.isoformat() if e.last_outcome_at else None,
            }
            for e in validated
        ]
        lead.summary["discredited"] = [
            {
                "key": e.key,
                "entity_path": e.entity_path,
                "confidence": round(e.confidence, 3),
                "failure_count": e.failure_count,
                "success_count": e.success_count,
                "last_outcome": e.last_outcome,
                "discredited_at": e.discredited_at.isoformat() if e.discredited_at else None,
                "value_preview": _preview(e.value),
                "replaced_by": replacements.get(e.entry_key, []),
            }
            for e in discredited
        ]
        if procedures:
            applicable_rows: list[dict[str, Any]] = []
            not_applicable_rows: list[dict[str, Any]] = []
            for e in procedures:
                status, detail = _preconditions_status(e.value, environment)
                row = {
                    "key": e.key,
                    "entity_path": e.entity_path,
                    "confidence": round(e.confidence, 3),
                    "evidence_status": e.evidence_status,
                    "success_count": e.success_count,
                    "failure_count": e.failure_count,
                    "goal": _procedure_goal(e.value),
                    "value_preview": _preview(e.value),
                    "written_at": e.provenance.written_at.isoformat(),
                    "last_outcome_at": (
                        e.last_outcome_at.isoformat() if e.last_outcome_at else None
                    ),
                    "applicability": status,
                }
                if detail:
                    row["applicability_detail"] = detail
                if status == "not_applicable":
                    not_applicable_rows.append(row)
                else:
                    applicable_rows.append(row)
            if applicable_rows:
                lead.summary["procedures"] = applicable_rows
            if not_applicable_rows:
                # Kept visible, apart: the agent should know a way exists and
                # why it does not apply here, not just fail to find it.
                lead.summary["procedures_not_applicable"] = not_applicable_rows
        statuses = [e.evidence_status for e in entries if not _is_synthetic(e.key)]
        lead.summary["guidance_strength"] = _guidance_strength(
            None, statuses, regime_shift=bool(shifted)
        )
        if shifted:
            lead.summary["regime_shift"] = {
                "suspected": True,
                "entries": [
                    {
                        "key": e.key,
                        "entity_path": e.entity_path,
                        "success_count": e.success_count,
                        "failure_count": e.failure_count,
                        "confidence": round(e.confidence, 3),
                    }
                    for e in shifted
                ],
                "message": (
                    f"{len(shifted)} previously validated "
                    f"entr{'y' if len(shifted) == 1 else 'ies'} "
                    "in this scope started failing recently. Something about the "
                    "environment has likely changed; verify before reusing them and "
                    "prefer entries validated since."
                ),
            }
        return statuses

    @staticmethod
    def _regime_shift(entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Entries that were validated repeatedly and whose record no longer
        supports acting on them: the signature of a rule that used to work.
        The predicate is ``amfs_core.evidence.regime_shifted``, shared with
        retrieve's ``regime_shift`` flag so the two surfaces never disagree
        about whether the world changed."""
        out = [e for e in entries if _regime_shifted(e)]
        out.sort(key=lambda e: (e.failure_count, e.success_count), reverse=True)
        return out[:_EVIDENCE_SECTION_LIMIT]

    @staticmethod
    def _replacements_from_lessons(entries: list[MemoryEntry]) -> dict[str, list[str]]:
        # Kept as a method for callers that patch or call it here; the logic
        # lives in amfs_core.evidence so the retrieve avoid list reads the
        # same lessons the same way.
        return _replacements_from_lessons(entries)

    def _compact(self, digests: list[Digest], entity_path: str) -> list[Digest]:
        """The lead digest only, with the sections an agent acts on."""
        lead = self._lead_digest(digests, entity_path)
        if lead is None:
            return digests[:1]
        keep = (
            "narrative", "hot_context", "validated", "discredited", "procedures",
            "procedures_not_applicable", "regime_shift", "tried_here", "explore",
            "guidance_strength",
        )
        summary = {k: lead.summary[k] for k in keep if k in lead.summary}
        narrative = summary.get("narrative")
        if isinstance(narrative, str) and len(narrative) > _COMPACT_NARRATIVE_CHARS:
            summary["narrative"] = narrative[:_COMPACT_NARRATIVE_CHARS].rstrip() + "…"
        hot = summary.get("hot_context")
        if isinstance(hot, list):
            summary["hot_context"] = [
                self._compact_hot_row(row) if isinstance(row, dict) else row for row in hot
            ]
        lead.summary = summary
        return [lead]

    @staticmethod
    def _compact_hot_row(row: dict[str, Any]) -> dict[str, Any]:
        """A hot-context row at compact size: the value cut to its head and the
        counters an agent does not act on dropped. The fields lineage needs to
        book the row as a read (key, entity_path, version, confidence,
        memory_type, agent, evidence counts, written_at for ``since``) stay."""
        out = {k: v for k, v in row.items() if k not in ("recall_count", "outcome_count", "posterior")}
        value = out.get("value")
        if isinstance(value, str) and len(value) > _COMPACT_VALUE_CHARS:
            out["value"] = value[: _COMPACT_VALUE_CHARS - 1].rstrip() + "…"
            out["value_truncated"] = True
        return out

    def _inject_who_to_ask(
        self,
        digests: list[Digest],
        entity_path: str,
        agent_id: str | None,
        branch: str,
    ) -> None:
        """Name the agents worth asking about this entity, and what to read.

        Injected at serve time rather than compiled into the digest, for two
        reasons that both rule compilation out. Authority depends on recency,
        and the worker recompiles on a debounce, so a compiled ranking is stale
        by construction. And this names *other* agents, so it has to be
        filtered per caller — a compiled digest is shared by everyone who asks
        for it, which is the wrong granularity for a disclosure surface.

        The caller-side visibility filter is the control that enforces the
        second point; see the http-server's briefing handler.
        """
        try:
            stats = self._adapter.agent_entity_stats(entity_path=entity_path)
        except Exception:
            logger.debug("who_to_ask injection failed for %s", entity_path, exc_info=True)
            return

        # Descendants are rolled up because the query already scopes by prefix,
        # and because an agent that owns `<path>/tokens` is exactly who you
        # want named when you ask about `<path>`.
        ranked = rank_authors(
            entity_path,
            stats=stats,
            limit=_WHO_TO_ASK_LIMIT,
            include_descendants=True,
        )

        # Asking an agent to read from itself is noise, not routing. Drop the
        # caller only after ranking, so its own writes still set the share
        # denominator and a colleague's contribution is not overstated.
        if agent_id:
            ranked = [a for a in ranked if a.agent_id != agent_id]
        if not ranked:
            return

        # Where each author's keys actually live, which is not necessarily the
        # path that was asked about now that the ranking spans the subtree.
        source_paths = _busiest_path_by_agent(
            stats, entity_path, {a.agent_id for a in ranked}
        )
        top_keys = self._top_keys_by_agent(source_paths, branch)

        block = []
        for author in ranked:
            source = source_paths.get(author.agent_id, entity_path)
            keys = top_keys.get(author.agent_id, [])
            item: dict[str, Any] = {
                "agent_id": author.agent_id,
                "reason": author.reason,
                "entry_count": author.entry_count,
                "share": round(author.share, 3),
                "validated_outcomes": author.validated_outcomes,
                # Named because it can sit below the path asked about, and a
                # reader wanting a key other than the first one needs to know
                # where to look for it.
                "entity_path": source,
                "top_keys": keys,
            }
            # The literal call closes the "I know who, but not what to read"
            # gap that makes a bare recommendation useless. It has to name the
            # path the key is stored under: addressed to the parent, the read
            # finds nothing and the recommendation is worse than none.
            if keys:
                item["call"] = (
                    f'amfs_read_from("{author.agent_id}", "{source}", "{keys[0]}")'
                )
            block.append(item)

        for d in digests:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                d.summary["who_to_ask"] = block
                return

        # Nothing compiled for this path to hang the block on — the briefing
        # came back with digests scoped elsewhere, or with agent briefs only.
        # Carry the recommendations on a digest of our own instead of dropping
        # them, the same way standalone hot context does when the compiled
        # digest is missing entirely.
        digests.append(Digest(
            digest_type=DigestType.ENTITY,
            scope=entity_path,
            summary={
                "narrative": f"No compiled digest yet for {entity_path}.",
                "who_to_ask": block,
            },
            entry_count=0,
            source_agents=[],
            compiled_at=datetime.now(timezone.utc),
            namespace=self._namespace,
            branch=branch,
        ))

    def _top_keys_by_agent(
        self,
        source_paths: dict[str, str],
        branch: str,
    ) -> dict[str, list[str]]:
        """Highest-priority keys each named agent wrote, on its own path.

        One search per agent, as before. The path varies per agent because
        ``search`` matches ``entity_path`` exactly, so asking about the parent
        returns nothing for an author whose work sits in a child of it.
        """
        result: dict[str, list[str]] = {}
        for aid, path in source_paths.items():
            try:
                entries = self._search(
                    SearchQuery(
                        entity_path=path,
                        agent_id=aid,
                        sort_by="priority",
                        limit=_WHO_TO_ASK_KEYS,
                        include_artifacts=False,
                    ),
                    # Without this the lookup silently reads main while the
                    # rest of the briefing is on the caller's branch, and the
                    # recommendation names an author but no key to read.
                    branch=branch,
                )
            except Exception:
                logger.debug("who_to_ask key lookup failed for %s", aid, exc_info=True)
                continue
            result[aid] = [e.key for e in entries]
        return result

    def _inject_hot_context(
        self,
        digests: list[Digest],
        entity_path: str,
        branch: str,
    ) -> None:
        """Attach top priority-scored entries to the lead entity digest.

        This gives agents the equivalent of soft attention over the memory
        store: the most important entries are always surfaced, not just those
        mentioned in the compiled digest summary.
        """
        try:
            entries = self._search(
                SearchQuery(
                    entity_path=entity_path,
                    sort_by="priority",
                    limit=_HOT_CONTEXT_SCAN_LIMIT,
                    # Working files shouldn't dominate the "what you know" context
                    # an agent reads at task start.
                    include_artifacts=False,
                    # Knowledge is written at "<repo>/deploy", never at "<repo>",
                    # so an exact match on a scope a caller derived rather than
                    # was handed reaches none of it. This is the same widening
                    # _inject_who_to_ask already does via rank_authors.
                    include_descendants=True,
                ),
                branch=branch,
            )
        except Exception:
            logger.debug("Hot-context injection failed for %s", entity_path, exc_info=True)
            return

        entries = _hot_context_entries(entries)
        if not entries:
            return
        hot_entries = [self._entry_brief(e) for e in entries]

        for d in digests:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                d.summary["hot_context"] = hot_entries
                break

    def _inject_standalone_hot_context(
        self,
        digests: list[Digest],
        entity_path: str,
        branch: str,
    ) -> None:
        """Create a minimal digest with hot-context when no compiled digest exists.

        Ensures agents always get priority-scored entries even for entities
        that haven't been compiled yet.
        """
        try:
            entries = self._search(
                SearchQuery(
                    entity_path=entity_path,
                    sort_by="priority",
                    limit=_HOT_CONTEXT_SCAN_LIMIT,
                    # Working files shouldn't dominate the "what you know" context
                    # an agent reads at task start.
                    include_artifacts=False,
                    # As in _inject_hot_context: the scope is a prefix, and an
                    # uncompiled entity is exactly the case where the caller
                    # named a repo root rather than a topic under it.
                    include_descendants=True,
                ),
                branch=branch,
            )
        except Exception:
            return

        if not entries:
            return

        # The digest is created whenever the scope holds *anything*, not only
        # when something qualifies for the hot context: this is the lead digest
        # the evidence sections attach to, and a scope whose knowledge is all
        # procedures (or all discredited) still has a ``procedures`` or
        # ``discredited`` section to show. Without it the agent would be
        # briefed with an empty list.
        fetched = len(entries)
        entries = _hot_context_entries(entries)
        hot_entries = [self._entry_brief(e) for e in entries]
        narrative = f"No compiled digest yet for {entity_path}. " + (
            "Showing top priority entries."
            if hot_entries
            else "See the evidence sections below."
        )

        digests.append(Digest(
            digest_type=DigestType.ENTITY,
            scope=entity_path,
            summary={
                "narrative": narrative,
                "hot_context": hot_entries,
            },
            entry_count=fetched,
            source_agents=[],
            compiled_at=datetime.now(timezone.utc),
            namespace=self._namespace,
            branch=branch,
        ))

    def _inject_consolidation_notice(
        self,
        digests: list[Digest],
        entity_path: str,
        branch: str,
    ) -> None:
        """Surface pending consolidation branches for this entity."""
        try:
            branches = self._adapter.list_branches(
                namespace=self._namespace,
                status="active",
            )
            pending = [
                b for b in branches
                if b.name.startswith(f"cortex/consolidation/{entity_path}/")
            ]
        except Exception:
            logger.debug("Consolidation notice injection failed", exc_info=True)
            return

        if not pending:
            return

        notice = {
            "pending_proposals": len(pending),
            "branch_names": [b.name for b in pending[:5]],
            "message": (
                f"There {'is' if len(pending) == 1 else 'are'} {len(pending)} pending "
                f"consolidation proposal{'s' if len(pending) != 1 else ''} for this entity. "
                f"Review at /cortex/proposals."
            ),
        }

        for d in digests:
            if d.digest_type == DigestType.ENTITY and d.scope == entity_path:
                d.summary["consolidation_notice"] = notice
                break

    def _score(
        self,
        digest: Digest,
        entity_path: str | None,
        agent_id: str | None,
        now: datetime,
    ) -> float:
        score = 0.0

        if entity_path:
            if digest.digest_type == DigestType.ENTITY and digest.scope == entity_path:
                score += 100.0
            elif digest.digest_type == DigestType.CONNECTION_MAP and digest.scope == entity_path:
                score += 80.0
            elif digest.digest_type == DigestType.SOURCE:
                touched = digest.summary.get("entities_touched", [])
                if entity_path in touched:
                    score += 60.0
            elif digest.digest_type == DigestType.AGENT_BRIEF:
                entities = digest.summary.get("entities_written", [])
                if entity_path in entities:
                    score += 40.0
            elif digest.digest_type == DigestType.CONNECTION_MAP:
                connected = digest.summary.get("connected_entities", [])
                if entity_path in connected:
                    score += 30.0

        if agent_id:
            if digest.digest_type == DigestType.AGENT_BRIEF and digest.scope == agent_id:
                score += 100.0
            elif digest.digest_type == DigestType.ENTITY:
                agents = digest.summary.get("agents", [])
                if agent_id in agents:
                    score += 50.0

        if score == 0:
            return 0.0

        age_hours = max((now - digest.compiled_at).total_seconds() / 3600, 0.01)
        recency_boost = min(10.0 / age_hours, 20.0)
        score += recency_boost

        entry_boost = min(digest.entry_count * 0.5, 15.0)
        score += entry_boost

        score += digest.anticipation_score * 30.0

        return score
