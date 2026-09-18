"""Request and response models for the AMFS HTTP API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WriteRequest(BaseModel):
    entity_path: str
    key: str
    value: Any = None
    confidence: float = 1.0
    pattern_refs: list[str] = Field(default_factory=list)
    memory_type: str = "fact"
    shared: bool = True
    branch: str = "main"
    agent_id: str | None = None
    #: The caller's session, so stored provenance names the agent session that
    #: made the write rather than the server process that served it. Decision
    #: traces are keyed on this same id, and without it the two cannot be
    #: joined. Optional: older clients omit it and fall back to the server's.
    session_id: str | None = None
    #: Return what is already stored beside this entry, and how much of it has
    #: never been read back. Off by default because it costs an aggregate on the
    #: write path, and a caller that will not show the block should not pay for
    #: it. Computing it *here* is the point: a client that asks separately spends
    #: another round trip, which on the hosted surface is another billed op.
    include_scope: bool = False


class OutcomeRequest(BaseModel):
    outcome_ref: str
    outcome_type: str
    causal_entry_keys: list[str] | None = None
    causal_confidence: float = 1.0
    agent_id: str | None = None
    #: The request that triggered the decision. Supervised training needs the
    #: prompt side of the pair and the trace otherwise records only what was
    #: decided. Scanned for secrets before it is persisted.
    task_input: str | None = None
    #: The agent's answer. Opt-in separately from ``task_input`` because it is
    #: the larger disclosure and only the assistant-model dataset needs it.
    response_text: str | None = None
    #: The actions taken, which is what training predicts from ``task_input``.
    #: Already scanned by the client before they reach here, and scanned again on
    #: the way in, because this endpoint is reachable by anything with a key.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    #: The client's session metadata. Only ``attributes`` (the trace's dimension
    #: bag) and ``llm_calls`` (token/cost records) are read from it; a remote
    #: client has no other way to get either onto the trace this commit builds,
    #: since the trace is assembled here from the server's own handle.
    session_metadata: dict[str, Any] | None = None
    #: The caller will post its own trace to ``POST /api/v1/traces`` immediately
    #: after this call, so this endpoint must not seal one of its own. The trace
    #: assembled here comes off the server's shared handle — see the note above —
    #: so alongside the caller's it is both the poorer copy and a second sealed
    #: trace for one outcome. Only the SDK sets this, and only because its
    #: ``commit_outcome`` posts the trace on every path out of itself.
    trace_follows: bool = False
    #: Failed attempts that preceded the terminal outcome, oldest first:
    #: ``{attempt, outcome_type, causal_entry_keys, action_indices, summary}``.
    #: Each is applied to its own causal entries before ``outcome_type`` is
    #: applied to ``causal_entry_keys`` — inside this one outcome and trace.
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    #: Index into ``tool_calls`` of the action that produced the terminal
    #: outcome, when there was more than one.
    final_action_index: int | None = None
    #: ``entry_key -> version`` the agent read for ``causal_entry_keys``; the
    #: outcome is applied only where the key still carries that claim.
    causal_entry_versions: dict[str, int] | None = None
    #: What happened to each decisive action (``amfs_core.actions.actions_taken``
    #: rows). Derived here from ``tool_calls`` + ``attempts`` +
    #: ``final_action_index`` when absent, so a direct REST client gets priors
    #: without computing them.
    actions_taken: list[dict[str, Any]] | None = None
    #: The entities this outcome is about. Defaults to the causal keys' paths
    #: plus ``entity_path``.
    entity_paths: list[str] | None = None
    entity_path: str | None = None
    #: Optional label for the kind of task ("card-declined ticket").
    situation: str | None = None


class SearchRequest(BaseModel):
    query: str | None = None
    entity_path: str | None = None
    min_confidence: float = 0.0
    max_confidence: float | None = None
    agent_id: str | None = None
    since: datetime | None = None
    pattern_ref: str | None = None
    limit: int = 100
    sort_by: str = "confidence"
    branch: str = "main"
    depth: int = 3
    # When True (default) artifacts are demoted to the bottom of results; when
    # False they are excluded entirely.
    include_artifacts: bool = True
    # Widens entity_path to that path and everything beneath it. Off by default
    # so an existing caller's exact-match query keeps returning exactly what it
    # did; see amfs_core.scope for the covering rule.
    include_descendants: bool = False


class AggregateRequest(BaseModel):
    """Server-side aggregation over the records stored under one entity path.

    entity_path is required: an unscoped query strips shared @room/ paths, so
    an unscoped aggregate would silently return empty for every room. The
    reducer runs after visibility filtering so a caller can never aggregate over
    entries they cannot read.
    """

    entity_path: str
    op: str = "count"
    field: str | None = None
    group_by: str | None = None
    row_path: str | None = None
    branch: str = "main"


class RetrieveRequest(BaseModel):
    """Semantic (meaning-based) retrieval request.

    Ranks entries by embedding similarity to the query, blended with recency
    and confidence. entity_path is optional — when omitted, searches across
    everything the caller can see.
    """

    query: str
    entity_path: str | None = None
    min_confidence: float = 0.0
    limit: int = 10
    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    confidence_weight: float = 0.2
    branch: str = "main"
    # When True (default) artifacts (stored source files) are demoted but still
    # returned; when False they are excluded entirely from results.
    include_artifacts: bool = True
    #: Weight of the outcome-evidence term in the blend (see
    #: ``amfs_core.evidence.evidence_signal``): a validated entry outranks an
    #: untested one at equal confidence, a contested one falls behind both.
    evidence_weight: float = 0.15
    #: Discredited entries — a failure left them under the discredit
    #: threshold and nothing has lifted them since — are excluded from the
    #: ranked results by default. Set to rank them like any other entry.
    include_discredited: bool = False
    #: Append the discredited entries that *would* have ranked, flagged
    #: ``_avoid: true`` with ``_score: 0``, after the results. What not to do is
    #: knowledge too; opt-in because a client that does not know the flag would
    #: read them as hits.
    include_avoid: bool = False
    #: Shrink the result list when the top hit is validated and clearly ahead:
    #: a memory the record has confirmed does not need nine alternatives beside
    #: it in the prompt. Opt-in.
    adaptive_k: bool = False
    #: The calling agent, for the per-agent exploration assignment in the
    #: recommendation. A fleet whose members pass distinct ids spreads its
    #: search over the untried actions instead of all trying the same one.
    agent_id: str | None = None
    #: Append action priors — what was tried on the most similar past tasks
    #: about ``entity_path`` and how it went — and a recommendation
    #: (act / explore / escalate) as a trailing ``{"_meta": true, ...}`` element.
    #: Requires ``entity_path``. Opt-in, because a client that does not know
    #: the element would read it as a hit.
    include_priors: bool = False
    #: The actions the caller could take, as action keys (``tool:action``). With
    #: them the priors can say which are untried here and the recommendation
    #: can say ``escalate`` when every one has failed; without them it never does.
    candidate_actions: list[str] | None = None
    #: What kind of task this is, embedded for the priors' nearest-neighbour
    #: lookup in place of the query when given.
    situation: str | None = None
    #: Return the fields an agent acts on and drop the bookkeeping (score
    #: breakdown, importance dimensions, integrity fields, long values trimmed).
    #: Roughly a third of the tokens of the full payload.
    compact: bool = False


class ContextRequest(BaseModel):
    label: str
    summary: str
    source: str | None = None
    agent_id: str | None = None


class CreateAPIKeyRequest(BaseModel):
    name: str
    key_type: str = "agent"
    scopes: list[dict[str, str]] = Field(default_factory=lambda: [{"pattern": "*", "permission": "read_write"}])
    rate_limit_rpm: int = 120
    expires_at: datetime | None = None


# ──────────────────────────────────────────────────────────────────────
# Teams (Pro)
# ──────────────────────────────────────────────────────────────────────


class CreateTeamRequest(BaseModel):
    name: str
    slug: str
    description: str = ""


class UpdateTeamRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class AddTeamMemberRequest(BaseModel):
    email: str
    display_name: str = ""
    role: str = "developer"


class UpdateTeamMemberRequest(BaseModel):
    role: str | None = None
    display_name: str | None = None


# ──────────────────────────────────────────────────────────────────────
# Patterns (Pro)
# ──────────────────────────────────────────────────────────────────────


class RunPatternDetectionRequest(BaseModel):
    entity_path: str | None = None
    agent_id: str | None = None
    stale_days: int = 14
    orphan_days: int = 7
    pr_stale_days: int = 3
    similarity_threshold: float = 0.75
    incident_threshold: int = 2


# ──────────────────────────────────────────────────────────────────────
# Agent Snapshots
# ──────────────────────────────────────────────────────────────────────


class CreateSnapshotRequest(BaseModel):
    name: str
    description: str = ""
    snapshot_data: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Events (Shared Pool Ingestion)
# ──────────────────────────────────────────────────────────────────────


class EventRequest(BaseModel):
    """Direct shared-pool event ingestion without connector framework."""

    source: str
    entity_path: str
    key: str
    value: Any = None
    event_type: str = "generic"
