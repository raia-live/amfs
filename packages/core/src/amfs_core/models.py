"""Pydantic models for AMFS memory entries, outcomes, and configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    """Classification of memory entries for type-specific behavior.

    Facts are objective and stable. Beliefs are subjective and decay faster.
    Experiences are append-only records of agent actions. Procedures are
    *how to do a task*: an ordered set of steps, with what to check before
    acting and what to do when a step fails. They are what the repair loop
    ships when a failure was about method rather than fact, so they decay
    slowest and the briefing lists them in their own section.
    """

    FACT = "fact"
    BELIEF = "belief"
    EXPERIENCE = "experience"
    PROCEDURE = "procedure"


class ProvenanceTier(int, Enum):
    """Quality tier derived from how a memory was created and validated.

    Tier 1 (highest): written by a production agent with outcome validation.
    Tier 4 (lowest): manually seeded, no empirical validation.
    """

    PRODUCTION_VALIDATED = 1
    PRODUCTION_OBSERVED = 2
    DEVELOPMENT = 3
    MANUAL = 4


class MemoryTier(int, Enum):
    """Hierarchical memory tier for prioritized retrieval.

    Hot entries are searched first; archive is only accessed when active
    tiers don't satisfy the query.
    """

    HOT = 1
    WARM = 2
    ARCHIVE = 3


class OutcomeType(str, Enum):
    """Types of outcomes that can affect memory confidence."""

    SUCCESS = "success"                      # confidence *= 1.03 (reinforce)
    MINOR_FAILURE = "minor_failure"          # confidence *= 0.92 (erode)
    FAILURE = "failure"                      # confidence *= 0.90 (erode)
    CRITICAL_FAILURE = "critical_failure"    # confidence *= 0.85 (erode)

    # Backward-compatible aliases (deprecated)
    CLEAN_DEPLOY = "clean_deploy"
    REGRESSION = "regression"
    P2_INCIDENT = "p2_incident"
    P1_INCIDENT = "p1_incident"


# Multipliers applied to confidence when an outcome is committed.
#
# Semantics (as of the confidence-direction fix): a SUCCESS *reinforces*
# confidence (>1.0) and failures *erode* it (<1.0), which matches the intuitive
# meaning of the field — a memory that keeps leading to good outcomes should be
# trusted more, not less. Failures move confidence more than a single success,
# so trust is easy to lose and slow to rebuild. The result is always clamped to
# [0.0, 1.0] via ``clamp_confidence`` (a success can never push past certainty,
# and repeated failures floor at zero rather than going negative).
OUTCOME_MULTIPLIERS: dict[OutcomeType | str, float] = {
    OutcomeType.CRITICAL_FAILURE: 0.85,
    OutcomeType.FAILURE: 0.90,
    OutcomeType.MINOR_FAILURE: 0.92,
    OutcomeType.SUCCESS: 1.03,
    # Legacy values
    OutcomeType.P1_INCIDENT: 0.85,
    OutcomeType.P2_INCIDENT: 0.90,
    OutcomeType.REGRESSION: 0.92,
    OutcomeType.CLEAN_DEPLOY: 1.03,
}

# Confidence is a probability-like quantity in [0, 1]. Outcome back-propagation
# multiplies by the factors above, so results must be clamped: successes saturate
# at 1.0 (never exceed certainty) and failures floor at 0.0 (never go negative).


def clamp_confidence(value: float) -> float:
    """Clamp a confidence value into the valid [0.0, 1.0] range."""
    return max(0.0, min(1.0, float(value)))

# Beliefs are penalised more by regressions and decay faster. Procedures are
# the slowest: a way of doing a task stays valid until an outcome says it
# does not, and outcomes — not age — are what should retire one.
MEMORY_TYPE_DECAY_MULTIPLIERS: dict[MemoryType, float] = {
    MemoryType.FACT: 1.0,
    MemoryType.BELIEF: 0.5,
    MemoryType.EXPERIENCE: 1.5,
    MemoryType.PROCEDURE: 2.0,
}

#: Keys a procedure's value is expected to carry when it is a dict. The
#: quality evaluator reports what is missing; nothing rejects the write.
PROCEDURE_REQUIRED_FIELDS: tuple[str, ...] = ("goal", "steps")
PROCEDURE_OPTIONAL_FIELDS: tuple[str, ...] = (
    "preconditions",
    "on_failure",
    "verify",
    # What holds after the procedure ran, so a later procedure can rely on it.
    "effects",
    # The procedures this one chains, as ``{"ref": "<entity>/<key>", "version": n}``.
    "depends_on",
    # Trace or outcome ids the procedure was distilled from.
    "evidence",
)

#: Keys of a ``preconditions`` dict that name the run's environment rather than
#: the task. ``{"runtime": "python3.12"}`` says the procedure applies only there;
#: a string precondition ("the repo has a lockfile") is about the task and is
#: matched by nobody but the agent.
PROCEDURE_ENVIRONMENT_PRECONDITIONS: tuple[str, ...] = (
    "model", "agent_version", "runtime", "platform",
)


def procedure_issues(value: Any) -> list[str]:
    """What a procedure value is missing, as short issue codes.

    A procedure is a dict with a ``goal`` and a non-empty ``steps`` list (each
    step a non-blank string or a dict with a non-blank ``action`` and, optionally,
    an ``action_key`` naming the tool call the step expects — the hook adherence
    and step-level attribution key on), optionally ``preconditions``,
    ``on_failure``, ``verify``, ``effects``, ``depends_on`` and ``evidence``. The
    optional fields are validated for shape only when present: ``effects`` and
    ``evidence`` are lists of strings, ``depends_on`` a list of ``{"ref", "version"}``
    dicts. A plain string is accepted when it reads as a numbered or bulleted
    list of at least two steps; anything else is ``not_structured``. Pure; used
    by ``amfs_core.quality``.
    """
    if isinstance(value, dict):
        out: list[str] = []
        goal = value.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            out.append("missing_goal")
        steps = value.get("steps")
        if not isinstance(steps, list) or not steps:
            out.append("missing_steps")
        else:
            for step in steps:
                text = step.get("action") if isinstance(step, dict) else step
                if isinstance(text, str) and text.strip():
                    if isinstance(step, dict) and "action_key" in step:
                        key = step.get("action_key")
                        if key is not None and (not isinstance(key, str) or not key.strip()):
                            out.append("malformed_step")
                            break
                    continue
                out.append("malformed_step")
                break
        for field in ("effects", "evidence"):
            if field in value and value[field] is not None:
                items = value[field]
                if not isinstance(items, list) or not all(
                    isinstance(i, str) and i.strip() for i in items
                ):
                    out.append(f"malformed_{field}")
        deps = value.get("depends_on")
        if deps is not None:
            ok = isinstance(deps, list) and all(
                isinstance(d, dict)
                and isinstance(d.get("ref"), str)
                and d["ref"].strip()
                and (d.get("version") is None or isinstance(d.get("version"), int))
                for d in deps
            )
            if not ok:
                out.append("malformed_depends_on")
        return out
    if isinstance(value, str):
        lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
        listed = [ln for ln in lines if _PROCEDURE_STEP_LINE.match(ln)]
        return [] if len(listed) >= 2 else ["not_structured"]
    return ["not_structured"]


def procedure_action_keys(value: Any) -> list[str]:
    """The ``action_key`` each dict step of a procedure names, in order, skipping
    steps that name none. Empty for string procedures. Pure."""
    if not isinstance(value, dict):
        return []
    steps = value.get("steps")
    if not isinstance(steps, list):
        return []
    out: list[str] = []
    for step in steps:
        if isinstance(step, dict):
            key = step.get("action_key")
            if isinstance(key, str) and key.strip():
                out.append(key.strip())
    return out


def preconditions_status(
    value: Any, environment: Mapping[str, Any] | None
) -> tuple[str, list[str]]:
    """Whether a procedure applies to a run in *environment*.

    Returns ``("applicable", [])`` when the procedure states no environment
    preconditions, when *environment* is empty (nothing to check against — the
    MCP case), or when every stated one matches. Returns
    ``("not_applicable", ["runtime: wants python3.12, run has python3.9"])`` on
    an explicit mismatch, and ``("unknown", ["runtime"])`` when the procedure
    constrains a key the environment does not report.

    Environment preconditions live in ``value["preconditions"]`` either as a
    dict (``{"runtime": "python3.12"}``; a list of accepted values is allowed)
    or, inside a list of preconditions, as dict items of the same shape. String
    preconditions describe the task and are never matched here. Matching is
    exact after ``strip()``, or a prefix match when the constraint ends in
    ``*`` (``"claude-*"``). Pure.
    """
    if not isinstance(value, dict):
        return "applicable", []
    constraints: dict[str, list[str]] = {}
    raw = value.get("preconditions")
    items: list[Any]
    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, list):
        items = [i for i in raw if isinstance(i, dict)]
    else:
        items = []
    for item in items:
        for key in PROCEDURE_ENVIRONMENT_PRECONDITIONS:
            if key not in item or item[key] is None:
                continue
            wanted = item[key]
            values = wanted if isinstance(wanted, list) else [wanted]
            cleaned = [str(v).strip() for v in values if str(v).strip()]
            if cleaned:
                constraints.setdefault(key, []).extend(cleaned)
    if not constraints:
        return "applicable", []
    env = {k: str(v).strip() for k, v in (environment or {}).items() if v is not None}
    if not env:
        return "applicable", []
    unknown: list[str] = []
    mismatched: list[str] = []
    for key, wanted in constraints.items():
        have = env.get(key)
        if not have:
            unknown.append(key)
            continue
        if any(_precondition_matches(w, have) for w in wanted):
            continue
        mismatched.append(f"{key}: wants {' or '.join(wanted)}, run has {have}")
    if mismatched:
        return "not_applicable", mismatched
    if unknown:
        return "unknown", unknown
    return "applicable", []


def _precondition_matches(wanted: str, have: str) -> bool:
    if wanted.endswith("*"):
        return have.startswith(wanted[:-1])
    return wanted == have


#: ``- step``, ``* step``, ``• step``, ``1. step``, ``1) step``.
_PROCEDURE_STEP_LINE = re.compile(r"^(?:[-*•]|\d+[.)])\s+\S")

_PRODUCTION_AGENT_PREFIXES = ("agent/", "prod/", "prod-")


class Provenance(BaseModel):
    """Tracks who wrote a memory entry and why."""

    agent_id: str
    session_id: str
    written_at: datetime
    pattern_refs: list[str] = Field(default_factory=list)


class ArtifactRef(BaseModel):
    """Reference to an external artifact (blob, file, model checkpoint, etc.)."""
    uri: str  # s3://bucket/path, file:///path, https://...
    media_type: str | None = None  # e.g. "application/json", "model/onnx"
    label: str | None = None  # human-readable description
    size_bytes: int | None = None


class QualityIssue(BaseModel):
    """A single quality issue found during write-time evaluation."""

    type: str
    message: str
    suggestion: str


class QualityReport(BaseModel):
    """Quality assessment of a memory write.

    Returned alongside the stored entry so agents can improve low-quality
    memories.  A score >= 0.8 is considered acceptable; below that the
    ``issues`` list contains actionable suggestions.
    """

    score: float = Field(ge=0.0, le=1.0)
    action: str = "stored_ok"
    issues: list[QualityIssue] = Field(default_factory=list)


class MemoryEntry(BaseModel):
    """A single versioned memory entry within the AMFS namespace."""

    amfs_version: str = "0.3.0"
    entity_path: str
    key: str
    version: int = 1
    value: Any = None
    provenance: Provenance
    confidence: float = 1.0
    outcome_count: int = 0
    # ── Outcome evidence ────────────────────────────────────────────────
    # Raw counts of committed outcomes that cited this entry, split by
    # direction, plus the recency-weighted evidence mass the evidence model
    # (``amfs_core.evidence``) turns into ``confidence``. ``prior_confidence``
    # is the confidence the author wrote; it is the Beta prior the posterior
    # shrinks toward, and it is what a fresh version resets to.
    success_count: int = 0
    failure_count: int = 0
    evidence_success: float = 0.0
    evidence_failure: float = 0.0
    prior_confidence: float | None = None
    last_outcome: str | None = None
    last_outcome_at: datetime | None = None
    # Set when a failure pushed the posterior below the discredit threshold;
    # cleared when later evidence lifts it back. Read paths exclude discredited
    # entries by default and render them as anti-patterns in briefings.
    discredited_at: datetime | None = None
    #: Distinct agent ids whose *successful* outcomes credited this version, most
    #: recent last, capped at ten by the writer. "Validated by 3 agents" is a
    #: stronger signal than one agent's repeated success, and a peer reading the
    #: entry can see who stands behind it.
    validators: list[str] = Field(default_factory=list)
    recall_count: int = 0
    priority_score: float | None = None
    tier: int = 3
    importance_score: float | None = None
    importance_dimensions: dict[str, float] | None = None
    ttl_at: datetime | None = None
    embedding: list[float] | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    memory_type: MemoryType = MemoryType.FACT
    # True when the value is a stored working file (source code, markup, config)
    # rather than a knowledge claim. Orthogonal to ``memory_type``; used to demote
    # artifacts in retrieve/search/briefing so they don't crowd out real facts.
    is_artifact: bool = False
    shared: bool = True
    branch: str = "main"
    content_hash: str | None = None
    integrity_chain: str | None = None
    commit_id: str | None = None

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """True when a TTL is set and has elapsed.

        Read paths use this to enforce ``ttl_at`` immediately, without waiting
        for the background ``LifecycleManager`` sweep to archive the entry.
        """
        if self.ttl_at is None:
            return False
        reference = now or datetime.now(timezone.utc)
        ttl = self.ttl_at
        if ttl.tzinfo is None:
            ttl = ttl.replace(tzinfo=timezone.utc)
        return ttl <= reference

    def effective_confidence(self, *, decay_half_life_days: float | None = None) -> float:
        """Confidence adjusted for four-signal decay: time, memory type, outcomes, and access frequency.

        Uses exponential decay: effective = stored * 0.5^(age_days / half_life).
        The effective half-life is modulated by:
          - memory_type: beliefs decay faster (0.5x), experiences slower (1.5x)
          - recall_count: frequently accessed entries decay slower via log1p
          - outcome_count: outcome-validated entries get 2x half-life boost
        Returns stored confidence unchanged when decay is disabled.
        """
        if decay_half_life_days is None or decay_half_life_days <= 0:
            return self.confidence
        age = datetime.now(timezone.utc) - self.provenance.written_at
        age_days = age.total_seconds() / 86400.0

        type_mult = MEMORY_TYPE_DECAY_MULTIPLIERS.get(self.memory_type, 1.0)
        base_half_life = decay_half_life_days * type_mult

        # Frequency-modulated decay: higher recall_count flattens the decay curve
        effective_half_life = base_half_life * (1 + math.log1p(self.recall_count))
        if self.outcome_count > 0:
            effective_half_life *= 2

        decay_factor = math.pow(0.5, age_days / effective_half_life)
        return self.confidence * decay_factor

    @property
    def entry_key(self) -> str:
        """Canonical key spec used for causal linking: ``entity_path/key``."""
        return f"{self.entity_path}/{self.key}"

    @property
    def evidence_status(self) -> str:
        """One of ``untested``, ``validated``, ``contested``, ``discredited``.

        The word an agent sees next to an entry. ``validated`` means the
        record supports acting on this — every outcome that cited it succeeded,
        or the posterior over its outcomes is high enough over enough of them
        (see ``amfs_core.labels``); ``contested`` means the record is mixed and
        thin, but the posterior is still above the discredit threshold;
        ``discredited`` means a failure pushed it below and nothing has lifted
        it since. ``untested`` entries have never been cited by an outcome, so
        their confidence is whatever the author claimed.
        """
        from .labels import evidence_label

        return evidence_label(
            success_count=self.success_count,
            failure_count=self.failure_count,
            evidence_success=self.evidence_success,
            evidence_failure=self.evidence_failure,
            discredited=self.discredited_at is not None,
            outcome_count=self.outcome_count,
            confidence=self.confidence,
        )

    @property
    def posterior(self) -> tuple[float, int]:
        """``(p_success, n)``: the record behind the label.

        ``p_success`` is the posterior mean of success from the entry's decayed
        outcome evidence under a neutral prior — an untested entry is 0.5, not
        whatever its author claimed — and ``n`` is how many outcomes it rests on.
        A 0.9 over twelve outcomes and a 0.9 over one are different things to
        act on, which is what the pair is for.
        """
        from .labels import posterior_mean

        return (
            round(posterior_mean(self.evidence_success, self.evidence_failure), 3),
            int(self.success_count) + int(self.failure_count),
        )

    @property
    def provenance_tier(self) -> ProvenanceTier:
        """Compute quality tier from provenance and outcome history.

        Production agents are identified by agent_id prefix conventions
        (``agent/``, ``prod/``, ``prod-``) or can be set explicitly via
        environment configuration.
        """
        is_production = any(
            self.provenance.agent_id.startswith(p) for p in _PRODUCTION_AGENT_PREFIXES
        )
        if is_production and self.outcome_count > 0:
            return ProvenanceTier.PRODUCTION_VALIDATED
        if is_production:
            return ProvenanceTier.PRODUCTION_OBSERVED
        if self.provenance.agent_id.startswith(("dev/", "test/", "dev-", "test-")):
            return ProvenanceTier.DEVELOPMENT
        if self.provenance.agent_id.startswith(("manual/", "seed/", "human/")):
            return ProvenanceTier.MANUAL
        # Default: if agent has outcomes it's treated as observed, otherwise dev
        if self.outcome_count > 0:
            return ProvenanceTier.PRODUCTION_OBSERVED
        return ProvenanceTier.DEVELOPMENT


class AttemptRecord(BaseModel):
    """One failed attempt inside a task that was eventually resolved.

    An agent that tries a remembered fix, sees it fail, and then succeeds on a
    different action has learned two things, and a single ``success`` outcome
    records only one of them. Recording the boundary between the attempts lets
    the entries the failed attempt relied on receive the failure, while the
    entries the final answer relied on receive the success — within one trace,
    so the task still has exactly one terminal label for training.

    ``causal_entry_keys`` are the ``entity_path/key`` specs read between the
    previous boundary and this one. ``action_indices`` point into the trace's
    ``tool_calls`` at the actions this attempt took, so a training exporter can
    pair them against the final action as a rejected/chosen contrast.
    """

    attempt: int
    outcome_type: OutcomeType = OutcomeType.MINOR_FAILURE
    causal_entry_keys: list[str] = Field(default_factory=list)
    #: ``entry_key -> version`` as read during this attempt. The outcome is
    #: applied only if the key still says what it said then; see
    #: ``OutcomeRecord.causal_entry_versions``.
    causal_entry_versions: dict[str, int] = Field(default_factory=dict)
    action_indices: list[int] = Field(default_factory=list)
    summary: str | None = None


class OutcomeRecord(BaseModel):
    """Records an outcome event that back-propagates to memory entries."""

    outcome_ref: str
    outcome_type: OutcomeType
    causal_confidence: float = 1.0
    committed_at: datetime
    causal_entry_keys: list[str] = Field(default_factory=list)
    #: ``entry_key -> version`` the agent actually read, for the keys above.
    #: Credit goes to the claim that was read: when the live version of a key
    #: differs from this one *and* its value changed in between (the agent's
    #: reflection rewrote the lesson before committing, or a colleague did),
    #: the outcome is not applied to the new claim. Keys absent from the map
    #: are applied unconditionally, as before.
    causal_entry_versions: dict[str, int] = Field(default_factory=dict)
    agent_id: str
    #: Failed attempts that preceded the terminal outcome, oldest first. Each is
    #: applied to its own causal entries before ``outcome_type`` is applied to
    #: ``causal_entry_keys``. Empty for the common single-shot task.
    attempts: list[AttemptRecord] = Field(default_factory=list)
    #: Index into the trace's ``tool_calls`` of the action that produced the
    #: terminal outcome, when the agent took more than one. Training pipelines
    #: use it as the supervised target instead of guessing "the first action".
    final_action_index: int | None = None
    #: What happened to each decisive action, derived at commit from
    #: ``tool_calls`` + ``attempts`` + ``final_action_index`` (see
    #: ``amfs_core.actions.actions_taken``): the action that ended each failed
    #: attempt is a loss, the terminal action carries the outcome. This is the
    #: replay-buffer row that action priors are computed from. Persisted on the
    #: outcome, never as a memory entry.
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)
    #: The entities this outcome is about — the explicit one the caller named plus
    #: those of the causal keys. Priors are scoped by overlap with the entity a
    #: later retrieve asks about; the outcomes table had no entity scope before.
    entity_paths: list[str] = Field(default_factory=list)
    #: Optional caller label for the kind of situation ("card-declined ticket"),
    #: stored verbatim for analysis; similarity itself comes from ``task_input``.
    situation: str | None = None
    #: Captured prompt and response, carried alongside the outcome rather than
    #: only on the trace. On the SaaS path the adapter's ``commit_outcome`` is what
    #: reaches the server, and the server seals its immutable trace from that call
    #: — so a capture that travelled only on the later ``save_trace`` was missing
    #: from the sealed copy that training and export actually read. Not persisted
    #: to the outcomes table; adapters write named columns.
    task_input: str | None = None
    response_text: str | None = None
    #: The actions taken, carried here for the same reason as the capture above.
    #: Typed as plain dicts because ``ToolCall`` is declared further down this
    #: module; they are the ``model_dump`` of one.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    #: The session's metadata as the trace will carry it — the ``attributes``
    #: bag and ``llm_calls`` list merged in — carried here for the same reason:
    #: the server seals from this call, and a bag that arrived only on the later
    #: trace left the sealed copy with no attributes and no token or cost figures.
    session_metadata: dict[str, Any] | None = None
    #: Set by ``AgentMemory.commit_outcome``, which always posts the full trace
    #: straight after this record goes out. It tells the server not to seal a
    #: trace of its own from this call, because a better one is seconds away.
    #: The server has no trace of the caller's to seal: it assembles one on its
    #: shared handle, so the causal entries, query events, state diff and
    #: session window come from whatever other requests left on that handle,
    #: and the session is the server process's rather than the caller's. Two
    #: sealed traces per outcome also doubled every count and average measured
    #: over them, and chained the fabricated one into a hash chain keyed by the
    #: process's session and therefore shared across accounts.
    #: Default ``False``, so a caller that posts no trace — a direct REST client
    #: — still gets the server-side seal, which for it is the only one there is.
    trace_follows: bool = False


class TraceEntry(BaseModel):
    """A snapshot of an entry that was read during a decision."""

    entity_path: str
    key: str
    version: int
    confidence: float
    value: Any = None
    memory_type: str | None = None
    written_by: str | None = None
    read_at: datetime | None = None
    duration_ms: float | None = None
    #: The outcome record as it stood at read time — untested / validated /
    #: contested / discredited and the tally behind it. ``None`` on traces
    #: sealed before the evidence model; readers treat that as untested.
    evidence_status: str | None = None
    success_count: int = 0
    failure_count: int = 0


class ExternalContext(BaseModel):
    """External context recorded during a decision session."""

    label: str
    summary: str
    source: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCall(BaseModel):
    """An action the agent took during a decision session.

    The action half of a supervised training pair: ``task_input`` on the trace is
    what was asked, and this is what the agent did about it. Recorded explicitly
    by the agent, because a memory layer sees only its own tools — it has no way
    to observe a call to the caller's deploy or refund tool.

    ``arguments`` is scanned for secrets before it is persisted, on the same
    grounds as the captured prompt: an action's parameters routinely carry
    credentials, and unlike free text they are structured in a way that makes
    them easy to feed straight into a dataset.
    """

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: Truncated rather than full, so a trace cannot become a copy of an API
    #: response. The hash below is what proves the result was not altered.
    result_summary: str = ""
    result_hash: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = 0
    source: str | None = None
    success: bool = True
    #: Explicit identity of the action for action-level learning, when the agent
    #: knows it better than the argument heuristic in ``amfs_core.actions`` does.
    action_key: str | None = None


class QueryEvent(BaseModel):
    """A search or list operation performed during a decision session."""

    operation: str  # "search" or "list"
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_count: int = 0
    duration_ms: float | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorEvent(BaseModel):
    """An error that occurred during a decision session."""

    operation: str  # "read", "write", "search", "tool", "adapter"
    error_type: str
    message: str
    stack_trace: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConfidenceChange(BaseModel):
    """Records a confidence change caused by an outcome."""

    entity_path: str
    key: str
    before: float
    after: float
    outcome_ref: str


class MemoryStateDiff(BaseModel):
    """Summary of memory changes during a session."""

    entries_created: int = 0
    entries_updated: int = 0
    confidence_changes: list[ConfidenceChange] = Field(default_factory=list)


class DecisionTrace(BaseModel):
    """A persisted record of the causal chain behind an outcome."""

    id: str = Field(default_factory=lambda: "")
    agent_id: str
    session_id: str
    outcome_ref: str | None = None
    outcome_type: str | None = None
    decision_summary: str | None = None
    # The request that triggered the decision, and optionally the agent's
    # answer. Without the prompt side of the pair a trace records what was
    # decided but not what was being asked, which is exactly what supervised
    # training needs. Both are opt-in per call and scanned for secrets before
    # they are persisted.
    task_input: str | None = None
    response_text: str | None = None
    #: The action side of the pair above. Empty on a trace whose agent recorded no
    #: action, which is the common case for a session that only read and wrote
    #: memory; supervised training needs both halves and skips such a trace.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    causal_entries: list[TraceEntry] = Field(default_factory=list)
    external_contexts: list[ExternalContext] = Field(default_factory=list)
    query_events: list[QueryEvent] = Field(default_factory=list)
    error_events: list[ErrorEvent] = Field(default_factory=list)
    state_diff: MemoryStateDiff | None = None
    session_metadata: SessionMetadata | None = None
    session_started_at: datetime | None = None
    session_ended_at: datetime | None = None
    session_duration_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    namespace: str = "default"


# ── Atomic commits ───────────────────────────────────────────────────


class CommitEntry(BaseModel):
    """A single key-write within an atomic commit."""

    entity_path: str
    key: str
    version: int
    content_hash: str | None = None


class Commit(BaseModel):
    """An atomic group of writes across multiple keys.

    Mirrors a git commit: unique id, author, message, tree hash, and
    parent pointer(s) for DAG traversal.
    """

    id: str
    message: str = ""
    author_agent_id: str
    session_id: str | None = None
    entries: list[CommitEntry] = Field(default_factory=list)
    tree_hash: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    branch: str = "main"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    namespace: str = "default"


class RecallConfig(BaseModel):
    """Weights for composite recall scoring.

    When no embedder is configured or an entry lacks an embedding vector,
    the semantic component scores 0.0 and the remaining weights (recency,
    confidence) dominate the composite score.
    """

    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    confidence_weight: float = 0.2
    #: Weight of the outcome-evidence term (``amfs_core.evidence.evidence_signal``,
    #: in [-1, 1]). Separate from confidence so a validated 0.9 outranks an
    #: untested 0.9 and a contested one falls behind both.
    evidence_weight: float = 0.15
    #: Discredited entries (a failure left them under the discredit threshold)
    #: are dropped from results unless this is set.
    include_discredited: bool = False
    #: Append recently discredited entries matching the query as an avoid list
    #: (flagged ``_avoid`` in the breakdown, scored 0, never booked as reads),
    #: so the agent is told what not to do instead of merely not being told.
    include_avoid: bool = False
    #: When the top hit is validated with no recent failure, return fewer
    #: results: the answer is known, and the rest is tokens.
    adaptive_k: bool = False
    recency_half_life_days: float = 30.0


class ScoredEntry(BaseModel):
    """A memory entry with composite recall score."""

    entry: MemoryEntry
    score: float
    #: Score components. Floats, plus the odd label the server adds alongside
    #: them (``evidence_status``, ``is_artifact``) so a caller can see *why* an
    #: entry ranked where it did without re-deriving it.
    breakdown: dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Filters for searching across memory entries.

    When *query* is set the adapter may use full-text search (e.g. Postgres
    tsvector) to filter/rank results.  Adapters that do not support FTS
    ignore the field — the SDK falls back to Python substring matching.
    """

    query: str | None = None
    entity_path: str | None = None
    entity_paths: list[str] | None = None
    # Widens *entity_path* from an exact match to that path and everything
    # beneath it, per amfs_core.scope.covers: "a/b" then reaches "a/b/c" but
    # never "a/bc". Off by default because a caller naming an exact key means
    # it; the callers that want this are the ones deriving a scope rather than
    # being handed one, such as a session-opening briefing on a repo root.
    include_descendants: bool = False
    min_confidence: float = 0.0
    max_confidence: float | None = None
    agent_id: str | None = None
    since: datetime | None = None
    pattern_ref: str | None = None
    limit: int = 100
    sort_by: str = "confidence"  # "confidence", "recency", "version", "priority"
    recall_config: RecallConfig | None = None
    depth: int = 3
    # When True (default) artifacts (stored source files) are demoted to the
    # bottom of results; when False they are excluded entirely.
    include_artifacts: bool = True


class TierConfig(BaseModel):
    """Configuration for tiered memory hierarchy.

    Controls the capacity of each tier and the weights used in the
    HMO-inspired priority scoring formula.
    """

    hot_capacity: int = 50
    warm_capacity: int = 200
    alpha: float = 1.0
    beta: float = 1.0
    decay_lambda: float = 0.1


class MemoryStats(BaseModel):
    """Aggregate statistics about memory state."""

    total_entries: int = 0
    total_entities: int = 0
    total_agents: int = 0
    agents: dict[str, int] = Field(default_factory=dict)
    entities: dict[str, int] = Field(default_factory=dict)
    confidence_avg: float = 0.0
    confidence_min: float = 0.0
    confidence_max: float = 0.0
    outcome_linked_count: int = 0
    oldest_entry_at: datetime | None = None
    newest_entry_at: datetime | None = None


class ScopeInfo(BaseModel):
    """Summary info about a scope (entity_path)."""

    path: str
    entry_count: int
    avg_confidence: float
    keys: list[str] = Field(default_factory=list)
    oldest: datetime | None = None
    newest: datetime | None = None


class ConflictPolicy(str, Enum):
    """How to handle writes when another agent modified the entry since our last read."""

    LAST_WRITE_WINS = "last_write_wins"
    RAISE = "raise"


class SemanticQuery(BaseModel):
    """Query for semantic (embedding-based) search."""

    text: str
    entity_path: str | None = None
    min_confidence: float = 0.0
    limit: int = 10
    min_similarity: float = 0.0


# ── Knowledge graph models ────────────────────────────────────────────


class GraphEdge(BaseModel):
    """A directed edge in the knowledge graph."""

    source_entity: str
    source_type: str
    relation: str
    target_entity: str
    target_type: str
    confidence: float = 1.0
    evidence_count: int = 1
    first_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: dict[str, Any] | None = None


class GraphNeighborQuery(BaseModel):
    """Parameters for a knowledge graph neighbor traversal."""

    entity: str
    relation: str | None = None
    direction: str = "both"  # "outgoing", "incoming", "both"
    min_confidence: float = 0.0
    depth: int = 1
    limit: int = 50


class DigestType(str, Enum):
    """Types of compiled knowledge digests produced by the Cortex."""

    ENTITY = "entity"
    AGENT_BRIEF = "agent_brief"
    SOURCE = "source"
    CONNECTION_MAP = "connection_map"
    TRACE_PATTERN = "trace_pattern"
    AGENT_CLUSTERS = "agent_clusters"


class Digest(BaseModel):
    """A compiled knowledge digest — distilled from raw memory entries."""

    digest_type: DigestType
    scope: str
    summary: dict[str, Any]
    entry_count: int = 0
    source_agents: list[str] = Field(default_factory=list)
    compiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    staleness_ms: int = 0
    anticipation_score: float = 0.0
    namespace: str = "default"
    branch: str = "main"


# ── Git-like timeline (OSS) ────────────────────────────────────────────


class EventType(str, Enum):
    """Types of events on the agent timeline (git commit log).

    Core event types (OSS): WRITE, READ, OUTCOME, WEBHOOK,
    BRIEF_COMPILED, CROSS_AGENT_READ. Branching event types (Pro):
    BRANCH_CREATED, BRANCH_MERGED, BRANCH_CLOSED, ACCESS_GRANTED,
    ACCESS_REVOKED, ROLLBACK, TAG_CREATED, CHERRY_PICK, FORK.
    """

    WRITE = "write"
    READ = "read"
    OUTCOME = "outcome"
    WEBHOOK = "webhook"
    BRIEF_COMPILED = "brief_compiled"
    CROSS_AGENT_READ = "cross_agent_read"
    BRANCH_CREATED = "branch_created"
    BRANCH_MERGED = "branch_merged"
    BRANCH_CLOSED = "branch_closed"
    ACCESS_GRANTED = "access_granted"
    ACCESS_REVOKED = "access_revoked"
    ROLLBACK = "rollback"
    TAG_CREATED = "tag_created"
    CHERRY_PICK = "cherry_pick"
    FORK = "fork"
    SNAPSHOT_TAKEN = "snapshot_taken"
    SNAPSHOT_RECOVERED = "snapshot_recovered"
    CONSOLIDATION_PROPOSED = "consolidation_proposed"
    CONSOLIDATION_APPROVED = "consolidation_approved"
    CONSOLIDATION_REJECTED = "consolidation_rejected"
    CONSOLIDATION_AUTO_MERGED = "consolidation_auto_merged"


class ConsolidationProposal(BaseModel):
    """A proposed memory consolidation, created on a branch for review.

    Tier A (auto-safe) proposals are applied directly; Tier B (semantic)
    proposals sit on a branch until a human or agent approves the merge.
    """

    id: str
    entity_path: str
    branch_name: str
    strategy: str
    risk_tier: str
    source_entry_keys: list[str]
    proposed_value: Any
    proposed_confidence: float
    compression_ratio: float
    rationale: str
    status: str = "pending"
    created_at: datetime
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class ConsolidationReport(BaseModel):
    """Summary report of a consolidation run across an entity."""

    entity_path: str
    auto_archived: int
    proposals_created: int
    proposals_auto_merged: int
    compression_ratio: float
    consolidated_at: datetime


class SessionMetadata(BaseModel):
    """Metadata about the agent's runtime environment.

    Captured once per session (typically via amfs_set_identity) and attached
    to the AgentProfile and DecisionTrace so every trace records which model,
    platform, and toolset produced the decisions.

    Extra keys are kept and serialised. Session-scoped data the SDKs collect for
    a trace — the ``attributes`` bag set by ``AgentMemory.set_session_attributes``
    and the ``llm_calls`` list built by ``AgentMemory.record_llm_call`` — rides
    here rather than as new fields on ``DecisionTrace``, and a subclass or
    extension may add keys of its own. Without ``extra="allow"`` such keys were
    dropped twice: once when a dict was validated into this model and again when
    a trace was dumped for the HTTP adapter, which serialises by declared type.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    client_name: str | None = None
    platform: str | None = None
    tools_available: list[str] = Field(default_factory=list)
    mcp_client_id: str | None = None
    mcp_session_id: str | None = None
    #: The customer's agent's own version and runtime (``"checkout-bot@2.3.1"``,
    #: ``"python3.12/linux"``). Together with ``model`` these are the confounders
    #: a canary or a before/after comparison must hold stable: a change in the
    #: agent that coincides with a change in its memory would otherwise be
    #: credited to the memory. Declared here so they survive validation and
    #: are lifted to reserved trace attributes; both are optional.
    agent_version: str | None = None
    runtime: str | None = None
    #: Where the terminal outcome came from, when the agent did not decide it
    #: itself: ``"ci"``, ``"human"``, ``"verifier"``, ``"customer"``. An outcome
    #: that carries ``verified_by`` is external evidence; one without is the
    #: agent's own declaration. ``evidence`` is a small free-form bag for the
    #: pointer (``{"run_id": ..., "url": ...}``).
    verified_by: str | None = None
    evidence: dict[str, Any] | None = None


#: The ``SessionMetadata`` keys that describe the run's environment. A
#: procedure's ``preconditions`` may constrain any of them; serving code matches
#: the two (see ``preconditions_status``).
ENVIRONMENT_KEYS: tuple[str, ...] = ("model", "agent_version", "runtime", "platform")


def environment_of(metadata: Mapping[str, Any] | BaseModel | None) -> dict[str, str]:
    """The environment a session ran in, as ``{key: value}`` for the
    :data:`ENVIRONMENT_KEYS` that are set. Accepts a ``SessionMetadata``, a plain
    dict of attributes, or ``None`` (an empty environment)."""
    if metadata is None:
        return {}
    data = metadata.model_dump() if isinstance(metadata, BaseModel) else dict(metadata)
    out: dict[str, str] = {}
    for key in ENVIRONMENT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out


class AgentProfile(BaseModel):
    """Declarative profile describing an agent's role and defaults."""

    description: str = ""
    default_branch: str = "main"
    auto_context_paths: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    session_metadata: SessionMetadata | None = None


class AgentCapability(BaseModel):
    """A declared capability or domain of expertise for an agent."""

    name: str
    description: str = ""
    entity_paths: list[str] = Field(default_factory=list)


class MemoryContract(BaseModel):
    """A contract specifying expectations for entries an agent writes.

    Contracts are validated at write time to enforce schema, TTL, and
    confidence expectations.
    """

    entity_path: str
    key_pattern: str = "*"
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    required_fields: list[str] = Field(default_factory=list)
    ttl_required: bool = False
    description: str = ""


class Agent(BaseModel):
    """Registered agent — auto-created on first write."""

    id: str = ""
    namespace: str = "default"
    agent_id: str
    display_name: str | None = None
    created_at: datetime | None = None
    last_active_at: datetime | None = None
    entry_count: int = 0
    profile: AgentProfile | None = None
    capabilities: list[AgentCapability] = Field(default_factory=list)
    contracts: list[MemoryContract] = Field(default_factory=list)


class AgentGroup(BaseModel):
    """User-defined group of agents."""

    id: str = ""
    namespace: str = "default"
    account_id: str | None = None
    name: str
    description: str = ""
    color: str | None = None
    icon: str | None = None
    position: float = 0.0
    auto_generated: bool = False
    source_cluster_id: str | None = None
    member_count: int = 0
    agent_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentCluster(BaseModel):
    """Auto-detected cluster of related agents."""

    cluster_id: str
    suggested_name: str
    agents: list[str]
    dominant_entities: list[str] = Field(default_factory=list)
    dominant_platform: str | None = None
    cohesion_score: float = 0.0
    rationale: str = ""
    total_entries: int = 0


class Event(BaseModel):
    """A single event on the agent timeline (the git commit log)."""

    id: str = ""
    namespace: str = "default"
    agent_id: str
    branch: str = "main"
    event_type: EventType
    summary: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    actor_agent_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Branching models (Pro — used by amfs_branching) ──────────────────


class BranchStatus(str, Enum):
    """Lifecycle state of a memory branch."""

    ACTIVE = "active"
    MERGED = "merged"
    CLOSED = "closed"


class Branch(BaseModel):
    """A named branch of agent memory."""

    id: str = ""
    namespace: str = "default"
    name: str
    parent_branch: str = "main"
    branched_at: datetime
    created_by: str
    description: str | None = None
    status: BranchStatus = BranchStatus.ACTIVE
    merged_at: datetime | None = None
    merged_by: str | None = None
    created_at: datetime | None = None
    head_commit_id: str | None = None
    base_commit_id: str | None = None


class BranchAccessPermission(str, Enum):
    """Permission level for external agents on a branch."""

    READ = "read"
    READ_WRITE = "read_write"


class BranchAccess(BaseModel):
    """Grant giving an external agent/team access to a branch."""

    id: str = ""
    namespace: str = "default"
    branch_name: str
    grantee_type: str  # "user", "team", "api_key"
    grantee_id: str
    permission: BranchAccessPermission = BranchAccessPermission.READ
    granted_by: str
    granted_at: datetime | None = None


class FieldChange(BaseModel):
    """A single field-level change within a JSON value (RFC 6901 path)."""

    path: str
    operation: str  # "add", "remove", "replace"
    old_value: Any = None
    new_value: Any = None


class DiffEntry(BaseModel):
    """One entry's difference between a branch and its parent."""

    entity_path: str
    key: str
    diff_type: str  # "added", "modified", "deleted"
    branch_value: Any = None
    parent_value: Any = None
    branch_confidence: float | None = None
    parent_confidence: float | None = None
    branch_shared: bool | None = None
    field_changes: list[FieldChange] = Field(default_factory=list)


class MemoryPatch(BaseModel):
    """A serializable set of field-level changes that can be applied to entries."""

    entity_path: str
    key: str
    changes: list[FieldChange] = Field(default_factory=list)
    source_version: int | None = None
    target_version: int | None = None


class MergeConflict(BaseModel):
    """A conflict detected during branch merge."""

    entity_path: str
    key: str
    branch_value: Any
    main_value: Any
    branch_version: int = 0
    main_version: int = 0
    reason: str = "both_modified"


class MergeStrategy(str, Enum):
    """How to resolve merge conflicts."""

    FAST_FORWARD = "fast_forward"
    BRANCH_WINS = "branch_wins"
    MAIN_WINS = "main_wins"
    MANUAL = "manual"


class MergeResult(BaseModel):
    """Result of merging a branch into its parent."""

    branch_name: str
    status: str  # "merged", "conflicts"
    merged_entries: int = 0
    conflicts: list[MergeConflict] = Field(default_factory=list)


class Tag(BaseModel):
    """A named point-in-time marker on a branch (like a git tag)."""

    id: str = ""
    namespace: str = "default"
    name: str
    branch: str = "main"
    tagged_at: datetime
    description: str | None = None
    created_by: str
    created_at: datetime | None = None
    event_id: str | None = None


class PullRequestStatus(str, Enum):
    """Lifecycle state of a pull request."""

    OPEN = "open"
    APPROVED = "approved"
    MERGED = "merged"
    CLOSED = "closed"


class PullRequest(BaseModel):
    """A pull request for merging a branch."""

    id: str = ""
    namespace: str = "default"
    title: str
    description: str | None = None
    source_branch: str
    target_branch: str = "main"
    status: PullRequestStatus = PullRequestStatus.OPEN
    created_by: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    merged_at: datetime | None = None
    merged_by: str | None = None
    closed_at: datetime | None = None
    closed_by: str | None = None
    merge_strategy: str | None = None


class PRReviewStatus(str, Enum):
    """Status of a PR review."""

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"


class PRReview(BaseModel):
    """A review on a pull request."""

    id: str = ""
    namespace: str = "default"
    pr_id: str
    reviewer: str
    status: PRReviewStatus
    comment: str | None = None
    entry_path: str | None = None
    created_at: datetime | None = None


# ── Config models ──────────────────────────────────────────────────────


class LayerConfig(BaseModel):
    """Configuration for a single storage layer."""

    adapter: str  # e.g. "filesystem", "postgres", "redis"
    options: dict[str, Any] = Field(default_factory=dict)


class AMFSConfig(BaseModel):
    """Top-level AMFS configuration."""

    namespace: str
    layers: dict[str, LayerConfig] = Field(default_factory=dict)
