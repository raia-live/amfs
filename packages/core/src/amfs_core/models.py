"""Pydantic models for AMFS memory entries, outcomes, and configuration."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Classification of memory entries for type-specific behavior.

    Facts are objective and stable. Beliefs are subjective and decay faster.
    Experiences are append-only records of agent actions.
    """

    FACT = "fact"
    BELIEF = "belief"
    EXPERIENCE = "experience"


class ProvenanceTier(int, Enum):
    """Quality tier derived from how a memory was created and validated.

    Tier 1 (highest): written by a production agent with outcome validation.
    Tier 4 (lowest): manually seeded, no empirical validation.
    """

    PRODUCTION_VALIDATED = 1
    PRODUCTION_OBSERVED = 2
    DEVELOPMENT = 3
    MANUAL = 4


class OutcomeType(str, Enum):
    """Types of outcomes that can affect memory confidence."""

    P1_INCIDENT = "p1_incident"      # confidence *= 1.15
    P2_INCIDENT = "p2_incident"      # confidence *= 1.10
    REGRESSION = "regression"        # confidence *= 1.08
    CLEAN_DEPLOY = "clean_deploy"    # confidence *= 0.97


# Multipliers applied to confidence when an outcome is committed.
OUTCOME_MULTIPLIERS: dict[OutcomeType, float] = {
    OutcomeType.P1_INCIDENT: 1.15,
    OutcomeType.P2_INCIDENT: 1.10,
    OutcomeType.REGRESSION: 1.08,
    OutcomeType.CLEAN_DEPLOY: 0.97,
}

# Beliefs are penalised more by regressions and decay faster.
MEMORY_TYPE_DECAY_MULTIPLIERS: dict[MemoryType, float] = {
    MemoryType.FACT: 1.0,
    MemoryType.BELIEF: 0.5,
    MemoryType.EXPERIENCE: 1.5,
}

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


class MemoryEntry(BaseModel):
    """A single versioned memory entry within the AMFS namespace."""

    amfs_version: str = "0.2.0"
    entity_path: str
    key: str
    version: int = 1
    value: Any = None
    provenance: Provenance
    confidence: float = 1.0
    outcome_count: int = 0
    ttl_at: datetime | None = None
    embedding: list[float] | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    memory_type: MemoryType = MemoryType.FACT
    shared: bool = True

    def effective_confidence(self, *, decay_half_life_days: float | None = None) -> float:
        """Confidence adjusted for time-based decay and memory type.

        Uses exponential decay: effective = stored * 0.5^(age_days / half_life).
        Entries validated by outcomes (outcome_count > 0) decay at half the rate.
        Beliefs decay faster (half_life * 0.5), experiences slower (half_life * 1.5).
        Returns stored confidence unchanged when decay is disabled.
        """
        if decay_half_life_days is None or decay_half_life_days <= 0:
            return self.confidence
        age = datetime.now(timezone.utc) - self.provenance.written_at
        age_days = age.total_seconds() / 86400.0

        type_mult = MEMORY_TYPE_DECAY_MULTIPLIERS.get(self.memory_type, 1.0)
        base_half_life = decay_half_life_days * type_mult
        effective_half_life = base_half_life * 2 if self.outcome_count > 0 else base_half_life

        decay_factor = math.pow(0.5, age_days / effective_half_life)
        return self.confidence * decay_factor

    @property
    def entry_key(self) -> str:
        """Canonical key spec used for causal linking: ``entity_path/key``."""
        return f"{self.entity_path}/{self.key}"

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


class OutcomeRecord(BaseModel):
    """Records an outcome event that back-propagates to memory entries."""

    outcome_ref: str
    outcome_type: OutcomeType
    causal_confidence: float = 1.0
    committed_at: datetime
    causal_entry_keys: list[str] = Field(default_factory=list)
    agent_id: str


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


class ExternalContext(BaseModel):
    """External context recorded during a decision session."""

    label: str
    summary: str
    source: str | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    causal_entries: list[TraceEntry] = Field(default_factory=list)
    external_contexts: list[ExternalContext] = Field(default_factory=list)
    query_events: list[QueryEvent] = Field(default_factory=list)
    error_events: list[ErrorEvent] = Field(default_factory=list)
    state_diff: MemoryStateDiff | None = None
    session_started_at: datetime | None = None
    session_ended_at: datetime | None = None
    session_duration_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    namespace: str = "default"


class RecallConfig(BaseModel):
    """Weights for composite recall scoring."""

    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    confidence_weight: float = 0.2
    recency_half_life_days: float = 30.0


class ScoredEntry(BaseModel):
    """A memory entry with composite recall score."""

    entry: MemoryEntry
    score: float
    breakdown: dict[str, float] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Filters for searching across memory entries."""

    entity_path: str | None = None
    entity_paths: list[str] | None = None
    min_confidence: float = 0.0
    max_confidence: float | None = None
    agent_id: str | None = None
    since: datetime | None = None
    pattern_ref: str | None = None
    limit: int = 100
    sort_by: str = "confidence"  # "confidence", "recency", "version"
    recall_config: RecallConfig | None = None


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


class LayerConfig(BaseModel):
    """Configuration for a single storage layer."""

    adapter: str  # e.g. "filesystem", "postgres", "redis"
    options: dict[str, Any] = Field(default_factory=dict)


class AMFSConfig(BaseModel):
    """Top-level AMFS configuration."""

    namespace: str
    layers: dict[str, LayerConfig] = Field(default_factory=dict)
