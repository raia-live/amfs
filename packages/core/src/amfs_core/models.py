"""Pydantic models for AMFS memory entries, outcomes, and configuration."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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


class Provenance(BaseModel):
    """Tracks who wrote a memory entry and why."""

    agent_id: str
    session_id: str
    written_at: datetime
    pattern_refs: list[str] = Field(default_factory=list)


class MemoryEntry(BaseModel):
    """A single versioned memory entry within the AMFS namespace."""

    amfs_version: str = "0.1.0"
    entity_path: str
    key: str
    version: int = 1
    value: Any = None
    provenance: Provenance
    confidence: float = 1.0
    outcome_count: int = 0
    ttl_at: datetime | None = None
    embedding: list[float] | None = None

    def effective_confidence(self, *, decay_half_life_days: float | None = None) -> float:
        """Confidence adjusted for time-based decay.

        Uses exponential decay: effective = stored * 0.5^(age_days / half_life).
        Entries validated by outcomes (outcome_count > 0) decay at half the rate.
        Returns stored confidence unchanged when decay is disabled.
        """
        if decay_half_life_days is None or decay_half_life_days <= 0:
            return self.confidence
        age = datetime.now(timezone.utc) - self.provenance.written_at
        age_days = age.total_seconds() / 86400.0
        effective_half_life = (
            decay_half_life_days * 2 if self.outcome_count > 0 else decay_half_life_days
        )
        decay_factor = math.pow(0.5, age_days / effective_half_life)
        return self.confidence * decay_factor

    @property
    def entry_key(self) -> str:
        """Canonical key spec used for causal linking: ``entity_path/key``."""
        return f"{self.entity_path}/{self.key}"


class OutcomeRecord(BaseModel):
    """Records an outcome event that back-propagates to memory entries."""

    outcome_ref: str
    outcome_type: OutcomeType
    causal_confidence: float = 1.0
    committed_at: datetime
    causal_entry_keys: list[str] = Field(default_factory=list)
    agent_id: str


class SearchQuery(BaseModel):
    """Filters for searching across memory entries."""

    entity_path: str | None = None
    min_confidence: float = 0.0
    max_confidence: float | None = None
    agent_id: str | None = None
    since: datetime | None = None
    pattern_ref: str | None = None
    limit: int = 100
    sort_by: str = "confidence"  # "confidence", "recency", "version"


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
