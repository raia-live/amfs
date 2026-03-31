"""Pydantic models for AMFS memory entries, outcomes, and configuration."""

from __future__ import annotations

from datetime import datetime
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


class OutcomeRecord(BaseModel):
    """Records an outcome event that back-propagates to memory entries."""

    outcome_ref: str
    outcome_type: OutcomeType
    causal_confidence: float = 1.0
    committed_at: datetime
    causal_entry_keys: list[str] = Field(default_factory=list)
    agent_id: str


class LayerConfig(BaseModel):
    """Configuration for a single storage layer."""

    adapter: str  # e.g. "filesystem", "postgres", "redis"
    options: dict[str, Any] = Field(default_factory=dict)


class AMFSConfig(BaseModel):
    """Top-level AMFS configuration."""

    namespace: str
    layers: dict[str, LayerConfig] = Field(default_factory=dict)
