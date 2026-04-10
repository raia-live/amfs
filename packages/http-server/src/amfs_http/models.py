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


class OutcomeRequest(BaseModel):
    outcome_ref: str
    outcome_type: str
    causal_entry_keys: list[str] | None = None
    causal_confidence: float = 1.0
    agent_id: str | None = None


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
