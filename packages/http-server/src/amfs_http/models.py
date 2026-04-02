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


class OutcomeRequest(BaseModel):
    outcome_ref: str
    outcome_type: str
    causal_entry_keys: list[str] | None = None
    causal_confidence: float = 1.0


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


class ContextRequest(BaseModel):
    label: str
    summary: str
    source: str | None = None
