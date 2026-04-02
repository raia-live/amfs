"""Base classes for external system connectors.

Every connector transforms external events into AMFS memory operations
(write + record_context). The abstract interface ensures all connectors
follow the same lifecycle: configure -> connect -> ingest -> disconnect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ConnectorConfig(BaseModel):
    """Base configuration for any connector."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    connector_type: str
    entity_path: str
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class IngestionResult(BaseModel):
    """Result of processing a single ingested event."""

    connector_id: UUID
    event_id: str
    entity_path: str
    key: str
    action: str  # "write", "context", "skip"
    success: bool = True
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ConnectorABC(ABC):
    """Abstract base class for external system connectors.

    Subclasses implement the transformation logic for turning
    external events into AMFS memory operations.
    """

    def __init__(self, config: ConnectorConfig) -> None:
        self._config = config

    @property
    def config(self) -> ConnectorConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @abstractmethod
    def transform(self, raw_event: dict[str, Any]) -> list[IngestionResult]:
        """Transform a raw external event into AMFS operations."""
        ...

    @abstractmethod
    def validate_event(self, raw_event: dict[str, Any]) -> bool:
        """Validate that a raw event is well-formed for this connector."""
        ...

    def extract_event_id(self, raw_event: dict[str, Any]) -> str:
        """Extract a unique event ID from the raw event.

        Default implementation uses a UUID. Override for idempotency.
        """
        return str(uuid4())
