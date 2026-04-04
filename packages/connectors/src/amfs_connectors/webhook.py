"""Generic webhook ingester for receiving events from external systems.

Provides HMAC signature verification, deduplication, size checks,
and a configurable transformation pipeline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from amfs_connectors.base import ConnectorConfig, IngestionResult

logger = logging.getLogger(__name__)


class WebhookConfig(ConnectorConfig):
    """Configuration for a webhook ingestion endpoint."""

    connector_type: str = "webhook"
    secret: str | None = None
    signature_header: str = "X-Webhook-Signature"
    signature_algo: str = "sha256"
    dedup_window_seconds: int = 300
    max_payload_bytes: int = 1_048_576  # 1MB


class WebhookEvent(BaseModel):
    """A received and validated webhook event."""

    id: UUID = Field(default_factory=uuid4)
    connector_id: UUID
    source: str
    event_type: str
    payload: dict[str, Any]
    headers: dict[str, str] = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signature_valid: bool | None = None


TransformFn = Callable[[WebhookEvent], list[IngestionResult]]


class WebhookIngester:
    """Receives, validates, and processes webhook events.

    Pipeline: size check -> signature verify -> dedup -> transform -> record.
    """

    def __init__(self, config: WebhookConfig, memory: Any = None) -> None:
        self._config = config
        self._memory = memory
        self._transforms: dict[str, TransformFn] = {}
        self._seen_events: dict[str, datetime] = {}

    @property
    def config(self) -> WebhookConfig:
        return self._config

    def register_transform(self, event_type_pattern: str, fn: TransformFn) -> None:
        """Register a transform function for an event type pattern.

        Patterns use simple prefix matching: ``pagerduty.incident.*``
        matches ``pagerduty.incident.triggered``, etc.  Use ``*`` as
        a catch-all.
        """
        self._transforms[event_type_pattern] = fn

    def verify_signature(self, payload: bytes, headers: dict[str, str]) -> bool:
        """Verify the HMAC signature of an incoming webhook payload."""
        if not self._config.secret:
            return True

        sig_header = headers.get(self._config.signature_header, "")
        if not sig_header:
            return False

        expected = hmac.new(
            self._config.secret.encode(),
            payload,
            getattr(hashlib, self._config.signature_algo),
        ).hexdigest()

        sig_value = sig_header.split("=", 1)[-1] if "=" in sig_header else sig_header
        return hmac.compare_digest(expected, sig_value)

    def is_duplicate(self, event_id: str) -> bool:
        """Check if we've already processed this event (idempotency guard)."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self._config.dedup_window_seconds
        expired = [k for k, v in self._seen_events.items() if v.timestamp() < cutoff]
        for k in expired:
            del self._seen_events[k]

        if event_id in self._seen_events:
            return True
        self._seen_events[event_id] = now
        return False

    def _find_transform(self, event_type: str) -> TransformFn | None:
        if event_type in self._transforms:
            return self._transforms[event_type]
        for pattern, fn in self._transforms.items():
            if pattern.endswith("*") and event_type.startswith(pattern[:-1]):
                return fn
        return self._transforms.get("*")

    def ingest(
        self,
        payload: bytes,
        headers: dict[str, str],
        *,
        source: str = "webhook",
        event_type: str = "generic",
        event_id: str | None = None,
    ) -> list[IngestionResult]:
        """Process an incoming webhook payload end-to-end."""
        results: list[IngestionResult] = []

        if len(payload) > self._config.max_payload_bytes:
            results.append(IngestionResult(
                connector_id=self._config.id,
                event_id=event_id or "unknown",
                entity_path=self._config.entity_path,
                key="error",
                action="skip",
                success=False,
                error=f"Payload too large: {len(payload)} bytes (max {self._config.max_payload_bytes})",
            ))
            return results

        if not self.verify_signature(payload, headers):
            results.append(IngestionResult(
                connector_id=self._config.id,
                event_id=event_id or "unknown",
                entity_path=self._config.entity_path,
                key="error",
                action="skip",
                success=False,
                error="Invalid webhook signature",
            ))
            return results

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            results.append(IngestionResult(
                connector_id=self._config.id,
                event_id=event_id or "unknown",
                entity_path=self._config.entity_path,
                key="error",
                action="skip",
                success=False,
                error=f"Invalid JSON: {e}",
            ))
            return results

        eid = event_id or data.get("id") or str(uuid4())

        if self.is_duplicate(eid):
            results.append(IngestionResult(
                connector_id=self._config.id,
                event_id=eid,
                entity_path=self._config.entity_path,
                key="duplicate",
                action="skip",
                success=True,
                details={"reason": "duplicate_event"},
            ))
            return results

        event = WebhookEvent(
            connector_id=self._config.id,
            source=source,
            event_type=event_type,
            payload=data,
            headers=headers,
        )

        transform = self._find_transform(event_type)
        if transform:
            results.extend(transform(event))
        else:
            results.append(IngestionResult(
                connector_id=self._config.id,
                event_id=eid,
                entity_path=self._config.entity_path,
                key=f"{event_type}-{eid[:8]}",
                action="write",
                success=True,
                details=data,
            ))

        return results
