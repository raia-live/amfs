"""MockAMFS test harness for connector authors.

Provides a lightweight in-memory AMFS substitute that records all
write() and record_context() calls, so connector tests can verify
their transform logic without a real AMFS instance or database.

Usage::

    from amfs_connectors.testing import MockAMFS

    def test_my_connector():
        mock = MockAMFS()
        connector = MyConnector(config)
        results = connector.transform(sample_event)

        # Apply results to mock
        mock.apply_results(results)

        assert mock.writes[0].key == "expected-key"
        assert len(mock.contexts) == 1
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from amfs_connectors.base import IngestionResult


class MockWrite(BaseModel):
    """A recorded write() call."""

    entity_path: str
    key: str
    value: Any
    confidence: float = 1.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MockContext(BaseModel):
    """A recorded record_context() call."""

    label: str
    summary: str
    source: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MockAMFS:
    """In-memory AMFS substitute for testing connectors.

    Records all memory operations so tests can assert on what the
    connector would have written to a real AMFS instance.
    """

    def __init__(self) -> None:
        self.writes: list[MockWrite] = []
        self.contexts: list[MockContext] = []
        self._entries: dict[str, dict[str, Any]] = {}

    def write(
        self,
        entity_path: str,
        key: str,
        value: Any,
        *,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Record a write operation."""
        self.writes.append(MockWrite(
            entity_path=entity_path,
            key=key,
            value=value,
            confidence=confidence,
        ))
        self._entries.setdefault(entity_path, {})[key] = value

    def record_context(
        self,
        label: str,
        summary: str,
        *,
        source: str | None = None,
    ) -> None:
        """Record a context operation."""
        self.contexts.append(MockContext(
            label=label,
            summary=summary,
            source=source,
        ))

    def read(self, entity_path: str, key: str) -> Any:
        """Read a previously written value."""
        return self._entries.get(entity_path, {}).get(key)

    def apply_results(self, results: list[IngestionResult]) -> None:
        """Apply a list of IngestionResults to the mock.

        - ``action="write"`` -> recorded as a write
        - ``action="context"`` -> recorded as context
        - ``action="skip"`` -> ignored
        """
        for r in results:
            if not r.success:
                continue
            if r.action == "write":
                self.write(r.entity_path, r.key, r.details)
            elif r.action == "context":
                self.record_context(
                    label=f"{r.entity_path}/{r.key}",
                    summary=str(r.details)[:500] if r.details else "",
                    source=r.entity_path,
                )

    def reset(self) -> None:
        """Clear all recorded operations."""
        self.writes.clear()
        self.contexts.clear()
        self._entries.clear()

    @property
    def total_operations(self) -> int:
        return len(self.writes) + len(self.contexts)
