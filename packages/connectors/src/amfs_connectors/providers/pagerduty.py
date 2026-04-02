"""PagerDuty connector -- transforms PD incident webhooks into AMFS context.

PagerDuty webhook v3 events are transformed into:
- record_context() calls for incident metadata
- memory writes for high-severity incident patterns
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from amfs_connectors.base import ConnectorABC, ConnectorConfig, IngestionResult
from amfs_connectors.webhook import WebhookEvent


class PagerDutyConfig(ConnectorConfig):
    """PagerDuty-specific connector configuration."""

    connector_type: str = "pagerduty"
    routing_key: str | None = None
    severity_threshold: str = "high"


SEVERITY_ORDER = {"critical": 0, "high": 1, "warning": 2, "info": 3}


class PagerDutyConnector(ConnectorABC):
    """PagerDuty incident connector.

    Registered as the ``pagerduty`` entry point.
    """

    def __init__(self, config: ConnectorConfig | None = None) -> None:
        super().__init__(config or ConnectorConfig(
            name="pagerduty",
            connector_type="pagerduty",
            entity_path="incidents",
        ))

    def validate_event(self, raw_event: dict[str, Any]) -> bool:
        return "event" in raw_event or "data" in raw_event

    def transform(self, raw_event: dict[str, Any]) -> list[IngestionResult]:
        event = WebhookEvent(
            connector_id=self._config.id,
            source="pagerduty",
            event_type=raw_event.get("event", {}).get("event_type", "incident"),
            payload=raw_event,
            headers={},
        )
        return transform_pagerduty_incident(event)


def transform_pagerduty_incident(event: WebhookEvent) -> list[IngestionResult]:
    """Transform a PagerDuty incident webhook into AMFS operations."""
    payload = event.payload
    results: list[IngestionResult] = []

    pd_event = payload.get("event", payload)
    event_type = pd_event.get("event_type", event.event_type)
    data = pd_event.get("data", pd_event)

    incident_id = data.get("id", str(uuid4())[:8])
    title = data.get("title", "Unknown incident")
    severity = data.get("severity", data.get("urgency", "info"))
    service_name = _extract_service(data)
    status = data.get("status", "triggered")

    entity_path = event.payload.get("_amfs_entity_path", f"incidents/{service_name}")
    key = f"pd-{incident_id}"

    value = {
        "incident_id": incident_id,
        "title": title,
        "severity": severity,
        "service": service_name,
        "status": status,
        "event_type": event_type,
        "assignees": _extract_assignees(data),
        "created_at": data.get("created_at"),
        "resolved_at": data.get("resolved_at"),
        "html_url": data.get("html_url"),
    }

    action = "context" if event_type in ("incident.resolved", "incident.acknowledged") else "write"

    results.append(IngestionResult(
        connector_id=event.connector_id,
        event_id=incident_id,
        entity_path=entity_path,
        key=key,
        action=action,
        success=True,
        details=value,
    ))

    return results


def _extract_service(data: dict[str, Any]) -> str:
    service = data.get("service", {})
    if isinstance(service, dict):
        return service.get("name", service.get("summary", "unknown-service"))
    return str(service) if service else "unknown-service"


def _extract_assignees(data: dict[str, Any]) -> list[str]:
    assignees = data.get("assignees", data.get("assignments", []))
    names: list[str] = []
    for a in assignees:
        if isinstance(a, dict):
            assignee = a.get("assignee", a)
            names.append(assignee.get("summary", assignee.get("name", "unknown")))
        else:
            names.append(str(a))
    return names
