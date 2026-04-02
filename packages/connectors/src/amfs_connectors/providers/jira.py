"""Jira connector -- transforms Jira webhook events into AMFS memory.

Handles issue_created, issue_updated, sprint_started, and sprint_closed
events, transforming them into structured writes keyed by project.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from amfs_connectors.base import ConnectorABC, ConnectorConfig, IngestionResult
from amfs_connectors.webhook import WebhookEvent


class JiraConfig(ConnectorConfig):
    """Jira-specific connector configuration."""

    connector_type: str = "jira"
    base_url: str | None = None
    tracked_events: list[str] = [
        "jira:issue_created",
        "jira:issue_updated",
        "sprint_started",
        "sprint_closed",
    ]


class JiraConnector(ConnectorABC):
    """Jira webhook event connector.

    Registered as the ``jira`` entry point.
    """

    def __init__(self, config: ConnectorConfig | None = None) -> None:
        super().__init__(config or ConnectorConfig(
            name="jira",
            connector_type="jira",
            entity_path="jira",
        ))

    def validate_event(self, raw_event: dict[str, Any]) -> bool:
        if "webhookEvent" in raw_event:
            return True
        if "sprint" in raw_event and "event" in raw_event:
            return True
        return False

    def extract_event_id(self, raw_event: dict[str, Any]) -> str:
        ts = raw_event.get("timestamp", "")
        issue = raw_event.get("issue", {})
        issue_key = issue.get("key", "")
        if issue_key:
            return f"jira-{issue_key}-{ts}"
        return f"jira-{uuid4().hex[:12]}"

    def transform(self, raw_event: dict[str, Any]) -> list[IngestionResult]:
        webhook_event_type = raw_event.get("webhookEvent", "")
        event = WebhookEvent(
            connector_id=self._config.id,
            source="jira",
            event_type=_normalize_event_type(webhook_event_type, raw_event),
            payload=raw_event,
            headers={},
        )
        return _transform_jira_event(event)


def _normalize_event_type(webhook_event: str, payload: dict[str, Any]) -> str:
    if webhook_event in ("jira:issue_created", "jira:issue_updated"):
        return webhook_event

    sprint = payload.get("sprint", {})
    if sprint:
        state = sprint.get("state", "").lower()
        if state == "active":
            return "sprint_started"
        if state == "closed":
            return "sprint_closed"

    return webhook_event or "unknown"


def _transform_jira_event(event: WebhookEvent) -> list[IngestionResult]:
    handler = _HANDLERS.get(event.event_type)
    if handler:
        return handler(event)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=str(uuid4())[:8],
        entity_path=_project_entity(event.payload),
        key=f"jira-event-{event.event_type}",
        action="context",
        success=True,
        details={"event_type": event.event_type},
    )]


def _project_entity(payload: dict[str, Any]) -> str:
    issue = payload.get("issue", {})
    fields = issue.get("fields", {})
    project = fields.get("project", {})
    key = project.get("key", "")
    if key:
        return f"jira/{key}"

    sprint = payload.get("sprint", {})
    origin = sprint.get("originBoardId", "")
    if origin:
        return f"jira/board-{origin}"

    return "jira/unknown"


def _handle_issue_created(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    issue = payload.get("issue", {})
    fields = issue.get("fields", {})
    issue_key = issue.get("key", "UNKNOWN-0")
    entity = _project_entity(payload)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=f"issue-{issue_key}",
        entity_path=entity,
        key=f"issue-{issue_key}-created",
        action="write",
        success=True,
        details={
            "event_type": "jira:issue_created",
            "issue_key": issue_key,
            "summary": fields.get("summary", ""),
            "issue_type": fields.get("issuetype", {}).get("name", ""),
            "priority": fields.get("priority", {}).get("name", ""),
            "status": fields.get("status", {}).get("name", ""),
            "assignee": _extract_user(fields.get("assignee")),
            "reporter": _extract_user(fields.get("reporter")),
            "labels": fields.get("labels", []),
        },
    )]


def _handle_issue_updated(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    issue = payload.get("issue", {})
    fields = issue.get("fields", {})
    issue_key = issue.get("key", "UNKNOWN-0")
    entity = _project_entity(payload)
    changelog = payload.get("changelog", {})
    changes = _extract_changes(changelog)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=f"issue-{issue_key}-update",
        entity_path=entity,
        key=f"issue-{issue_key}-updated",
        action="write",
        success=True,
        details={
            "event_type": "jira:issue_updated",
            "issue_key": issue_key,
            "summary": fields.get("summary", ""),
            "status": fields.get("status", {}).get("name", ""),
            "priority": fields.get("priority", {}).get("name", ""),
            "assignee": _extract_user(fields.get("assignee")),
            "changes": changes,
        },
    )]


def _handle_sprint_started(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    sprint = payload.get("sprint", {})
    sprint_name = sprint.get("name", "Unknown Sprint")
    sprint_id = sprint.get("id", str(uuid4())[:8])
    entity = _project_entity(payload)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=f"sprint-{sprint_id}-started",
        entity_path=entity,
        key=f"sprint-{sprint_id}-started",
        action="write",
        success=True,
        details={
            "event_type": "sprint_started",
            "sprint_id": sprint_id,
            "sprint_name": sprint_name,
            "state": sprint.get("state", "active"),
            "start_date": sprint.get("startDate", ""),
            "end_date": sprint.get("endDate", ""),
            "goal": sprint.get("goal", ""),
        },
    )]


def _handle_sprint_closed(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    sprint = payload.get("sprint", {})
    sprint_name = sprint.get("name", "Unknown Sprint")
    sprint_id = sprint.get("id", str(uuid4())[:8])
    entity = _project_entity(payload)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=f"sprint-{sprint_id}-closed",
        entity_path=entity,
        key=f"sprint-{sprint_id}-closed",
        action="write",
        success=True,
        details={
            "event_type": "sprint_closed",
            "sprint_id": sprint_id,
            "sprint_name": sprint_name,
            "state": sprint.get("state", "closed"),
            "start_date": sprint.get("startDate", ""),
            "end_date": sprint.get("endDate", ""),
            "complete_date": sprint.get("completeDate", ""),
            "goal": sprint.get("goal", ""),
        },
    )]


def _extract_user(user: dict[str, Any] | None) -> str:
    if not user:
        return "unassigned"
    return user.get("displayName", user.get("name", user.get("emailAddress", "unknown")))


def _extract_changes(changelog: dict[str, Any]) -> list[dict[str, str]]:
    items = changelog.get("items", [])
    return [
        {
            "field": item.get("field", ""),
            "from": item.get("fromString", ""),
            "to": item.get("toString", ""),
        }
        for item in items
    ]


_HANDLERS: dict[str, Any] = {
    "jira:issue_created": _handle_issue_created,
    "jira:issue_updated": _handle_issue_updated,
    "sprint_started": _handle_sprint_started,
    "sprint_closed": _handle_sprint_closed,
}
