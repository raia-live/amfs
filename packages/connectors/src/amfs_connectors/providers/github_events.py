"""GitHub connector -- transforms GitHub webhook events into AMFS memory.

Handles push, pull_request, issues, and deployment_status events,
transforming them into structured AMFS writes keyed by repository.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from amfs_connectors.base import ConnectorABC, ConnectorConfig, IngestionResult
from amfs_connectors.webhook import WebhookEvent


class GitHubConfig(ConnectorConfig):
    """GitHub-specific connector configuration."""

    connector_type: str = "github"
    webhook_secret: str | None = None
    tracked_events: list[str] = [
        "push",
        "pull_request",
        "issues",
        "deployment_status",
    ]


class GitHubConnector(ConnectorABC):
    """GitHub webhook event connector.

    Registered as the ``github`` entry point.
    """

    def __init__(self, config: ConnectorConfig | None = None) -> None:
        super().__init__(config or ConnectorConfig(
            name="github",
            connector_type="github",
            entity_path="github",
        ))

    def validate_event(self, raw_event: dict[str, Any]) -> bool:
        if "action" in raw_event and "repository" in raw_event:
            return True
        if "ref" in raw_event and "commits" in raw_event:
            return True
        if "deployment_status" in raw_event:
            return True
        return False

    def extract_event_id(self, raw_event: dict[str, Any]) -> str:
        if "hook_id" in raw_event and "action" in raw_event:
            return f"gh-{raw_event['hook_id']}-{raw_event['action']}"
        return f"gh-{uuid4().hex[:12]}"

    def transform(self, raw_event: dict[str, Any]) -> list[IngestionResult]:
        event = WebhookEvent(
            connector_id=self._config.id,
            source="github",
            event_type=_detect_event_type(raw_event),
            payload=raw_event,
            headers={},
        )
        return _transform_github_event(event)


def _detect_event_type(payload: dict[str, Any]) -> str:
    if "commits" in payload and "ref" in payload:
        return "push"
    if "pull_request" in payload:
        return "pull_request"
    if "issue" in payload and "pull_request" not in payload:
        return "issues"
    if "deployment_status" in payload:
        return "deployment_status"
    return payload.get("action", "unknown")


def _transform_github_event(event: WebhookEvent) -> list[IngestionResult]:
    handler = _HANDLERS.get(event.event_type)
    if handler:
        return handler(event)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=str(uuid4())[:8],
        entity_path=_repo_entity(event.payload),
        key=f"gh-event-{event.event_type}",
        action="context",
        success=True,
        details={"event_type": event.event_type},
    )]


def _repo_entity(payload: dict[str, Any]) -> str:
    repo = payload.get("repository", {})
    name = repo.get("full_name", repo.get("name", "unknown"))
    return f"github/{name}"


def _handle_push(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    repo = _repo_entity(payload)
    ref = payload.get("ref", "")
    branch = ref.rsplit("/", 1)[-1] if "/" in ref else ref
    commits = payload.get("commits", [])
    pusher = payload.get("pusher", {}).get("name", "unknown")
    head_commit = payload.get("head_commit", {})

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=head_commit.get("id", str(uuid4())[:8]),
        entity_path=repo,
        key=f"push-{branch}-{head_commit.get('id', 'latest')[:8]}",
        action="write",
        success=True,
        details={
            "event_type": "push",
            "branch": branch,
            "pusher": pusher,
            "commit_count": len(commits),
            "head_message": head_commit.get("message", ""),
            "head_sha": head_commit.get("id", ""),
            "compare_url": payload.get("compare", ""),
        },
    )]


def _handle_pull_request(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    repo = _repo_entity(payload)
    action = payload.get("action", "unknown")
    pr = payload.get("pull_request", {})
    number = pr.get("number", 0)
    merged = pr.get("merged", False)

    effective_action = "merged" if action == "closed" and merged else action

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=f"pr-{number}",
        entity_path=repo,
        key=f"pr-{number}-{effective_action}",
        action="write",
        success=True,
        details={
            "event_type": "pull_request",
            "action": effective_action,
            "number": number,
            "title": pr.get("title", ""),
            "author": pr.get("user", {}).get("login", "unknown"),
            "base_branch": pr.get("base", {}).get("ref", ""),
            "head_branch": pr.get("head", {}).get("ref", ""),
            "merged": merged,
            "html_url": pr.get("html_url", ""),
        },
    )]


def _handle_issues(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    repo = _repo_entity(payload)
    action = payload.get("action", "unknown")
    issue = payload.get("issue", {})
    number = issue.get("number", 0)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=f"issue-{number}",
        entity_path=repo,
        key=f"issue-{number}-{action}",
        action="write",
        success=True,
        details={
            "event_type": "issues",
            "action": action,
            "number": number,
            "title": issue.get("title", ""),
            "author": issue.get("user", {}).get("login", "unknown"),
            "state": issue.get("state", ""),
            "labels": [l.get("name", "") for l in issue.get("labels", [])],
            "html_url": issue.get("html_url", ""),
        },
    )]


def _handle_deployment_status(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    repo = _repo_entity(payload)
    dep_status = payload.get("deployment_status", {})
    deployment = payload.get("deployment", {})
    status_id = dep_status.get("id", str(uuid4())[:8])
    state = dep_status.get("state", "unknown")
    environment = dep_status.get("environment", deployment.get("environment", "unknown"))

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=f"deploy-{status_id}",
        entity_path=repo,
        key=f"deploy-{environment}-{state}",
        action="write",
        success=True,
        details={
            "event_type": "deployment_status",
            "state": state,
            "environment": environment,
            "description": dep_status.get("description", ""),
            "creator": dep_status.get("creator", {}).get("login", "unknown"),
            "ref": deployment.get("ref", ""),
            "sha": deployment.get("sha", ""),
            "target_url": dep_status.get("target_url", ""),
        },
    )]


_HANDLERS: dict[str, Any] = {
    "push": _handle_push,
    "pull_request": _handle_pull_request,
    "issues": _handle_issues,
    "deployment_status": _handle_deployment_status,
}
