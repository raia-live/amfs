"""Slack connector -- transforms Slack Events API payloads into AMFS context.

Handles message, app_mention, and reaction_added events,
transforming them into context records keyed by channel.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from amfs_connectors.base import ConnectorABC, ConnectorConfig, IngestionResult
from amfs_connectors.webhook import WebhookEvent


class SlackConfig(ConnectorConfig):
    """Slack-specific connector configuration."""

    connector_type: str = "slack"
    signing_secret: str | None = None
    tracked_events: list[str] = [
        "message",
        "app_mention",
        "reaction_added",
    ]
    max_text_length: int = 500


class SlackConnector(ConnectorABC):
    """Slack Events API connector.

    Registered as the ``slack`` entry point.
    """

    def __init__(self, config: ConnectorConfig | None = None) -> None:
        super().__init__(config or ConnectorConfig(
            name="slack",
            connector_type="slack",
            entity_path="slack",
        ))

    def validate_event(self, raw_event: dict[str, Any]) -> bool:
        if raw_event.get("type") == "url_verification":
            return True
        event = raw_event.get("event", {})
        return bool(event.get("type"))

    def extract_event_id(self, raw_event: dict[str, Any]) -> str:
        return raw_event.get("event_id", f"slack-{uuid4().hex[:12]}")

    def transform(self, raw_event: dict[str, Any]) -> list[IngestionResult]:
        if raw_event.get("type") == "url_verification":
            return [IngestionResult(
                connector_id=self._config.id,
                event_id="url_verification",
                entity_path=self._config.entity_path,
                key="url-verification",
                action="skip",
                success=True,
                details={"challenge": raw_event.get("challenge", "")},
            )]

        inner = raw_event.get("event", {})
        event = WebhookEvent(
            connector_id=self._config.id,
            source="slack",
            event_type=inner.get("type", "unknown"),
            payload=raw_event,
            headers={},
        )
        return _transform_slack_event(event)


def _transform_slack_event(event: WebhookEvent) -> list[IngestionResult]:
    handler = _HANDLERS.get(event.event_type)
    if handler:
        return handler(event)

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=event.payload.get("event_id", str(uuid4())[:8]),
        entity_path=_channel_entity(event.payload.get("event", {})),
        key=f"slack-event-{event.event_type}",
        action="context",
        success=True,
        details={"event_type": event.event_type},
    )]


def _channel_entity(inner: dict[str, Any]) -> str:
    channel = inner.get("channel", inner.get("item", {}).get("channel", "unknown"))
    return f"slack/{channel}"


def _truncate(text: str, max_len: int = 500) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _handle_message(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    inner = payload.get("event", {})

    if inner.get("subtype") == "bot_message":
        return [IngestionResult(
            connector_id=event.connector_id,
            event_id=payload.get("event_id", str(uuid4())[:8]),
            entity_path=_channel_entity(inner),
            key=f"msg-bot-{inner.get('ts', 'unknown')}",
            action="skip",
            success=True,
            details={"reason": "bot_message"},
        )]

    channel = _channel_entity(inner)
    ts = inner.get("ts", str(uuid4())[:8])
    user = inner.get("user", "unknown")
    text = _truncate(inner.get("text", ""))
    thread_ts = inner.get("thread_ts")

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=payload.get("event_id", f"slack-{ts}"),
        entity_path=channel,
        key=f"msg-{ts}",
        action="context",
        success=True,
        details={
            "event_type": "message",
            "user": user,
            "text": text,
            "thread_ts": thread_ts,
            "channel": inner.get("channel", ""),
            "is_thread_reply": thread_ts is not None and thread_ts != ts,
        },
    )]


def _handle_app_mention(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    inner = payload.get("event", {})
    channel = _channel_entity(inner)
    ts = inner.get("ts", str(uuid4())[:8])

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=payload.get("event_id", f"slack-mention-{ts}"),
        entity_path=channel,
        key=f"mention-{ts}",
        action="write",
        success=True,
        details={
            "event_type": "app_mention",
            "user": inner.get("user", "unknown"),
            "text": _truncate(inner.get("text", "")),
            "thread_ts": inner.get("thread_ts"),
            "channel": inner.get("channel", ""),
        },
    )]


def _handle_reaction_added(event: WebhookEvent) -> list[IngestionResult]:
    payload = event.payload
    inner = payload.get("event", {})
    item = inner.get("item", {})
    channel = _channel_entity(inner)
    reaction = inner.get("reaction", "unknown")

    return [IngestionResult(
        connector_id=event.connector_id,
        event_id=payload.get("event_id", f"slack-reaction-{reaction}"),
        entity_path=channel,
        key=f"reaction-{reaction}-{item.get('ts', 'unknown')}",
        action="context",
        success=True,
        details={
            "event_type": "reaction_added",
            "user": inner.get("user", "unknown"),
            "reaction": reaction,
            "item_type": item.get("type", ""),
            "item_ts": item.get("ts", ""),
            "channel": item.get("channel", ""),
        },
    )]


_HANDLERS: dict[str, Any] = {
    "message": _handle_message,
    "app_mention": _handle_app_mention,
    "reaction_added": _handle_reaction_added,
}
