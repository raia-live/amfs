"""Server-Sent Events for real-time AMFS watch notifications."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from amfs_core.models import MemoryEntry

logger = logging.getLogger(__name__)


class SSEManager:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._room_subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, entity_path: str = "*") -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if entity_path not in self._subscribers:
            self._subscribers[entity_path] = []
        self._subscribers[entity_path].append(queue)
        return queue

    def unsubscribe(self, entity_path: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subscribers.get(entity_path, [])
        if queue in subs:
            subs.remove(queue)

    def subscribe_room(self, room_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to all events for a specific room."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if room_id not in self._room_subscribers:
            self._room_subscribers[room_id] = []
        self._room_subscribers[room_id].append(queue)
        return queue

    def unsubscribe_room(self, room_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._room_subscribers.get(room_id, [])
        if queue in subs:
            subs.remove(queue)

    def broadcast(self, entry: MemoryEntry) -> None:
        data = entry.model_dump(mode="json")
        data.pop("embedding", None)
        event = {"type": "write", "entry": data}
        for pattern, queues in self._subscribers.items():
            if (
                pattern == "*"
                or entry.entity_path == pattern
                or entry.entity_path.startswith(pattern + "/")
            ):
                for queue in queues:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.debug("SSE queue full, dropping event")

    def broadcast_room_event(
        self,
        room_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast a room-scoped event (join, leave, write, read, etc.)."""
        event = {"type": event_type, "room_id": room_id, **data}
        queues = self._room_subscribers.get(room_id, [])
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Room SSE queue full, dropping event for room %s", room_id)

    async def event_generator(self, entity_path: str = "*", predicate=None):
        """Yield SSE events, optionally filtered per subscriber.

        ``predicate`` receives the entry dict and returns True when the
        subscriber may see it (used for per-user visibility). Events that
        fail the predicate — or whose predicate raises — are dropped.
        """
        queue = self.subscribe(entity_path)
        try:
            while True:
                event = await queue.get()
                if predicate is not None:
                    try:
                        if not predicate(event.get("entry", {})):
                            continue
                    except Exception:
                        logger.debug("SSE visibility predicate failed — dropping event", exc_info=True)
                        continue
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["entry"], default=str),
                }
        finally:
            self.unsubscribe(entity_path, queue)

    async def room_event_generator(self, room_id: str):
        """SSE generator for room-scoped events."""
        queue = self.subscribe_room(room_id)
        try:
            while True:
                event = await queue.get()
                event_type = event.pop("type", "room_event")
                yield {
                    "event": event_type,
                    "data": json.dumps(event, default=str),
                }
        finally:
            self.unsubscribe_room(room_id, queue)
