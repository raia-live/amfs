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

    async def event_generator(self, entity_path: str = "*"):
        queue = self.subscribe(entity_path)
        try:
            while True:
                event = await queue.get()
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["entry"], default=str),
                }
        finally:
            self.unsubscribe(entity_path, queue)
