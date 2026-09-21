"""Server-Sent Events for real-time AMFS watch notifications.

Subscribers are ``asyncio.Queue`` objects owned by the event loop that opened
the stream. Broadcasts, though, arrive from wherever the write happened: an
``async`` route on that loop, or — since the sync-bodied routes run on FastAPI's
threadpool — a worker thread. ``asyncio.Queue.put_nowait`` is not thread-safe:
called from another thread it appends the item but wakes the waiting consumer
through ``call_soon``, which does not rouse a sleeping loop, so the event sits
until something else happens to wake it (and raises under asyncio debug mode).
The manager records each queue's loop at subscribe time and, when a broadcast
comes from outside it, hands the put to ``call_soon_threadsafe``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from amfs_core.models import MemoryEntry

logger = logging.getLogger(__name__)


class SSEManager:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._room_subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        # The loop each queue belongs to, keyed by the queue's id. Subscribe and
        # unsubscribe run on that loop; broadcasts may run on any thread, so the
        # registries are read under the lock and iterated as copies.
        self._loops: dict[int, asyncio.AbstractEventLoop] = {}
        self._lock = threading.Lock()

    # ── Subscriptions ────────────────────────────────────────────────

    def _new_queue(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:  # subscribed outside a loop (tests); deliver directly
            loop = None
        if loop is not None:
            with self._lock:
                self._loops[id(queue)] = loop
        return queue

    def _forget_queue(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._loops.pop(id(queue), None)

    def subscribe(self, entity_path: str = "*") -> asyncio.Queue[dict[str, Any]]:
        queue = self._new_queue()
        with self._lock:
            self._subscribers.setdefault(entity_path, []).append(queue)
        return queue

    def unsubscribe(self, entity_path: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subs = self._subscribers.get(entity_path, [])
            if queue in subs:
                subs.remove(queue)
        self._forget_queue(queue)

    def subscribe_room(self, room_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to all events for a specific room."""
        queue = self._new_queue()
        with self._lock:
            self._room_subscribers.setdefault(room_id, []).append(queue)
        return queue

    def unsubscribe_room(self, room_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subs = self._room_subscribers.get(room_id, [])
            if queue in subs:
                subs.remove(queue)
        self._forget_queue(queue)

    # ── Delivery ─────────────────────────────────────────────────────

    @staticmethod
    def _put(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any], what: str) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug("SSE queue full, dropping %s", what)

    def _deliver(
        self, queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any], what: str
    ) -> None:
        """Put ``event`` on ``queue`` from whatever thread this is.

        On the queue's own loop the put is direct. From any other thread — a
        threadpool worker running a sync route — it is scheduled onto that loop
        with ``call_soon_threadsafe``, which is the one asyncio entry point that
        may be called from outside the loop and that wakes it.
        """
        with self._lock:
            owner = self._loops.get(id(queue))
        if owner is None:
            self._put(queue, event, what)
            return
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is owner:
            self._put(queue, event, what)
            return
        if owner.is_closed():
            return
        try:
            owner.call_soon_threadsafe(self._put, queue, event, what)
        except RuntimeError:  # the loop closed between the check and the call
            logger.debug("SSE subscriber loop closed, dropping %s", what)

    def broadcast(self, entry: MemoryEntry) -> None:
        data = entry.model_dump(mode="json")
        data.pop("embedding", None)
        event = {"type": "write", "entry": data}
        with self._lock:
            targets = [
                queue
                for pattern, queues in self._subscribers.items()
                if (
                    pattern == "*"
                    or entry.entity_path == pattern
                    or entry.entity_path.startswith(pattern + "/")
                )
                for queue in queues
            ]
        for queue in targets:
            self._deliver(queue, event, "event")

    def broadcast_room_event(
        self,
        room_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast a room-scoped event (join, leave, write, read, etc.)."""
        event = {"type": event_type, "room_id": room_id, **data}
        with self._lock:
            targets = list(self._room_subscribers.get(room_id, []))
        for queue in targets:
            # Each subscriber pops "type" from its own copy in the generator.
            self._deliver(queue, dict(event), f"event for room {room_id}")

    # ── Generators ───────────────────────────────────────────────────

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
