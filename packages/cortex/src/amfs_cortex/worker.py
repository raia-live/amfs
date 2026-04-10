"""CortexWorker — event-driven streaming worker that keeps digests warm.

Listens to Postgres NOTIFY events for new writes and triggers debounced
digest recompilation. Designed to run as a standalone process or embedded
in the HTTP server.

Supports a drift gate: only recompiles a scope when its fingerprint
has changed beyond a configurable threshold since the last compilation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from amfs_cortex.compiler import DigestCompiler

logger = logging.getLogger(__name__)

_ACTIVITY_LOG_MAX = 200


class CortexWorker:
    """Event-driven streaming worker that keeps digests warm."""

    def __init__(
        self,
        dsn: str,
        compiler: DigestCompiler,
        debounce_ms: int = 3000,
        use_advisory_lock: bool = True,
        drift_threshold: float = 0.0,
    ) -> None:
        self._dsn = dsn
        self._compiler = compiler
        self._debounce_ms = debounce_ms
        self._use_advisory_lock = use_advisory_lock
        self._drift_threshold = drift_threshold
        self._pending: dict[str, float] = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._events_processed = 0
        self._digests_compiled = 0
        self._drift_skipped = 0
        self._started_at: float | None = None
        self._last_event_at: float | None = None
        self._activity_log: deque[dict[str, Any]] = deque(maxlen=_ACTIVITY_LOG_MAX)
        self._outcome_wiring: Any = None
        self._hot_tracker: Any = None
        self._pro_forwarder: Any = None
        self._throughput_buckets: deque[dict[str, Any]] = deque(maxlen=60)
        self._current_bucket_ts: float = 0
        self._current_bucket_count: int = 0
        self._anchor_fingerprints: dict[str, str] = {}

    @property
    def activity_log(self) -> list[dict[str, Any]]:
        """Recent compilation and event activity."""
        return list(self._activity_log)

    @property
    def throughput(self) -> list[dict[str, Any]]:
        """Per-minute throughput buckets (last 60 minutes)."""
        return list(self._throughput_buckets)

    @property
    def stats(self) -> dict:
        """Current worker statistics."""
        return {
            "events_processed": self._events_processed,
            "digests_compiled": self._digests_compiled,
            "drift_skipped": self._drift_skipped,
            "pending_scopes": len(self._pending),
            "uptime_seconds": round(time.monotonic() - self._started_at, 1) if self._started_at else 0,
            "last_event_ago_ms": round((time.monotonic() - self._last_event_at) * 1000) if self._last_event_at else None,
            "drift_threshold": self._drift_threshold,
            "anchored_scopes": len(self._anchor_fingerprints),
        }

    def run(self) -> None:
        """Main event loop. Blocks until stop() is called.

        Automatically reconnects on connection loss with exponential backoff.
        """
        self._started_at = time.monotonic()
        logger.info("Cortex worker starting (debounce=%dms, advisory_lock=%s)",
                     self._debounce_ms, self._use_advisory_lock)

        backoff = 1.0
        max_backoff = 60.0

        while not self._stop.is_set():
            try:
                self._run_once()
                # Clean exit (stop() was called)
                break
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception(
                    "Cortex worker connection lost — reconnecting in %.0fs", backoff
                )
                self._activity_log.append({
                    "type": "connection_lost",
                    "backoff_s": backoff,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self._stop.wait(backoff)
                backoff = min(backoff * 2, max_backoff)

        logger.info("Cortex worker stopped")

    def _run_once(self) -> None:
        """Single connection lifecycle. Raises on connection loss."""
        import psycopg

        conn = psycopg.connect(self._dsn, autocommit=True)
        try:
            if self._use_advisory_lock:
                if not self._try_acquire_lock(conn):
                    self._standby_loop(conn)
                    return

            logger.info("Cortex worker active — listening for events")

            self._compiler.recompile_all()

            conn.execute("LISTEN amfs_write")
            conn.execute("LISTEN amfs_outcome")

            recompile_thread = threading.Thread(
                target=self._recompile_loop, daemon=True, name="cortex-recompile"
            )
            recompile_thread.start()

            while not self._stop.is_set():
                for notify in conn.notifies(timeout=5.0):
                    if self._stop.is_set():
                        break
                    self._handle_notify(notify)

        finally:
            conn.close()

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop.set()

    def _try_acquire_lock(self, conn) -> bool:
        """Try to acquire the advisory lock. Returns True if acquired."""
        row = conn.execute(
            "SELECT pg_try_advisory_lock(hashtext('amfs_cortex_worker'))"
        ).fetchone()
        acquired = row[0] if row else False
        if acquired:
            logger.info("Advisory lock acquired — this worker is active")
        else:
            logger.info("Advisory lock held by another worker — entering standby")
        return acquired

    def _standby_loop(self, conn) -> None:
        """Poll for the advisory lock until acquired or stopped."""
        while not self._stop.is_set():
            self._stop.wait(30.0)
            if self._stop.is_set():
                break
            if self._try_acquire_lock(conn):
                logger.info("Standby worker promoted to active")
                self._compiler.recompile_all()
                conn.execute("LISTEN amfs_write")
                conn.execute("LISTEN amfs_outcome")

                recompile_thread = threading.Thread(
                    target=self._recompile_loop, daemon=True, name="cortex-recompile"
                )
                recompile_thread.start()

                while not self._stop.is_set():
                    for notify in conn.notifies(timeout=5.0):
                        if self._stop.is_set():
                            break
                        self._handle_notify(notify)
                break

    def _handle_notify(self, notify) -> None:
        """Process a NOTIFY event from Postgres."""
        self._events_processed += 1
        self._last_event_at = time.monotonic()
        self._record_throughput()

        try:
            payload = json.loads(notify.payload) if notify.payload else {}
        except (json.JSONDecodeError, TypeError):
            return

        channel = notify.channel

        if channel == "amfs_write":
            entity_path = payload.get("entity_path", "")
            agent_id = payload.get("agent_id", "")
            branch = payload.get("branch", "main")
            if entity_path:
                with self._lock:
                    self._pending[f"entity:{entity_path}@{branch}"] = time.monotonic()
            if agent_id:
                with self._lock:
                    if agent_id.startswith(("webhook/", "external/")):
                        source = agent_id.split("/", 1)[1]
                        self._pending[f"source:{source}@{branch}"] = time.monotonic()
                    else:
                        self._pending[f"agent:{agent_id}@{branch}"] = time.monotonic()

                if self._hot_tracker:
                    self._hot_tracker.record_activity(agent_id, entity_path, "write")

            self._activity_log.append({
                "type": "event_received",
                "channel": channel,
                "entity_path": entity_path,
                "agent_id": agent_id,
                "branch": branch,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        elif channel == "amfs_outcome":
            if self._outcome_wiring:
                updated = self._outcome_wiring.process_outcome_event(payload)
                self._activity_log.append({
                    "type": "outcome_processed",
                    "outcome_ref": payload.get("outcome_ref", ""),
                    "outcome_type": payload.get("outcome_type", ""),
                    "digests_updated": updated,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        if self._pro_forwarder:
            self._pro_forwarder.enqueue({"channel": channel, **payload})

    def _recompile_loop(self) -> None:
        """Background thread that recompiles debounced pending digests."""
        while not self._stop.is_set():
            self._stop.wait(0.5)
            self._recompile_pending()

    def _recompile_pending(self) -> None:
        """Recompile all digests that have been pending longer than debounce."""
        now = time.monotonic()
        ready: list[str] = []

        with self._lock:
            for scope, ts in list(self._pending.items()):
                if (now - ts) * 1000 >= self._debounce_ms:
                    ready.append(scope)
            for scope in ready:
                del self._pending[scope]

        for scope_with_branch in ready:
            if "@" in scope_with_branch:
                scope, branch = scope_with_branch.rsplit("@", 1)
            else:
                scope, branch = scope_with_branch, "main"

            if self._drift_threshold > 0 and not self._should_recompile(scope_with_branch, scope, branch):
                continue

            try:
                t0 = time.monotonic()
                result = self._compiler.compile(scope, branch=branch)
                elapsed_ms = round((time.monotonic() - t0) * 1000)
                if result:
                    self._digests_compiled += 1
                    fp = self._compiler.compute_scope_fingerprint(scope, branch=branch)
                    if fp:
                        self._anchor_fingerprints[scope_with_branch] = fp
                    self._activity_log.append({
                        "type": "digest_compiled",
                        "scope": scope,
                        "branch": branch,
                        "digest_type": result.digest_type.value,
                        "entry_count": result.entry_count,
                        "elapsed_ms": elapsed_ms,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception:
                logger.exception("Failed to compile digest for %s (branch=%s)", scope, branch)
                self._activity_log.append({
                    "type": "compilation_error",
                    "scope": scope,
                    "branch": branch,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    def _should_recompile(self, scope_with_branch: str, scope: str, branch: str) -> bool:
        """Check whether the scope has drifted enough to warrant recompilation."""
        anchor = self._anchor_fingerprints.get(scope_with_branch)
        if anchor is None:
            return True

        current = self._compiler.compute_scope_fingerprint(scope, branch=branch)
        if current is None:
            return True
        if current == anchor:
            self._drift_skipped += 1
            logger.debug("Drift gate: skipping %s (fingerprint unchanged)", scope)
            self._activity_log.append({
                "type": "drift_skipped",
                "scope": scope,
                "branch": branch,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return False

        return True

    def _record_throughput(self) -> None:
        """Track per-minute event throughput."""
        now = time.time()
        bucket = int(now // 60) * 60
        if bucket != self._current_bucket_ts:
            if self._current_bucket_ts > 0:
                self._throughput_buckets.append({
                    "timestamp": datetime.fromtimestamp(
                        self._current_bucket_ts, tz=timezone.utc
                    ).isoformat(),
                    "events": self._current_bucket_count,
                })
            self._current_bucket_ts = bucket
            self._current_bucket_count = 0
        self._current_bucket_count += 1
