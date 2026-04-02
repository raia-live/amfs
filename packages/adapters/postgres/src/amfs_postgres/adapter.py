"""PostgresAdapter — AMFS adapter backed by PostgreSQL with psycopg3."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg.rows import dict_row

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.exceptions import AdapterError, VersionConflictError
from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    ArtifactRef,
    MemoryEntry,
    MemoryType,
    OutcomeRecord,
    Provenance,
)

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


class PostgresAdapter(AdapterABC):
    """Store AMFS entries in PostgreSQL.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string.
    namespace:
        Logical namespace for this adapter instance.
    auto_schema:
        If True, create tables/triggers on init.
    """

    def __init__(
        self,
        dsn: str,
        namespace: str = "default",
        *,
        auto_schema: bool = True,
    ) -> None:
        self._dsn = dsn
        self._namespace = namespace
        self._conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        if auto_schema:
            self._apply_schema()
        self._listen_thread: threading.Thread | None = None
        self._listen_stop = threading.Event()
        self._watchers: dict[str, list[Callable[[MemoryEntry], None]]] = {}

    def _apply_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM amfs_memory_entries
                WHERE namespace = %s AND entity_path = %s AND key = %s
                  AND superseded_at IS NULL
                ORDER BY version DESC LIMIT 1
                """,
                (self._namespace, entity_path, key),
            )
            row = cur.fetchone()
        if row is None:
            return None
        entry = self._row_to_entry(row)
        if entry.confidence < min_confidence:
            return None
        return entry

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        with self._conn.transaction():
            with self._conn.cursor() as cur:
                # Find current version
                cur.execute(
                    """
                    SELECT version FROM amfs_memory_entries
                    WHERE namespace = %s AND entity_path = %s AND key = %s
                      AND superseded_at IS NULL
                    ORDER BY version DESC LIMIT 1
                    FOR UPDATE
                    """,
                    (self._namespace, entry.entity_path, entry.key),
                )
                row = cur.fetchone()
                current_version = row["version"] if row else 0
                new_version = current_version + 1

                if entry.version > 1 and entry.version != new_version:
                    raise VersionConflictError(
                        entry.entity_path, entry.key, entry.version, current_version
                    )

                # Supersede old version
                if row:
                    cur.execute(
                        """
                        UPDATE amfs_memory_entries
                        SET superseded_at = NOW()
                        WHERE namespace = %s AND entity_path = %s AND key = %s
                          AND superseded_at IS NULL
                        """,
                        (self._namespace, entry.entity_path, entry.key),
                    )

                # Insert new version
                entry_id = uuid.uuid4()
                cur.execute(
                    """
                    INSERT INTO amfs_memory_entries (
                        id, namespace, entity_path, key, version, value,
                        agent_id, session_id, written_at, pattern_refs,
                        confidence, outcome_count, ttl_at, memory_type,
                        artifact_refs
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(entry_id),
                        self._namespace,
                        entry.entity_path,
                        entry.key,
                        new_version,
                        json.dumps(entry.value, default=str),
                        entry.provenance.agent_id,
                        entry.provenance.session_id,
                        entry.provenance.written_at,
                        entry.provenance.pattern_refs,
                        entry.confidence,
                        entry.outcome_count,
                        entry.ttl_at,
                        entry.memory_type.value,
                        json.dumps([ref.model_dump(mode="json") for ref in entry.artifact_refs], default=str),
                    ),
                )

        return entry.model_copy(update={"version": new_version})

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        conditions = ["namespace = %s"]
        params: list[Any] = [self._namespace]

        if entity_path is not None:
            conditions.append("entity_path = %s")
            params.append(entity_path)

        if not include_superseded:
            conditions.append("superseded_at IS NULL")

        where = " AND ".join(conditions)
        query = f"SELECT * FROM amfs_memory_entries WHERE {where} ORDER BY entity_path, key, version"

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [self._row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # watch
    # ------------------------------------------------------------------

    def watch(
        self,
        entity_path: str,
        callback: Callable[[MemoryEntry], None],
    ) -> WatchHandle:
        if entity_path not in self._watchers:
            self._watchers[entity_path] = []
        self._watchers[entity_path].append(callback)
        self._ensure_listener()

        def cancel() -> None:
            cbs = self._watchers.get(entity_path, [])
            if callback in cbs:
                cbs.remove(callback)

        return WatchHandle(cancel)

    def _ensure_listener(self) -> None:
        if self._listen_thread is not None and self._listen_thread.is_alive():
            return
        self._listen_stop.clear()
        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="amfs-pg-listen"
        )
        self._listen_thread.start()

    def _listen_loop(self) -> None:
        try:
            listen_conn = psycopg.connect(self._dsn, autocommit=True)
            listen_conn.execute("LISTEN amfs_write")
            while not self._listen_stop.is_set():
                for notify in listen_conn.notifies(timeout=1.0):
                    try:
                        payload = json.loads(notify.payload)
                        ep = payload.get("entity_path", "")
                        for watch_ep, callbacks in self._watchers.items():
                            if ep == watch_ep or ep.startswith(watch_ep + "/"):
                                entry = self.read(ep, payload["key"])
                                if entry:
                                    for cb in callbacks:
                                        cb(entry)
                    except Exception:
                        logger.debug("Listen: error processing notify", exc_info=True)
            listen_conn.close()
        except Exception:
            logger.debug("Listen loop exited", exc_info=True)

    # ------------------------------------------------------------------
    # commit_outcome
    # ------------------------------------------------------------------

    def commit_outcome(self, record: OutcomeRecord) -> list[MemoryEntry]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO amfs_outcomes (
                    namespace, outcome_ref, outcome_type, causal_confidence,
                    committed_at, causal_entry_keys, agent_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    self._namespace,
                    record.outcome_ref,
                    record.outcome_type.value,
                    record.causal_confidence,
                    record.committed_at,
                    record.causal_entry_keys,
                    record.agent_id,
                ),
            )

        # Read back the affected entries (trigger already propagated)
        updated: list[MemoryEntry] = []
        for spec in record.causal_entry_keys:
            parts = spec.rsplit("/", 1)
            if len(parts) != 2:
                continue
            ep, key = parts
            entry = self.read(ep, key)
            if entry:
                updated.append(entry)
        return updated

    # ------------------------------------------------------------------
    # list_outcomes
    # ------------------------------------------------------------------

    def list_outcomes(
        self,
        *,
        entity_path: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list["OutcomeRecord"]:
        conditions = ["namespace = %s"]
        params: list[Any] = [self._namespace]

        if entity_path is not None:
            conditions.append("EXISTS (SELECT 1 FROM unnest(causal_entry_keys) AS ek WHERE ek LIKE %s)")
            params.append(f"{entity_path}/%")

        if since is not None:
            conditions.append("committed_at >= %s")
            params.append(since)

        where = " AND ".join(conditions)
        query = f"""
            SELECT outcome_ref, outcome_type, causal_confidence,
                   committed_at, causal_entry_keys, agent_id
            FROM amfs_outcomes
            WHERE {where}
            ORDER BY committed_at DESC
            LIMIT %s
        """
        params.append(limit)

        from amfs_core.models import OutcomeRecord, OutcomeType

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        results: list[OutcomeRecord] = []
        for row in rows:
            try:
                otype = OutcomeType(row["outcome_type"])
            except ValueError:
                continue
            results.append(OutcomeRecord(
                outcome_ref=row["outcome_ref"],
                outcome_type=otype,
                causal_confidence=float(row["causal_confidence"]),
                committed_at=row["committed_at"],
                causal_entry_keys=row.get("causal_entry_keys") or [],
                agent_id=row["agent_id"],
            ))
        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: dict[str, Any]) -> MemoryEntry:
        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value)
        raw_type = row.get("memory_type", "fact")
        try:
            memory_type = MemoryType(raw_type)
        except ValueError:
            memory_type = MemoryType.FACT
        return MemoryEntry(
            entity_path=row["entity_path"],
            key=row["key"],
            version=row["version"],
            value=value,
            provenance=Provenance(
                agent_id=row["agent_id"],
                session_id=row["session_id"],
                written_at=row["written_at"],
                pattern_refs=row.get("pattern_refs") or [],
            ),
            confidence=float(row["confidence"]),
            outcome_count=row["outcome_count"],
            ttl_at=row.get("ttl_at"),
            artifact_refs=[ArtifactRef.model_validate(r) for r in (row.get("artifact_refs") or [])],
            memory_type=memory_type,
        )

    def close(self) -> None:
        """Stop listener and close connection."""
        self._listen_stop.set()
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=2)
        self._conn.close()
