"""PostgresAdapter — AMFS adapter backed by PostgreSQL with psycopg3."""

from __future__ import annotations

import contextlib
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
from amfs_core.embedder import EmbedderABC
from amfs_core.exceptions import AdapterError, VersionConflictError
from amfs_core.models import (
    OUTCOME_MULTIPLIERS,
    ArtifactRef,
    DecisionTrace,
    ErrorEvent,
    ExternalContext,
    MemoryEntry,
    MemoryStateDiff,
    MemoryStats,
    MemoryType,
    OutcomeRecord,
    OutcomeType,
    Provenance,
    QueryEvent,
    SearchQuery,
    SemanticQuery,
    TraceEntry,
)

try:
    from psycopg_pool import ConnectionPool as _ConnectionPool
except ImportError:  # pragma: no cover
    _ConnectionPool = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


class _SingleConnectionPool:
    """Minimal pool shim wrapping a single connection for environments
    without ``psycopg_pool`` installed.  Provides the same
    ``.connection()`` context-manager interface so the adapter code
    doesn't need branching.
    """

    def __init__(self, dsn: str, **kwargs: Any) -> None:
        connect_kwargs = kwargs.get("kwargs", {})
        self._conn = psycopg.connect(dsn, **connect_kwargs)

    @contextlib.contextmanager
    def connection(self):  # noqa: ANN204
        yield self._conn

    def close(self) -> None:
        self._conn.close()


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
    min_pool_size:
        Minimum number of connections in the pool (requires ``psycopg_pool``).
    max_pool_size:
        Maximum number of connections in the pool (requires ``psycopg_pool``).
    """

    def __init__(
        self,
        dsn: str,
        namespace: str = "default",
        *,
        auto_schema: bool = True,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._namespace = namespace
        self._has_embedding_col = False
        pool_kwargs: dict[str, Any] = {
            "kwargs": {"row_factory": dict_row, "autocommit": True},
        }
        if _ConnectionPool is not None:
            self._pool = _ConnectionPool(
                dsn,
                min_size=min_pool_size,
                max_size=max_pool_size,
                **pool_kwargs,
            )
        else:
            logger.info(
                "psycopg_pool not installed — falling back to single connection"
            )
            self._pool = _SingleConnectionPool(dsn, **pool_kwargs)
        if auto_schema:
            self._apply_schema()
        self._detect_optional_columns()
        self._listen_thread: threading.Thread | None = None
        self._listen_stop = threading.Event()
        self._watchers: dict[str, list[Callable[[MemoryEntry], None]]] = {}

    def _apply_schema(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)

    def _detect_optional_columns(self) -> None:
        """Check whether optional columns (embedding, search_tsv) exist.

        These are added by migration 002_search_indexes.sql and require
        the pgvector extension.  The adapter works without them.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'amfs_memory_entries'
                      AND column_name = 'embedding'
                    """,
                )
                self._has_embedding_col = cur.fetchone() is not None

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
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
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
    # read_at_version (historical)
    # ------------------------------------------------------------------

    def read_at_version(
        self,
        entity_path: str,
        key: str,
        version: int,
    ) -> MemoryEntry | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM amfs_memory_entries
                    WHERE namespace = %s AND entity_path = %s AND key = %s
                      AND version = %s
                    """,
                    (self._namespace, entity_path, key, version),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        with self._pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
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
                            entry.entity_path,
                            entry.key,
                            entry.version,
                            current_version,
                        )

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

                    entry_id = uuid.uuid4()

                    columns = [
                        "id", "namespace", "entity_path", "key", "version",
                        "value", "agent_id", "session_id", "written_at",
                        "pattern_refs", "confidence", "outcome_count",
                        "ttl_at", "memory_type", "shared", "artifact_refs",
                    ]
                    params: list[Any] = [
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
                        entry.shared,
                        json.dumps(
                            [ref.model_dump(mode="json") for ref in entry.artifact_refs],
                            default=str,
                        ),
                    ]

                    if self._has_embedding_col and entry.embedding:
                        columns.append("embedding")
                        params.append(
                            f"[{','.join(str(v) for v in entry.embedding)}]"
                        )

                    cols_sql = ", ".join(columns)
                    placeholders = ", ".join(["%s"] * len(params))
                    cur.execute(
                        f"""
                        INSERT INTO amfs_memory_entries ({cols_sql})
                        VALUES ({placeholders})
                        """,
                        params,
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

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [self._row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # search (full-text via tsvector when available, SQL filter otherwise)
    # ------------------------------------------------------------------

    def search(self, query: SearchQuery) -> list[MemoryEntry]:
        """Search using PostgreSQL full-text search when a text query is available."""
        conditions = ["namespace = %s", "superseded_at IS NULL"]
        params: list[Any] = [self._namespace]

        if query.entity_path is not None:
            conditions.append("entity_path = %s")
            params.append(query.entity_path)
        if query.min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(query.min_confidence)
        if query.max_confidence is not None:
            conditions.append("confidence <= %s")
            params.append(query.max_confidence)
        if query.agent_id is not None:
            conditions.append("agent_id = %s")
            params.append(query.agent_id)
        if query.since is not None:
            conditions.append("written_at >= %s")
            params.append(query.since)
        if query.pattern_ref is not None:
            conditions.append("%s = ANY(pattern_refs)")
            params.append(query.pattern_ref)

        order_map = {
            "confidence": "confidence DESC",
            "recency": "written_at DESC",
            "version": "version DESC",
        }
        order = order_map.get(query.sort_by, "confidence DESC")

        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM amfs_memory_entries
            WHERE {where}
            ORDER BY {order}
            LIMIT %s
        """
        params.append(query.limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # semantic_search (pgvector cosine similarity)
    # ------------------------------------------------------------------

    def semantic_search(
        self, query: SemanticQuery, embedder: EmbedderABC
    ) -> list[tuple[MemoryEntry, float]]:
        """Search using pgvector cosine similarity.

        Falls back to the base in-memory implementation when the
        ``embedding`` column is not available (pgvector not installed).
        """
        if not self._has_embedding_col:
            return super().semantic_search(query, embedder)

        query_vec = embedder.embed(query.text)

        conditions = ["namespace = %s", "superseded_at IS NULL", "embedding IS NOT NULL"]
        params: list[Any] = [self._namespace]

        if query.entity_path is not None:
            conditions.append("entity_path = %s")
            params.append(query.entity_path)
        if query.min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(query.min_confidence)

        where = " AND ".join(conditions)
        vec_str = f"[{','.join(str(v) for v in query_vec)}]"

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT *, 1 - (embedding <=> %s::vector) AS similarity
                    FROM amfs_memory_entries
                    WHERE {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [vec_str] + params + [vec_str, query.limit],
                )
                rows = cur.fetchall()

        results: list[tuple[MemoryEntry, float]] = []
        for row in rows:
            sim = float(row.get("similarity", 0))
            if sim >= query.min_similarity:
                results.append((self._row_to_entry(row), sim))
        return results

    # ------------------------------------------------------------------
    # stats (SQL aggregates)
    # ------------------------------------------------------------------

    def stats(self) -> MemoryStats:
        """Compute stats using SQL aggregates instead of full scan."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        COUNT(DISTINCT entity_path) as total_entities,
                        COUNT(DISTINCT agent_id) as total_agents,
                        AVG(confidence) as confidence_avg,
                        MIN(confidence) as confidence_min,
                        MAX(confidence) as confidence_max,
                        COUNT(*) FILTER (WHERE outcome_count > 0) as outcome_linked_count,
                        MIN(written_at) as oldest_entry_at,
                        MAX(written_at) as newest_entry_at
                    FROM amfs_memory_entries
                    WHERE namespace = %s AND superseded_at IS NULL
                    """,
                    (self._namespace,),
                )
                row = cur.fetchone()

                cur.execute(
                    """
                    SELECT agent_id, COUNT(*) as cnt
                    FROM amfs_memory_entries
                    WHERE namespace = %s AND superseded_at IS NULL
                    GROUP BY agent_id
                    """,
                    (self._namespace,),
                )
                agent_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT entity_path, COUNT(*) as cnt
                    FROM amfs_memory_entries
                    WHERE namespace = %s AND superseded_at IS NULL
                    GROUP BY entity_path
                    """,
                    (self._namespace,),
                )
                entity_rows = cur.fetchall()

        if not row or row["total_entries"] == 0:
            return MemoryStats()

        return MemoryStats(
            total_entries=row["total_entries"],
            total_entities=row["total_entities"],
            total_agents=row["total_agents"],
            agents={r["agent_id"]: r["cnt"] for r in agent_rows},
            entities={r["entity_path"]: r["cnt"] for r in entity_rows},
            confidence_avg=float(row["confidence_avg"] or 0),
            confidence_min=float(row["confidence_min"] or 0),
            confidence_max=float(row["confidence_max"] or 0),
            outcome_linked_count=row["outcome_linked_count"],
            oldest_entry_at=row["oldest_entry_at"],
            newest_entry_at=row["newest_entry_at"],
        )

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
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
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
    ) -> list[OutcomeRecord]:
        conditions = ["namespace = %s"]
        params: list[Any] = [self._namespace]

        if entity_path is not None:
            conditions.append(
                "EXISTS (SELECT 1 FROM unnest(causal_entry_keys) AS ek WHERE ek LIKE %s)"
            )
            params.append(f"{entity_path}/%")

        if since is not None:
            conditions.append("committed_at >= %s")
            params.append(since)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT outcome_ref, outcome_type, causal_confidence,
                   committed_at, causal_entry_keys, agent_id
            FROM amfs_outcomes
            WHERE {where}
            ORDER BY committed_at DESC
            LIMIT %s
        """
        params.append(limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[OutcomeRecord] = []
        for row in rows:
            try:
                otype = OutcomeType(row["outcome_type"])
            except ValueError:
                continue
            results.append(
                OutcomeRecord(
                    outcome_ref=row["outcome_ref"],
                    outcome_type=otype,
                    causal_confidence=float(row["causal_confidence"]),
                    committed_at=row["committed_at"],
                    causal_entry_keys=row.get("causal_entry_keys") or [],
                    agent_id=row["agent_id"],
                )
            )
        return results

    # ------------------------------------------------------------------
    # decision traces
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> DecisionTrace | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, agent_id, session_id, outcome_ref, outcome_type,
                           decision_summary, causal_entries, external_contexts,
                           query_events, session_started_at, session_ended_at,
                           session_duration_ms,
                           error_events, state_diff, created_at
                    FROM amfs_decision_traces
                    WHERE id = %s AND namespace = %s
                    """,
                    (trace_id, self._namespace),
                )
                row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_trace(row)

    def save_trace(self, trace: DecisionTrace) -> DecisionTrace:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_decision_traces
                        (namespace, agent_id, session_id, outcome_ref, outcome_type,
                         decision_summary, causal_entries, external_contexts,
                         query_events, session_started_at, session_ended_at,
                         session_duration_ms,
                         error_events, state_diff, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                            %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                    RETURNING id, created_at
                    """,
                    (
                        self._namespace,
                        trace.agent_id,
                        trace.session_id,
                        trace.outcome_ref,
                        trace.outcome_type,
                        trace.decision_summary,
                        json.dumps([e.model_dump(mode="json") for e in trace.causal_entries]),
                        json.dumps([c.model_dump(mode="json") for c in trace.external_contexts]),
                        json.dumps([q.model_dump(mode="json") for q in trace.query_events]),
                        trace.session_started_at,
                        trace.session_ended_at,
                        trace.session_duration_ms,
                        json.dumps([e.model_dump(mode="json") for e in trace.error_events]),
                        json.dumps(trace.state_diff.model_dump(mode="json")) if trace.state_diff else None,
                        trace.created_at,
                    ),
                )
                row = cur.fetchone()
        return trace.model_copy(update={"id": str(row["id"]), "created_at": row["created_at"]})

    def list_traces(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        limit: int = 100,
    ) -> list[DecisionTrace]:
        conditions = ["namespace = %s"]
        params: list[Any] = [self._namespace]

        if entity_path is not None:
            conditions.append(
                "EXISTS (SELECT 1 FROM jsonb_array_elements(causal_entries) AS ce "
                "WHERE ce->>'entity_path' = %s)"
            )
            params.append(entity_path)
        if agent_id is not None:
            conditions.append("agent_id = %s")
            params.append(agent_id)
        if outcome_type is not None:
            conditions.append("outcome_type = %s")
            params.append(outcome_type)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT id, agent_id, session_id, outcome_ref, outcome_type,
                   decision_summary, causal_entries, external_contexts,
                   query_events, session_started_at, session_ended_at,
                   session_duration_ms,
                   error_events, state_diff, created_at
            FROM amfs_decision_traces
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT %s
        """
        params.append(limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_trace(row) for row in rows]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _row_to_trace(self, row: dict[str, Any]) -> DecisionTrace:
        ce_raw = row["causal_entries"] or []
        ec_raw = row["external_contexts"] or []
        qe_raw = row.get("query_events") or []
        ee_raw = row.get("error_events") or []
        sd_raw = row.get("state_diff")
        if isinstance(ce_raw, str):
            ce_raw = json.loads(ce_raw)
        if isinstance(ec_raw, str):
            ec_raw = json.loads(ec_raw)
        if isinstance(qe_raw, str):
            qe_raw = json.loads(qe_raw)
        if isinstance(ee_raw, str):
            ee_raw = json.loads(ee_raw)
        if isinstance(sd_raw, str):
            sd_raw = json.loads(sd_raw)

        duration = row.get("session_duration_ms")
        if duration is not None:
            duration = float(duration)

        return DecisionTrace(
            id=str(row["id"]),
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            outcome_ref=row.get("outcome_ref"),
            outcome_type=row.get("outcome_type"),
            decision_summary=row.get("decision_summary"),
            causal_entries=[TraceEntry(**e) for e in ce_raw],
            external_contexts=[ExternalContext(**c) for c in ec_raw],
            query_events=[QueryEvent(**q) for q in qe_raw],
            error_events=[ErrorEvent(**e) for e in ee_raw],
            state_diff=MemoryStateDiff(**sd_raw) if sd_raw else None,
            session_started_at=row.get("session_started_at"),
            session_ended_at=row.get("session_ended_at"),
            session_duration_ms=duration,
            created_at=row["created_at"],
            namespace=self._namespace,
        )

    @staticmethod
    def _row_to_entry(row: dict[str, Any]) -> MemoryEntry:
        value = row["value"]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
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
            shared=row.get("shared", True),
        )

    def close(self) -> None:
        """Stop listener and close connection pool."""
        self._listen_stop.set()
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=2)
        self._pool.close()
