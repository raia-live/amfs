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
    Agent,
    ArtifactRef,
    Branch,
    BranchAccess,
    BranchAccessPermission,
    BranchStatus,
    DecisionTrace,
    DiffEntry,
    Digest,
    DigestType,
    ErrorEvent,
    Event,
    EventType,
    ExternalContext,
    MemoryEntry,
    MemoryStateDiff,
    MemoryStats,
    MemoryType,
    MergeConflict,
    MergeResult,
    MergeStrategy,
    OutcomeRecord,
    OutcomeType,
    PRReview,
    PRReviewStatus,
    PullRequest,
    PullRequestStatus,
    Provenance,
    GraphEdge,
    GraphNeighborQuery,
    QueryEvent,
    SearchQuery,
    SemanticQuery,
    Tag,
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


class _TenantRLSConnection:
    """Applies ``amfs.current_account_id`` when a request-scoped tenant is set."""

    def __init__(self, inner_ctx: Any) -> None:
        self._inner_ctx = inner_ctx

    def __enter__(self) -> Any:
        from amfs_postgres.tenant_context import get_request_tenant_account_id

        conn = self._inner_ctx.__enter__()
        tid = get_request_tenant_account_id()
        if tid:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('amfs.current_account_id', %s, false)",
                    (tid,),
                )
        return conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner_ctx.__exit__(exc_type, exc, tb)


class _TenantRLSPoolWrapper:
    """Wraps a psycopg pool so each checkout applies RLS session vars."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def connection(self) -> _TenantRLSConnection:
        return _TenantRLSConnection(self._inner.connection())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


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
        self._has_search_tsv = False
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
        self._pool = _TenantRLSPoolWrapper(self._pool)
        if auto_schema:
            self._apply_schema()
        self._detect_optional_columns()
        self._listen_thread: threading.Thread | None = None
        self._listen_stop = threading.Event()
        self._watchers: dict[str, list[Callable[[MemoryEntry], None]]] = {}

    def _apply_schema(self, *, retries: int = 3) -> None:
        import time as _time

        for attempt in range(1, retries + 1):
            try:
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(_SCHEMA_SQL)
                        self._apply_migrations(cur)
                return
            except Exception as exc:
                is_retryable = "deadlock" in str(exc).lower() or "lock" in str(exc).lower()
                if is_retryable and attempt < retries:
                    wait = 0.5 * (2 ** (attempt - 1))
                    logger.warning(
                        "Schema apply attempt %d/%d hit %s — retrying in %.1fs",
                        attempt, retries, type(exc).__name__, wait,
                    )
                    _time.sleep(wait)
                else:
                    raise

    @staticmethod
    def _apply_migrations(cur: psycopg.Cursor) -> None:
        """Additive migrations for columns added after initial schema."""
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS shared BOOLEAN NOT NULL DEFAULT TRUE
        """)
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS branch TEXT NOT NULL DEFAULT 'main'
        """)
        cur.execute("""
            ALTER TABLE amfs_outcomes
            ADD COLUMN IF NOT EXISTS branch TEXT NOT NULL DEFAULT 'main'
        """)
        cur.execute("""
            ALTER TABLE amfs_decision_traces
            ADD COLUMN IF NOT EXISTS branch TEXT NOT NULL DEFAULT 'main'
        """)
        cur.execute("""
            ALTER TABLE amfs_digests
            ADD COLUMN IF NOT EXISTS branch TEXT NOT NULL DEFAULT 'main'
        """)
        cur.execute("""
            ALTER TABLE amfs_digests DROP CONSTRAINT IF EXISTS uq_digest
        """)
        cur.execute("""
            ALTER TABLE amfs_digests
            ADD CONSTRAINT uq_digest UNIQUE (namespace, branch, digest_type, scope)
        """)
        cur.execute("""
            ALTER TABLE amfs_api_keys
            ADD COLUMN IF NOT EXISTS default_branch TEXT
        """)
        cur.execute("""
            ALTER TABLE amfs_detected_patterns
            ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'collaboration'
        """)
        cur.execute("""
            ALTER TABLE amfs_detected_patterns DROP CONSTRAINT IF EXISTS chk_pattern_type
        """)
        cur.execute("""
            ALTER TABLE amfs_detected_patterns ADD CONSTRAINT chk_pattern_type CHECK (
                pattern_type IN (
                    'knowledge_conflict', 'stale_knowledge', 'orphaned_branch',
                    'redundant_writes', 'single_point_of_knowledge', 'passive_consumer',
                    'unreviewed_changes', 'recurring_failure',
                    'hot_entity', 'stale_cluster', 'confidence_drift'
                )
            )
        """)
        cur.execute("""
            CREATE OR REPLACE FUNCTION amfs_notify_write() RETURNS TRIGGER AS $$
            BEGIN
                IF NEW.superseded_at IS NULL THEN
                    PERFORM pg_notify('amfs_write', json_build_object(
                        'namespace', NEW.namespace,
                        'entity_path', NEW.entity_path,
                        'key', NEW.key,
                        'version', NEW.version,
                        'agent_id', NEW.agent_id,
                        'branch', NEW.branch
                    )::TEXT);
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        # HMO feature columns (frequency decay, tiered memory, importance)
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS recall_count INTEGER DEFAULT 0
        """)
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS priority_score NUMERIC(10,6)
        """)
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS tier SMALLINT DEFAULT 3
        """)
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS importance_score NUMERIC(6,4)
        """)
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS importance_dimensions JSONB
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_entries_hot
            ON amfs_memory_entries (namespace, entity_path)
            WHERE tier = 1 AND superseded_at IS NULL
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_entries_warm
            ON amfs_memory_entries (namespace, entity_path)
            WHERE tier <= 2 AND superseded_at IS NULL
        """)
        cur.execute("""
            CREATE OR REPLACE FUNCTION amfs_propagate_outcome() RETURNS TRIGGER AS $$
            DECLARE
                multiplier NUMERIC;
                entry_key TEXT;
                parts TEXT[];
                ep TEXT;
                k TEXT;
                cur RECORD;
            BEGIN
                CASE NEW.outcome_type
                    WHEN 'critical_failure' THEN multiplier := 1.15;
                    WHEN 'failure' THEN multiplier := 1.10;
                    WHEN 'minor_failure' THEN multiplier := 1.08;
                    WHEN 'success' THEN multiplier := 0.97;
                    WHEN 'p1_incident' THEN multiplier := 1.15;
                    WHEN 'p2_incident' THEN multiplier := 1.10;
                    WHEN 'regression' THEN multiplier := 1.08;
                    WHEN 'clean_deploy' THEN multiplier := 0.97;
                    ELSE multiplier := 1.0;
                END CASE;

                FOREACH entry_key IN ARRAY NEW.causal_entry_keys
                LOOP
                    IF position('/' in entry_key) = 0 THEN
                        CONTINUE;
                    END IF;
                    k := substring(entry_key from '([^/]+)$');
                    ep := left(entry_key, length(entry_key) - length(k) - 1);

                    SELECT * INTO cur FROM amfs_memory_entries
                    WHERE namespace = NEW.namespace
                      AND entity_path = ep
                      AND key = k
                      AND superseded_at IS NULL
                    ORDER BY version DESC LIMIT 1;

                    IF FOUND THEN
                        UPDATE amfs_memory_entries
                        SET superseded_at = NOW()
                        WHERE id = cur.id;

                        INSERT INTO amfs_memory_entries (
                            namespace, entity_path, key, version, value,
                            agent_id, session_id, written_at, pattern_refs,
                            confidence, outcome_count, recall_count,
                            ttl_at, memory_type, shared, artifact_refs
                        ) VALUES (
                            cur.namespace, cur.entity_path, cur.key, cur.version + 1, cur.value,
                            cur.agent_id, cur.session_id, cur.written_at, cur.pattern_refs,
                            cur.confidence * multiplier * NEW.causal_confidence,
                            cur.outcome_count + 1, cur.recall_count,
                            cur.ttl_at, cur.memory_type,
                            cur.shared, cur.artifact_refs
                        );
                    END IF;
                END LOOP;

                PERFORM pg_notify('amfs_outcome', json_build_object(
                    'namespace', NEW.namespace,
                    'outcome_ref', NEW.outcome_ref,
                    'outcome_type', NEW.outcome_type,
                    'agent_id', NEW.agent_id,
                    'causal_confidence', NEW.causal_confidence
                )::TEXT);

                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        # Soft-delete for team members (user removal flow)
        cur.execute("""
            ALTER TABLE amfs_team_members
            ADD COLUMN IF NOT EXISTS removed_at TIMESTAMPTZ
        """)
        cur.execute("""
            ALTER TABLE amfs_team_members
            ADD COLUMN IF NOT EXISTS removed_by TEXT
        """)
        # Replace full unique constraint with partial index (active members only)
        cur.execute("""
            ALTER TABLE amfs_team_members
            DROP CONSTRAINT IF EXISTS uq_team_member
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_team_member_active
            ON amfs_team_members (team_id, email)
            WHERE removed_at IS NULL
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_team_members_removed
            ON amfs_team_members (namespace, email)
            WHERE removed_at IS NOT NULL
        """)

    def _detect_optional_columns(self) -> None:
        """Check whether optional columns (embedding, search_tsv) exist.

        These are added by migration 002_search_indexes.sql and may require
        the pgvector extension.  The adapter works without them.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'amfs_memory_entries'
                      AND column_name IN ('embedding', 'search_tsv')
                    """,
                )
                found = {row["column_name"] for row in cur.fetchall()}
                self._has_embedding_col = "embedding" in found
                self._has_search_tsv = "search_tsv" in found

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
        branch: str = "main",
    ) -> MemoryEntry | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM amfs_memory_entries
                    WHERE namespace = %s AND branch = %s
                      AND entity_path = %s AND key = %s
                      AND superseded_at IS NULL
                    ORDER BY version DESC LIMIT 1
                    """,
                    (self._namespace, branch, entity_path, key),
                )
                row = cur.fetchone()

                if row is None and branch != "main":
                    branch_info = cur.execute(
                        "SELECT parent_branch, branched_at FROM amfs_branches WHERE namespace = %s AND name = %s",
                        (self._namespace, branch),
                    ).fetchone()
                    if branch_info:
                        cur.execute(
                            """
                            SELECT * FROM amfs_memory_entries
                            WHERE namespace = %s AND branch = %s
                              AND entity_path = %s AND key = %s
                              AND superseded_at IS NULL
                              AND written_at <= %s
                            ORDER BY version DESC LIMIT 1
                            """,
                            (self._namespace, branch_info["parent_branch"],
                             entity_path, key, branch_info["branched_at"]),
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
        branch = entry.branch or "main"
        with self._pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT version FROM amfs_memory_entries
                        WHERE namespace = %s AND branch = %s
                          AND entity_path = %s AND key = %s
                          AND superseded_at IS NULL
                        ORDER BY version DESC LIMIT 1
                        FOR UPDATE
                        """,
                        (self._namespace, branch, entry.entity_path, entry.key),
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
                            WHERE namespace = %s AND branch = %s
                              AND entity_path = %s AND key = %s
                              AND superseded_at IS NULL
                            """,
                            (self._namespace, branch, entry.entity_path, entry.key),
                        )

                    entry_id = uuid.uuid4()

                    columns = [
                        "id", "namespace", "entity_path", "key", "version",
                        "value", "agent_id", "session_id", "written_at",
                        "pattern_refs", "confidence", "outcome_count",
                        "ttl_at", "memory_type", "shared", "artifact_refs",
                        "branch",
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
                        branch,
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
        branch: str = "main",
    ) -> list[MemoryEntry]:
        conditions = ["namespace = %s", "branch = %s"]
        params: list[Any] = [self._namespace, branch]

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

    def search(self, query: SearchQuery, *, branch: str = "main") -> list[MemoryEntry]:
        """Search using PostgreSQL full-text search when a text query is available.

        When ``SearchQuery.query`` is set and the ``search_tsv`` column exists
        the adapter adds a ``search_tsv @@ plainto_tsquery(...)`` filter and,
        when ``sort_by`` is the default ``"confidence"``, orders by ts_rank so
        keyword relevance is factored in.  Explicit ``sort_by`` values
        (``"recency"``, ``"version"``) are respected as-is.
        """
        use_fts = bool(query.query and getattr(self, "_has_search_tsv", False))

        conditions = ["namespace = %s", "branch = %s", "superseded_at IS NULL"]
        params: list[Any] = [self._namespace, branch]

        if query.depth < 3:
            conditions.append("tier <= %s")
            params.append(query.depth)

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

        if use_fts:
            conditions.append("search_tsv @@ plainto_tsquery('english', %s)")
            params.append(query.query)

        order_map = {
            "confidence": "confidence DESC",
            "recency": "written_at DESC",
            "version": "version DESC",
        }

        if use_fts and query.sort_by == "confidence":
            order = "ts_rank(search_tsv, plainto_tsquery('english', %s)) DESC, confidence DESC"
            params.append(query.query)
        else:
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
    # Recall tracking + tiered memory (mutable in-place updates)
    # ------------------------------------------------------------------

    def increment_recall_count(
        self,
        entity_path: str,
        key: str,
        *,
        branch: str = "main",
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """UPDATE amfs_memory_entries
                   SET recall_count = recall_count + 1
                   WHERE namespace = %s AND branch = %s
                     AND entity_path = %s AND key = %s
                     AND superseded_at IS NULL""",
                (self._namespace, branch, entity_path, key),
            )

    def update_tiers(
        self,
        tier_assignments: dict[str, int],
        scores: dict[str, float],
        *,
        branch: str = "main",
    ) -> int:
        if not tier_assignments:
            return 0
        updated = 0
        with self._pool.connection() as conn:
            for entry_key, tier_val in tier_assignments.items():
                if "/" not in entry_key:
                    continue
                k = entry_key.rsplit("/", 1)[-1]
                ep = entry_key[: len(entry_key) - len(k) - 1]
                score = scores.get(entry_key)
                result = conn.execute(
                    """UPDATE amfs_memory_entries
                       SET tier = %s, priority_score = %s
                       WHERE namespace = %s AND branch = %s
                         AND entity_path = %s AND key = %s
                         AND superseded_at IS NULL""",
                    (tier_val, score, self._namespace, branch, ep, k),
                )
                if result.rowcount:
                    updated += result.rowcount
        return updated

    # ------------------------------------------------------------------
    # Knowledge graph (Postgres implementation)
    # ------------------------------------------------------------------

    def upsert_graph_edge(
        self,
        edge: GraphEdge,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> GraphEdge:
        provenance_json = json.dumps(edge.provenance) if edge.provenance else None
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_knowledge_graph
                        (namespace, branch, source_entity, source_type, relation,
                         target_entity, target_type, confidence, evidence_count,
                         first_seen, last_seen, provenance)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (namespace, branch, source_entity, relation, target_entity)
                    DO UPDATE SET
                        evidence_count = amfs_knowledge_graph.evidence_count + 1,
                        last_seen = NOW(),
                        confidence = GREATEST(amfs_knowledge_graph.confidence, EXCLUDED.confidence),
                        provenance = EXCLUDED.provenance,
                        target_type = EXCLUDED.target_type,
                        source_type = EXCLUDED.source_type
                    RETURNING *
                    """,
                    (
                        namespace, branch,
                        edge.source_entity, edge.source_type, edge.relation,
                        edge.target_entity, edge.target_type,
                        edge.confidence, edge.evidence_count,
                        edge.first_seen, edge.last_seen,
                        provenance_json,
                    ),
                )
                row = cur.fetchone()

        if row is None:
            return edge
        return GraphEdge(
            source_entity=row["source_entity"],
            source_type=row["source_type"],
            relation=row["relation"],
            target_entity=row["target_entity"],
            target_type=row["target_type"],
            confidence=row["confidence"],
            evidence_count=row["evidence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            provenance=row["provenance"],
        )

    def graph_neighbors(
        self,
        query: GraphNeighborQuery,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> list[GraphEdge]:
        if query.depth <= 1:
            return self._graph_neighbors_single(query, namespace=namespace, branch=branch)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                direction_clause = self._graph_direction_seed(query.direction)
                entity_params = self._graph_entity_params(query.entity, query.direction)
                cur.execute(
                    f"""
                    WITH RECURSIVE graph_walk AS (
                        SELECT *, 1 AS depth FROM amfs_knowledge_graph
                        WHERE namespace = %s AND branch = %s
                          AND confidence >= %s
                          AND ({direction_clause})
                        UNION ALL
                        SELECT g.*, gw.depth + 1
                        FROM amfs_knowledge_graph g
                        JOIN graph_walk gw ON g.source_entity = gw.target_entity
                        WHERE gw.depth < %s
                          AND g.namespace = %s AND g.branch = %s
                          AND g.confidence >= %s
                    )
                    SELECT * FROM graph_walk
                    ORDER BY depth, confidence DESC
                    LIMIT %s
                    """,
                    (
                        namespace, branch, query.min_confidence,
                        *entity_params,
                        query.depth, namespace, branch, query.min_confidence,
                        query.limit,
                    ),
                )
                rows = cur.fetchall()

        return [self._row_to_graph_edge(r) for r in rows]

    def _graph_neighbors_single(
        self,
        query: GraphNeighborQuery,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> list[GraphEdge]:
        conditions = ["namespace = %s", "branch = %s", "confidence >= %s"]
        params: list[Any] = [namespace, branch, query.min_confidence]

        direction_clause = self._graph_direction_seed(query.direction)
        conditions.append(f"({direction_clause})")
        params.extend(self._graph_entity_params(query.entity, query.direction))

        if query.relation:
            conditions.append("relation = %s")
            params.append(query.relation)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM amfs_knowledge_graph
            WHERE {where}
            ORDER BY confidence DESC
            LIMIT %s
        """
        params.append(query.limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_graph_edge(r) for r in rows]

    @staticmethod
    def _graph_direction_seed(direction: str) -> str:
        src = "(source_entity = %s OR source_entity LIKE %s)"
        tgt = "(target_entity = %s OR target_entity LIKE %s)"
        if direction == "outgoing":
            return src
        elif direction == "incoming":
            return tgt
        return f"({src} OR {tgt})"

    @staticmethod
    def _graph_entity_params(entity: str, direction: str) -> list[str]:
        """Return SQL params for the direction seed clause with prefix matching."""
        prefix = entity + "/%"
        if direction == "outgoing":
            return [entity, prefix]
        elif direction == "incoming":
            return [entity, prefix]
        elif direction == "both":
            return [entity, prefix, entity, prefix]
        return [entity, prefix, entity, prefix]

    def list_graph_edges(
        self,
        *,
        entity: str | None = None,
        relation: str | None = None,
        min_confidence: float = 0.0,
        namespace: str = "default",
        branch: str = "main",
        limit: int = 500,
    ) -> list[GraphEdge]:
        conditions = ["namespace = %s", "branch = %s"]
        params: list[Any] = [namespace, branch]

        if entity is not None:
            prefix = entity + "/%"
            conditions.append(
                "((source_entity = %s OR source_entity LIKE %s)"
                " OR (target_entity = %s OR target_entity LIKE %s))"
            )
            params.extend([entity, prefix, entity, prefix])
        if relation is not None:
            conditions.append("relation = %s")
            params.append(relation)
        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM amfs_knowledge_graph
            WHERE {where}
            ORDER BY evidence_count DESC, confidence DESC
            LIMIT %s
        """
        params.append(limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_graph_edge(r) for r in rows]

    def graph_stats(
        self,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> dict:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_edges,
                        COUNT(DISTINCT source_entity) + COUNT(DISTINCT target_entity) as approx_nodes,
                        AVG(confidence) as avg_confidence,
                        AVG(evidence_count) as avg_evidence
                    FROM amfs_knowledge_graph
                    WHERE namespace = %s AND branch = %s
                    """,
                    (namespace, branch),
                )
                row = cur.fetchone()
                if not row:
                    return {}

                cur.execute(
                    """
                    SELECT relation, COUNT(*) as cnt
                    FROM amfs_knowledge_graph
                    WHERE namespace = %s AND branch = %s
                    GROUP BY relation ORDER BY cnt DESC
                    """,
                    (namespace, branch),
                )
                relation_counts = {r["relation"]: r["cnt"] for r in cur.fetchall()}

        return {
            "total_edges": row["total_edges"],
            "approx_nodes": row["approx_nodes"],
            "avg_confidence": float(row["avg_confidence"] or 0),
            "avg_evidence": float(row["avg_evidence"] or 0),
            "relations": relation_counts,
        }

    @staticmethod
    def _row_to_graph_edge(row: dict) -> GraphEdge:
        return GraphEdge(
            source_entity=row["source_entity"],
            source_type=row["source_type"],
            relation=row["relation"],
            target_entity=row["target_entity"],
            target_type=row["target_type"],
            confidence=row["confidence"],
            evidence_count=row["evidence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            provenance=row.get("provenance"),
        )

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
            recall_count=row.get("recall_count", 0),
            priority_score=float(row["priority_score"]) if row.get("priority_score") is not None else None,
            tier=row.get("tier", 3),
            importance_score=float(row["importance_score"]) if row.get("importance_score") is not None else None,
            importance_dimensions=row.get("importance_dimensions"),
            ttl_at=row.get("ttl_at"),
            artifact_refs=[ArtifactRef.model_validate(r) for r in (row.get("artifact_refs") or [])],
            memory_type=memory_type,
            shared=row.get("shared", True),
            branch=row.get("branch", "main"),
        )

    # ── Digest storage (Memory Cortex) ──────────────────────────────

    def upsert_digest(self, digest: Digest) -> None:
        """Insert or update a compiled digest."""
        branch = digest.branch or "main"
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO amfs_digests
                   (namespace, branch, digest_type, scope, summary, entry_count,
                    source_agents, anticipation_score, compiled_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT ON CONSTRAINT uq_digest
                   DO UPDATE SET summary = EXCLUDED.summary,
                                 entry_count = EXCLUDED.entry_count,
                                 source_agents = EXCLUDED.source_agents,
                                 anticipation_score = EXCLUDED.anticipation_score,
                                 compiled_at = EXCLUDED.compiled_at""",
                (
                    digest.namespace,
                    branch,
                    digest.digest_type.value,
                    digest.scope,
                    json.dumps(digest.summary),
                    digest.entry_count,
                    digest.source_agents,
                    digest.anticipation_score,
                    digest.compiled_at,
                ),
            )

    def get_digest(
        self,
        digest_type: DigestType,
        scope: str,
        namespace: str = "default",
        branch: str = "main",
    ) -> Digest | None:
        """Read a single digest by type and scope."""
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                row = cur.execute(
                    """SELECT * FROM amfs_digests
                       WHERE namespace = %s AND branch = %s
                         AND digest_type = %s AND scope = %s""",
                    (namespace, branch, digest_type.value, scope),
                ).fetchone()
        return self._row_to_digest(row) if row else None

    def list_digests(
        self,
        digest_type: DigestType | None = None,
        namespace: str = "default",
        branch: str = "main",
    ) -> list[Digest]:
        """List digests, optionally filtered by type and branch."""
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                if digest_type:
                    rows = cur.execute(
                        """SELECT * FROM amfs_digests
                           WHERE namespace = %s AND branch = %s AND digest_type = %s
                           ORDER BY compiled_at DESC""",
                        (namespace, branch, digest_type.value),
                    ).fetchall()
                else:
                    rows = cur.execute(
                        """SELECT * FROM amfs_digests
                           WHERE namespace = %s AND branch = %s
                           ORDER BY compiled_at DESC""",
                        (namespace, branch),
                    ).fetchall()
        return [self._row_to_digest(r) for r in rows]

    @staticmethod
    def _row_to_digest(row: dict[str, Any]) -> Digest:
        summary = row["summary"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        return Digest(
            digest_type=DigestType(row["digest_type"]),
            scope=row["scope"],
            summary=summary,
            entry_count=row.get("entry_count", 0),
            source_agents=row.get("source_agents") or [],
            compiled_at=row["compiled_at"],
            anticipation_score=float(row.get("anticipation_score", 0.0)),
            branch=row.get("branch", "main"),
            namespace=row.get("namespace", "default"),
        )

    # ── Agent registration (Pro) ────────────────────────────────────────

    def ensure_agent(self, agent_id: str, namespace: str = "default") -> Agent:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_agents (namespace, agent_id)
                    VALUES (%s, %s)
                    ON CONFLICT ON CONSTRAINT uq_agent DO UPDATE
                        SET last_active_at = NOW(),
                            entry_count = amfs_agents.entry_count + 1
                    RETURNING id, namespace, agent_id, display_name,
                              created_at, last_active_at, entry_count
                    """,
                    (namespace, agent_id),
                )
                row = cur.fetchone()
        return self._row_to_agent(row)

    def get_agent(self, agent_id: str, namespace: str = "default") -> Agent | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, namespace, agent_id, display_name,
                           created_at, last_active_at, entry_count
                    FROM amfs_agents
                    WHERE namespace = %s AND agent_id = %s
                    """,
                    (namespace, agent_id),
                )
                row = cur.fetchone()
        return self._row_to_agent(row) if row else None

    def list_agents(self, namespace: str = "default") -> list[Agent]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, namespace, agent_id, display_name,
                           created_at, last_active_at, entry_count
                    FROM amfs_agents
                    WHERE namespace = %s
                    ORDER BY last_active_at DESC
                    """,
                    (namespace,),
                )
                rows = cur.fetchall()
        return [self._row_to_agent(r) for r in rows]

    @staticmethod
    def _row_to_agent(row: dict[str, Any]) -> Agent:
        return Agent(
            id=str(row["id"]),
            namespace=row["namespace"],
            agent_id=row["agent_id"],
            display_name=row.get("display_name"),
            created_at=row["created_at"],
            last_active_at=row["last_active_at"],
            entry_count=row.get("entry_count", 0),
        )

    # ── Event log / timeline (Pro) ────────────────────────────────────

    def log_event(self, event: Event) -> Event:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_events
                        (namespace, agent_id, branch, event_type,
                         summary, details, actor_agent_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        event.namespace,
                        event.agent_id,
                        event.branch,
                        event.event_type.value if isinstance(event.event_type, EventType) else event.event_type,
                        event.summary,
                        json.dumps(event.details, default=str),
                        event.actor_agent_id,
                        event.created_at,
                    ),
                )
                row = cur.fetchone()
        return event.model_copy(update={"id": str(row["id"]), "created_at": row["created_at"]})

    def list_events(
        self,
        agent_id: str,
        namespace: str = "default",
        *,
        branch: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Event]:
        conditions = ["namespace = %s", "agent_id = %s"]
        params: list[Any] = [namespace, agent_id]

        if branch is not None:
            conditions.append("branch = %s")
            params.append(branch)
        if event_type is not None:
            conditions.append("event_type = %s")
            params.append(event_type)
        if since is not None:
            conditions.append("created_at >= %s")
            params.append(since)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT id, namespace, agent_id, branch, event_type,
                   summary, details, actor_agent_id, created_at
            FROM amfs_events
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT %s
        """
        params.append(limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_event(r) for r in rows]

    def get_event(self, event_id: str, namespace: str = "default") -> Event | None:
        sql = """
            SELECT id, namespace, agent_id, branch, event_type,
                   summary, details, actor_agent_id, created_at
            FROM amfs_events
            WHERE id = %s AND namespace = %s
            LIMIT 1
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (event_id, namespace))
                row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    @staticmethod
    def _row_to_event(row: dict[str, Any]) -> Event:
        details = row.get("details") or {}
        if isinstance(details, str):
            details = json.loads(details)
        try:
            etype = EventType(row["event_type"])
        except ValueError:
            etype = EventType.WRITE
        return Event(
            id=str(row["id"]),
            namespace=row["namespace"],
            agent_id=row["agent_id"],
            branch=row.get("branch", "main"),
            event_type=etype,
            summary=row.get("summary"),
            details=details,
            actor_agent_id=row.get("actor_agent_id"),
            created_at=row["created_at"],
        )

    # ── Branch management (Pro) ───────────────────────────────────────

    def create_branch(self, branch: Branch) -> Branch:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_branches
                        (namespace, name, parent_branch, branched_at,
                         created_by, description, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        branch.namespace,
                        branch.name,
                        branch.parent_branch,
                        branch.branched_at,
                        branch.created_by,
                        branch.description,
                        branch.status.value if isinstance(branch.status, BranchStatus) else branch.status,
                    ),
                )
                row = cur.fetchone()
        return branch.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
        })

    def get_branch(self, name: str, namespace: str = "default") -> Branch | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM amfs_branches
                    WHERE namespace = %s AND name = %s
                    """,
                    (namespace, name),
                )
                row = cur.fetchone()
        return self._row_to_branch(row) if row else None

    def list_branches(
        self, namespace: str = "default", *, status: str | None = None
    ) -> list[Branch]:
        conditions = ["namespace = %s"]
        params: list[Any] = [namespace]
        if status is not None:
            conditions.append("status = %s")
            params.append(status)

        where = " AND ".join(conditions)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM amfs_branches
                    WHERE {where}
                    ORDER BY created_at DESC
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_branch(r) for r in rows]

    def close_branch(self, name: str, namespace: str = "default") -> Branch:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE amfs_branches
                    SET status = 'closed'
                    WHERE namespace = %s AND name = %s
                    RETURNING *
                    """,
                    (namespace, name),
                )
                row = cur.fetchone()
        if row is None:
            raise AdapterError(f"Branch '{name}' not found")
        return self._row_to_branch(row)

    def diff_branch(self, name: str, namespace: str = "default") -> list[DiffEntry]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT parent_branch, branched_at FROM amfs_branches WHERE namespace = %s AND name = %s",
                    (namespace, name),
                )
                branch_row = cur.fetchone()
                if branch_row is None:
                    return []

                parent = branch_row["parent_branch"]
                branched_at = branch_row["branched_at"]

                cur.execute(
                    """
                    SELECT be.entity_path, be.key, be.value AS branch_value,
                           be.confidence AS branch_confidence, be.shared AS branch_shared,
                           pe.value AS parent_value, pe.confidence AS parent_confidence,
                           CASE WHEN pe.id IS NULL THEN 'added' ELSE 'modified' END AS diff_type
                    FROM amfs_memory_entries be
                    LEFT JOIN amfs_memory_entries pe
                        ON pe.namespace = be.namespace
                       AND pe.entity_path = be.entity_path
                       AND pe.key = be.key
                       AND pe.branch = %s
                       AND pe.superseded_at IS NULL
                    WHERE be.namespace = %s AND be.branch = %s
                      AND be.superseded_at IS NULL
                    """,
                    (parent, namespace, name),
                )
                rows = cur.fetchall()

        results: list[DiffEntry] = []
        for r in rows:
            bv = r["branch_value"]
            if isinstance(bv, str):
                try:
                    bv = json.loads(bv)
                except (json.JSONDecodeError, ValueError):
                    pass
            pv = r.get("parent_value")
            if isinstance(pv, str):
                try:
                    pv = json.loads(pv)
                except (json.JSONDecodeError, ValueError):
                    pass
            results.append(DiffEntry(
                entity_path=r["entity_path"],
                key=r["key"],
                diff_type=r["diff_type"],
                branch_value=bv,
                parent_value=pv,
                branch_confidence=float(r["branch_confidence"]) if r.get("branch_confidence") is not None else None,
                parent_confidence=float(r["parent_confidence"]) if r.get("parent_confidence") is not None else None,
                branch_shared=r.get("branch_shared"),
            ))
        return results

    def merge_branch(
        self,
        name: str,
        namespace: str = "default",
        *,
        strategy: MergeStrategy = MergeStrategy.FAST_FORWARD,
        resolve_conflicts: dict[str, str] | None = None,
    ) -> MergeResult:
        resolve_conflicts = resolve_conflicts or {}

        with self._pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM amfs_branches WHERE namespace = %s AND name = %s FOR UPDATE",
                        (namespace, name),
                    )
                    branch_row = cur.fetchone()
                    if branch_row is None:
                        raise AdapterError(f"Branch '{name}' not found")
                    if branch_row["status"] != "active":
                        raise AdapterError(f"Branch '{name}' is not active (status: {branch_row['status']})")

                    parent = branch_row["parent_branch"]
                    branched_at = branch_row["branched_at"]

                    cur.execute(
                        """
                        SELECT be.*, pe.value AS parent_value, pe.version AS parent_version,
                               pe.id AS parent_id,
                               CASE WHEN pe.id IS NULL THEN 'added' ELSE 'modified' END AS diff_type
                        FROM amfs_memory_entries be
                        LEFT JOIN amfs_memory_entries pe
                            ON pe.namespace = be.namespace
                           AND pe.entity_path = be.entity_path
                           AND pe.key = be.key
                           AND pe.branch = %s
                           AND pe.superseded_at IS NULL
                        WHERE be.namespace = %s AND be.branch = %s
                          AND be.superseded_at IS NULL
                        """,
                        (parent, namespace, name),
                    )
                    branch_entries = cur.fetchall()

                    conflicts: list[MergeConflict] = []
                    merged = 0

                    for be in branch_entries:
                        entry_spec = f"{be['entity_path']}/{be['key']}"

                        if be["diff_type"] == "modified":
                            cur.execute(
                                """
                                SELECT version FROM amfs_memory_entries
                                WHERE namespace = %s AND branch = %s
                                  AND entity_path = %s AND key = %s
                                  AND written_at > %s AND superseded_at IS NULL
                                ORDER BY version DESC LIMIT 1
                                """,
                                (namespace, parent, be["entity_path"], be["key"], branched_at),
                            )
                            main_changed = cur.fetchone()
                            if main_changed:
                                if strategy == MergeStrategy.BRANCH_WINS or resolve_conflicts.get(entry_spec) == "branch":
                                    pass  # proceed with merge
                                elif strategy == MergeStrategy.MAIN_WINS or resolve_conflicts.get(entry_spec) == "main":
                                    continue
                                else:
                                    pv = be.get("parent_value")
                                    if isinstance(pv, str):
                                        try:
                                            pv = json.loads(pv)
                                        except (json.JSONDecodeError, ValueError):
                                            pass
                                    bv = be["value"]
                                    if isinstance(bv, str):
                                        try:
                                            bv = json.loads(bv)
                                        except (json.JSONDecodeError, ValueError):
                                            pass
                                    conflicts.append(MergeConflict(
                                        entity_path=be["entity_path"],
                                        key=be["key"],
                                        branch_value=bv,
                                        main_value=pv,
                                        branch_version=be["version"],
                                        main_version=main_changed["version"],
                                    ))
                                    continue

                        if be.get("parent_id"):
                            cur.execute(
                                "UPDATE amfs_memory_entries SET superseded_at = NOW() WHERE id = %s",
                                (be["parent_id"],),
                            )

                        cur.execute(
                            """
                            SELECT COALESCE(MAX(version), 0) + 1 AS next_ver
                            FROM amfs_memory_entries
                            WHERE namespace = %s AND branch = %s
                              AND entity_path = %s AND key = %s
                            """,
                            (namespace, parent, be["entity_path"], be["key"]),
                        )
                        next_ver = cur.fetchone()["next_ver"]

                        columns = [
                            "namespace", "entity_path", "key", "version", "value",
                            "agent_id", "session_id", "written_at", "pattern_refs",
                            "confidence", "outcome_count", "ttl_at", "memory_type",
                            "shared", "artifact_refs", "branch",
                        ]
                        params_list: list[Any] = [
                            namespace, be["entity_path"], be["key"], next_ver,
                            json.dumps(be["value"], default=str),
                            be["agent_id"], be["session_id"], be["written_at"],
                            be.get("pattern_refs") or [],
                            be["confidence"], be["outcome_count"],
                            be.get("ttl_at"), be.get("memory_type", "fact"),
                            be.get("shared", True), json.dumps(be.get("artifact_refs") or [], default=str),
                            parent,
                        ]
                        cols_sql = ", ".join(columns)
                        placeholders = ", ".join(["%s"] * len(params_list))
                        cur.execute(
                            f"INSERT INTO amfs_memory_entries ({cols_sql}) VALUES ({placeholders})",
                            params_list,
                        )
                        merged += 1

                    if conflicts:
                        return MergeResult(
                            branch_name=name,
                            status="conflicts",
                            merged_entries=merged,
                            conflicts=conflicts,
                        )

                    cur.execute(
                        """
                        UPDATE amfs_branches
                        SET status = 'merged', merged_at = NOW(), merged_by = %s
                        WHERE namespace = %s AND name = %s
                        """,
                        (branch_row["created_by"], namespace, name),
                    )

        return MergeResult(branch_name=name, status="merged", merged_entries=merged)

    @staticmethod
    def _row_to_branch(row: dict[str, Any]) -> Branch:
        try:
            status = BranchStatus(row["status"])
        except ValueError:
            status = BranchStatus.ACTIVE
        return Branch(
            id=str(row["id"]),
            namespace=row["namespace"],
            name=row["name"],
            parent_branch=row["parent_branch"],
            branched_at=row["branched_at"],
            created_by=row["created_by"],
            description=row.get("description"),
            status=status,
            merged_at=row.get("merged_at"),
            merged_by=row.get("merged_by"),
            created_at=row.get("created_at"),
        )

    # ── Branch access control (Pro) ───────────────────────────────────

    def grant_branch_access(self, access: BranchAccess) -> BranchAccess:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_branch_access
                        (namespace, branch_name, grantee_type, grantee_id,
                         permission, granted_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT uq_branch_access DO UPDATE
                        SET permission = EXCLUDED.permission,
                            granted_by = EXCLUDED.granted_by,
                            granted_at = NOW()
                    RETURNING id, granted_at
                    """,
                    (
                        access.namespace,
                        access.branch_name,
                        access.grantee_type,
                        access.grantee_id,
                        access.permission.value if isinstance(access.permission, BranchAccessPermission) else access.permission,
                        access.granted_by,
                    ),
                )
                row = cur.fetchone()
        return access.model_copy(update={
            "id": str(row["id"]),
            "granted_at": row["granted_at"],
        })

    def revoke_branch_access(
        self, branch_name: str, grantee_type: str, grantee_id: str,
        namespace: str = "default",
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                DELETE FROM amfs_branch_access
                WHERE namespace = %s AND branch_name = %s
                  AND grantee_type = %s AND grantee_id = %s
                """,
                (namespace, branch_name, grantee_type, grantee_id),
            )

    def list_branch_access(
        self, branch_name: str, namespace: str = "default"
    ) -> list[BranchAccess]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM amfs_branch_access
                    WHERE namespace = %s AND branch_name = %s
                    ORDER BY granted_at DESC
                    """,
                    (namespace, branch_name),
                )
                rows = cur.fetchall()
        return [self._row_to_branch_access(r) for r in rows]

    def check_branch_access(
        self, branch_name: str, api_key_id: str, namespace: str = "default"
    ) -> str | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ba.permission FROM amfs_branch_access ba
                    WHERE ba.namespace = %s AND ba.branch_name = %s
                      AND (
                        (ba.grantee_type = 'api_key' AND ba.grantee_id = %s)
                        OR (ba.grantee_type = 'team' AND ba.grantee_id IN (
                            SELECT t.slug FROM amfs_teams t
                            JOIN amfs_team_members tm ON tm.team_id = t.id
                            JOIN amfs_api_keys ak ON ak.id::text = %s
                            WHERE tm.email = ak.name AND tm.namespace = %s
                        ))
                      )
                    LIMIT 1
                    """,
                    (namespace, branch_name, api_key_id, api_key_id, namespace),
                )
                row = cur.fetchone()
        return row["permission"] if row else None

    @staticmethod
    def _row_to_branch_access(row: dict[str, Any]) -> BranchAccess:
        try:
            perm = BranchAccessPermission(row["permission"])
        except ValueError:
            perm = BranchAccessPermission.READ
        return BranchAccess(
            id=str(row["id"]),
            namespace=row["namespace"],
            branch_name=row["branch_name"],
            grantee_type=row["grantee_type"],
            grantee_id=row["grantee_id"],
            permission=perm,
            granted_by=row["granted_by"],
            granted_at=row.get("granted_at"),
        )

    # ── Tags / Snapshots (Pro) ────────────────────────────────────────

    def create_tag(self, tag: Tag) -> Tag:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_tags
                        (namespace, name, branch, tagged_at, description, created_by, event_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (tag.namespace, tag.name, tag.branch, tag.tagged_at,
                     tag.description, tag.created_by, tag.event_id),
                )
                row = cur.fetchone()
        return tag.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
        })

    def get_tag(self, name: str, namespace: str = "default") -> Tag | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM amfs_tags WHERE namespace = %s AND name = %s",
                    (namespace, name),
                )
                row = cur.fetchone()
        return self._row_to_tag(row) if row else None

    def list_tags(
        self, namespace: str = "default", *, branch: str | None = None
    ) -> list[Tag]:
        conditions = ["namespace = %s"]
        params: list[Any] = [namespace]
        if branch is not None:
            conditions.append("branch = %s")
            params.append(branch)
        where = " AND ".join(conditions)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM amfs_tags WHERE {where} ORDER BY tagged_at DESC",
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_tag(r) for r in rows]

    def delete_tag(self, name: str, namespace: str = "default") -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM amfs_tags WHERE namespace = %s AND name = %s",
                (namespace, name),
            )

    @staticmethod
    def _row_to_tag(row: dict[str, Any]) -> Tag:
        return Tag(
            id=str(row["id"]),
            namespace=row["namespace"],
            name=row["name"],
            branch=row.get("branch", "main"),
            tagged_at=row["tagged_at"],
            description=row.get("description"),
            created_by=row["created_by"],
            created_at=row.get("created_at"),
            event_id=row.get("event_id"),
        )

    # ── Pull Requests (Pro) ─────────────────────────────────────────────

    def create_pull_request(self, pr: PullRequest) -> PullRequest:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_pull_requests
                        (namespace, title, description, source_branch,
                         target_branch, status, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at, updated_at
                    """,
                    (pr.namespace, pr.title, pr.description, pr.source_branch,
                     pr.target_branch, pr.status.value if isinstance(pr.status, PullRequestStatus) else pr.status,
                     pr.created_by),
                )
                row = cur.fetchone()
        return pr.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    def get_pull_request(self, pr_id: str, namespace: str = "default") -> PullRequest | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM amfs_pull_requests WHERE id = %s::uuid AND namespace = %s",
                    (pr_id, namespace),
                )
                row = cur.fetchone()
        return self._row_to_pr(row) if row else None

    def list_pull_requests(
        self, namespace: str = "default", *, status: str | None = None
    ) -> list[PullRequest]:
        conditions = ["namespace = %s"]
        params: list[Any] = [namespace]
        if status:
            conditions.append("status = %s")
            params.append(status)
        where = " AND ".join(conditions)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM amfs_pull_requests WHERE {where} ORDER BY created_at DESC",
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_pr(r) for r in rows]

    def update_pull_request_status(
        self, pr_id: str, status: str, *, by: str = "", namespace: str = "default"
    ) -> PullRequest:
        updates = ["status = %s", "updated_at = NOW()"]
        params: list[Any] = [status]

        if status == "merged":
            updates.extend(["merged_at = NOW()", "merged_by = %s"])
            params.append(by)
        elif status == "closed":
            updates.extend(["closed_at = NOW()", "closed_by = %s"])
            params.append(by)

        params.extend([pr_id, namespace])
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""UPDATE amfs_pull_requests SET {', '.join(updates)}
                        WHERE id = %s::uuid AND namespace = %s RETURNING *""",
                    params,
                )
                row = cur.fetchone()
        if row is None:
            raise AdapterError(f"PR '{pr_id}' not found")
        return self._row_to_pr(row)

    def add_pr_review(self, review: PRReview) -> PRReview:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_pr_reviews
                        (namespace, pr_id, reviewer, status, comment, entry_path)
                    VALUES (%s, %s::uuid, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (review.namespace, review.pr_id, review.reviewer,
                     review.status.value if isinstance(review.status, PRReviewStatus) else review.status,
                     review.comment, review.entry_path),
                )
                row = cur.fetchone()
        return review.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
        })

    def list_pr_reviews(self, pr_id: str, namespace: str = "default") -> list[PRReview]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM amfs_pr_reviews WHERE pr_id = %s::uuid AND namespace = %s ORDER BY created_at",
                    (pr_id, namespace),
                )
                rows = cur.fetchall()
        results: list[PRReview] = []
        for r in rows:
            try:
                st = PRReviewStatus(r["status"])
            except ValueError:
                st = PRReviewStatus.COMMENTED
            results.append(PRReview(
                id=str(r["id"]),
                namespace=r["namespace"],
                pr_id=str(r["pr_id"]),
                reviewer=r["reviewer"],
                status=st,
                comment=r.get("comment"),
                entry_path=r.get("entry_path"),
                created_at=r.get("created_at"),
            ))
        return results

    @staticmethod
    def _row_to_pr(row: dict[str, Any]) -> PullRequest:
        try:
            st = PullRequestStatus(row["status"])
        except ValueError:
            st = PullRequestStatus.OPEN
        return PullRequest(
            id=str(row["id"]),
            namespace=row["namespace"],
            title=row["title"],
            description=row.get("description"),
            source_branch=row["source_branch"],
            target_branch=row["target_branch"],
            status=st,
            created_by=row["created_by"],
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            merged_at=row.get("merged_at"),
            merged_by=row.get("merged_by"),
            closed_at=row.get("closed_at"),
            closed_by=row.get("closed_by"),
            merge_strategy=row.get("merge_strategy"),
        )

    # ── Rollback (Pro) ────────────────────────────────────────────────

    def rollback_to_timestamp(
        self,
        agent_id: str,
        branch: str,
        timestamp: datetime,
        namespace: str = "default",
    ) -> int:
        """Supersede all entries written after timestamp and restore state."""
        restored = 0
        with self._pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT DISTINCT entity_path, key FROM amfs_memory_entries
                        WHERE namespace = %s AND branch = %s
                          AND written_at > %s AND superseded_at IS NULL
                        """,
                        (namespace, branch, timestamp),
                    )
                    to_revert = cur.fetchall()

                    for row in to_revert:
                        ep, key = row["entity_path"], row["key"]
                        cur.execute(
                            """
                            UPDATE amfs_memory_entries
                            SET superseded_at = NOW()
                            WHERE namespace = %s AND branch = %s
                              AND entity_path = %s AND key = %s
                              AND superseded_at IS NULL
                            """,
                            (namespace, branch, ep, key),
                        )
                        cur.execute(
                            """
                            SELECT * FROM amfs_memory_entries
                            WHERE namespace = %s AND branch = %s
                              AND entity_path = %s AND key = %s
                              AND written_at <= %s
                            ORDER BY version DESC LIMIT 1
                            """,
                            (namespace, branch, ep, key, timestamp),
                        )
                        old = cur.fetchone()
                        if old:
                            cur.execute(
                                """
                                UPDATE amfs_memory_entries
                                SET superseded_at = NULL
                                WHERE id = %s
                                """,
                                (old["id"],),
                            )
                            restored += 1
        return restored

    # ── Fork (Pro) ────────────────────────────────────────────────────

    def fork_agent(
        self,
        source_agent_id: str,
        target_agent_id: str,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> int:
        """Copy all live entries from source agent's branch into target agent's main."""
        self.ensure_agent(target_agent_id, namespace)
        copied = 0
        with self._pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM amfs_memory_entries
                        WHERE namespace = %s AND branch = %s
                          AND superseded_at IS NULL
                        """,
                        (namespace, branch),
                    )
                    rows = cur.fetchall()

                    for row in rows:
                        cur.execute(
                            """
                            INSERT INTO amfs_memory_entries
                            (namespace, entity_path, key, version, value,
                             agent_id, session_id, tool_context, pattern_refs,
                             written_at, confidence, outcome_count,
                             ttl_at, artifact_refs, memory_type, shared,
                             branch, embedding)
                            VALUES
                            (%s, %s, %s, 1, %s,
                             %s, %s, %s, %s,
                             NOW(), %s, 0,
                             %s, %s, %s, %s,
                             'main', %s)
                            ON CONFLICT ON CONSTRAINT uq_entry_version DO NOTHING
                            """,
                            (
                                namespace,
                                row["entity_path"],
                                row["key"],
                                json.dumps(row["value"]) if not isinstance(row["value"], str) else row["value"],
                                target_agent_id,
                                row.get("session_id", ""),
                                row.get("tool_context", ""),
                                row.get("pattern_refs", []),
                                row["confidence"],
                                row.get("ttl_at"),
                                row.get("artifact_refs", []),
                                row.get("memory_type", "fact"),
                                row.get("shared", False),
                                row.get("embedding"),
                            ),
                        )
                        copied += cur.rowcount
        return copied

    def close(self) -> None:
        """Stop listener and close connection pool."""
        self._listen_stop.set()
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=2)
        self._pool.close()
