"""AsyncPostgresAdapter — async psycopg3 adapter for AMFS HTTP server hot paths.

Standalone async adapter that mirrors the sync PostgresAdapter's hot-path
methods using psycopg AsyncConnectionPool. Used exclusively by the HTTP
server to avoid blocking the FastAPI event loop.

The sync PostgresAdapter, SDK, MCP, and Cortex are completely unaffected.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from amfs_postgres._fts import or_tsquery
from amfs_core.content import classify_artifact
from amfs_core.exceptions import VersionConflictError
from amfs_core.models import (
    Agent,
    ArtifactRef,
    Event,
    EventType,
    GraphEdge,
    MemoryEntry,
    MemoryType,
    Provenance,
    SearchQuery,
    SemanticQuery,
)

from amfs_postgres.adapter import (
    _EXCLUDE_SHARED_PATHS,
    PostgresAdapter,
    pool_bounds,
    connection_options,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Async RLS wrappers — mirrors _TenantRLSConnection / _TenantRLSPoolWrapper
# ──────────────────────────────────────────────────────────────────────


class _AsyncTenantRLSConnection:
    """Async equivalent of _TenantRLSConnection.

    Replicates the exact same RLS behaviour:
    1. Always set GUC on every checkout (even when tid is empty).
    2. Empty string for no tenant (NULLIF handles it in RLS policies).
    3. Reads from the same contextvars used by the sync version.
    """

    def __init__(self, inner_ctx: Any) -> None:
        self._inner_ctx = inner_ctx

    async def __aenter__(self) -> Any:
        from amfs_postgres.tenant_context import get_request_tenant_account_id
        from amfs_postgres.tenant_gucs import aset_tenant_gucs

        conn = await self._inner_ctx.__aenter__()
        async with conn.cursor() as cur:
            await aset_tenant_gucs(cur, local=False)

        if not get_request_tenant_account_id():
            logger.warning(
                "async pool checkout with no tenant account_id — RLS-protected "
                "reads will return no rows"
            )
        return conn

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return await self._inner_ctx.__aexit__(exc_type, exc, tb)


class _AsyncTenantRLSPoolWrapper:
    """Wraps an AsyncConnectionPool so each checkout applies RLS session vars."""

    def __init__(self, inner: AsyncConnectionPool) -> None:
        self._inner = inner

    def connection(self) -> _AsyncTenantRLSConnection:
        return _AsyncTenantRLSConnection(self._inner.connection())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ──────────────────────────────────────────────────────────────────────
# AsyncPostgresAdapter
# ──────────────────────────────────────────────────────────────────────


class AsyncPostgresAdapter:
    """Async adapter for AMFS HTTP server hot-path endpoints.

    Provides async versions of read, write, list, search, ensure_agent,
    log_event, upsert_graph_edge, and increment_recall_count — enough
    to service the four hot-path REST endpoints without blocking the
    FastAPI event loop.

    Does NOT extend AdapterABC or PostgresAdapter.  Reuses static
    helpers (_row_to_entry, _row_to_agent) from the sync adapter.
    """

    _AGENT_COLUMNS = PostgresAdapter._AGENT_COLUMNS

    def __init__(
        self,
        dsn: str,
        namespace: str = "default",
        *,
        min_pool_size: int | None = None,
        max_pool_size: int | None = None,
        statement_timeout: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._namespace = namespace
        self._has_embedding_col = False
        self._has_search_tsv = False
        self._has_is_artifact_col = False
        connect_kwargs: dict[str, Any] = {"row_factory": dict_row, "autocommit": True}
        # The ceiling belongs here most of all. This adapter serves the hot-path
        # REST endpoints, so it holds the statements a caller is actually waiting
        # on — and until this line existed, AMFS_POSTGRES_STATEMENT_TIMEOUT
        # reached only the sync adapter, leaving the request path it was
        # introduced to protect without one.
        options = connection_options(statement_timeout)
        if options:
            connect_kwargs["options"] = options
        min_size, max_size = pool_bounds(min_pool_size, max_pool_size)
        self._pool = _AsyncTenantRLSPoolWrapper(
            AsyncConnectionPool(
                dsn,
                min_size=min_size,
                max_size=max_size,
                kwargs=connect_kwargs,
                open=False,
            )
        )

    async def open(self) -> None:
        """Open the connection pool and detect optional columns."""
        await self._pool._inner.open()
        await self._detect_optional_columns()

    async def close(self) -> None:
        """Shut down the connection pool."""
        await self._pool._inner.close()

    # ── column detection ────────────────────────────────────────────

    async def _detect_optional_columns(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'amfs_memory_entries'
                      AND column_name IN ('embedding', 'search_tsv', 'is_artifact')
                    """,
                )
                rows = await cur.fetchall()
                found = {row["column_name"] for row in rows}
                self._has_embedding_col = "embedding" in found
                self._has_search_tsv = "search_tsv" in found
                self._has_is_artifact_col = "is_artifact" in found

    # ── helpers (delegated to sync adapter statics) ─────────────────

    @staticmethod
    def _row_to_entry(row: dict[str, Any]) -> MemoryEntry:
        return PostgresAdapter._row_to_entry(row)

    @staticmethod
    def _row_to_agent(row: dict[str, Any]) -> Agent:
        return PostgresAdapter._row_to_agent(row)

    def _get_current_account_id(self) -> str | None:
        try:
            from amfs_postgres.tenant_context import get_request_tenant_account_id
            return get_request_tenant_account_id()
        except ImportError:
            return None

    # ──────────────────────────────────────────────────────────────
    # 1. read
    # ──────────────────────────────────────────────────────────────

    async def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
        branch: str = "main",
    ) -> MemoryEntry | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT * FROM amfs_memory_entries
                    WHERE namespace = %s AND branch = %s
                      AND entity_path = %s AND key = %s
                      AND superseded_at IS NULL
                    ORDER BY version DESC LIMIT 1
                    """,
                    (self._namespace, branch, entity_path, key),
                )
                row = await cur.fetchone()

                if row is None and branch != "main":
                    await cur.execute(
                        "SELECT parent_branch, branched_at FROM amfs_branches WHERE namespace = %s AND name = %s",
                        (self._namespace, branch),
                    )
                    branch_info = await cur.fetchone()
                    if branch_info:
                        await cur.execute(
                            """
                            SELECT * FROM amfs_memory_entries
                            WHERE namespace = %s AND branch = %s
                              AND entity_path = %s AND key = %s
                              AND superseded_at IS NULL
                              AND written_at <= %s
                            ORDER BY version DESC LIMIT 1
                            """,
                            (
                                self._namespace,
                                branch_info["parent_branch"],
                                entity_path,
                                key,
                                branch_info["branched_at"],
                            ),
                        )
                        row = await cur.fetchone()

        if row is None:
            return None
        entry = self._row_to_entry(row)
        if entry.confidence < min_confidence:
            return None
        return entry

    # ──────────────────────────────────────────────────────────────
    # 2. write
    # ──────────────────────────────────────────────────────────────

    async def write(self, entry: MemoryEntry) -> MemoryEntry:
        branch = entry.branch or "main"
        inserted_row = None
        async with self._pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
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
                    row = await cur.fetchone()
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
                        await cur.execute(
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

                    # Classify at the persistence chokepoint (the HTTP hot path
                    # builds entries inline and bypasses CoWEngine). The HTTP
                    # handler already embeds a descriptor for artifacts before
                    # calling write; this just guarantees the flag is set.
                    is_artifact = bool(entry.is_artifact) or classify_artifact(
                        entry.key, entry.value
                    )

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

                    if self._has_is_artifact_col:
                        columns.append("is_artifact")
                        params.append(is_artifact)

                    if self._has_embedding_col and entry.embedding:
                        columns.append("embedding")
                        params.append(
                            f"[{','.join(str(v) for v in entry.embedding)}]"
                        )

                    cols_sql = ", ".join(columns)
                    placeholders = ", ".join(["%s"] * len(params))
                    await cur.execute(
                        f"""
                        INSERT INTO amfs_memory_entries ({cols_sql})
                        VALUES ({placeholders})
                        RETURNING id, account_id
                        """,
                        params,
                    )
                    inserted_row = await cur.fetchone()

        if inserted_row is None:
            logger.error(
                "async write: INSERT returned no row — entry not persisted "
                "(entity_path=%s, key=%s, ns=%s)",
                entry.entity_path, entry.key, self._namespace,
            )
            raise RuntimeError(
                f"Write failed: INSERT did not persist for "
                f"{entry.entity_path}/{entry.key}"
            )

        if inserted_row.get("account_id") is None:
            logger.error(
                "async write: INSERT persisted with NULL account_id — "
                "entry will be invisible to RLS reads "
                "(entity_path=%s, key=%s, id=%s)",
                entry.entity_path, entry.key, inserted_row["id"],
            )

        return entry.model_copy(update={"version": new_version, "is_artifact": is_artifact})

    # ──────────────────────────────────────────────────────────────
    # 3. list
    # ──────────────────────────────────────────────────────────────

    async def list(
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
        else:
            conditions.append(_EXCLUDE_SHARED_PATHS)

        if not include_superseded:
            conditions.append("superseded_at IS NULL")

        where = " AND ".join(conditions)
        query = f"SELECT * FROM amfs_memory_entries WHERE {where} ORDER BY entity_path, key, version"

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

        return [self._row_to_entry(r) for r in rows]

    # ──────────────────────────────────────────────────────────────
    # 4. search
    # ──────────────────────────────────────────────────────────────

    async def search(self, query: SearchQuery, *, branch: str = "main") -> list[MemoryEntry]:
        use_fts = bool(query.query and self._has_search_tsv)
        col_ready = self._has_is_artifact_col
        tsq_sql, tsq_params = or_tsquery(query.query) if use_fts else ("", [])

        conditions = ["namespace = %s", "branch = %s", "superseded_at IS NULL"]
        params: list[Any] = [self._namespace, branch]

        if query.depth < 3:
            conditions.append("tier <= %s")
            params.append(query.depth)

        if col_ready and not query.include_artifacts:
            conditions.append("is_artifact IS NOT TRUE")

        if query.entity_path is not None:
            conditions.append("entity_path = %s")
            params.append(query.entity_path)
        else:
            conditions.append(_EXCLUDE_SHARED_PATHS)
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
            conditions.append(f"search_tsv @@ {tsq_sql}")
            params.extend(tsq_params)
        elif query.query and query.recall_config is None:
            conditions.append(
                "(key ILIKE %s OR entity_path ILIKE %s OR value::text ILIKE %s)"
            )
            like_pattern = f"%{query.query}%"
            params.extend([like_pattern, like_pattern, like_pattern])

        order_map = {
            "confidence": "confidence DESC",
            "recency": "written_at DESC",
            "version": "version DESC",
        }

        if use_fts and query.sort_by == "confidence":
            order = f"ts_rank(search_tsv, {tsq_sql}) DESC, confidence DESC"
            params.extend(tsq_params)
        else:
            order = order_map.get(query.sort_by, "confidence DESC")

        # Demote artifacts beneath equally-ranked facts (leading sort key when
        # the column is ready; Python fallback below otherwise).
        fetch_limit = query.limit
        if col_ready and query.sort_by != "priority":
            order = f"is_artifact ASC, {order}"
        elif not col_ready:
            fetch_limit = min(query.limit * 3, 1000)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM amfs_memory_entries
            WHERE {where}
            ORDER BY {order}
            LIMIT %s
        """
        params.append(fetch_limit)

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()

        entries = [self._row_to_entry(r) for r in rows]
        if not col_ready:
            if not query.include_artifacts:
                entries = [e for e in entries if not classify_artifact(e.key, e.value)]
            else:
                entries.sort(key=lambda e: classify_artifact(e.key, e.value))
            entries = entries[: query.limit]
        return entries

    # ──────────────────────────────────────────────────────────────
    # 4b. semantic_search (pgvector cosine, account-scoped via RLS pool)
    # ──────────────────────────────────────────────────────────────

    async def semantic_search(
        self,
        query: SemanticQuery,
        embedder: Any,
        *,
        branch: str = "main",
    ) -> list[tuple[MemoryEntry, float]]:
        """Rank entries by pgvector cosine similarity to the query embedding.

        Account isolation is enforced by the RLS pool wrapper (same as
        ``search``), so results are already scoped to the request's account
        before the LIMIT — never do a global top-K then filter, which would
        both leak scan scope and drop the caller's own matches.

        Returns (entry, similarity) pairs ordered by similarity desc. Empty
        list when the pgvector column is absent (caller falls back to lexical).
        """
        if not self._has_embedding_col:
            return []

        query_vec = embedder.embed(query.text)
        vec_str = f"[{','.join(str(v) for v in query_vec)}]"

        conditions = [
            "namespace = %s",
            "branch = %s",
            "superseded_at IS NULL",
            "embedding IS NOT NULL",
        ]
        params: list[Any] = [self._namespace, branch]
        if query.entity_path is not None:
            conditions.append("entity_path = %s")
            params.append(query.entity_path)
        else:
            conditions.append(_EXCLUDE_SHARED_PATHS)
        if query.min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(query.min_confidence)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT *, 1 - (embedding <=> %s::vector) AS similarity
            FROM amfs_memory_entries
            WHERE {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, [vec_str] + params + [vec_str, query.limit])
                rows = await cur.fetchall()

        results: list[tuple[MemoryEntry, float]] = []
        for row in rows:
            sim = float(row.get("similarity", 0) or 0)
            if sim >= query.min_similarity:
                results.append((self._row_to_entry(row), sim))
        return results

    # ──────────────────────────────────────────────────────────────
    # 5. ensure_agent
    # ──────────────────────────────────────────────────────────────

    async def ensure_agent(self, agent_id: str, namespace: str = "default") -> Agent:
        account_id = self._get_current_account_id()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO amfs_agents (namespace, agent_id, account_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT ON CONSTRAINT uq_agent DO UPDATE
                        SET last_active_at = NOW(),
                            entry_count = amfs_agents.entry_count + 1,
                            account_id = COALESCE(amfs_agents.account_id, EXCLUDED.account_id)
                    RETURNING {self._AGENT_COLUMNS}
                    """,
                    (namespace, agent_id, account_id),
                )
                row = await cur.fetchone()
        return self._row_to_agent(row)

    async def register_agent(self, agent_id: str, namespace: str = "default") -> Agent:
        """Ensure an agent row exists WITHOUT counting it as a memory write.

        Mirrors :meth:`PostgresAdapter.register_agent` — used when an agent
        announces itself (set_identity / profile update) so it appears on the
        dashboard before writing any memory. Does not inflate ``entry_count``.
        """
        account_id = self._get_current_account_id()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO amfs_agents (namespace, agent_id, account_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT ON CONSTRAINT uq_agent DO UPDATE
                        SET last_active_at = NOW(),
                            account_id = COALESCE(amfs_agents.account_id, EXCLUDED.account_id)
                    RETURNING {self._AGENT_COLUMNS}
                    """,
                    (namespace, agent_id, account_id),
                )
                row = await cur.fetchone()
        return self._row_to_agent(row)

    # ──────────────────────────────────────────────────────────────
    # 6. log_event
    # ──────────────────────────────────────────────────────────────

    async def log_event(self, event: Event) -> Event:
        account_id = self._get_current_account_id()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO amfs_events
                        (namespace, agent_id, branch, event_type,
                         summary, details, actor_agent_id, created_at, account_id)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
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
                        account_id,
                    ),
                )
                row = await cur.fetchone()
        return event.model_copy(update={"id": str(row["id"]), "created_at": row["created_at"]})

    # ──────────────────────────────────────────────────────────────
    # 7. upsert_graph_edge
    # ──────────────────────────────────────────────────────────────

    async def upsert_graph_edge(
        self,
        edge: GraphEdge,
        *,
        namespace: str = "default",
        branch: str = "main",
    ) -> GraphEdge:
        provenance_json = json.dumps(edge.provenance) if edge.provenance else None
        account_id = self._get_current_account_id()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO amfs_knowledge_graph
                        (namespace, branch, source_entity, source_type, relation,
                         target_entity, target_type, confidence, evidence_count,
                         first_seen, last_seen, provenance, account_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (namespace, branch, source_entity, relation, target_entity, account_id)
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
                        account_id,
                    ),
                )
                row = await cur.fetchone()

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

    # ──────────────────────────────────────────────────────────────
    # 8. increment_recall_count
    # ──────────────────────────────────────────────────────────────

    async def increment_recall_count(
        self,
        entity_path: str,
        key: str,
        *,
        branch: str = "main",
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """UPDATE amfs_memory_entries
                   SET recall_count = recall_count + 1
                   WHERE namespace = %s AND branch = %s
                     AND entity_path = %s AND key = %s
                     AND superseded_at IS NULL""",
                (self._namespace, branch, entity_path, key),
            )
