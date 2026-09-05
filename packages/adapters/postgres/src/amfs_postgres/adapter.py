"""PostgresAdapter — AMFS adapter backed by PostgreSQL with psycopg3."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg.rows import dict_row

from amfs_postgres._fts import or_tsquery
from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.content import ARTIFACT_PENALTY, classify_artifact, embedding_input
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
    Commit,
    CommitEntry,
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
    SessionMetadata,
    Tag,
    ToolCall,
    TraceEntry,
)

try:
    from psycopg_pool import ConnectionPool as _ConnectionPool
except ImportError:  # pragma: no cover
    _ConnectionPool = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Entity paths of the form @<name>/<topic> denote a shared namespace: one
# whose contents belong to a group rather than to the account reading them.
# Access to them is granted separately, outside this adapter.
#
# The rule below is that they are only ever returned when they were asked for
# by name. A query that names no entity_path is an ambient one — "what do I
# know", "what is relevant here" — and its results land in an agent's context
# without anyone having decided they should. Shared content reaching that
# position means a stranger who can write to a shared namespace can put text
# in front of an agent that is working on something else entirely, which is
# the shape of a prompt-injection attack and does not require the attacker to
# do anything but write an ordinary memory.
#
# Asking for the path explicitly is the act of deciding to trust it. So
# read(), and any search scoped to the path, still return everything; it is
# only the unscoped queries that filter it out.
# The wildcards are doubled because psycopg parses '%' as a placeholder marker
# whenever a query is executed with parameters, and '%/' is not a placeholder.
# Doubling is safe in both directions: psycopg unescapes it to '@%/%', and on
# the paths that pass the string through verbatim, consecutive LIKE wildcards
# collapse, so '@%%/%%' matches exactly what '@%/%' does.
_EXCLUDE_SHARED_PATHS = "entity_path NOT LIKE '@%%/%%'"

_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")

_DEFAULT_POOL_MIN = 2
_DEFAULT_POOL_MAX = 10


def _pool_size_from_env(name: str, default: int) -> int:
    """A pool bound read from the environment, or the default.

    A bad value is a misconfiguration, not a reason to refuse to start: a typo
    in a deployment variable would otherwise take a service down at boot, which
    is worse than running on the default and saying so.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        size = int(raw)
    except ValueError:
        logger.warning("%s=%r is not a number — using %d", name, raw, default)
        return default
    if size < 1:
        logger.warning("%s=%d is below 1 — using %d", name, size, default)
        return default
    return size


def pool_bounds(
    min_pool_size: int | None = None, max_pool_size: int | None = None
) -> tuple[int, int]:
    """How many connections one adapter may hold.

    Every process holding a pool multiplies against every instance of it, and
    the database has a fixed ``max_connections``. That budget is a property of
    the deployment, not of the code, so it belongs in the environment —
    ``AMFS_POSTGRES_POOL_MIN`` and ``AMFS_POSTGRES_POOL_MAX`` — where an
    operator scaling a service out can spend it without a release.

    An explicit argument still wins, because a caller that names a size is
    describing what that particular adapter is for: a background worker asking
    for two connections should get two, whatever the request path is allowed.
    """
    low = (
        min_pool_size
        if min_pool_size is not None
        else _pool_size_from_env("AMFS_POSTGRES_POOL_MIN", _DEFAULT_POOL_MIN)
    )
    high = (
        max_pool_size
        if max_pool_size is not None
        else _pool_size_from_env("AMFS_POSTGRES_POOL_MAX", _DEFAULT_POOL_MAX)
    )
    # psycopg_pool refuses a minimum above the maximum. Reaching that means the
    # two numbers came from different places — one argument, one variable — and
    # the ceiling is the one to respect, since it is what the database can bear.
    return min(low, high), high


#: Passed as ``statement_timeout`` to mean "no ceiling for this process",
#: distinct from passing nothing, which means "whatever the environment says".
NO_STATEMENT_TIMEOUT = "0"


def statement_timeout_options(override: str | None = None) -> str | None:
    """The libpq ``options`` string carrying the statement ceiling, if any.

    A ceiling on how long any one statement may run. Off unless asked for,
    because the right ceiling depends on the deployment: a request-serving
    process wants one near its own request timeout, while a consolidation or
    backfill job may legitimately run for much longer, and a default here would
    abort the second kind of work in installations that never asked for it.

    Worth setting on anything serving HTTP. A statement that outlives the
    request that asked for it holds its locks with nobody left to want the
    answer, which is how a single slow query became a production outage on
    2026-08-12; see _apply_schema.

    ``override`` exists because this is applied through libpq ``options``, which
    makes it a property of the *connection* and therefore of the whole pool.
    That is the right shape for a service but the wrong shape for a process that
    does both kinds of work: a bulk load, a per-partition index build and an
    embedding backfill are all single statements that legitimately outlast any
    request, and inheriting a request-shaped ceiling from a shared environment
    variable aborts them partway. A process that knows what it is passes its own
    value, or :data:`NO_STATEMENT_TIMEOUT` to opt out, rather than depending on
    how the environment happens to be configured around it.
    """
    timeout = (
        override
        if override is not None
        else os.environ.get("AMFS_POSTGRES_STATEMENT_TIMEOUT", "")
    ).strip()
    return f"-c statement_timeout={timeout}" if timeout else None


#: pgvector's accepted values. ``off`` is its default.
_HNSW_ITERATIVE_SCAN_MODES = frozenset({"off", "strict_order", "relaxed_order"})


def hnsw_iterative_scan_option() -> str | None:
    """The libpq ``options`` fragment enabling iterative HNSW scans, if asked.

    Without this, an HNSW index *post-filters*: it walks the global proximity
    graph for the nearest vectors and only then discards the ones failing the
    query's ``WHERE`` clause. A search restricted to a small slice of a large
    table -- one tenant, one room, one dataset -- can therefore come back short
    or empty while the rows it wanted sit unvisited. That failure is quiet, and
    a quiet wrong answer is worse than a slow one.

    ``hnsw.iterative_scan`` makes the scan keep going until it has enough rows
    that actually satisfy the filter. It needs pgvector >= 0.8.0.

    Off unless asked for, and deliberately not on by default: this is applied
    through libpq ``options``, so on a server whose pgvector predates 0.8.0 the
    setting is unrecognised and *every connection fails to open*. An installer
    who has not pinned their extension version should not have the adapter stop
    working because of a flag they never set. Deployments that control their
    database version -- which is the same set that can pin the image -- opt in.
    """
    mode = os.environ.get("AMFS_POSTGRES_HNSW_ITERATIVE_SCAN", "").strip().lower()
    if not mode:
        return None
    if mode not in _HNSW_ITERATIVE_SCAN_MODES:
        raise ValueError(
            "AMFS_POSTGRES_HNSW_ITERATIVE_SCAN must be one of "
            f"{sorted(_HNSW_ITERATIVE_SCAN_MODES)}, got {mode!r}"
        )
    return f"-c hnsw.iterative_scan={mode}"


def connection_options(statement_timeout: str | None = None) -> str | None:
    """Everything this process wants set on each connection, as one string.

    libpq takes repeated ``-c`` flags separated by spaces, so the fragments
    compose. None when nothing is configured, so callers can leave ``options``
    off the connection entirely rather than passing an empty string.
    """
    fragments = [
        fragment
        for fragment in (
            statement_timeout_options(statement_timeout),
            hnsw_iterative_scan_option(),
        )
        if fragment
    ]
    return " ".join(fragments) if fragments else None


def _tables_created_by(sql: str) -> frozenset[str]:
    """The tables a chunk of DDL creates.

    Reads the SQL rather than taking a list, so that adding a table anywhere the
    bootstrap runs is covered without anyone remembering to update a second
    place — which is the kind of upkeep that silently stops happening.
    """
    # Partitions (``CREATE TABLE x PARTITION OF parent``) are excluded: they
    # exist only once the parent is partitioned, which on a database where the
    # partitioning migration was deliberately skipped is never, and a table
    # that can never be present would make the schema never "current" and
    # re-run the DDL on every start — the convoy this check exists to stop.
    return frozenset(
        re.findall(
            r"CREATE TABLE IF NOT EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)\b(?!\s+PARTITION\s+OF\b)",
            sql,
            re.IGNORECASE,
        )
    )

# SQL mirror of amfs_core.aggregates.recall_tokens_saved: per reuse, credit
# the measured content size (~CHARS_PER_TOKEN chars/token) clamped to
# [floor, ceil]. Constants are imported so SQL and Python can't drift.
from amfs_core.aggregates import (  # noqa: E402
    CHARS_PER_TOKEN as _CHARS_PER_TOKEN,
    RECALL_TOKENS_CEIL as _RECALL_TOKENS_CEIL,
    RECALL_TOKENS_FLOOR as _RECALL_TOKENS_FLOOR,
)

_RECALLED_TOKENS_SQL = (
    "COALESCE(SUM(recall_count * LEAST(GREATEST("
    f"length(value::text) / {_CHARS_PER_TOKEN}, {_RECALL_TOKENS_FLOOR}), "
    f"{_RECALL_TOKENS_CEIL})), 0)"
)


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
    """Applies the four tenant settings on every checkout, inside a transaction.

    All four are set every time, whether or not a request tenant is present:
    with a tenant, to the real values; without one, to empty, which ``NULLIF``
    in the policies turns into NULL so nothing matches. Setting them
    unconditionally is the point — skipping the write when there is no tenant
    would leave the previous borrower of this pooled connection in place, and
    ``amfs.current_user_id`` is the setting that grants reach across accounts.

    The checkout opens an explicit transaction and sets them *inside* it, so
    Postgres discards them at commit and nothing is left on the backend for the
    next borrower. That is what makes it safe to put a transaction-mode pooler
    in front: session-scoped settings would land on whichever backend served the
    statement that set them, and the next query could be answered by a backend
    still carrying another tenant. See :mod:`amfs_postgres.tenant_gucs`.

    ``transactional=False`` is the maintenance path, for the two kinds of work
    that cannot run in a caller-imposed transaction — DDL, and the backfills
    that call an embedder between statements. Those get all four blanked instead
    of set, which still overwrites the previous borrower but writes no tenant.

    The whole checkout being one transaction is also why the methods below no
    longer call ``conn.commit()``: psycopg forbids an explicit commit inside a
    ``transaction()`` block, and the block commits on its own successful exit.
    """

    _tenantless_checkouts = 0

    def __init__(self, inner_ctx: Any, *, transactional: bool = True) -> None:
        self._inner_ctx = inner_ctx
        self._transactional = transactional
        self._stack: contextlib.ExitStack | None = None

    def __enter__(self) -> Any:
        from amfs_postgres.tenant_gucs import (
            blank_tenant_gucs,
            require_open_transaction,
            set_tenant_gucs,
        )

        # ExitStack rather than nested try/finally so that a failure while
        # setting the tenant unwinds the transaction and returns the connection
        # in the right order, without this class having to get that right twice.
        stack = contextlib.ExitStack()
        try:
            conn = stack.enter_context(self._inner_ctx)
            if self._transactional:
                stack.enter_context(conn.transaction())
                require_open_transaction(conn)
                with conn.cursor() as cur:
                    set_tenant_gucs(cur, local=True)
                self._warn_if_tenantless()
            else:
                with conn.cursor() as cur:
                    blank_tenant_gucs(cur)
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return conn

    @staticmethod
    def _warn_if_tenantless() -> None:
        """Report tenant-less checkouts, but on a curve rather than every time.

        A checkout with no tenant is worth knowing about, because in a request
        path it means reads are about to return nothing. It is also completely
        normal in a worker or a CLI, which have no request to carry one — and
        those are exactly the processes that check out connections in a tight
        loop, where one warning per checkout buries the log it was meant to
        inform. Powers of two keep the first few and then get out of the way.
        """
        from amfs_postgres.tenant_context import get_request_tenant_account_id

        if get_request_tenant_account_id():
            return
        _TenantRLSConnection._tenantless_checkouts += 1
        count = _TenantRLSConnection._tenantless_checkouts
        if count & (count - 1):  # not a power of two
            return
        logger.warning(
            "Postgres checkout with no tenant account_id (%d so far) — "
            "RLS-protected reads will return no rows. Expected in workers and "
            "CLIs; a bug in a request path.",
            count,
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if self._stack is None:
            return None
        return self._stack.__exit__(exc_type, exc, tb)


class _TenantRLSPoolWrapper:
    """Wraps a psycopg pool so each checkout scopes RLS vars to a transaction."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def connection(self, *, transactional: bool = True) -> _TenantRLSConnection:
        return _TenantRLSConnection(
            self._inner.connection(), transactional=transactional
        )

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
        Defaults to ``AMFS_POSTGRES_POOL_MIN``, or 2.
    max_pool_size:
        Maximum number of connections in the pool (requires ``psycopg_pool``).
        Defaults to ``AMFS_POSTGRES_POOL_MAX``, or 10.
    """

    def __init__(
        self,
        dsn: str,
        namespace: str = "default",
        *,
        auto_schema: bool = True,
        min_pool_size: int | None = None,
        max_pool_size: int | None = None,
        embedder: EmbedderABC | None = None,
        embedding_dim: int = 384,
        statement_timeout: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._namespace = namespace
        self._has_embedding_col = False
        self._has_search_tsv = False
        self._has_is_artifact_col = False
        # When an embedder is provided, embeddings are computed at write time and
        # persisted, so ANN retrieval (semantic_search / pgvector HNSW) works
        # without a separate backfill pass. embedding_dim must match the column
        # dimension (see ensure_embedding_column). None => write-time embedding off.
        self._embedder = embedder
        self._embedding_dim = embedding_dim
        connect_kwargs: dict[str, Any] = {
            "row_factory": dict_row,
            "autocommit": True,
        }
        options = connection_options(statement_timeout)
        if options:
            connect_kwargs["options"] = options
        pool_kwargs: dict[str, Any] = {"kwargs": connect_kwargs}
        if _ConnectionPool is not None:
            from amfs_postgres.tenant_gucs import reset_tenant_gucs

            min_size, max_size = pool_bounds(min_pool_size, max_pool_size)
            self._pool = _ConnectionPool(
                dsn,
                min_size=min_size,
                max_size=max_size,
                # Blank the tenant settings as the connection goes back. Every
                # checkout already overwrites all four, so this is not what
                # prevents a leak in normal use; it prevents one from outliving
                # a checkout that raised partway through, and it means an
                # idle pooled connection is never sitting on a real tenant.
                reset=reset_tenant_gucs,
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
            # Every start, not only when the schema changes: the partitions
            # that need to exist move forward one month at a time and the
            # schema marker does not. One catalog query on a warm database.
            try:
                self.ensure_trace_partitions()
            except Exception:  # noqa: BLE001
                logger.warning("trace partitioning: partition upkeep skipped", exc_info=True)
        self._detect_optional_columns()
        self._listen_thread: threading.Thread | None = None
        self._listen_stop = threading.Event()
        self._watchers: dict[str, list[Callable[[MemoryEntry], None]]] = {}

    def _maintenance_connection(self) -> Any:
        """A checkout with no transaction, for DDL and the backfills.

        Everything that touches tenant rows goes through the ordinary
        ``self._pool.connection()``, which wraps the whole block in one
        transaction and scopes the four settings to it. This is the documented
        exception, and it is deliberately awkward to reach: the callers that
        need it cannot live inside a single caller-imposed transaction at all --
        DDL because one transaction around ``schema.sql`` would hold ACCESS
        EXCLUSIVE on every object for the whole run, the backfills because they
        call an embedder between statements and fall back to more SQL after a
        caught error.

        The checkout blanks the four settings rather than setting them, so the
        connection never inherits the previous borrower's tenant. For DDL that
        is the whole story, and it fails closed: under FORCE ROW LEVEL SECURITY
        empty reads zero rows rather than all of them, so DDL that grew a tenant
        query by accident would return nothing rather than everything.

        The backfills do touch tenant rows, so they cannot run on the blanked
        settings this leaves behind -- they open a short transaction per
        statement with :meth:`_tenant_statement` and set a real tenant inside
        it. What they get from here is only the absence of an enclosing
        transaction.
        """
        return self._pool.connection(transactional=False)

    @contextlib.contextmanager
    def _tenant_statement(self, conn: Any) -> Iterator[Any]:
        """A short transaction on an existing checkout, carrying a real tenant.

        For the backfills, which run on :meth:`_maintenance_connection` but --
        unlike DDL -- read and update ``amfs_memory_entries``. That table has
        FORCE ROW LEVEL SECURITY, so on the blanked settings a maintenance
        checkout leaves behind, their ``SELECT`` matches no rows and the loops
        examine and update nothing at all, silently. They need a tenant.

        The tenant is still transaction-local, because a session-scoped one is
        the leak :mod:`amfs_postgres.tenant_gucs` exists to remove. So it is set
        per statement here rather than once per checkout, which costs a round
        trip per transaction and buys back both of the properties the backfills
        depend on: no embedder call happens inside a transaction, and each
        statement commits on its own, so one row's failure leaves the rest of
        the batch committed and a fallback issued after a caught error runs in a
        new transaction rather than a failed one.
        """
        from amfs_postgres.tenant_gucs import (
            require_open_transaction,
            set_tenant_gucs,
        )

        with conn.transaction():
            require_open_transaction(conn)
            with conn.cursor() as cur:
                set_tenant_gucs(cur, local=True)
                yield cur

    def _migrations_source(self) -> str | None:
        """The text of _apply_migrations, or None if it cannot be read.

        The migrations are Python rather than a file, so both the fingerprint and
        the table-presence check have to reach for them through inspect. None on a
        frozen or zipped install, where callers fall back to the behaviour from
        before any of this existed: apply the schema.
        """
        try:
            import inspect

            from amfs_postgres import trace_partitions

            # The partitioning and retention DDL lives in its own module, and a
            # change there is a schema change just as much as one in here.
            return inspect.getsource(type(self)._apply_migrations) + inspect.getsource(
                trace_partitions
            )
        except Exception:  # noqa: BLE001
            return None

    def _schema_fingerprint(self) -> str | None:
        """Identity of the schema this build would apply, or None if unknowable.

        Covers the migrations as well as schema.sql, because they add columns and
        indexes of their own — fingerprinting only the file would skip a container
        whose code carries a migration the database has not seen. Reading the
        source is how that stays automatic; a hand-bumped revision constant is a
        step someone eventually forgets, and forgetting it here means a column
        silently never gets added.
        """
        migrations = self._migrations_source()
        if migrations is None:
            return None
        return hashlib.sha256(
            (_SCHEMA_SQL + migrations).encode("utf-8")
        ).hexdigest()

    def _expected_tables(self) -> frozenset[str]:
        """Every table the bootstrap creates, from both places it creates them.

        schema.sql is not the whole schema: _apply_migrations adds the agent-group
        tables. Checking only the file would leave those three outside the
        presence check, so dropping one of them while the fingerprint still
        matched would be waved through — the exact hole the presence check was
        added to close, just further from the eye.
        """
        return _tables_created_by(_SCHEMA_SQL) | _tables_created_by(
            self._migrations_source() or ""
        )

    def _apply_schema(self, *, retries: int = 3) -> None:
        """Bring the database up to this build's schema, at most once per build.

        Every process that constructs an adapter runs this, which on a warm
        database used to mean re-running the whole DDL on every container start.
        That is not free even when every statement is a no-op: ALTER TABLE and
        CREATE INDEX ask for ACCESS EXCLUSIVE, and a *waiting* exclusive lock
        blocks every reader that arrives behind it. One long-running SELECT is
        then enough to stall the table, and because a stalled table stalls
        startup, the platform responds by starting more containers, each adding
        another exclusive lock to the queue. That took production down on
        2026-08-12: 29 queued copies of this DDL behind a single orphaned query.

        So the fast path checks a fingerprint and touches nothing, and the slow
        path refuses to queue: lock_timeout means a container that cannot get the
        lock promptly fails this attempt instead of joining the convoy it would
        otherwise extend.

        Runs on ``_maintenance_connection`` rather than the ordinary checkout,
        which is the one place that matters here: the ordinary one wraps the
        block in a transaction, and one transaction around all of this DDL would
        hold every ACCESS EXCLUSIVE lock it takes until the last statement
        finished, turning the convoy above into a guaranteed one. It would also
        undo the retry, which depends on partial progress being kept.
        """
        import time as _time

        fingerprint = self._schema_fingerprint()
        if fingerprint and self._schema_is_current(fingerprint):
            return

        for attempt in range(1, retries + 1):
            try:
                with self._maintenance_connection() as conn:
                    with conn.cursor() as cur:
                        # Bounds the wait for a lock, not the work: a genuine
                        # first-run CREATE INDEX on a large table is allowed to
                        # take as long as it takes.
                        #
                        # Reset in the finally because the pool runs autocommit,
                        # which makes plain SET last for the life of the session.
                        # Left set, this connection would go back to the pool and
                        # hand a 5s lock timeout to ordinary writes, turning
                        # everyday contention into failed requests. SET LOCAL is
                        # not the answer either: outside a transaction it is a
                        # no-op, and schema.sql is not something to wrap in one.
                        cur.execute("SET lock_timeout = '5s'")
                        try:
                            cur.execute(_SCHEMA_SQL)
                            complete = self._apply_migrations(cur)
                            # A migration that failed for a transient reason
                            # reports False so the marker is withheld and the
                            # next start tries again; everything else is done.
                            if fingerprint and complete is not False:
                                self._record_schema_fingerprint(cur, fingerprint)
                        finally:
                            cur.execute("SET lock_timeout = DEFAULT")
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

    def _schema_is_current(self, fingerprint: str) -> bool:
        """Whether this database already has the schema this build would apply.

        The marker alone is not enough to answer this, and trusting it on its own
        was a bug: it lives in a table of its own, so it survives anything that
        removes the tables it vouches for — a dropped table, a restore from an
        older dump, a test fixture resetting state — and the next process would
        sail past the DDL and then fail on the first query against a table that
        is no longer there. So every table the bootstrap creates is checked too,
        against the catalog, which costs one indexed lookup and takes no locks.

        Deliberately silent about every failure. Anything unreadable means "not
        current", which costs one pass of idempotent DDL — the behaviour from
        before this check existed. Raising instead would turn a missing metadata
        row into a process that will not boot.
        """
        try:
            expected = self._expected_tables()
            # Build metadata and pg_catalog, no tenant rows, and it runs before
            # the schema it is checking for necessarily exists — so the
            # maintenance checkout, same as the DDL it guards.
            with self._maintenance_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM amfs_schema_state WHERE fingerprint = %s",
                        (fingerprint,),
                    )
                    if cur.fetchone() is None:
                        return False
                    cur.execute(
                        """
                        SELECT count(*) AS present
                        FROM pg_catalog.pg_class c
                        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relkind IN ('r', 'p')
                          AND n.nspname = ANY (current_schemas(false))
                          AND c.relname = ANY (%s)
                        """,
                        (list(expected),),
                    )
                    row = cur.fetchone()
                    return bool(expected) and row["present"] == len(expected)
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _record_schema_fingerprint(cur: psycopg.Cursor, fingerprint: str) -> None:
        """Note that this database has been given this build's schema.

        A row per schema, not one row for "the current schema". Two builds are
        alive at once during any rolling deploy, and a single row makes them
        fight over it: the older instance finds the newer hash, decides it is out
        of date, re-applies its DDL and overwrites the marker, at which point the
        newer instance does the same in reverse. Every scale-up on either
        revision would then take ACCESS EXCLUSIVE locks — the convoy this whole
        mechanism exists to prevent, arriving exactly at deploy time, when the
        most instances are starting at once.

        Recording each schema separately means both builds see themselves as
        applied and neither re-applies. The DDL is additive, so a database that
        has been given two schemas satisfies both. Growth is one row per distinct
        schema version the deployment has ever run, which is not a table anyone
        needs to prune; it is also not a history worth reading, so a row is
        replaced rather than duplicated when the same schema is applied again.

        Deliberately carries no account_id and no row-level security policy,
        unlike every table that holds tenant data. It records a hash of this
        build's DDL, which is the same for every tenant on the deployment and is
        not anybody's data. Nothing tenant-scoped may be added to it later
        without giving it a policy first.
        """
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS amfs_schema_state (
                fingerprint TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        # Scoped to this fingerprint: an unqualified DELETE is what let one build
        # erase another's marker. Not an upsert, to avoid depending on a unique
        # index that databases created by the first version of this code do not
        # have.
        cur.execute(
            "DELETE FROM amfs_schema_state WHERE fingerprint = %s",
            (fingerprint,),
        )
        cur.execute(
            "INSERT INTO amfs_schema_state (fingerprint) VALUES (%s)",
            (fingerprint,),
        )

    @staticmethod
    def _apply_migrations(cur: psycopg.Cursor) -> bool:
        """Additive migrations for columns added after initial schema.

        Returns False when a migration could not be applied for a reason that
        may clear on its own, so ``_apply_schema`` withholds the schema marker
        and the next start retries. True (or None, from older code paths)
        means the database is at this build's schema.
        """
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS shared BOOLEAN NOT NULL DEFAULT TRUE
        """)
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS branch TEXT NOT NULL DEFAULT 'main'
        """)
        # Artifact flag: distinguishes stored working files (source code, markup,
        # config) from knowledge facts so retrieval can demote them. Partial index
        # stays tiny because only the artifact rows are indexed.
        cur.execute("""
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS is_artifact BOOLEAN NOT NULL DEFAULT FALSE
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_entries_is_artifact
            ON amfs_memory_entries (is_artifact) WHERE is_artifact
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
            ALTER TABLE amfs_decision_traces
            ADD COLUMN IF NOT EXISTS session_metadata JSONB DEFAULT '{}'
        """)
        # The request that triggered the decision, and optionally the agent's
        # response. Supervised training needs the prompt side of the pair, and
        # the trace previously recorded only what was decided.
        cur.execute("""
            ALTER TABLE amfs_decision_traces
            ADD COLUMN IF NOT EXISTS task_input TEXT
        """)
        cur.execute("""
            ALTER TABLE amfs_decision_traces
            ADD COLUMN IF NOT EXISTS response_text TEXT
        """)
        # The action side of that pair. Training predicts the action from the
        # request, so a trace carrying only the request is half an example.
        cur.execute("""
            ALTER TABLE amfs_decision_traces
            ADD COLUMN IF NOT EXISTS tool_calls JSONB DEFAULT '[]'
        """)
        # Entry-to-trace resolution for knowledge lineage. The shared session id
        # is the only link from a written entry back to the trace that committed
        # it, since causal_entries records reads rather than writes.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_traces_session
            ON amfs_decision_traces (namespace, agent_id, session_id)
            WHERE session_id IS NOT NULL
        """)
        cur.execute("""
            ALTER TABLE amfs_digests
            ADD COLUMN IF NOT EXISTS branch TEXT NOT NULL DEFAULT 'main'
        """)
        cur.execute("""
            ALTER TABLE amfs_digests
            ADD COLUMN IF NOT EXISTS account_id UUID
        """)
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE amfs_digests DROP CONSTRAINT IF EXISTS uq_digest;
                ALTER TABLE amfs_digests
                    ADD CONSTRAINT uq_digest
                    UNIQUE (namespace, branch, digest_type, scope, account_id);
            EXCEPTION WHEN others THEN
                RAISE NOTICE 'uq_digest constraint migration skipped: %', SQLERRM;
            END $$
        """)
        cur.execute("""
            ALTER TABLE amfs_api_keys
            ADD COLUMN IF NOT EXISTS default_branch TEXT
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_detected_patterns') THEN
                    ALTER TABLE amfs_detected_patterns
                        ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'collaboration';
                    ALTER TABLE amfs_detected_patterns
                        DROP CONSTRAINT IF EXISTS chk_pattern_type;
                    ALTER TABLE amfs_detected_patterns
                        ADD CONSTRAINT chk_pattern_type CHECK (
                            pattern_type IN (
                                'knowledge_conflict', 'stale_knowledge', 'orphaned_branch',
                                'redundant_writes', 'single_point_of_knowledge', 'passive_consumer',
                                'unreviewed_changes', 'recurring_failure',
                                'hot_entity', 'stale_cluster', 'confidence_drift'
                            )
                        );
                END IF;
            END $$
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
                        'branch', NEW.branch,
                        'account_id', NEW.account_id
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
            ALTER TABLE amfs_memory_entries
            ADD COLUMN IF NOT EXISTS account_id UUID
        """)
        cur.execute("""
            ALTER TABLE amfs_outcomes
            ADD COLUMN IF NOT EXISTS account_id UUID
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
                -- SUCCESS reinforces confidence (>1.0), failures erode it (<1.0).
                -- Result is clamped to [0,1] below.
                CASE NEW.outcome_type
                    WHEN 'critical_failure' THEN multiplier := 0.85;
                    WHEN 'failure' THEN multiplier := 0.90;
                    WHEN 'minor_failure' THEN multiplier := 0.92;
                    WHEN 'success' THEN multiplier := 1.03;
                    WHEN 'p1_incident' THEN multiplier := 0.85;
                    WHEN 'p2_incident' THEN multiplier := 0.90;
                    WHEN 'regression' THEN multiplier := 0.92;
                    WHEN 'clean_deploy' THEN multiplier := 1.03;
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
                            ttl_at, memory_type, shared, artifact_refs,
                            is_artifact
                        ) VALUES (
                            cur.namespace, cur.entity_path, cur.key, cur.version + 1, cur.value,
                            cur.agent_id, cur.session_id, cur.written_at, cur.pattern_refs,
                            LEAST(1.0, GREATEST(0.0, cur.confidence * multiplier * NEW.causal_confidence)),
                            cur.outcome_count + 1, cur.recall_count,
                            cur.ttl_at, cur.memory_type,
                            cur.shared, cur.artifact_refs,
                            cur.is_artifact
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
        # Agent profile persistence
        cur.execute("""
            ALTER TABLE amfs_agents
            ADD COLUMN IF NOT EXISTS profile JSONB
        """)
        cur.execute("""
            ALTER TABLE amfs_agents
            ADD COLUMN IF NOT EXISTS capabilities JSONB DEFAULT '[]'::jsonb
        """)
        cur.execute("""
            ALTER TABLE amfs_agents
            ADD COLUMN IF NOT EXISTS contracts JSONB DEFAULT '[]'::jsonb
        """)
        # Tenant isolation — account_id scoping on optional tables.
        # These tables are created by separate migration files and may not
        # exist in minimal test environments, so guard with IF EXISTS.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_digests_account
                ON amfs_digests (account_id) WHERE account_id IS NOT NULL
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_knowledge_graph') THEN
                    ALTER TABLE amfs_knowledge_graph
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    ALTER TABLE amfs_knowledge_graph
                        DROP CONSTRAINT IF EXISTS uq_graph_edge;
                    ALTER TABLE amfs_knowledge_graph
                        ADD CONSTRAINT uq_graph_edge
                        UNIQUE (namespace, branch, source_entity, relation, target_entity, account_id);
                    CREATE INDEX IF NOT EXISTS idx_kg_account
                        ON amfs_knowledge_graph (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_events') THEN
                    ALTER TABLE amfs_events
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    CREATE INDEX IF NOT EXISTS idx_events_account
                        ON amfs_events (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_agents') THEN
                    ALTER TABLE amfs_agents
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    CREATE INDEX IF NOT EXISTS idx_agents_account
                        ON amfs_agents (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_branches') THEN
                    ALTER TABLE amfs_branches
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    CREATE INDEX IF NOT EXISTS idx_branches_account
                        ON amfs_branches (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_tags') THEN
                    ALTER TABLE amfs_tags
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    CREATE INDEX IF NOT EXISTS idx_tags_account
                        ON amfs_tags (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_pull_requests') THEN
                    ALTER TABLE amfs_pull_requests
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    CREATE INDEX IF NOT EXISTS idx_prs_account
                        ON amfs_pull_requests (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_branch_access') THEN
                    ALTER TABLE amfs_branch_access
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    CREATE INDEX IF NOT EXISTS idx_branch_access_account
                        ON amfs_branch_access (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'amfs_pr_reviews') THEN
                    ALTER TABLE amfs_pr_reviews
                        ADD COLUMN IF NOT EXISTS account_id UUID;
                    CREATE INDEX IF NOT EXISTS idx_pr_reviews_account
                        ON amfs_pr_reviews (account_id) WHERE account_id IS NOT NULL;
                END IF;
            END $$
        """)

        # Multi-tenant unique constraints: include account_id so different
        # tenants can have entries/agents/branches/tags with the same names.
        #
        # Each DROP+ADD pair runs inside an explicit transaction (BEGIN/COMMIT)
        # so that if ADD fails, the DROP is rolled back and the old constraint
        # stays in place.  The pool uses autocommit=True, so without an
        # explicit transaction each statement would commit immediately.
        _CONSTRAINT_MIGRATIONS: list[tuple[str, str, str, list[str]]] = [
            (
                "amfs_memory_entries",
                "uq_entry_version",
                "UNIQUE (namespace, entity_path, key, version, account_id)",
                ["namespace", "entity_path", "key", "version", "account_id"],
            ),
            (
                "amfs_agents",
                "uq_agent",
                "UNIQUE (namespace, agent_id, account_id)",
                ["namespace", "agent_id", "account_id"],
            ),
            (
                "amfs_branches",
                "uq_branch_name",
                "UNIQUE (namespace, name, account_id)",
                ["namespace", "name", "account_id"],
            ),
            (
                "amfs_tags",
                "uq_tag_name",
                "UNIQUE (namespace, name, account_id)",
                ["namespace", "name", "account_id"],
            ),
            (
                "amfs_branch_access",
                "uq_branch_access",
                "UNIQUE (namespace, branch_name, grantee_type, grantee_id, account_id)",
                ["namespace", "branch_name", "grantee_type", "grantee_id", "account_id"],
            ),
        ]
        for table, cname, cdef, cols in _CONSTRAINT_MIGRATIONS:
            try:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = %s",
                    (table,),
                )
                if cur.fetchone() is None:
                    continue

                col_list = ", ".join(cols)
                null_clause = " AND ".join(f"{c} IS NOT NULL" for c in cols)

                cur.execute(f"""
                    DELETE FROM "{table}" a USING (
                        SELECT ctid, ROW_NUMBER() OVER (
                            PARTITION BY {col_list} ORDER BY ctid
                        ) AS rn
                        FROM "{table}"
                        WHERE {null_clause}
                    ) b
                    WHERE a.ctid = b.ctid AND b.rn > 1
                """)
                if cur.rowcount and cur.rowcount > 0:
                    logger.warning(
                        "Removed %d duplicate rows from %s before adding %s",
                        cur.rowcount, table, cname,
                    )

                cur.execute("BEGIN")
                cur.execute(
                    f'ALTER TABLE "{table}" DROP CONSTRAINT IF EXISTS {cname}'
                )
                cur.execute(
                    f'ALTER TABLE "{table}" ADD CONSTRAINT {cname} {cdef}'
                )
                cur.execute("COMMIT")
            except Exception as exc:
                try:
                    cur.execute("ROLLBACK")
                except Exception:
                    pass
                logger.warning(
                    "Constraint migration %s.%s failed (non-fatal): %s",
                    table, cname, exc,
                )

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash_active
            ON amfs_api_keys (key_hash) WHERE active = true
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_entries_current_branch
            ON amfs_memory_entries (namespace, branch, entity_path, key)
            WHERE superseded_at IS NULL
        """)
        # The per-agent activity listing: current entries by one agent, newest
        # first, keyset-paged on written_at. Without this it is a scan of every
        # current entry in the namespace filtered by agent afterwards.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_entries_agent_written
            ON amfs_memory_entries (namespace, agent_id, written_at DESC)
            WHERE superseded_at IS NULL
        """)

        cur.execute("""
            ALTER TABLE amfs_api_keys
            ADD COLUMN IF NOT EXISTS created_by UUID
        """)
        # Agent groups (user-defined and auto-generated)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS amfs_agent_groups (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace TEXT NOT NULL DEFAULT 'default',
                account_id UUID,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                color TEXT DEFAULT NULL,
                icon TEXT DEFAULT NULL,
                position FLOAT DEFAULT 0,
                auto_generated BOOLEAN DEFAULT FALSE,
                source_cluster_id TEXT DEFAULT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT uq_agent_group UNIQUE (namespace, account_id, name)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_groups_account
                ON amfs_agent_groups(account_id) WHERE account_id IS NOT NULL
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS amfs_agent_group_members (
                group_id UUID NOT NULL REFERENCES amfs_agent_groups(id) ON DELETE CASCADE,
                agent_id TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default',
                added_at TIMESTAMPTZ DEFAULT NOW(),
                added_by TEXT DEFAULT 'user',
                PRIMARY KEY (group_id, agent_id)
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_group_member_unique
                ON amfs_agent_group_members(namespace, agent_id)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS amfs_agent_group_suggestions_dismissed (
                account_id UUID NOT NULL,
                cluster_id TEXT NOT NULL,
                dismissed_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (account_id, cluster_id)
            )
        """)

        # Decision traces at volume: monthly range partitioning by created_at,
        # then the GIN index the entity_path filter needs. The partitioning
        # step is a one-time table rebuild inside its own transaction (see
        # trace_partitions.migrate_to_partitioned for the approach and its
        # rollback). It comes after every ALTER above so the parent is built
        # from a table that already has every column; the index comes after
        # the partitioning so it is created once, on the parent, rather than
        # on a table about to be rebuilt.
        from amfs_postgres import trace_partitions

        complete = True
        try:
            trace_partitions.migrate_to_partitioned(cur)
        except Exception:  # noqa: BLE001
            # Already rolled back and logged. Not re-raised: a failed rebuild
            # must not stop the process booting, since the flat table still
            # works. Reported as incomplete so the next start tries again.
            complete = False
        trace_partitions.create_causal_entries_index(cur)
        try:
            trace_partitions.ensure_partitions(cur)
        except Exception:  # noqa: BLE001
            logger.warning("trace partitioning: could not create partitions ahead", exc_info=True)
        return complete

    # ------------------------------------------------------------------
    # decision-trace partition upkeep and retention
    # ------------------------------------------------------------------

    def ensure_trace_partitions(self, months_ahead: int = 2) -> list[str]:
        """Create trace partitions for this month and *months_ahead* more.

        Runs at adapter start and is safe to call any time: it reads the
        catalog and creates only what is missing, so on a warm database it is
        one query and no DDL. Returns the partition names it created. A no-op
        when the table is not partitioned.
        """
        from amfs_postgres import trace_partitions

        with self._maintenance_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET lock_timeout = '5s'")
                try:
                    return trace_partitions.ensure_partitions(cur, months_ahead=months_ahead)
                finally:
                    cur.execute("SET lock_timeout = DEFAULT")

    def apply_trace_retention(
        self,
        *,
        hot_days: int = 30,
        strip_payloads: bool = True,
        drop_after_days: int | None = None,
    ) -> dict[str, Any]:
        """Age decision traces out of the hot tier; see ``trace_partitions.apply_retention``.

        Payload stripping is scoped to this adapter's namespace and keeps every
        trace, its causal entries, contexts and metadata; only ``task_input``,
        ``response_text``, ``tool_calls``, ``query_events`` and ``error_events``
        are cleared for traces older than *hot_days*. *drop_after_days*, when
        set, drops whole monthly partitions older than that — across every
        namespace, since a partition is a month, not a tenant — so it is off
        by default. Nothing schedules this; call it from a job or the
        ``python -m amfs_postgres.retention`` entry point.

        Runs on the maintenance checkout, which carries no tenant, so on a
        deployment that scopes traces with row-level security it runs as the
        table owner or not at all.
        """
        from amfs_postgres import trace_partitions

        with self._maintenance_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET lock_timeout = '5s'")
                try:
                    return trace_partitions.apply_retention(
                        cur,
                        namespace=self._namespace,
                        hot_days=hot_days,
                        strip_payloads=strip_payloads,
                        drop_after_days=drop_after_days,
                    )
                finally:
                    cur.execute("SET lock_timeout = DEFAULT")

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
                      AND column_name IN ('embedding', 'search_tsv', 'is_artifact')
                    """,
                )
                found = {row["column_name"] for row in cur.fetchall()}
                self._has_embedding_col = "embedding" in found
                self._has_search_tsv = "search_tsv" in found
                self._has_is_artifact_col = "is_artifact" in found

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
        if entry.is_expired():
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

                    # Classify at the persistence chokepoint so every write path
                    # (HTTP async hot path, TransactionBuffer, Cortex, lifecycle,
                    # snapshot — all of which bypass CoWEngine) gets a consistent
                    # flag, and so a new version after a redact/revert re-classifies.
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

                    # Write-time embedding: compute and persist an embedding when
                    # the caller did not supply one and an embedder is configured.
                    # Artifacts embed a clean descriptor (filename + symbols), not
                    # the noisy, 512-token-truncated raw blob. Guarded so an
                    # embedder failure never blocks a write.
                    embedding = entry.embedding
                    if (embedding is None and self._embedder is not None
                            and self._has_embedding_col):
                        try:
                            _, embed_text = embedding_input(entry.key, entry.value)
                            embedding = self._embedder.embed(embed_text)
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "write-time embedding failed for %s/%s; storing without vector",
                                entry.entity_path, entry.key, exc_info=True,
                            )
                            embedding = None

                    if self._has_embedding_col and embedding:
                        columns.append("embedding")
                        params.append(
                            f"[{','.join(str(v) for v in embedding)}]"
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

        return entry.model_copy(update={
            "version": new_version,
            "embedding": embedding,
            "is_artifact": is_artifact,
        })

    def ensure_embedding_column(self, dim: int | None = None) -> None:
        """Create the pgvector embedding column + HNSW index at a given dimension.

        Idempotent when the column already exists at the right dimension. Use
        this to provision a store for a larger embedder (e.g. bge-large at 1024)
        instead of the default vector(384) from migration 002.

        Requires the pgvector extension (``CREATE EXTENSION IF NOT EXISTS vector``).
        This alters schema and, if the column exists at a different dimension,
        does nothing (Postgres cannot change a vector column's dimension in
        place — drop it first, see migration 005). Not called automatically.

        TODO(integration-test): covered by tests/integration/test_postgres_embeddings.py
        which requires AMFS_TEST_PG_DSN + pgvector; unvalidated in unit CI.
        """
        target = dim or self._embedding_dim
        # DDL: CREATE INDEX ... hnsw on a populated table is long and takes
        # ACCESS EXCLUSIVE, so it is not something to hold a transaction across.
        with self._maintenance_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"ALTER TABLE amfs_memory_entries "
                    f"ADD COLUMN IF NOT EXISTS embedding vector({int(target)})"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entries_embedding "
                    "ON amfs_memory_entries USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                )
        self._detect_optional_columns()

    def backfill_embeddings(
        self, *, batch_size: int = 200, max_rows: int | None = None
    ) -> int:
        """Compute and store embeddings for existing rows that lack one.

        Returns the number of rows updated. Requires a configured embedder and
        the embedding column. Intended as a migration step after enabling
        write-time embeddings on an existing store, and as the repair path for
        rows an import wrote while no embedder was configured.

        Walks a keyset cursor over ``id`` to completion, in the same shape as
        :meth:`backfill_is_artifact`. Both properties matter, and neither was
        true before:

        * **It finishes.** The previous version read one batch and returned, so
          a caller asking to backfill a store with more than ``batch_size``
          missing embeddings silently got ``batch_size`` of them and no
          indication there was more.
        * **A row that cannot be embedded cannot starve the ones behind it.**
          The previous version ordered by ``written_at`` and skipped failures
          with ``continue``, leaving them ``NULL``. Every later call therefore
          re-read the same permanently-failing rows at the head of the batch --
          a value the embedder chokes on is enough to stall the backfill
          forever, and the symptom is a store that never finishes backfilling
          rather than an error. Advancing by ``id`` regardless of outcome is
          what fixes it, since progress no longer depends on success.

        ``max_rows`` bounds the work for a caller that wants to make progress
        without running to completion, such as a worker doing this between other
        jobs. It counts rows *examined*, not rows successfully embedded, and the
        difference is the whole value of it: counting successes would mean a run
        of rows the embedder cannot handle does not count against the budget, so
        a caller asking for 1000 rows of work would keep fetching batches until
        it found 1000 that worked -- walking every remaining row in the store if
        none of them do. A bound that a bad row can lift is not a bound, and the
        caller that needs one is precisely the one on a time slice.

        The return value is still the number of rows *updated*, which is the
        useful figure for a migration and the one the integration test asserts
        on. When the two differ, the warning below says by how much.

        One cost this does not pay off: each call starts its cursor from the
        beginning, so rows that permanently fail are re-examined by every
        subsequent call. That is bounded by the number of such rows rather than
        growing, and it does not block progress -- a call still reaches new rows
        after paying it -- so it is left as a known overhead instead of being
        fixed with a resume cursor that no caller yet wants. If failures ever
        grow to the point of eating a time slice, that is the fix.

        TODO(integration-test): covered by tests/integration/test_postgres_embeddings.py
        (needs AMFS_TEST_PG_DSN + pgvector); unvalidated in unit CI.
        """
        if self._embedder is None or not self._has_embedding_col:
            return 0
        updated = 0
        failed = 0
        examined = 0
        last_id = "00000000-0000-0000-0000-000000000000"
        while True:
            if max_rows is not None and examined >= max_rows:
                break
            limit = batch_size
            if max_rows is not None:
                limit = min(limit, max_rows - examined)
            # Maintenance checkout: this block calls the embedder once per row,
            # so an ordinary one would hold a transaction open across a whole
            # batch of network round trips. The tenant comes from the short
            # per-statement transactions below rather than from the checkout,
            # which blanks it -- these rows are tenant rows, and RLS matches
            # none of them on a blank tenant. Per-statement commit is also what
            # lets one row's embedding failure leave the rest of the batch
            # committed, which is the documented behaviour above.
            with self._maintenance_connection() as conn:
                with self._tenant_statement(conn) as cur:
                    cur.execute(
                        "SELECT id, value FROM amfs_memory_entries "
                        "WHERE namespace = %s AND embedding IS NULL AND id > %s "
                        "ORDER BY id LIMIT %s",
                        (self._namespace, last_id, limit),
                    )
                    rows = cur.fetchall()
                if not rows:
                    break
                for r in rows:
                    last_id = r["id"]
                    examined += 1
                    # Decoding is separate from embedding on purpose. Both
                    # used to share one try, so a value that is a bare
                    # string -- not JSON, and perfectly embeddable -- raised
                    # JSONDecodeError and was counted as a row that could
                    # not be embedded. It then kept its NULL embedding and
                    # was re-read and re-failed by every later backfill.
                    # backfill_is_artifact already falls back this way.
                    raw = r["value"]
                    if isinstance(raw, str):
                        try:
                            value = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            value = raw
                    else:
                        value = raw
                    # Outside any transaction on purpose: this is the network
                    # round trip the maintenance checkout exists to keep out
                    # of one.
                    try:
                        vec = self._embedder.embed_value(value)
                    except Exception:  # noqa: BLE001
                        failed += 1
                        logger.warning(
                            "backfill embedding failed for row %s", r["id"],
                            exc_info=True,
                        )
                        continue
                    with self._tenant_statement(conn) as cur:
                        cur.execute(
                            "UPDATE amfs_memory_entries SET embedding = %s "
                            "WHERE id = %s",
                            (f"[{','.join(str(v) for v in vec)}]", r["id"]),
                        )
                    updated += 1
        if failed:
            logger.warning(
                "backfill_embeddings: examined %d rows, updated %d, %d could not "
                "be embedded and remain without one",
                examined, updated, failed,
            )
        return updated

    def backfill_is_artifact(self, *, batch_size: int = 200, reembed: bool = True) -> int:
        """Classify existing rows and set ``is_artifact``; re-embed newly-flagged
        artifacts from a clean descriptor.

        This fixes two things on historical data: rows written before the flag
        existed are classified, and code blobs whose vectors were the noisy,
        512-token-truncated raw content are re-embedded from a filename+symbols
        descriptor so they stop spuriously matching generic prose queries.

        Idempotent and resumable: only rows whose flag actually flips are
        touched (and only those are re-embedded), so re-running is cheap.
        Returns the number of rows updated.

        TODO(integration-test): exercised against a real DB in
        tests/integration/test_postgres_embeddings.py (needs AMFS_TEST_PG_DSN).
        """
        if not self._has_is_artifact_col:
            return 0
        can_embed = reembed and self._embedder is not None and self._has_embedding_col
        updated = 0
        last_id = "00000000-0000-0000-0000-000000000000"
        while True:
            # Maintenance checkout, for two reasons. It embeds per row, so an
            # ordinary transactional one would stay open across a batch of
            # network calls; and the re-embed fallback below issues an UPDATE
            # from inside an ``except``, which after a failed statement in the
            # same transaction is InFailedSqlTransaction rather than a fallback.
            # The tenant comes from the short per-statement transactions rather
            # than the checkout, which blanks it: these are tenant rows, and RLS
            # matches none of them on a blank tenant.
            with self._maintenance_connection() as conn:
                with self._tenant_statement(conn) as cur:
                    cur.execute(
                        "SELECT id, key, value, is_artifact FROM amfs_memory_entries "
                        "WHERE namespace = %s AND id > %s "
                        "ORDER BY id LIMIT %s",
                        (self._namespace, last_id, batch_size),
                    )
                    rows = cur.fetchall()
                if not rows:
                    break
                for r in rows:
                    last_id = r["id"]
                    stored = bool(r["is_artifact"])
                    try:
                        value = (
                            json.loads(r["value"])
                            if isinstance(r["value"], str)
                            else r["value"]
                        )
                    except (json.JSONDecodeError, ValueError):
                        value = r["value"]
                    is_art = classify_artifact(r["key"], value)
                    if is_art == stored:
                        continue
                    if is_art and can_embed:
                        try:
                            _, text = embedding_input(r["key"], value)
                            vec = self._embedder.embed(text)
                            with self._tenant_statement(conn) as cur:
                                cur.execute(
                                    "UPDATE amfs_memory_entries "
                                    "SET is_artifact = %s, embedding = %s WHERE id = %s",
                                    (is_art, f"[{','.join(str(v) for v in vec)}]", r["id"]),
                                )
                        except Exception:  # noqa: BLE001
                            logger.warning(
                                "backfill re-embed failed for row %s", r["id"],
                                exc_info=True,
                            )
                            with self._tenant_statement(conn) as cur:
                                cur.execute(
                                    "UPDATE amfs_memory_entries "
                                    "SET is_artifact = %s WHERE id = %s",
                                    (is_art, r["id"]),
                                )
                    else:
                        with self._tenant_statement(conn) as cur:
                            cur.execute(
                                "UPDATE amfs_memory_entries SET is_artifact = %s WHERE id = %s",
                                (is_art, r["id"]),
                            )
                    updated += 1
        return updated

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
        else:
            conditions.append(_EXCLUDE_SHARED_PATHS)

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
        the adapter adds an OR-combined tsquery filter (see ``_fts.or_tsquery``)
        so multi-term queries recall documents matching ANY term instead of
        requiring every term, and — when ``sort_by`` is the default
        ``"confidence"`` — orders by ts_rank so keyword relevance is factored
        in.  Explicit ``sort_by`` values (``"recency"``, ``"version"``) are
        respected as-is.
        """
        use_fts = bool(query.query and getattr(self, "_has_search_tsv", False))
        col_ready = getattr(self, "_has_is_artifact_col", False)
        tsq_sql, tsq_params = or_tsquery(query.query) if use_fts else ("", [])

        conditions = ["namespace = %s", "branch = %s", "superseded_at IS NULL"]
        params: list[Any] = [self._namespace, branch]

        if query.depth < 3:
            conditions.append("tier <= %s")
            params.append(query.depth)

        # Exclude artifacts entirely at the SQL layer when the column is ready.
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

        use_priority_sort = query.sort_by == "priority"
        if use_priority_sort:
            order = "confidence DESC"
            fetch_limit = min(query.limit * 3, 1000)
        elif use_fts and query.sort_by == "confidence":
            order = f"ts_rank(search_tsv, {tsq_sql}) DESC, confidence DESC"
            params.extend(tsq_params)
            fetch_limit = query.limit
        else:
            order = order_map.get(query.sort_by, "confidence DESC")
            fetch_limit = query.limit

        # Demote artifacts beneath equally-ranked facts. With the column ready we
        # do it in SQL (leading sort key); otherwise we over-fetch and demote in
        # Python below so the pre-backfill window still behaves.
        if col_ready and not use_priority_sort:
            order = f"is_artifact ASC, {order}"
        elif not col_ready:
            fetch_limit = min(max(fetch_limit, query.limit * 3), 1000)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM amfs_memory_entries
            WHERE {where}
            ORDER BY {order}
            LIMIT %s
        """
        params.append(fetch_limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        entries = [self._row_to_entry(r) for r in rows]

        def _art(e: MemoryEntry) -> bool:
            return bool(e.is_artifact) if col_ready else classify_artifact(e.key, e.value)

        # Read-time fallback (column not yet backfilled): exclude or stable-demote
        # artifacts in Python. Stable so non-artifact ordering is preserved.
        if not col_ready:
            if not query.include_artifacts:
                entries = [e for e in entries if not _art(e)]
            elif not use_priority_sort:
                entries.sort(key=_art)

        if use_priority_sort:
            from amfs_core.tiering import PriorityScorer
            scorer = PriorityScorer()
            scores = scorer.score_batch(entries)
            entries.sort(
                key=lambda e: scores.get(e.entry_key, 0.0)
                * (ARTIFACT_PENALTY if _art(e) else 1.0),
                reverse=True,
            )
            entries = entries[: query.limit]
        elif not col_ready:
            entries = entries[: query.limit]

        return entries

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
        else:
            conditions.append(_EXCLUDE_SHARED_PATHS)
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
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
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

        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = "AND account_id IS NULL"
            acct_params = []

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
                          {account_filter}
                          AND ({direction_clause})
                        UNION ALL
                        SELECT g.*, gw.depth + 1
                        FROM amfs_knowledge_graph g
                        JOIN graph_walk gw ON g.source_entity = gw.target_entity
                        WHERE gw.depth < %s
                          AND g.namespace = %s AND g.branch = %s
                          AND g.confidence >= %s
                          {account_filter}
                    )
                    SELECT * FROM graph_walk
                    ORDER BY depth, confidence DESC
                    LIMIT %s
                    """,
                    (
                        namespace, branch, query.min_confidence,
                        *acct_params,
                        *entity_params,
                        query.depth, namespace, branch, query.min_confidence,
                        *acct_params,
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

        account_id = self._get_current_account_id()
        if account_id:
            conditions.append("account_id = %s")
            params.append(account_id)
        else:
            conditions.append("account_id IS NULL")

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

        account_id = self._get_current_account_id()
        if account_id:
            conditions.append("account_id = %s")
            params.append(account_id)
        else:
            conditions.append("account_id IS NULL")

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
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = "AND account_id IS NULL"
            acct_params = []

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) as total_edges,
                        COUNT(DISTINCT source_entity) + COUNT(DISTINCT target_entity) as approx_nodes,
                        AVG(confidence) as avg_confidence,
                        AVG(evidence_count) as avg_evidence
                    FROM amfs_knowledge_graph
                    WHERE namespace = %s AND branch = %s {account_filter}
                    """,
                    (namespace, branch, *acct_params),
                )
                row = cur.fetchone()
                if not row:
                    return {}

                cur.execute(
                    f"""
                    SELECT relation, COUNT(*) as cnt
                    FROM amfs_knowledge_graph
                    WHERE namespace = %s AND branch = %s {account_filter}
                    GROUP BY relation ORDER BY cnt DESC
                    """,
                    (namespace, branch, *acct_params),
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
        """Compute stats using SQL aggregates instead of full scan.

        Unscoped by construction, so shared namespaces are excluded here for
        the same reason as in entity_summaries and stats_extended.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
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
                      AND {_EXCLUDE_SHARED_PATHS}
                    """,
                    (self._namespace,),
                )
                row = cur.fetchone()

                cur.execute(
                    f"""
                    SELECT agent_id, COUNT(*) as cnt
                    FROM amfs_memory_entries
                    WHERE namespace = %s AND superseded_at IS NULL
                      AND {_EXCLUDE_SHARED_PATHS}
                    GROUP BY agent_id
                    """,
                    (self._namespace,),
                )
                agent_rows = cur.fetchall()

                cur.execute(
                    f"""
                    SELECT entity_path, COUNT(*) as cnt
                    FROM amfs_memory_entries
                    WHERE namespace = %s AND superseded_at IS NULL
                      AND {_EXCLUDE_SHARED_PATHS}
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

    def entity_summaries(
        self,
        *,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Per-entity aggregates via GROUP BY — never loads entry values."""
        conditions = ["namespace = %s", "branch = 'main'", "superseded_at IS NULL"]
        params: list[Any] = [self._namespace]
        if agent_ids is not None:
            conditions.append("agent_id = ANY(%s)")
            params.append(list(agent_ids))
        # No entity_path argument to this one: it is unscoped by construction,
        # so shared namespaces are always excluded. It groups by entity_path,
        # which would otherwise list a shared namespace's topics as though
        # they were this account's own.
        conditions.append(_EXCLUDE_SHARED_PATHS)
        where = " AND ".join(conditions)

        sql = f"""
            SELECT entity_path,
                   COUNT(*) AS entry_count,
                   AVG(confidence) AS avg_confidence,
                   MAX(written_at) AS last_updated,
                   (ARRAY_AGG(agent_id ORDER BY written_at DESC))[1] AS last_agent,
                   ARRAY_AGG(DISTINCT agent_id) AS agents,
                   -- The Postgres schema has no content_hash column (hashes
                   -- live only on filesystem-backed entries), so hashed_count
                   -- is always 0 here — matching entity_summaries_from_entries,
                   -- which sees content_hash=None on every Postgres row.
                   0::bigint AS hashed_count,
                   COALESCE(SUM(recall_count), 0) AS total_recalls,
                   {_RECALLED_TOKENS_SQL} AS recalled_tokens_saved
            FROM amfs_memory_entries
            WHERE {where}
            GROUP BY entity_path
            ORDER BY MAX(written_at) DESC
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            {
                "entity_path": r["entity_path"],
                "entry_count": r["entry_count"],
                "avg_confidence": float(r["avg_confidence"] or 0),
                "last_updated": r["last_updated"],
                "last_agent": r["last_agent"],
                "agents": sorted(r["agents"] or []),
                "hashed_count": r["hashed_count"],
                "total_recalls": int(r["total_recalls"] or 0),
                "recalled_tokens_saved": int(r["recalled_tokens_saved"] or 0),
            }
            for r in rows
        ]

    def agent_entity_stats(
        self,
        *,
        entity_path: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Per-(agent, entity) aggregates via GROUP BY — never loads values.

        The authority ranking runs over every pair on the account, so this must
        not be a full scan pulled into Python: an account with 5k entries would
        otherwise ship all of them over the wire to count rows.
        """
        conditions = ["namespace = %s", "branch = 'main'", "superseded_at IS NULL"]
        params: list[Any] = [self._namespace]
        if agent_ids is not None:
            conditions.append("agent_id = ANY(%s)")
            params.append(list(agent_ids))
        if entity_path:
            # Descendants included: a/b covers a/b/c but never a/bc, which is
            # why this is an equality OR a prefix-with-separator rather than a
            # bare LIKE 'a/b%'.
            prefix = entity_path.rstrip("/")
            conditions.append("(entity_path = %s OR entity_path LIKE %s)")
            params.extend([prefix, prefix + "/%"])
        else:
            # Same reasoning as entity_summaries: unscoped, this groups by
            # entity_path, so a shared namespace's topics would be listed as
            # though they were this account's own. Naming a path is the act of
            # opting into it.
            conditions.append(_EXCLUDE_SHARED_PATHS)
        where = " AND ".join(conditions)

        sql = f"""
            SELECT agent_id,
                   entity_path,
                   COUNT(*) AS entry_count,
                   AVG(confidence) AS avg_confidence,
                   MAX(written_at) AS last_written,
                   MIN(written_at) AS first_written,
                   COALESCE(SUM(recall_count), 0) AS total_recalls,
                   COUNT(*) FILTER (WHERE outcome_count > 0) AS outcome_linked_count
            FROM amfs_memory_entries
            WHERE {where}
            GROUP BY agent_id, entity_path
            ORDER BY entity_path, entry_count DESC, agent_id
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            {
                "agent_id": r["agent_id"],
                "entity_path": r["entity_path"],
                "entry_count": int(r["entry_count"]),
                "avg_confidence": float(r["avg_confidence"] or 0),
                "last_written": r["last_written"],
                "first_written": r["first_written"],
                "total_recalls": int(r["total_recalls"] or 0),
                "outcome_linked_count": int(r["outcome_linked_count"] or 0),
            }
            for r in rows
        ]

    def stats_extended(
        self,
        *,
        agent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extended aggregate stats in SQL (recalls, weekly deltas, types)."""
        conditions = ["namespace = %s", "superseded_at IS NULL"]
        params: list[Any] = [self._namespace]
        if agent_ids is not None:
            conditions.append("agent_id = ANY(%s)")
            params.append(list(agent_ids))
        # Unscoped by construction, like entity_summaries: there is no
        # entity_path argument to opt into a shared namespace, and the
        # breakdown below groups by entity_path. Without this, a shared
        # namespace's topics appear in the account's own entity list and its
        # entries inflate every total on the stats page.
        conditions.append(_EXCLUDE_SHARED_PATHS)
        where = " AND ".join(conditions)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_entries,
                        COUNT(DISTINCT entity_path) AS total_entities,
                        COUNT(DISTINCT agent_id) AS total_agents,
                        AVG(confidence) AS confidence_avg,
                        MIN(confidence) AS confidence_min,
                        MAX(confidence) AS confidence_max,
                        COUNT(*) FILTER (WHERE outcome_count > 0) AS outcome_linked_count,
                        MIN(written_at) AS oldest_entry_at,
                        MAX(written_at) AS newest_entry_at,
                        COALESCE(SUM(recall_count), 0) AS total_recalls,
                        {_RECALLED_TOKENS_SQL} AS recalled_tokens_saved,
                        COUNT(*) FILTER (WHERE written_at >= NOW() - INTERVAL '7 days')
                            AS entries_this_week,
                        COUNT(*) FILTER (
                            WHERE written_at >= NOW() - INTERVAL '14 days'
                              AND written_at < NOW() - INTERVAL '7 days'
                        ) AS entries_last_week
                    FROM amfs_memory_entries
                    WHERE {where}
                    """,
                    params,
                )
                row = cur.fetchone()

                cur.execute(
                    f"""
                    SELECT agent_id, COUNT(*) AS cnt
                    FROM amfs_memory_entries WHERE {where} GROUP BY agent_id
                    """,
                    params,
                )
                agent_rows = cur.fetchall()

                cur.execute(
                    f"""
                    SELECT entity_path, COUNT(*) AS cnt
                    FROM amfs_memory_entries WHERE {where} GROUP BY entity_path
                    """,
                    params,
                )
                entity_rows = cur.fetchall()

                cur.execute(
                    f"""
                    SELECT memory_type, COUNT(*) AS cnt
                    FROM amfs_memory_entries WHERE {where} GROUP BY memory_type
                    """,
                    params,
                )
                type_rows = cur.fetchall()

                # Weekly reuse deltas from the timestamped event log (READ /
                # CROSS_AGENT_READ events). recall_count on entries is a bare
                # counter, so this is the only source of *when* reuse happened.
                #
                # Scoped explicitly by account, as in graph_stats: amfs_events
                # has no row-level security, so unlike every other query here
                # this one is not isolated by RLS. Without the filter each
                # account is shown the whole deployment's reuse — measured on
                # production, one account's true +106% rendered as +472%.
                event_conditions = [
                    "namespace = %s",
                    "event_type IN ('read', 'cross_agent_read')",
                ]
                event_params: list[Any] = [self._namespace]
                event_account_id = self._get_current_account_id()
                if event_account_id:
                    event_conditions.append("account_id = %s")
                    event_params.append(event_account_id)
                else:
                    event_conditions.append("account_id IS NULL")
                if agent_ids is not None:
                    event_conditions.append("agent_id = ANY(%s)")
                    event_params.append(list(agent_ids))
                event_where = " AND ".join(event_conditions)
                recalls_this_week = 0
                recalls_last_week = 0
                try:
                    cur.execute(
                        f"""
                        SELECT
                            COUNT(*) FILTER (
                                WHERE created_at >= NOW() - INTERVAL '7 days'
                            ) AS recalls_this_week,
                            COUNT(*) FILTER (
                                WHERE created_at >= NOW() - INTERVAL '14 days'
                                  AND created_at < NOW() - INTERVAL '7 days'
                            ) AS recalls_last_week
                        FROM amfs_events
                        WHERE {event_where}
                        """,
                        event_params,
                    )
                    event_row = cur.fetchone()
                    if event_row:
                        recalls_this_week = int(event_row["recalls_this_week"] or 0)
                        recalls_last_week = int(event_row["recalls_last_week"] or 0)
                except Exception:
                    # Older deployments without the events table: deltas stay 0
                    # and the dashboard hides the trend line.
                    pass

        return {
            "recalls_this_week": recalls_this_week,
            "recalls_last_week": recalls_last_week,
            "total_entries": row["total_entries"] if row else 0,
            "total_entities": row["total_entities"] if row else 0,
            "total_agents": row["total_agents"] if row else 0,
            "agents": {r["agent_id"]: r["cnt"] for r in agent_rows},
            "entities": {r["entity_path"]: r["cnt"] for r in entity_rows},
            "confidence_avg": float(row["confidence_avg"] or 0) if row else 0.0,
            "confidence_min": float(row["confidence_min"] or 0) if row else 0.0,
            "confidence_max": float(row["confidence_max"] or 0) if row else 0.0,
            "outcome_linked_count": row["outcome_linked_count"] if row else 0,
            "oldest_entry_at": row["oldest_entry_at"] if row else None,
            "newest_entry_at": row["newest_entry_at"] if row else None,
            "total_recalls": int(row["total_recalls"]) if row else 0,
            "recalled_tokens_saved": int(row["recalled_tokens_saved"]) if row else 0,
            "entries_this_week": row["entries_this_week"] if row else 0,
            "entries_last_week": row["entries_last_week"] if row else 0,
            "memory_type_counts": {
                (r["memory_type"] or "fact"): r["cnt"] for r in type_rows
            },
        }

    def share_stats(
        self,
        *,
        since: datetime | None = None,
        pair_limit: int = 20,
        agent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Cross-agent share aggregate via JSONB unnest — full traces never
        leave the database."""
        conditions = [
            "t.namespace = %s",
            "NULLIF(ce->>'written_by', '') IS NOT NULL",
            "ce->>'written_by' <> t.agent_id",
        ]
        params: list[Any] = [self._namespace]
        if since is not None:
            conditions.append("t.created_at >= %s")
            params.append(since)
        if agent_ids is not None:
            conditions.append("t.agent_id = ANY(%s)")
            params.append(list(agent_ids))
            conditions.append("ce->>'written_by' = ANY(%s)")
            params.append(list(agent_ids))
        where = " AND ".join(conditions)

        sql = f"""
            SELECT t.agent_id AS reader, ce->>'written_by' AS author,
                   COUNT(*) AS cnt
            FROM amfs_decision_traces t
            CROSS JOIN LATERAL jsonb_array_elements(t.causal_entries) AS ce
            WHERE {where}
            GROUP BY t.agent_id, ce->>'written_by'
            ORDER BY cnt DESC
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        total = sum(r["cnt"] for r in rows)
        pairs = [
            {"reader": r["reader"], "author": r["author"], "count": r["cnt"]}
            for r in rows[:pair_limit]
        ]
        return {"total": total, "pairs": pairs}

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

    def _outcome_conditions(
        self,
        *,
        entity_path: str | None,
        since: datetime | None,
        outcome_ref: str | None = None,
    ) -> tuple[str, list[Any]]:
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

        if outcome_ref is not None:
            conditions.append("outcome_ref = %s")
            params.append(outcome_ref)

        return " AND ".join(conditions), params

    def count_outcomes(
        self,
        *,
        entity_path: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """``COUNT(*)`` over outcomes, so the number is right past any page cap."""
        where, params = self._outcome_conditions(entity_path=entity_path, since=since)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) AS cnt FROM amfs_outcomes WHERE {where}", params)
                row = cur.fetchone()
        return int(row["cnt"]) if row else 0

    def list_outcomes(
        self,
        *,
        entity_path: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
        outcome_ref: str | None = None,
    ) -> list[OutcomeRecord]:
        where, params = self._outcome_conditions(
            entity_path=entity_path, since=since, outcome_ref=outcome_ref
        )
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
                           decision_summary, task_input, response_text, tool_calls,
                           causal_entries, external_contexts,
                           query_events, session_started_at, session_ended_at,
                           session_duration_ms,
                           error_events, state_diff, session_metadata, created_at
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
                         error_events, state_diff, session_metadata, created_at,
                         task_input, response_text, tool_calls)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                            %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                            %s, %s, %s::jsonb)
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
                        # Dumped like every other model field on this row. The
                        # field is typed, so anything arriving over
                        # ``POST /api/v1/traces`` is a ``SessionMetadata`` by the
                        # time it reaches here and ``json.dumps`` cannot encode
                        # it — the trace was lost with a 500. A plain dict still
                        # gets through ``model_copy(update=...)``, which does not
                        # validate, so both shapes are accepted.
                        json.dumps(
                            trace.session_metadata.model_dump(mode="json")
                            if isinstance(trace.session_metadata, SessionMetadata)
                            else trace.session_metadata
                        ) if trace.session_metadata else "{}",
                        trace.created_at,
                        trace.task_input,
                        trace.response_text,
                        json.dumps([t.model_dump(mode="json") for t in trace.tool_calls]),
                    ),
                )
                row = cur.fetchone()
        return trace.model_copy(update={"id": str(row["id"]), "created_at": row["created_at"]})

    @staticmethod
    def _trace_conditions(
        namespace: str,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        cursor: str | None = None,
    ) -> tuple[str, list[Any]]:
        """WHERE clause shared by the trace list and count queries.

        Shared between the sync and async adapters, which is why it is static:
        the two must agree on what a page contains or a cursor issued by one
        would mean something else to the other.
        """
        conditions = ["namespace = %s"]
        params: list[Any] = [namespace]

        if entity_path is not None:
            # Containment rather than ``jsonb_array_elements``: ``@>`` is what
            # the jsonb_path_ops GIN index on causal_entries can serve, and it
            # asks the same question — is there an element whose entity_path
            # is this string.
            conditions.append("causal_entries @> %s::jsonb")
            params.append(json.dumps([{"entity_path": entity_path}]))
        if agent_id is not None:
            conditions.append("agent_id = %s")
            params.append(agent_id)
        if outcome_type is not None:
            conditions.append("outcome_type = %s")
            params.append(outcome_type)
        if since is not None:
            conditions.append("created_at >= %s")
            params.append(since)
        if until is not None:
            conditions.append("created_at < %s")
            params.append(until)
        if cursor:
            from amfs_core.pagination import decode_cursor

            cursor_ts, cursor_id = decode_cursor(cursor)
            # The row comparison is the keyset step; the plain ``created_at <=``
            # beside it is redundant for correctness but is a single-column
            # predicate the planner can use to prune partitions, which the
            # row comparison is not.
            conditions.append("created_at <= %s")
            params.append(cursor_ts)
            conditions.append("(created_at, id) < (%s, %s::uuid)")
            params.extend([cursor_ts, str(cursor_id)])

        return " AND ".join(conditions), params

    _TRACE_COLUMNS = """
        id, agent_id, session_id, outcome_ref, outcome_type,
        decision_summary, task_input, response_text, tool_calls,
        causal_entries, external_contexts,
        query_events, session_started_at, session_ended_at,
        session_duration_ms,
        error_events, state_diff, session_metadata, created_at
    """

    def list_traces(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        cursor: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[DecisionTrace]:
        where, params = self._trace_conditions(
            self._namespace,
            entity_path=entity_path,
            agent_id=agent_id,
            outcome_type=outcome_type,
            since=since,
            until=until,
            cursor=cursor,
        )
        sql = f"""
            SELECT {self._TRACE_COLUMNS}
            FROM amfs_decision_traces
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """
        params.append(limit)
        if not cursor and offset > 0:
            sql += " OFFSET %s"
            params.append(offset)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_trace(row) for row in rows]

    def count_traces(
        self,
        *,
        entity_path: str | None = None,
        agent_id: str | None = None,
        outcome_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        where, params = self._trace_conditions(
            self._namespace,
            entity_path=entity_path,
            agent_id=agent_id,
            outcome_type=outcome_type,
            since=since,
            until=until,
        )
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT count(*) AS cnt FROM amfs_decision_traces WHERE {where}", params
                )
                row = cur.fetchone()
        return int(row["cnt"]) if row else 0

    _TRACE_READ_COUNTS_SQL = """
        SELECT ce->>'entity_path' AS entity_path, ce->>'key' AS key, COUNT(*) AS cnt
        FROM amfs_decision_traces t
        CROSS JOIN LATERAL jsonb_array_elements(t.causal_entries) AS ce
        WHERE t.namespace = %s AND t.agent_id = %s
        GROUP BY ce->>'entity_path', ce->>'key'
    """

    @staticmethod
    def _read_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for r in rows:
            ep, key = r["entity_path"], r["key"]
            if ep is None or key is None:
                continue
            counts.setdefault(ep, {})[key] = int(r["cnt"])
        return counts

    def trace_read_counts(self, agent_id: str) -> dict[str, dict[str, int]]:
        """Per-entry read counts across an agent's traces, aggregated in SQL."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(self._TRACE_READ_COUNTS_SQL, (self._namespace, agent_id))
                rows = cur.fetchall()
        return self._read_counts_from_rows(rows)

    _ENTRIES_FOR_AGENT_ORDER = "ORDER BY written_at DESC, entity_path DESC, key DESC, version DESC"

    @staticmethod
    def _entries_for_agent_conditions(
        namespace: str,
        agent_id: str,
        *,
        since: datetime | None,
        until: datetime | None,
        cursor: str | None,
        exclude_prefixes: tuple[str, ...],
    ) -> tuple[str, list[Any]]:
        conditions = [
            "namespace = %s",
            "agent_id = %s",
            "superseded_at IS NULL",
            "branch = 'main'",
        ]
        params: list[Any] = [namespace, agent_id]
        for prefix in exclude_prefixes:
            conditions.append("entity_path NOT LIKE %s")
            params.append(prefix.replace("%", r"\%").replace("_", r"\_") + "%")
        if since is not None:
            conditions.append("written_at >= %s")
            params.append(since)
        if until is not None:
            conditions.append("written_at < %s")
            params.append(until)
        if cursor:
            from amfs_core.pagination import decode_cursor

            cursor_ts, tiebreak = decode_cursor(cursor)
            try:
                ep, key, version = tiebreak
            except (TypeError, ValueError) as exc:
                from amfs_core.pagination import InvalidCursorError

                raise InvalidCursorError("cursor does not belong to this listing") from exc
            conditions.append(
                "(written_at, entity_path, key, version) < (%s, %s, %s, %s)"
            )
            params.extend([cursor_ts, ep, key, int(version)])
        return " AND ".join(conditions), params

    def list_entries_for_agent(
        self,
        agent_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        cursor: str | None = None,
        limit: int = 100,
        exclude_prefixes: tuple[str, ...] = ("_system/",),
    ) -> list[MemoryEntry]:
        where, params = self._entries_for_agent_conditions(
            self._namespace,
            agent_id,
            since=since,
            until=until,
            cursor=cursor,
            exclude_prefixes=exclude_prefixes,
        )
        sql = f"""
            SELECT * FROM amfs_memory_entries
            WHERE {where}
            {self._ENTRIES_FOR_AGENT_ORDER}
            LIMIT %s
        """
        params.append(limit)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_entry(r) for r in rows]

    def list_traces_for_entry(
        self,
        *,
        agent_id: str,
        session_id: str | None,
        written_at: datetime | None = None,
        limit: int = 5,
    ) -> list[DecisionTrace]:
        """Traces from the session that wrote an entry, nearest write first.

        Uses idx_traces_session (namespace, agent_id, session_id).
        """
        if not session_id:
            return []

        conditions = ["namespace = %s", "agent_id = %s", "session_id = %s"]
        params: list[Any] = [self._namespace, agent_id, session_id]

        if written_at is not None:
            # A trace committed *before* the write cannot be the one that
            # committed it. Ordering by distance after the write picks the
            # right trace when a long session committed several outcomes.
            order = "ORDER BY (created_at >= %s) DESC, ABS(EXTRACT(EPOCH FROM (created_at - %s))) ASC"
            params.extend([written_at, written_at])
        else:
            order = "ORDER BY created_at DESC"

        where = " AND ".join(conditions)
        sql = f"""
            SELECT id, agent_id, session_id, outcome_ref, outcome_type,
                   decision_summary, task_input, response_text,
                   causal_entries, external_contexts,
                   query_events, session_started_at, session_ended_at,
                   session_duration_ms,
                   error_events, state_diff, session_metadata, created_at
            FROM amfs_decision_traces
            WHERE {where}
            {order}
            LIMIT %s
        """
        params.append(limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_trace(row) for row in rows]

    # ------------------------------------------------------------------
    # commits
    # ------------------------------------------------------------------
    #
    # These three inherited the base class's placeholders — a no-op save,
    # ``None``, and ``[]`` — for as long as this adapter has existed. The
    # effect was silent and complete: ``TransactionBuffer.flush`` assembled a
    # Commit, handed it to ``save_commit``, and it was dropped, so every
    # transaction returned a real id that resolved to nothing. ``commit_log``
    # was empty for every account and ``common_ancestor`` found no ancestor for
    # any pair of commits, including two that obviously shared one. All three
    # read as an honest answer about a history that had not been written yet,
    # which is why nothing complained.

    def save_commit(self, commit: Commit) -> None:
        """Persist a commit.

        Idempotent on the id, so saving the same commit object twice is safe
        and nothing about a commit is mutable enough to need an update branch.

        That is narrower than it sounds, and worth being exact about: the id is
        a fresh uuid4 minted per flush, not a hash of the contents. A retried
        flush is therefore a *different* commit, and ``ON CONFLICT`` will not
        collapse the two. It protects a replay of one commit object, not a
        retry of the operation that produced it.

        This also runs on its own connection, after ``write_batch`` has already
        committed the entries. If this insert fails, the entries stay and the
        commit row does not, leaving versions that name a commit nobody can
        look up. See ``TransactionBuffer.flush``.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_commits
                        (id, namespace, branch, message, author_agent_id,
                         session_id, entries, tree_hash, parent_ids, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        commit.id,
                        self._namespace,
                        commit.branch,
                        commit.message,
                        commit.author_agent_id,
                        commit.session_id,
                        json.dumps([e.model_dump(mode="json") for e in commit.entries]),
                        commit.tree_hash,
                        json.dumps(list(commit.parent_ids)),
                        commit.created_at,
                    ),
                )

    def get_commit(self, commit_id: str) -> Commit | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, namespace, branch, message, author_agent_id,
                           session_id, entries, tree_hash, parent_ids, created_at
                    FROM amfs_commits
                    WHERE id = %s AND namespace = %s
                    """,
                    (commit_id, self._namespace),
                )
                row = cur.fetchone()

        return self._row_to_commit(row) if row is not None else None

    def list_commits(
        self,
        *,
        branch: str = "main",
        limit: int = 50,
        namespace: str = "default",
    ) -> list[Commit]:
        """Commits, newest first.

        ``namespace`` is accepted for the interface but the adapter's own
        namespace is what is queried, as everywhere else here: it is fixed at
        construction and is the tenant boundary this connection was opened for.
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, namespace, branch, message, author_agent_id,
                           session_id, entries, tree_hash, parent_ids, created_at
                    FROM amfs_commits
                    WHERE namespace = %s AND branch = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (self._namespace, branch, limit),
                )
                rows = cur.fetchall()

        return [self._row_to_commit(row) for row in rows]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_commit(row: dict[str, Any]) -> Commit:
        # psycopg returns JSONB already decoded, but the same rows arrive as
        # strings through a plain cursor, so both are handled — the trace
        # reader above does the same for the same reason.
        entries = row.get("entries") or []
        parents = row.get("parent_ids") or []
        if isinstance(entries, str):
            entries = json.loads(entries)
        if isinstance(parents, str):
            parents = json.loads(parents)

        return Commit(
            id=row["id"],
            namespace=row["namespace"],
            branch=row["branch"],
            message=row["message"] or "",
            author_agent_id=row["author_agent_id"],
            session_id=row.get("session_id"),
            entries=[CommitEntry.model_validate(e) for e in entries],
            tree_hash=row.get("tree_hash"),
            parent_ids=list(parents),
            created_at=row["created_at"],
        )

    def _row_to_trace(self, row: dict[str, Any]) -> DecisionTrace:
        ce_raw = row["causal_entries"] or []
        ec_raw = row["external_contexts"] or []
        qe_raw = row.get("query_events") or []
        ee_raw = row.get("error_events") or []
        tc_raw = row.get("tool_calls") or []
        sd_raw = row.get("state_diff")
        if isinstance(tc_raw, str):
            tc_raw = json.loads(tc_raw)
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

        sm_raw = row.get("session_metadata") or {}
        if isinstance(sm_raw, str):
            sm_raw = json.loads(sm_raw)

        return DecisionTrace(
            id=str(row["id"]),
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            outcome_ref=row.get("outcome_ref"),
            outcome_type=row.get("outcome_type"),
            decision_summary=row.get("decision_summary"),
            task_input=row.get("task_input"),
            response_text=row.get("response_text"),
            tool_calls=[ToolCall(**t) for t in tc_raw],
            causal_entries=[TraceEntry(**e) for e in ce_raw],
            external_contexts=[ExternalContext(**c) for c in ec_raw],
            query_events=[QueryEvent(**q) for q in qe_raw],
            error_events=[ErrorEvent(**e) for e in ee_raw],
            state_diff=MemoryStateDiff(**sd_raw) if sd_raw else None,
            session_metadata=sm_raw,
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
            is_artifact=bool(row.get("is_artifact", False)),
        )

    # ── Digest storage (Memory Cortex) ──────────────────────────────

    def _get_current_account_id(self) -> str | None:
        """Read the current tenant account ID from thread-local context."""
        try:
            from amfs_postgres.tenant_context import get_request_tenant_account_id
            return get_request_tenant_account_id()
        except ImportError:
            return None

    def upsert_digest(self, digest: Digest) -> None:
        """Insert or update a compiled digest."""
        branch = digest.branch or "main"
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO amfs_digests
                   (namespace, branch, digest_type, scope, summary, entry_count,
                    source_agents, anticipation_score, compiled_at, account_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (namespace, branch, digest_type, scope, account_id)
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
                    account_id,
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
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                if account_id:
                    row = cur.execute(
                        """SELECT * FROM amfs_digests
                           WHERE namespace = %s AND branch = %s
                             AND digest_type = %s AND scope = %s
                             AND account_id = %s""",
                        (namespace, branch, digest_type.value, scope, account_id),
                    ).fetchone()
                else:
                    row = cur.execute(
                        """SELECT * FROM amfs_digests
                           WHERE namespace = %s AND branch = %s
                             AND digest_type = %s AND scope = %s
                             AND account_id IS NULL""",
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
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                if account_id:
                    account_filter = "AND account_id = %s"
                    base_params: list[Any] = [namespace, branch]
                    extra_params: list[Any] = [account_id]
                else:
                    account_filter = "AND account_id IS NULL"
                    base_params = [namespace, branch]
                    extra_params = []

                if digest_type:
                    rows = cur.execute(
                        f"""SELECT * FROM amfs_digests
                           WHERE namespace = %s AND branch = %s AND digest_type = %s
                           {account_filter}
                           ORDER BY compiled_at DESC""",
                        (*base_params, digest_type.value, *extra_params),
                    ).fetchall()
                else:
                    rows = cur.execute(
                        f"""SELECT * FROM amfs_digests
                           WHERE namespace = %s AND branch = %s
                           {account_filter}
                           ORDER BY compiled_at DESC""",
                        (*base_params, *extra_params),
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

    _AGENT_COLUMNS = """id, namespace, agent_id, display_name,
                    created_at, last_active_at, entry_count,
                    profile, capabilities, contracts"""

    def ensure_agent(self, agent_id: str, namespace: str = "default") -> Agent:
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
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
                row = cur.fetchone()
        return self._row_to_agent(row)

    def register_agent(self, agent_id: str, namespace: str = "default") -> Agent:
        """Ensure an agent row exists WITHOUT counting it as a memory write.

        Unlike ``ensure_agent`` this does not increment ``entry_count`` — it is
        used when an agent announces itself (e.g. via set_identity / profile
        update) so the agent appears on the dashboard before it writes any
        memory. Idempotent: refreshes ``last_active_at`` on conflict.
        """
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
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
                row = cur.fetchone()
        return self._row_to_agent(row)

    def get_agent(self, agent_id: str, namespace: str = "default") -> Agent | None:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {self._AGENT_COLUMNS}
                    FROM amfs_agents
                    WHERE namespace = %s AND agent_id = %s {account_filter}
                    """,
                    (namespace, agent_id, *acct_params),
                )
                row = cur.fetchone()
        return self._row_to_agent(row) if row else None

    def list_agents(self, namespace: str = "default") -> list[Agent]:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT {self._AGENT_COLUMNS}
                    FROM amfs_agents
                    WHERE namespace = %s {account_filter}
                    ORDER BY last_active_at DESC
                    """,
                    (namespace, *acct_params),
                )
                rows = cur.fetchall()
        return [self._row_to_agent(r) for r in rows]

    @staticmethod
    def _row_to_agent(row: dict[str, Any]) -> Agent:
        from amfs_core.models import AgentProfile, AgentCapability, MemoryContract

        profile_raw = row.get("profile")
        profile = AgentProfile.model_validate(profile_raw) if profile_raw else None

        caps_raw = row.get("capabilities") or []
        capabilities = [AgentCapability.model_validate(c) for c in caps_raw]

        contracts_raw = row.get("contracts") or []
        contracts = [MemoryContract.model_validate(c) for c in contracts_raw]

        return Agent(
            id=str(row["id"]),
            namespace=row["namespace"],
            agent_id=row["agent_id"],
            display_name=row.get("display_name"),
            created_at=row["created_at"],
            last_active_at=row["last_active_at"],
            entry_count=row.get("entry_count", 0),
            profile=profile,
            capabilities=capabilities,
            contracts=contracts,
        )

    def update_agent_profile(
        self,
        agent_id: str,
        profile: Any,
        namespace: str = "default",
    ) -> Agent:
        # register (not ensure) — announcing a profile is not a memory write,
        # so it must not inflate entry_count.
        self.register_agent(agent_id, namespace)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE amfs_agents
                    SET profile = %s
                    WHERE namespace = %s AND agent_id = %s
                    RETURNING {self._AGENT_COLUMNS}
                    """,
                    (json.dumps(profile.model_dump()), namespace, agent_id),
                )
                row = cur.fetchone()
        return self._row_to_agent(row)

    def update_agent_capabilities(
        self,
        agent_id: str,
        capabilities: list[Any],
        namespace: str = "default",
    ) -> Agent:
        self.ensure_agent(agent_id, namespace)
        caps_json = json.dumps([c.model_dump() for c in capabilities])
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE amfs_agents
                    SET capabilities = %s
                    WHERE namespace = %s AND agent_id = %s
                    RETURNING {self._AGENT_COLUMNS}
                    """,
                    (caps_json, namespace, agent_id),
                )
                row = cur.fetchone()
        return self._row_to_agent(row)

    def update_agent_contracts(
        self,
        agent_id: str,
        contracts: list[Any],
        namespace: str = "default",
    ) -> Agent:
        self.ensure_agent(agent_id, namespace)
        contracts_json = json.dumps([c.model_dump() for c in contracts])
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE amfs_agents
                    SET contracts = %s
                    WHERE namespace = %s AND agent_id = %s
                    RETURNING {self._AGENT_COLUMNS}
                    """,
                    (contracts_json, namespace, agent_id),
                )
                row = cur.fetchone()
        return self._row_to_agent(row)

    # ── Agent groups ──────────────────────────────────────────────────

    def create_agent_group(self, group, namespace: str = "default"):
        from amfs_core.models import AgentGroup

        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_agent_groups
                        (namespace, account_id, name, description, color, icon,
                         position, auto_generated, source_cluster_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at, updated_at
                    """,
                    (
                        namespace,
                        account_id,
                        group.name,
                        group.description or "",
                        group.color,
                        group.icon,
                        group.position,
                        group.auto_generated,
                        group.source_cluster_id,
                    ),
                )
                row = cur.fetchone()
        return AgentGroup(
            id=str(row["id"]),
            namespace=namespace,
            account_id=str(account_id) if account_id else None,
            name=group.name,
            description=group.description or "",
            color=group.color,
            icon=group.icon,
            position=group.position,
            auto_generated=group.auto_generated,
            source_cluster_id=group.source_cluster_id,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_agent_groups(self, namespace: str = "default") -> list:
        from amfs_core.models import AgentGroup

        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND g.account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT g.id, g.namespace, g.account_id, g.name,
                           g.description, g.color, g.icon, g.position,
                           g.auto_generated, g.source_cluster_id,
                           g.created_at, g.updated_at,
                           COALESCE(ARRAY_AGG(m.agent_id) FILTER (WHERE m.agent_id IS NOT NULL), '{{}}'::TEXT[]) AS agent_ids,
                           COUNT(m.agent_id) AS member_count
                    FROM amfs_agent_groups g
                    LEFT JOIN amfs_agent_group_members m ON m.group_id = g.id
                    WHERE g.namespace = %s {account_filter}
                    GROUP BY g.id
                    ORDER BY g.position, g.created_at
                    """,
                    (namespace, *acct_params),
                )
                rows = cur.fetchall()

        return [
            AgentGroup(
                id=str(r["id"]),
                namespace=r["namespace"],
                account_id=str(r["account_id"]) if r["account_id"] else None,
                name=r["name"],
                description=r["description"] or "",
                color=r["color"],
                icon=r["icon"],
                position=float(r["position"]),
                auto_generated=r["auto_generated"],
                source_cluster_id=r["source_cluster_id"],
                member_count=r["member_count"],
                agent_ids=list(r["agent_ids"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def get_agent_group(self, group_id: str, namespace: str = "default"):
        from amfs_core.models import AgentGroup

        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND g.account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT g.id, g.namespace, g.account_id, g.name,
                           g.description, g.color, g.icon, g.position,
                           g.auto_generated, g.source_cluster_id,
                           g.created_at, g.updated_at,
                           COALESCE(ARRAY_AGG(m.agent_id) FILTER (WHERE m.agent_id IS NOT NULL), '{{}}'::TEXT[]) AS agent_ids,
                           COUNT(m.agent_id) AS member_count
                    FROM amfs_agent_groups g
                    LEFT JOIN amfs_agent_group_members m ON m.group_id = g.id
                    WHERE g.namespace = %s AND g.id = %s {account_filter}
                    GROUP BY g.id
                    """,
                    (namespace, group_id, *acct_params),
                )
                row = cur.fetchone()

        if not row:
            return None
        return AgentGroup(
            id=str(row["id"]),
            namespace=row["namespace"],
            account_id=str(row["account_id"]) if row["account_id"] else None,
            name=row["name"],
            description=row["description"] or "",
            color=row["color"],
            icon=row["icon"],
            position=float(row["position"]),
            auto_generated=row["auto_generated"],
            source_cluster_id=row["source_cluster_id"],
            member_count=row["member_count"],
            agent_ids=list(row["agent_ids"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_agent_group(
        self,
        group_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
        icon: str | None = None,
        position: float | None = None,
        namespace: str = "default",
    ):
        from amfs_core.models import AgentGroup

        fields: list[str] = []
        params: list[Any] = []
        if name is not None:
            fields.append("name = %s")
            params.append(name)
        if description is not None:
            fields.append("description = %s")
            params.append(description)
        if color is not None:
            fields.append("color = %s")
            params.append(color)
        if icon is not None:
            fields.append("icon = %s")
            params.append(icon)
        if position is not None:
            fields.append("position = %s")
            params.append(position)

        if not fields:
            return self.get_agent_group(group_id, namespace)

        fields.append("updated_at = NOW()")
        set_clause = ", ".join(fields)

        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            params.append(account_id)
        else:
            account_filter = ""

        params.extend([namespace, group_id])

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE amfs_agent_groups
                    SET {set_clause}
                    WHERE namespace = %s AND id = %s {account_filter}
                    RETURNING id, namespace, account_id, name, description,
                              color, icon, position, auto_generated,
                              source_cluster_id, created_at, updated_at
                    """,
                    params,
                )
                row = cur.fetchone()
        if not row:
            return None
        return AgentGroup(
            id=str(row["id"]),
            namespace=row["namespace"],
            account_id=str(row["account_id"]) if row["account_id"] else None,
            name=row["name"],
            description=row["description"] or "",
            color=row["color"],
            icon=row["icon"],
            position=float(row["position"]),
            auto_generated=row["auto_generated"],
            source_cluster_id=row["source_cluster_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def delete_agent_group(self, group_id: str, namespace: str = "default") -> bool:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM amfs_agent_groups
                    WHERE namespace = %s AND id = %s {account_filter}
                    """,
                    (namespace, group_id, *acct_params),
                )
                deleted = cur.rowcount > 0
        return deleted

    def add_agents_to_group(
        self,
        group_id: str,
        agent_ids: list[str],
        added_by: str = "user",
        namespace: str = "default",
    ) -> int:
        if not agent_ids:
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM amfs_agent_group_members
                    WHERE namespace = %s AND agent_id = ANY(%s)
                    """,
                    (namespace, agent_ids),
                )
                values = [(group_id, aid, namespace, added_by) for aid in agent_ids]
                cur.executemany(
                    """
                    INSERT INTO amfs_agent_group_members
                        (group_id, agent_id, namespace, added_by)
                    VALUES (%s, %s, %s, %s)
                    """,
                    values,
                )
                count = cur.rowcount
        return count

    def remove_agents_from_group(
        self,
        group_id: str,
        agent_ids: list[str],
        namespace: str = "default",
    ) -> int:
        if not agent_ids:
            return 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM amfs_agent_group_members
                    WHERE group_id = %s AND namespace = %s AND agent_id = ANY(%s)
                    """,
                    (group_id, namespace, agent_ids),
                )
                count = cur.rowcount
        return count

    def reorder_agent_groups(
        self,
        positions: list[tuple[str, float]],
        namespace: str = "default",
    ) -> None:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
        else:
            account_filter = ""

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for group_id, position in positions:
                    params: list[Any] = [position, namespace, group_id]
                    if account_id:
                        params.append(account_id)
                    cur.execute(
                        f"""
                        UPDATE amfs_agent_groups
                        SET position = %s, updated_at = NOW()
                        WHERE namespace = %s AND id = %s {account_filter}
                        """,
                        params,
                    )

    def list_agents_enriched(self, namespace: str = "default") -> list[dict[str, Any]]:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND a.account_id = %s"
            entry_account_filter = "AND me.account_id = %s"
            graph_account_filter = "AND kg.account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            entry_account_filter = ""
            graph_account_filter = ""
            acct_params = []

        sql = f"""
            WITH agent_entries AS (
                SELECT
                    me.agent_id,
                    me.entity_path,
                    me.memory_type,
                    COUNT(*) AS cnt,
                    MAX(me.written_at) AS last_written
                FROM amfs_memory_entries me
                WHERE me.namespace = %s
                  AND me.superseded_at IS NULL
                  {entry_account_filter}
                GROUP BY me.agent_id, me.entity_path, me.memory_type
            ),
            agent_stats AS (
                SELECT
                    ae.agent_id,
                    SUM(ae.cnt)::INT AS entries_written,
                    COUNT(DISTINCT ae.entity_path) AS entities_touched,
                    MAX(ae.last_written) AS last_active,
                    SPLIT_PART(
                        MODE() WITHIN GROUP (ORDER BY ae.entity_path),
                        '/', 1
                    ) AS primary_repo,
                    jsonb_agg(DISTINCT ae.entity_path) AS entity_paths,
                    jsonb_object_agg(
                        ae.memory_type,
                        ae.cnt
                    ) FILTER (WHERE ae.memory_type IS NOT NULL) AS type_dist
                FROM agent_entries ae
                GROUP BY ae.agent_id
            ),
            agent_collab AS (
                SELECT
                    kg.target_entity AS agent_id,
                    ARRAY_AGG(DISTINCT kg.source_entity) AS collaborators
                FROM amfs_knowledge_graph kg
                WHERE kg.namespace = %s
                  AND kg.relation = 'learned_from'
                  {graph_account_filter}
                GROUP BY kg.target_entity
            )
            SELECT
                a.agent_id,
                a.created_at,
                COALESCE(s.entries_written, 0) AS entries_written,
                COALESCE(s.entities_touched, 0) AS entities_touched,
                COALESCE(s.last_active, a.last_active_at) AS last_active,
                s.primary_repo,
                COALESCE(s.entity_paths, '[]'::jsonb) AS entity_paths,
                COALESCE(s.type_dist, '{{}}'::jsonb) AS type_dist,
                COALESCE(c.collaborators, '{{}}'::TEXT[]) AS collaborators,
                gm.group_id::TEXT AS group_id,
                COALESCE(
                    a.profile->>'platform',
                    a.profile->'session_metadata'->>'platform'
                ) AS platform,
                COALESCE(
                    a.profile->>'model',
                    a.profile->'session_metadata'->>'model'
                ) AS model,
                COALESCE(
                    a.profile->'inferred_tags',
                    a.profile->'tags'
                ) AS inferred_tags,
                a.profile->>'description' AS description
            FROM amfs_agents a
            LEFT JOIN agent_stats s ON s.agent_id = a.agent_id
            LEFT JOIN agent_collab c ON c.agent_id = a.agent_id
            LEFT JOIN amfs_agent_group_members gm
                ON gm.agent_id = a.agent_id AND gm.namespace = a.namespace
            WHERE a.namespace = %s {account_filter}
            ORDER BY last_active DESC NULLS LAST
        """
        params: list[Any] = [
            namespace, *acct_params,
            namespace, *acct_params,
            namespace, *acct_params,
        ]

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            tags = r["inferred_tags"]
            if isinstance(tags, str):
                tags = json.loads(tags)
            results.append({
                "agent_id": r["agent_id"],
                "created_at": r["created_at"],
                "entries_written": r["entries_written"],
                "entities_touched": r["entities_touched"],
                "last_active": r["last_active"],
                "primary_repo": r["primary_repo"],
                "entity_paths": r["entity_paths"] if isinstance(r["entity_paths"], list) else json.loads(r["entity_paths"]) if isinstance(r["entity_paths"], str) else r["entity_paths"],
                "type_dist": r["type_dist"] if isinstance(r["type_dist"], dict) else json.loads(r["type_dist"]) if isinstance(r["type_dist"], str) else r["type_dist"],
                "collaborators": list(r["collaborators"]),
                "group_id": r["group_id"],
                "platform": r["platform"],
                "model": r["model"],
                "inferred_tags": tags if isinstance(tags, list) else [],
                "description": r["description"],
            })
        return results

    def get_agent_activity_histogram(
        self,
        agent_id: str,
        days: int = 7,
        namespace: str = "default",
    ) -> list[int]:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND e.account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []

        sql = f"""
            SELECT d.day::date AS day,
                   COUNT(e.id)::INT AS cnt
            FROM generate_series(
                (CURRENT_DATE - INTERVAL '%s days'),
                CURRENT_DATE,
                INTERVAL '1 day'
            ) AS d(day)
            LEFT JOIN amfs_events e
                ON e.agent_id = %s
               AND e.namespace = %s
               AND e.created_at::date = d.day::date
               {account_filter}
            GROUP BY d.day
            ORDER BY d.day
        """
        params: list[Any] = [days - 1, agent_id, namespace, *acct_params]

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [r["cnt"] for r in rows]

    def dismiss_cluster_suggestion(self, cluster_id: str, account_id: str) -> None:
        acct = self._get_current_account_id() or account_id
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_agent_group_suggestions_dismissed
                        (account_id, cluster_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (acct, cluster_id),
                )

    def list_dismissed_cluster_ids(self, account_id: str) -> list[str]:
        acct = self._get_current_account_id() or account_id
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT cluster_id
                    FROM amfs_agent_group_suggestions_dismissed
                    WHERE account_id = %s
                    """,
                    (acct,),
                )
                rows = cur.fetchall()
        return [r["cluster_id"] for r in rows]

    # ── Event log / timeline (Pro) ────────────────────────────────────

    def log_event(self, event: Event) -> Event:
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
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
        offset: int = 0,
        cursor: str | None = None,
        until: datetime | None = None,
    ) -> list[Event]:
        where, params = self._event_conditions(
            namespace,
            agent_id,
            account_id=self._get_current_account_id(),
            branch=branch,
            event_type=event_type,
            since=since,
            until=until,
            cursor=cursor,
        )
        sql = f"""
            SELECT id, namespace, agent_id, branch, event_type,
                   summary, details, actor_agent_id, created_at
            FROM amfs_events
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """
        params.append(limit)
        if not cursor and offset > 0:
            sql += " OFFSET %s"
            params.append(offset)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _event_conditions(
        namespace: str,
        agent_id: str,
        *,
        account_id: str | None,
        branch: str | None,
        event_type: str | None,
        since: datetime | None,
        until: datetime | None,
        cursor: str | None,
    ) -> tuple[str, list[Any]]:
        """WHERE clause for the timeline, shared with the async adapter."""
        conditions = ["namespace = %s", "agent_id = %s"]
        params: list[Any] = [namespace, agent_id]

        if account_id:
            conditions.append("account_id = %s")
            params.append(account_id)
        if branch is not None:
            conditions.append("branch = %s")
            params.append(branch)
        if event_type is not None:
            conditions.append("event_type = %s")
            params.append(event_type)
        if since is not None:
            conditions.append("created_at >= %s")
            params.append(since)
        if until is not None:
            conditions.append("created_at < %s")
            params.append(until)
        if cursor:
            from amfs_core.pagination import decode_cursor

            cursor_ts, cursor_id = decode_cursor(cursor)
            conditions.append("(created_at, id) < (%s, %s::uuid)")
            params.extend([cursor_ts, str(cursor_id)])
        return " AND ".join(conditions), params

    def get_event(self, event_id: str, namespace: str = "default") -> Event | None:
        """Fetch one event by id, scoped to the caller's account.

        The id comes from the client: the rollback endpoints resolve
        ``target_event_id`` through here and take the timestamp — and, in the
        branching plugin, the agent — off whatever comes back. amfs_events has
        no RLS, so without the filter that lookup reaches into other accounts
        and an id from one becomes a rollback target in another. Missing tenant
        context matches only untenanted rows rather than everything: a 404 is
        the safe failure here, an unscoped read is not.
        """
        conditions = ["id = %s", "namespace = %s"]
        params: list[Any] = [event_id, namespace]
        account_id = self._get_current_account_id()
        if account_id:
            conditions.append("account_id = %s")
            params.append(account_id)
        else:
            conditions.append("account_id IS NULL")

        sql = f"""
            SELECT id, namespace, agent_id, branch, event_type,
                   summary, details, actor_agent_id, created_at
            FROM amfs_events
            WHERE {" AND ".join(conditions)}
            LIMIT 1
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
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
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_branches
                        (namespace, name, parent_branch, branched_at,
                         created_by, description, status, account_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                        account_id,
                    ),
                )
                row = cur.fetchone()
        return branch.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
        })

    def get_branch(self, name: str, namespace: str = "default") -> Branch | None:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM amfs_branches
                    WHERE namespace = %s AND name = %s {account_filter}
                    """,
                    (namespace, name, *acct_params),
                )
                row = cur.fetchone()
        return self._row_to_branch(row) if row else None

    def list_branches(
        self, namespace: str = "default", *, status: str | None = None
    ) -> list[Branch]:
        conditions = ["namespace = %s"]
        params: list[Any] = [namespace]

        account_id = self._get_current_account_id()
        if account_id:
            conditions.append("account_id = %s")
            params.append(account_id)

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
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE amfs_branches
                    SET status = 'closed'
                    WHERE namespace = %s AND name = %s {account_filter}
                    RETURNING *
                    """,
                    (namespace, name, *acct_params),
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
                        if self._has_is_artifact_col:
                            columns.append("is_artifact")
                            params_list.append(be.get("is_artifact", False))
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
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_branch_access
                        (namespace, branch_name, grantee_type, grantee_id,
                         permission, granted_by, account_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT uq_branch_access DO UPDATE
                        SET permission = EXCLUDED.permission,
                            granted_by = EXCLUDED.granted_by,
                            granted_at = NOW(),
                            account_id = COALESCE(amfs_branch_access.account_id, EXCLUDED.account_id)
                    RETURNING id, granted_at
                    """,
                    (
                        access.namespace,
                        access.branch_name,
                        access.grantee_type,
                        access.grantee_id,
                        access.permission.value if isinstance(access.permission, BranchAccessPermission) else access.permission,
                        access.granted_by,
                        account_id,
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
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            conn.execute(
                f"""
                DELETE FROM amfs_branch_access
                WHERE namespace = %s AND branch_name = %s
                  AND grantee_type = %s AND grantee_id = %s
                  {account_filter}
                """,
                (namespace, branch_name, grantee_type, grantee_id, *acct_params),
            )

    def list_branch_access(
        self, branch_name: str, namespace: str = "default"
    ) -> list[BranchAccess]:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM amfs_branch_access
                    WHERE namespace = %s AND branch_name = %s
                    {account_filter}
                    ORDER BY granted_at DESC
                    """,
                    (namespace, branch_name, *acct_params),
                )
                rows = cur.fetchall()
        return [self._row_to_branch_access(r) for r in rows]

    def check_branch_access(
        self, branch_name: str, api_key_id: str, namespace: str = "default"
    ) -> str | None:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND ba.account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT ba.permission FROM amfs_branch_access ba
                    WHERE ba.namespace = %s AND ba.branch_name = %s
                      {account_filter}
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
                    (namespace, branch_name, *acct_params, api_key_id, api_key_id, namespace),
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
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_tags
                        (namespace, name, branch, tagged_at, description, created_by, event_id, account_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (tag.namespace, tag.name, tag.branch, tag.tagged_at,
                     tag.description, tag.created_by, tag.event_id, account_id),
                )
                row = cur.fetchone()
        return tag.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
        })

    def get_tag(self, name: str, namespace: str = "default") -> Tag | None:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM amfs_tags WHERE namespace = %s AND name = %s {account_filter}",
                    (namespace, name, *acct_params),
                )
                row = cur.fetchone()
        return self._row_to_tag(row) if row else None

    def list_tags(
        self, namespace: str = "default", *, branch: str | None = None
    ) -> list[Tag]:
        conditions = ["namespace = %s"]
        params: list[Any] = [namespace]

        account_id = self._get_current_account_id()
        if account_id:
            conditions.append("account_id = %s")
            params.append(account_id)

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
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            conn.execute(
                f"DELETE FROM amfs_tags WHERE namespace = %s AND name = %s {account_filter}",
                (namespace, name, *acct_params),
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
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_pull_requests
                        (namespace, title, description, source_branch,
                         target_branch, status, created_by, account_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at, updated_at
                    """,
                    (pr.namespace, pr.title, pr.description, pr.source_branch,
                     pr.target_branch, pr.status.value if isinstance(pr.status, PullRequestStatus) else pr.status,
                     pr.created_by, account_id),
                )
                row = cur.fetchone()
        return pr.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    def get_pull_request(self, pr_id: str, namespace: str = "default") -> PullRequest | None:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM amfs_pull_requests WHERE id = %s::uuid AND namespace = %s {account_filter}",
                    (pr_id, namespace, *acct_params),
                )
                row = cur.fetchone()
        return self._row_to_pr(row) if row else None

    def list_pull_requests(
        self, namespace: str = "default", *, status: str | None = None
    ) -> list[PullRequest]:
        conditions = ["namespace = %s"]
        params: list[Any] = [namespace]

        account_id = self._get_current_account_id()
        if account_id:
            conditions.append("account_id = %s")
            params.append(account_id)

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
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            params.append(account_id)
        else:
            account_filter = ""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""UPDATE amfs_pull_requests SET {', '.join(updates)}
                        WHERE id = %s::uuid AND namespace = %s {account_filter} RETURNING *""",
                    params,
                )
                row = cur.fetchone()
        if row is None:
            raise AdapterError(f"PR '{pr_id}' not found")
        return self._row_to_pr(row)

    def add_pr_review(self, review: PRReview) -> PRReview:
        account_id = self._get_current_account_id()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO amfs_pr_reviews
                        (namespace, pr_id, reviewer, status, comment, entry_path, account_id)
                    VALUES (%s, %s::uuid, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (review.namespace, review.pr_id, review.reviewer,
                     review.status.value if isinstance(review.status, PRReviewStatus) else review.status,
                     review.comment, review.entry_path, account_id),
                )
                row = cur.fetchone()
        return review.model_copy(update={
            "id": str(row["id"]),
            "created_at": row["created_at"],
        })

    def list_pr_reviews(self, pr_id: str, namespace: str = "default") -> list[PRReview]:
        account_id = self._get_current_account_id()
        if account_id:
            account_filter = "AND account_id = %s"
            acct_params: list[Any] = [account_id]
        else:
            account_filter = ""
            acct_params = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM amfs_pr_reviews WHERE pr_id = %s::uuid AND namespace = %s {account_filter} ORDER BY created_at",
                    (pr_id, namespace, *acct_params),
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
                             branch, embedding, is_artifact)
                            VALUES
                            (%s, %s, %s, 1, %s,
                             %s, %s, %s, %s,
                             NOW(), %s, 0,
                             %s, %s, %s, %s,
                             'main', %s, %s)
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
                                row.get("is_artifact", False),
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
