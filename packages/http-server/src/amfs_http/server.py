"""AMFS HTTP/REST API server — universal access to Agent Memory over HTTP.

Provides a FastAPI application exposing the full AMFS API as REST endpoints
with SSE streaming for real-time watch notifications and optional API key
authentication.

Run directly::

    amfs-http                          # default 0.0.0.0:8741
    amfs-http --port 9000 --host 127.0.0.1
    amfs-http --reload                 # development mode
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from amfs import AgentMemory, MemoryType, OutcomeType
from amfs.config import load_config_or_default
from pydantic import BaseModel, Field
from amfs_core.models import (
    AgentGroup,
    AMFSConfig,
    DecisionTrace,
    Event,
    EventType,
    GraphEdge,
    LayerConfig,
    MemoryEntry,
    SearchQuery,
    SemanticQuery,
)
from amfs_core.quality import HeuristicQualityEvaluator

from amfs_http.auth import verify_api_key
from amfs_http.models import (
    AddTeamMemberRequest,
    ContextRequest,
    CreateAPIKeyRequest,
    CreateSnapshotRequest,
    CreateTeamRequest,
    EventRequest,
    OutcomeRequest,
    RetrieveRequest,
    RunPatternDetectionRequest,
    SearchRequest,
    UpdateTeamMemberRequest,
    UpdateTeamRequest,
    WriteRequest,
)
from amfs_http.sse import SSEManager

logger = logging.getLogger(__name__)

# ── Async adapter (hot-path, non-blocking) ──────────────────────────
_async_adapter = None  # AsyncPostgresAdapter | None, set in lifespan


@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ARG001
    """Open/close the async connection pool for hot-path DB access."""
    global _async_adapter
    dsn = os.environ.get("AMFS_POSTGRES_DSN")
    if dsn and not os.environ.get("AMFS_HTTP_URL"):
        try:
            from amfs_postgres.async_adapter import AsyncPostgresAdapter
            ns = os.environ.get("AMFS_NAMESPACE", "default")
            _async_adapter = AsyncPostgresAdapter(dsn=dsn, namespace=ns)
            await _async_adapter.open()
            logger.info("Async Postgres adapter started (namespace=%s)", ns)
        except Exception:
            logger.warning("Failed to start async adapter — falling back to sync", exc_info=True)
            _async_adapter = None
    yield
    if _async_adapter is not None:
        await _async_adapter.close()
        logger.info("Async Postgres adapter closed")


app = FastAPI(
    title="AMFS HTTP API",
    description="Agent Memory File System — REST API with SSE support",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _dashboard_tenant_middleware(request: Request, call_next):
    """Optional X-AMFS-Dashboard-Account-Id + secret → amfs.current_account_id on DB connections."""
    from amfs_http.tenant_middleware import apply_tenant_headers_from_request, clear_tenant_headers

    did_set = apply_tenant_headers_from_request(request)
    try:
        return await call_next(request)
    finally:
        if did_set:
            clear_tenant_headers()


_memory: AgentMemory | None = None
_sse_manager = SSEManager()
_bg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="amfs-bg")
_known_agents: set[str] = set()

# ── Semantic embedder (shared by write-time embedding + /retrieve) ──────
# One instance for the whole process so write and query vectors come from the
# SAME model (otherwise cosine similarity is meaningless). Env-gated and fully
# crash-safe: if the embedder can't be built, we return None and every caller
# falls back to lexical behaviour — a bad rollout degrades to the status quo,
# it never breaks writes or reads.
_UNSET_EMBEDDER: Any = object()
_server_embedder: Any = _UNSET_EMBEDDER


def _embeddings_enabled() -> bool:
    return os.environ.get("AMFS_ENABLE_EMBEDDINGS", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _get_server_embedder():
    """Return the process-wide embedder, or None if disabled/unavailable."""
    global _server_embedder
    if _server_embedder is _UNSET_EMBEDDER:
        _server_embedder = None
        if _embeddings_enabled():
            try:
                from amfs_core.default_embedder import create_default_embedder

                _server_embedder = create_default_embedder()
                logger.info(
                    "Semantic embedder ready: %s",
                    type(_server_embedder).__name__,
                )
            except Exception:  # noqa: BLE001 - never block startup on the embedder
                logger.warning(
                    "Embedder init failed — semantic retrieval disabled, "
                    "falling back to lexical search",
                    exc_info=True,
                )
                _server_embedder = None
        else:
            logger.info("AMFS_ENABLE_EMBEDDINGS is off — semantic retrieval disabled")
    return _server_embedder

_immutable_trace_store = None
try:
    from amfs_traces.api import mount_pro_routes
    mount_pro_routes(app)
    logger.info("Pro trace endpoints mounted at /api/v1/pro/traces")

    from amfs_traces.store import PostgresImmutableTraceStore
    from amfs_traces.crypto import seal, get_signing_key, get_signing_key_id
    from amfs_traces.models import ImmutableDecisionTrace, TraceEntry, TraceExternalContext
    _HAS_PRO_TRACES = True
except ImportError:
    _HAS_PRO_TRACES = False

try:
    from amfs_cortex_pro import mount_cortex_pro
    mount_cortex_pro(app)
    logger.info("Pro Cortex endpoints mounted (local)")
except ImportError:
    from amfs_http.pro_proxy import mount_pro_proxy
    mount_pro_proxy(app)

def _get_memory() -> AgentMemory:
    """Lazily initialise the shared AgentMemory singleton."""
    global _memory
    if _memory is not None:
        return _memory

    agent_id = os.environ.get("AMFS_AGENT_ID", "http-server")

    http_url = os.environ.get("AMFS_HTTP_URL")
    if http_url:
        if os.environ.get("AMFS_POSTGRES_DSN"):
            logger.warning(
                "Both AMFS_HTTP_URL and AMFS_POSTGRES_DSN are set. "
                "The HTTP adapter takes precedence — direct DB access "
                "is bypassed in favour of the authenticated HTTP API."
            )
        try:
            from amfs_adapter_http import HttpAdapter

            api_key = os.environ.get("AMFS_API_KEY", "")
            logger.info(
                "AMFS HTTP adapter mode — routing through %s", http_url
            )
            adapter = HttpAdapter(base_url=http_url, api_key=api_key)
            _memory = AgentMemory(agent_id=agent_id, adapter=adapter)
            return _memory
        except ImportError:
            logger.warning(
                "AMFS_HTTP_URL is set but amfs-adapter-http is not installed. "
                "Falling back to local adapter. "
                "Install with: pip install amfs-adapter-http"
            )

    config = _resolve_config()

    ttl_interval_str = os.environ.get("AMFS_TTL_SWEEP_INTERVAL")
    ttl_sweep_interval = float(ttl_interval_str) if ttl_interval_str else 300.0

    logger.info("AMFS HTTP server starting — agent_id=%s", agent_id)
    mem = AgentMemory(
        agent_id=agent_id,
        config_path=None,
        adapter=None,
        ttl_sweep_interval=ttl_sweep_interval,
    )

    mem._config = config
    from amfs.factory import create_adapter_from_config

    adapter = create_adapter_from_config(config)
    mem._adapter = adapter
    mem._engine._adapter = adapter
    mem._propagator._adapter = adapter

    # Share the process-wide embedder so mem.write() embeds on the sync
    # fallback path and mem.semantic_search() works. The hot async path embeds
    # explicitly in the write endpoint. Safe no-op when embeddings are disabled.
    embedder = _get_server_embedder()
    if embedder is not None:
        mem._embedder = embedder
        try:
            if hasattr(adapter, "_embedder") and getattr(adapter, "_embedder", None) is None:
                adapter._embedder = embedder
        except Exception:  # noqa: BLE001 - adapter embedding is best-effort
            logger.debug("Could not attach embedder to adapter", exc_info=True)

    _memory = mem
    return _memory


try:
    from amfs_pro_api import mount_pro_api
    mount_pro_api(app, get_memory=_get_memory)
    logger.info("Pro API endpoints mounted (intelligence, extraction)")
except ImportError:
    pass

try:
    from amfs_rooms import mount_rooms
    mount_rooms(app, get_memory=_get_memory)
    logger.info("Room endpoints mounted")
except ImportError:
    pass

try:
    from amfs_http.openrouter_proxy import mount_openrouter_proxy
    mount_openrouter_proxy(app, get_memory=_get_memory)
except Exception:  # noqa: BLE001 - proxy is optional; never block startup
    logger.debug("OpenRouter proxy not mounted", exc_info=True)

try:
    from amfs_tenant.http_deps import mount_scope_enforcement
    mount_scope_enforcement(app)
except ImportError:
    pass


def _resolve_config() -> AMFSConfig:
    """Resolve AMFS configuration from environment or config files.

    Priority:
    1. AMFS_POSTGRES_DSN env var -> Postgres adapter
    2. AMFS_DATA_DIR env var -> filesystem adapter at that path
    3. amfs.yaml discovery -> load from file
    4. Default -> filesystem adapter at .amfs/
    """
    postgres_dsn = os.environ.get("AMFS_POSTGRES_DSN")
    if postgres_dsn:
        return AMFSConfig(
            namespace="default",
            layers={
                "primary": LayerConfig(
                    adapter="postgres",
                    options={"dsn": postgres_dsn},
                )
            },
        )

    data_dir = os.environ.get("AMFS_DATA_DIR")
    if data_dir:
        return AMFSConfig(
            namespace="default",
            layers={
                "primary": LayerConfig(
                    adapter="filesystem",
                    options={"root": data_dir},
                )
            },
        )

    return load_config_or_default()


def _entry_to_response(entry: MemoryEntry) -> dict[str, Any]:
    """Convert a MemoryEntry to a JSON-safe dict, stripping embeddings."""
    data = entry.model_dump(mode="json")
    data.pop("embedding", None)
    return data


def _get_visibility_filter(request: Request):
    """Return the UserVisibilityFilter from request.state, or None."""
    return getattr(request.state, "visibility_filter", None)


def _ensure_agent_owner(
    request: Request, agent_id: str, namespace: str = "default"
) -> None:
    """Link the agent to the API key's owner user in the Pro agents table.

    This is a no-op when amfs_rooms is not installed or the request
    has no tenant context.  The linkage lets the UserVisibilityFilter
    discover which agents belong to which user.
    """
    ctx = getattr(request.state, "tenant_ctx", None)
    user_id = getattr(request.state, "user_id", None)
    if not ctx or not user_id:
        return
    try:
        from amfs_rooms.visibility import ensure_agent_owner

        mem = _get_memory()
        pool = getattr(mem._adapter, "_pool", None)
        if pool is not None:
            ensure_agent_owner(
                pool,
                agent_id=agent_id,
                owner_user_id=user_id,
                account_id=ctx.account_id,
                namespace=namespace,
            )
    except ImportError:
        pass
    except Exception:
        logger.debug("Failed to ensure agent owner for %s", agent_id, exc_info=True)


# ──────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/health")
async def health_v1() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/auth/whoami")
async def whoami(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return information about the authenticated caller.

    When Pro middleware is active, this returns account, key type, scopes,
    and rate limit info. In OSS mode, returns basic auth status.
    """
    ctx = getattr(request.state, "tenant_ctx", None)
    user_id = getattr(request.state, "user_id", None)
    vis = _get_visibility_filter(request)

    if ctx is not None:
        result: dict[str, Any] = {
            "authenticated": True,
            "mode": "pro",
            "account_id": str(ctx.account_id),
            "actor_id": str(ctx.actor_id),
            "key_type": ctx.key_type.value if ctx.key_type else None,
            "role": ctx.role.value if ctx.role else None,
            "scopes": [
                {
                    "entity_path_pattern": s.entity_path_pattern,
                    "permission": s.permission.value,
                }
                for s in ctx.scopes
            ],
            "rate_limit_rpm": ctx.rate_limit_rpm,
            "is_admin": ctx.is_admin,
            "user_id": str(user_id) if user_id else None,
            "visibility_filter_active": vis is not None,
            "visibility_filtering": vis is not None and vis.should_filter() if vis else False,
        }
        if vis is not None and vis.should_filter():
            try:
                result["visible_agents"] = sorted(vis.get_visible_agent_ids())
            except Exception:
                result["visible_agents_error"] = True
        return result
    return {
        "authenticated": _auth is not None,
        "mode": "oss",
        "user_id": str(user_id) if user_id else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Entries
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/entries/{entity_path:path}/{key}")
async def read_entry(
    request: Request,
    entity_path: str,
    key: str,
    branch: str = Query("main"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    if _async_adapter is not None:
        try:
            entry = await _async_adapter.read(entity_path, key, branch=branch)
        except Exception:
            logger.warning("Async read failed for %s/%s — falling back to sync", entity_path, key, exc_info=True)
            entry = None
        if entry is None:
            entry = mem.read(entity_path, key, branch=branch)
            if entry is not None:
                logger.warning(
                    "Async adapter missed entry %s/%s but sync found it — RLS context mismatch",
                    entity_path, key,
                )
        if entry is not None:
            asyncio.create_task(_async_adapter.increment_recall_count(entity_path, key, branch=branch))
    else:
        entry = mem.read(entity_path, key, branch=branch)
    if entry is None:
        return {"status": "not_found", "entity_path": entity_path, "key": key}

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_entry_visible(entry):
        return {"status": "not_found", "entity_path": entity_path, "key": key}

    return _entry_to_response(entry)


@app.get("/api/v1/quality/{entity_path:path}/{key}")
async def entry_quality(
    entity_path: str,
    key: str,
    branch: str = Query("main"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Compute a quality report for a stored entry on demand."""
    mem = _get_memory()
    entry = mem.read(entity_path, key, branch=branch)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    try:
        existing_entries = mem.list(entity_path)
        existing_keys = [e.key for e in existing_entries if e.key != key]
    except Exception:
        existing_keys = []

    evaluator = HeuristicQualityEvaluator()
    mt = entry.memory_type.value if hasattr(entry.memory_type, "value") else str(entry.memory_type)
    report = evaluator.evaluate(
        entry.value,
        entity_path=entity_path,
        key=key,
        confidence=entry.confidence,
        memory_type=mt,
        pattern_refs=list(entry.provenance.pattern_refs),
        existing_keys=existing_keys,
    )
    return {
        "entity_path": entity_path,
        "key": key,
        "quality": report.model_dump(mode="json"),
    }


@app.post("/api/v1/entries")
async def write_entry(
    req: WriteRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    type_map = {
        "fact": MemoryType.FACT,
        "belief": MemoryType.BELIEF,
        "experience": MemoryType.EXPERIENCE,
    }
    mt = type_map.get(req.memory_type.lower(), MemoryType.FACT)

    original_agent = mem._tagger.agent_id if req.agent_id else None
    if req.agent_id:
        mem._tagger.agent_id = req.agent_id

    try:
        _used_async = False
        if _async_adapter is not None:
            _agent_ns = _async_adapter._namespace
            if req.agent_id:
                _agent_cache_key = f"{req.agent_id}:{_agent_ns}"
                if _agent_cache_key not in _known_agents:
                    try:
                        await _async_adapter.ensure_agent(req.agent_id, _agent_ns)
                        _known_agents.add(_agent_cache_key)
                    except Exception:
                        pass
                    _ensure_agent_owner(request, req.agent_id, _agent_ns)

            from amfs_core.models import Provenance
            provenance = mem._tagger.tag(pattern_refs=req.pattern_refs or None)
            entry_obj = MemoryEntry(
                entity_path=req.entity_path,
                key=req.key,
                version=1,
                value=req.value,
                provenance=provenance,
                confidence=req.confidence,
                memory_type=mt,
                shared=req.shared,
                branch=req.branch,
            )
            # Write-time embedding for semantic retrieval. The async adapter
            # persists entry.embedding when the pgvector column exists; without
            # this the hot write path stores no vector (embeddings never land).
            # Crash-safe: a failure here just stores the entry without a vector.
            _embedder = _get_server_embedder()
            if _embedder is not None:
                try:
                    entry_obj = entry_obj.model_copy(
                        update={"embedding": _embedder.embed_value(req.value)}
                    )
                except Exception:  # noqa: BLE001 - never fail a write on embedding
                    logger.warning(
                        "write-time embedding failed for %s/%s — storing without vector",
                        req.entity_path, req.key, exc_info=True,
                    )
            try:
                entry = await _async_adapter.write(entry_obj)
                _used_async = True
            except Exception:
                logger.warning(
                    "Async write failed for %s/%s — falling back to sync adapter",
                    req.entity_path, req.key, exc_info=True,
                )
                entry = mem.write(
                    req.entity_path,
                    req.key,
                    req.value,
                    confidence=req.confidence,
                    pattern_refs=req.pattern_refs or None,
                    memory_type=mt,
                    shared=req.shared,
                    branch=req.branch,
                )
        if not _used_async and _async_adapter is None:
            if req.agent_id:
                _agent_cache_key = f"{req.agent_id}:{mem.namespace}"
                if _agent_cache_key not in _known_agents:
                    try:
                        mem._adapter.ensure_agent(req.agent_id, mem.namespace)
                        _known_agents.add(_agent_cache_key)
                    except Exception:
                        pass
                    _ensure_agent_owner(request, req.agent_id, mem.namespace)
            entry = mem.write(
                req.entity_path,
                req.key,
                req.value,
                confidence=req.confidence,
                pattern_refs=req.pattern_refs or None,
                memory_type=mt,
                shared=req.shared,
                branch=req.branch,
            )
    finally:
        if original_agent is not None:
            mem._tagger.agent_id = original_agent
    _sse_manager.broadcast(entry)

    _resource = f"{req.entity_path}/{req.key}"
    _ip = request.client.host if request.client else None
    _agent = entry.provenance.agent_id
    _ek = f"{entry.entity_path}/{entry.key}"
    _ns = _async_adapter._namespace if _async_adapter else mem.namespace
    _branch = entry.branch or "main"
    _confidence = entry.confidence

    if _async_adapter is not None:
        async def _bg_async_side_effects() -> None:
            try:
                await _async_adapter.log_event(Event(
                    namespace=_ns,
                    agent_id=_agent,
                    branch=_branch,
                    event_type=EventType.WRITE,
                    summary=f"Wrote {req.entity_path}/{req.key} v{entry.version}",
                    details={
                        "entity_path": req.entity_path,
                        "key": req.key,
                        "version": entry.version,
                        "confidence": _confidence,
                        "memory_type": mt.value,
                        "shared": req.shared,
                    },
                ))
            except Exception:
                logger.debug("bg: Failed to log write event", exc_info=True)
            try:
                await _async_adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=_agent,
                        source_type="agent",
                        relation="wrote",
                        target_entity=_ek,
                        target_type="entry",
                        confidence=_confidence,
                        provenance={"agent_id": _agent, "trigger": "write"},
                    ),
                    namespace=_ns,
                    branch=_branch,
                )
            except Exception:
                logger.debug("bg: Failed to materialize wrote edge", exc_info=True)

        asyncio.create_task(_bg_async_side_effects())
    else:
        _tenant_account_id: str | None = None
        _tenant_team_id: str | None = None
        _tenant_is_admin: bool = False
        try:
            from amfs_postgres.tenant_context import (
                get_request_tenant_account_id,
                get_request_tenant_team_id,
                get_request_is_account_admin,
            )
            _tenant_account_id = get_request_tenant_account_id()
            _tenant_team_id = get_request_tenant_team_id()
            _tenant_is_admin = get_request_is_account_admin()
        except ImportError:
            pass

        def _bg_write_side_effects() -> None:
            try:
                from amfs_postgres.tenant_context import (
                    set_tls_tenant_account_id,
                    set_tls_tenant_team_id,
                    set_tls_is_account_admin,
                    clear_tls_tenant_account_id,
                    clear_tls_tenant_team_id,
                    clear_tls_is_account_admin,
                )
                set_tls_tenant_account_id(_tenant_account_id)
                set_tls_tenant_team_id(_tenant_team_id)
                set_tls_is_account_admin(_tenant_is_admin)
            except ImportError:
                pass

            try:
                _audit_log("memory.write", resource=_resource, ip_address=_ip)
            except Exception:
                logger.debug("bg: Failed to write audit log", exc_info=True)
            try:
                mem._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=_agent,
                        source_type="agent",
                        relation="wrote",
                        target_entity=_ek,
                        target_type="entry",
                        confidence=_confidence,
                        provenance={"agent_id": _agent, "trigger": "write"},
                    ),
                    namespace=_ns,
                    branch=_branch,
                )
            except Exception:
                logger.debug("bg: Failed to materialize wrote edge", exc_info=True)
            finally:
                try:
                    from amfs_postgres.tenant_context import (
                        clear_tls_tenant_account_id,
                        clear_tls_tenant_team_id,
                        clear_tls_is_account_admin,
                    )
                    clear_tls_tenant_account_id()
                    clear_tls_tenant_team_id()
                    clear_tls_is_account_admin()
                except ImportError:
                    pass

        _bg_executor.submit(_bg_write_side_effects)

    return _entry_to_response(entry)


@app.get("/api/v1/entries")
async def list_entries(
    request: Request,
    entity_path: str | None = Query(None),
    branch: str = Query("main"),
    include_superseded: bool = Query(False),
    limit: int | None = Query(None, ge=1, le=10_000),
    offset: int = Query(0, ge=0),
    sort: str | None = Query(None, pattern="^(written_at|recall_count)$"),
    fields: str | None = Query(None, pattern="^meta$"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        from amfs_postgres.tenant_context import get_request_tenant_account_id
        _tls_acct = get_request_tenant_account_id()
    except ImportError:
        _tls_acct = "NO_MODULE"
    _state_acct = getattr(request.state, "account_id", None)
    _state_user = getattr(request.state, "user_id", None)
    _has_ctx = getattr(request.state, "tenant_ctx", None) is not None
    logger.warning(
        "[TLS-DIAG] /entries tls_account=%s state_account=%s state_user=%s has_tenant_ctx=%s",
        _tls_acct, _state_acct, _state_user, _has_ctx,
    )
    mem = _get_memory()
    if _async_adapter is not None:
        try:
            entries = await _async_adapter.list(entity_path, branch=branch, include_superseded=include_superseded)
        except Exception:
            logger.warning("Async list failed for %s — falling back to sync", entity_path, exc_info=True)
            entries = []
        if not entries:
            sync_entries = mem.list(entity_path, branch=branch, include_superseded=include_superseded)
            if sync_entries:
                logger.warning(
                    "Async adapter returned 0 entries for %s but sync found %d — RLS context mismatch",
                    entity_path, len(sync_entries),
                )
                entries = sync_entries
    else:
        entries = mem.list(entity_path, branch=branch, include_superseded=include_superseded)
    total_before = len(entries)

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        entries = vis.filter_entries(entries)
        logger.warning(
            "[ENTRIES] entity_path=%s mem.list=%d after_filter=%d user_agents=%s",
            entity_path, total_before, len(entries),
            sorted(vis.get_user_agents()) if vis else "N/A",
        )
    else:
        logger.warning(
            "[ENTRIES] entity_path=%s mem.list=%d NO_FILTER vis=%s",
            entity_path, total_before, vis,
        )

    # Sorting/pagination/meta happen after the visibility filter so callers
    # can never page past entries they aren't allowed to see. Defaults keep
    # the historical behavior (full list, full fields) intact.
    if sort == "written_at":
        entries = sorted(entries, key=lambda e: e.provenance.written_at, reverse=True)
    elif sort == "recall_count":
        entries = sorted(entries, key=lambda e: e.recall_count, reverse=True)

    total = len(entries)
    if offset:
        entries = entries[offset:]
    if limit is not None:
        entries = entries[:limit]

    payload = [_entry_to_response(e) for e in entries]
    if fields == "meta":
        for item in payload:
            item.pop("value", None)
            item.pop("artifact_refs", None)

    return {"entries": payload, "total": total}


@app.get("/api/v1/entities")
async def list_entity_summaries(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Per-entity aggregates without entry values — a few KB instead of the
    multi-MB /entries payload. Dashboards should prefer this endpoint."""
    mem = _get_memory()

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        # Room visibility can't be expressed as a per-agent SQL filter, so
        # filter entries in Python and aggregate with the shared helper.
        from amfs_core.aggregates import entity_summaries_from_entries

        entries = vis.filter_entries(mem.list())
        summaries = entity_summaries_from_entries(entries)
    else:
        summaries = mem._adapter.entity_summaries()

    return json.loads(json.dumps({"entities": summaries}, default=str))


# ──────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/search")
async def search_entries(
    request: Request,
    req: SearchRequest,
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    branch = getattr(req, "branch", "main") or "main"
    sq = SearchQuery(
        query=req.query,
        entity_path=req.entity_path,
        min_confidence=req.min_confidence,
        max_confidence=req.max_confidence,
        agent_id=req.agent_id,
        since=req.since,
        pattern_ref=req.pattern_ref,
        sort_by=req.sort_by,
        limit=req.limit,
        depth=req.depth,
    )
    mem = _get_memory()
    if _async_adapter is not None:
        try:
            results = await _async_adapter.search(sq, branch=branch)
        except Exception:
            logger.warning("Async search failed — falling back to sync", exc_info=True)
            results = []
        if not results:
            try:
                sync_results = mem._adapter.search(sq, branch=branch)
            except TypeError:
                sync_results = mem._adapter.search(sq)
            if sync_results:
                logger.warning(
                    "Async search returned 0 results but sync found %d — RLS context mismatch",
                    len(sync_results),
                )
                results = sync_results
    else:
        try:
            results = mem._adapter.search(sq, branch=branch)
        except TypeError:
            results = mem._adapter.search(sq)

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        results = vis.filter_entries(results)

    return [_entry_to_response(e) for e in results]


@app.post("/api/v1/retrieve")
async def retrieve_entries(
    request: Request,
    req: RetrieveRequest,
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Semantic (meaning-based) retrieval.

    Ranks entries by embedding similarity to the query, blended with recency
    and confidence, so plain-language queries match paraphrased memories.
    Account isolation comes from the RLS-scoped async adapter; user/room
    visibility is then enforced by UserVisibilityFilter (same as /search).

    Degrades gracefully to lexical search when the embedder or pgvector column
    is unavailable, so this endpoint never returns worse than /search.
    """
    from datetime import datetime as _dt, timezone as _tz

    branch = req.branch or "main"
    vis = _get_visibility_filter(request)
    embedder = _get_server_embedder()

    pairs: list[tuple[MemoryEntry, float]] = []
    if embedder is not None and _async_adapter is not None:
        # Over-fetch a candidate pool so the recency/confidence blend has
        # headroom beyond pure similarity, then trim to `limit` after scoring.
        pool = max(req.limit * 5, 50)
        sq = SemanticQuery(
            text=req.query,
            entity_path=req.entity_path,
            min_confidence=req.min_confidence,
            limit=pool,
        )
        try:
            pairs = await _async_adapter.semantic_search(sq, embedder, branch=branch)
        except Exception:
            logger.warning("semantic_search failed — falling back to lexical", exc_info=True)
            pairs = []

    if pairs:
        if vis is not None and vis.should_filter():
            allowed = {id(e) for e in vis.filter_entries([e for e, _ in pairs])}
            pairs = [(e, s) for e, s in pairs if id(e) in allowed]

        now = _dt.now(_tz.utc)
        half_life = 30.0
        scored: list[tuple[MemoryEntry, float, dict[str, float]]] = []
        for entry, sim in pairs:
            written = getattr(entry.provenance, "written_at", None)
            if written is not None:
                if written.tzinfo is None:
                    written = written.replace(tzinfo=_tz.utc)
                age_days = max(0.0, (now - written).total_seconds() / 86400.0)
                recency = 0.5 ** (age_days / half_life)
            else:
                recency = 0.0
            conf = float(entry.confidence)
            score = (
                req.semantic_weight * sim
                + req.recency_weight * recency
                + req.confidence_weight * conf
            )
            scored.append((entry, score, {
                "semantic": round(sim, 4),
                "recency": round(recency, 4),
                "confidence": round(conf, 4),
            }))

        scored.sort(key=lambda t: t[1], reverse=True)
        out: list[dict[str, Any]] = []
        for entry, score, breakdown in scored[: req.limit]:
            data = _entry_to_response(entry)
            data["_score"] = round(score, 4)
            data["_breakdown"] = breakdown
            out.append(data)
        return out

    # ── Lexical fallback (embedder/pgvector unavailable, or no vector hits) ──
    sq_lex = SearchQuery(
        query=req.query,
        entity_path=req.entity_path,
        min_confidence=req.min_confidence,
        limit=req.limit,
        sort_by="confidence",
        depth=3,
    )
    mem = _get_memory()
    results: list[MemoryEntry] = []
    if _async_adapter is not None:
        try:
            results = await _async_adapter.search(sq_lex, branch=branch)
        except Exception:
            results = []
    if not results:
        try:
            results = mem._adapter.search(sq_lex, branch=branch)
        except TypeError:
            results = mem._adapter.search(sq_lex)
        except Exception:
            results = []
    if vis is not None and vis.should_filter():
        results = vis.filter_entries(results)
    return [_entry_to_response(e) for e in results]


# ──────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/stats")
async def get_stats(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        # Visibility-scoped stats. This branch must return a SUPERSET of the
        # MemoryStats shape (which the unfiltered branch below produces),
        # because the client parses /stats via MemoryStats.model_validate —
        # any missing field silently defaults (confidence→0.0, outcome→0) and
        # any mis-named key (e.g. "oldest_entry" vs "oldest_entry_at") is
        # dropped to None. Room visibility semantics (co-member entries on
        # shared entity paths) can't be expressed as a plain agent_id filter,
        # so this path filters in Python and aggregates via the same shared
        # helper the adapter defaults use.
        from amfs_core.aggregates import extended_stats_from_entries

        entries = vis.filter_entries(mem.list())
        scoped = extended_stats_from_entries(entries)
        return json.loads(json.dumps(scoped, default=str))

    stats = mem._adapter.stats_extended()
    return json.loads(json.dumps(stats, default=str))


# ──────────────────────────────────────────────────────────────────────
# Integrity verification
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/verify")
async def verify_integrity(
    body: dict[str, Any] = {},
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    entity_path = body.get("entity_path")
    return mem.verify(entity_path)


# ──────────────────────────────────────────────────────────────────────
# Atomic commits
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/commits")
async def create_commit(
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    writes = body.get("writes", [])
    message = body.get("message", "")
    with mem.transaction(message) as tx:
        for w in writes:
            tx.write(w["entity_path"], w["key"], w.get("value"))
    return {
        "commit_id": tx.commit.id if tx.commit else None,
        "message": message,
        "entries_written": len(writes),
    }


@app.get("/api/v1/commits")
async def list_commits(
    limit: int = Query(50, ge=1, le=500),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    commits = mem.commit_log(limit=limit)
    return {
        "commits": [json.loads(json.dumps(c.model_dump(mode="json"), default=str)) for c in commits],
        "count": len(commits),
    }


@app.get("/api/v1/commits/{commit_id}")
async def get_commit(
    commit_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    commit = mem.get_commit(commit_id)
    if commit is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Commit not found")
    return json.loads(json.dumps(commit.model_dump(mode="json"), default=str))


@app.post("/api/v1/merge-base")
async def merge_base(
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    ancestor = mem.common_ancestor(body["commit_a"], body["commit_b"])
    return {
        "ancestor_commit_id": ancestor,
        "commit_a": body["commit_a"],
        "commit_b": body["commit_b"],
    }


# ──────────────────────────────────────────────────────────────────────
# Agent binding
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents/{agent_id:path}/profile")
async def get_agent_profile(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return an agent's profile, stats, and registration info.

    When no explicit profile has been registered, synthesises one from the
    agent's actual activity — entity paths become auto-inferred capabilities
    and memory-type distribution is included.
    """
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    agent = mem._adapter.get_agent(agent_id, namespace=mem.namespace)

    entries = [
        e for e in mem.list()
        if e.provenance.agent_id == agent_id
        and not e.entity_path.startswith("_system/")
    ]
    entities_touched = {e.entity_path for e in entries}
    last_active = max(
        (e.provenance.written_at for e in entries),
        default=None,
    )
    traces = mem._adapter.list_traces(agent_id=agent_id, limit=10000)

    result: dict[str, Any] = {
        "agentId": agent_id,
        "entriesWritten": len(entries),
        "entitiesTouched": len(entities_touched),
        "entityPaths": sorted(entities_touched),
        "totalReads": sum(len(t.causal_entries) for t in traces),
        "decisionTraces": len(traces),
        "lastActive": last_active.isoformat() if last_active else None,
    }

    if agent:
        profile = agent.profile
        capabilities = agent.capabilities
        contracts = agent.contracts

        if not profile and not capabilities and entries:
            profile, capabilities = _synthesize_agent_profile(
                agent_id, entries, traces,
            )

        result["profile"] = profile.model_dump() if profile else {}
        result["capabilities"] = [c.model_dump() for c in capabilities]
        result["contracts"] = [c.model_dump() for c in contracts]
        result["displayName"] = agent.display_name
        result["createdAt"] = agent.created_at.isoformat() if agent.created_at else None
    return result


def _synthesize_agent_profile(
    agent_id: str,
    entries: list,
    traces: list,
) -> tuple:
    """Build an AgentProfile and capabilities from observed activity."""
    from amfs_core.models import AgentProfile, AgentCapability, MemoryType
    from collections import Counter

    entities_touched = {e.entity_path for e in entries}
    memory_types: Counter[str] = Counter()
    key_prefixes: Counter[str] = Counter()
    for e in entries:
        mt = e.memory_type if hasattr(e, "memory_type") and e.memory_type else MemoryType.FACT
        memory_types[mt.value if hasattr(mt, "value") else str(mt)] += 1
        prefix = e.key.split("-")[0] if "-" in e.key else e.key
        key_prefixes[prefix] += 1

    top_prefixes = [p for p, _ in key_prefixes.most_common(5)]
    type_summary = ", ".join(
        f"{count} {mtype}" for mtype, count in memory_types.most_common()
    )
    desc_parts = []
    if type_summary:
        desc_parts.append(f"Writes: {type_summary}.")
    if entities_touched:
        desc_parts.append(
            f"Active across {len(entities_touched)} "
            f"entit{'y' if len(entities_touched) == 1 else 'ies'}."
        )
    if traces:
        desc_parts.append(f"{len(traces)} decision trace(s) recorded.")

    profile = AgentProfile(
        description=" ".join(desc_parts),
        auto_context_paths=sorted(entities_touched)[:10],
        tags=_infer_tags(agent_id, top_prefixes, memory_types),
    )

    capabilities = []
    entity_groups: dict[str, list[str]] = {}
    for ep in sorted(entities_touched):
        group = ep.split("/")[0] if "/" in ep else ep
        entity_groups.setdefault(group, []).append(ep)

    for group, paths in entity_groups.items():
        capabilities.append(AgentCapability(
            name=group,
            description=f"Works on {len(paths)} entit{'y' if len(paths) == 1 else 'ies'} under {group}/",
            entity_paths=paths[:10],
        ))

    return profile, capabilities


def _infer_tags(
    agent_id: str,
    key_prefixes: list[str],
    memory_types: "Counter",
) -> list[str]:
    """Derive a small set of tags from the agent's activity."""
    tags: list[str] = []
    prefix_tag_map = {
        "task": "task-executor",
        "pattern": "pattern-detector",
        "risk": "risk-assessor",
        "decision": "decision-maker",
        "action": "action-logger",
    }
    for prefix in key_prefixes:
        if prefix in prefix_tag_map and prefix_tag_map[prefix] not in tags:
            tags.append(prefix_tag_map[prefix])

    if memory_types.get("belief", 0) > memory_types.get("fact", 0):
        tags.append("hypothesis-driven")
    if memory_types.get("experience", 0) > 0:
        tags.append("experiential")

    return tags[:5]


@app.put("/api/v1/agents/{agent_id:path}/profile")
async def update_agent_profile(
    request: Request,
    agent_id: str,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from amfs_core.models import AgentProfile

    mem = _get_memory()
    profile = AgentProfile.model_validate(body)
    agent = mem._adapter.update_agent_profile(agent_id, profile)
    # Link the agent to the API key's owner so it shows on the user's dashboard
    # immediately — set_identity announces through this endpoint before the
    # agent has written any memory, so the write-path owner link never fires.
    _ensure_agent_owner(request, agent_id, mem.namespace)
    return json.loads(json.dumps(agent.model_dump(mode="json"), default=str))


@app.put("/api/v1/agents/{agent_id:path}/capabilities")
async def update_agent_capabilities(
    agent_id: str,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from amfs_core.models import AgentCapability

    mem = _get_memory()
    capabilities = [AgentCapability.model_validate(c) for c in body.get("capabilities", [])]
    agent = mem._adapter.update_agent_capabilities(agent_id, capabilities)
    return json.loads(json.dumps(agent.model_dump(mode="json"), default=str))


@app.put("/api/v1/agents/{agent_id:path}/contracts")
async def update_agent_contracts(
    agent_id: str,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from amfs_core.models import MemoryContract

    mem = _get_memory()
    contracts = [MemoryContract.model_validate(c) for c in body.get("contracts", [])]
    agent = mem._adapter.update_agent_contracts(agent_id, contracts)
    return json.loads(json.dumps(agent.model_dump(mode="json"), default=str))


@app.get("/api/v1/agents/discover")
async def discover_agents(
    request: Request,
    capability: str | None = Query(None),
    entity_path: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    agents = mem.discover_agents(capability=capability, entity_path=entity_path)

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        agents = [a for a in agents if vis.is_agent_visible(a.agent_id)]

    return {
        "agents": [json.loads(json.dumps(a.model_dump(mode="json"), default=str)) for a in agents],
        "count": len(agents),
    }


# ──────────────────────────────────────────────────────────────────────
# Diff & patch
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/diff")
async def compute_diff(
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    return mem.diff(
        body["entity_path"],
        body["key"],
        body.get("old_version"),
    )


@app.post("/api/v1/patches")
async def create_patch(
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    return mem.create_patch(
        body["entity_path"],
        body["key"],
        body.get("source_version"),
    )


# ──────────────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/history/{entity_path:path}/{key}")
async def get_history(
    entity_path: str,
    key: str,
    since: str | None = Query(None),
    until: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None

    versions = mem.history(entity_path, key, since=since_dt, until=until_dt)
    return {
        "entity_path": entity_path,
        "key": key,
        "version_count": len(versions),
        "versions": [_entry_to_response(e) for e in versions],
    }


# ──────────────────────────────────────────────────────────────────────
# Outcomes
# ──────────────────────────────────────────────────────────────────────

_OUTCOME_TYPE_MAP = {
    "success": OutcomeType.SUCCESS,
    "minor_failure": OutcomeType.MINOR_FAILURE,
    "failure": OutcomeType.FAILURE,
    "critical_failure": OutcomeType.CRITICAL_FAILURE,
    # Legacy values (backward compatible)
    "clean_deploy": OutcomeType.CLEAN_DEPLOY,
    "regression": OutcomeType.REGRESSION,
    "p2_incident": OutcomeType.P2_INCIDENT,
    "p1_incident": OutcomeType.P1_INCIDENT,
}


def _get_immutable_store():
    """Lazily create the immutable trace store (Pro only)."""
    global _immutable_trace_store
    if _immutable_trace_store is not None:
        return _immutable_trace_store
    if not _HAS_PRO_TRACES:
        return None
    dsn = os.environ.get("AMFS_POSTGRES_DSN")
    if not dsn:
        return None
    try:
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        _immutable_trace_store = PostgresImmutableTraceStore(conn)
        logger.info("Immutable trace store initialized")
        return _immutable_trace_store
    except Exception:
        logger.debug("Failed to init immutable trace store", exc_info=True)
        return None


_seal_sequence: dict[str, int] = {}


def _auto_seal_trace(mem: AgentMemory) -> str | None:
    """If Pro traces are available, auto-seal the last OSS trace as immutable."""
    if not _HAS_PRO_TRACES:
        return None
    oss_trace = getattr(mem, "_last_trace", None)
    if oss_trace is None:
        return None
    store = _get_immutable_store()
    if store is None:
        return None
    try:
        from uuid import UUID as _UUID

        now = datetime.now(timezone.utc)
        causal = [
            TraceEntry(
                entity_path=e.entity_path,
                key=e.key,
                version=e.version,
                confidence=e.confidence,
                value=e.value,
                memory_type=getattr(e, "memory_type", None) or "fact",
                written_by=getattr(e, "written_by", None),
                read_at=getattr(e, "read_at", None) or now,
            )
            for e in (oss_trace.causal_entries or [])
        ]
        contexts = [
            TraceExternalContext(
                label=c.label,
                summary=c.summary,
                source=getattr(c, "source", None),
                recorded_at=getattr(c, "recorded_at", None) or now,
            )
            for c in (oss_trace.external_contexts or [])
        ]

        session_id = mem.session_id
        seq = _seal_sequence.get(session_id, 0)
        parent_hash = store.get_latest_hash(session_id)

        trace_id = _UUID(oss_trace.id) if oss_trace.id else None

        account_id = None
        try:
            from amfs_postgres.tenant_context import get_request_tenant_account_id
            tid = get_request_tenant_account_id()
            if tid:
                account_id = _UUID(tid)
        except (ImportError, ValueError):
            pass

        imm = ImmutableDecisionTrace(
            **({"id": trace_id} if trace_id else {}),
            **({"account_id": account_id} if account_id else {}),
            agent_id=mem.agent_id,
            session_id=session_id,
            sequence_number=seq,
            outcome_ref=oss_trace.outcome_ref,
            outcome_type=oss_trace.outcome_type,
            decision_summary=getattr(oss_trace, "decision_summary", None),
            causal_entries=causal,
            external_contexts=contexts,
            created_at=now,
        )
        sealed = seal(
            imm,
            get_signing_key(),
            parent_hash=parent_hash,
            sequence_number=seq,
            signing_key_id=get_signing_key_id(),
        )
        saved = store.save(sealed)
        _seal_sequence[session_id] = seq + 1
        logger.info("Auto-sealed immutable trace %s for outcome %s",
                     saved.id, oss_trace.outcome_ref)
        return str(saved.id)
    except Exception:
        logger.warning("Failed to auto-seal immutable trace", exc_info=True)
        return None


@app.post("/api/v1/outcomes")
async def commit_outcome(
    req: OutcomeRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    otype = _OUTCOME_TYPE_MAP.get(req.outcome_type.lower())
    if otype is None:
        valid = ", ".join(_OUTCOME_TYPE_MAP.keys())
        return {"error": f"Invalid outcome_type '{req.outcome_type}'. Must be one of: {valid}"}

    original_agent = mem._tagger.agent_id if req.agent_id else None
    if req.agent_id:
        mem._tagger.agent_id = req.agent_id
        try:
            mem._adapter.ensure_agent(req.agent_id, mem.namespace)
        except Exception:
            pass
        _ensure_agent_owner(request, req.agent_id, mem.namespace)
    try:
        entries = mem.commit_outcome(
            req.outcome_ref,
            otype,
            causal_entry_keys=req.causal_entry_keys,
            causal_confidence=req.causal_confidence,
        )
    finally:
        if original_agent is not None:
            mem._tagger.agent_id = original_agent
    _audit_log(
        "outcome.commit",
        resource=req.outcome_ref,
        ip_address=request.client.host if request.client else None,
    )

    immutable_trace_id = _auto_seal_trace(mem)

    result: dict[str, Any] = {
        "outcome_ref": req.outcome_ref,
        "outcome_type": req.outcome_type,
        "affected_entries": len(entries),
        "entries": [_entry_to_response(e) for e in entries],
    }
    if immutable_trace_id:
        result["immutable_trace_id"] = immutable_trace_id
    return result


@app.get("/api/v1/outcomes")
async def list_outcomes(
    entity_path: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    since_dt = datetime.fromisoformat(since) if since else None
    records = mem._adapter.list_outcomes(
        entity_path=entity_path,
        since=since_dt,
        limit=limit,
    )
    return {"outcomes": [r.model_dump(mode="json") for r in records]}


# ──────────────────────────────────────────────────────────────────────
# Context & Explain
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/context")
async def record_context(
    req: ContextRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    mem.record_context(req.label, req.summary, source=req.source)
    return {"recorded": req.label, "source": req.source}


@app.get("/api/v1/explain")
async def explain(
    outcome_ref: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    return mem.explain(outcome_ref)


# ──────────────────────────────────────────────────────────────────────
# Decision Traces
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/traces")
async def list_traces(
    entity_path: str | None = Query(None),
    agent_id: str | None = Query(None),
    outcome_type: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    traces = mem._adapter.list_traces(
        entity_path=entity_path,
        agent_id=agent_id,
        outcome_type=outcome_type,
        limit=limit,
    )
    return {"traces": [t.model_dump(mode="json") for t in traces]}


@app.post("/api/v1/traces")
async def save_trace(
    req: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Persist a decision trace (used by HttpAdapter.save_trace)."""
    body = await req.json()
    trace = DecisionTrace.model_validate(body)
    mem = _get_memory()
    saved = mem._adapter.save_trace(trace)
    return saved.model_dump(mode="json")


# NOTE: must be registered before /api/v1/traces/{trace_id} so "share-stats"
# isn't captured as a trace_id.
@app.get("/api/v1/traces/share-stats")
async def get_share_stats(
    request: Request,
    since: datetime | None = Query(None),
    pair_limit: int = Query(20, ge=1, le=100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Cross-agent knowledge-share totals and top reader/author pairs,
    aggregated server-side so full traces never cross the wire."""
    mem = _get_memory()

    vis = _get_visibility_filter(request)
    agent_ids: list[str] | None = None
    if vis is not None and vis.should_filter():
        agent_ids = sorted(vis.get_visible_agent_ids())

    stats = mem._adapter.share_stats(
        since=since, pair_limit=pair_limit, agent_ids=agent_ids
    )
    return json.loads(json.dumps(stats, default=str))


@app.get("/api/v1/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    trace = mem._adapter.get_trace(trace_id)
    if trace is None:
        return JSONResponse({"error": "Trace not found"}, status_code=404)
    return trace.model_dump(mode="json")


@app.post("/api/v1/traces/{trace_id}/explain")
async def explain_trace(
    trace_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    api_key = os.environ.get("AMFS_LLM_API_KEY", "")
    if not api_key:
        return JSONResponse(
            {"error": "LLM not configured. Set AMFS_LLM_API_KEY to enable AI explanations."},
            status_code=503,
        )

    mem = _get_memory()
    trace = mem._adapter.get_trace(trace_id)
    if trace is None:
        return JSONResponse({"error": "Trace not found"}, status_code=404)

    provider = os.environ.get("AMFS_LLM_PROVIDER", "openai")
    model = os.environ.get("AMFS_LLM_MODEL", "gpt-4o-mini")

    td = trace.model_dump(mode="json")

    entries_desc = []
    for e in td.get("causal_entries", []):
        desc = f"- {e['entity_path']}/{e['key']} (v{e['version']}, confidence: {e['confidence']:.0%})"
        if e.get("value"):
            desc += f"\n  Value: {json.dumps(e['value'], default=str)}"
        if e.get("memory_type"):
            desc += f"\n  Type: {e['memory_type']}"
        if e.get("written_by"):
            desc += f"\n  Written by: {e['written_by']}"
        entries_desc.append(desc)

    contexts_desc = []
    for c in td.get("external_contexts", []):
        desc = f"- {c['label']}: {c['summary']}"
        if c.get("source"):
            desc += f" (source: {c['source']})"
        contexts_desc.append(desc)

    queries_desc = []
    for q in td.get("query_events", []):
        desc = f"- {q['operation']}({json.dumps(q.get('parameters', {}))}) → {q.get('result_count', 0)} results"
        if q.get("duration_ms"):
            desc += f" in {q['duration_ms']:.1f}ms"
        queries_desc.append(desc)

    errors_desc = []
    for e in td.get("error_events", []):
        errors_desc.append(f"- [{e['operation']}] {e['error_type']}: {e['message']}")

    diff_desc = ""
    sd = td.get("state_diff")
    if sd:
        diff_desc = f"Entries created: {sd.get('entries_created', 0)}, updated: {sd.get('entries_updated', 0)}"
        for cc in sd.get("confidence_changes", []):
            diff_desc += f"\n  {cc['entity_path']}/{cc['key']}: {cc['before']:.0%} → {cc['after']:.0%}"

    duration_str = ""
    if td.get("session_duration_ms"):
        mins = td["session_duration_ms"] / 60000
        duration_str = f"{mins:.0f} minutes" if mins >= 1 else f"{td['session_duration_ms']:.0f}ms"

    prompt = f"""You are analyzing an AI agent's decision trace from AMFS (Agent Memory File System). Your job is to explain what happened in clear, actionable language that helps a human understand the agent's reasoning and the impact of its decision.

DECISION TRACE DATA:
- Agent: {td.get('agent_id')}
- Outcome Reference: {td.get('outcome_ref', 'None')}
- Outcome Type: {td.get('outcome_type', 'None')}
- Decision Summary: {td.get('decision_summary', 'No summary provided')}
- Session Duration: {duration_str or 'Unknown'}

MEMORY ENTRIES READ (what the agent knew):
{chr(10).join(entries_desc) if entries_desc else 'None'}

EXTERNAL SOURCES CONSULTED:
{chr(10).join(contexts_desc) if contexts_desc else 'None'}

SEARCHES PERFORMED:
{chr(10).join(queries_desc) if queries_desc else 'None'}

ERRORS ENCOUNTERED:
{chr(10).join(errors_desc) if errors_desc else 'None'}

STATE CHANGES:
{diff_desc or 'None'}

Respond with a JSON object containing these fields:
- "narrative": A 2-3 paragraph human-readable story explaining what happened. Start with what the agent was trying to do, then what information it gathered and from where, then what decision it made and why, and finally what the outcome was. Use specific data values from the trace. Write as if explaining to a team lead.
- "key_findings": An array of 3-5 bullet points highlighting the most important facts that influenced the decision. Each should be a complete sentence.
- "risk_assessment": A paragraph assessing risks. For incidents/regressions, explain what went wrong. For clean deploys, explain what risks were mitigated and what could still go wrong.
- "confidence_analysis": A paragraph explaining why the confidence levels are what they are, referencing specific before/after changes if available.
- "recommendations": An array of 2-4 actionable recommendations for what should happen next based on this decision trace.

Return ONLY valid JSON, no markdown formatting."""

    try:
        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                return JSONResponse(
                    {"error": "openai package not installed. Run: pip install openai"},
                    status_code=503,
                )
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            result_text = response.choices[0].message.content or "{}"
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                return JSONResponse(
                    {"error": "anthropic package not installed. Run: pip install anthropic"},
                    status_code=503,
                )
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt + "\n\nRespond with JSON only."}],
            )
            result_text = response.content[0].text
        else:
            return JSONResponse({"error": f"Unsupported LLM provider: {provider}"}, status_code=400)

        explanation = json.loads(result_text)

        for field in ["narrative", "key_findings", "risk_assessment", "confidence_analysis", "recommendations"]:
            if field not in explanation:
                explanation[field] = [] if field in ("key_findings", "recommendations") else ""

        return {"explanation": explanation, "model": model, "provider": provider}

    except json.JSONDecodeError:
        return JSONResponse({"error": "LLM returned invalid JSON", "raw": result_text[:500]}, status_code=502)
    except Exception as exc:
        logger.exception("LLM explain failed")
        return JSONResponse({"error": f"LLM call failed: {exc}"}, status_code=502)


# ──────────────────────────────────────────────────────────────────────
# Admin — Usage
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/usage")
async def get_usage(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    st = mem.stats()
    outcomes = mem._adapter.list_outcomes(limit=10000)

    top_agents = sorted(st.agents.items(), key=lambda x: x[1], reverse=True)[:10]
    top_entities = sorted(st.entities.items(), key=lambda x: x[1], reverse=True)[:10]

    api_key_count = 0
    requests_today = 0
    requests_this_month = 0
    pool = _get_db_pool()
    ns = _get_namespace()

    if pool is not None:
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT count(*) AS cnt FROM amfs_api_keys WHERE namespace = %s AND active = true",
                        (ns,),
                    )
                    row = cur.fetchone()
                    api_key_count = row["cnt"] if row else 0

                    cur.execute(
                        """SELECT count(*) AS cnt FROM amfs_audit_log
                           WHERE namespace = %s
                             AND created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')""",
                        (ns,),
                    )
                    row = cur.fetchone()
                    requests_today = row["cnt"] if row else 0

                    cur.execute(
                        """SELECT count(*) AS cnt FROM amfs_audit_log
                           WHERE namespace = %s
                             AND created_at >= date_trunc('month', now() AT TIME ZONE 'UTC')""",
                        (ns,),
                    )
                    row = cur.fetchone()
                    requests_this_month = row["cnt"] if row else 0
        except Exception:
            pass

    return {
        "requestsToday": requests_today,
        "requestsThisMonth": requests_this_month,
        "peakRpm": 0,
        "avgLatencyMs": 0,
        "quotas": [
            {"label": "Memory entries", "current": st.total_entries, "limit": 0},
            {"label": "Decision traces", "current": len(outcomes), "limit": 0},
            {"label": "API keys", "current": api_key_count, "limit": 0},
            {"label": "Users", "current": st.total_agents, "limit": 0},
        ],
        "topAgents": [
            {"agentId": aid, "requests": count} for aid, count in top_agents
        ],
        "topEntities": [
            {"entityPath": ep, "reads": 0, "writes": count}
            for ep, count in top_entities
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents")
async def list_agents(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all known agents with entry counts and last activity."""
    try:
        from amfs_postgres.tenant_context import get_request_tenant_account_id
        _tls_acct = get_request_tenant_account_id()
    except ImportError:
        _tls_acct = "NO_MODULE"
    _state_acct = getattr(request.state, "account_id", None)
    _state_user = getattr(request.state, "user_id", None)
    _has_ctx = getattr(request.state, "tenant_ctx", None) is not None
    logger.warning(
        "[TLS-DIAG] /agents tls_account=%s state_account=%s state_user=%s has_tenant_ctx=%s",
        _tls_acct, _state_acct, _state_user, _has_ctx,
    )
    mem = _get_memory()
    entries = mem.list()

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        pre_filter = len(entries)
        entries = vis.filter_entries(entries)
        logger.warning(
            "[AGENTS] mem.list=%d after_entry_filter=%d user_agents=%s",
            pre_filter, len(entries), sorted(vis.get_user_agents()),
        )
    else:
        logger.warning(
            "[AGENTS] mem.list=%d NO_FILTER vis=%s",
            len(entries), vis,
        )

    agent_data: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e.entity_path.startswith("_system/"):
            continue
        aid = e.provenance.agent_id
        if aid not in agent_data:
            agent_data[aid] = {
                "agent_id": aid,
                "entries_written": 0,
                "entities_touched": set(),
                "last_active": e.provenance.written_at,
                "first_seen": e.provenance.written_at,
            }
        agent_data[aid]["entries_written"] += 1
        agent_data[aid]["entities_touched"].add(e.entity_path)
        if e.provenance.written_at > agent_data[aid]["last_active"]:
            agent_data[aid]["last_active"] = e.provenance.written_at
        if e.provenance.written_at and (
            agent_data[aid]["first_seen"] is None
            or e.provenance.written_at < agent_data[aid]["first_seen"]
        ):
            agent_data[aid]["first_seen"] = e.provenance.written_at

    if vis is not None and vis.should_filter():
        own = vis.get_user_agents()
        before_own_filter = list(agent_data.keys())
        agent_data = {aid: d for aid, d in agent_data.items() if aid in own}
        # Include owner-linked agents that have written zero entries (e.g. an
        # agent that called set_identity over MCP but hasn't written memory
        # yet). Without this they never appear on the dashboard.
        for aid in own:
            if aid not in agent_data:
                agent_data[aid] = {
                    "agent_id": aid,
                    "entries_written": 0,
                    "entities_touched": set(),
                    "last_active": None,
                    "first_seen": None,
                }
        logger.warning(
            "[AGENTS] own_filter: before=%s after=%s own_set=%s",
            sorted(before_own_filter), sorted(agent_data.keys()), sorted(own),
        )

    agent_registration: dict[str, dict[str, Any]] = {}
    known_agent_ids = list(agent_data.keys())
    if known_agent_ids:
        try:
            from amfs_postgres.adapter import PostgresAdapter
            adapter = mem._adapter
            if isinstance(adapter, PostgresAdapter):
                with adapter._pool.connection() as conn:
                    with conn.cursor() as cur:
                        placeholders = ", ".join(["%s"] * len(known_agent_ids))
                        cur.execute(
                            f"SELECT agent_id, created_at, last_active_at, profile "
                            f"FROM amfs_agents "
                            f"WHERE namespace = %s AND agent_id IN ({placeholders})",
                            [adapter._namespace, *known_agent_ids],
                        )
                        for row in cur.fetchall():
                            agent_registration[row["agent_id"]] = {
                                "created_at": row["created_at"],
                                "last_active_at": row.get("last_active_at"),
                                "profile": row.get("profile"),
                            }
        except (ImportError, Exception):
            pass

    agent_descriptions: dict[str, dict[str, Any]] = {}
    try:
        desc_entries = mem.list("_system/agents")
        for de in desc_entries:
            val = de.value if isinstance(de.value, dict) else {}
            agent_descriptions[de.key] = {
                "description": val.get("description", ""),
                "platform": val.get("platform", ""),
            }
    except Exception:
        pass

    agents = []
    for ad in sorted(agent_data.values(), key=lambda x: x["entries_written"], reverse=True):
        desc_info = agent_descriptions.get(ad["agent_id"], {})
        reg = agent_registration.get(ad["agent_id"], {})
        created = reg.get("created_at") or ad.get("first_seen")
        # Zero-write agents have no entry-derived activity; fall back to the
        # registration row (set via set_identity / profile update).
        last_active = ad["last_active"] or reg.get("last_active_at")
        description = desc_info.get("description", "")
        platform = desc_info.get("platform", "")
        prof = reg.get("profile")
        if isinstance(prof, dict):
            if not description:
                description = prof.get("description", "") or ""
            if not platform:
                sm = prof.get("session_metadata")
                if isinstance(sm, dict):
                    platform = sm.get("platform", "") or ""
        agents.append({
            "agentId": ad["agent_id"],
            "entriesWritten": ad["entries_written"],
            "entitiesTouched": len(ad["entities_touched"]),
            "lastActive": last_active.isoformat() if last_active else None,
            "createdAt": created.isoformat() if created else None,
            "description": description,
            "platform": platform,
        })
    return {"agents": agents}


@app.get("/api/v1/agents/{agent_id:path}/memory-graph")
async def agent_memory_graph(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """All entries written by or read by this agent, grouped by entity."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    entries = mem.list()
    traces = mem._adapter.list_traces(agent_id=agent_id, limit=10000)

    written_by_agent = [
        e for e in entries
        if e.provenance.agent_id == agent_id and not e.entity_path.startswith("_system/")
    ]
    entities_written: dict[str, list[dict]] = {}
    for e in written_by_agent:
        ep = e.entity_path
        if ep not in entities_written:
            entities_written[ep] = []
        entities_written[ep].append({
            "key": e.key,
            "version": e.version,
            "confidence": e.confidence,
            "memoryType": e.memory_type.value if hasattr(e.memory_type, "value") else str(e.memory_type),
            "writtenAt": e.provenance.written_at.isoformat(),
        })

    # Build read counts from both traces (causal_entries) and timeline events.
    read_entities: dict[str, dict[str, int]] = {}
    for t in traces:
        for ce in t.causal_entries:
            ep = ce.entity_path
            key = ce.key
            if ep not in read_entities:
                read_entities[ep] = {}
            read_entities[ep][key] = read_entities[ep].get(key, 0) + 1

    # Supplement with READ events from the timeline (persisted independently).
    total_read_events = 0
    try:
        read_events = mem._adapter.list_events(
            agent_id, mem.namespace, event_type="read", limit=10000,
        )
        cross_read_events = mem._adapter.list_events(
            agent_id, mem.namespace, event_type="cross_agent_read", limit=10000,
        )
        for ev in read_events + cross_read_events:
            ep = ev.details.get("entity_path", "")
            key = ev.details.get("key", "")
            if ep and key:
                if ep not in read_entities:
                    read_entities[ep] = {}
                read_entities[ep][key] = read_entities[ep].get(key, 0) + 1
        total_read_events = len(read_events) + len(cross_read_events)
    except Exception:
        pass

    entry_authors: dict[str, str] = {}
    for e in entries:
        entry_authors[f"{e.entity_path}/{e.key}"] = e.provenance.agent_id

    cross_agent_reads: dict[str, list[dict]] = {}
    for ep, keys in read_entities.items():
        for key, count in keys.items():
            author = entry_authors.get(f"{ep}/{key}")
            if author and author != agent_id:
                if author not in cross_agent_reads:
                    cross_agent_reads[author] = []
                cross_agent_reads[author].append({
                    "entityPath": ep,
                    "key": key,
                    "readCount": count,
                })

    nodes = []
    for ep in sorted(set(list(entities_written.keys()) + list(read_entities.keys()))):
        nodes.append({
            "entityPath": ep,
            "writtenEntries": entities_written.get(ep, []),
            "readCounts": read_entities.get(ep, {}),
        })

    return {
        "agentId": agent_id,
        "nodes": nodes,
        "traceCount": len(traces),
        "totalWritten": len(written_by_agent),
        "totalReads": total_read_events,
        "crossAgentReads": cross_agent_reads,
    }


@app.get("/api/v1/agents/{agent_id:path}/cross-agent-reads")
async def agent_cross_reads(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Which other agents' memory this agent has read."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    entries = mem.list()
    traces = mem._adapter.list_traces(agent_id=agent_id, limit=10000)

    entry_authors: dict[str, str] = {}
    for e in entries:
        entry_authors[f"{e.entity_path}/{e.key}"] = e.provenance.agent_id

    read_entities: dict[str, dict[str, int]] = {}
    for t in traces:
        for ce in t.causal_entries:
            ep = ce.entity_path
            key = ce.key
            if ep not in read_entities:
                read_entities[ep] = {}
            read_entities[ep][key] = read_entities[ep].get(key, 0) + 1

    cross_reads: dict[str, list[dict[str, Any]]] = {}
    for ep, keys in read_entities.items():
        for key, count in keys.items():
            author = entry_authors.get(f"{ep}/{key}")
            if author and author != agent_id:
                if author not in cross_reads:
                    cross_reads[author] = []
                cross_reads[author].append({
                    "entityPath": ep,
                    "key": key,
                    "readCount": count,
                })

    return {
        "agentId": agent_id,
        "readsFrom": cross_reads,
        "agentsReadFrom": list(cross_reads.keys()),
        "totalCrossAgentReads": sum(
            r["readCount"] for reads in cross_reads.values() for r in reads
        ),
    }


@app.get("/api/v1/agents/{agent_id:path}/recall/{entity_path:path}/{key}")
async def agent_recall(
    request: Request,
    agent_id: str,
    entity_path: str,
    key: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Recall an agent's own memory for a key (agent-scoped read).

    Unlike the generic read endpoint, this returns only entries written
    by the specified agent — what that agent's brain actually knows.
    """
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    original_agent = mem._tagger.agent_id
    mem._tagger.agent_id = agent_id
    try:
        entry = mem.recall(entity_path, key)
    finally:
        mem._tagger.agent_id = original_agent
    if entry is None:
        return {"status": "not_found", "agentId": agent_id,
                "entityPath": entity_path, "key": key}
    return _entry_to_response(entry)


@app.get("/api/v1/agents/{agent_id:path}/entries")
async def agent_entries(
    request: Request,
    agent_id: str,
    entity_path: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all entries written by a specific agent."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    entries = mem.search(entity_path=entity_path, agent_id=agent_id)
    return {
        "agentId": agent_id,
        "count": len(entries),
        "entries": [_entry_to_response(e) for e in entries],
    }


@app.get("/api/v1/agents/{agent_id:path}/read-from/{source_agent_id}/{entity_path:path}/{key}")
async def agent_read_from(
    request: Request,
    agent_id: str,
    source_agent_id: str,
    entity_path: str,
    key: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Read a specific key from another agent's memory (cross-agent read)."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(source_agent_id):
        return {"status": "not_found", "sourceAgentId": source_agent_id,
                "entityPath": entity_path, "key": key}

    mem = _get_memory()
    entries = mem.search(entity_path=entity_path, agent_id=source_agent_id)
    matching = [e for e in entries if e.key == key]
    if not matching:
        return {"status": "not_found", "sourceAgentId": source_agent_id,
                "entityPath": entity_path, "key": key}
    return _entry_to_response(matching[0])


@app.get("/api/v1/agents/{agent_id:path}/activity")
async def agent_activity(
    request: Request,
    agent_id: str,
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Timeline of writes, outcomes, reads, and other events for this agent."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    entries = [
        e for e in mem.list()
        if e.provenance.agent_id == agent_id and not e.entity_path.startswith("_system/")
    ]
    entries.sort(key=lambda e: e.provenance.written_at, reverse=True)

    writes = [
        {
            "type": "write",
            "entityPath": e.entity_path,
            "key": e.key,
            "version": e.version,
            "confidence": e.confidence,
            "timestamp": e.provenance.written_at.isoformat(),
        }
        for e in entries[:limit]
    ]

    traces = mem._adapter.list_traces(agent_id=agent_id, limit=limit)
    outcomes = [
        {
            "type": "outcome",
            "outcomeRef": t.outcome_ref,
            "outcomeType": t.outcome_type,
            "causalEntryCount": len(t.causal_entries),
            "timestamp": t.created_at.isoformat(),
        }
        for t in traces if t.outcome_ref
    ]

    events = mem._adapter.list_events(agent_id, mem.namespace, limit=limit)
    event_items = [
        {
            "type": evt.event_type.value if hasattr(evt.event_type, "value") else str(evt.event_type),
            "summary": evt.summary,
            "details": evt.details,
            "actorAgentId": evt.actor_agent_id,
            "timestamp": evt.created_at.isoformat(),
        }
        for evt in events
    ]

    timeline = sorted(
        writes + outcomes + event_items,
        key=lambda x: x["timestamp"],
        reverse=True,
    )[:limit]
    return {"agentId": agent_id, "timeline": timeline}


@app.get("/api/v1/agents/{agent_id:path}/timeline")
async def agent_timeline(
    request: Request,
    agent_id: str,
    event_type: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Git-like event log for an agent — every write, outcome, and
    cross-agent read is recorded as an event on the agent's timeline."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    since_dt = datetime.fromisoformat(since) if since else None
    events = mem._adapter.list_events(
        agent_id, mem.namespace,
        event_type=event_type,
        since=since_dt, limit=limit,
    )
    return {
        "agentId": agent_id,
        "events": [e.model_dump(mode="json") for e in events],
        "count": len(events),
    }


class LogEventRequest(BaseModel):
    agent_id: str
    event_type: str
    summary: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    actor_agent_id: str | None = None
    branch: str = "main"


@app.post("/api/v1/timeline/events")
async def log_timeline_event(
    body: LogEventRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Log a timeline event for an agent."""
    mem = _get_memory()
    try:
        event_type_enum = EventType(body.event_type)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown event_type: {body.event_type}"},
        )
    event = Event(
        namespace=mem.namespace,
        agent_id=body.agent_id,
        branch=body.branch,
        event_type=event_type_enum,
        summary=body.summary,
        details=body.details,
        actor_agent_id=body.actor_agent_id,
    )
    saved = mem._adapter.log_event(event)
    return saved.model_dump(mode="json")


class UpsertGraphEdgeRequest(BaseModel):
    source_entity: str
    source_type: str = "agent"
    relation: str
    target_entity: str
    target_type: str = "agent"
    provenance: dict[str, Any] = Field(default_factory=dict)
    branch: str = "main"


@app.post("/api/v1/graph/edges")
async def upsert_graph_edge_endpoint(
    body: UpsertGraphEdgeRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Create or update a knowledge graph edge."""
    mem = _get_memory()
    edge = GraphEdge(
        source_entity=body.source_entity,
        source_type=body.source_type,
        relation=body.relation,
        target_entity=body.target_entity,
        target_type=body.target_type,
        provenance=body.provenance,
    )
    mem._adapter.upsert_graph_edge(edge, namespace=mem.namespace, branch=body.branch)
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────
# Agent Groups & Enriched Agents
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents/enriched")
async def list_agents_enriched(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return enriched agent info with activity histograms."""
    mem = _get_memory()
    agents = mem._adapter.list_agents_enriched(namespace=mem.namespace)
    for agent in agents[:100]:
        aid = agent.get("agent_id") or agent.get("agentId", "")
        if aid:
            agent["activity_histogram"] = mem._adapter.get_agent_activity_histogram(
                aid, days=7, namespace=mem.namespace,
            )
    return {"agents": agents}


@app.get("/api/v1/agent-groups")
async def list_agent_groups(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all agent groups."""
    mem = _get_memory()
    groups = mem._adapter.list_agent_groups(namespace=mem.namespace)
    return {"groups": [g.model_dump(mode="json") for g in groups]}


@app.post("/api/v1/agent-groups", status_code=201)
async def create_agent_group_endpoint(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Create a new agent group."""
    body = await request.json()
    name = body.get("name")
    if not name:
        return JSONResponse(
            status_code=400,
            content={"error": "name is required"},
        )
    mem = _get_memory()
    group = AgentGroup(
        namespace=mem.namespace,
        name=name,
        description=body.get("description", ""),
        color=body.get("color"),
        icon=body.get("icon"),
        position=body.get("position", 0.0),
        auto_generated=body.get("autoGenerated", False),
        source_cluster_id=body.get("sourceClusterId"),
    )
    created = mem._adapter.create_agent_group(group, namespace=mem.namespace)
    return created.model_dump(mode="json")


@app.put("/api/v1/agent-groups/{group_id}")
async def update_agent_group_endpoint(
    group_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Update an existing agent group."""
    body = await request.json()
    mem = _get_memory()
    kwargs: dict[str, Any] = {}
    for field in ("name", "description", "color", "icon", "position"):
        if field in body and body[field] is not None:
            kwargs[field] = body[field]
    updated = mem._adapter.update_agent_group(
        group_id, namespace=mem.namespace, **kwargs,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Group not found")
    return updated.model_dump(mode="json")


@app.delete("/api/v1/agent-groups/{group_id}")
async def delete_agent_group_endpoint(
    group_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete an agent group."""
    mem = _get_memory()
    deleted = mem._adapter.delete_agent_group(group_id, namespace=mem.namespace)
    return {"deleted": deleted}


@app.post("/api/v1/agent-groups/{group_id}/members")
async def add_group_members(
    group_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Add agents to a group."""
    body = await request.json()
    agent_ids = body.get("agent_ids")
    if not agent_ids or not isinstance(agent_ids, list):
        return JSONResponse(
            status_code=400,
            content={"error": "agent_ids list is required"},
        )
    mem = _get_memory()
    count = mem._adapter.add_agents_to_group(
        group_id, agent_ids, namespace=mem.namespace,
    )
    return {"added": count}


@app.delete("/api/v1/agent-groups/{group_id}/members")
async def remove_group_members(
    group_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Remove agents from a group."""
    body = await request.json()
    agent_ids = body.get("agent_ids")
    if not agent_ids or not isinstance(agent_ids, list):
        return JSONResponse(
            status_code=400,
            content={"error": "agent_ids list is required"},
        )
    mem = _get_memory()
    count = mem._adapter.remove_agents_from_group(
        group_id, agent_ids, namespace=mem.namespace,
    )
    return {"removed": count}


@app.put("/api/v1/agent-groups/reorder")
async def reorder_agent_groups_endpoint(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Reorder agent groups by setting new positions."""
    body = await request.json()
    positions = body.get("positions")
    if not positions or not isinstance(positions, list):
        return JSONResponse(
            status_code=400,
            content={"error": "positions list is required"},
        )
    tuples = [(p["group_id"], p["position"]) for p in positions]
    mem = _get_memory()
    mem._adapter.reorder_agent_groups(tuples, namespace=mem.namespace)
    return {"ok": True}


@app.get("/api/v1/agent-groups/suggestions")
async def agent_group_suggestions(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return cluster-based group suggestions, excluding dismissed ones."""
    from amfs_core.models import DigestType

    mem = _get_memory()
    adapter = mem._adapter
    digest = adapter.get_digest(
        DigestType.AGENT_CLUSTERS,
        f"account:{mem.namespace}",
        namespace=mem.namespace,
    )
    if not digest:
        return {"suggestions": []}

    try:
        dismissed = set(adapter.list_dismissed_cluster_ids(mem.namespace))
    except Exception:
        dismissed = set()

    clusters = digest.summary.get("clusters", [])
    suggestions = [c for c in clusters if c.get("cluster_id") not in dismissed]
    return {"suggestions": suggestions}


@app.post("/api/v1/agent-groups/suggestions/{cluster_id}/accept", status_code=201)
async def accept_cluster_suggestion(
    cluster_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Accept a cluster suggestion — create a group from it."""
    from amfs_core.models import DigestType

    mem = _get_memory()
    adapter = mem._adapter
    digest = adapter.get_digest(
        DigestType.AGENT_CLUSTERS,
        f"account:{mem.namespace}",
        namespace=mem.namespace,
    )
    if not digest:
        raise HTTPException(status_code=404, detail="No cluster digest found")

    clusters = digest.summary.get("clusters", [])
    cluster = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    group = AgentGroup(
        namespace=mem.namespace,
        name=cluster.get("suggested_name", cluster_id),
        description=cluster.get("rationale", ""),
        auto_generated=True,
        source_cluster_id=cluster_id,
    )
    created = adapter.create_agent_group(group, namespace=mem.namespace)

    agent_ids = cluster.get("agents", [])
    if agent_ids:
        adapter.add_agents_to_group(created.id, agent_ids, namespace=mem.namespace)

    return created.model_dump(mode="json")


@app.post("/api/v1/agent-groups/suggestions/{cluster_id}/dismiss")
async def dismiss_cluster_suggestion_endpoint(
    cluster_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Dismiss a cluster suggestion so it no longer appears."""
    mem = _get_memory()
    mem._adapter.dismiss_cluster_suggestion(cluster_id, mem.namespace)
    return {"dismissed": True}


@app.post("/api/v1/agent-groups/recompute")
async def recompute_clusters(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Trigger recomputation of agent clusters."""
    try:
        from amfs_cortex.compiler import DigestCompiler
        from amfs_postgres.adapter import PostgresAdapter

        mem = _get_memory()
        adapter = mem._adapter
        if not isinstance(adapter, PostgresAdapter):
            return JSONResponse(
                status_code=400,
                content={"error": "Cluster recomputation requires Postgres adapter"},
            )
        compiler = DigestCompiler(
            adapter=adapter,
            namespace=mem.namespace,
        )
        compiler.compile(f"cluster:account:{mem.namespace}")
    except ImportError:
        return JSONResponse(
            status_code=400,
            content={"error": "amfs-cortex package is required for recomputation"},
        )
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────
# Agent Snapshots
# ──────────────────────────────────────────────────────────────────────


SNAPSHOT_ENTITY = "_system/agent-snapshots"


@app.post("/api/v1/agents/{agent_id:path}/snapshots")
async def create_snapshot(
    agent_id: str,
    req: CreateSnapshotRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Create a named snapshot of an agent's brain state."""
    mem = _get_memory()
    snapshot_id = f"snap-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    snapshot_value = {
        "id": snapshot_id,
        "agent_id": agent_id,
        "name": req.name,
        "description": req.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": req.snapshot_data.get("stats", {}),
        "data": req.snapshot_data,
    }

    original_agent = mem._tagger.agent_id
    mem._tagger.agent_id = agent_id
    try:
        mem.write(
            SNAPSHOT_ENTITY,
            f"{agent_id}/{snapshot_id}",
            snapshot_value,
            confidence=1.0,
        )
    finally:
        mem._tagger.agent_id = original_agent

    try:
        mem._adapter.log_event(Event(
            namespace=mem.namespace,
            agent_id=agent_id,
            branch="main",
            event_type=EventType.SNAPSHOT_TAKEN,
            summary=f"Snapshot '{req.name}' taken",
            details={
                "snapshot_id": snapshot_id,
                "name": req.name,
                "description": req.description,
                **snapshot_value.get("stats", {}),
            },
        ))
    except Exception:
        logger.debug("Failed to log snapshot event", exc_info=True)

    return {
        "id": snapshot_id,
        "name": req.name,
        "agent_id": agent_id,
        "created_at": snapshot_value["created_at"],
    }


@app.get("/api/v1/agents/{agent_id:path}/snapshots")
async def list_snapshots(
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all snapshots for an agent."""
    mem = _get_memory()
    entries = mem.list(entity_path=SNAPSHOT_ENTITY)
    prefix = f"{agent_id}/"
    snapshots = []
    for e in entries:
        if not e.key.startswith(prefix):
            continue
        val = e.value if isinstance(e.value, dict) else {}
        snapshots.append({
            "id": val.get("id", e.key.split("/")[-1]),
            "name": val.get("name", e.key),
            "description": val.get("description", ""),
            "agent_id": agent_id,
            "created_at": val.get("created_at", e.provenance.written_at.isoformat()),
            "stats": val.get("stats", {}),
        })
    snapshots.sort(key=lambda s: s["created_at"], reverse=True)
    return {"snapshots": snapshots, "count": len(snapshots)}


@app.get("/api/v1/agents/{agent_id:path}/snapshots/{snapshot_id}")
async def get_snapshot(
    agent_id: str,
    snapshot_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get the full data for a specific snapshot."""
    mem = _get_memory()
    key = f"{agent_id}/{snapshot_id}"
    entry = mem.recall(SNAPSHOT_ENTITY, key)
    if entry is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    val = entry.value if isinstance(entry.value, dict) else {}
    return val


@app.delete("/api/v1/agents/{agent_id:path}/snapshots/{snapshot_id}")
async def delete_snapshot(
    agent_id: str,
    snapshot_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete a snapshot."""
    mem = _get_memory()
    key = f"{agent_id}/{snapshot_id}"
    entry = mem.recall(SNAPSHOT_ENTITY, key)
    if entry is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    mem.write(SNAPSHOT_ENTITY, key, None, confidence=0.0)
    return {"deleted": True, "id": snapshot_id}


@app.post("/api/v1/agents/{agent_id:path}/snapshots/{snapshot_id}/recover")
async def recover_snapshot(
    agent_id: str,
    snapshot_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Recover an agent's memory to the state captured in a snapshot."""
    mem = _get_memory()
    key = f"{agent_id}/{snapshot_id}"
    entry = mem.recall(SNAPSHOT_ENTITY, key)
    if entry is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    val = entry.value if isinstance(entry.value, dict) else {}
    created_at = val.get("created_at")
    if not created_at:
        raise HTTPException(status_code=400, detail="Snapshot missing timestamp")

    timestamp = datetime.fromisoformat(created_at)
    count = mem._adapter.rollback_to_timestamp(
        agent_id, "main", timestamp, mem.namespace,
    )

    try:
        mem._adapter.log_event(Event(
            namespace=mem.namespace,
            agent_id=agent_id,
            branch="main",
            event_type=EventType.SNAPSHOT_RECOVERED,
            summary=f"Recovered from snapshot '{val.get('name', snapshot_id)}'",
            details={
                "snapshot_id": snapshot_id,
                "name": val.get("name", ""),
                "recovered_to": created_at,
                "entries_restored": count,
            },
        ))
    except Exception:
        logger.debug("Failed to log recovery event", exc_info=True)

    return {
        "recovered": True,
        "snapshot_id": snapshot_id,
        "entries_restored": count,
        "recovered_to": created_at,
    }


# ──────────────────────────────────────────────────────────────────────
# Rollback
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/rollback")
async def rollback(
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Rollback an agent's memory to a specific event or timestamp.

    Accepts either ``target_event_id`` (looks up the event to get its
    timestamp and agent) or ``target_timestamp`` + ``agent_id``.
    """
    mem = _get_memory()
    target_event_id = body.get("target_event_id")
    target_timestamp = body.get("target_timestamp")
    agent_id = body.get("agent_id")

    if target_event_id:
        event = mem._adapter.get_event(target_event_id, mem.namespace)
        if event is None and agent_id:
            for e in mem._adapter.list_events(agent_id, mem.namespace, limit=10000):
                if e.id == target_event_id:
                    event = e
                    break
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        timestamp = event.created_at
        agent_id = agent_id or event.agent_id
    elif target_timestamp:
        if not agent_id:
            raise HTTPException(
                status_code=400,
                detail="agent_id is required when using target_timestamp",
            )
        timestamp = datetime.fromisoformat(target_timestamp)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide target_event_id or target_timestamp",
        )

    count = mem._adapter.rollback_to_timestamp(
        agent_id, "main", timestamp, mem.namespace,
    )

    try:
        mem._adapter.log_event(Event(
            namespace=mem.namespace,
            agent_id=agent_id,
            branch="main",
            event_type=EventType.ROLLBACK,
            summary=f"Rolled back to {timestamp.isoformat()}",
            details={
                "rolled_back_to": timestamp.isoformat(),
                "entries_restored": count,
                "source_event_id": target_event_id,
            },
        ))
    except Exception:
        logger.debug("Failed to log rollback event", exc_info=True)

    return {
        "entries_restored": count,
        "rolled_back_to": timestamp.isoformat(),
        "agent_id": agent_id,
    }


# ──────────────────────────────────────────────────────────────────────
# Agent-scoped branches & pull requests
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents/{agent_id:path}/branches")
async def agent_branches(
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Branches created or merged by this agent."""
    mem = _get_memory()
    try:
        all_branches = mem.list_branches()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        return {"agentId": agent_id, "branches": [], "count": 0}
    scoped = [
        b for b in all_branches
        if getattr(b, "created_by", None) == agent_id
        or getattr(b, "merged_by", None) == agent_id
    ]
    return {
        "agentId": agent_id,
        "branches": [b.model_dump(mode="json") if hasattr(b, "model_dump") else b for b in scoped],
        "count": len(scoped),
    }


@app.get("/api/v1/agents/{agent_id:path}/pull-requests")
async def agent_pull_requests(
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Pull requests created, merged, or closed by this agent."""
    mem = _get_memory()
    try:
        all_prs = mem.list_pull_requests()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        return {"agentId": agent_id, "pullRequests": [], "count": 0}
    scoped = [
        pr for pr in all_prs
        if getattr(pr, "created_by", None) == agent_id
        or getattr(pr, "merged_by", None) == agent_id
        or getattr(pr, "closed_by", None) == agent_id
    ]
    return {
        "agentId": agent_id,
        "pullRequests": [pr.model_dump(mode="json") if hasattr(pr, "model_dump") else pr for pr in scoped],
        "count": len(scoped),
    }


# ──────────────────────────────────────────────────────────────────────
# Pro Branching Plugin (amfs_branching — proprietary)
# ──────────────────────────────────────────────────────────────────────

try:
    from amfs_branching import mount_branching_routes  # type: ignore[import-not-found]
    mount_branching_routes(app, get_memory=_get_memory)
    logger.info("Pro branching routes mounted")
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────
# Control Plane Plugin (amfs_control_plane — proprietary SaaS billing)
# ──────────────────────────────────────────────────────────────────────

try:
    from amfs_control_plane import mount_control_plane  # type: ignore[import-not-found]
    mount_control_plane(app)
    logger.info("Control-plane routes mounted (auth, billing, Stripe webhook)")
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────
# Admin — API Keys
# ──────────────────────────────────────────────────────────────────────


def _get_db_pool():
    """Return the underlying database pool if using the Postgres adapter."""
    mem = _get_memory()
    adapter = mem._adapter
    if hasattr(adapter, "_pool"):
        return adapter._pool
    return None


def _get_namespace() -> str:
    """Return the namespace for the current adapter.

    All admin queries MUST use this to scope data to the correct tenant.
    """
    mem = _get_memory()
    adapter = mem._adapter
    if hasattr(adapter, "_namespace"):
        return adapter._namespace
    return "default"


def _audit_log(
    action: str,
    *,
    resource: str | None = None,
    actor_type: str = "api_key",
    actor_name: str = "api",
    ip_address: str | None = None,
) -> None:
    """Write an entry to the audit log. Silently no-ops without Postgres."""
    pool = _get_db_pool()
    if pool is None:
        return
    ns = _get_namespace()
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO amfs_audit_log
                           (namespace, actor_type, actor_name, action, resource, ip_address)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (ns, actor_type, actor_name, action, resource, ip_address),
                )
    except Exception:
        logger.debug("Failed to write audit log", exc_info=True)


@app.get("/api/v1/admin/api-keys")
async def list_api_keys(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"keys": []}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, name, prefix, key_type, active, scopes,
                          rate_limit_rpm, last_used, created_at, expires_at
                   FROM amfs_api_keys
                   WHERE namespace = %s
                   ORDER BY created_at DESC""",
                (ns,),
            )
            rows = cur.fetchall()
    keys = []
    for row in rows:
        scopes = row["scopes"] or []
        if isinstance(scopes, str):
            scopes = json.loads(scopes)
        keys.append({
            "id": str(row["id"]),
            "name": row["name"],
            "prefix": row["prefix"],
            "keyType": row["key_type"],
            "active": row["active"],
            "scopes": scopes,
            "rateLimitRpm": row["rate_limit_rpm"],
            "lastUsed": row["last_used"].isoformat() if row["last_used"] else None,
            "createdAt": row["created_at"].isoformat(),
            "expiresAt": row["expires_at"].isoformat() if row["expires_at"] else None,
        })
    return {"keys": keys}


@app.post("/api/v1/admin/api-keys")
async def create_api_key(
    req: CreateAPIKeyRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "API key management requires a Postgres backend"}

    raw_key = f"amfs_{secrets.token_urlsafe(32)}"
    prefix = raw_key[:12]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    ns = _get_namespace()

    user_id = getattr(request.state, "user_id", None)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO amfs_api_keys
                       (namespace, name, key_hash, prefix, key_type, scopes, rate_limit_rpm, expires_at, created_by)
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                   RETURNING id, created_at""",
                (
                    ns,
                    req.name,
                    key_hash,
                    prefix,
                    req.key_type,
                    json.dumps(req.scopes),
                    req.rate_limit_rpm,
                    req.expires_at,
                    str(user_id) if user_id else None,
                ),
            )
            row = cur.fetchone()

    _audit_log(
        "api_key.create",
        resource=req.name,
        ip_address=request.client.host if request.client else None,
    )

    return {
        "id": str(row["id"]),
        "name": req.name,
        "key": raw_key,
        "prefix": prefix,
        "keyType": req.key_type,
        "createdAt": row["created_at"].isoformat(),
    }


@app.delete("/api/v1/admin/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "API key management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE amfs_api_keys SET active = FALSE
                   WHERE id = %s::uuid AND namespace = %s
                   RETURNING id, name""",
                (key_id, ns),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Key not found"}
    _audit_log(
        "api_key.revoke",
        resource=row.get("name", key_id),
        ip_address=request.client.host if request.client else None,
    )
    return {"revoked": str(row["id"])}


# ──────────────────────────────────────────────────────────────────────
# Admin — Audit Log
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/audit")
async def list_audit_log(
    action: str | None = Query(None),
    limit: int = Query(200),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"entries": []}

    ns = _get_namespace()
    conditions = ["namespace = %s"]
    params: list[Any] = [ns]

    if action is not None and action != "all":
        conditions.append("action = %s")
        params.append(action)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, actor_type, actor_name, action, resource,
               ip_address, created_at
        FROM amfs_audit_log
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    entries = []
    for row in rows:
        entries.append({
            "id": str(row["id"]),
            "actorType": row["actor_type"],
            "actorName": row["actor_name"],
            "action": row["action"],
            "resource": row["resource"],
            "ipAddress": row["ip_address"],
            "createdAt": row["created_at"].isoformat(),
        })
    return {"entries": entries}


# ──────────────────────────────────────────────────────────────────────
# Patterns — OSS: list pattern_refs used across entries
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/patterns")
async def list_patterns(
    entity_path: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List unique pattern_refs used across memory entries with usage counts."""
    mem = _get_memory()
    entries = mem.list(entity_path)

    pattern_counts: dict[str, int] = {}
    pattern_entities: dict[str, set[str]] = {}
    for entry in entries:
        for ref in entry.provenance.pattern_refs:
            pattern_counts[ref] = pattern_counts.get(ref, 0) + 1
            if ref not in pattern_entities:
                pattern_entities[ref] = set()
            pattern_entities[ref].add(entry.entity_path)

    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {
        "patterns": [
            {
                "pattern_ref": ref,
                "usage_count": count,
                "entity_paths": sorted(pattern_entities[ref]),
            }
            for ref, count in sorted_patterns
        ],
        "total": len(pattern_counts),
    }


# ──────────────────────────────────────────────────────────────────────
# Admin — Teams (Pro)
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/teams")
async def list_teams(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"teams": []}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.id, t.name, t.slug, t.description,
                          t.created_at, t.updated_at,
                          COUNT(m.id) AS member_count
                   FROM amfs_teams t
                   LEFT JOIN amfs_team_members m
                       ON m.team_id = t.id AND m.removed_at IS NULL
                   WHERE t.namespace = %s
                   GROUP BY t.id
                   ORDER BY t.created_at DESC""",
                (ns,),
            )
            rows = cur.fetchall()
    return {
        "teams": [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "slug": row["slug"],
                "description": row["description"],
                "memberCount": row["member_count"],
                "createdAt": row["created_at"].isoformat(),
                "updatedAt": row["updated_at"].isoformat(),
            }
            for row in rows
        ]
    }


@app.post("/api/v1/admin/teams")
async def create_team(
    req: CreateTeamRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO amfs_teams (namespace, name, slug, description)
                   VALUES (%s, %s, %s, %s)
                   RETURNING id, created_at, updated_at""",
                (ns, req.name, req.slug, req.description),
            )
            row = cur.fetchone()
    _audit_log(
        "team.create",
        resource=req.slug,
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "name": req.name,
        "slug": req.slug,
        "description": req.description,
        "createdAt": row["created_at"].isoformat(),
        "updatedAt": row["updated_at"].isoformat(),
    }


@app.patch("/api/v1/admin/teams/{team_id}")
async def update_team(
    team_id: str,
    req: UpdateTeamRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}

    updates: list[str] = []
    params: list[Any] = []
    if req.name is not None:
        updates.append("name = %s")
        params.append(req.name)
    if req.description is not None:
        updates.append("description = %s")
        params.append(req.description)

    if not updates:
        return {"error": "No fields to update"}

    updates.append("updated_at = NOW()")
    ns = _get_namespace()
    params.extend([team_id, ns])

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE amfs_teams SET {', '.join(updates)}
                    WHERE id = %s::uuid AND namespace = %s
                    RETURNING id, name, slug, description, created_at, updated_at""",
                params,
            )
            row = cur.fetchone()

    if row is None:
        return {"error": "Team not found"}
    _audit_log(
        "team.update",
        resource=str(row["slug"]),
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "slug": row["slug"],
        "description": row["description"],
        "createdAt": row["created_at"].isoformat(),
        "updatedAt": row["updated_at"].isoformat(),
    }


@app.delete("/api/v1/admin/teams/{team_id}")
async def delete_team(
    team_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM amfs_teams
                   WHERE id = %s::uuid AND namespace = %s
                   RETURNING id, slug""",
                (team_id, ns),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Team not found"}
    _audit_log(
        "team.delete",
        resource=row.get("slug", team_id),
        ip_address=request.client.host if request.client else None,
    )
    return {"deleted": str(row["id"])}


# ──────────────────────────────────────────────────────────────────────
# Admin — Team Members (Pro)
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/teams/{team_id}/members")
async def list_team_members(
    team_id: str,
    include_removed: bool = Query(False),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"members": []}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if include_removed:
                cur.execute(
                    """SELECT id, email, display_name, role,
                              invited_at, accepted_at, created_at,
                              removed_at, removed_by
                       FROM amfs_team_members
                       WHERE team_id = %s::uuid AND namespace = %s
                       ORDER BY created_at""",
                    (team_id, ns),
                )
            else:
                cur.execute(
                    """SELECT id, email, display_name, role,
                              invited_at, accepted_at, created_at,
                              removed_at, removed_by
                       FROM amfs_team_members
                       WHERE team_id = %s::uuid AND namespace = %s
                         AND removed_at IS NULL
                       ORDER BY created_at""",
                    (team_id, ns),
                )
            rows = cur.fetchall()
    return {
        "members": [
            {
                "id": str(row["id"]),
                "email": row["email"],
                "displayName": row["display_name"],
                "role": row["role"],
                "invitedAt": row["invited_at"].isoformat(),
                "acceptedAt": row["accepted_at"].isoformat() if row["accepted_at"] else None,
                "createdAt": row["created_at"].isoformat(),
                "removedAt": row["removed_at"].isoformat() if row.get("removed_at") else None,
                "removedBy": row.get("removed_by"),
            }
            for row in rows
        ]
    }


@app.post("/api/v1/admin/teams/{team_id}/members")
async def add_team_member(
    team_id: str,
    req: AddTeamMemberRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # If this email was previously removed from this team, reinstate instead of duplicating
            cur.execute(
                """SELECT id FROM amfs_team_members
                   WHERE team_id = %s::uuid AND email = %s AND namespace = %s
                     AND removed_at IS NOT NULL""",
                (team_id, req.email, ns),
            )
            existing_removed = cur.fetchone()
            if existing_removed:
                cur.execute(
                    """UPDATE amfs_team_members
                       SET removed_at = NULL, removed_by = NULL,
                           role = %s, display_name = %s
                       WHERE id = %s::uuid
                       RETURNING id, invited_at, created_at""",
                    (req.role, req.display_name, existing_removed["id"]),
                )
                row = cur.fetchone()
            else:
                cur.execute(
                    """INSERT INTO amfs_team_members
                       (namespace, team_id, email, display_name, role)
                       VALUES (%s, %s::uuid, %s, %s, %s)
                       RETURNING id, invited_at, created_at""",
                    (ns, team_id, req.email, req.display_name, req.role),
                )
                row = cur.fetchone()
    _audit_log(
        "team.member.add",
        resource=f"{team_id}/{req.email}",
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "teamId": team_id,
        "email": req.email,
        "displayName": req.display_name,
        "role": req.role,
        "invitedAt": row["invited_at"].isoformat(),
        "createdAt": row["created_at"].isoformat(),
    }


@app.patch("/api/v1/admin/teams/{team_id}/members/{member_id}")
async def update_team_member(
    team_id: str,
    member_id: str,
    req: UpdateTeamMemberRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}

    updates: list[str] = []
    params: list[Any] = []
    if req.role is not None:
        updates.append("role = %s")
        params.append(req.role)
    if req.display_name is not None:
        updates.append("display_name = %s")
        params.append(req.display_name)

    if not updates:
        return {"error": "No fields to update"}

    ns = _get_namespace()
    params.extend([member_id, team_id, ns])

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE amfs_team_members SET {', '.join(updates)}
                    WHERE id = %s::uuid AND team_id = %s::uuid AND namespace = %s
                    RETURNING id, email, display_name, role, invited_at, accepted_at, created_at""",
                params,
            )
            row = cur.fetchone()

    if row is None:
        return {"error": "Member not found"}
    _audit_log(
        "team.member.update",
        resource=f"{team_id}/{row['email']}",
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "email": row["email"],
        "displayName": row["display_name"],
        "role": row["role"],
        "invitedAt": row["invited_at"].isoformat(),
        "acceptedAt": row["accepted_at"].isoformat() if row["accepted_at"] else None,
        "createdAt": row["created_at"].isoformat(),
    }


@app.delete("/api/v1/admin/teams/{team_id}/members/{member_id}")
async def remove_team_member(
    team_id: str,
    member_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    removed_by = request.headers.get("X-AMFS-Dashboard-Actor", "api")
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE amfs_team_members
                   SET removed_at = NOW(), removed_by = %s
                   WHERE id = %s::uuid AND team_id = %s::uuid
                     AND namespace = %s AND removed_at IS NULL
                   RETURNING id, email""",
                (removed_by, member_id, team_id, ns),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Member not found"}
    _audit_log(
        "team.member.remove",
        resource=f"{team_id}/{row['email']}",
        ip_address=request.client.host if request.client else None,
    )
    return {"deleted": str(row["id"])}


@app.post("/api/v1/admin/teams/{team_id}/members/{member_id}/reinstate")
async def reinstate_team_member(
    team_id: str,
    member_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Re-activate a previously removed team member."""
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE amfs_team_members
                   SET removed_at = NULL, removed_by = NULL
                   WHERE id = %s::uuid AND team_id = %s::uuid
                     AND namespace = %s AND removed_at IS NOT NULL
                   RETURNING id, email, display_name, role,
                             invited_at, accepted_at, created_at""",
                (member_id, team_id, ns),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Removed member not found"}
    _audit_log(
        "team.member.reinstate",
        resource=f"{team_id}/{row['email']}",
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "email": row["email"],
        "displayName": row["display_name"],
        "role": row["role"],
        "invitedAt": row["invited_at"].isoformat(),
        "acceptedAt": row["accepted_at"].isoformat() if row["accepted_at"] else None,
        "createdAt": row["created_at"].isoformat(),
        "reinstated": True,
    }


@app.get("/api/v1/admin/members/check-email")
async def check_member_email(
    email: str = Query(...),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Check if an email is associated with any active or removed team memberships.

    Called by the dashboard OAuth flow to determine if a returning user
    should be blocked (removed) or allowed through.

    Returns status: "active", "removed", or "not_found".
    """
    pool = _get_db_pool()
    if pool is None:
        return {"email": email, "status": "unknown", "memberships": []}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT m.id, m.team_id, t.name AS team_name, m.role,
                          m.removed_at, m.removed_by,
                          m.created_at, m.accepted_at
                   FROM amfs_team_members m
                   JOIN amfs_teams t ON t.id = m.team_id
                   WHERE m.email = %s AND m.namespace = %s
                   ORDER BY m.removed_at NULLS FIRST""",
                (email, ns),
            )
            rows = cur.fetchall()
    if not rows:
        return {"email": email, "status": "not_found", "memberships": []}
    active = [r for r in rows if r["removed_at"] is None]
    removed = [r for r in rows if r["removed_at"] is not None]
    if active:
        status = "active"
    elif removed:
        status = "removed"
    else:
        status = "not_found"
    return {
        "email": email,
        "status": status,
        "memberships": [
            {
                "id": str(r["id"]),
                "teamId": str(r["team_id"]),
                "teamName": r["team_name"],
                "role": r["role"],
                "removedAt": r["removed_at"].isoformat() if r["removed_at"] else None,
                "removedBy": r["removed_by"],
                "createdAt": r["created_at"].isoformat(),
                "acceptedAt": r["accepted_at"].isoformat() if r["accepted_at"] else None,
            }
            for r in rows
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Admin — Pattern Detection (Pro)
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/patterns")
async def list_detected_patterns(
    pattern_type: str | None = Query(None),
    category: str | None = Query(None),
    severity: str | None = Query(None),
    resolved: bool | None = Query(None),
    agent_id: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List previously detected patterns from the database."""
    try:
        from amfs_patterns.detector import PATTERN_CATEGORIES, PATTERN_METADATA
    except ImportError:
        PATTERN_CATEGORIES = {}
        PATTERN_METADATA = {}

    pool = _get_db_pool()
    if pool is None:
        return {"patterns": [], "categories": PATTERN_CATEGORIES, "metadata": PATTERN_METADATA}

    ns = _get_namespace()
    conditions = ["namespace = %s"]
    params: list[Any] = [ns]

    if pattern_type is not None:
        conditions.append("pattern_type = %s")
        params.append(pattern_type)
    if category is not None:
        conditions.append("category = %s")
        params.append(category)
    if severity is not None:
        conditions.append("severity = %s")
        params.append(severity)
    if resolved is not None:
        conditions.append("resolved = %s")
        params.append(resolved)
    if agent_id is not None:
        conditions.append("(details->>'agent' = %s OR details->>'agent_a' = %s OR details->>'agent_b' = %s)")
        params.extend([agent_id, agent_id, agent_id])

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, pattern_type, severity, entity_path, description,
               details, resolved, detected_at, resolved_at,
               COALESCE(category, 'collaboration') as category
        FROM amfs_detected_patterns
        WHERE {where}
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
            detected_at DESC
        LIMIT %s
    """
    params.append(limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return {
        "patterns": [
            {
                "id": str(row["id"]),
                "patternType": row["pattern_type"],
                "severity": row["severity"],
                "category": row["category"],
                "entityPath": row["entity_path"],
                "description": row["description"],
                "details": row["details"] or {},
                "resolved": row["resolved"],
                "detectedAt": row["detected_at"].isoformat(),
                "resolvedAt": row["resolved_at"].isoformat() if row["resolved_at"] else None,
            }
            for row in rows
        ],
        "categories": PATTERN_CATEGORIES,
        "metadata": PATTERN_METADATA,
    }


@app.post("/api/v1/admin/patterns/scan")
async def run_pattern_scan(
    req: RunPatternDetectionRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run the collaboration-aware pattern detector and persist results."""
    try:
        from amfs_patterns import PatternDetector
        from amfs_patterns.detector import PATTERN_CATEGORIES, PATTERN_METADATA
    except ImportError:
        return {"error": "amfs-patterns package not installed"}

    mem = _get_memory()
    entries = mem.list(req.entity_path)
    outcomes = mem._adapter.list_outcomes(entity_path=req.entity_path, limit=10000)

    if req.agent_id:
        entries = [e for e in entries if e.provenance.agent_id == req.agent_id]

    branches: list[Any] = []
    pull_requests: list[Any] = []
    try:
        branches = mem.list_branches()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        pass
    try:
        pull_requests = mem.list_pull_requests()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        pass

    detector = PatternDetector(
        stale_days=req.stale_days,
        orphan_days=req.orphan_days,
        pr_stale_days=req.pr_stale_days,
        similarity_threshold=req.similarity_threshold,
        incident_threshold=req.incident_threshold,
    )
    report = detector.analyze(
        entries,
        outcome_data=outcomes,
        branches=branches,
        pull_requests=pull_requests,
    )

    pool = _get_db_pool()
    persisted = 0
    if pool is not None:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM amfs_detected_patterns WHERE resolved = FALSE")
                for p in report.patterns:
                    cur.execute(
                        """INSERT INTO amfs_detected_patterns
                               (pattern_type, severity, category, entity_path,
                                description, details, detected_at)
                           VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)""",
                        (
                            p.pattern_type,
                            p.severity,
                            p.category,
                            p.entity_path,
                            p.description,
                            json.dumps(p.details, default=str),
                            p.detected_at,
                        ),
                    )
                    persisted += 1

    _audit_log(
        "patterns.scan",
        resource=req.entity_path or "*",
        ip_address=request.client.host if request.client else None,
    )

    return {
        "scannedEntries": report.scanned_entries,
        "scannedOutcomes": report.scanned_outcomes,
        "scanDurationMs": round(report.scan_duration_ms, 2),
        "patternsFound": len(report.patterns),
        "patternsPersisted": persisted,
        "patterns": [
            {
                "patternType": p.pattern_type,
                "severity": p.severity,
                "category": p.category,
                "entityPath": p.entity_path,
                "description": p.description,
                "details": p.details,
                "detectedAt": p.detected_at.isoformat(),
            }
            for p in report.patterns
        ],
        "categories": PATTERN_CATEGORIES,
        "metadata": PATTERN_METADATA,
    }


@app.patch("/api/v1/admin/patterns/{pattern_id}/resolve")
async def resolve_pattern(
    pattern_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Mark a detected pattern as resolved."""
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Pattern management requires a Postgres backend"}
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE amfs_detected_patterns
                   SET resolved = TRUE, resolved_at = NOW()
                   WHERE id = %s::uuid
                   RETURNING id, pattern_type, entity_path""",
                (pattern_id,),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Pattern not found"}
    _audit_log(
        "patterns.resolve",
        resource=pattern_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"resolved": str(row["id"]), "patternType": row["pattern_type"]}


# ──────────────────────────────────────────────────────────────────────
# Pro — Expertise Heatmap
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/pro/graph/neighbors")
async def graph_neighbors(
    entity: str = Query(...),
    relation: str | None = Query(None),
    direction: str = Query("both"),
    min_confidence: float = Query(0.0),
    depth: int = Query(1, ge=1, le=5),
    limit: int = Query(200, ge=1, le=1000),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Traverse the knowledge graph from an entity."""
    mem = _get_memory()
    try:
        edges = mem.graph_neighbors(
            entity,
            relation=relation,
            direction=direction,
            min_confidence=min_confidence,
            depth=depth,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("graph_neighbors failed for %s: %s", entity, exc)
        return JSONResponse(
            {"entity": entity, "edges": [], "count": 0, "error": str(exc)},
            status_code=200,
        )
    return {
        "entity": entity,
        "edges": [e.model_dump(mode="json") for e in edges],
        "count": len(edges),
    }


@app.get("/api/v1/pro/graph/expertise")
async def expertise_graph(
    request: Request,
    agent_id: str | None = Query(None),
    limit_agents: int = Query(30, ge=1, le=200),
    limit_entities: int = Query(30, ge=1, le=200),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Build an agent×entity expertise heatmap.

    Returns a list of agents, entities, and cells with scores derived
    from write counts. When ``agent_id`` is provided, results are scoped
    to that single agent.
    """
    mem = _get_memory()
    entries = mem.list()

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        entries = vis.filter_entries(entries)

    visible_agents: set[str] | None = None
    if vis is not None and vis.should_filter():
        visible_agents = vis.get_user_agents()

    agent_entity_weights: dict[str, dict[str, int]] = {}
    agent_totals: dict[str, int] = {}
    entity_totals: dict[str, int] = {}

    for e in entries:
        aid = e.provenance.agent_id
        if agent_id and aid != agent_id:
            continue
        if visible_agents is not None and aid not in visible_agents:
            continue
        ep = e.entity_path
        agent_totals[aid] = agent_totals.get(aid, 0) + 1
        entity_totals[ep] = entity_totals.get(ep, 0) + 1
        if aid not in agent_entity_weights:
            agent_entity_weights[aid] = {}
        agent_entity_weights[aid][ep] = agent_entity_weights[aid].get(ep, 0) + 1

    top_agents = [
        a for a, _ in sorted(agent_totals.items(), key=lambda x: x[1], reverse=True)
    ][:limit_agents]
    top_agent_set = set(top_agents)

    relevant_entities: set[str] = set()
    for aid in top_agent_set:
        relevant_entities.update(agent_entity_weights.get(aid, {}).keys())
    top_entities = [
        ep
        for ep, _ in sorted(
            [(ep, entity_totals.get(ep, 0)) for ep in relevant_entities],
            key=lambda x: x[1],
            reverse=True,
        )
    ][:limit_entities]
    top_entity_set = set(top_entities)

    cells: list[dict[str, Any]] = []
    for aid in top_agents:
        for ep, weight in agent_entity_weights.get(aid, {}).items():
            if ep in top_entity_set:
                cells.append({
                    "agent": aid,
                    "entity": ep,
                    "score": weight,
                    "relations": ["writes"],
                })

    return {
        "agents": top_agents,
        "entities": top_entities,
        "cells": cells,
    }


@app.post("/api/v1/pro/graph/backfill")
async def graph_backfill(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Backfill knowledge graph edges from existing entries and outcomes.

    Materializes edges from: pattern_refs → 'references' edges,
    outcome causal chains → 'informed' + 'read' edges,
    agent writes → 'wrote' edges.
    """
    mem = _get_memory()
    entries = mem.list()
    try:
        outcomes = mem._adapter.list_outcomes() if hasattr(mem._adapter, "list_outcomes") else []
    except Exception:
        outcomes = []

    created = 0
    errors = 0

    for e in entries:
        aid = e.provenance.agent_id
        ep = e.entity_path
        ek = f"{ep}/{e.key}"

        try:
            mem._adapter.upsert_graph_edge(
                GraphEdge(
                    source_entity=aid,
                    source_type="agent",
                    relation="wrote",
                    target_entity=ek,
                    target_type="entry",
                    confidence=e.confidence,
                    provenance={"agent_id": aid, "trigger": "backfill"},
                ),
                namespace=mem.namespace,
                branch=e.branch or "main",
            )
            created += 1
        except Exception:
            errors += 1

        for ref in (e.provenance.pattern_refs or []):
            try:
                mem._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=ek,
                        source_type="entry",
                        relation="references",
                        target_entity=ref,
                        target_type="entry",
                        provenance={"agent_id": aid, "trigger": "backfill"},
                    ),
                    namespace=mem.namespace,
                    branch=e.branch or "main",
                )
                created += 1
            except Exception:
                errors += 1

    for o in outcomes:
        otype = getattr(o, "outcome_type", None)
        edge_conf = 1.0 if otype and otype.value == "success" else 0.7
        aid = getattr(o, "agent_id", "unknown")
        for ek in getattr(o, "causal_entry_keys", []):
            try:
                mem._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=ek,
                        source_type="entry",
                        relation="informed",
                        target_entity=o.outcome_ref,
                        target_type="outcome",
                        confidence=edge_conf,
                        provenance={"agent_id": aid, "trigger": "backfill"},
                    ),
                    namespace=mem.namespace,
                    branch="main",
                )
                created += 1
            except Exception:
                errors += 1

    return {"edges_created": created, "errors": errors, "entries_scanned": len(entries), "outcomes_scanned": len(outcomes)}


# ──────────────────────────────────────────────────────────────────────
# Pro — HMO Memory Tiers
# ──────────────────────────────────────────────────────────────────────


def _compute_tiers(entries: list) -> tuple[dict[str, int], dict[str, float]]:
    """Score entries and assign HMO tiers (Hot/Warm/Archive)."""
    from amfs_core.tiering import PriorityScorer, TierAssigner

    scorer = PriorityScorer()
    assigner = TierAssigner()
    return assigner.assign_with_scores(entries, scorer)


@app.get("/api/v1/pro/tiers/distribution")
async def tiers_distribution(
    agent_id: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return HMO tier distribution (Hot / Warm / Archive)."""
    mem = _get_memory()
    entries = mem.list()
    if agent_id:
        entries = [e for e in entries if e.provenance.agent_id == agent_id]

    tiers, scores = _compute_tiers(entries)

    hot = warm = archive = scored_count = 0
    score_sum = 0.0
    for key, tier in tiers.items():
        if tier == 1:
            hot += 1
        elif tier == 2:
            warm += 1
        else:
            archive += 1
        s = scores.get(key)
        if s is not None:
            scored_count += 1
            score_sum += s

    return {
        "total": len(entries),
        "hot": hot,
        "warm": warm,
        "archive": archive,
        "scored": scored_count,
        "avg_priority_score": round(score_sum / scored_count, 6) if scored_count else None,
    }


@app.get("/api/v1/pro/tiers/entries")
async def tiers_entries(
    tier: int = Query(..., ge=1, le=3),
    limit: int = Query(50, ge=1, le=500),
    agent_id: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return entries for a given HMO tier as a flat array."""
    mem = _get_memory()
    entries = mem.list()
    if agent_id:
        entries = [e for e in entries if e.provenance.agent_id == agent_id]

    tier_map, score_map = _compute_tiers(entries)

    filtered = [e for e in entries if tier_map.get(e.entry_key) == tier]
    filtered.sort(key=lambda e: score_map.get(e.entry_key, 0.0), reverse=True)

    return [
        {
            "entity_path": e.entity_path,
            "key": e.key,
            "confidence": e.confidence,
            "tier": tier,
            "priority_score": round(score_map.get(e.entry_key, 0.0), 6),
            "recall_count": getattr(e, "recall_count", 0),
            "importance_score": getattr(e, "importance_score", None),
            "written_at": e.provenance.written_at.isoformat(),
            "agent_id": e.provenance.agent_id,
        }
        for e in filtered[:limit]
    ]


# ──────────────────────────────────────────────────────────────────────
# SSE Stream
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/stream")
async def stream(
    entity_path: str = Query("*"),
    _auth: str | None = Depends(verify_api_key),
) -> EventSourceResponse:
    return EventSourceResponse(_sse_manager.event_generator(entity_path))


# ──────────────────────────────────────────────────────────────────────
# System Config
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/config")
async def get_system_config(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return system configuration and active Pro module status."""
    mem = _get_memory()
    adapter_type = type(mem._adapter).__name__

    pro_modules: dict[str, bool] = {}
    for mod_name in ("amfs_traces", "amfs_pro_api", "amfs_critic", "amfs_distiller",
                     "amfs_retrieval", "amfs_ml", "amfs_safety", "amfs_extraction",
                     "amfs_pro_connectors"):
        try:
            __import__(mod_name)
            pro_modules[mod_name] = True
        except ImportError:
            pro_modules[mod_name] = False

    return {
        "adapter": adapter_type,
        "namespace": getattr(mem, "_namespace", os.environ.get("AMFS_NAMESPACE", "default")),
        "agent_id": mem.agent_id,
        "session_id": mem.session_id,
        "postgres_configured": bool(os.environ.get("AMFS_POSTGRES_DSN")),
        "llm_configured": bool(os.environ.get("AMFS_LLM_API_KEY")),
        "extraction_enabled": os.environ.get("AMFS_AUTO_EXTRACT", "").lower() == "true",
        "otel_enabled": bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")),
        "pro_modules": pro_modules,
    }


# ──────────────────────────────────────────────────────────────────────
# Connectors / Webhook Ingestion
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/connectors")
async def list_connectors(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List available and installed connectors."""
    try:
        from amfs_connectors import ConnectorRegistry

        registry = ConnectorRegistry()
        return {
            "connectors": registry.list_installed(),
            "total": len(registry.list_available()),
        }
    except ImportError:
        return {"connectors": [], "total": 0}


# ──────────────────────────────────────────────────────────────────────
# Memory Cortex (Briefing + Status)
# ──────────────────────────────────────────────────────────────────────


def _is_agent_visible_for_entity(
    agent_id: str,
    entity_path: str,
    user_agents: set[str],
    room_map: dict[str, set[str]],
) -> bool:
    """Check if an agent is visible in the context of a specific entity_path.

    Mirrors the logic of UserVisibilityFilter.is_entry_visible but works
    with raw agent_id + entity_path instead of requiring a full entry object.
    """
    if agent_id in user_agents:
        return True
    room_members = room_map.get(entity_path)
    if room_members:
        if agent_id in room_members:
            return True
        if agent_id in ("amfs-server", "system", "amfs"):
            return True
    return False


def _filter_briefing_digests(vis: Any, digests: list) -> list:
    """Filter briefing digests and their hot_context through visibility.

    Ensures briefing only surfaces data the user could also access via
    amfs_read / amfs_search, maintaining consistency across all endpoints.
    """
    user_agents = vis.get_user_agents()
    room_map = vis.get_room_map()
    filtered = []

    for digest in digests:
        scope = digest.scope
        source_agents = digest.source_agents

        if "hot_context" in digest.summary:
            digest.summary["hot_context"] = [
                h for h in digest.summary["hot_context"]
                if _is_agent_visible_for_entity(
                    h.get("agent", ""), scope, user_agents, room_map,
                )
            ]

        if source_agents:
            has_visible = any(
                _is_agent_visible_for_entity(a, scope, user_agents, room_map)
                for a in source_agents
            )
            if not has_visible:
                continue
        else:
            has_room_access = scope in room_map
            has_visible_hot = bool(digest.summary.get("hot_context"))
            if not has_room_access and not has_visible_hot:
                continue

        filtered.append(digest)

    return filtered


@app.get("/api/v1/briefing")
async def get_briefing(
    request: Request,
    entity_path: str | None = Query(None),
    agent_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get a ranked briefing of compiled knowledge digests."""
    mem = _get_memory()
    digests = mem.briefing(
        entity_path=entity_path,
        agent_id=agent_id,
        limit=limit,
    )

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        digests = _filter_briefing_digests(vis, digests)

    return {
        "digests": [d.model_dump(mode="json") for d in digests],
        "total": len(digests),
    }


@app.get("/api/v1/cortex/status")
async def cortex_status(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get Cortex worker status and digest statistics."""
    mem = _get_memory()
    try:
        digests = mem.briefing(limit=0)
    except Exception:
        digests = []

    try:
        from amfs_postgres.adapter import PostgresAdapter

        adapter = mem._adapter
        if isinstance(adapter, PostgresAdapter):
            all_digests = adapter.list_digests()

            vis = _get_visibility_filter(request)
            if vis is not None and vis.should_filter():
                all_digests = _filter_briefing_digests(vis, all_digests)

            return {
                "status": "active" if all_digests else "idle",
                "digest_count": len(all_digests),
                "digest_types": {
                    "entity": sum(1 for d in all_digests if d.digest_type.value == "entity"),
                    "agent_brief": sum(1 for d in all_digests if d.digest_type.value == "agent_brief"),
                    "source": sum(1 for d in all_digests if d.digest_type.value == "source"),
                },
            }
    except ImportError:
        pass
    except Exception:
        logger.exception("Failed to fetch Cortex status")

    return {"status": "unavailable", "digest_count": 0}


@app.get("/api/v1/cortex/digests")
async def list_cortex_digests(
    request: Request,
    digest_type: str | None = Query(None),
    scope: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List compiled digests, optionally filtered by type and scope."""
    try:
        from amfs_postgres.adapter import PostgresAdapter
        from amfs_core.models import DigestType

        mem = _get_memory()
        adapter = mem._adapter
        if isinstance(adapter, PostgresAdapter):
            dt = DigestType(digest_type) if digest_type else None
            namespace = getattr(adapter, "_namespace", "default")
            digests = adapter.list_digests(digest_type=dt, namespace=namespace)
            if scope:
                digests = [d for d in digests if d.scope == scope]

            vis = _get_visibility_filter(request)
            if vis is not None and vis.should_filter():
                digests = _filter_briefing_digests(vis, digests)

            return {
                "digests": [d.model_dump(mode="json") for d in digests],
                "total": len(digests),
            }
    except (ImportError, ValueError):
        pass
    except Exception:
        logger.exception("Failed to list cortex digests")

    return {"digests": [], "total": 0}


@app.get("/api/v1/cortex/activity")
async def cortex_activity(
    limit: int = Query(default=50, le=200),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get recent Cortex compilation and event activity."""
    if _cortex_worker:
        log = _cortex_worker.activity_log
        recent = log[-limit:] if len(log) > limit else log
        return {
            "events": list(reversed(recent)),
            "total": len(log),
            "throughput": _cortex_worker.throughput,
            "stats": _cortex_worker.stats,
        }
    return {"events": [], "total": 0, "throughput": [], "stats": None}


@app.post("/api/v1/cortex/recompile")
async def cortex_recompile(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Trigger a full digest recompilation."""
    if not _cortex_worker:
        raise HTTPException(status_code=503, detail="Cortex worker not running")
    count = _cortex_worker._compiler.recompile_all()
    return {"recompiled": count}


# ------------------------------------------------------------------
# Consolidation (Cortex compaction) endpoints
# ------------------------------------------------------------------


@app.post("/api/v1/cortex/consolidate")
async def run_consolidation(
    body: dict[str, Any] | None = None,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run Tier A (auto-safe) consolidation.

    Optionally accepts ``entity_path`` to consolidate a single entity.
    """
    from amfs_cortex.consolidator import ConsolidationStrategy

    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem._namespace
    branch = (body or {}).get("branch", "main")
    entity_path = (body or {}).get("entity_path")

    strategy = ConsolidationStrategy(adapter, namespace=namespace)
    if entity_path:
        report = strategy.run_entity(entity_path, branch=branch)
    else:
        report = strategy.run(branch=branch)

    return report.model_dump()


@app.get("/api/v1/cortex/consolidation/candidates")
async def list_consolidation_candidates(
    entity_path: str = Query(...),
    branch: str = Query("main"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List Tier B consolidation candidates (proposals) for an entity."""
    from amfs_cortex.consolidator import ConsolidationStrategy

    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem._namespace

    strategy = ConsolidationStrategy(adapter, namespace=namespace)
    proposals = strategy.find_consolidation_candidates(entity_path, branch=branch)

    return {
        "entity_path": entity_path,
        "proposals": [p.model_dump() for p in proposals],
    }


@app.get("/api/v1/cortex/consolidation/proposals")
async def list_consolidation_proposals(
    entity_path: str | None = Query(None),
    status: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List consolidation proposals from persisted branches.

    Scans branches matching ``cortex/consolidation/`` and returns
    structured proposal metadata extracted from each branch's diff.
    """
    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem._namespace

    try:
        branches = adapter.list_branches(namespace=namespace, status="active" if status != "all" else None)
    except Exception:
        return {"proposals": [], "total": 0}

    consolidation_branches = [
        b for b in branches if b.name.startswith("cortex/consolidation/")
    ]
    if entity_path:
        consolidation_branches = [
            b for b in consolidation_branches
            if b.name.startswith(f"cortex/consolidation/{entity_path}/")
        ]

    if status and status in ("approved", "rejected"):
        branch_status = "merged" if status == "approved" else "closed"
        branches_all = adapter.list_branches(namespace=namespace, status=branch_status)
        extra = [b for b in branches_all if b.name.startswith("cortex/consolidation/")]
        if entity_path:
            extra = [b for b in extra if b.name.startswith(f"cortex/consolidation/{entity_path}/")]
        consolidation_branches.extend(extra)

    all_proposals = []
    for b in consolidation_branches:
        parts = b.name.split("/")
        ep = "/".join(parts[2:-1]) if len(parts) > 3 else parts[2] if len(parts) >= 3 else "unknown"

        branch_status_map = {"active": "pending", "merged": "approved", "closed": "rejected"}
        prop_status = branch_status_map.get(b.status.value if hasattr(b.status, "value") else str(b.status), "pending")

        if status and status != "all" and prop_status != status:
            continue

        try:
            diff = adapter.diff_branch(b.name, namespace=namespace)
            entry_keys = [f"{d.entity_path}/{d.key}" for d in diff.entries] if hasattr(diff, "entries") else []
        except Exception:
            diff = None
            entry_keys = []

        proposed_value = None
        proposed_confidence = 0.0
        if diff and hasattr(diff, "entries") and diff.entries:
            first = diff.entries[0]
            proposed_value = first.branch_value if hasattr(first, "branch_value") else None
            proposed_confidence = 0.8

        all_proposals.append({
            "id": b.id or b.name,
            "entity_path": ep,
            "branch_name": b.name,
            "strategy": b.description.split(":")[0].strip().lower().replace(" ", "_") if b.description and ":" in b.description else "consolidation",
            "risk_tier": "review_required",
            "source_entry_keys": entry_keys,
            "proposed_value": proposed_value,
            "proposed_confidence": proposed_confidence,
            "compression_ratio": max(len(entry_keys), 1),
            "rationale": b.description or "Consolidation proposal",
            "status": prop_status,
            "created_at": (b.created_at or b.branched_at).isoformat() if (b.created_at or b.branched_at) else None,
            "reviewed_by": b.merged_by,
            "reviewed_at": b.merged_at.isoformat() if b.merged_at else None,
        })

    return {
        "proposals": all_proposals,
        "total": len(all_proposals),
    }


@app.get("/api/v1/cortex/consolidation/status")
async def consolidation_status(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get consolidation health metrics for the dashboard."""
    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem._namespace

    consolidation_runs = 0
    total_auto_archived = 0
    if _cortex_worker:
        consolidation_runs = _cortex_worker._consolidation_runs
        for entry in _cortex_worker._activity_log:
            if entry.get("type") == "consolidation_run":
                total_auto_archived += entry.get("auto_archived", 0)

    try:
        branches = adapter.list_branches(namespace=namespace, status="active")
        pending_branches = [
            b for b in branches if b.name.startswith("cortex/consolidation/")
        ]
    except Exception:
        pending_branches = []

    entities_ready: list[str] = []
    try:
        entries = adapter.list()
        entity_counts: dict[str, int] = {}
        for e in entries:
            entity_counts[e.entity_path] = entity_counts.get(e.entity_path, 0) + 1
        for ep, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            if count >= 10:
                entities_ready.append(ep)
            if len(entities_ready) >= 10:
                break
    except Exception:
        pass

    return {
        "consolidation_runs": consolidation_runs,
        "auto_archived": total_auto_archived,
        "pending_proposals": len(pending_branches),
        "entities_ready": entities_ready,
    }


@app.post("/api/v1/webhooks/{connector_name}")
async def ingest_webhook(
    connector_name: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Receive and process a webhook event through the connector framework."""
    try:
        from amfs_connectors import WebhookIngester, WebhookConfig
    except ImportError:
        raise HTTPException(status_code=501, detail="Connector framework not installed")

    body = await request.body()
    headers = dict(request.headers)

    mem = _get_memory()

    secret_env = f"AMFS_CONNECTOR_{connector_name.upper().replace('-', '_')}_SECRET"
    secret = os.environ.get(secret_env)

    config = WebhookConfig(
        name=connector_name,
        connector_type="webhook",
        entity_path=connector_name,
        secret=secret,
    )
    ingester = WebhookIngester(config, memory=mem)

    try:
        from amfs_connectors import ConnectorRegistry

        registry = ConnectorRegistry()
        connector = registry.get(connector_name)
        if connector:
            ingester.register_transform("*", connector.transform)
    except Exception:
        pass

    event_type = headers.get("x-event-type", "generic")
    event_id = headers.get("x-event-id")

    results = ingester.ingest(
        body,
        headers,
        source=connector_name,
        event_type=event_type,
        event_id=event_id,
    )

    original_agent = mem._tagger.agent_id
    mem._tagger.agent_id = f"webhook/{connector_name}"
    persisted = 0
    try:
        for r in results:
            if r.success and r.action == "write":
                entry = mem.write(
                    r.entity_path,
                    r.key,
                    r.details,
                    confidence=1.0,
                    memory_type=MemoryType.EXPERIENCE,
                )
                _sse_manager.broadcast(entry)
                persisted += 1
    finally:
        mem._tagger.agent_id = original_agent

    return {
        "results": [r.model_dump(mode="json") for r in results],
        "total": len(results),
        "persisted": persisted,
    }


@app.post("/api/v1/events")
async def ingest_event(
    body: EventRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Ingest an event directly into the shared memory pool.

    Simple alternative to the webhook/connector framework for apps that
    just want to push context into AMFS.
    """
    mem = _get_memory()

    original_agent = mem._tagger.agent_id
    mem._tagger.agent_id = f"external/{body.source}"
    try:
        entry = mem.write(
            body.entity_path,
            body.key,
            body.value,
            confidence=1.0,
            memory_type=MemoryType.EXPERIENCE,
        )
        _sse_manager.broadcast(entry)
    finally:
        mem._tagger.agent_id = original_agent

    return {
        "status": "ok",
        "entity_path": body.entity_path,
        "key": body.key,
        "source": body.source,
        "agent_id": f"external/{body.source}",
    }


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amfs-http",
        description="AMFS HTTP/REST API server with SSE support",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AMFS_HTTP_HOST", "0.0.0.0"),
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.environ.get("AMFS_HTTP_PORT", "8741")),
        help="Port to bind (default: 8741)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--with-cortex",
        action="store_true",
        default=False,
        help="Run embedded Cortex worker in-process (for single-instance deployments)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("AMFS_HTTP_WORKERS", "1")),
        help="Number of uvicorn worker processes (default: 1, env: AMFS_HTTP_WORKERS)",
    )
    return parser.parse_args()


_cortex_worker = None


def _make_tenant_provider(dsn: str):
    """Build a tenant_provider callback for the Cortex worker.

    Returns a callable that queries the ``accounts`` table (Pro/SaaS only)
    for all tenant UUIDs.  Falls back to ``[None]`` for OSS deployments
    that don't have the ``accounts`` table.
    """

    def _provider() -> list:
        try:
            import psycopg

            with psycopg.connect(dsn, autocommit=True) as conn:
                rows = conn.execute(
                    "SELECT id::text FROM accounts"
                ).fetchall()
                if rows:
                    return [r[0] for r in rows]
        except Exception:
            pass
        return [None]

    return _provider


def main() -> None:
    """Run the AMFS HTTP server via uvicorn."""
    global _cortex_worker
    args = _parse_args()

    if args.with_cortex:
        dsn = os.environ.get("AMFS_POSTGRES_DSN")
        if dsn:
            import threading

            try:
                from amfs_postgres.adapter import PostgresAdapter
                from amfs_cortex.compiler import DigestCompiler
                from amfs_cortex.worker import CortexWorker

                namespace = os.environ.get("AMFS_NAMESPACE", "default")
                adapter = PostgresAdapter(dsn=dsn, namespace=namespace)

                strategies = []
                try:
                    from amfs_cortex_pro import get_pro_strategies
                    strategies = get_pro_strategies()
                    logger.info("Pro compilation strategies loaded")
                except ImportError:
                    pass

                compiler = DigestCompiler(
                    adapter=adapter,
                    strategies=strategies or None,
                    namespace=namespace,
                )
                tenant_provider = _make_tenant_provider(dsn)
                _cortex_worker = CortexWorker(
                    dsn=dsn,
                    compiler=compiler,
                    use_advisory_lock=False,
                    tenant_provider=tenant_provider,
                )

                try:
                    from amfs_cortex_pro import get_outcome_wiring, HotContextTracker
                    wiring = get_outcome_wiring(adapter, namespace)
                    if wiring:
                        _cortex_worker._outcome_wiring = wiring
                        logger.info("Outcome wiring attached to embedded Cortex worker")
                    tracker = HotContextTracker()
                    _cortex_worker._hot_tracker = tracker
                    logger.info("Hot context tracker attached to embedded Cortex worker")
                except ImportError:
                    pass

                from amfs_http.pro_proxy import create_forwarder
                forwarder = create_forwarder()
                if forwarder:
                    _cortex_worker._pro_forwarder = forwarder

                t = threading.Thread(target=_cortex_worker.run, daemon=True, name="cortex-embedded")
                t.start()
                logger.info("Embedded Cortex worker started")
            except ImportError:
                logger.warning("--with-cortex requires amfs-cortex package")
            except Exception:
                logger.exception("Failed to start embedded Cortex worker — server will run without Cortex")
        else:
            logger.warning("--with-cortex requires AMFS_POSTGRES_DSN")

    workers = args.workers
    if args.reload and workers > 1:
        logger.warning("--reload is incompatible with --workers > 1; forcing workers=1")
        workers = 1

    logger.info(
        "Starting AMFS HTTP server on %s:%d (workers=%d)", args.host, args.port, workers
    )
    uvicorn.run(
        "amfs_http.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
    )


if __name__ == "__main__":
    main()
