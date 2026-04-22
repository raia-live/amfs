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
import hashlib
import json
import logging
import os
import secrets
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
    AMFSConfig,
    DecisionTrace,
    Event,
    EventType,
    GraphEdge,
    LayerConfig,
    MemoryEntry,
    SearchQuery,
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
    RunPatternDetectionRequest,
    SearchRequest,
    UpdateTeamMemberRequest,
    UpdateTeamRequest,
    WriteRequest,
)
from amfs_http.sse import SSEManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AMFS HTTP API",
    description="Agent Memory File System — REST API with SSE support",
    version="0.1.0",
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

    apply_tenant_headers_from_request(request)
    try:
        return await call_next(request)
    finally:
        clear_tenant_headers()


_memory: AgentMemory | None = None
_sse_manager = SSEManager()

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
    if ctx is not None:
        return {
            "authenticated": True,
            "mode": "pro",
            "account_id": str(ctx.account_id),
            "actor_id": str(ctx.actor_id),
            "key_type": ctx.key_type.value if ctx.key_type else None,
            "scopes": [
                {
                    "entity_path_pattern": s.entity_path_pattern,
                    "permission": s.permission.value,
                }
                for s in ctx.scopes
            ],
            "rate_limit_rpm": ctx.rate_limit_rpm,
            "is_admin": ctx.is_admin,
        }
    return {
        "authenticated": _auth is not None,
        "mode": "oss",
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
            mem._adapter.ensure_agent(req.agent_id, mem.namespace)
        except Exception:
            pass
        _ensure_agent_owner(request, req.agent_id, mem.namespace)
    try:
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
    _audit_log(
        "memory.write",
        resource=f"{req.entity_path}/{req.key}",
        ip_address=request.client.host if request.client else None,
    )

    try:
        agent = entry.provenance.agent_id
        ek = f"{entry.entity_path}/{entry.key}"
        mem._adapter.upsert_graph_edge(
            GraphEdge(
                source_entity=agent,
                source_type="agent",
                relation="wrote",
                target_entity=ek,
                target_type="entry",
                confidence=entry.confidence,
                provenance={"agent_id": agent, "trigger": "write"},
            ),
            namespace=mem.namespace,
            branch=entry.branch or "main",
        )
    except Exception:
        logger.debug("Failed to materialize wrote edge", exc_info=True)

    return _entry_to_response(entry)


@app.get("/api/v1/entries")
async def list_entries(
    request: Request,
    entity_path: str | None = Query(None),
    branch: str = Query("main"),
    include_superseded: bool = Query(False),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    entries = mem.list(entity_path, branch=branch, include_superseded=include_superseded)

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        entries = vis.filter_entries(entries)

    return {"entries": [_entry_to_response(e) for e in entries]}


# ──────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/search")
async def search_entries(
    request: Request,
    req: SearchRequest,
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    mem = _get_memory()
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
    try:
        results = mem._adapter.search(sq, branch=branch)
    except TypeError:
        results = mem._adapter.search(sq)

    vis = _get_visibility_filter(request)
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
        entries = mem.list()
        entries = vis.filter_entries(entries)

        entity_set: set[str] = set()
        agent_set: set[str] = set()
        for e in entries:
            entity_set.add(e.entity_path)
            agent_set.add(e.provenance.agent_id)

        return {
            "total_entries": len(entries),
            "total_entities": len(entity_set),
            "total_agents": len(agent_set),
            "oldest_entry": min(
                (e.provenance.written_at for e in entries if e.provenance.written_at),
                default=None,
            ),
            "newest_entry": max(
                (e.provenance.written_at for e in entries if e.provenance.written_at),
                default=None,
            ),
        }

    stats = mem.stats()
    return json.loads(json.dumps(stats.model_dump(mode="json"), default=str))


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
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return an agent's profile, stats, and registration info.

    When no explicit profile has been registered, synthesises one from the
    agent's actual activity — entity paths become auto-inferred capabilities
    and memory-type distribution is included.
    """
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
    agent_id: str,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from amfs_core.models import AgentProfile

    mem = _get_memory()
    profile = AgentProfile.model_validate(body)
    agent = mem._adapter.update_agent_profile(agent_id, profile)
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
    mem = _get_memory()
    entries = mem.list()

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        entries = vis.filter_entries(entries)

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
                            f"SELECT agent_id, created_at FROM amfs_agents "
                            f"WHERE namespace = %s AND agent_id IN ({placeholders})",
                            [adapter._namespace, *known_agent_ids],
                        )
                        for row in cur.fetchall():
                            agent_registration[row["agent_id"]] = {"created_at": row["created_at"]}
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
        agents.append({
            "agentId": ad["agent_id"],
            "entriesWritten": ad["entries_written"],
            "entitiesTouched": len(ad["entities_touched"]),
            "lastActive": ad["last_active"].isoformat() if ad["last_active"] else None,
            "createdAt": created.isoformat() if created else None,
            "description": desc_info.get("description", ""),
            "platform": desc_info.get("platform", ""),
        })
    return {"agents": agents}


@app.get("/api/v1/agents/{agent_id:path}/memory-graph")
async def agent_memory_graph(
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """All entries written by or read by this agent, grouped by entity."""
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
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Which other agents' memory this agent has read."""
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
    agent_id: str,
    entity_path: str,
    key: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Recall an agent's own memory for a key (agent-scoped read).

    Unlike the generic read endpoint, this returns only entries written
    by the specified agent — what that agent's brain actually knows.
    """
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
    agent_id: str,
    entity_path: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all entries written by a specific agent."""
    mem = _get_memory()
    entries = mem.search(entity_path=entity_path, agent_id=agent_id)
    return {
        "agentId": agent_id,
        "count": len(entries),
        "entries": [_entry_to_response(e) for e in entries],
    }


@app.get("/api/v1/agents/{agent_id:path}/read-from/{source_agent_id}/{entity_path:path}/{key}")
async def agent_read_from(
    agent_id: str,
    source_agent_id: str,
    entity_path: str,
    key: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Read a specific key from another agent's memory (cross-agent read)."""
    mem = _get_memory()
    entries = mem.search(entity_path=entity_path, agent_id=source_agent_id)
    matching = [e for e in entries if e.key == key]
    if not matching:
        return {"status": "not_found", "sourceAgentId": source_agent_id,
                "entityPath": entity_path, "key": key}
    return _entry_to_response(matching[0])


@app.get("/api/v1/agents/{agent_id:path}/activity")
async def agent_activity(
    agent_id: str,
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Timeline of writes, outcomes, reads, and other events for this agent."""
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
    agent_id: str,
    event_type: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Git-like event log for an agent — every write, outcome, and
    cross-agent read is recorded as an event on the agent's timeline."""
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

    agent_entity_weights: dict[str, dict[str, int]] = {}
    agent_totals: dict[str, int] = {}
    entity_totals: dict[str, int] = {}

    for e in entries:
        aid = e.provenance.agent_id
        if agent_id and aid != agent_id:
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


@app.get("/api/v1/briefing")
async def get_briefing(
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
    return {
        "digests": [d.model_dump(mode="json") for d in digests],
        "total": len(digests),
    }


@app.get("/api/v1/cortex/status")
async def cortex_status(
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

    logger.info("Starting AMFS HTTP server on %s:%d", args.host, args.port)
    uvicorn.run(
        "amfs_http.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
