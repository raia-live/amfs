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
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from amfs import AgentMemory, MemoryType, OutcomeType
from amfs.config import load_config_or_default
from amfs_core.models import AMFSConfig, LayerConfig, MemoryEntry, SearchQuery

from amfs_http.auth import verify_api_key
from amfs_http.models import (
    AddTeamMemberRequest,
    ContextRequest,
    CreateAPIKeyRequest,
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
    _memory = AgentMemory(
        agent_id=agent_id,
        config_path=None,
        adapter=None,
        ttl_sweep_interval=ttl_sweep_interval,
    )

    _memory._config = config
    from amfs.factory import create_adapter_from_config

    adapter = create_adapter_from_config(config)
    _memory._adapter = adapter
    _memory._engine._adapter = adapter
    _memory._propagator._adapter = adapter

    return _memory


try:
    from amfs_pro_api import mount_pro_api
    mount_pro_api(app, get_memory=_get_memory)
    logger.info("Pro API endpoints mounted (intelligence, extraction)")
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


# ──────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
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
    entity_path: str,
    key: str,
    branch: str = Query("main"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    entry = mem.read(entity_path, key, branch=branch)
    if entry is None:
        return {"status": "not_found", "entity_path": entity_path, "key": key}
    return _entry_to_response(entry)


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
    return _entry_to_response(entry)


@app.get("/api/v1/entries")
async def list_entries(
    entity_path: str | None = Query(None),
    branch: str = Query("main"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    entries = mem.list(entity_path, branch=branch)
    return {"entries": [_entry_to_response(e) for e in entries]}


# ──────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/search")
async def search_entries(
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
    )
    try:
        results = mem._adapter.search(sq, branch=branch)
    except TypeError:
        results = mem._adapter.search(sq)

    if req.query and not getattr(mem._adapter, "_has_search_tsv", False):
        query_lower = req.query.lower()
        results = [
            e
            for e in results
            if query_lower in e.key.lower()
            or query_lower in str(e.value).lower()
            or query_lower in e.entity_path.lower()
        ]

    return [_entry_to_response(e) for e in results]


# ──────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/stats")
async def get_stats(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    stats = mem.stats()
    return json.loads(json.dumps(stats.model_dump(mode="json"), default=str))


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

        imm = ImmutableDecisionTrace(
            agent_id=mem.agent_id,
            session_id=session_id,
            sequence_number=seq,
            outcome_ref=oss_trace.outcome_ref,
            outcome_type=oss_trace.outcome_type,
            decision_summary=getattr(oss_trace, "decision_summary", None),
            causal_entries=causal,
            external_contexts=contexts,
            created_at=datetime.now(timezone.utc),
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


@app.get("/api/v1/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    trace = mem._adapter.get_trace(trace_id)
    if trace is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Trace not found"}, status_code=404)
    return trace.model_dump(mode="json")


@app.post("/api/v1/traces/{trace_id}/explain")
async def explain_trace(
    trace_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from fastapi.responses import JSONResponse

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

    return {
        "requestsToday": 0,
        "requestsThisMonth": 0,
        "peakRpm": 0,
        "avgLatencyMs": 0,
        "quotas": [
            {"label": "Memory entries", "current": st.total_entries, "limit": 0},
            {"label": "Decision traces", "current": len(outcomes), "limit": 0},
            {"label": "API keys", "current": 0, "limit": 0},
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
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all known agents with entry counts and last activity."""
    mem = _get_memory()
    entries = mem.list()
    agent_data: dict[str, dict[str, Any]] = {}
    for e in entries:
        aid = e.provenance.agent_id
        if aid not in agent_data:
            agent_data[aid] = {
                "agent_id": aid,
                "entries_written": 0,
                "entities_touched": set(),
                "last_active": e.provenance.written_at,
            }
        agent_data[aid]["entries_written"] += 1
        agent_data[aid]["entities_touched"].add(e.entity_path)
        if e.provenance.written_at > agent_data[aid]["last_active"]:
            agent_data[aid]["last_active"] = e.provenance.written_at

    agents = []
    for ad in sorted(agent_data.values(), key=lambda x: x["entries_written"], reverse=True):
        agents.append({
            "agentId": ad["agent_id"],
            "entriesWritten": ad["entries_written"],
            "entitiesTouched": len(ad["entities_touched"]),
            "lastActive": ad["last_active"].isoformat() if ad["last_active"] else None,
        })
    return {"agents": agents}


@app.get("/api/v1/agents/{agent_id}/memory-graph")
async def agent_memory_graph(
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """All entries written by or read by this agent, grouped by entity."""
    mem = _get_memory()
    entries = mem.list()
    traces = mem._adapter.list_traces(agent_id=agent_id, limit=10000)

    written_by_agent = [e for e in entries if e.provenance.agent_id == agent_id]
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

    read_entities: dict[str, dict[str, int]] = {}
    for t in traces:
        for ce in t.causal_entries:
            ep = ce.entity_path
            key = ce.key
            if ep not in read_entities:
                read_entities[ep] = {}
            read_entities[ep][key] = read_entities[ep].get(key, 0) + 1

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
        "crossAgentReads": cross_agent_reads,
    }


@app.get("/api/v1/agents/{agent_id}/cross-agent-reads")
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


@app.get("/api/v1/agents/{agent_id}/recall/{entity_path:path}/{key}")
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


@app.get("/api/v1/agents/{agent_id}/entries")
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


@app.get("/api/v1/agents/{agent_id}/read-from/{source_agent_id}/{entity_path:path}/{key}")
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


@app.get("/api/v1/agents/{agent_id}/activity")
async def agent_activity(
    agent_id: str,
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Timeline of writes and outcomes for this agent."""
    mem = _get_memory()
    entries = [e for e in mem.list() if e.provenance.agent_id == agent_id]
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

    timeline = sorted(writes + outcomes, key=lambda x: x["timestamp"], reverse=True)[:limit]
    return {"agentId": agent_id, "timeline": timeline}


@app.get("/api/v1/agents/{agent_id}/timeline")
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
# Admin — API Keys
# ──────────────────────────────────────────────────────────────────────


def _get_db_pool():
    """Return the underlying database pool if using the Postgres adapter."""
    mem = _get_memory()
    adapter = mem._adapter
    if hasattr(adapter, "_pool"):
        return adapter._pool
    return None


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
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO amfs_audit_log
                           (actor_type, actor_name, action, resource, ip_address)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (actor_type, actor_name, action, resource, ip_address),
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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, name, prefix, key_type, active, scopes,
                          rate_limit_rpm, last_used, created_at, expires_at
                   FROM amfs_api_keys ORDER BY created_at DESC"""
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

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO amfs_api_keys
                       (name, key_hash, prefix, key_type, scopes, rate_limit_rpm, expires_at)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                   RETURNING id, created_at""",
                (
                    req.name,
                    key_hash,
                    prefix,
                    req.key_type,
                    json.dumps(req.scopes),
                    req.rate_limit_rpm,
                    req.expires_at,
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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE amfs_api_keys SET active = FALSE WHERE id = %s::uuid RETURNING id, name",
                (key_id,),
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

    conditions = ["1=1"]
    params: list[Any] = []

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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.id, t.name, t.slug, t.description,
                          t.created_at, t.updated_at,
                          COUNT(m.id) AS member_count
                   FROM amfs_teams t
                   LEFT JOIN amfs_team_members m ON m.team_id = t.id
                   GROUP BY t.id
                   ORDER BY t.created_at DESC"""
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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO amfs_teams (name, slug, description)
                   VALUES (%s, %s, %s)
                   RETURNING id, created_at, updated_at""",
                (req.name, req.slug, req.description),
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
    params.append(team_id)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE amfs_teams SET {', '.join(updates)}
                    WHERE id = %s::uuid
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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM amfs_teams WHERE id = %s::uuid RETURNING id, slug",
                (team_id,),
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
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"members": []}
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, email, display_name, role,
                          invited_at, accepted_at, created_at
                   FROM amfs_team_members
                   WHERE team_id = %s::uuid
                   ORDER BY created_at""",
                (team_id,),
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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO amfs_team_members (team_id, email, display_name, role)
                   VALUES (%s::uuid, %s, %s, %s)
                   RETURNING id, invited_at, created_at""",
                (team_id, req.email, req.display_name, req.role),
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

    params.extend([member_id, team_id])

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE amfs_team_members SET {', '.join(updates)}
                    WHERE id = %s::uuid AND team_id = %s::uuid
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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM amfs_team_members
                   WHERE id = %s::uuid AND team_id = %s::uuid
                   RETURNING id, email""",
                (member_id, team_id),
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


# ──────────────────────────────────────────────────────────────────────
# Admin — Pattern Detection (Pro)
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/patterns")
async def list_detected_patterns(
    pattern_type: str | None = Query(None),
    severity: str | None = Query(None),
    resolved: bool | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List previously detected patterns from the database."""
    pool = _get_db_pool()
    if pool is None:
        return {"patterns": []}

    conditions = ["1=1"]
    params: list[Any] = []

    if pattern_type is not None:
        conditions.append("pattern_type = %s")
        params.append(pattern_type)
    if severity is not None:
        conditions.append("severity = %s")
        params.append(severity)
    if resolved is not None:
        conditions.append("resolved = %s")
        params.append(resolved)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, pattern_type, severity, entity_path, description,
               details, resolved, detected_at, resolved_at
        FROM amfs_detected_patterns
        WHERE {where}
        ORDER BY detected_at DESC
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
                "entityPath": row["entity_path"],
                "description": row["description"],
                "details": row["details"] or {},
                "resolved": row["resolved"],
                "detectedAt": row["detected_at"].isoformat(),
                "resolvedAt": row["resolved_at"].isoformat() if row["resolved_at"] else None,
            }
            for row in rows
        ]
    }


@app.post("/api/v1/admin/patterns/scan")
async def run_pattern_scan(
    req: RunPatternDetectionRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run the pattern detector and persist results."""
    try:
        from amfs_patterns import PatternDetector
    except ImportError:
        return {"error": "amfs-patterns package not installed"}

    mem = _get_memory()
    entries = mem.list(req.entity_path)
    outcomes = mem._adapter.list_outcomes(entity_path=req.entity_path, limit=10000)

    detector = PatternDetector(
        incident_threshold=req.incident_threshold,
        stale_days=req.stale_days,
        hot_entity_stddev=req.hot_entity_stddev,
        drift_stddev=req.drift_stddev,
    )
    report = detector.analyze(entries, outcome_data=outcomes)

    pool = _get_db_pool()
    persisted = 0
    if pool is not None:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for p in report.patterns:
                    cur.execute(
                        """INSERT INTO amfs_detected_patterns
                               (pattern_type, severity, entity_path, description,
                                details, detected_at)
                           VALUES (%s, %s, %s, %s, %s::jsonb, %s)""",
                        (
                            p.pattern_type,
                            p.severity,
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
                "entityPath": p.entity_path,
                "description": p.description,
                "details": p.details,
                "detectedAt": p.detected_at.isoformat(),
            }
            for p in report.patterns
        ],
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
    from fastapi.responses import JSONResponse

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
    limit_agents: int = Query(30, ge=1, le=200),
    limit_entities: int = Query(30, ge=1, le=200),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Build an agent×entity expertise heatmap.

    Returns a list of agents, entities, and cells with scores derived
    from write counts. The dashboard renders this as a heatmap table.
    """
    mem = _get_memory()
    entries = mem.list()

    agent_entity_weights: dict[str, dict[str, int]] = {}
    agent_totals: dict[str, int] = {}
    entity_totals: dict[str, int] = {}

    for e in entries:
        aid = e.provenance.agent_id
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
    from amfs_core.models import GraphEdge

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
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all compiled digests."""
    try:
        from amfs_postgres.adapter import PostgresAdapter
        from amfs_core.models import DigestType

        mem = _get_memory()
        adapter = mem._adapter
        if isinstance(adapter, PostgresAdapter):
            dt = DigestType(digest_type) if digest_type else None
            digests = adapter.list_digests(digest_type=dt)
            return {
                "digests": [d.model_dump(mode="json") for d in digests],
                "total": len(digests),
            }
    except (ImportError, ValueError, Exception):
        pass

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
                _cortex_worker = CortexWorker(
                    dsn=dsn,
                    compiler=compiler,
                    use_advisory_lock=False,
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
