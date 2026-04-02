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
import json
import logging
import os
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from amfs import AgentMemory, MemoryType, OutcomeType
from amfs.config import load_config_or_default
from amfs_core.models import AMFSConfig, LayerConfig, MemoryEntry

from amfs_http.auth import verify_api_key
from amfs_http.models import ContextRequest, OutcomeRequest, SearchRequest, WriteRequest
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

_memory: AgentMemory | None = None
_sse_manager = SSEManager()


def _get_memory() -> AgentMemory:
    """Lazily initialise the shared AgentMemory singleton."""
    global _memory
    if _memory is not None:
        return _memory

    agent_id = os.environ.get("AMFS_AGENT_ID", "http-server")
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


# ──────────────────────────────────────────────────────────────────────
# Entries
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/entries/{entity_path:path}/{key}")
async def read_entry(
    entity_path: str,
    key: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    entry = mem.read(entity_path, key)
    if entry is None:
        return {"status": "not_found", "entity_path": entity_path, "key": key}
    return _entry_to_response(entry)


@app.post("/api/v1/entries")
async def write_entry(
    req: WriteRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    type_map = {
        "fact": MemoryType.FACT,
        "belief": MemoryType.BELIEF,
        "experience": MemoryType.EXPERIENCE,
    }
    mt = type_map.get(req.memory_type.lower(), MemoryType.FACT)

    entry = mem.write(
        req.entity_path,
        req.key,
        req.value,
        confidence=req.confidence,
        pattern_refs=req.pattern_refs or None,
        memory_type=mt,
    )
    _sse_manager.broadcast(entry)
    return _entry_to_response(entry)


@app.get("/api/v1/entries")
async def list_entries(
    entity_path: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    mem = _get_memory()
    entries = mem.list(entity_path)
    return [_entry_to_response(e) for e in entries]


# ──────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/search")
async def search_entries(
    req: SearchRequest,
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    mem = _get_memory()
    results = mem.search(
        entity_path=req.entity_path,
        min_confidence=req.min_confidence,
        agent_id=req.agent_id,
        sort_by=req.sort_by,
        limit=req.limit,
    )

    if req.query:
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
    "p1_incident": OutcomeType.P1_INCIDENT,
    "p2_incident": OutcomeType.P2_INCIDENT,
    "regression": OutcomeType.REGRESSION,
    "clean_deploy": OutcomeType.CLEAN_DEPLOY,
}


@app.post("/api/v1/outcomes")
async def commit_outcome(
    req: OutcomeRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    otype = _OUTCOME_TYPE_MAP.get(req.outcome_type.lower())
    if otype is None:
        valid = ", ".join(_OUTCOME_TYPE_MAP.keys())
        return {"error": f"Invalid outcome_type '{req.outcome_type}'. Must be one of: {valid}"}

    entries = mem.commit_outcome(
        req.outcome_ref,
        otype,
        causal_entry_keys=req.causal_entry_keys,
        causal_confidence=req.causal_confidence,
    )
    return {
        "outcome_ref": req.outcome_ref,
        "outcome_type": req.outcome_type,
        "affected_entries": len(entries),
        "entries": [_entry_to_response(e) for e in entries],
    }


@app.get("/api/v1/outcomes")
async def list_outcomes(
    entity_path: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    mem = _get_memory()
    since_dt = datetime.fromisoformat(since) if since else None
    records = mem._adapter.list_outcomes(
        entity_path=entity_path,
        since=since_dt,
        limit=limit,
    )
    return [r.model_dump(mode="json") for r in records]


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
# SSE Stream
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/stream")
async def stream(
    entity_path: str = Query("*"),
    _auth: str | None = Depends(verify_api_key),
) -> EventSourceResponse:
    return EventSourceResponse(_sse_manager.event_generator(entity_path))


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
    return parser.parse_args()


def main() -> None:
    """Run the AMFS HTTP server via uvicorn."""
    args = _parse_args()
    logger.info("Starting AMFS HTTP server on %s:%d", args.host, args.port)
    uvicorn.run(
        "amfs_http.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
