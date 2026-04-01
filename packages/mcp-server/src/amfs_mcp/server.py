"""AMFS MCP Server — exposes Agent Memory as MCP tools.

Designed for Cursor, Claude Code, and any MCP-compatible AI agent.
One AgentMemory instance persists for the lifetime of the server process,
giving agents a continuous session with automatic causal tracking.

Supports two transports:

- **stdio** (default) — for Cursor and Claude Code local MCP integration.
- **streamable-http** — for remote/team access over HTTP. Ideal when the
  AMFS server runs on a shared host and multiple agents connect remotely.

Transport selection via CLI or environment:

    amfs-mcp-server                           # stdio (default)
    amfs-mcp-server --transport http          # streamable-http on 0.0.0.0:8000/mcp
    amfs-mcp-server --transport http --port 9000 --host 127.0.0.1 --path /amfs

    AMFS_TRANSPORT=http amfs-mcp-server       # env-based selection
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from fastmcp import FastMCP

from amfs import AgentMemory, OutcomeType
from amfs.config import load_config_or_default
from amfs_core.models import AMFSConfig, LayerConfig

from amfs_mcp.agent_id import detect_agent_id

logger = logging.getLogger(__name__)

mcp = FastMCP(name="amfs")

_memory: AgentMemory | None = None


def _get_memory() -> AgentMemory:
    """Lazily initialise the shared AgentMemory singleton."""
    global _memory
    if _memory is not None:
        return _memory

    agent_id = detect_agent_id()
    config = _resolve_config()

    logger.info("AMFS MCP server starting — agent_id=%s", agent_id)
    _memory = AgentMemory(agent_id=agent_id, config_path=None, adapter=None)

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
    1. AMFS_POSTGRES_DSN env var → Postgres adapter
    2. AMFS_DATA_DIR env var → filesystem adapter at that path
    3. amfs.yaml discovery → load from file
    4. Default → filesystem adapter at .amfs/
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


def _serialize_entry(entry: Any) -> dict[str, Any]:
    """Convert a MemoryEntry to a JSON-safe dict for MCP responses."""
    data = entry.model_dump(mode="json")
    data.pop("embedding", None)
    return data


# ──────────────────────────────────────────────────────────────────────
# MCP Tools
# ──────────────────────────────────────────────────────────────────────


@mcp.tool
def amfs_read(entity_path: str, key: str) -> str:
    """Read a memory entry by entity path and key.

    Returns the full entry as JSON including value, confidence, provenance,
    and version. Returns a message if the entry does not exist.

    Example: amfs_read("checkout-service", "retry-pattern")
    """
    mem = _get_memory()
    entry = mem.read(entity_path, key)
    if entry is None:
        return json.dumps({"status": "not_found", "entity_path": entity_path, "key": key})
    return json.dumps(_serialize_entry(entry), default=str)


@mcp.tool
def amfs_write(
    entity_path: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    pattern_refs: list[str] | None = None,
) -> str:
    """Write a memory entry with automatic provenance tracking.

    The agent_id and session_id are auto-detected from the environment.
    Use this after completing a task, discovering a pattern, or recording
    a decision.

    Args:
        entity_path: Hierarchical path like "repo/service" (e.g. "amfs/core-engine")
        key: Name for this piece of knowledge (e.g. "retry-pattern", "risk-signals")
        value: The knowledge to store — can be plain text or JSON string
        confidence: How confident you are (0.0-1.0, default 1.0)
        pattern_refs: Optional list of related pattern keys for cross-referencing

    Example: amfs_write("checkout-service", "retry-pattern", '{"max_retries": 3}')
    """
    mem = _get_memory()

    parsed_value: Any = value
    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass

    entry = mem.write(
        entity_path,
        key,
        parsed_value,
        confidence=confidence,
        pattern_refs=pattern_refs,
    )
    return json.dumps(_serialize_entry(entry), default=str)


@mcp.tool
def amfs_search(
    query: str | None = None,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    agent_id: str | None = None,
    sort_by: str = "confidence",
    limit: int = 20,
) -> str:
    """Search across all memory entries with filters.

    Use this before starting work to find context about the entity you're
    modifying, or to check if another agent already solved a similar problem.

    Args:
        query: Optional text to match in keys/values (basic substring match)
        entity_path: Filter to a specific entity path
        min_confidence: Minimum confidence threshold (0.0-1.0)
        agent_id: Filter to entries from a specific agent
        sort_by: Sort order — "confidence", "recency", or "version"
        limit: Maximum results to return

    Example: amfs_search(entity_path="checkout-service", min_confidence=0.5)
    """
    mem = _get_memory()
    results = mem.search(
        entity_path=entity_path,
        min_confidence=min_confidence,
        agent_id=agent_id,
        sort_by=sort_by,
        limit=limit,
    )

    if query:
        query_lower = query.lower()
        results = [
            e
            for e in results
            if query_lower in e.key.lower()
            or query_lower in str(e.value).lower()
            or query_lower in e.entity_path.lower()
        ]

    return json.dumps([_serialize_entry(e) for e in results], default=str)


@mcp.tool
def amfs_list(entity_path: str | None = None) -> str:
    """List all current memory entries, optionally filtered to an entity path.

    Use to explore what knowledge exists for a given service or module.

    Args:
        entity_path: Optional entity path to filter (e.g. "checkout-service")

    Example: amfs_list("checkout-service")
    """
    mem = _get_memory()
    entries = mem.list(entity_path)
    return json.dumps([_serialize_entry(e) for e in entries], default=str)


@mcp.tool
def amfs_stats() -> str:
    """Get aggregate statistics about the memory store.

    Returns total entries, entities, agents, confidence distribution,
    and time range. Useful for understanding the current state of
    shared knowledge.
    """
    mem = _get_memory()
    stats = mem.stats()
    return json.dumps(stats.model_dump(mode="json"), default=str)


@mcp.tool
def amfs_commit_outcome(
    outcome_ref: str,
    outcome_type: str,
) -> str:
    """Record an outcome and auto-link it to everything read this session.

    Call this when something significant happens — a deployment succeeds,
    a bug is found, an incident occurs. The outcome automatically back-
    propagates confidence changes to all entries that influenced the decision.

    Args:
        outcome_ref: Reference identifier (e.g. "INC-2047", "deploy-v1.2.3", "PR-456")
        outcome_type: One of "p1_incident", "p2_incident", "regression", "clean_deploy"

    Example: amfs_commit_outcome("INC-2047", "p1_incident")
    """
    mem = _get_memory()

    type_map = {
        "p1_incident": OutcomeType.P1_INCIDENT,
        "p2_incident": OutcomeType.P2_INCIDENT,
        "regression": OutcomeType.REGRESSION,
        "clean_deploy": OutcomeType.CLEAN_DEPLOY,
    }

    otype = type_map.get(outcome_type.lower())
    if otype is None:
        valid = ", ".join(type_map.keys())
        return json.dumps({
            "error": f"Invalid outcome_type '{outcome_type}'. Must be one of: {valid}"
        })

    entries = mem.commit_outcome(outcome_ref, otype)
    return json.dumps(
        {
            "outcome_ref": outcome_ref,
            "outcome_type": outcome_type,
            "affected_entries": len(entries),
            "entries": [_serialize_entry(e) for e in entries],
        },
        default=str,
    )


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

_TRANSPORT_ALIASES: dict[str, str] = {
    "stdio": "stdio",
    "http": "streamable-http",
    "streamable-http": "streamable-http",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amfs-mcp-server",
        description="AMFS MCP Server — shared agent memory over MCP",
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "http", "streamable-http"],
        default=None,
        help='Transport to use: "stdio" (default) or "http" / "streamable-http"',
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to bind for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="URL path for HTTP transport (default: /mcp)",
    )
    return parser.parse_args()


def create_server() -> FastMCP:
    """Return the configured FastMCP server instance (for programmatic use)."""
    return mcp


def main() -> None:
    """Run the AMFS MCP server.

    Transport is resolved in order:
    1. ``--transport`` CLI flag
    2. ``AMFS_TRANSPORT`` env var
    3. Default: ``stdio``
    """
    args = _parse_args()

    raw_transport = (
        args.transport
        or os.environ.get("AMFS_TRANSPORT")
        or "stdio"
    )
    transport = _TRANSPORT_ALIASES.get(raw_transport, raw_transport)

    if transport == "streamable-http":
        host = args.host or os.environ.get("AMFS_HOST", "0.0.0.0")
        port = args.port or int(os.environ.get("AMFS_PORT", "8000"))
        path = args.path or os.environ.get("AMFS_PATH", "/mcp")
        logger.info("Starting AMFS MCP server — transport=%s %s:%d%s", transport, host, port, path)
        mcp.run(transport=transport, host=host, port=port, path=path)
    else:
        logger.info("Starting AMFS MCP server — transport=stdio")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
