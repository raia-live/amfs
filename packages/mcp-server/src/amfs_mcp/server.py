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

from amfs import AgentMemory, MemoryType, OutcomeType
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

    ttl_interval_str = os.environ.get("AMFS_TTL_SWEEP_INTERVAL")
    ttl_sweep_interval = float(ttl_interval_str) if ttl_interval_str else 300.0

    logger.info("AMFS MCP server starting — agent_id=%s", agent_id)
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
    memory_type: str = "fact",
    artifact_refs: list[dict[str, Any]] | None = None,
    shared: bool = True,
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
        memory_type: One of "fact" (default), "belief", or "experience"
        artifact_refs: Optional list of external artifact references. Each dict
            should have "uri" (required), and optionally "media_type", "label",
            "size_bytes".
        shared: If True (default), other agents can read this entry. If False,
            only the writing agent can access it — useful for internal reasoning,
            scratchpad notes, or sensitive context.

    Example: amfs_write("checkout-service", "retry-pattern", '{"max_retries": 3}')
    Example private: amfs_write("checkout-service", "internal-notes", "...", shared=False)
    """
    from amfs_core.models import ArtifactRef

    mem = _get_memory()

    parsed_value: Any = value
    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass

    type_map = {"fact": MemoryType.FACT, "belief": MemoryType.BELIEF, "experience": MemoryType.EXPERIENCE}
    mt = type_map.get(memory_type.lower(), MemoryType.FACT)

    parsed_artifact_refs = [
        ArtifactRef.model_validate(r) for r in (artifact_refs or [])
    ]

    entry = mem.write(
        entity_path,
        key,
        parsed_value,
        confidence=confidence,
        pattern_refs=pattern_refs,
        memory_type=mt,
        artifact_refs=parsed_artifact_refs,
        shared=shared,
    )
    return json.dumps(_serialize_entry(entry), default=str)


@mcp.tool
def amfs_search(
    query: str | None = None,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    max_confidence: float | None = None,
    agent_id: str | None = None,
    since: str | None = None,
    pattern_ref: str | None = None,
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
        max_confidence: Maximum confidence threshold (0.0-1.0)
        agent_id: Filter to entries from a specific agent
        since: Optional ISO timestamp to filter entries written after this time
        pattern_ref: Filter to entries tagged with this pattern reference
        sort_by: Sort order — "confidence", "recency", or "version"
        limit: Maximum results to return

    Example: amfs_search(entity_path="checkout-service", min_confidence=0.5)
    """
    from datetime import datetime as dt

    mem = _get_memory()
    since_dt = dt.fromisoformat(since) if since else None

    results = mem.search(
        entity_path=entity_path,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        agent_id=agent_id,
        since=since_dt,
        pattern_ref=pattern_ref,
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


@mcp.tool
def amfs_history(
    entity_path: str,
    key: str,
    since: str | None = None,
    until: str | None = None,
) -> str:
    """Get the full version history of a memory entry over time.

    Returns all CoW versions of a key, showing how the value and confidence
    evolved. Useful for temporal reasoning — "how did this decision change?"

    Args:
        entity_path: Entity path (e.g. "checkout-service")
        key: Memory key to trace (e.g. "retry-pattern")
        since: Optional ISO timestamp to filter versions after this time
        until: Optional ISO timestamp to filter versions before this time

    Example: amfs_history("checkout-service", "retry-pattern")
    """
    from datetime import datetime as dt

    mem = _get_memory()
    since_dt = dt.fromisoformat(since) if since else None
    until_dt = dt.fromisoformat(until) if until else None

    versions = mem.history(entity_path, key, since=since_dt, until=until_dt)
    return json.dumps(
        {
            "entity_path": entity_path,
            "key": key,
            "version_count": len(versions),
            "versions": [_serialize_entry(e) for e in versions],
        },
        default=str,
    )


@mcp.tool
def amfs_record_context(
    label: str,
    summary: str,
    source: str = "",
) -> str:
    """Record external context that influenced this session's decisions.

    Call this after consulting an external tool, API, or data source.
    The context is added to the causal chain returned by amfs_explain(),
    making decision traces complete.

    Args:
        label: Short name for the context (e.g. "pagerduty-incidents", "git-log")
        summary: Brief summary of what was found
        source: Optional source identifier (e.g. "PagerDuty API", "git")

    Example: amfs_record_context("git-log", "15 commits since last deploy", "git")
    """
    mem = _get_memory()
    mem.record_context(label, summary, source=source or None)
    return json.dumps({"recorded": label, "source": source or None})


@mcp.tool
def amfs_recall(entity_path: str, key: str) -> str:
    """Recall YOUR OWN memory for a key — what do I know about this?

    Unlike amfs_read (which returns the latest version by any agent),
    amfs_recall returns only entries written by you. Use this to check
    your own knowledge before acting.

    Args:
        entity_path: Entity path (e.g. "checkout-service")
        key: Memory key (e.g. "retry-pattern")

    Example: amfs_recall("checkout-service", "retry-pattern")
    """
    mem = _get_memory()
    entry = mem.recall(entity_path, key)
    if entry is None:
        return json.dumps({"status": "not_found", "entity_path": entity_path, "key": key,
                           "hint": "You have not written this key. Try amfs_read() for shared knowledge."})
    return json.dumps(_serialize_entry(entry), default=str)


@mcp.tool
def amfs_my_entries(entity_path: str | None = None) -> str:
    """List all entries written by YOU — what's in my brain?

    Returns only entries authored by this agent. Optionally filter to
    a specific entity path.

    Args:
        entity_path: Optional entity path filter

    Example: amfs_my_entries("checkout-service")
    """
    mem = _get_memory()
    entries = mem.my_entries(entity_path)
    return json.dumps({
        "agent_id": mem.agent_id,
        "count": len(entries),
        "entries": [_serialize_entry(e) for e in entries],
    }, default=str)


@mcp.tool
def amfs_read_from(agent_id: str, entity_path: str, key: str) -> str:
    """Read a specific key from ANOTHER agent's memory.

    Use this when you want to explicitly learn from another agent's
    experience. The read is tracked for causal tracing.

    Args:
        agent_id: The agent whose memory to read from
        entity_path: Entity path (e.g. "checkout-service")
        key: Memory key (e.g. "retry-pattern")

    Example: amfs_read_from("deploy-agent", "checkout-service", "deploy-config")
    """
    mem = _get_memory()
    entry = mem.read_from(agent_id, entity_path, key)
    if entry is None:
        return json.dumps({"status": "not_found", "agent_id": agent_id,
                           "entity_path": entity_path, "key": key})
    return json.dumps(_serialize_entry(entry), default=str)


@mcp.tool
def amfs_cross_agent_reads() -> str:
    """Show which other agents' memory this agent has read.

    Returns a mapping of other agent IDs to the specific entity/key pairs
    read from them, with read counts. Use this to understand inter-agent
    communication and memory sharing relationships.

    Answers questions like:
    - "Which agents have I talked to?"
    - "What memory did I get from agent X?"
    - "Who wrote the knowledge I'm relying on?"

    Example response:
    {
      "agent_id": "review-agent",
      "reads_from": {
        "deploy-agent": [
          {"entity_path": "checkout-service", "key": "retry-pattern", "read_count": 3}
        ]
      },
      "agents_read_from": ["deploy-agent"]
    }
    """
    mem = _get_memory()
    cross_reads = mem.cross_agent_reads()
    return json.dumps({
        "agent_id": mem.agent_id,
        "reads_from": cross_reads,
        "agents_read_from": list(cross_reads.keys()),
        "total_cross_agent_reads": sum(
            r["read_count"] for reads in cross_reads.values() for r in reads
        ),
    }, default=str)


@mcp.tool
def amfs_explain(outcome_ref: str | None = None) -> str:
    """Explain the causal chain — which memories influenced this session's decisions.

    Shows every memory the agent read (in order) before committing an outcome.
    This is production-grounded explainability: not what the LLM inferred,
    but which stored knowledge actually drove the decision.

    Args:
        outcome_ref: Optional outcome reference to label the explanation

    Example: amfs_explain("deploy-v1.2.3")
    """
    mem = _get_memory()
    explanation = mem.explain(outcome_ref)
    return json.dumps(explanation, default=str)


@mcp.tool
def amfs_briefing(
    entity_path: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
) -> str:
    """Get a compiled knowledge briefing — what you should know right now.

    Returns pre-compiled digests from the Memory Cortex, ranked by relevance.
    Digests include entity summaries, agent brain briefs, and external source
    summaries. Much faster and more complete than manual search.

    Args:
        entity_path: Focus on this entity (e.g. "checkout-service")
        agent_id: Focus on this agent's context (defaults to current agent)
        limit: Max digests to return (default 10)

    Example: amfs_briefing(entity_path="checkout-service")
    """
    mem = _get_memory()
    digests = mem.briefing(
        entity_path=entity_path,
        agent_id=agent_id,
        limit=limit,
    )
    return json.dumps(
        [d.model_dump(mode="json") for d in digests],
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
