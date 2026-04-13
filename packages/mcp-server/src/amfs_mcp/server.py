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
import time
from typing import Any

from fastmcp import FastMCP

from amfs import AgentMemory, MemoryType, OutcomeType
from amfs.config import load_config_or_default
from amfs_core.abc import AdapterABC
from amfs_core.models import AMFSConfig, LayerConfig
from amfs_core.quality import (
    HeuristicQualityEvaluator,
    MemoryQualityEvaluator,
    NoOpQualityEvaluator,
)

from amfs_mcp.agent_id import detect_agent_id, detect_platform

logger = logging.getLogger(__name__)

mcp = FastMCP(name="amfs")

# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

_quality_evaluator: MemoryQualityEvaluator | None = None


def _get_quality_evaluator() -> MemoryQualityEvaluator:
    """Return the singleton quality evaluator, respecting AMFS_QUALITY_FEEDBACK."""
    global _quality_evaluator
    if _quality_evaluator is None:
        enabled = os.environ.get("AMFS_QUALITY_FEEDBACK", "1") not in ("0", "false", "off")
        _quality_evaluator = HeuristicQualityEvaluator() if enabled else NoOpQualityEvaluator()
    return _quality_evaluator

# ---------------------------------------------------------------------------
# Identity-scoped memory management
#
# In stdio mode Cursor multiplexes all conversations through a single MCP
# process.  A naïve global AgentMemory singleton means set_identity() in
# one chat silently poisons writes from another.
#
# Fix: maintain one *adapter* (storage layer) shared across all identities,
# but a separate AgentMemory per identity — each with its own CausalTagger,
# ReadTracker, and session_id.  A staleness guard prevents rapid cross-
# conversation identity switches from clobbering each other.
# ---------------------------------------------------------------------------

_adapter: AdapterABC | None = None
_memories: dict[str, AgentMemory] = {}
_active_identity: str | None = None
_last_activity: float = 0.0

_IDENTITY_COOLDOWN_SECONDS = 30


def _get_adapter() -> AdapterABC:
    """Lazily initialise the shared storage adapter (one per process)."""
    global _adapter
    if _adapter is not None:
        return _adapter

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
            _adapter = HttpAdapter(base_url=http_url, api_key=api_key)
            return _adapter
        except ImportError:
            logger.warning(
                "AMFS_HTTP_URL is set but amfs-adapter-http is not installed. "
                "Falling back to local adapter. "
                "Install with: pip install amfs-adapter-http"
            )

    config = _resolve_config()
    from amfs.factory import create_adapter_from_config

    _adapter = create_adapter_from_config(config)
    return _adapter


def _get_memory() -> AgentMemory:
    """Return the AgentMemory for the currently active identity.

    Each identity gets its own AgentMemory instance (own CausalTagger,
    ReadTracker, session_id) while sharing the underlying adapter.
    """
    global _last_activity
    _last_activity = time.monotonic()

    name = _active_identity or detect_agent_id()

    if name in _memories:
        return _memories[name]

    adapter = _get_adapter()
    ttl_interval_str = os.environ.get("AMFS_TTL_SWEEP_INTERVAL")
    ttl_sweep_interval = float(ttl_interval_str) if ttl_interval_str else 300.0

    logger.info("AMFS MCP server — creating memory for agent_id=%s", name)
    mem = AgentMemory(
        agent_id=name,
        adapter=adapter,
        ttl_sweep_interval=ttl_sweep_interval,
    )
    _memories[name] = mem
    return mem


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
def amfs_set_identity(name: str, description: str | None = None) -> str:
    """Set the agent identity for this conversation. CALL THIS FIRST before any other AMFS tool.

    Without this, all your work is attributed to a generic default identity.

    MANDATORY WORKFLOW — follow this order every session:
    1. amfs_set_identity(name, description)  ← you are here
    2. amfs_briefing(entity_path="repo/module")  ← get compiled context before starting work
    3. Do your work, calling amfs_write() for important discoveries, decisions, and patterns
    4. amfs_record_context() for external tool results and user decisions as they happen
    5. amfs_commit_outcome("task-ref", "success|failure")  ← always do this when done

    Naming rules:
    - Use kebab-case role names that persist across conversations: "api-agent", "auth-debugger", "infra-agent"
    - BAD: "fix-button-color" (too specific), "agent-1" (meaningless)
    - If continuing previous work, reuse the same name to build on that agent's knowledge
    - The description should say what you're doing right now

    Entity path convention: use "repo/module" paths (e.g. "myapp/auth", "amfs/core-engine")

    Confidence guidelines: 1.0=verified fact, 0.7-0.9=high confidence, 0.4-0.6=hypothesis, <0.4=speculative

    Args:
        name: Short, descriptive name (e.g. "dashboard-fixer", "auth-debugger",
              "mcp-integration"). Use kebab-case.
        description: Optional one-line description of what this agent is doing.

    Example: amfs_set_identity("tenant-isolation-agent", "Fixing RLS propagation for Pro tenancy")
    """
    global _active_identity, _last_activity
    now = time.monotonic()

    # Idempotent: same name is always a no-op
    if _active_identity == name:
        _last_activity = now
        mem = _get_memory()
        return json.dumps({
            "identity": name,
            "session_id": mem.session_id,
            "status": "already_active",
        }, default=str)

    # Guard: reject if a *different* identity was recently active
    if _active_identity is not None:
        elapsed = now - _last_activity
        if elapsed < _IDENTITY_COOLDOWN_SECONDS:
            return json.dumps({
                "error": "identity_conflict",
                "current_identity": _active_identity,
                "requested_identity": name,
                "seconds_since_last_activity": round(elapsed, 1),
                "cooldown_seconds": _IDENTITY_COOLDOWN_SECONDS,
                "hint": (
                    "Another conversation recently set a different identity. "
                    "In stdio mode all conversations share one MCP process. "
                    f"Wait {_IDENTITY_COOLDOWN_SECONDS - elapsed:.0f}s or "
                    "continue using the current identity."
                ),
            })
        logger.info(
            "Identity switch: %s → %s (previous idle for %.1fs)",
            _active_identity, name, elapsed,
        )

    old_identity = _active_identity or detect_agent_id()
    _active_identity = name
    _last_activity = now

    mem = _get_memory()

    # Log identity change for debugging — but do NOT write to _system/agents.
    # Writing system metadata as regular memory entries pollutes agent stats
    # (entry counts, entity lists) and creates ghost agents on the dashboard.
    # Agent identity is already tracked via provenance on every real entry.
    logger.info(
        "Identity set: %s → %s (session=%s, platform=%s, description=%s)",
        old_identity, name, mem.session_id, detect_platform(), description or "",
    )

    result: dict[str, Any] = {
        "previous_identity": old_identity,
        "new_identity": name,
        "session_id": mem.session_id,
    }
    if description:
        result["description"] = description
    return json.dumps(result, default=str)


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

    WHEN TO WRITE (only write things that help a future agent):
    - After completing a task: key="task-summary-<desc>", value="what you did and why"
    - When discovering a pattern: key="pattern-<name>", add pattern_refs for cross-referencing
    - When finding a bug or risk: key="risk-<name>", use memory_type="belief" for hypotheses
    - When making a non-obvious decision: key="decision-<topic>", include rationale
    - When logging actions: key="action-<desc>", use memory_type="experience" (decays slower)

    DON'T write trivial info ("added a comment") — write things a colleague would need.
    Keep values concise but informative. Think of it as a note to a future agent.

    Args:
        entity_path: Hierarchical path like "repo/service" (e.g. "amfs/core-engine")
        key: Name for this piece of knowledge (e.g. "retry-pattern", "risk-signals")
        value: The knowledge to store — can be plain text or JSON string
        confidence: How confident you are (1.0=verified, 0.7-0.9=high, 0.4-0.6=hypothesis, <0.4=speculative)
        pattern_refs: Optional list of related pattern keys for cross-referencing
        memory_type: One of "fact" (default, normal decay), "belief" (decays faster), or "experience" (decays slower)
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

    evaluator = _get_quality_evaluator()
    quality_report = None
    try:
        existing_entries = mem.list(entity_path)
        existing_keys = [e.key for e in existing_entries if e.key != key]
    except Exception:
        existing_keys = []
    try:
        quality_report = evaluator.evaluate(
            parsed_value,
            entity_path=entity_path,
            key=key,
            confidence=confidence,
            memory_type=memory_type,
            pattern_refs=pattern_refs,
            existing_keys=existing_keys,
        )
    except Exception:
        logger.debug("Quality evaluation failed, skipping", exc_info=True)

    result: dict[str, Any] = {"entry": _serialize_entry(entry)}
    if quality_report is not None:
        result["quality"] = quality_report.model_dump(mode="json")
    return json.dumps(result, default=str)


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
    depth: int = 3,
) -> str:
    """Search across all memory entries with filters.

    Use this before starting work to find context about the entity you're
    modifying, or to check if another agent already solved a similar problem.

    When a Postgres adapter with tsvector support is configured, the query
    text is used for full-text search.  Otherwise falls back to Python
    substring matching on keys/values.

    Args:
        query: Optional text to search for (full-text when available, substring fallback)
        entity_path: Filter to a specific entity path
        min_confidence: Minimum confidence threshold (0.0-1.0)
        max_confidence: Maximum confidence threshold (0.0-1.0)
        agent_id: Filter to entries from a specific agent
        since: Optional ISO timestamp to filter entries written after this time
        pattern_ref: Filter to entries tagged with this pattern reference
        sort_by: Sort order — "confidence", "recency", or "version"
        limit: Maximum results to return
        depth: Tier depth (1=hot only, 2=hot+warm, 3=all tiers)

    Example: amfs_search(entity_path="checkout-service", min_confidence=0.5)
    """
    from datetime import datetime as dt

    mem = _get_memory()
    since_dt = dt.fromisoformat(since) if since else None

    results = mem.search(
        query=query,
        entity_path=entity_path,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        agent_id=agent_id,
        since=since_dt,
        pattern_ref=pattern_ref,
        sort_by=sort_by,
        limit=limit,
        depth=depth,
    )

    if not results:
        return json.dumps({
            "status": "empty",
            "count": 0,
            "message": "No entries matched your search filters.",
            "filters": {
                "query": query,
                "entity_path": entity_path,
                "agent_id": agent_id,
                "min_confidence": min_confidence,
            },
        })
    return json.dumps({
        "count": len(results),
        "entries": [_serialize_entry(e) for e in results],
    }, default=str)


@mcp.tool
def amfs_retrieve(
    query: str,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    semantic_weight: float = 0.5,
    recency_weight: float = 0.3,
    confidence_weight: float = 0.2,
    depth: int = 3,
) -> str:
    """Find the most relevant memories for a natural language query.

    Blends semantic similarity, recency, and confidence into a single
    ranked list.  Use this when you need to find memories by meaning,
    not exact key/value match.  Use amfs_search for structured filtering.

    Args:
        query: Natural language query describing what you're looking for
        entity_path: Optional entity path filter
        min_confidence: Minimum confidence threshold (0.0-1.0)
        limit: Maximum results to return
        semantic_weight: Weight for semantic similarity (0.0-1.0)
        recency_weight: Weight for recency (0.0-1.0)
        confidence_weight: Weight for confidence (0.0-1.0)
        depth: Tier depth (1=hot only, 2=hot+warm, 3=all tiers)

    Returns ranked results with score breakdowns showing how each
    signal contributed to the final ranking.
    """
    from amfs_core.models import RecallConfig

    mem = _get_memory()
    recall_config = RecallConfig(
        semantic_weight=semantic_weight,
        recency_weight=recency_weight,
        confidence_weight=confidence_weight,
    )

    results = mem.search(
        query=query,
        entity_path=entity_path,
        min_confidence=min_confidence,
        limit=limit,
        recall_config=recall_config,
        depth=depth,
    )

    serialized = []
    for scored in results:
        data = _serialize_entry(scored.entry)
        data["_score"] = round(scored.score, 4)
        data["_breakdown"] = {k: round(v, 4) for k, v in scored.breakdown.items()}
        serialized.append(data)

    if not serialized:
        return json.dumps({
            "status": "empty",
            "count": 0,
            "message": "No entries matched your query.",
            "query": query,
            "entity_path": entity_path,
        })
    return json.dumps({
        "count": len(serialized),
        "entries": serialized,
    }, default=str)


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
    if not entries:
        return json.dumps({
            "status": "empty",
            "count": 0,
            "entity_path": entity_path,
            "message": "No entries found." + (
                " Try without an entity_path filter to see all entries."
                if entity_path else ""
            ),
        })
    return json.dumps({
        "count": len(entries),
        "entries": [_serialize_entry(e) for e in entries],
    }, default=str)


@mcp.tool
def amfs_graph_neighbors(
    entity: str,
    relation: str | None = None,
    direction: str = "both",
    min_confidence: float = 0.0,
    depth: int = 1,
    limit: int = 50,
) -> str:
    """Explore the knowledge graph around an entity.

    Shows what services, agents, patterns, and outcomes are connected
    to the given entity, with relationship types and confidence scores.
    Use depth > 1 for multi-hop traversal.

    Args:
        entity: The entity to explore (e.g. "checkout-service/retry-pattern")
        relation: Optional filter by relation type (e.g. "references", "informed")
        direction: Edge direction — "outgoing", "incoming", or "both"
        min_confidence: Minimum edge confidence (0.0-1.0)
        depth: Traversal depth (1 = direct neighbors, 2+ = multi-hop)
        limit: Maximum edges to return
    """
    mem = _get_memory()
    edges = mem.graph_neighbors(
        entity,
        relation=relation,
        direction=direction,
        min_confidence=min_confidence,
        depth=depth,
        limit=limit,
    )
    if not edges:
        return json.dumps({
            "status": "empty",
            "count": 0,
            "entity": entity,
            "message": "No graph edges found for this entity.",
        })
    return json.dumps({
        "count": len(edges),
        "edges": [e.model_dump(mode="json") for e in edges],
    }, default=str)


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

    ALWAYS call this at the end of meaningful work. Without it, the decision
    trace (which memories were read, what contexts were gathered, what was
    decided) is lost when the session ends.

    This snapshots all reads, writes, recorded contexts, and decisions from
    this session into a persisted DecisionTrace. It also back-propagates
    confidence changes: entries linked to successes stabilize, entries linked
    to failures get flagged for review.

    Args:
        outcome_ref: Reference identifier (e.g. "INC-2047", "task-42", "PR-456")
        outcome_type: One of "success", "minor_failure", "failure", "critical_failure"

    Example: amfs_commit_outcome("task-42", "success")
    Example: amfs_commit_outcome("deploy-v2", "failure")
    """
    mem = _get_memory()

    type_map = {
        "success": OutcomeType.SUCCESS,
        "minor_failure": OutcomeType.MINOR_FAILURE,
        "failure": OutcomeType.FAILURE,
        "critical_failure": OutcomeType.CRITICAL_FAILURE,
        "clean_deploy": OutcomeType.CLEAN_DEPLOY,
        "regression": OutcomeType.REGRESSION,
        "p2_incident": OutcomeType.P2_INCIDENT,
        "p1_incident": OutcomeType.P1_INCIDENT,
    }

    otype = type_map.get(outcome_type.lower())
    if otype is None:
        valid = ", ".join(type_map.keys())
        return json.dumps({
            "error": f"Invalid outcome_type '{outcome_type}'. Must be one of: {valid}"
        })

    entries = mem.commit_outcome(outcome_ref, otype)
    trace = getattr(mem, "_last_trace", None)
    result: dict[str, Any] = {
        "outcome_ref": outcome_ref,
        "outcome_type": outcome_type,
        "affected_entries": len(entries),
        "entries": [_serialize_entry(e) for e in entries],
    }
    if trace and getattr(trace, "id", None):
        result["trace_id"] = trace.id
        result["causal_entries"] = len(trace.causal_entries)
        result["external_contexts"] = len(trace.external_contexts)
        result["session_duration_ms"] = trace.session_duration_ms
    return json.dumps(result, default=str)


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

    Call this AS IT HAPPENS — not all at the end. This preserves causal order.

    Use this for:
    - External tool results: amfs_record_context("git-log", "15 commits since last deploy", "git")
    - User decisions: amfs_record_context("user-decision", "User chose X over Y", "chat")
    - Architecture decisions: amfs_record_context("arch-decision", "Using Redis for cache", "analysis")
    - API responses: amfs_record_context("pagerduty", "3 SEV-1 in 24h", "PagerDuty API")

    The context is added to the causal chain persisted by amfs_commit_outcome(),
    making decision traces complete and explainable.

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
def amfs_list_traces(
    entity_path: str | None = None,
    agent_id: str | None = None,
    outcome_type: str | None = None,
    limit: int = 20,
) -> str:
    """Browse persisted decision traces from past sessions.

    Each trace captures the full causal chain: which memories were read,
    what external context was gathered, what decisions were made, and the
    final outcome. Use this to learn from past decisions before making
    similar ones.

    Args:
        entity_path: Filter to traces involving this entity
        agent_id: Filter to traces from a specific agent
        outcome_type: Filter by outcome type (success, failure, etc.)
        limit: Maximum traces to return (default 20)

    Example: amfs_list_traces(entity_path="checkout-service", limit=5)
    """
    mem = _get_memory()
    traces = mem._adapter.list_traces(
        entity_path=entity_path,
        agent_id=agent_id,
        outcome_type=outcome_type,
        limit=limit,
    )
    return json.dumps(
        [
            {
                "id": t.id,
                "agent_id": t.agent_id,
                "outcome_ref": t.outcome_ref,
                "outcome_type": t.outcome_type,
                "decision_summary": t.decision_summary,
                "causal_entries": len(t.causal_entries),
                "external_contexts": len(t.external_contexts),
                "session_duration_ms": t.session_duration_ms,
                "created_at": t.created_at,
            }
            for t in traces
        ],
        default=str,
    )


@mcp.tool
def amfs_get_trace(trace_id: str) -> str:
    """Retrieve a full decision trace by ID.

    Returns the complete causal chain: every memory read, external context,
    query, error, and the outcome. Use this to understand exactly what
    information drove a past decision.

    Args:
        trace_id: The trace ID (from amfs_list_traces or amfs_commit_outcome)

    Example: amfs_get_trace("abc123-def456")
    """
    mem = _get_memory()
    trace = mem._adapter.get_trace(trace_id)
    if trace is None:
        return json.dumps({"status": "not_found", "trace_id": trace_id})
    return json.dumps(trace.model_dump(mode="json"), default=str)


@mcp.tool
def amfs_briefing(
    entity_path: str | None = None,
    agent_id: str | None = None,
    limit: int = 10,
) -> str:
    """Get a compiled knowledge briefing — call this at the START of every session after setting identity.

    Returns pre-compiled digests from the Memory Cortex, ranked by relevance.
    This is your most important context-gathering step — it tells you what other
    agents know, recent risks, and confidence-ranked facts about the entity you're
    about to work on. Call this BEFORE reading code or making decisions.

    After briefing, use amfs_recall() for specific keys you remember, or
    amfs_search() for broader queries.

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
    if not digests:
        hint = (
            "No compiled briefings yet. "
            "Use amfs_search() or amfs_recall() to find existing memories, "
            "or amfs_write() to start building knowledge."
        )
        return json.dumps({"status": "empty", "message": hint})
    return json.dumps(
        [d.model_dump(mode="json") for d in digests],
        default=str,
    )


# ──────────────────────────────────────────────────────────────────────
# Timeline (git log)
# ──────────────────────────────────────────────────────────────────────


@mcp.tool
def amfs_timeline(
    limit: int = 50,
    event_type: str | None = None,
    since: str | None = None,
) -> str:
    """View recent events on this agent's timeline (git commit log).

    Every write, outcome, and cross-agent read is recorded as an event.
    Use this to see the history of what happened to your agent's memory.

    Args:
        limit: Max events to return (default 50)
        event_type: Filter by type (write, outcome, cross_agent_read, etc.)
        since: ISO timestamp to get events after

    Example: amfs_timeline(limit=20, event_type="write")
    """
    from datetime import datetime as dt
    mem = _get_memory()
    since_dt = dt.fromisoformat(since) if since else None
    events = mem._adapter.list_events(
        mem.agent_id,
        mem._config.namespace,
        event_type=event_type,
        since=since_dt,
        limit=limit,
    )
    return json.dumps({
        "events": [e.model_dump(mode="json") for e in events],
        "count": len(events),
    }, default=str)


@mcp.tool
def amfs_verify(
    entity_path: str | None = None,
) -> str:
    """Verify the content integrity of your memory store.

    Checks that stored content hashes match actual values, and that
    integrity chains link correctly across entry versions. Use this
    to detect corruption or tampering.

    Args:
        entity_path: Optional scope — verify only entries under this path.
                     If omitted, verifies all entries.

    Returns a report with total_checked, valid count, and any corrupted
    entries or chain breaks found.
    """
    mem = _get_memory()
    report = mem.verify(entity_path)
    return json.dumps(report, default=str)


@mcp.tool
def amfs_commit_batch(
    writes: list[dict[str, Any]],
    message: str = "",
) -> str:
    """Atomically write multiple memory entries as a single commit.

    All writes succeed or fail together — no partial updates. Each write
    in the batch is a dict with at least "entity_path", "key", "value",
    and optionally "confidence", "memory_type", "pattern_refs", "shared".

    Args:
        writes: List of write operations, each a dict with entity_path, key, value
        message: Commit message describing what these changes represent

    Example:
        amfs_commit_batch([
            {"entity_path": "myapp/auth", "key": "session-config", "value": "{}"},
            {"entity_path": "myapp/auth", "key": "token-ttl", "value": "3600"}
        ], message="Update auth configuration")
    """
    mem = _get_memory()
    with mem.transaction(message) as tx:
        for w in writes:
            ep = w["entity_path"]
            key = w["key"]
            raw_value = w.get("value", "")
            parsed_value: Any = raw_value
            if isinstance(raw_value, str):
                try:
                    parsed_value = json.loads(raw_value)
                except (json.JSONDecodeError, TypeError):
                    pass

            kwargs: dict[str, Any] = {}
            if "confidence" in w:
                kwargs["confidence"] = w["confidence"]
            if "memory_type" in w:
                type_map = {"fact": MemoryType.FACT, "belief": MemoryType.BELIEF, "experience": MemoryType.EXPERIENCE}
                kwargs["memory_type"] = type_map.get(str(w["memory_type"]).lower(), MemoryType.FACT)
            if "shared" in w:
                kwargs["shared"] = w["shared"]
            if "pattern_refs" in w:
                kwargs["pattern_refs"] = w["pattern_refs"]

            tx.write(ep, key, parsed_value, **kwargs)

    return json.dumps({
        "commit_id": tx.commit.id if tx.commit else None,
        "message": message,
        "entries_written": len(writes),
    }, default=str)


@mcp.tool
def amfs_commit_log(
    limit: int = 20,
) -> str:
    """View the commit log — atomic groups of writes with messages.

    Shows commits newest first, including the entries that were part
    of each commit.

    Args:
        limit: Maximum number of commits to return (default 20)
    """
    mem = _get_memory()
    commits = mem.commit_log(limit=limit)
    return json.dumps({
        "commits": [c.model_dump(mode="json") for c in commits],
        "count": len(commits),
    }, default=str)


@mcp.tool
def amfs_merge_base(
    commit_a: str,
    commit_b: str,
) -> str:
    """Find the common ancestor of two commits (merge-base).

    Given two commit IDs, walks the DAG to find the most recent
    commit that is an ancestor of both.

    Args:
        commit_a: First commit ID
        commit_b: Second commit ID
    """
    mem = _get_memory()
    ancestor = mem.common_ancestor(commit_a, commit_b)
    return json.dumps({
        "ancestor_commit_id": ancestor,
        "commit_a": commit_a,
        "commit_b": commit_b,
    })


@mcp.tool
def amfs_set_profile(
    description: str = "",
    tags: list[str] | None = None,
    auto_context_paths: list[str] | None = None,
) -> str:
    """Set your agent profile — description, tags, and auto-context paths.

    The profile helps other agents discover you and understand your role.

    Args:
        description: What this agent does (e.g. "Manages auth configuration")
        tags: Searchable tags (e.g. ["auth", "security", "backend"])
        auto_context_paths: Entity paths to auto-load on session start
    """
    mem = _get_memory()
    mem.set_profile(description, tags=tags, auto_context_paths=auto_context_paths)
    return json.dumps({"status": "ok", "agent_id": mem.agent_id})


@mcp.tool
def amfs_declare_capability(
    name: str,
    description: str = "",
    entity_paths: list[str] | None = None,
) -> str:
    """Declare a capability this agent has.

    Capabilities help other agents discover who knows about what.

    Args:
        name: Short capability name (e.g. "database-migrations")
        description: What this capability means
        entity_paths: Which entity paths this capability covers
    """
    mem = _get_memory()
    mem.declare_capability(name, description, entity_paths)
    return json.dumps({"status": "ok", "capability": name})


@mcp.tool
def amfs_discover_agents(
    capability: str | None = None,
    entity_path: str | None = None,
) -> str:
    """Discover other agents by capability or entity path.

    Use this to find which agents know about a topic or work on a codebase area.

    Args:
        capability: Filter by capability name
        entity_path: Filter by entity path relevance
    """
    mem = _get_memory()
    agents = mem.discover_agents(capability=capability, entity_path=entity_path)
    return json.dumps({
        "agents": [a.model_dump(mode="json") for a in agents],
        "count": len(agents),
    }, default=str)


@mcp.tool
def amfs_set_contract(
    entity_path: str,
    key_pattern: str = "*",
    min_confidence: float = 0.0,
    required_fields: list[str] | None = None,
    ttl_required: bool = False,
    description: str = "",
) -> str:
    """Set a memory contract — enforce schema/confidence expectations on writes.

    Contracts define what quality and structure is expected for memory
    entries matching a given entity_path and key pattern.

    Args:
        entity_path: The entity path this contract applies to
        key_pattern: Glob pattern for keys (default "*" matches all)
        min_confidence: Minimum confidence required
        required_fields: Fields that must be present in the value (if dict)
        ttl_required: Whether a TTL must be set
        description: Human-readable description of this contract
    """
    mem = _get_memory()
    mem.set_contracts([{
        "entity_path": entity_path,
        "key_pattern": key_pattern,
        "min_confidence": min_confidence,
        "required_fields": required_fields or [],
        "ttl_required": ttl_required,
        "description": description,
    }])
    return json.dumps({"status": "ok", "entity_path": entity_path, "key_pattern": key_pattern})


@mcp.tool
def amfs_diff(
    entity_path: str,
    key: str,
    old_version: int | None = None,
) -> str:
    """Compute a structural diff for a memory entry between versions.

    Shows field-level changes (add/remove/replace) with JSON Pointer paths.
    If old_version is not specified, diffs between the two most recent versions.

    Args:
        entity_path: The entity path (e.g. "repo/service")
        key: The key to diff
        old_version: Optional version to diff from (defaults to previous version)
    """
    mem = _get_memory()
    result = mem.diff(entity_path, key, old_version)
    return json.dumps(result, default=str)


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
