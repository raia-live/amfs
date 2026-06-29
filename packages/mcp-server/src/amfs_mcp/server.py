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
from pathlib import Path
from typing import Any

from fastmcp import Context, FastMCP

from amfs import AgentMemory, MemoryType, OutcomeType, SessionMetadata
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

_INSTRUCTIONS = """\
You have access to AMFS (Agent Memory File System) — persistent memory that survives across sessions, agents, and machines. Use it to build institutional knowledge over time.

## MANDATORY FIRST STEP: Set Your Identity

You MUST call `amfs_set_identity` before doing anything else. Without it, all your work is attributed to a generic default.

```
amfs_set_identity("<role-name>", "<one-line description of current task>", model="<your-model-name>")
```

Use kebab-case role/domain names that persist across conversations (e.g. "dashboard-agent", "api-agent"). If continuing work a previous agent started, use the same name. Always pass `model=` with your LLM model name (e.g. "claude-4-opus", "gpt-4o") — this is recorded in decision traces.

**Sticky identity**: Once set, your identity persists across sessions automatically. You only need to call `amfs_set_identity` if you want to change to a different role or update your description.

- `amfs_whoami()` — check which identity is active and where it came from
- `amfs_reset_identity()` — clear the sticky identity and revert to auto-detection

## Operations Cost

Every operation has a cost — be deliberate:
- **Read** (1 op): `amfs_read`, `amfs_search`, `amfs_list`, `amfs_recall`, `amfs_briefing`, `amfs_retrieve`
- **Write** (2 ops): `amfs_write`, `amfs_record_context`
- **Commit** (FREE): `amfs_commit_outcome` — always commit, never skip

## Workflow

1. **Get briefed**: `amfs_briefing(entity_path="<repo>/<module>")` — one briefing (1 op) replaces many individual reads. Always start here.
2. **Read & explore**: `amfs_read` for known keys, `amfs_search(query="...")` for text search, `amfs_retrieve` for semantic retrieval, `amfs_graph_neighbors` for related entities.
3. **Record decisions as they happen**: `amfs_record_context("user-decision", "User chose X over Y", source="chat")`. Only record meaningful decisions — not every micro-step.
4. **Write knowledge**: `amfs_write("<repo>/<module>", "task-summary-<desc>", "<what and why>")`. Use `memory_type="belief"` for hypotheses, `"experience"` for actions taken.
5. **Commit outcomes**: `amfs_commit_outcome("<ref>", "success")` after completing significant work. This is FREE and snapshots the full decision trace.

## What to Save (worth 2 ops)

- Decisions with rationale (why X over Y)
- Discovered patterns reusable across sessions
- Risks, bugs, or gotchas that would trip up future agents
- Task summaries after completing significant work
- External tool outputs worth preserving

## What NOT to Save (waste of ops)

- Trivial changes the IDE/VCS already tracks ("added a comment", "renamed variable")
- Anything recomputable from the current codebase in under 5 seconds
- Transient debug output or one-sentence entries (too low signal for 2 ops)

## Entity Naming

Use `{repo}/{service-or-module}` paths (e.g. `myapp/auth`, `amfs/core-engine`). Avoid generic paths like `project` or `code`.

## Key Patterns

- **Patterns**: `amfs_write(..., "pattern-<name>", "...", pattern_refs=["related-key"])`
- **Risks**: `amfs_write(..., "risk-<name>", "...", confidence=0.8, memory_type="belief")`
- **Cross-agent reads**: `amfs_read_from("<agent_id>", ...)` for tracked knowledge transfer.

## Confidence Scale

- 1.0 = verified fact | 0.7-0.9 = high confidence | 0.4-0.6 = hypothesis | < 0.4 = speculative
- Beliefs should have confidence < 0.9 — if you're sure enough for 0.9+, it's a fact.

## Avoid These

- Writing every file edit — only write knowledge, not changelogs
- Forgetting `amfs_commit_outcome` — it's free and preserves the decision trace
- Generic entity paths like `project` — use `{repo}/{module}`
- Setting confidence=1.0 on beliefs — beliefs are hypotheses, keep < 0.9
- Skipping `amfs_briefing` — one briefing replaces many individual reads

## Quality Feedback

`amfs_write` responses include a `quality` score. If below 0.8, review `issues` and rewrite with more detail. Use `pattern_refs` to link related entries.
"""

_toolset = os.environ.get("AMFS_TOOLSET", "all").lower()


def _create_server(toolset: str) -> FastMCP:
    """Build FastMCP with tag filtering, compatible with v2 and v3+."""
    if toolset != "all":
        try:
            server = FastMCP(name="amfs", instructions=_INSTRUCTIONS, include_tags={"core"})
        except TypeError:
            server = FastMCP(name="amfs", instructions=_INSTRUCTIONS)
            server.enable(tags={"core"}, only=True)
        logger.info(
            "AMFS toolset: core (15 essential tools). "
            "Set AMFS_TOOLSET=all to expose all 36 tools."
        )
    else:
        server = FastMCP(name="amfs", instructions=_INSTRUCTIONS)
        logger.info("AMFS toolset: all (36 tools)")
    return server


mcp = _create_server(_toolset)

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
#
# Sticky identity: the last identity set via amfs_set_identity() is
# persisted to ~/.amfs/.identity so it survives process restarts
# (e.g. Claude Desktop spawns a fresh MCP process per conversation).
# The user only needs to call set_identity once; subsequent sessions
# auto-restore it until the user explicitly changes it.
# ---------------------------------------------------------------------------

_adapter: AdapterABC | None = None
_memories: dict[str, AgentMemory] = {}
_active_identity: str | None = None
_last_activity: float = 0.0
_session_metadata: SessionMetadata | None = None

_IDENTITY_COOLDOWN_SECONDS = 30


def _identity_file() -> Path:
    """Path to the sticky identity file."""
    return Path.home() / ".amfs" / ".identity"


def _save_sticky_identity(name: str) -> None:
    """Persist the active identity to disk so it survives process restarts."""
    try:
        path = _identity_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    except OSError:
        logger.debug("Could not save sticky identity to %s", _identity_file(), exc_info=True)


def _load_sticky_identity() -> str | None:
    """Load the previously set identity from disk, if it exists."""
    try:
        path = _identity_file()
        if path.is_file():
            name = path.read_text(encoding="utf-8").strip()
            if name:
                return name
    except OSError:
        logger.debug("Could not read sticky identity from %s", _identity_file(), exc_info=True)
    return None


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
            if not api_key:
                logger.warning(
                    "AMFS_HTTP_URL is set but AMFS_API_KEY is empty or missing. "
                    "API requests will likely fail with 401 Unauthorized. "
                    "Set AMFS_API_KEY in your MCP server env configuration."
                )
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

    Resolution order: in-process global → sticky file → auto-detect.
    """
    global _active_identity, _last_activity
    _last_activity = time.monotonic()

    if _active_identity is None:
        sticky = _load_sticky_identity()
        if sticky:
            _active_identity = sticky
            logger.info("Restored sticky identity: %s", sticky)

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


def _enrich_metadata_from_context(ctx: Context | None) -> None:
    """Passively capture MCP client metadata from the FastMCP Context.

    Called on tool invocations that have a Context parameter.  Merges
    client_id and session_id from the MCP handshake into the module-level
    SessionMetadata without overwriting fields the agent explicitly set.
    """
    global _session_metadata
    if ctx is None:
        return
    try:
        client_id = ctx.client_id
        session_id = ctx.session_id if ctx.request_context else None
    except Exception:
        return

    if client_id is None and session_id is None:
        return

    if _session_metadata is None:
        _session_metadata = SessionMetadata()

    if client_id and not _session_metadata.mcp_client_id:
        _session_metadata.mcp_client_id = client_id
    if session_id and not _session_metadata.mcp_session_id:
        _session_metadata.mcp_session_id = session_id

    mem = _memories.get(_active_identity or "")
    if mem:
        mem.session_metadata = _session_metadata


def _serialize_entry(entry: Any) -> dict[str, Any]:
    """Convert a MemoryEntry to a JSON-safe dict for MCP responses."""
    data = entry.model_dump(mode="json")
    data.pop("embedding", None)
    return data


# ──────────────────────────────────────────────────────────────────────
# MCP Tools
# ──────────────────────────────────────────────────────────────────────


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": False, "idempotentHint": True})
def amfs_set_identity(
    name: str,
    description: str | None = None,
    model: str | None = None,
    client_name: str | None = None,
    tools_available: list[str] | None = None,
    ctx: Context | None = None,
) -> str:
    """Set the agent identity for this conversation. CALL THIS FIRST before any other AMFS tool.

    Without this, all your work is attributed to a generic default identity.

    **Sticky identity**: Once set, the identity is saved to disk and automatically
    restored in future sessions. You only need to call this again to change roles
    or update the description. If the identity is already active (from a previous
    session or the same session), this is a no-op.

    MANDATORY WORKFLOW — follow this order every session:
    1. amfs_set_identity(name, description, model)  ← you are here
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
        model: The LLM model you are running as (e.g. "claude-4-opus",
               "gpt-4o", "claude-3.5-sonnet"). You know your own model name — pass it here.
        client_name: The IDE or client platform (e.g. "cursor", "claude-code",
                     "windsurf", "cline"). Auto-detected when possible.
        tools_available: Optional list of tool names available to you in this session
                         (e.g. ["Shell", "Read", "Write", "Grep"]). Helps AMFS understand
                         what capabilities drove a decision.

    Example: amfs_set_identity("tenant-isolation-agent", "Fixing RLS propagation for Pro tenancy", model="claude-4-opus")
    """
    global _active_identity, _last_activity, _session_metadata
    now = time.monotonic()

    # Build / update session metadata from explicit params + MCP context
    if model or client_name or tools_available:
        if _session_metadata is None:
            _session_metadata = SessionMetadata()
        if model:
            _session_metadata.model = model
        if client_name:
            _session_metadata.client_name = client_name
        if tools_available:
            _session_metadata.tools_available = tools_available

    if _session_metadata is None:
        _session_metadata = SessionMetadata()
    _session_metadata.platform = detect_platform()

    _enrich_metadata_from_context(ctx)

    # Idempotent: same name is always a no-op (but still update metadata)
    if _active_identity == name:
        _last_activity = now
        mem = _get_memory()
        mem.session_metadata = _session_metadata
        return json.dumps({
            "identity": name,
            "session_id": mem.session_id,
            "status": "already_active",
            "sticky": True,
            "session_metadata": _session_metadata.model_dump(mode="json"),
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
    _save_sticky_identity(name)

    mem = _get_memory()
    mem.session_metadata = _session_metadata

    logger.info(
        "Identity set: %s → %s (session=%s, platform=%s, model=%s, description=%s)",
        old_identity, name, mem.session_id, detect_platform(),
        model or "unknown", description or "",
    )

    # Register / update the agent profile with the description and session
    # metadata so it shows up on the dashboard.  Fire-and-forget best-effort
    # — identity setting always succeeds even if profile update fails.
    try:
        from amfs_core.models import AgentProfile

        current_agent = mem._adapter.get_agent(name, namespace=mem.namespace)
        existing = current_agent.profile if current_agent else None
        profile = AgentProfile(
            description=description or (existing.description if existing else ""),
            default_branch=existing.default_branch if existing else "main",
            auto_context_paths=existing.auto_context_paths if existing else [],
            tags=existing.tags if existing else [],
            session_metadata=_session_metadata,
        )
        mem._adapter.update_agent_profile(name, profile, namespace=mem.namespace)
    except Exception:
        logger.debug("Could not update agent profile for %s", name, exc_info=True)

    result: dict[str, Any] = {
        "previous_identity": old_identity,
        "new_identity": name,
        "session_id": mem.session_id,
        "sticky": True,
        "hint": "Identity saved — it will be auto-restored in future sessions.",
        "session_metadata": _session_metadata.model_dump(mode="json"),
    }
    if description:
        result["description"] = description
    return json.dumps(result, default=str)


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
def amfs_whoami() -> str:
    """Show the current active identity and whether it was restored from disk.

    Useful for debugging identity issues or confirming which agent name
    is being used for reads/writes.
    """
    sticky = _load_sticky_identity()
    result: dict[str, Any] = {
        "active_identity": _active_identity,
        "sticky_identity": sticky,
        "detected_identity": detect_agent_id(),
        "effective_identity": _active_identity or sticky or detect_agent_id(),
        "platform": detect_platform(),
        "identity_file": str(_identity_file()),
    }
    if _session_metadata:
        result["session_metadata"] = _session_metadata.model_dump(mode="json")
    return json.dumps(result)


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": False})
def amfs_reset_identity() -> str:
    """Clear the sticky identity so it reverts to auto-detection.

    After calling this, the identity will fall back to environment-based
    detection (e.g. "cursor/username", "claude-code/username") until
    amfs_set_identity is called again.
    """
    global _active_identity
    old = _active_identity
    _active_identity = None
    try:
        path = _identity_file()
        if path.is_file():
            path.unlink()
    except OSError:
        logger.debug("Could not remove sticky identity file", exc_info=True)
    return json.dumps({
        "previous_identity": old,
        "current_identity": detect_agent_id(),
        "sticky_cleared": True,
    })


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True})
def amfs_write(
    entity_path: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    pattern_refs: list[str] | None = None,
    memory_type: str = "fact",
    artifact_refs: list[dict[str, Any]] | None = None,
    shared: bool = True,
    branch: str | None = None,
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
        branch: Optional branch name to write to (defaults to "main").

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

    write_kwargs: dict[str, Any] = dict(
        confidence=confidence,
        pattern_refs=pattern_refs,
        memory_type=mt,
        artifact_refs=parsed_artifact_refs,
        shared=shared,
    )
    if branch:
        write_kwargs["branch"] = branch

    entry = mem.write(entity_path, key, parsed_value, **write_kwargs)

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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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
        sort_by: Sort order — "confidence", "recency", "version", or "priority"
        limit: Maximum results to return
        depth: Tier depth (1=hot only, 2=hot+warm, 3=all tiers)

    Example: amfs_search(entity_path="checkout-service", min_confidence=0.5)
    """
    from datetime import datetime as dt

    mem = _get_memory()
    try:
        since_dt = dt.fromisoformat(since) if since else None
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid 'since' timestamp: {e}"})

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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
def amfs_stats() -> str:
    """Get aggregate statistics about the memory store.

    Returns total entries, entities, agents, confidence distribution,
    and time range. Useful for understanding the current state of
    shared knowledge.
    """
    mem = _get_memory()
    stats = mem.stats()
    return json.dumps(stats.model_dump(mode="json"), default=str)


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": False, "destructiveHint": False})
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
        outcome_type: One of "success", "minor_failure", "failure", "critical_failure",
            "clean_deploy", "regression", "p2_incident", "p1_incident"

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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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
    try:
        since_dt = dt.fromisoformat(since) if since else None
        until_dt = dt.fromisoformat(until) if until else None
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid timestamp: {e}"})

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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": False, "destructiveHint": False})
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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
    try:
        traces = mem._adapter.list_traces(
            entity_path=entity_path,
            agent_id=agent_id,
            outcome_type=outcome_type,
            limit=limit,
        )
    except Exception as e:
        return json.dumps({"error": f"Could not list traces: {e}"})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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
    try:
        trace = mem._adapter.get_trace(trace_id)
    except Exception as e:
        return json.dumps({"error": f"Could not retrieve trace: {e}"})
    if trace is None:
        return json.dumps({"status": "not_found", "trace_id": trace_id})
    return json.dumps(trace.model_dump(mode="json"), default=str)


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
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
    if digests:
        return json.dumps(
            [d.model_dump(mode="json") for d in digests],
            default=str,
        )

    # Fallback: if adapter.briefing() is unavailable (older amfs-adapter-http)
    # or Cortex has no compiled digests yet, synthesize a summary from entries.
    entries = mem.list(entity_path) if entity_path else []
    if entries:
        keys = [e.key for e in entries]
        agents = sorted({e.provenance.agent_id for e in entries})
        avg_conf = sum(e.confidence for e in entries) / len(entries)
        return json.dumps({
            "status": "synthesized",
            "message": (
                "No compiled Cortex digests available. "
                "Showing a summary built from existing memory entries."
            ),
            "entity_path": entity_path,
            "entry_count": len(entries),
            "avg_confidence": round(avg_conf, 2),
            "contributing_agents": agents,
            "keys": keys,
            "entries": [_serialize_entry(e) for e in entries[:limit]],
        }, default=str)

    hint = (
        "No compiled briefings yet. "
        "Use amfs_search() or amfs_recall() to find existing memories, "
        "or amfs_write() to start building knowledge."
    )
    return json.dumps({"status": "empty", "message": hint})


# ──────────────────────────────────────────────────────────────────────
# Timeline (git log)
# ──────────────────────────────────────────────────────────────────────


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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
    try:
        since_dt = dt.fromisoformat(since) if since else None
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid 'since' timestamp: {e}"})
    try:
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
    except Exception as e:
        return json.dumps({"error": f"Timeline unavailable: {e}"})


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": False, "destructiveHint": False})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": False, "destructiveHint": False})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": False, "destructiveHint": False})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": False, "destructiveHint": False})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True})
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


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True, "openWorldHint": True})
def amfs_export_training_data(
    entity_paths: list[str] | None = None,
    min_confidence: float = 0.7,
    format: str = "sft",
    limit: int = 100,
) -> str:
    """Export outcome-validated knowledge as a fine-tuning dataset.

    Generates training examples from decision traces in SFT, DPO, or
    Reward Model format. Only includes entries meeting the confidence
    threshold and linked to production outcomes.

    Requires AMFS_HTTP_URL to be set and the amfs-pro-api package
    installed on the server.

    Args:
        entity_paths: Filter to specific entities (exports all if omitted).
        min_confidence: Minimum confidence threshold (default 0.7).
        format: Export format — "sft", "dpo", or "reward_model".
        limit: Maximum examples to generate.

    Returns JSON with format, num_examples, and examples array.
    """
    base_url = os.environ.get("AMFS_HTTP_URL")
    if not base_url:
        return json.dumps({"error": "AMFS_HTTP_URL not set — export requires an HTTP API connection"})
    try:
        import httpx
        headers = {}
        api_key = os.environ.get("AMFS_API_KEY", "")
        if api_key:
            headers["X-AMFS-API-Key"] = api_key
        params: dict[str, Any] = {"format": format, "min_confidence": min_confidence, "limit": limit}
        if entity_paths:
            params["entity_path"] = entity_paths[0]
        resp = httpx.get(f"{base_url}/api/v1/pro/export", params=params, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            return json.dumps(resp.json(), default=str)
        return json.dumps({"error": f"Export endpoint returned {resp.status_code}", "detail": resp.text})
    except ImportError:
        return json.dumps({"error": "httpx not installed — install it for export support"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ──────────────────────────────────────────────────────────────────────
# Consolidation tools
# ──────────────────────────────────────────────────────────────────────


def _http_api_call(method: str, path: str, *, params: dict | None = None, body: dict | None = None) -> str:
    """Shared helper for tools that proxy to the AMFS HTTP API."""
    base_url = os.environ.get("AMFS_HTTP_URL")
    if not base_url:
        return json.dumps({"error": "AMFS_HTTP_URL not set — this feature requires an HTTP API connection"})
    try:
        import httpx
        headers: dict[str, str] = {}
        api_key = os.environ.get("AMFS_API_KEY", "")
        if api_key:
            headers["X-AMFS-API-Key"] = api_key
        if method == "GET":
            resp = httpx.get(f"{base_url}{path}", params=params, headers=headers, timeout=30.0)
        else:
            resp = httpx.post(f"{base_url}{path}", params=params, json=body or {}, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            return json.dumps(resp.json(), default=str)
        return json.dumps({"error": f"API returned {resp.status_code}", "detail": resp.text})
    except ImportError:
        return json.dumps({"error": "httpx not installed — required for HTTP API features"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True, "openWorldHint": True})
def amfs_consolidation_status() -> str:
    """Check the current memory consolidation status.

    Returns metrics about automatic memory consolidation: how many entries
    have been auto-archived, how many proposals are pending review, and
    the overall health of the consolidation system.
    """
    return _http_api_call("GET", "/api/v1/cortex/consolidation/status")


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True, "openWorldHint": True})
def amfs_consolidation_proposals(
    status: str = "pending",
    limit: int = 20,
) -> str:
    """List memory consolidation proposals.

    Consolidation proposals are created by the Cortex when it detects
    opportunities to compress, merge, or archive memory entries. Tier B
    proposals require human/agent review before merging.

    Args:
        status: Filter by proposal status — "pending", "approved", "rejected"
        limit: Maximum proposals to return

    Example: amfs_consolidation_proposals(status="pending")
    """
    return _http_api_call("GET", "/api/v1/cortex/consolidation/proposals",
                          params={"status": status, "limit": limit})


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": True, "openWorldHint": True})
def amfs_consolidation_candidates(
    limit: int = 20,
) -> str:
    """List entries that are candidates for consolidation.

    These are entries the Cortex has identified as potentially redundant,
    superseded, or compressible. Review these to understand what the
    consolidation system would act on.

    Args:
        limit: Maximum candidates to return
    """
    return _http_api_call("GET", "/api/v1/cortex/consolidation/candidates",
                          params={"limit": limit})


@mcp.tool(tags={"extended"}, annotations={"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True})
def amfs_consolidate(
    dry_run: bool = True,
) -> str:
    """Trigger a memory consolidation cycle.

    Runs the Cortex consolidation strategy which:
    - Tier A (automatic): archives superseded beliefs, prunes stale entries
    - Tier B (proposals): creates consolidation proposals for convergent
      knowledge and outcome rollups that need review

    Args:
        dry_run: If True (default), shows what would happen without making
            changes. Set to False to actually run consolidation.

    Example: amfs_consolidate(dry_run=True)
    """
    return _http_api_call("POST", "/api/v1/cortex/consolidate",
                          body={"dry_run": dry_run})


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
    parser.add_argument(
        "--toolset",
        choices=["core", "all"],
        default=None,
        help=(
            'Tool surface to expose: "all" (36 tools, default) '
            'or "core" (15 essential tools). Also settable via AMFS_TOOLSET env var. '
            "Use core for Claude Desktop to improve tool discovery reliability."
        ),
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

    Toolset is resolved in order:
    1. ``--toolset`` CLI flag
    2. ``AMFS_TOOLSET`` env var
    3. Default: ``core`` (15 essential tools)
    """
    args = _parse_args()

    if args.toolset and args.toolset != _toolset:
        logger.warning(
            "CLI --toolset=%s differs from AMFS_TOOLSET=%s. "
            "Restart with AMFS_TOOLSET=%s for reliable results.",
            args.toolset, _toolset, args.toolset,
        )

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
        logger.info(
            "Starting AMFS MCP server — transport=%s %s:%d%s toolset=%s",
            transport, host, port, path, _toolset,
        )
        mcp.run(transport=transport, host=host, port=port, path=path)
    else:
        logger.info(
            "Starting AMFS MCP server — transport=stdio toolset=%s",
            _toolset,
        )
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
