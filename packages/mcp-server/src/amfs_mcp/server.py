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
from collections.abc import Iterable
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
SenseLab is your persistent memory — it remembers ANYTHING the user wants to keep, across every session, tool, and machine. That includes **personal facts and preferences** (food, people, important dates, how they like things done) AND **work context** (decisions, patterns, code, runbooks). It is general-purpose memory for the whole person, NOT just a coding-agent tool.

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
- **Free**: `amfs_record_action` — costs nothing, so record every consequential action
- **Commit** (FREE): `amfs_commit_outcome` — always commit, never skip

## Finding / recalling memories (READ THIS)

When the user asks you to find, recall, look up, remember, or check what's stored
about ANYTHING — or you just want to see if relevant memory exists — call
`amfs_retrieve(query="<the user's words>")`. That's it. You do NOT need to know
where it's stored: **entity_path and key are optional**. `amfs_retrieve` searches
by meaning across everything you can see (all of your agents, all entities), so a
plain phrase like "my favorite ice cream" or "the deploy runbook" is enough.

**RECALL-FIRST RULE (do not skip):** If the user asks what you know / remember /
have saved about anything — INCLUDING personal facts and preferences ("what food
do I like?") — you MUST call `amfs_retrieve` BEFORE answering. Never tell the user
you have no memory of something without checking first.

**Common misconceptions to avoid:**
- SenseLab stores **personal facts and preferences** (food, people, plans, habits)
  just as much as code/decisions. NEVER tell the user "SenseLab isn't a preferences
  database", "it only stores work/agent memories", or "that's not what it's for".
  It stores whatever was saved — so just search and see.
- `amfs_retrieve` is full free-text *semantic* search. Never tell the user memory
  "only works by exact key" or that you "need an entity_path/key" — you don't.
- Don't stop at `amfs_read`/`amfs_recall` returning nothing: those need an EXACT
  key. A miss there means "try `amfs_retrieve`", not "nothing is stored".

- Reach for `amfs_retrieve` FIRST for any natural-language lookup.
- Only use `amfs_read(entity_path, key)` when you ALREADY know the exact path AND
  key — never guess them. If you're guessing, use `amfs_retrieve` instead.
- Use `amfs_search` when you want structured filters (by agent, confidence, date,
  pattern) or an exact keyword; it also accepts a bare `query`.

## Workflow

1. **Get briefed**: `amfs_briefing(entity_path="<repo>/<module>")` — one briefing (1 op) replaces many individual reads. Always start here.
2. **Recall**: `amfs_retrieve(query="...")` to find anything by meaning (no path/key needed); `amfs_search(query="...")` for filtered/keyword search; `amfs_read(path, key)` only when you know the exact coordinates; `amfs_graph_neighbors` for related entities.
3. **Record decisions as they happen**: `amfs_record_context("user-decision", "User chose X over Y", source="chat")`. Only record meaningful decisions — not every micro-step.
4. **Record actions as you take them**: `amfs_record_action("deploy_rollback", {"service": "checkout"})` right after a consequential action — deploy, rollback, refund, file edit, PR. Record failed ones too with `success=False`. AMFS sees only its own tools, so your other tools are invisible unless you record them.
5. **Write knowledge**: `amfs_write("<repo>/<module>", "task-summary-<desc>", "<what and why>")`. Use `memory_type="belief"` for hypotheses, `"experience"` for actions taken.
6. **Commit outcomes**: `amfs_commit_outcome("<ref>", "success", task_input="<the request that started this>")` after completing significant work. This is FREE and snapshots the full decision trace. Pass `task_input` whenever you have it — without it the trace records what you decided but not what you were asked.

## What to Save (worth 2 ops)

- **Personal facts & preferences** the user shares ("I like pizza", names of people,
  important dates, how they like things done) — save whenever they state one or say "remember…"
- Decisions with rationale (why X over Y)
- Discovered patterns reusable across sessions
- Risks, bugs, or gotchas that would trip up future agents
- Task summaries after completing significant work
- External tool outputs worth preserving

## What NOT to Save (waste of ops)

- Trivial changes the IDE/VCS already tracks ("added a comment", "renamed variable")
- Anything trivially recomputable from the current context in under 5 seconds
- Transient debug output or one-sentence entries (too low signal for 2 ops)

## Entity Naming

Use short hierarchical paths. For work: `{repo}/{service-or-module}` (e.g. `myapp/auth`, `amfs/core-engine`). For personal memory: `user/preferences`, `user/people`, `user/plans`, etc. Avoid generic paths like `project` or `code`. When recalling you never need to remember the path — `amfs_retrieve` searches across all of them by meaning.

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

# Surfaced in the amfs_set_identity response so the agent (and, through it, the
# user) immediately learns what SenseLab makes possible now that it's connected.
# This is the reliable channel: agents always read tool output, even when they
# skip the server `instructions` block.
_GETTING_STARTED = {
    "you_are_connected": "SenseLab is a shared memory across all your tools and agents. "
    "Anything saved from one tool (e.g. Cursor) is recallable from another (e.g. Claude Desktop) on the same account.",
    "recall_anything": "To find what's stored — including personal facts and preferences — "
    "call amfs_retrieve(query=\"<the user's words>\"). No entity_path/key needed; it searches by meaning.",
    "remember_something": "To save something for the future, call "
    "amfs_write(entity_path, key, value). Use it whenever the user says 'remember…' or shares a durable fact/decision.",
    "get_briefed": "Before working in a codebase, call amfs_briefing(entity_path=\"repo/module\") for compiled context.",
    "recall_first_rule": "When the user asks what you know/remember/have saved about ANYTHING, "
    "call amfs_retrieve BEFORE answering — never say you have no memory of it without checking.",
}


def _default_entity_path() -> str | None:
    """The home entity_path for this environment, from ``AMFS_ENTITY_PATH``.

    Set by hosts that spin up disposable, single-purpose environments (e.g. an
    ephemeral sandbox or a CI job) so a fresh process auto-hydrates the right
    memory without the agent having to guess a path. When set, it becomes the
    default for ``amfs_briefing`` and is advertised to the agent through the
    server instructions and the ``amfs_set_identity`` response.
    """
    ep = os.environ.get("AMFS_ENTITY_PATH")
    return ep.strip() if ep and ep.strip() else None


def _build_instructions() -> str:
    """Return the server instructions, appending the bound entity when set."""
    base = _INSTRUCTIONS
    ep = _default_entity_path()
    if ep:
        base += (
            f"\n\n## This environment is bound to `{ep}`\n\n"
            f"`AMFS_ENTITY_PATH` is set, so this machine/session has a home "
            f"entity: **{ep}**. Right after `amfs_set_identity`, call "
            f"`amfs_briefing(entity_path=\"{ep}\")` to load everything prior "
            f"sessions here learned, and default your reads/writes to this "
            f"path unless the task clearly concerns another entity. Calling "
            f"`amfs_briefing()` with no argument also defaults to `{ep}` here.\n"
        )
    return base


def _getting_started() -> dict[str, str]:
    """The getting-started block, with the bound entity path baked in when set."""
    gs = dict(_GETTING_STARTED)
    ep = _default_entity_path()
    if ep:
        gs["get_briefed"] = (
            f'This environment is bound to entity_path="{ep}". Call '
            f'amfs_briefing(entity_path="{ep}") now to load what prior '
            f"sessions here learned before doing any work."
        )
        gs["bound_entity_path"] = ep
    return gs


_toolset = os.environ.get("AMFS_TOOLSET", "all").lower()


def _create_server(toolset: str) -> FastMCP:
    """Build FastMCP with tag filtering, compatible with v2 and v3+."""
    instructions = _build_instructions()
    if toolset != "all":
        try:
            server = FastMCP(name="amfs", instructions=instructions, include_tags={"core"})
        except TypeError:
            server = FastMCP(name="amfs", instructions=instructions)
            server.enable(tags={"core"}, only=True)
        logger.info(
            "AMFS toolset: core (essential tools incl. amfs_retrieve). "
            "Set AMFS_TOOLSET=all to expose all 36 tools."
        )
    else:
        server = FastMCP(name="amfs", instructions=instructions)
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


def _legacy_identity_file() -> Path:
    """Path to the pre-per-client shared sticky identity file.

    This machine-global file was shared by EVERY MCP client, so a work
    identity set in one client (e.g. Cursor) leaked into another (e.g. Claude
    Desktop), overriding its auto-detected identity and hiding the user's real
    memories. We no longer read it for auto-restore; kept only so it can be
    cleaned up on the next save/reset.
    """
    return Path.home() / ".amfs" / ".identity"


def _identity_file() -> Path:
    """Path to the sticky identity file, scoped per client.

    Keyed by the detected platform (cursor / claude-code / cli) so each client
    restores its OWN last identity and Cursor work identities never leak into
    Claude Desktop (which detects as "cli") or vice versa. Note: distinct
    non-IDE clients (Claude Desktop, headless CLI) all map to "cli" and share
    one file — acceptable, since the reported leak is across IDE vs desktop.
    """
    return Path.home() / ".amfs" / f".identity-{detect_platform()}"


def _save_sticky_identity(name: str) -> None:
    """Persist the active identity to disk so it survives process restarts."""
    try:
        path = _identity_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    except OSError:
        logger.debug("Could not save sticky identity to %s", _identity_file(), exc_info=True)
    # Best-effort: retire the legacy machine-global file so stale cross-client
    # state can't be restored by an older client on this machine.
    try:
        legacy = _legacy_identity_file()
        if legacy.is_file():
            legacy.unlink()
    except OSError:
        logger.debug("Could not remove legacy sticky identity file", exc_info=True)


def _load_sticky_identity() -> str | None:
    """Load the previously set identity for THIS client, if it exists.

    Only reads the per-client file — never the legacy shared file — so a work
    identity set in another client cannot bleed into this session.
    """
    try:
        path = _identity_file()
        if path.is_file():
            name = path.read_text(encoding="utf-8").strip()
            if name:
                return name
    except OSError:
        logger.debug("Could not read sticky identity from %s", _identity_file(), exc_info=True)
    return None


def _call_compat(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn(**kwargs)`` resiliently across amfs-mcp / amfs SDK version skew.

    The hosted connector can run a newer amfs-mcp (whose read tools pass
    ``include_artifacts``) against an older installed amfs SDK whose
    ``retrieve``/``search`` predate that parameter — which otherwise raises
    ``TypeError: ... unexpected keyword argument 'include_artifacts'`` and
    hard-breaks recall on clients like Claude Desktop. Retry once without the
    offending kwarg so retrieval degrades gracefully instead of failing.
    """
    try:
        return fn(**kwargs)
    except TypeError as e:
        if "include_artifacts" in kwargs and "include_artifacts" in str(e):
            kwargs.pop("include_artifacts", None)
            return fn(**kwargs)
        raise


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


# A stored source file can run past 50K characters, so a 20-result list response
# can reach ~87K tokens — more context than the memories it recalls are worth.
# List responses carry a preview instead; the exact-read tools stay whole.
LIST_VALUE_CHAR_LIMIT = 2000


def _serialize_entries(entries: Iterable[Any]) -> list[dict[str, Any]]:
    """Serialize entries for a multi-result response, previewing long values."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        data = _serialize_entry(entry)
        value = data.get("value")
        # Structured values are the biggest ones in practice (investigations and
        # design plans reach 20K characters as objects), so measure what a client
        # actually receives rather than only checking plain strings.
        rendered = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(rendered) > LIST_VALUE_CHAR_LIMIT:
            data["value"] = rendered[:LIST_VALUE_CHAR_LIMIT]
            data["value_truncated"] = True
            data["value_chars"] = len(rendered)
            if not isinstance(value, str):
                data["value_format"] = "json"
            data["full_value"] = (
                f"Preview only — {LIST_VALUE_CHAR_LIMIT} of {len(rendered)} characters. "
                f"Call amfs_read(entity_path=\"{data.get('entity_path')}\", "
                f"key=\"{data.get('key')}\") for the whole value."
            )
        out.append(data)
    return out


# ──────────────────────────────────────────────────────────────────────
# MCP Tools
# ──────────────────────────────────────────────────────────────────────


def _bind_home_entity_paths(
    mem: Any, name: str, description: str | None, *, persist: bool
) -> list[str]:
    """Persist the AMFS_ENTITY_PATH binding on the agent's profile and return the
    ``auto_context_paths`` it should hydrate now (the bound home entity first).

    Shared by the new-identity and already-active paths so a disposable
    environment still gets its binding applied — and the ``next_step`` hint back —
    when sticky identity was already restored (or the same name is re-set), the
    case that previously returned early and skipped both.

    ``persist`` is True for a fresh identity, which always rewrites the profile to
    record the description and session metadata. On the already-active path the
    profile is rewritten only when there is actually a bound path to apply, so a
    plain idempotent re-affirmation with no AMFS_ENTITY_PATH costs no write.
    """
    try:
        from amfs_core.models import AgentProfile

        current_agent = mem._adapter.get_agent(name, namespace=mem.namespace)
        existing = current_agent.profile if current_agent else None
        auto_paths = list(existing.auto_context_paths) if existing else []
        # It goes to the *front*: next_step and the hydration hint brief
        # auto_paths[0], and the environment's binding must win over any path
        # already saved on the profile — that's the point of AMFS_ENTITY_PATH.
        bound = _default_entity_path()
        if bound:
            auto_paths = [bound] + [p for p in auto_paths if p != bound]
        if persist or bound:
            profile = AgentProfile(
                description=description or (existing.description if existing else ""),
                default_branch=existing.default_branch if existing else "main",
                auto_context_paths=auto_paths,
                tags=existing.tags if existing else [],
                session_metadata=_session_metadata,
            )
            mem._adapter.update_agent_profile(name, profile, namespace=mem.namespace)
        return auto_paths
    except Exception:
        logger.debug("Could not update agent profile for %s", name, exc_info=True)
        return [p for p in [_default_entity_path()] if p]


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

    # Idempotent on the identity itself, but not a no-op: sticky restoration (or
    # any earlier _get_memory call) leaves _active_identity == name, so this is
    # the common path in a disposable environment. It must still apply the
    # AMFS_ENTITY_PATH binding and hand back the auto-context paths / next_step,
    # or the environment that relies on that binding for hydration never gets it.
    if _active_identity == name:
        _last_activity = now
        mem = _get_memory()
        mem.session_metadata = _session_metadata
        auto_paths = _bind_home_entity_paths(mem, name, description, persist=False)
        result: dict[str, Any] = {
            "identity": name,
            "session_id": mem.session_id,
            "status": "already_active",
            "sticky": True,
            "getting_started": _getting_started(),
            "session_metadata": _session_metadata.model_dump(mode="json"),
        }
        if auto_paths:
            result["auto_context_paths"] = auto_paths
            result["next_step"] = (
                f'Call amfs_briefing(entity_path="{auto_paths[0]}") now to load '
                f"prior context before working."
            )
        if description:
            result["description"] = description
        return json.dumps(result, default=str)

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
    # metadata (and the AMFS_ENTITY_PATH binding) so it shows up on the dashboard
    # and the environment's home entity is surfaced back as real auto-context.
    # Best-effort — identity setting always succeeds even if the profile update
    # fails. Shared with the already-active path above.
    auto_paths = _bind_home_entity_paths(mem, name, description, persist=True)

    result: dict[str, Any] = {
        "previous_identity": old_identity,
        "new_identity": name,
        "session_id": mem.session_id,
        "sticky": True,
        "hint": "Identity saved — it will be auto-restored in future sessions.",
        "getting_started": _getting_started(),
        "session_metadata": _session_metadata.model_dump(mode="json"),
    }
    # Surface the entity paths this agent should load now (auto-context
    # hydration): a bound AMFS_ENTITY_PATH plus any paths saved on the profile.
    if auto_paths:
        result["auto_context_paths"] = auto_paths
        result["next_step"] = (
            f'Call amfs_briefing(entity_path="{auto_paths[0]}") now to load '
            f"prior context before working."
        )
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
    bound = _default_entity_path()
    if bound:
        result["bound_entity_path"] = bound
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
    for path in (_identity_file(), _legacy_identity_file()):
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            logger.debug("Could not remove sticky identity file %s", path, exc_info=True)
    return json.dumps({
        "previous_identity": old,
        "current_identity": detect_agent_id(),
        "sticky_cleared": True,
    })


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
def amfs_read(entity_path: str, key: str) -> str:
    """Read a single memory entry when you ALREADY know its exact path and key.

    Only use this when you have the exact entity_path AND key (e.g. from a prior
    search/retrieve result). Do NOT guess coordinates — if you're trying to find
    something by meaning or the user gave you a plain phrase, use `amfs_retrieve`
    instead (it needs no path/key).

    Returns the full entry as JSON including value, confidence, provenance,
    and version. Returns a not_found message if the entry does not exist.

    Example: amfs_read("checkout-service", "retry-pattern")
    """
    mem = _get_memory()
    entry = mem.read(entity_path, key)
    if entry is None:
        return json.dumps({"status": "not_found", "entity_path": entity_path, "key": key,
                           "hint": "No entry at that exact path/key. If you were guessing the "
                                   "coordinates, call amfs_retrieve(query=\"<the user's words>\") to "
                                   "search by meaning instead — do NOT tell the user nothing is stored "
                                   "until you've tried amfs_retrieve."})
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
    result["next"] = (
        "Saved. You — or your agents in any other tool on this account — can recall this "
        "later just by asking in plain language; call amfs_retrieve(query=\"...\") "
        "(no path/key needed). Tell the user it's saved and recallable from their other tools."
    )
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
    include_artifacts: bool = True,
) -> str:
    """Search across all memory entries with structured filters or exact keywords.

    Use this before starting work to find context about the entity you're
    modifying, or to check if another agent already solved a similar problem.
    For plain natural-language recall ("what's my favorite ice cream?"), prefer
    `amfs_retrieve` — it ranks by meaning and needs no filters. Reach for this
    tool when you want to filter by agent/confidence/date/pattern or match an
    exact keyword. `query`, `entity_path`, and all filters are optional.

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
        include_artifacts: When False, exclude stored source files from results
            (they are demoted by default)

    Example: amfs_search(entity_path="checkout-service", min_confidence=0.5)

    Long values come back as a 2000-character preview with `value_truncated`
    and `full_value` set; call amfs_read for the whole value.
    """
    from datetime import datetime as dt

    mem = _get_memory()
    try:
        since_dt = dt.fromisoformat(since) if since else None
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid 'since' timestamp: {e}"})

    results = _call_compat(
        mem.search,
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
        include_artifacts=include_artifacts,
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
        "entries": _serialize_entries(results),
    }, default=str)


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
def amfs_retrieve(
    query: str,
    entity_path: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 10,
    semantic_weight: float = 0.5,
    recency_weight: float = 0.3,
    confidence_weight: float = 0.2,
    depth: int = 3,
    include_artifacts: bool = True,
) -> str:
    """Find memories by meaning — the default tool for any recall/lookup.

    Use this FIRST whenever the user asks you to find, recall, look up, or
    remember something, or when you want to check whether relevant memory
    exists. Just pass the user's own words as `query`. You do NOT need to know
    where it's stored: `entity_path` is optional and, when omitted, this searches
    across everything you can see (all your agents and all entities).

    It blends semantic similarity, recency, and confidence into one ranked list,
    so paraphrases and synonyms match even without exact keywords. Prefer this
    over `amfs_read` (which needs exact coordinates) and over `amfs_search`
    (which is for structured filtering / exact keywords).

    Args:
        query: Natural language query in the user's own words (e.g. "my favorite ice cream")
        entity_path: Optional entity path filter
        min_confidence: Minimum confidence threshold (0.0-1.0)
        limit: Maximum results to return
        semantic_weight: Weight for semantic similarity (0.0-1.0)
        recency_weight: Weight for recency (0.0-1.0)
        confidence_weight: Weight for confidence (0.0-1.0)
        depth: Tier depth (1=hot only, 2=hot+warm, 3=all tiers)
        include_artifacts: When False, exclude stored source files from results
            (they are demoted by default so genuine facts rank above code)

    Returns ranked results with score breakdowns showing how each
    signal contributed to the final ranking.

    Long values come back as a 2000-character preview with `value_truncated`
    and `full_value` set; call amfs_read for the whole value.
    """
    from amfs_core.models import RecallConfig

    mem = _get_memory()
    recall_config = RecallConfig(
        semantic_weight=semantic_weight,
        recency_weight=recency_weight,
        confidence_weight=confidence_weight,
    )

    # Prefer server-side semantic retrieval (embedder + pgvector live on the
    # server); falls back to client-side composite scoring for local adapters.
    results = _call_compat(
        mem.retrieve,
        query=query,
        entity_path=entity_path,
        min_confidence=min_confidence,
        limit=limit,
        recall_config=recall_config,
        include_artifacts=include_artifacts,
    )

    serialized = []
    for scored, data in zip(results, _serialize_entries(s.entry for s in results)):
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
        "entries": _serialize_entries(entries),
    }, default=str)


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
def amfs_export(
    entity_path: str,
    format: str = "jsonl",
    row_path: str | None = None,
    inline_char_budget: int = 20000,
) -> str:
    """Bulk-fetch every entry under an entity path with FULL values, to a file.

    Use this instead of many amfs_read calls when you need all the data under a
    path (e.g. an entire room). Unlike amfs_list/amfs_search/amfs_retrieve, this
    does NOT truncate values — but to keep your context clean it writes the full
    data to a local file and returns a manifest (path, count, keys), not the
    values. Read the file with your filesystem tool, or hand its path to a
    subagent. Because it is on disk, the data survives conversation compaction:
    re-crunching is a file read, not a re-fetch.

    Args:
        entity_path: Path to export (required). For a room, pass its shared
            path, e.g. "@my-room/listings". An entity path is required —
            unscoped exports would drop shared/room data.
        format: "jsonl" (default), "json", or "csv". csv/jsonl combine well
            with row_path for tabular datasets.
        row_path: Optional dotted field holding a list of records to flatten
            into one row each (e.g. "listings" turns 26 batch entries into one
            row per listing). Omit to export whole entries.
        inline_char_budget: If the whole payload is at or under this many
            characters it is ALSO returned inline (for clients without a
            filesystem tool). Larger payloads return the path only.

    Example: amfs_export("@deals-room/listings", format="csv", row_path="listings")
    """
    import csv
    import io
    import uuid

    from amfs_core.aggregates import coerce_value, iter_rows

    fmt = format.lower()
    if fmt not in ("jsonl", "json", "csv"):
        return json.dumps({"error": f"unknown format {format!r}; use jsonl, json, or csv"})

    mem = _get_memory()
    entries = mem.list(entity_path)
    if not entries:
        return json.dumps({
            "status": "empty",
            "count": 0,
            "entity_path": entity_path,
            "message": "No entries found for that entity path.",
        })

    keys = [getattr(e, "key", None) for e in entries]
    if row_path:
        records: list[Any] = iter_rows(entries, row_path)
    else:
        records = [
            {**_serialize_entry(e), "value": coerce_value(getattr(e, "value", None))}
            for e in entries
        ]

    if fmt == "json":
        content = json.dumps(records, default=str, indent=2)
    elif fmt == "jsonl":
        content = "\n".join(json.dumps(r, default=str) for r in records)
    else:  # csv
        dict_rows = [r for r in records if isinstance(r, dict)]
        columns: list[str] = []
        for r in dict_rows:
            for k in r.keys():
                if k not in columns:
                    columns.append(k)
        buf = io.StringIO()
        if columns:
            writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for r in dict_rows:
                writer.writerow({
                    k: (v if isinstance(v, (str, int, float, bool)) or v is None
                        else json.dumps(v, default=str))
                    for k, v in r.items()
                })
        else:
            writer = csv.writer(buf)
            writer.writerow(["value"])
            for r in records:
                writer.writerow([json.dumps(r, default=str)])
        content = buf.getvalue()

    export_dir = os.path.join(os.path.expanduser("~"), ".amfs", "exports")
    os.makedirs(export_dir, exist_ok=True)
    ext = {"jsonl": "jsonl", "json": "json", "csv": "csv"}[fmt]
    safe = "".join(c if c.isalnum() else "-" for c in entity_path).strip("-") or "export"
    path = os.path.join(export_dir, f"{safe}-{uuid.uuid4().hex[:8]}.{ext}")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as exc:
        return json.dumps({"error": f"could not write export file: {exc}"})

    manifest: dict[str, Any] = {
        "path": path,
        "format": fmt,
        "entity_path": entity_path,
        "row_path": row_path,
        "entry_count": len(entries),
        "record_count": len(records),
        "total_chars": len(content),
        "keys": keys,
        "next": (
            "Full data written to 'path'. Read it with your filesystem tool or "
            "hand the path to a subagent — it survives compaction. For numbers, "
            "prefer amfs_aggregate so records never enter your context."
        ),
    }
    # Gate the inline copy on the size it will ACTUALLY occupy in this manifest,
    # not on len(content): a CSV file is compact, but manifest["inline"] embeds
    # the raw records which re-serialize as JSON — far larger for csv/json — so
    # comparing the file size would let a small CSV smuggle a huge inline blob
    # past the context budget.
    inline_serialized = json.dumps(records, default=str)
    manifest["inline_chars"] = len(inline_serialized)
    if len(inline_serialized) <= inline_char_budget:
        manifest["inline"] = records
    return json.dumps(manifest, default=str)


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
def amfs_aggregate(
    entity_path: str,
    op: str = "count",
    field: str | None = None,
    group_by: str | None = None,
    row_path: str | None = None,
) -> str:
    """Compute an aggregate over records under an entity path — server-side.

    Use this to answer quantitative questions ("average price", "total yield by
    city", "how many listings") WITHOUT pulling any records into your context.
    Only the computed result comes back. Far cheaper than exporting and doing
    the math token-by-token.

    Args:
        entity_path: Path to aggregate over (required). For a room use its
            shared path, e.g. "@my-room/listings".
        op: "count" (default), "sum", "mean", "min", "max", or "stats"
            (all numeric stats at once). Numeric ops require `field`.
        field: Dotted field to aggregate, e.g. "price" or "meta.yield".
        group_by: Optional dotted field to group results by (e.g. "city").
        row_path: Optional dotted field holding a list of records to flatten
            into rows first (e.g. "listings" for batch entries).

    Example: amfs_aggregate("@deals-room/listings", op="mean", field="price", group_by="city", row_path="listings")
    """
    body = {
        "entity_path": entity_path,
        "op": op,
        "field": field,
        "group_by": group_by,
        "row_path": row_path,
    }
    if os.environ.get("AMFS_HTTP_URL"):
        return _http_api_call("POST", "/api/v1/aggregate", body=body)

    # Local fallback (filesystem / local-Postgres installs): compute over the
    # same shared aggregates module so results match the server path.
    from amfs_core.aggregates import aggregate_entries

    mem = _get_memory()
    entries = mem.list(entity_path)
    try:
        result = aggregate_entries(
            entries, op=op, field=field, group_by=group_by, row_path=row_path
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    result["entity_path"] = entity_path
    result["entries_scanned"] = len(entries)
    return json.dumps(result, default=str)


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


def _training_note(missing: list[str]) -> str:
    """Explain which half of the training pair the sealed trace is missing.

    Worded to be actionable without inviting invention. A request exists for
    every session, so a missing ``task_input`` is always a gap worth closing;
    an action does not, and a session that only read and wrote memory has none
    to record. Prompting for actions unconditionally would ask agents to make
    them up, which costs more than the missing example.
    """
    if missing == ["task_input"]:
        return (
            "The actions on this trace are recorded but the request that "
            "prompted them is not, so it says what was done without saying "
            "what was asked. Pass task_input on this call, in the words the "
            "request arrived in."
        )
    if missing == ["tool_calls"]:
        return (
            "The request on this trace is recorded but no action is, so there "
            "is no decision in it to learn from. Call amfs_record_action as "
            "each consequential action is taken, before committing. A session "
            "that only read and wrote memory rightly has none, and an invented "
            "action is worse than the gap."
        )
    return (
        "This trace carries neither the request that started the work nor any "
        "recorded action, so it is not a training example. Pass task_input on "
        "this call, and call amfs_record_action as each consequential action "
        "is taken. A session that only read and wrote memory rightly records "
        "no action, and an invented one is worse than the gap."
    )


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": False, "destructiveHint": False})
def amfs_commit_outcome(
    outcome_ref: str,
    outcome_type: str,
    task_input: str | None = None,
) -> str:
    """Record an outcome and auto-link it to everything read this session.

    ALWAYS call this at the end of meaningful work. Without it, the decision
    trace (which memories were read, what contexts were gathered, what was
    decided) is lost when the session ends.

    This snapshots all reads, writes, recorded contexts, and decisions from
    this session into a persisted DecisionTrace. It also back-propagates
    confidence changes: entries linked to successes stabilize, entries linked
    to failures get flagged for review.

    Back-propagation applies to memories READ this session, since those are the
    ones the outcome is evidence about. A session that only wrote therefore
    returns affected_entries: 0 and still saves a full trace — check
    entries_created, entries_updated and external_contexts for what it holds.

    Args:
        outcome_ref: Reference identifier (e.g. "INC-2047", "task-42", "PR-456")
        outcome_type: One of "success", "minor_failure", "failure", "critical_failure",
            "clean_deploy", "regression", "p2_incident", "p1_incident"
        task_input: Optional. The request that triggered this work, in the words
            it arrived in (e.g. the user's ask, the alert payload, the ticket
            body). Pass it whenever you have it: it is the prompt half of the
            trace, and without it the trace records what you decided but not
            what you were asked. Secrets are scanned and redacted before storage.

    Example: amfs_commit_outcome("task-42", "success")
    Example: amfs_commit_outcome("INC-2047", "success", task_input="API p99 latency above 2s in us-east")
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

    entries = mem.commit_outcome(outcome_ref, otype, task_input=task_input)
    trace = getattr(mem, "_last_trace", None)
    result: dict[str, Any] = {
        "outcome_ref": outcome_ref,
        "outcome_type": outcome_type,
        "affected_entries": len(entries),
        "entries": [_serialize_entry(e) for e in entries],
    }
    if trace is not None:
        # Only the id is adapter-dependent — the filesystem adapter persists a
        # trace without minting one. Gating the whole block on it dropped the
        # counts too, which is what made a committed trace look like nothing
        # had been recorded.
        if getattr(trace, "id", None):
            result["trace_id"] = trace.id
        result["causal_entries"] = len(trace.causal_entries)
        result["external_contexts"] = len(trace.external_contexts)
        # Reported for the same reason as the two counts above it, and it is the
        # only one an agent can check its own recording against: actions are the
        # half of a training example that has to be recorded deliberately, and
        # the safety gate drops any whose arguments cannot be redacted. Without
        # this number a session that recorded ten actions and landed none looks
        # exactly like one that landed all ten, and the absence surfaces much
        # later as an export with nothing in it.
        result["tool_calls"] = len(getattr(trace, "tool_calls", None) or [])
        # Supervised training pairs the request with the action taken in
        # response, and the exporter skips a trace missing either half. Said
        # here because this call is the last moment anyone can tell: an account
        # finds out otherwise when readiness reports hundreds of outcomes and
        # nothing to train on, months after the sessions that produced them.
        # Read off the trace rather than off the arguments, so a task_input the
        # SDK never stored or the gate redacted is reported as absent.
        missing = [
            half
            for half, present in (
                ("task_input", bool(getattr(trace, "task_input", None))),
                ("tool_calls", bool(result["tool_calls"])),
            )
            if not present
        ]
        result["training"] = {"trainable": not missing}
        if missing:
            result["training"]["missing"] = missing
            result["training"]["note"] = _training_note(missing)
        result["session_duration_ms"] = trace.session_duration_ms
        diff = getattr(trace, "state_diff", None)
        if diff is not None:
            result["entries_created"] = diff.entries_created
            result["entries_updated"] = diff.entries_updated
        if not entries:
            # A write-only session commits a complete trace and still reports
            # zero here, which reads like the call did nothing. Say what the
            # number actually measures so it is not mistaken for a failure.
            #
            # Reads and adjustments are not the same count: the adapter skips
            # any causal entry it cannot resolve, so a session that read can
            # still adjust nothing. Saying "nothing was read" in that case
            # would contradict the causal_entries beside it.
            if trace.causal_entries:
                result["note"] = (
                    "affected_entries counts only memories whose confidence "
                    "this outcome adjusted. This session's reads are listed "
                    "in causal_entries, but none of them could be updated — "
                    "usually because the entry no longer resolves. The trace "
                    "was still saved."
                )
            else:
                result["note"] = (
                    "affected_entries counts only memories whose confidence "
                    "this outcome adjusted, which is the set that was read "
                    "during the session. Nothing was read, so there was "
                    "nothing to adjust. The trace was still saved, including "
                    "any writes and contexts."
                )
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


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": False, "destructiveHint": False})
def amfs_record_action(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    result: str = "",
    success: bool = True,
) -> str:
    """Record an action you took — the tool you called and what you passed it.

    Call this AS IT HAPPENS, right after taking a consequential action: deploying,
    rolling back, editing a file, refunding, sending, opening a PR, changing a
    setting. Not for reading or searching — record what you *did*, not what you
    looked at. Use amfs_record_context for what you learned.

    This is the counterpart to task_input on amfs_commit_outcome. Together they
    form a complete record of the decision: what you were asked, and what you did
    about it. AMFS cannot see this by itself — it observes only its own tools, so
    a call to your deploy or refund tool is invisible unless you record it.

    Args:
        tool_name: The tool or operation you invoked (e.g. "deploy_rollback",
            "refund_payment", "edit_file"). Use the real name, consistently —
            the same action should always be recorded under the same name.
        arguments: What you passed it, as a dict (e.g. {"service": "checkout",
            "to_version": "v41"}). Secrets are scanned and redacted before
            storage; an action whose arguments cannot be cleared is dropped.
        result: Optional. What came back, or a short description of the effect.
            Stored truncated — it is a record, not a cache.
        success: Whether the action succeeded. Record failed actions too; a trace
            committed as a failure is more useful when it shows what was tried.

    Example: amfs_record_action("deploy_rollback", {"service": "checkout", "to_version": "v41"})
    Example: amfs_record_action("refund_payment", {"charge_id": "ch_123"}, result="refunded", success=True)
    """
    mem = _get_memory()
    mem.record_action(
        tool_name,
        arguments or {},
        result=result,
        success=success,
    )
    return json.dumps({"recorded_action": tool_name, "success": success})


@mcp.tool(tags={"core"}, annotations={"readOnlyHint": True})
def amfs_recall(entity_path: str, key: str) -> str:
    """Recall YOUR OWN memory for an EXACT key — needs the precise path + key.

    NOTE: despite the name, this is NOT free-text recall — it requires the exact
    entity_path AND key. If the user asked in plain language ("what food do I
    like?") or you're guessing the key, use `amfs_retrieve(query=...)` instead —
    it searches by meaning with no path/key needed. Only use amfs_recall to
    re-read a key you already know you wrote.

    Args:
        entity_path: Entity path (e.g. "checkout-service")
        key: Memory key (e.g. "retry-pattern")

    Example: amfs_recall("checkout-service", "retry-pattern")
    """
    mem = _get_memory()
    entry = mem.recall(entity_path, key)
    if entry is None:
        return json.dumps({"status": "not_found", "entity_path": entity_path, "key": key,
                           "hint": "No entry at that exact key. Do NOT conclude nothing is stored — "
                                   "call amfs_retrieve(query=\"<the user's words>\") to search by meaning "
                                   "across everything you can see (no path/key needed)."})
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
        "entries": _serialize_entries(entries),
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
    # When the host bound this environment to an entity (AMFS_ENTITY_PATH),
    # default to it so a fresh, disposable process hydrates the right memory
    # without the agent having to know the path.
    if entity_path is None:
        entity_path = _default_entity_path()
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
            "entries": _serialize_entries(entries[:limit]),
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
