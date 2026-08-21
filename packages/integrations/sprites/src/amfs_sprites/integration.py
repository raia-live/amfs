"""Provision and hydrate AMFS memory inside a Fly.io Sprite.

The lifecycle inside a Sprite is:

1. **provision** — resolve the durable ``entity_path`` and connect to hosted
   SenseLab (or a local store), returning a :class:`SpriteSession`.
2. **hydrate** — turn everything prior sessions learned about this entity into a
   system-prompt block, so the agent boots already knowing the context.
3. **commit** — snapshot the decision trace and (optionally) a task summary when
   the work finishes, so the *next* Sprite is even better briefed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from amfs import AgentMemory, OutcomeType

_DEFAULT_API_URL = "https://amfs-login.sense-lab.ai"
_DEFAULT_AGENT_ID = "sprite-agent"
_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def derive_entity_path(*parts: str, prefix: str | None = "sprites") -> str:
    """Build a stable ``entity_path`` from identifying parts.

    The entity path is the durable key that ties every Sprite working on the
    same thing to one shared memory, so it must be derived from *stable*
    inputs (org, project, service) — never from the ephemeral Sprite/machine
    ID, which changes on every spin-up and would scatter memory across
    single-use paths.

    Args:
        *parts: Ordered, stable identifiers, e.g. ``("acme", "checkout")``.
        prefix: Leading namespace segment (default ``"sprites"``). Pass
            ``None`` to omit it.

    Returns:
        A slugified, slash-joined path, e.g. ``"sprites/acme/checkout"``.

    Raises:
        ValueError: If no non-empty parts are provided.
    """
    segments: list[str] = []
    if prefix:
        segments.append(prefix)
    for part in parts:
        slug = _SLUG_RE.sub("-", str(part).strip().lower()).strip("-")
        if slug:
            segments.append(slug)
    if len(segments) <= (1 if prefix else 0):
        raise ValueError("derive_entity_path requires at least one non-empty part")
    return "/".join(segments)


def _build_adapter(api_url: str | None, api_key: str | None) -> Any | None:
    """Return an HttpAdapter when a URL is configured, else None (local store)."""
    if not api_url:
        return None
    try:
        from amfs_adapter_http import HttpAdapter
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Connecting to hosted SenseLab needs the HTTP adapter. "
            "Install it with:  pip install 'amfs-sprites[http]'"
        ) from exc
    return HttpAdapter(base_url=api_url, api_key=api_key or "")


def provision_memory(
    *,
    entity_path: str | None = None,
    agent_id: str | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    description: str | None = None,
    model: str | None = None,
    set_env: bool = True,
) -> SpriteSession:
    """Connect a Sprite to its durable memory and return a :class:`SpriteSession`.

    Resolution order for every field is: explicit argument → environment
    variable → default. The environment defaults mean a Sprite whose base image
    already exports ``AMFS_ENTITY_PATH`` / ``AMFS_HTTP_URL`` / ``AMFS_API_KEY``
    can call ``provision_memory()`` with no arguments at all.

    Args:
        entity_path: The durable home entity. Falls back to ``AMFS_ENTITY_PATH``.
        agent_id: Role name for this agent. Falls back to ``AMFS_AGENT_ID``,
            then ``"sprite-agent"``. Use a stable, kebab-case role.
        api_url: Hosted SenseLab base URL. Falls back to ``AMFS_HTTP_URL``.
            When neither is set, memory uses the local filesystem store.
        api_key: SenseLab API key. Falls back to ``AMFS_API_KEY``.
        description: One-line description of what this agent is doing now.
        model: The LLM model driving the agent (recorded on decision traces).
        set_env: When True (default), export ``AMFS_ENTITY_PATH`` back into the
            process environment so any child MCP server inherits the binding.

    Raises:
        ValueError: If no entity_path can be resolved.
    """
    entity_path = entity_path or os.environ.get("AMFS_ENTITY_PATH")
    if not entity_path:
        raise ValueError(
            "No entity_path provided and AMFS_ENTITY_PATH is not set. "
            "Pass entity_path=... or derive one with derive_entity_path()."
        )

    agent_id = agent_id or os.environ.get("AMFS_AGENT_ID") or _DEFAULT_AGENT_ID
    api_url = api_url or os.environ.get("AMFS_HTTP_URL")
    api_key = api_key or os.environ.get("AMFS_API_KEY")
    # An explicit URL with a key but the SaaS default host missing is a common
    # mistake; fall back to the hosted default only when a key is present.
    if api_key and not api_url:
        api_url = _DEFAULT_API_URL

    if set_env:
        # So a co-located MCP server (stdio) inherits the same binding and the
        # agent's tool calls land on the same entity.
        os.environ["AMFS_ENTITY_PATH"] = entity_path

    adapter = _build_adapter(api_url, api_key)
    if adapter is not None:
        memory = AgentMemory(agent_id=agent_id, adapter=adapter)
    else:
        memory = AgentMemory(agent_id=agent_id)

    return SpriteSession(
        memory=memory,
        entity_path=entity_path,
        agent_id=agent_id,
        remote=adapter is not None,
        description=description,
        model=model,
    )


def hydrate_prompt(
    memory: AgentMemory,
    entity_path: str,
    *,
    limit: int = 8,
    header: str | None = None,
    include_briefing: bool = True,
) -> str:
    """Render prior memory for ``entity_path`` as a system-prompt block.

    This is the "magic moment" on Sprites: a brand-new microVM produces a prompt
    that already contains what earlier sessions learned. Injecting the returned
    string into the agent's system prompt works for *any* agent, whether or not
    it speaks MCP.

    Args:
        memory: A provisioned :class:`~amfs.AgentMemory`.
        entity_path: The entity to load context for.
        limit: Maximum memory entries to include.
        header: Optional first line; a sensible default is used when omitted.
        include_briefing: When True, prepend the compiled Cortex narrative if the
            backend provides one (hosted SenseLab; a no-op on local stores).

    Returns:
        A markdown block, or a short "fresh start" note when nothing is stored.
    """
    lines: list[str] = []

    narrative = ""
    if include_briefing:
        narrative = _briefing_narrative(memory, entity_path)

    entries = _top_entries(memory, entity_path, limit)

    if not narrative and not entries:
        return (
            f"{header or '## Memory'}\n\n"
            f"No prior memory found for `{entity_path}` yet — this is a fresh "
            f"start. As you work, save durable decisions, patterns, and risks so "
            f"the next session picks up where you leave off."
        )

    lines.append(header or f"## What earlier sessions learned about `{entity_path}`")
    lines.append("")
    lines.append(
        "You are continuing work that other sessions started. This context was "
        "loaded from SenseLab, your persistent memory. Treat it as established "
        "knowledge; build on it rather than rediscovering it."
    )

    if narrative:
        lines.append("")
        lines.append(narrative.strip())

    if entries:
        lines.append("")
        lines.append("### Key memories")
        for e in entries:
            lines.append(f"- **{e['key']}** (confidence {e['confidence']:.2f}): {e['value']}")

    lines.append("")
    lines.append(
        "When this task finishes, record what you did and why so the next "
        "session is even better briefed."
    )
    return "\n".join(lines)


def _briefing_narrative(memory: AgentMemory, entity_path: str) -> str:
    """Best-effort compiled narrative for an entity (empty if unavailable)."""
    try:
        digests = memory.briefing(entity_path=entity_path)
    except Exception:
        return ""
    for digest in digests or []:
        data = digest.model_dump() if hasattr(digest, "model_dump") else digest
        if not isinstance(data, dict):
            continue
        summary = data.get("summary")
        if isinstance(summary, dict):
            narrative = summary.get("narrative")
            if isinstance(narrative, str) and narrative.strip():
                return narrative
    return ""


def _top_entries(memory: AgentMemory, entity_path: str, limit: int) -> list[dict[str, Any]]:
    """Return the highest-confidence entries under an entity, newest tie-broken."""
    try:
        results = memory.search(entity_path=entity_path, sort_by="confidence", limit=limit)
    except Exception:
        try:
            results = memory.list(entity_path)[:limit]
        except Exception:
            return []
    out: list[dict[str, Any]] = []
    for entry in results:
        value = getattr(entry, "value", "")
        rendered = value if isinstance(value, str) else str(value)
        if len(rendered) > 500:
            rendered = rendered[:500] + " …"
        out.append(
            {
                "key": getattr(entry, "key", "?"),
                "value": rendered,
                "confidence": float(getattr(entry, "confidence", 1.0) or 1.0),
            }
        )
    return out


def commit_sprite_outcome(
    memory: AgentMemory,
    outcome_ref: str,
    outcome_type: str | OutcomeType = OutcomeType.SUCCESS,
    *,
    task_input: str | None = None,
    summary: str | None = None,
    entity_path: str | None = None,
) -> None:
    """Persist the session's decision trace, plus an optional task summary.

    Call this when the Sprite finishes meaningful work. The trace is what makes
    the *next* Sprite smarter; the summary is a durable note future agents read.

    Args:
        memory: The provisioned :class:`~amfs.AgentMemory`.
        outcome_ref: A reference for this outcome (ticket, task, or PR id).
        outcome_type: ``"success"`` / ``"minor_failure"`` / ``"failure"`` /
            ``"critical_failure"`` (or an :class:`~amfs.OutcomeType`).
        task_input: The request that started the work, in its original words.
        summary: A short note to persist under the entity as an ``experience``.
        entity_path: Where to store the summary (required if summary is given).
    """
    if isinstance(outcome_type, str):
        outcome_type = OutcomeType(outcome_type)

    if summary:
        if not entity_path:
            raise ValueError("entity_path is required when a summary is provided")
        memory.write(
            entity_path,
            f"sprite-outcome-{outcome_ref}",
            summary,
            memory_type="experience",
        )

    memory.commit_outcome(outcome_ref, outcome_type, task_input=task_input)


@dataclass
class SpriteSession:
    """A provisioned memory session bound to one Sprite's home entity.

    Prefer the convenience methods here over the module-level functions; they
    carry the resolved ``entity_path`` so calls stay short. Usable as a context
    manager so the underlying memory is closed on exit.
    """

    memory: AgentMemory
    entity_path: str
    agent_id: str
    remote: bool = False
    description: str | None = None
    model: str | None = None
    _closed: bool = field(default=False, repr=False)

    def hydrate_prompt(self, *, limit: int = 8, header: str | None = None) -> str:
        """Render prior memory for this session's entity as a prompt block."""
        return hydrate_prompt(self.memory, self.entity_path, limit=limit, header=header)

    def write(self, key: str, value: Any, **kwargs: Any) -> Any:
        """Write a memory under this session's entity_path."""
        return self.memory.write(self.entity_path, key, value, **kwargs)

    def commit_outcome(
        self,
        outcome_ref: str,
        outcome_type: str | OutcomeType = OutcomeType.SUCCESS,
        *,
        task_input: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Commit the decision trace (and optional summary) for this session."""
        commit_sprite_outcome(
            self.memory,
            outcome_ref,
            outcome_type,
            task_input=task_input,
            summary=summary,
            entity_path=self.entity_path if summary else None,
        )

    def close(self) -> None:
        """Release the underlying memory's resources."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self.memory, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> SpriteSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
