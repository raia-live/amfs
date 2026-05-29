"""DigestCompiler — orchestrates compilation strategies to produce digests."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Protocol

from amfs_core.models import Digest, DigestType, Event, EventType

if TYPE_CHECKING:
    from amfs_postgres.adapter import PostgresAdapter

logger = logging.getLogger(__name__)


class CompilationStrategy(Protocol):
    """Interface for digest compilation strategies.

    OSS ships RuleBasedStrategy. Pro can provide LLMStrategy,
    OutcomeCalibratedStrategy, etc.
    """

    def compile_entity(self, entity_path: str, adapter: PostgresAdapter, namespace: str, branch: str = "main") -> Digest | None: ...
    def compile_agent_brief(self, agent_id: str, adapter: PostgresAdapter, namespace: str, branch: str = "main") -> Digest | None: ...
    def compile_source(self, source_id: str, adapter: PostgresAdapter, namespace: str, branch: str = "main") -> Digest | None: ...
    def compile_connection_map(self, entity_path: str, adapter: PostgresAdapter, namespace: str, branch: str = "main") -> Digest | None: ...
    def compile_trace_patterns(self, entity_path: str, adapter: PostgresAdapter, namespace: str, branch: str = "main") -> Digest | None: ...


class DigestCompiler:
    """Compiles raw memory into structured digests using pluggable strategies."""

    def __init__(
        self,
        adapter: PostgresAdapter,
        strategies: list[CompilationStrategy] | None = None,
        namespace: str = "default",
        branch: str = "main",
    ) -> None:
        self._adapter = adapter
        self._namespace = namespace
        self._branch = branch
        if strategies:
            self._strategy = strategies[0]
        else:
            from amfs_cortex.strategies import RuleBasedStrategy
            self._strategy = RuleBasedStrategy()

    @property
    def branch(self) -> str:
        return self._branch

    @branch.setter
    def branch(self, value: str) -> None:
        self._branch = value

    def compile(self, scope_key: str, *, branch: str | None = None) -> Digest | None:
        """Compile a digest for the given scope key.

        Scope keys are prefixed: 'entity:path', 'agent:id', 'source:id'.
        """
        b = branch or self._branch
        kind, _, scope = scope_key.partition(":")
        if not scope:
            return None

        digest: Digest | None = None
        if kind == "entity":
            digest = self._strategy.compile_entity(scope, self._adapter, self._namespace, b)
        elif kind == "agent":
            digest = self._strategy.compile_agent_brief(scope, self._adapter, self._namespace, b)
        elif kind == "source":
            digest = self._strategy.compile_source(scope, self._adapter, self._namespace, b)
        elif kind == "connection":
            digest = self._strategy.compile_connection_map(scope, self._adapter, self._namespace, b)
        elif kind == "trace":
            digest = self._strategy.compile_trace_patterns(scope, self._adapter, self._namespace, b)
        else:
            logger.warning("Unknown scope kind: %s", kind)
            return None

        if digest:
            digest.branch = b
            self._adapter.upsert_digest(digest)
            logger.debug("Compiled %s digest for %s (branch=%s)", kind, scope, b)
            self._log_brief_compiled(digest, kind, scope, b)

        return digest

    def _log_brief_compiled(self, digest: Digest, kind: str, scope: str, branch: str) -> None:
        """Log a brief_compiled timeline event for the relevant agent(s)."""
        try:
            agent_ids: list[str] = []
            if kind == "agent":
                agent_ids = [scope]
            elif digest.source_agents:
                agent_ids = digest.source_agents

            for aid in agent_ids:
                self._adapter.log_event(Event(
                    namespace=self._namespace,
                    agent_id=aid,
                    branch=branch,
                    event_type=EventType.BRIEF_COMPILED,
                    summary=f"Compiled {kind} digest for {scope}",
                    details={
                        "digest_type": digest.digest_type.value,
                        "scope": scope,
                        "entry_count": digest.entry_count,
                        "entities_written": list(
                            digest.summary.get("top_entities", {}).keys()
                        ) if isinstance(digest.summary.get("top_entities"), dict) else [],
                    },
                ))
        except Exception:
            logger.debug("Failed to log brief_compiled event for %s:%s", kind, scope, exc_info=True)

    def recompile_all(self, *, branch: str | None = None) -> int:
        """Recompile all digests from scratch. Returns count of digests compiled."""
        b = branch or self._branch
        entries = self._adapter.list(branch=b)
        entity_paths: set[str] = set()
        agent_ids: set[str] = set()
        source_ids: set[str] = set()

        for entry in entries:
            entity_paths.add(entry.entity_path)
            aid = entry.provenance.agent_id
            if aid.startswith("webhook/"):
                source_ids.add(aid.split("/", 1)[1])
            elif aid.startswith("external/"):
                source_ids.add(aid.split("/", 1)[1])
            else:
                agent_ids.add(aid)

        count = 0
        errors = 0
        for ep in entity_paths:
            try:
                if self.compile(f"entity:{ep}", branch=b):
                    count += 1
            except Exception:
                errors += 1
                logger.exception("Failed to compile entity digest for %s", ep)
            try:
                if self.compile(f"connection:{ep}", branch=b):
                    count += 1
            except Exception:
                errors += 1
                logger.exception("Failed to compile connection digest for %s", ep)
        for aid in agent_ids:
            try:
                if self.compile(f"agent:{aid}", branch=b):
                    count += 1
            except Exception:
                errors += 1
                logger.exception("Failed to compile agent digest for %s", aid)
        for sid in source_ids:
            try:
                if self.compile(f"source:{sid}", branch=b):
                    count += 1
            except Exception:
                errors += 1
                logger.exception("Failed to compile source digest for %s", sid)

        logger.info("Recompiled %d digests (%d entities, %d agents, %d sources, %d errors) branch=%s",
                     count, len(entity_paths), len(agent_ids), len(source_ids), errors, b)
        return count

    def compute_scope_fingerprint(self, scope_key: str, *, branch: str | None = None) -> str | None:
        """Compute a lightweight fingerprint for a scope's current state.

        Used by the drift gate to decide whether recompilation is worthwhile.
        The fingerprint captures entry count, newest written_at, and average
        confidence — changes in any of these signal meaningful state drift.
        """
        b = branch or self._branch
        kind, _, scope = scope_key.partition(":")
        if not scope:
            return None

        try:
            if kind == "entity":
                entries = self._adapter.list(scope, branch=b)
            elif kind == "agent":
                entries = [
                    e for e in self._adapter.list(branch=b)
                    if e.provenance.agent_id == scope
                ]
            elif kind == "source":
                entries = [
                    e for e in self._adapter.list(branch=b)
                    if e.provenance.agent_id.endswith(f"/{scope}")
                ]
            elif kind == "connection":
                entries = self._adapter.list(scope, branch=b)
            else:
                return None
        except Exception:
            return None

        if not entries:
            return hashlib.sha256(b"empty").hexdigest()[:16]

        count = len(entries)
        newest = max(e.provenance.written_at for e in entries)
        avg_conf = sum(e.confidence for e in entries) / count
        top_keys = sorted(entries, key=lambda e: e.confidence, reverse=True)[:3]
        top_key_str = "|".join(e.key for e in top_keys)

        raw = f"{count}:{newest.isoformat()}:{avg_conf:.6f}:{top_key_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
