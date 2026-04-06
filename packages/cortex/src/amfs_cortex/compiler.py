"""DigestCompiler — orchestrates compilation strategies to produce digests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from amfs_core.models import Digest, DigestType

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
        else:
            logger.warning("Unknown scope kind: %s", kind)
            return None

        if digest:
            digest.branch = b
            self._adapter.upsert_digest(digest)
            logger.debug("Compiled %s digest for %s (branch=%s)", kind, scope, b)

        return digest

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
        for ep in entity_paths:
            if self.compile(f"entity:{ep}", branch=b):
                count += 1
        for aid in agent_ids:
            if self.compile(f"agent:{aid}", branch=b):
                count += 1
        for sid in source_ids:
            if self.compile(f"source:{sid}", branch=b):
                count += 1

        logger.info("Recompiled %d digests (%d entities, %d agents, %d sources) branch=%s",
                     count, len(entity_paths), len(agent_ids), len(source_ids), b)
        return count
