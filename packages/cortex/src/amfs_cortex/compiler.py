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

    def compile_entity(self, entity_path: str, adapter: PostgresAdapter, namespace: str) -> Digest | None: ...
    def compile_agent_brief(self, agent_id: str, adapter: PostgresAdapter, namespace: str) -> Digest | None: ...
    def compile_source(self, source_id: str, adapter: PostgresAdapter, namespace: str) -> Digest | None: ...


class DigestCompiler:
    """Compiles raw memory into structured digests using pluggable strategies."""

    def __init__(
        self,
        adapter: PostgresAdapter,
        strategies: list[CompilationStrategy] | None = None,
        namespace: str = "default",
    ) -> None:
        self._adapter = adapter
        self._namespace = namespace
        if strategies:
            self._strategy = strategies[0]
        else:
            from amfs_cortex.strategies import RuleBasedStrategy
            self._strategy = RuleBasedStrategy()

    def compile(self, scope_key: str) -> Digest | None:
        """Compile a digest for the given scope key.

        Scope keys are prefixed: 'entity:path', 'agent:id', 'source:id'.
        """
        kind, _, scope = scope_key.partition(":")
        if not scope:
            return None

        digest: Digest | None = None
        if kind == "entity":
            digest = self._strategy.compile_entity(scope, self._adapter, self._namespace)
        elif kind == "agent":
            digest = self._strategy.compile_agent_brief(scope, self._adapter, self._namespace)
        elif kind == "source":
            digest = self._strategy.compile_source(scope, self._adapter, self._namespace)
        else:
            logger.warning("Unknown scope kind: %s", kind)
            return None

        if digest:
            self._adapter.upsert_digest(digest)
            logger.debug("Compiled %s digest for %s", kind, scope)

        return digest

    def recompile_all(self) -> int:
        """Recompile all digests from scratch. Returns count of digests compiled."""
        entries = self._adapter.list(namespace=self._namespace)
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
            if self.compile(f"entity:{ep}"):
                count += 1
        for aid in agent_ids:
            if self.compile(f"agent:{aid}"):
                count += 1
        for sid in source_ids:
            if self.compile(f"source:{sid}"):
                count += 1

        logger.info("Recompiled %d digests (%d entities, %d agents, %d sources)",
                     count, len(entity_paths), len(agent_ids), len(source_ids))
        return count
