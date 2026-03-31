"""AgentMemory — the main SDK entry point for agents."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.engine import CausalTagger, CoWEngine
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import MemoryEntry, OutcomeRecord, OutcomeType
from amfs_core.outcome import OutcomeBackPropagator

from amfs.config import load_config_or_default
from amfs.factory import create_adapter_from_config


class AgentMemory:
    """High-level API for agents to read, write, and observe shared memory.

    Usage::

        mem = AgentMemory(agent_id="review-agent")
        mem.write("checkout-service", "retry-pattern", {"max_retries": 3})
        entry = mem.read("checkout-service", "retry-pattern")
        mem.close()

    Or as a context manager::

        with AgentMemory(agent_id="review-agent") as mem:
            mem.write("checkout-service", "retry-pattern", {"max_retries": 3})
    """

    def __init__(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        config_path: Path | None = None,
        adapter: AdapterABC | None = None,
        ttl_sweep_interval: float | None = None,
    ) -> None:
        self._config = load_config_or_default(config_path)

        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = create_adapter_from_config(self._config)

        self._tagger = CausalTagger(agent_id, session_id)
        self._engine = CoWEngine(self._adapter, self._tagger)
        self._propagator = OutcomeBackPropagator(self._adapter)

        self._lifecycle: LifecycleManager | None = None
        if ttl_sweep_interval is not None:
            self._lifecycle = LifecycleManager(self._adapter, interval=ttl_sweep_interval)
            self._lifecycle.start()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self._tagger.agent_id

    @property
    def session_id(self) -> str:
        return self._tagger.session_id

    @property
    def namespace(self) -> str:
        return self._config.namespace

    @property
    def adapter(self) -> AdapterABC:
        return self._adapter

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def read(
        self,
        entity_path: str,
        key: str,
        *,
        min_confidence: float = 0.0,
    ) -> MemoryEntry | None:
        """Read the current version of a key."""
        return self._engine.read(entity_path, key, min_confidence=min_confidence)

    def write(
        self,
        entity_path: str,
        key: str,
        value: Any,
        *,
        confidence: float = 1.0,
        ttl_at: datetime | None = None,
        pattern_refs: list[str] | None = None,
    ) -> MemoryEntry:
        """Write a new version of a key with automatic provenance."""
        return self._engine.write(
            entity_path,
            key,
            value,
            confidence=confidence,
            ttl_at=ttl_at,
            pattern_refs=pattern_refs,
        )

    def list(
        self,
        entity_path: str | None = None,
        *,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """List current entries, optionally filtered to an entity path."""
        return self._engine.list(entity_path, include_superseded=include_superseded)

    def watch(
        self,
        entity_path: str,
        callback: Any,
    ) -> WatchHandle:
        """Watch for writes to any key under an entity path."""
        return self._adapter.watch(entity_path, callback)

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def commit_outcome(
        self,
        outcome_ref: str,
        outcome_type: OutcomeType,
        causal_entry_keys: list[str],
        *,
        causal_confidence: float = 1.0,
    ) -> list[MemoryEntry]:
        """Record an outcome and back-propagate confidence changes."""
        record = OutcomeBackPropagator.make_record(
            outcome_ref=outcome_ref,
            outcome_type=outcome_type,
            causal_entry_keys=causal_entry_keys,
            agent_id=self.agent_id,
            causal_confidence=causal_confidence,
        )
        return self._propagator.propagate(record)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop background threads and clean up resources."""
        if self._lifecycle is not None:
            self._lifecycle.stop()
        if hasattr(self._adapter, "close"):
            self._adapter.close()  # type: ignore[attr-defined]

    def __enter__(self) -> AgentMemory:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
