"""AMFS Core — models, engine, and adapter ABC."""

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.engine import CausalTagger, CoWEngine
from amfs_core.exceptions import (
    AMFSError,
    AdapterError,
    EntryNotFoundError,
    LockTimeoutError,
    VersionConflictError,
)
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import AMFSConfig, LayerConfig, MemoryEntry, OutcomeRecord, OutcomeType, Provenance
from amfs_core.outcome import OutcomeBackPropagator

__all__ = [
    "AMFSConfig",
    "AMFSError",
    "AdapterABC",
    "AdapterError",
    "CausalTagger",
    "CoWEngine",
    "EntryNotFoundError",
    "LayerConfig",
    "LifecycleManager",
    "LockTimeoutError",
    "MemoryEntry",
    "OutcomeBackPropagator",
    "OutcomeRecord",
    "OutcomeType",
    "Provenance",
    "VersionConflictError",
    "WatchHandle",
]
