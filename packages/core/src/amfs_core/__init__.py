"""AMFS Core — models, engine, and adapter ABC."""

from amfs_core.abc import AdapterABC, WatchHandle
from amfs_core.embedder import EmbedderABC, cosine_similarity
from amfs_core.engine import CausalTagger, CoWEngine, ReadTracker
from amfs_core.exceptions import (
    AMFSError,
    AdapterError,
    EntryNotFoundError,
    LockTimeoutError,
    StaleWriteError,
    VersionConflictError,
)
from amfs_core.lifecycle import LifecycleManager
from amfs_core.models import (
    AMFSConfig,
    ConflictPolicy,
    LayerConfig,
    MemoryEntry,
    MemoryStats,
    OutcomeRecord,
    OutcomeType,
    Provenance,
    SearchQuery,
    SemanticQuery,
)
from amfs_core.outcome import OutcomeBackPropagator

__all__ = [
    "AMFSConfig",
    "AMFSError",
    "AdapterABC",
    "AdapterError",
    "CausalTagger",
    "ConflictPolicy",
    "CoWEngine",
    "EmbedderABC",
    "EntryNotFoundError",
    "LayerConfig",
    "LifecycleManager",
    "LockTimeoutError",
    "MemoryEntry",
    "MemoryStats",
    "OutcomeBackPropagator",
    "OutcomeRecord",
    "OutcomeType",
    "Provenance",
    "ReadTracker",
    "SearchQuery",
    "SemanticQuery",
    "StaleWriteError",
    "VersionConflictError",
    "WatchHandle",
    "cosine_similarity",
]
