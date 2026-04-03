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
    ConfidenceChange,
    ConflictPolicy,
    DecisionTrace,
    ErrorEvent,
    ExternalContext,
    LayerConfig,
    MemoryEntry,
    MemoryStateDiff,
    MemoryStats,
    OutcomeRecord,
    OutcomeType,
    Provenance,
    QueryEvent,
    SearchQuery,
    SemanticQuery,
    TraceEntry,
)
from amfs_core.outcome import OutcomeBackPropagator

__all__ = [
    "AMFSConfig",
    "AMFSError",
    "AdapterABC",
    "AdapterError",
    "CausalTagger",
    "ConfidenceChange",
    "ConflictPolicy",
    "CoWEngine",
    "DecisionTrace",
    "EmbedderABC",
    "EntryNotFoundError",
    "ErrorEvent",
    "ExternalContext",
    "LayerConfig",
    "LifecycleManager",
    "LockTimeoutError",
    "MemoryEntry",
    "MemoryStateDiff",
    "MemoryStats",
    "OutcomeBackPropagator",
    "OutcomeRecord",
    "OutcomeType",
    "Provenance",
    "QueryEvent",
    "ReadTracker",
    "SearchQuery",
    "SemanticQuery",
    "StaleWriteError",
    "TraceEntry",
    "VersionConflictError",
    "WatchHandle",
    "cosine_similarity",
]
