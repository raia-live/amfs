"""TransactionBuffer — collects writes and flushes them as an atomic commit."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from amfs_core.hashing import content_hash, integrity_chain_hash, tree_hash
from amfs_core.models import (
    ArtifactRef,
    Commit,
    CommitEntry,
    MemoryEntry,
    MemoryType,
    Provenance,
)


class PendingWrite:
    """A buffered write waiting to be flushed in a commit."""

    __slots__ = (
        "entity_path", "key", "value", "confidence", "pattern_refs",
        "memory_type", "artifact_refs", "shared", "embedding",
        "importance_score", "importance_dimensions",
    )

    def __init__(
        self,
        entity_path: str,
        key: str,
        value: Any,
        *,
        confidence: float = 1.0,
        pattern_refs: list[str] | None = None,
        memory_type: MemoryType = MemoryType.FACT,
        artifact_refs: list[ArtifactRef] | None = None,
        shared: bool = True,
        embedding: list[float] | None = None,
        importance_score: float | None = None,
        importance_dimensions: dict[str, float] | None = None,
    ) -> None:
        self.entity_path = entity_path
        self.key = key
        self.value = value
        self.confidence = confidence
        self.pattern_refs = pattern_refs
        self.memory_type = memory_type
        self.artifact_refs = artifact_refs
        self.shared = shared
        self.embedding = embedding
        self.importance_score = importance_score
        self.importance_dimensions = importance_dimensions


class TransactionBuffer:
    """Accumulates writes, then flushes them as a single atomic commit.

    Usage::

        buf = TransactionBuffer(agent_id="my-agent", session_id="sess-1")
        buf.write("repo/svc", "key1", {"data": 1})
        buf.write("repo/svc", "key2", {"data": 2})
        commit, entries = buf.flush(adapter, tagger)
    """

    def __init__(
        self,
        agent_id: str,
        session_id: str,
        *,
        branch: str = "main",
        namespace: str = "default",
    ) -> None:
        self._agent_id = agent_id
        self._session_id = session_id
        self._branch = branch
        self._namespace = namespace
        self._pending: list[PendingWrite] = []
        self._message: str = ""

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def set_message(self, message: str) -> None:
        self._message = message

    def write(
        self,
        entity_path: str,
        key: str,
        value: Any,
        **kwargs: Any,
    ) -> None:
        self._pending.append(PendingWrite(entity_path, key, value, **kwargs))

    def flush(self, adapter: Any, tagger: Any) -> tuple[Commit, list[MemoryEntry]]:
        """Materialize all pending writes into MemoryEntry objects and persist atomically.

        Returns the Commit and the list of written entries.
        Raises ValueError if the buffer is empty.
        """
        if not self._pending:
            raise ValueError("Cannot flush an empty transaction")

        commit_id = uuid.uuid4().hex[:12]
        entries: list[MemoryEntry] = []
        commit_entries: list[CommitEntry] = []

        for pw in self._pending:
            current = adapter.read(pw.entity_path, pw.key, branch=self._branch)
            next_version = (current.version + 1) if current else 1
            value_hash = content_hash(pw.value)
            prev_chain = current.integrity_chain if current else None
            chain = integrity_chain_hash(value_hash, prev_chain)

            entry = MemoryEntry(
                entity_path=pw.entity_path,
                key=pw.key,
                version=next_version,
                value=pw.value,
                provenance=tagger.tag(pattern_refs=pw.pattern_refs),
                confidence=pw.confidence,
                outcome_count=current.outcome_count if current else 0,
                recall_count=current.recall_count if current else 0,
                importance_score=pw.importance_score,
                importance_dimensions=pw.importance_dimensions,
                artifact_refs=pw.artifact_refs or [],
                memory_type=pw.memory_type,
                shared=pw.shared,
                branch=self._branch,
                embedding=pw.embedding,
                content_hash=value_hash,
                integrity_chain=chain,
                commit_id=commit_id,
            )
            entries.append(entry)
            commit_entries.append(CommitEntry(
                entity_path=pw.entity_path,
                key=pw.key,
                version=next_version,
                content_hash=value_hash,
            ))

        written = adapter.write_batch(entries)

        entry_hashes = [e.content_hash for e in written if e.content_hash]
        t_hash = tree_hash(entry_hashes) if entry_hashes else None

        commit = Commit(
            id=commit_id,
            message=self._message,
            author_agent_id=self._agent_id,
            session_id=self._session_id,
            entries=commit_entries,
            tree_hash=t_hash,
            branch=self._branch,
            namespace=self._namespace,
        )
        adapter.save_commit(commit)

        self._pending.clear()
        return commit, written
