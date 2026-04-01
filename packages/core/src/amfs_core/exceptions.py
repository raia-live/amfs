"""AMFS exception hierarchy."""

from __future__ import annotations


class AMFSError(Exception):
    """Base exception for all AMFS errors."""


class AdapterError(AMFSError):
    """Raised when an adapter operation fails."""


class EntryNotFoundError(AMFSError):
    """Raised when a requested memory entry does not exist."""


class VersionConflictError(AMFSError):
    """Raised on a concurrent-write version conflict."""

    def __init__(self, entity_path: str, key: str, expected: int, actual: int) -> None:
        self.entity_path = entity_path
        self.key = key
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Version conflict for {entity_path}/{key}: "
            f"expected {expected}, found {actual}"
        )


class StaleWriteError(AMFSError):
    """Raised when an agent writes a key that was modified by another agent
    since the last read (conflict_policy=RAISE).

    Attributes ``read_version`` and ``current_version`` let the caller
    decide how to merge.
    """

    def __init__(
        self,
        entity_path: str,
        key: str,
        read_version: int,
        current_version: int,
        current_agent: str,
    ) -> None:
        self.entity_path = entity_path
        self.key = key
        self.read_version = read_version
        self.current_version = current_version
        self.current_agent = current_agent
        super().__init__(
            f"Stale write to {entity_path}/{key}: you read v{read_version} "
            f"but v{current_version} now exists (written by {current_agent})"
        )


class LockTimeoutError(AMFSError):
    """Raised when an advisory lock cannot be acquired within the timeout."""
