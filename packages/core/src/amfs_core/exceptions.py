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


class LockTimeoutError(AMFSError):
    """Raised when an advisory lock cannot be acquired within the timeout."""
