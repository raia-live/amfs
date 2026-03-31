"""Advisory flock-based file lock for filesystem adapter."""

from __future__ import annotations

import fcntl
import time
from pathlib import Path
from types import TracebackType

from amfs_core.exceptions import LockTimeoutError


class AdvisoryLock:
    """A per-key advisory lock using flock().

    Usage::

        with AdvisoryLock(lock_dir / "my.lock"):
            # exclusive access
            ...
    """

    def __init__(self, lock_path: Path, *, timeout: float = 5.0) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._fd: int | None = None

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(self._lock_path, "w")  # noqa: SIM115
        self._fd_obj = fd
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    fd.close()
                    raise LockTimeoutError(
                        f"Could not acquire lock {self._lock_path} "
                        f"within {self._timeout}s"
                    )
                time.sleep(0.01)

    def release(self) -> None:
        if hasattr(self, "_fd_obj") and self._fd_obj is not None:
            try:
                fcntl.flock(self._fd_obj.fileno(), fcntl.LOCK_UN)
            finally:
                self._fd_obj.close()
                self._fd_obj = None  # type: ignore[assignment]

    def __enter__(self) -> AdvisoryLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()
