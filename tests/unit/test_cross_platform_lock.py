"""Tests for AdvisoryLock cross-platform behavior."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from amfs_core.exceptions import LockTimeoutError
from amfs_core.lock import AdvisoryLock


def test_advisory_lock_acquire_and_release(tmp_path: Path) -> None:
    lock_file = tmp_path / "test.lock"
    lock = AdvisoryLock(lock_file, timeout=1.0)
    with lock:
        assert lock_file.exists()


def test_advisory_lock_timeout(tmp_path: Path) -> None:
    lock_file = tmp_path / "test.lock"
    lock1 = AdvisoryLock(lock_file, timeout=1.0)
    lock1.acquire()
    try:
        lock2 = AdvisoryLock(lock_file, timeout=0.1)
        start = time.monotonic()
        with pytest.raises(LockTimeoutError):
            lock2.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.09
    finally:
        lock1.release()


def test_advisory_lock_reacquire_after_release(tmp_path: Path) -> None:
    lock_file = tmp_path / "test.lock"
    lock1 = AdvisoryLock(lock_file, timeout=0.5)
    lock1.acquire()
    lock1.release()

    lock2 = AdvisoryLock(lock_file, timeout=0.5)
    with lock2:
        assert lock_file.exists()
