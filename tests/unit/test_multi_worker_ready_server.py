"""The HTTP server behaves the same in every uvicorn worker process.

Three things had to hold before ``--workers`` above one was safe:

* The embedded Cortex worker starts from the app lifespan, once per serving
  process, rather than from ``main()`` — which runs only in the supervisor,
  leaving ``/api/v1/cortex/*`` in the children reading a ``None`` global.
* ``_model_executor`` is sized to the CPUs the container may use, not the
  host's core count.
* Routes whose bodies are wholly synchronous are plain ``def`` so FastAPI runs
  them on the threadpool; declared ``async`` they held the loop for the length
  of every query and every pool checkout.
"""

from __future__ import annotations

import argparse
import builtins
import inspect
import io
import os
import threading

import pytest
from amfs_http import server
from fastapi.testclient import TestClient

# ── Cortex starts per process, from the lifespan ─────────────────────


@pytest.fixture
def no_db(monkeypatch):
    monkeypatch.delenv("AMFS_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("AMFS_HTTP_URL", raising=False)
    monkeypatch.setattr(server, "_cortex_worker", None)


def test_lifespan_starts_cortex_when_the_flag_is_in_the_environment(monkeypatch, no_db) -> None:
    monkeypatch.setenv("AMFS_WITH_CORTEX", "1")
    calls: list[int] = []
    monkeypatch.setattr(server, "_start_embedded_cortex", lambda: calls.append(os.getpid()))

    with TestClient(server.app):
        pass

    assert calls == [os.getpid()], "the lifespan must start Cortex exactly once in this process"


def test_lifespan_leaves_cortex_alone_without_the_flag(monkeypatch, no_db) -> None:
    monkeypatch.delenv("AMFS_WITH_CORTEX", raising=False)
    monkeypatch.setattr(
        server, "_start_embedded_cortex", lambda: pytest.fail("Cortex started without the flag")
    )
    with TestClient(server.app):
        pass


def test_lifespan_stops_the_worker_it_started(monkeypatch, no_db) -> None:
    class Worker:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    worker = Worker()
    monkeypatch.setenv("AMFS_WITH_CORTEX", "true")
    monkeypatch.setattr(
        server, "_start_embedded_cortex", lambda: setattr(server, "_cortex_worker", worker)
    )

    with TestClient(server.app):
        assert server._cortex_worker is worker
    assert worker.stopped, "shutdown must stop the per-process worker"


def test_start_embedded_cortex_is_idempotent(monkeypatch, no_db) -> None:
    sentinel = object()
    monkeypatch.setattr(server, "_cortex_worker", sentinel)
    monkeypatch.setenv("AMFS_POSTGRES_DSN", "postgresql://never-connected")
    server._start_embedded_cortex()  # must not try to build a second worker
    assert server._cortex_worker is sentinel


def test_start_embedded_cortex_needs_a_dsn(monkeypatch, no_db) -> None:
    server._start_embedded_cortex()
    assert server._cortex_worker is None


def test_main_hands_the_flag_to_the_workers_through_the_environment(monkeypatch, no_db) -> None:
    """``main()`` runs in the supervisor only; the children start their own Cortex."""
    monkeypatch.delenv("AMFS_WITH_CORTEX", raising=False)
    monkeypatch.setattr(
        server,
        "_parse_args",
        lambda: argparse.Namespace(
            host="127.0.0.1", port=0, reload=False, workers=2, with_cortex=True
        ),
    )
    monkeypatch.setattr(
        server, "_start_embedded_cortex", lambda: pytest.fail("main() must not start Cortex itself")
    )
    ran: dict = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: ran.update(k))

    server.main()

    assert os.environ.get("AMFS_WITH_CORTEX") == "1"
    assert ran.get("workers") == 2


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("YES", True), ("on", True),
                                          ("0", False), ("", False), ("false", False)])
def test_env_flag(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("AMFS_TEST_FLAG", raw)
    assert server._env_flag("AMFS_TEST_FLAG") is expected


# ── Model executor sized to the CPUs we may use ──────────────────────


def _fake_cgroup(monkeypatch, files: dict[str, str]) -> None:
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) in files:
            return io.StringIO(files[str(path)])
        if str(path).startswith("/sys/fs/cgroup/"):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


@pytest.fixture
def host_has_8_cpus(monkeypatch):
    monkeypatch.delenv("AMFS_MODEL_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)


def test_cgroup_v2_quota_wins_over_the_host_count(monkeypatch, host_has_8_cpus) -> None:
    _fake_cgroup(monkeypatch, {"/sys/fs/cgroup/cpu.max": "200000 100000\n"})
    assert server._effective_cpu_count() == 2


def test_fractional_quota_rounds_up(monkeypatch, host_has_8_cpus) -> None:
    _fake_cgroup(monkeypatch, {"/sys/fs/cgroup/cpu.max": "150000 100000\n"})
    assert server._effective_cpu_count() == 2


def test_unlimited_cgroup_falls_back_to_the_host(monkeypatch, host_has_8_cpus) -> None:
    _fake_cgroup(monkeypatch, {"/sys/fs/cgroup/cpu.max": "max 100000\n"})
    assert server._effective_cpu_count() == 8


def test_cgroup_v1_quota_is_read_too(monkeypatch, host_has_8_cpus) -> None:
    _fake_cgroup(monkeypatch, {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "400000\n",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
    })
    assert server._effective_cpu_count() == 4


def test_affinity_mask_caps_the_count(monkeypatch, host_has_8_cpus) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 3, 5}, raising=False)
    _fake_cgroup(monkeypatch, {})
    assert server._effective_cpu_count() == 3


def test_no_cgroup_and_no_affinity_uses_the_host(monkeypatch, host_has_8_cpus) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    _fake_cgroup(monkeypatch, {})
    assert server._effective_cpu_count() == 8


def test_override_wins_and_never_goes_below_one(monkeypatch, host_has_8_cpus) -> None:
    monkeypatch.setenv("AMFS_MODEL_THREADS", "3")
    assert server._effective_cpu_count() == 3
    monkeypatch.setenv("AMFS_MODEL_THREADS", "0")
    assert server._effective_cpu_count() == 1
    monkeypatch.setenv("AMFS_MODEL_THREADS", "lots")
    _fake_cgroup(monkeypatch, {"/sys/fs/cgroup/cpu.max": "100000 100000\n"})
    assert server._effective_cpu_count() == 1, "a bad override is ignored, not fatal"


# ── Sync-bodied routes run on the threadpool ─────────────────────────

SYNC_ROUTES = [
    "entry_quality", "get_usage", "list_api_keys", "create_api_key", "revoke_api_key",
    "list_audit_log", "list_teams", "create_team", "update_team", "delete_team",
    "list_team_members", "add_team_member", "update_team_member", "remove_team_member",
    "reinstate_team_member", "check_member_email", "list_detected_patterns",
    "run_pattern_scan", "resolve_pattern",
]


@pytest.mark.parametrize("name", SYNC_ROUTES)
def test_sync_bodied_routes_are_plain_def(name) -> None:
    fn = getattr(server, name)
    assert not inspect.iscoroutinefunction(fn), (
        f"{name} has no await; as async def its pool.connection() would hold the loop"
    )
    src = inspect.getsource(fn)
    assert "await " not in src, f"{name} gained an await; make it async def again"


def test_a_sync_route_runs_off_the_event_loop(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def pool_probe():
        seen["thread"] = threading.current_thread().name
        return None

    monkeypatch.setattr(server, "_get_db_pool", pool_probe)
    monkeypatch.setattr(server, "verify_api_key", lambda *a, **k: None)
    server.app.dependency_overrides[server.verify_api_key] = lambda: None
    try:
        with TestClient(server.app) as client:
            resp = client.get("/api/v1/admin/patterns")
    finally:
        server.app.dependency_overrides.pop(server.verify_api_key, None)

    assert resp.status_code == 200, resp.text
    assert seen["thread"].startswith("AnyIO worker thread"), seen


# ── The sync search fallback ─────────────────────────────────────────


def test_sync_search_tolerates_adapters_without_branch() -> None:
    class Old:
        def search(self, sq):
            return ["old"]

    class New:
        def search(self, sq, branch=None):
            return ["new", branch]

    assert server._sync_search(Old(), "q", "main") == ["old"]
    assert server._sync_search(New(), "q", "main") == ["new", "main"]
