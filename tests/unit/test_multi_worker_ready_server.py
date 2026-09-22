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
import ast
import builtins
import inspect
import io
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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
            host="127.0.0.1", port=0, reload=False, workers=2, with_cortex=True,
            worker_healthcheck_timeout=120,
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


# ── Worker health check ──────────────────────────────────────────────────
#
# uvicorn's multi-worker supervisor kills any worker that misses a ping for
# ``timeout_worker_healthcheck`` seconds (default 5) and respawns it. A fresh
# worker on a busy 2-vCPU instance takes longer than that to import the module
# and load the models, so with the default one slot per instance was killed
# every 8 seconds and never served a request (2026-09-21). The server must pass
# a realistic window through, and it must be tunable without a code change.


def test_worker_healthcheck_default_is_two_minutes(monkeypatch) -> None:
    monkeypatch.delenv("AMFS_HTTP_WORKER_HEALTHCHECK_S", raising=False)
    monkeypatch.setattr("sys.argv", ["amfs-http"])
    args = server._parse_args()
    assert args.worker_healthcheck_timeout == server.DEFAULT_WORKER_HEALTHCHECK_S == 120


def test_worker_healthcheck_reads_the_environment_and_the_flag(monkeypatch) -> None:
    monkeypatch.setenv("AMFS_HTTP_WORKER_HEALTHCHECK_S", "45")
    monkeypatch.setattr("sys.argv", ["amfs-http"])
    assert server._parse_args().worker_healthcheck_timeout == 45
    monkeypatch.setattr("sys.argv", ["amfs-http", "--worker-healthcheck-timeout", "7"])
    assert server._parse_args().worker_healthcheck_timeout == 7


def test_main_passes_the_healthcheck_window_to_uvicorn(monkeypatch, no_db) -> None:
    monkeypatch.setattr(
        server,
        "_parse_args",
        lambda: argparse.Namespace(
            host="127.0.0.1", port=0, reload=False, workers=2, with_cortex=False,
            worker_healthcheck_timeout=90,
        ),
    )
    # uvicorn really accepts it under that name: a typo would be swallowed by
    # the stand-in below and only surface as a TypeError in production.
    assert "timeout_worker_healthcheck" in inspect.signature(server.uvicorn.run).parameters
    ran: dict = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: ran.update(k))

    server.main()

    assert ran.get("timeout_worker_healthcheck") == 90


def test_the_uvicorn_floor_is_where_the_healthcheck_argument_appeared() -> None:
    """main() passes timeout_worker_healthcheck unconditionally; uvicorn grew it in 0.37.0.

    The lock resolves higher, so this guards the *declared* floor, which is
    what anyone installing the package against their own pins gets.
    """
    import re
    from pathlib import Path

    pyproject = (Path(server.__file__).parents[2] / "pyproject.toml").read_text()
    m = re.search(r'"uvicorn\[standard\]>=(\d+)\.(\d+)', pyproject)
    assert m, "uvicorn floor not declared"
    assert (int(m.group(1)), int(m.group(2))) >= (0, 37)


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


# Routes that are async without awaiting anything, on purpose: the health
# probes must answer even when every threadpool token is taken, and ``stream``
# subscribes an asyncio queue that has to be created on the serving loop.
LOOP_ROUTES_WITHOUT_AWAIT = {"root", "health", "health_v1", "stream"}


def _route_handlers() -> list[ast.AsyncFunctionDef | ast.FunctionDef]:
    tree = ast.parse(Path(inspect.getsourcefile(server)).read_text())
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and getattr(d.func, "attr", "") in (
                "get", "post", "put", "delete", "patch",
            ):
                out.append(node)
                break
    return out


def test_no_route_is_async_def_without_an_await() -> None:
    """The Agents page froze prod for 160 s per load through this hole.

    ``recompute_clusters``, ``list_agents_enriched`` and ``reuse_summary`` were
    ``async def`` with fully synchronous bodies, so a minutes-long compile ran
    on the event loop and every other request on the instance waited. An
    await-less ``async def`` route is always this bug: FastAPI would run the
    same body on the threadpool if it were a plain ``def``.
    """
    offenders = []
    for node in _route_handlers():
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if node.name in LOOP_ROUTES_WITHOUT_AWAIT:
            continue
        awaits = any(
            isinstance(x, (ast.Await, ast.AsyncFor, ast.AsyncWith))
            for x in ast.walk(node)
        )
        if not awaits:
            offenders.append(node.name)
    assert offenders == [], (
        "async def routes with no await run their whole body on the event loop; "
        f"make them plain def (or add them to LOOP_ROUTES_WITHOUT_AWAIT with a reason): {offenders}"
    )
    assert len(_route_handlers()) > 100, "the route scan found too little to be trusted"


def test_the_allowlisted_loop_routes_still_exist_and_are_async() -> None:
    names = {n.name for n in _route_handlers() if isinstance(n, ast.AsyncFunctionDef)}
    assert LOOP_ROUTES_WITHOUT_AWAIT <= names


# ── Cluster recompute: one compile at a time, reused within the window ────


class _ClusterCompileSpy:
    def __init__(self, calls: list, gate: threading.Event | None = None):
        self.calls, self.gate = calls, gate

    def __call__(self, *, adapter, namespace):
        return self

    def compile(self, scope):
        self.calls.append((scope, threading.current_thread().name))
        if self.gate is not None:
            self.gate.wait(5)


def _wire_recompute(monkeypatch, calls, gate=None):
    from amfs_postgres.adapter import PostgresAdapter

    adapter = object.__new__(PostgresAdapter)
    mem = SimpleNamespace(_adapter=adapter, namespace="acct-1")
    monkeypatch.setattr(server, "_get_memory", lambda: mem)
    import amfs_cortex.compiler as compiler_mod

    monkeypatch.setattr(compiler_mod, "DigestCompiler", _ClusterCompileSpy(calls, gate))
    server._cluster_recompute_last.clear()


def _req(account_id: str | None = None):
    state = SimpleNamespace()
    if account_id is not None:
        state.account_id = account_id
    return SimpleNamespace(state=state)


def test_recompute_is_a_plain_def_and_compiles_once_per_window(monkeypatch) -> None:
    calls: list = []
    _wire_recompute(monkeypatch, calls)
    assert not inspect.iscoroutinefunction(server.recompute_clusters)

    assert server.recompute_clusters(_req()) == {"ok": True}
    assert server.recompute_clusters(_req()) == {"ok": True, "skipped": "recent"}
    assert server.recompute_clusters(_req(), force=True) == {"ok": True}
    assert [c[0] for c in calls] == ["cluster:account:acct-1"] * 2


def test_recompute_window_expires(monkeypatch) -> None:
    calls: list = []
    _wire_recompute(monkeypatch, calls)
    monkeypatch.setattr(server, "_CLUSTER_RECOMPUTE_MIN_INTERVAL_S", 0.0)
    server.recompute_clusters(_req())
    server.recompute_clusters(_req())
    assert len(calls) == 2


def test_recompute_window_is_per_tenant_account(monkeypatch) -> None:
    """Multi-tenant prod runs every account under one namespace behind RLS.

    One account's compile must not mark the next account on the same instance
    as fresh; the tenant middleware's ``request.state.account_id`` keys the
    window, and a single-tenant server (no account on the request) falls back
    to the namespace.
    """
    calls: list = []
    _wire_recompute(monkeypatch, calls)
    assert server.recompute_clusters(_req("acct-A")) == {"ok": True}
    assert server.recompute_clusters(_req("acct-B")) == {"ok": True}
    assert server.recompute_clusters(_req("acct-A")) == {"ok": True, "skipped": "recent"}
    assert server.recompute_clusters(_req("acct-B")) == {"ok": True, "skipped": "recent"}
    assert server.recompute_clusters(_req()) == {"ok": True}, "no account: its own window"
    assert len(calls) == 3


def test_concurrent_recomputes_do_not_stack(monkeypatch) -> None:
    """Two Agents-page mounts at once: one compile runs, the other reuses it."""
    calls: list = []
    gate = threading.Event()
    _wire_recompute(monkeypatch, calls, gate)
    results: list = []
    t1 = threading.Thread(target=lambda: results.append(server.recompute_clusters(_req())))
    t1.start()
    for _ in range(100):
        if calls:
            break
        time.sleep(0.01)
    t2 = threading.Thread(target=lambda: results.append(server.recompute_clusters(_req())))
    t2.start()
    time.sleep(0.05)
    assert len(calls) == 1, "the second call must wait for the first, not compile alongside it"
    gate.set()
    t1.join(5); t2.join(5)
    assert len(calls) == 1
    assert sorted(r.get("skipped", "") for r in results) == ["", "recent"]


# ── Acting for an agent never touches the shared tagger ─────────────────────


def test_no_route_swaps_the_shared_tagger() -> None:
    """The race Bugbot named on #440.

    ``mem._tagger.agent_id = X ... finally: restore`` was atomic only while the
    body ran on the event loop. On the threadpool two requests interleave and
    a commit is signed with a neighbour's identity. Every route that acts for
    an agent goes through ``AgentMemory.as_agent`` (a per-request handle with
    its own tagger) instead; ``write_entry`` sets the session on *its handle*,
    which is the one assignment allowed.
    """
    src = Path(inspect.getsourcefile(server)).read_text()
    assert "_tagger.agent_id =" not in src
    assert "mem._tagger.session_id =" not in src
    assert src.count("_tagger.session_id =") == 2, "only per-request handles set a session"


def test_commit_acts_through_a_handle_not_the_shared_tagger(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _Tx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, *a, **k):
            seen["wrote"] = a

        commit = SimpleNamespace(id="c-1", model_dump=lambda mode="json": {"id": "c-1"})
        entries = ["one"]

    class _Handle:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self._tagger = SimpleNamespace(agent_id=agent_id, session_id="sess-h")

        def transaction(self, message):
            seen["tx"] = (self._tagger.agent_id, self._tagger.session_id, message)
            return _Tx()

    shared = SimpleNamespace(
        agent_id="http-server",
        namespace="default",
        _tagger=SimpleNamespace(agent_id="http-server", session_id="sess-0"),
        _adapter=SimpleNamespace(ensure_agent=lambda *a, **k: None),
        as_agent=lambda aid: _Handle(aid),
        transaction=lambda message: pytest.fail("the shared handle must not commit"),
    )
    monkeypatch.setattr(server, "_get_memory", lambda: shared)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    body = {
        "message": "m", "agent_id": "alice", "session_id": "sess-9",
        "writes": [{"entity_path": "acme/x", "key": "k", "value": "v"}],
    }
    server.create_commit(body, SimpleNamespace(headers={}), None)

    assert seen["tx"] == ("alice", "sess-9", "m")
    assert seen["wrote"] == ("acme/x", "k", "v")
    assert (shared._tagger.agent_id, shared._tagger.session_id) == ("http-server", "sess-0")


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


# ── Sync routes may build the singleton concurrently ─────────────────


def test_get_memory_builds_exactly_one_instance_under_concurrent_first_calls(monkeypatch) -> None:
    """Threadpool routes racing into _get_memory must share one AgentMemory."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    built: list[object] = []

    def slow_build():
        # Widen the window: without the lock every racer gets past the
        # ``is None`` check before the first one assigns.
        time.sleep(0.05)
        obj = object()
        built.append(obj)
        server._memory = obj  # what the real builder does
        return obj

    monkeypatch.setattr(server, "_memory", None)
    monkeypatch.setattr(server, "_build_memory", slow_build)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: server._get_memory(), range(8)))

    assert len(built) == 1, f"built {len(built)} AgentMemory instances; the lock is not held"
    assert all(r is built[0] for r in results)


# ── SSE broadcasts from threadpool threads reach loop-bound subscribers ──


def test_room_broadcast_from_a_worker_thread_wakes_the_subscriber() -> None:
    """The sync rooms routes broadcast from the threadpool; the loop must wake."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from amfs_http.sse import SSEManager

    async def scenario() -> tuple[dict, str]:
        mgr = SSEManager()
        gen = mgr.room_event_generator("room-1")
        first = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0)  # let the generator subscribe

        loop_thread = threading.current_thread().name
        seen: dict[str, str] = {}

        def from_thread() -> None:
            seen["thread"] = threading.current_thread().name
            mgr.broadcast_room_event("room-1", "discussion_message", {"agent_id": "a"})

        with ThreadPoolExecutor(max_workers=1) as pool:
            await asyncio.get_running_loop().run_in_executor(pool, from_thread)

        assert seen["thread"] != loop_thread, "test must broadcast off the loop"
        # A put_nowait from the other thread leaves the loop asleep; only a
        # threadsafe hand-off delivers within a bounded wait with nothing
        # else happening on the loop.
        event = await asyncio.wait_for(first, timeout=2.0)
        await gen.aclose()
        return event, seen["thread"]

    event, _ = asyncio.run(scenario())
    assert event["event"] == "discussion_message"
    assert '"agent_id": "a"' in event["data"]


def test_broadcast_on_the_loop_is_still_direct() -> None:
    import asyncio

    from amfs_http.sse import SSEManager

    async def scenario() -> dict:
        mgr = SSEManager()
        queue = mgr.subscribe_room("r")
        mgr.broadcast_room_event("r", "join", {"user_id": "u"})
        return queue.get_nowait()  # already there, no loop turn needed

    assert asyncio.run(scenario())["type"] == "join"


def test_unsubscribe_forgets_the_queue_loop() -> None:
    import asyncio

    from amfs_http.sse import SSEManager

    async def scenario() -> int:
        mgr = SSEManager()
        q = mgr.subscribe("*")
        assert len(mgr._loops) == 1
        mgr.unsubscribe("*", q)
        return len(mgr._loops)

    assert asyncio.run(scenario()) == 0


def test_a_subscriber_whose_loop_is_gone_is_skipped_quietly() -> None:
    import asyncio

    from amfs_http.sse import SSEManager

    mgr = SSEManager()

    async def subscribe():
        return mgr.subscribe_room("r")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(subscribe())
    finally:
        loop.close()
    # The subscriber's loop is closed: broadcasting must neither raise nor hang.
    mgr.broadcast_room_event("r", "join", {})


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
