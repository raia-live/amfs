"""Tier 2 of the repair loop, served by the harness: the customer's replay endpoint.

When the platform has drafted a fix whose Tier 1 stand-in replay cannot vouch for it
(``test_inconclusive``: the record holds the outcome of the action the agent took, not of
the one the fix recommends), it asks the *agent's own* runtime to replay the failing cases
with memory on the fix's branch — one signed webhook per case, the OSS SDK's
``amfs.replay`` contract. In production that endpoint is the customer's; in the benchmark
it is this module: one HTTP server per runner process, reachable from the SaaS through an
ngrok tunnel, dispatching each request to the cell whose agent it names. The cell re-runs
the episode the case came from — same task, same environment, same model, memory on the
branch — and commits the outcome with ``case_id`` / ``fix_id`` / ``memory_branch`` so the
platform grades it (HYPOTHESIS_RSI.md, amendment (e): E2b promoted to the E1 mechanism).

Environment:

``CL_RSI_TIER2``          ``1`` to serve the endpoint and register it on every RSI cell's agent.
``CL_REPLAY_PORT``        local port (default 8787).
``CL_REPLAY_PUBLIC_URL``  the public base URL (``https://…``) if a tunnel is already up;
                          otherwise ``ngrok http <port>`` is started and its URL read from the
                          local ngrok API.
``CL_REPLAY_SECRET``      the webhook secret; a random one per process when unset. The secret
                          is registered with the platform and never written to results.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request, urlopen

from amfs.replay import EVENT_PING, ReplayError, ReplayRequest, parse_replay_request

log = logging.getLogger(__name__)

PATH = "/amfs/replay"
SEEN = 4096


def tier2_enabled() -> bool:
    return os.environ.get("CL_RSI_TIER2", "0").strip().lower() in ("1", "true", "yes")


class ReplayHub:
    """One per process. ``get()`` starts it on first use; ``register`` binds an agent id to
    the arm that answers for it (the cell), ``handle`` is the endpoint."""

    _instance: ReplayHub | None = None
    _guard = threading.Lock()

    @classmethod
    def get(cls) -> ReplayHub:
        with cls._guard:
            if cls._instance is None:
                hub = cls()
                hub.start()
                cls._instance = hub
            return cls._instance

    def __init__(self) -> None:
        self.port = int(os.environ.get("CL_REPLAY_PORT", "8787") or 8787)
        self.secret = os.environ.get("CL_REPLAY_SECRET") or secrets.token_urlsafe(32)
        self.public_base = (os.environ.get("CL_REPLAY_PUBLIC_URL") or "").rstrip("/")
        self.url: str = ""
        self._arms: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._seen: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._pool = ThreadPoolExecutor(max_workers=int(os.environ.get("CL_REPLAY_WORKERS", "6") or 6),
                                        thread_name_prefix="cl-replay")
        self._server: ThreadingHTTPServer | None = None
        self._ngrok: subprocess.Popen[bytes] | None = None
        self.stats: dict[str, int] = {"received": 0, "pings": 0, "accepted": 0, "duplicates": 0,
                                      "refused": 0, "unknown_agent": 0, "past_deadline": 0}

    # -- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "cl-harness-replay/1"

            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("replay endpoint: " + fmt, *args)

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] != PATH:
                    self._send(404, {"ok": False, "error": "not found"})
                    return
                self._send(200, {"ok": True, "receiver": "cl-harness", "agents": len(hub._arms)})

            def do_POST(self) -> None:  # noqa: N802
                if self.path.split("?", 1)[0] != PATH:
                    self._send(404, {"ok": False, "error": "not found"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                status, payload = hub.handle(self.headers, body)
                self._send(status, payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        try:
            if not self.public_base:
                self.public_base = self._tunnel()
        except Exception:
            # Release the port: a second attempt (the runner retries a failed cell) must not
            # die on "address already in use" behind the real error.
            self._server.server_close()
            self._server = None
            raise
        threading.Thread(target=self._server.serve_forever, name="cl-replay-http", daemon=True).start()
        self.url = f"{self.public_base}{PATH}"
        log.info("replay endpoint up: %s -> 127.0.0.1:%d", self.url, self.port)

    # -- the tunnel -----------------------------------------------------------------------
    _API_PORTS = (4040, 4041, 4042)

    def _tunnels(self) -> list[tuple[int, dict[str, Any]]]:
        """Every tunnel any local ngrok agent reports, with the API port it came from."""
        out: list[tuple[int, dict[str, Any]]] = []
        for api_port in self._API_PORTS:
            try:
                with urlopen(f"http://127.0.0.1:{api_port}/api/tunnels", timeout=2) as r:  # noqa: S310
                    out.extend((api_port, t) for t in (json.load(r).get("tunnels") or []))
            except Exception:  # noqa: BLE001
                continue
        return out

    def _ours(self, tunnels: list[tuple[int, dict[str, Any]]]) -> str | None:
        for _, t in tunnels:
            addr = str((t.get("config") or {}).get("addr") or "")
            if addr.endswith(f":{self.port}") and str(t.get("public_url", "")).startswith("https://"):
                return str(t["public_url"]).rstrip("/")
        return None

    def _tunnel(self) -> str:
        """The https URL that reaches this port. A free ngrok account runs one agent at a
        time, so: reuse a tunnel an agent already has to this port; else ask a running agent
        to open one (its local API); else start ``ngrok http <port>`` and read its URL."""
        existing = self._tunnels()
        url = self._ours(existing)
        if url:
            return url
        if existing:
            api_port = existing[0][0]
            body = json.dumps({"addr": str(self.port), "proto": "http", "name": f"cl-replay-{self.port}"}).encode()
            req = Request(f"http://127.0.0.1:{api_port}/api/tunnels", data=body,  # noqa: S310
                          headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(req, timeout=10) as r:  # noqa: S310
                    t = json.load(r)
                if str(t.get("public_url", "")).startswith("https://"):
                    return str(t["public_url"]).rstrip("/")
            except Exception as exc:  # noqa: BLE001
                others = sorted({str((t.get("config") or {}).get("addr")) for _, t in existing})
                raise RuntimeError(
                    f"an ngrok agent is already running with tunnel(s) to {others} and would not add one "
                    f"for :{self.port} ({type(exc).__name__}); stop it, or set CL_REPLAY_PUBLIC_URL"
                ) from exc
        self._ngrok = subprocess.Popen(["ngrok", "http", str(self.port), "--log=stdout", "--log-format=json"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._ngrok.poll() is not None:
                raise RuntimeError(f"ngrok exited with {self._ngrok.returncode} (another agent session on the account?)")
            url = self._ours(self._tunnels())
            if url:
                return url
            time.sleep(0.5)
        raise RuntimeError("ngrok did not report an https tunnel within 30 s (CL_REPLAY_PUBLIC_URL to bypass)")

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        self._pool.shutdown(wait=False)
        if self._ngrok is not None:
            self._ngrok.terminate()

    # -- registry -----------------------------------------------------------------------
    def register(self, agent_id: str, arm: Any) -> None:
        with self._lock:
            self._arms[agent_id] = arm

    def unregister(self, agent_id: str) -> None:
        with self._lock:
            self._arms.pop(agent_id, None)

    # -- the endpoint -----------------------------------------------------------------
    def handle(self, headers: Any, body: bytes) -> tuple[int, dict[str, Any]]:
        self.stats["received"] += 1
        try:
            request = parse_replay_request(headers, body, secret=self.secret)
        except ReplayError as exc:
            self.stats["refused"] += 1
            return exc.status, {"ok": False, "error": str(exc), "code": exc.code}
        if request.event == EVENT_PING:
            self.stats["pings"] += 1
            return 200, {"ok": True, "event": EVENT_PING, "agent_id": request.agent_id}
        with self._lock:
            prior = self._seen.get(request.delivery_id)
            if prior is not None:
                self.stats["duplicates"] += 1
                return 200, {**prior, "duplicate": True}
            arm = self._arms.get(request.agent_id)
            if arm is None:
                # The cell is over (or never ran here): nothing can replay this case. A 4xx
                # other than 429 tells the sender not to retry.
                self.stats["unknown_agent"] += 1
                return 404, {"ok": False, "code": "unknown_agent", "agent_id": request.agent_id}
            if request.past_deadline():
                self.stats["past_deadline"] += 1
                return 410, {"ok": False, "code": "past_deadline", "case_id": request.case_id}
            answer = {"ok": True, "accepted": True, "delivery_id": request.delivery_id,
                      "case_id": request.case_id, "branch": request.branch}
            self._seen[request.delivery_id] = answer
            while len(self._seen) > SEEN:
                self._seen.popitem(last=False)
            self.stats["accepted"] += 1
        self._pool.submit(self._run, arm, request)
        return 202, answer

    def _run(self, arm: Any, request: ReplayRequest) -> None:
        try:
            outcome = arm.replay(request)
        except Exception as exc:  # noqa: BLE001 - never let a worker die silently
            log.exception("replay of case %s failed", request.case_id)
            outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        with self._lock:
            prior = self._seen.get(request.delivery_id) or {}
            self._seen[request.delivery_id] = {**prior, "ran": True, **(outcome or {})}

    def report(self) -> dict[str, Any]:
        from urllib.parse import urlsplit

        return {"webhook_host": urlsplit(self.url).hostname if self.url else None, "port": self.port,
                "agents": len(self._arms), **self.stats}


__all__ = ["PATH", "ReplayHub", "tier2_enabled"]
