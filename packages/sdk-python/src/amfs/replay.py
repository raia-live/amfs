"""Answer a replay request: the customer's end of the repair loop's Tier 2.

When SenseLab has drafted a fix for one of your agent's behaviours and wants
proof from *your* agent before it ships, it sends one signed webhook per test
case::

    POST <your replay URL>
    X-AMFS-Event: replay_requested
    X-AMFS-Delivery: <uuid, unique per request>
    X-AMFS-Signature: sha256=<hex HMAC-SHA256(secret, raw body)>

    {"event": "replay_requested", "fix_id": ..., "agent_id": ..., "branch":
     "repair/<fix-id>", "case_id": ..., "task_input": ..., "expected": {...},
     "deadline_at": ..., ...}

Your side has three obligations, and this module discharges all of them so a
receiver is a handful of lines:

1. **Verify the signature** over the raw body (:func:`verify_replay_signature`).
2. **Run the agent on ``task_input`` with its memory on ``branch``** — the
   branch holds the fix; a run on ``main`` never read it and proves nothing.
3. **Commit the outcome with ``attributes.case_id`` and
   ``attributes.memory_branch``** so the grader can find the trace. The
   :class:`ReplayReceiver` sets both, and an ``AgentMemory`` on a branch
   stamps ``memory_branch`` on every commit regardless.

The sender waits ten seconds and retries a non-2xx a bounded number of times,
so the receiver acknowledges with ``202`` at once and runs the agent on a
worker thread; delivery is at-least-once, so a delivery id seen before is
acknowledged again without a second run. A ``ping`` event, sent from the
settings page to check the endpoint, is answered ``200``.

Framework-agnostic: :meth:`ReplayReceiver.handle` takes headers and the raw
body and returns a status and a JSON-able dict, so it mounts in Flask,
FastAPI, Django or the stdlib server :func:`serve` starts (``amfs replay
serve`` on the command line).

Example::

    from amfs import AgentMemory
    from amfs.replay import ReplayReceiver

    def run(task_input: str, memory: AgentMemory) -> str:
        return my_agent.answer(task_input, memory=memory)   # reads on the branch

    receiver = ReplayReceiver(secret=os.environ["AMFS_REPLAY_SECRET"], run=run)

    @app.post("/amfs/replay")                                # any framework
    def replay():
        status, body = receiver.handle(dict(request.headers), request.get_data())
        return jsonify(body), status
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from amfs_core.models import OutcomeType

from amfs.memory import MEMORY_BRANCH_ATTRIBUTE

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-AMFS-Signature"
EVENT_HEADER = "X-AMFS-Event"
DELIVERY_HEADER = "X-AMFS-Delivery"

EVENT_REPLAY_REQUESTED = "replay_requested"
EVENT_PING = "ping"

#: The attributes the grader reads a replayed trace through. Both are
#: required: the case says which test this run answers, the branch that it
#: read the fix.
CASE_ID_ATTRIBUTE = "case_id"
FIX_ID_ATTRIBUTE = "fix_id"
DELIVERY_ID_ATTRIBUTE = "replay_delivery_id"

#: How many delivery ids the receiver remembers for de-duplication. A replay
#: is at most a few dozen cases and a redelivery comes within minutes; the
#: window is generous for that and bounded so a long-lived process does not
#: grow without limit.
SEEN_DELIVERIES = 4096

#: How long a replay run may take before the receiver gives up on it and
#: commits a failure so the case is graded rather than left missing.
DEFAULT_RUN_TIMEOUT_SECONDS = 600.0


class ReplayError(Exception):
    """A request the receiver refuses. ``status`` is the HTTP status to
    answer with — a 4xx other than 429 tells the sender not to retry."""

    def __init__(self, status: int, message: str, *, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class SignatureError(ReplayError):
    """The signature is missing or does not match the body."""

    def __init__(self, message: str = "signature missing or invalid") -> None:
        super().__init__(401, message, code="bad_signature")


class PayloadError(ReplayError):
    """The body is not a replay request the receiver understands."""

    def __init__(self, message: str) -> None:
        super().__init__(400, message, code="bad_payload")


def sign_replay_body(secret: str, body: bytes) -> str:
    """The ``X-AMFS-Signature`` value for *body*: ``sha256=<hex>``. The same
    function the sender uses; here for simulations and tests."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_replay_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Whether *header* is the signature of *body* under *secret*. Compared in
    constant time. A missing or malformed header is ``False``, never an error."""
    if not header or not isinstance(header, str):
        return False
    scheme, _, digest = header.strip().partition("=")
    if scheme.lower() != "sha256" or not digest:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, digest.strip().lower())


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    """Case-insensitive header lookup over whatever mapping the framework
    hands over (a dict, a WSGI environ-derived object, ``request.headers``)."""
    if hasattr(headers, "get"):
        direct = headers.get(name)
        if direct is not None:
            return str(direct)
    wanted = name.lower()
    for key in headers:
        if str(key).lower() == wanted:
            value = headers[key]
            return None if value is None else str(value)
    return None


def _parse_when(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True)
class ReplayRequest:
    """One verified replay request, as the sender framed it."""

    event: str
    delivery_id: str
    fix_id: str
    agent_id: str
    branch: str
    case_id: str
    case_set_id: str | None
    task_input: str
    expected: dict[str, Any] = field(default_factory=dict)
    deadline_at: datetime | None = None
    sent_at: datetime | None = None
    instructions: str | None = None

    @property
    def outcome_ref(self) -> str:
        """The outcome reference the replay commits under."""
        return f"replay:{self.fix_id[:8]}:{self.case_id[:8]}"

    @property
    def attributes(self) -> dict[str, str]:
        """The session attributes the committed outcome must carry."""
        return {
            CASE_ID_ATTRIBUTE: self.case_id,
            MEMORY_BRANCH_ATTRIBUTE: self.branch,
            FIX_ID_ATTRIBUTE: self.fix_id,
            DELIVERY_ID_ATTRIBUTE: self.delivery_id,
        }

    def past_deadline(self, now: datetime | None = None) -> bool:
        if self.deadline_at is None:
            return False
        return (now or datetime.now(UTC)) >= self.deadline_at


def parse_replay_request(
    headers: Mapping[str, Any], body: bytes, *, secret: str | None
) -> ReplayRequest:
    """Verify and parse one delivery. Raises :class:`SignatureError` when
    *secret* is set and the signature does not match, :class:`PayloadError`
    when the body is not a request the receiver can act on.

    ``secret=None`` skips verification — for a receiver behind its own
    authentication, and for local simulations. A registered webhook without
    a secret is sent unsigned, so the two must agree.
    """
    if secret:
        if not verify_replay_signature(secret, body, _header(headers, SIGNATURE_HEADER)):
            raise SignatureError()
    try:
        payload = json.loads(body.decode("utf-8") if isinstance(body, bytes | bytearray) else body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PayloadError(f"body is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PayloadError("body must be a JSON object")

    event = str(payload.get("event") or _header(headers, EVENT_HEADER) or "").strip()
    delivery_id = str(payload.get("delivery_id") or _header(headers, DELIVERY_HEADER) or "").strip()
    if event == EVENT_PING:
        return ReplayRequest(
            event=EVENT_PING,
            delivery_id=delivery_id,
            fix_id="",
            agent_id=str(payload.get("agent_id") or ""),
            branch="",
            case_id="",
            case_set_id=None,
            task_input="",
            sent_at=_parse_when(payload.get("sent_at")),
        )
    if event != EVENT_REPLAY_REQUESTED:
        raise PayloadError(f"unknown event {event!r}")

    missing = [k for k in ("fix_id", "branch", "case_id", "task_input") if not payload.get(k)]
    if not delivery_id:
        missing.insert(0, "delivery_id")
    if missing:
        raise PayloadError("replay request is missing " + ", ".join(missing))
    branch = str(payload["branch"]).strip()
    if branch == "main":
        # The whole point of a replay is a run that read the fix. A request
        # naming main is not one the sender makes; refuse it rather than
        # commit a trace the grader would take for a repaired run.
        raise PayloadError("a replay cannot run on main")
    expected = payload.get("expected")
    return ReplayRequest(
        event=EVENT_REPLAY_REQUESTED,
        delivery_id=delivery_id,
        fix_id=str(payload["fix_id"]),
        agent_id=str(payload.get("agent_id") or ""),
        branch=branch,
        case_id=str(payload["case_id"]),
        case_set_id=str(payload["case_set_id"]) if payload.get("case_set_id") else None,
        task_input=str(payload["task_input"]),
        expected=dict(expected) if isinstance(expected, dict) else {},
        deadline_at=_parse_when(payload.get("deadline_at")),
        sent_at=_parse_when(payload.get("sent_at")),
        instructions=str(payload["instructions"]) if payload.get("instructions") else None,
    )


@dataclass
class ReplayResult:
    """What a run hands back. A runner may return this, a plain string (the
    agent's answer, taken as a success), or a ``(answer, outcome_type)`` pair."""

    response_text: str | None = None
    outcome_type: OutcomeType = OutcomeType.SUCCESS
    tool_calls: list[dict[str, Any]] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> ReplayResult:
        if isinstance(value, ReplayResult):
            return value
        if value is None:
            return cls(response_text=None)
        if isinstance(value, str):
            return cls(response_text=value)
        if isinstance(value, tuple) and len(value) == 2:
            text, outcome = value
            return cls(
                response_text=None if text is None else str(text), outcome_type=_outcome(outcome)
            )
        if isinstance(value, dict):
            return cls(
                response_text=value.get("response_text"),
                outcome_type=_outcome(value.get("outcome_type", OutcomeType.SUCCESS)),
                tool_calls=value.get("tool_calls"),
                attributes=dict(value.get("attributes") or {}),
            )
        return cls(response_text=str(value))


def _outcome(value: Any) -> OutcomeType:
    if isinstance(value, OutcomeType):
        return value
    if isinstance(value, bool):
        return OutcomeType.SUCCESS if value else OutcomeType.FAILURE
    return OutcomeType(str(value))


Runner = Callable[[str, Any], Any]
"""``run(task_input, memory) -> ReplayResult | str | (str, outcome) | dict``.

*memory* is an ``AgentMemory`` already on the request's branch; read through
it and the run reads the fix. Return the agent's answer; the receiver commits
it. Raise, and the receiver commits a failure with the error as the answer so
the case is graded rather than left missing."""

MemoryFactory = Callable[[ReplayRequest], Any]
"""``memory_factory(request) -> AgentMemory`` on ``request.branch``. The
default builds ``AgentMemory(agent_id=request.agent_id or agent_id,
branch=request.branch)`` from the process's configuration."""


class ReplayReceiver:
    """The receiving end of the replay webhook.

    :param secret: the shared secret registered with the webhook; ``None``
        accepts unsigned requests (see :func:`parse_replay_request`).
    :param run: the :data:`Runner` that answers a case.
    :param memory_factory: how to build the memory for a request; defaults to
        an ``AgentMemory`` on the request's branch.
    :param agent_id: the agent id for the default factory when the request
        names none.
    :param background: acknowledge with 202 and run on a worker thread
        (default). ``False`` runs inline and answers 200 with the outcome —
        for tests and for agents that finish inside the sender's timeout.
    :param run_timeout: seconds a background run may take before the receiver
        commits a failure for it (``replay_error="timeout"``) so the case is
        still graded. The runner cannot be interrupted: it keeps its thread,
        and an answer it gives after the deadline is discarded rather than
        committed over the failure. Inline runs (``background=False``) are
        not bounded — they run inside the sender's request. ``None`` or
        ``0`` disables the bound.
    """

    def __init__(
        self,
        *,
        secret: str | None,
        run: Runner,
        memory_factory: MemoryFactory | None = None,
        agent_id: str | None = None,
        background: bool = True,
        run_timeout: float = DEFAULT_RUN_TIMEOUT_SECONDS,
        max_workers: int = 4,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._secret = secret
        self._run = run
        self._memory_factory = memory_factory or self._default_memory
        # Only what the receiver opened is the receiver's to close.
        self._owns_memory = memory_factory is None
        self._agent_id = agent_id
        self._background = background
        self._run_timeout = run_timeout
        self._clock = clock or (lambda: datetime.now(UTC))
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="amfs-replay")
            if background
            else None
        )
        self._seen: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._inflight = 0

    # -- the HTTP-facing entry point -------------------------------------

    def handle(self, headers: Mapping[str, Any], body: bytes) -> tuple[int, dict[str, Any]]:
        """Answer one HTTP request. Returns ``(status, json_body)``.

        * ``ping`` → 200.
        * a signature that does not match → 401 (the sender stops retrying).
        * a body the receiver cannot act on → 400.
        * a request past its deadline → 410 (too late to be graded).
        * a delivery seen before → 200 with the first answer, no second run.
        * otherwise → 202 and the run starts (or 200 with its outcome when
          the receiver runs inline).
        """
        try:
            request = parse_replay_request(headers, body, secret=self._secret)
        except ReplayError as exc:
            logger.info("replay request refused: %s", exc)
            return exc.status, {"ok": False, "error": str(exc), "code": exc.code}

        if request.event == EVENT_PING:
            return 200, {"ok": True, "event": EVENT_PING, "agent_id": request.agent_id}

        with self._lock:
            prior = self._seen.get(request.delivery_id)
            if prior is not None:
                return 200, {**prior, "duplicate": True}
            if request.past_deadline(self._clock()):
                return 410, {
                    "ok": False,
                    "code": "past_deadline",
                    "case_id": request.case_id,
                    "error": f"deadline {request.deadline_at.isoformat()} has passed",  # type: ignore[union-attr]
                }
            answer: dict[str, Any] = {
                "ok": True,
                "accepted": True,
                "delivery_id": request.delivery_id,
                "case_id": request.case_id,
                "branch": request.branch,
            }
            self._remember(request.delivery_id, answer)

        if self._executor is None:
            outcome = self.replay(request)
            answer = {**answer, "accepted": False, "ran": True, **outcome}
            with self._lock:
                self._remember(request.delivery_id, answer)
            return 200, answer

        with self._lock:
            self._inflight += 1
        self._executor.submit(self._run_in_background, request)
        return 202, answer

    # -- the run ----------------------------------------------------------

    def replay(self, request: ReplayRequest, *, timeout: float | None = None) -> dict[str, Any]:
        """Run one case and commit its outcome on the request's branch. Returns
        a summary (``outcome_type``, ``outcome_ref``, ``trace_id`` when the
        adapter reports one). A runner that raises is committed as a failure
        with the error as the answer, and one that outlasts *timeout* seconds
        as a failure with ``replay_error="timeout"``: a graded failure tells
        the loop more than a case that never came back."""
        memory = self._memory_factory(request)
        if hasattr(memory, "checkout") and getattr(memory, "branch", None) != request.branch:
            memory.checkout(request.branch)
        result = self._answer(request, memory, timeout)
        attributes = {**result.attributes, **request.attributes}
        try:
            memory.commit_outcome(
                request.outcome_ref,
                result.outcome_type,
                task_input=request.task_input,
                response_text=result.response_text,
                tool_calls=result.tool_calls,
                attributes=attributes,
            )
        finally:
            closer = getattr(memory, "close", None)
            if callable(closer) and self._owns_memory:
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    logger.debug("closing the replay memory failed", exc_info=True)
        trace = getattr(memory, "_last_trace", None)
        trace_id = getattr(trace, "id", None) or getattr(trace, "trace_id", None)
        return {
            "outcome_type": result.outcome_type.value,
            "outcome_ref": request.outcome_ref,
            "trace_id": str(trace_id) if trace_id else None,
        }

    def _call_runner(self, request: ReplayRequest, memory: Any) -> ReplayResult:
        """The runner's answer, coerced; a raise is an answer too."""
        try:
            return ReplayResult.coerce(self._run(request.task_input, memory))
        except Exception as exc:  # noqa: BLE001 - the runner is the customer's code
            logger.warning("replay of case %s raised", request.case_id, exc_info=True)
            return ReplayResult(
                response_text=f"replay runner raised {type(exc).__name__}: {exc}"[:2000],
                outcome_type=OutcomeType.FAILURE,
                attributes={"replay_error": type(exc).__name__},
            )

    def _answer(self, request: ReplayRequest, memory: Any, timeout: float | None) -> ReplayResult:
        """The runner's answer within *timeout* seconds, or a failure that says
        it did not come. The runner runs on its own daemon thread when bounded:
        a thread cannot be interrupted, so a late answer is left where it is
        and never committed over the failure already recorded."""
        if not timeout or timeout <= 0:
            return self._call_runner(request, memory)
        box: dict[str, ReplayResult] = {}

        def target() -> None:
            box["result"] = self._call_runner(request, memory)

        worker = threading.Thread(
            target=target, name=f"amfs-replay-run-{request.case_id[:8]}", daemon=True
        )
        worker.start()
        worker.join(timeout)
        if worker.is_alive() or "result" not in box:
            logger.warning(
                "replay of case %s did not finish within %ss; committing a failure",
                request.case_id, timeout,
            )
            return ReplayResult(
                response_text=f"replay runner did not finish within {timeout:g}s",
                outcome_type=OutcomeType.FAILURE,
                attributes={"replay_error": "timeout"},
            )
        return box["result"]

    def _run_in_background(self, request: ReplayRequest) -> None:
        try:
            outcome = self.replay(request, timeout=self._run_timeout)
            with self._lock:
                prior = self._seen.get(request.delivery_id) or {}
                self._remember(request.delivery_id, {**prior, "ran": True, **outcome})
        except Exception:  # noqa: BLE001 - never let a worker die silently
            logger.exception("replay of case %s could not be committed", request.case_id)
            with self._lock:
                prior = self._seen.get(request.delivery_id) or {}
                self._remember(
                    request.delivery_id, {**prior, "ran": False, "error": "commit failed"}
                )
        finally:
            with self._lock:
                self._inflight -= 1

    def drain(self, timeout: float | None = None) -> bool:
        """Wait for background runs to finish. ``True`` when none are left."""
        if self._executor is None:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if self._inflight == 0:
                    return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def status(self, delivery_id: str) -> dict[str, Any] | None:
        """What became of a delivery, if the receiver still remembers it."""
        with self._lock:
            return dict(self._seen[delivery_id]) if delivery_id in self._seen else None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    # -- internals ----------------------------------------------------------

    def _remember(self, delivery_id: str, answer: dict[str, Any]) -> None:
        self._seen[delivery_id] = answer
        self._seen.move_to_end(delivery_id)
        while len(self._seen) > SEEN_DELIVERIES:
            self._seen.popitem(last=False)

    def _default_memory(self, request: ReplayRequest) -> Any:
        from amfs.memory import AgentMemory

        return AgentMemory(
            agent_id=request.agent_id or self._agent_id or "replay", branch=request.branch
        )


# ---------------------------------------------------------------------------
# A stdlib server, for ``amfs replay serve`` and for agents without a web
# framework of their own.
# ---------------------------------------------------------------------------


def serve(
    receiver: ReplayReceiver,
    *,
    host: str = "0.0.0.0",
    port: int = 8787,
    path: str = "/amfs/replay",
    ready: Callable[[str], None] | None = None,
) -> None:
    """Serve *receiver* at ``http://host:port{path}`` until interrupted. Any
    other path is 404; ``GET {path}`` answers 200 so a load balancer can
    check the process is up. Blocks; :func:`make_server` for a handle."""
    server = make_server(receiver, host=host, port=port, path=path)
    if ready is not None:
        ready(f"http://{server.server_address[0]}:{server.server_address[1]}{path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        receiver.drain(timeout=receiver._run_timeout)
        receiver.close()


def make_server(
    receiver: ReplayReceiver, *, host: str = "127.0.0.1", port: int = 0, path: str = "/amfs/replay"
):
    """A ``ThreadingHTTPServer`` mounting *receiver* at *path*. ``port=0``
    picks a free one; read ``server.server_address``."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        server_version = "amfs-replay/1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            logger.debug("replay server: " + fmt, *args)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path.split("?", 1)[0] != path:
                self._send(404, {"ok": False, "error": "not found"})
                return
            self._send(200, {"ok": True, "receiver": "amfs-replay", "path": path})

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            if self.path.split("?", 1)[0] != path:
                self._send(404, {"ok": False, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, payload = receiver.handle(self.headers, body)
            self._send(status, payload)

    return ThreadingHTTPServer((host, port), Handler)


__all__ = [
    "CASE_ID_ATTRIBUTE",
    "DELIVERY_HEADER",
    "EVENT_HEADER",
    "EVENT_PING",
    "EVENT_REPLAY_REQUESTED",
    "MEMORY_BRANCH_ATTRIBUTE",
    "PayloadError",
    "ReplayError",
    "ReplayReceiver",
    "ReplayRequest",
    "ReplayResult",
    "SIGNATURE_HEADER",
    "SignatureError",
    "make_server",
    "parse_replay_request",
    "serve",
    "sign_replay_body",
    "verify_replay_signature",
]
