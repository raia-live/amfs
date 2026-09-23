"""AMFS HTTP/REST API server — universal access to Agent Memory over HTTP.

Provides a FastAPI application exposing the full AMFS API as REST endpoints
with SSE streaming for real-time watch notifications and optional API key
authentication.

Run directly::

    amfs-http                          # default 0.0.0.0:8741
    amfs-http --port 9000 --host 127.0.0.1
    amfs-http --reload                 # development mode
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import contextvars
import functools
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta, timezone
from collections.abc import Mapping
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from amfs import AgentMemory, MemoryType, OutcomeType
from amfs.config import load_config_or_default
from amfs.memory import validate_session_attributes
from pydantic import BaseModel, Field, ValidationError
from amfs_core.aggregates import (
    REUSE_CREDIT_K,
    entry_content_chars,
    recall_tokens_for_chars,
)
from amfs_core.ranking import composite_score
from amfs_core.ranking import entry_text as _entry_text
from amfs_core.scope import SqlScope
from amfs_core.ranking import keyword_coverage as _keyword_coverage
from amfs_core.reuse_value import REUSE_VALUE_HEADER, reuse_value_block
from amfs_core.capture import scan_captured_arguments, scan_captured_text
from amfs_core.actions import CONTRAST_MIN_W as _CONTRAST_MIN_W
from amfs_core.actions import actions_taken as derive_actions_taken
from amfs_core.engine import read_tracker_scope
from amfs_core.evidence import evidence_signal as _evidence_signal
from amfs_core.evidence import is_success as _evidence_is_success
from amfs_core.evidence import regime_shifted as _regime_shifted
from amfs_core.models import (
    AgentGroup,
    AMFSConfig,
    AttemptRecord,
    DecisionTrace,
    Event,
    EventType,
    GraphEdge,
    LayerConfig,
    MemoryEntry,
    SearchQuery,
    SemanticQuery,
    SessionMetadata,
)
from amfs_core.pagination import (
    InvalidCursorError,
    Page,
    clamp_limit,
    decode_cursor,
    encode_cursor,
    entry_tiebreak,
    max_scan_rows,
    page_from_overfetch,
)
from amfs_core.quality import HeuristicQualityEvaluator

from amfs_http.auth import verify_api_key
from amfs_http.models import (
    AddTeamMemberRequest,
    AggregateRequest,
    ContextRequest,
    CreateAPIKeyRequest,
    CreateSnapshotRequest,
    CreateTeamRequest,
    EventRequest,
    OutcomeRequest,
    RetrieveRequest,
    RunPatternDetectionRequest,
    SearchRequest,
    UpdateTeamMemberRequest,
    UpdateTeamRequest,
    WriteRequest,
)
from amfs_http.sse import SSEManager

logger = logging.getLogger(__name__)


def _server_version() -> str:
    """Return the deployed amfs-http-server package version.

    Sourced from installed package metadata so it changes on every release —
    unlike the per-entry ``amfs_version`` schema tag. This is the value to use
    when telling deploys/revisions apart.
    """
    try:
        from importlib.metadata import version

        return version("amfs-http-server")
    except Exception:
        return "unknown"


# Set by the deploy pipeline (e.g. the git SHA) so a running revision is
# identifiable even between version bumps. Empty when not provided.
_BUILD_SHA = os.environ.get("AMFS_BUILD_SHA", "")
_SCHEMA_VERSION = MemoryEntry.model_fields["amfs_version"].default

# ── Async adapter (hot-path, non-blocking) ──────────────────────────
_async_adapter = None  # AsyncPostgresAdapter | None, set in lifespan


@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ARG001
    """Open/close the async connection pool for hot-path DB access, and run
    the embedded Cortex worker for the life of this process.

    The Cortex worker starts here rather than in ``main()`` because this is the
    hook that runs once in every process that serves requests. With
    ``--workers`` above one, uvicorn imports ``amfs_http.server:app`` afresh
    in each child; ``main()`` runs only in the supervisor, so a worker started
    there is invisible to the ``/api/v1/cortex/*`` routes in the children —
    they read the module global and saw ``None``. Starting it per process
    keeps those routes truthful and costs one LISTEN connection and a
    two-connection pool per worker instead of per instance.
    """
    global _async_adapter, _cortex_worker
    dsn = os.environ.get("AMFS_POSTGRES_DSN")
    if dsn and not os.environ.get("AMFS_HTTP_URL"):
        try:
            from amfs_postgres.async_adapter import AsyncPostgresAdapter
            ns = os.environ.get("AMFS_NAMESPACE", "default")
            _async_adapter = AsyncPostgresAdapter(dsn=dsn, namespace=ns)
            await _async_adapter.open()
            logger.info("Async Postgres adapter started (namespace=%s)", ns)
        except Exception:
            logger.warning("Failed to start async adapter — falling back to sync", exc_info=True)
            _async_adapter = None
    if _env_flag("AMFS_WITH_CORTEX"):
        _start_embedded_cortex()
    yield
    if _cortex_worker is not None:
        try:
            _cortex_worker.stop()
        except Exception:  # noqa: BLE001 - shutting down; nothing to do about it
            logger.debug("Cortex worker stop raised", exc_info=True)
    if _async_adapter is not None:
        await _async_adapter.close()
        logger.info("Async Postgres adapter closed")


def _env_flag(name: str) -> bool:
    """Whether environment variable *name* is set to a truthy value."""
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


app = FastAPI(
    title="AMFS HTTP API",
    description="Agent Memory File System — REST API with SSE support",
    version=_server_version(),
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _dashboard_tenant_middleware(request: Request, call_next):
    """Optional X-AMFS-Dashboard-Account-Id + secret → amfs.current_account_id on DB connections."""
    from amfs_http.tenant_middleware import apply_tenant_headers_from_request, clear_tenant_headers

    did_set = apply_tenant_headers_from_request(request)
    try:
        return await call_next(request)
    finally:
        if did_set:
            clear_tenant_headers()


@app.middleware("http")
async def _read_tracker_scope_middleware(request: Request, call_next):
    """Give every request its own session on the shared memory handle's tracker.

    ``_get_memory`` returns one ``AgentMemory`` for the whole process, so a single
    ``ReadTracker`` sits behind every request. It accumulates what the reads
    returned, along with the contexts, queries and writes of the session, and it is
    emptied only inside ``commit_outcome`` — so without a scope what one request
    left there was still in place for the next, and anything reading it back
    described a session belonging to no one caller.

    A request is the right unit: within one, the reads and any commit belong to the
    same caller, and across requests they do not, because a remote caller's own
    session lives in its own process and names its causal entries when it commits.

    Registered last so it runs outermost — Starlette builds the stack in reverse
    registration order — because it has to still be in force while the middlewares
    inside it unwind, or something tearing down could write into the next request's
    session.
    """
    with read_tracker_scope():
        return await call_next(request)


_memory: AgentMemory | None = None
_sse_manager = SSEManager()

# Routers mounted onto this app from outside the package cannot reach a
# module-level private, so the manager is published on app.state, which a
# route reaches through its own Request. Without this the room event stream
# has no manager to find and answers 503 on every connection, and the
# broadcasts aimed at it are dropped in silence — which is how it behaved.
app.state.sse_manager = _sse_manager

_bg_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="amfs-bg")

# ── Keeping the event loop free ────────────────────────────────────────
# This server is one uvicorn process per instance, so everything that runs
# synchronously inside an ``async def`` route stalls every other request on
# the instance for as long as it takes. Three things did, and together they
# were the production bottleneck measured on 2026-09-18 (a bare ``/stats``
# at 27 s while the database sat at 3% CPU): the ONNX embedder on every write
# and retrieve, the Pro cross-encoder rerank over up to thirty documents on
# every retrieve, and ``commit_outcome`` calling the sync adapter end to end.
#
# Model work goes to ``_model_executor``, sized to the CPUs because ONNX
# releases the GIL and more threads than cores only thrash. Sync adapter
# work goes to ``_db_executor``; its size is a ceiling on concurrent
# checkouts from the sync pool, not a throughput target. ``_offload`` copies
# the calling context into the thread so the tenant ContextVars the RLS pool
# reads are the request's, not the thread's leftovers.
#
# Routes whose whole body is synchronous — the admin, team, API-key and
# pattern routes that open ``pool.connection()`` and return — are plain
# ``def``, on purpose. FastAPI runs a ``def`` endpoint on the anyio worker
# pool, with the caller's contextvars copied in, so the body needs no
# ``_offload`` wrapping; declared ``async`` the same body holds the loop for
# the length of the query, and for however long the pool checkout has to wait
# when the pool is busy. Do not "fix" one back to ``async def`` unless it
# gains an ``await``.


def _effective_cpu_count() -> int:
    """The CPUs this process may actually use, not the ones the host has.

    ``os.cpu_count()`` reports the machine. In a container it is the node's
    core count, while the scheduler holds the process to whatever the cgroup
    quota or affinity mask says — on Cloud Run a 2-vCPU service can read 8 or
    more here. Sizing a thread pool for the host's cores then puts several
    ONNX threads on every core the container is allowed, which is the thrash
    the pool was sized to avoid.

    Takes the smallest of: the affinity mask (Linux), the cgroup v2 quota
    (``cpu.max``, ``quota/period`` rounded up), the cgroup v1 quota, and
    ``os.cpu_count()``. ``AMFS_MODEL_THREADS`` overrides all of it for the
    deployment that knows better. Never below one.
    """
    override = os.environ.get("AMFS_MODEL_THREADS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning("AMFS_MODEL_THREADS=%r is not an integer; ignoring", override)

    candidates: list[int] = []
    host = os.cpu_count()
    if host:
        candidates.append(host)
    try:
        candidates.append(len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            with open(quota_path) as fh:
                first = fh.read().split()
            if period_path is None:
                quota_s, period_s = first[0], first[1]
            else:
                quota_s = first[0]
                with open(period_path) as fh:
                    period_s = fh.read().split()[0]
            if quota_s in ("max", "-1"):
                continue
            quota, period = int(quota_s), int(period_s)
            if quota > 0 and period > 0:
                candidates.append(max(1, -(-quota // period)))
        except (OSError, ValueError, IndexError):
            continue
    return max(1, min(candidates) if candidates else 2)


_model_executor = ThreadPoolExecutor(
    max_workers=_effective_cpu_count(), thread_name_prefix="amfs-model"
)
_db_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="amfs-db")


async def _offload(executor: ThreadPoolExecutor, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn(*args, **kwargs)`` on ``executor`` with the current context."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(
        executor, functools.partial(ctx.run, functools.partial(fn, *args, **kwargs))
    )


def _sync_search(adapter: Any, sq: Any, branch: Any) -> list[Any]:
    """``adapter.search`` for adapters with and without a ``branch`` parameter.

    Meant to run on ``_db_executor`` via ``_offload``; the sync search is the
    fallback the async path takes when it returns nothing, and it must not
    hold the loop any more than the primary path does.
    """
    try:
        return adapter.search(sq, branch=branch)
    except TypeError:
        return adapter.search(sq)


_known_agents: set[str] = set()
# Tracks (agent, namespace, user) triples whose owner linkage was already
# upserted, so the hot write path doesn't repeat the DB call. Kept separate
# from _known_agents: an agent may be registered before its owner is known
# (e.g. first write arrived without user attribution).
_owner_linked_agents: set[str] = set()

# ── Semantic embedder (shared by write-time embedding + /retrieve) ──────
# One instance for the whole process so write and query vectors come from the
# SAME model (otherwise cosine similarity is meaningless). Env-gated and fully
# crash-safe: if the embedder can't be built, we return None and every caller
# falls back to lexical behaviour — a bad rollout degrades to the status quo,
# it never breaks writes or reads.
_UNSET_EMBEDDER: Any = object()
_server_embedder: Any = _UNSET_EMBEDDER


def _embeddings_enabled() -> bool:
    return os.environ.get("AMFS_ENABLE_EMBEDDINGS", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _get_server_embedder():
    """Return the process-wide embedder, or None if disabled/unavailable."""
    global _server_embedder
    if _server_embedder is _UNSET_EMBEDDER:
        _server_embedder = None
        if _embeddings_enabled():
            try:
                from amfs_core.default_embedder import create_default_embedder

                _server_embedder = create_default_embedder()
                logger.info(
                    "Semantic embedder ready: %s",
                    type(_server_embedder).__name__,
                )
            except Exception:  # noqa: BLE001 - never block startup on the embedder
                logger.warning(
                    "Embedder init failed — semantic retrieval disabled, "
                    "falling back to lexical search",
                    exc_info=True,
                )
                _server_embedder = None
        else:
            logger.info("AMFS_ENABLE_EMBEDDINGS is off — semantic retrieval disabled")
    return _server_embedder


# Routers mounted from outside this package need the same model that produced
# the vectors they search, or cosine similarity between them is meaningless.
# Published as the accessor rather than the instance: resolving it loads an
# ONNX model, and doing that at import time would move a multi-second cost into
# startup for every deployment, including the ones that never embed anything.
app.state.get_embedder = _get_server_embedder


# ── Injectable retrieval enhancers (Pro layer) ─────────────────────────
# The Pro layer registers a cross-encoder reranker and/or an LLM query
# rewriter into the SINGLE /api/v1/retrieve path (see set_retrieval_enhancers,
# called from mount_pro_api). Keeping ranking here — rather than in a separate
# Pro endpoint — means account (RLS) + per-user/room (UserVisibilityFilter)
# isolation is enforced in exactly one place and can never drift. Both are
# no-ops when unset, so a pure-OSS deployment is unchanged.
_retrieval_reranker: Any = None
_retrieval_query_rewriter: Any = None


def set_retrieval_enhancers(reranker: Any = None, query_rewriter: Any = None) -> None:
    """Register optional retrieval enhancers used by /api/v1/retrieve.

    - ``reranker``: object with ``available: bool`` and
      ``rerank(query, docs) -> list[float]`` (e.g. amfs_retrieval.CrossEncoderReranker).
    - ``query_rewriter``: object with ``expand(query) -> list[str]``
      (e.g. amfs_retrieval.LLMQueryRewriter). Should return ``[query]`` when
      disabled.
    """
    global _retrieval_reranker, _retrieval_query_rewriter
    if reranker is not None:
        _retrieval_reranker = reranker
    if query_rewriter is not None:
        _retrieval_query_rewriter = query_rewriter


# Benchmark/system scratch namespaces must never outrank a user's real memory
# in recall (the incident that motivated this had bench rows crowding out real
# task summaries). Matched against the leading segment of entity_path.
#
# Defined in amfs_core.exclusions, not here. The pattern used to be written out
# in four places — this one and three in the hosted packages, each carrying a
# comment asking whoever edited it to remember the others — and the aggregates
# need it as SQL as well as Python, which is two more chances for the same
# rule to say two things.
from amfs_core.exclusions import (  # noqa: E402
    EXCLUDED_ENTITY_RE as _EXCLUDED_ENTITY_RE,
)
from amfs_core.exclusions import (  # noqa: E402
    is_excluded_entity as _is_excluded_entity,
)
from amfs_core.evidence import (  # noqa: E402
    DISCREDIT_THRESHOLD,
    blend_local_evidence as _blend_local_evidence,
    is_synthetic_key as _is_synthetic_key,
    locally_discredited as _locally_discredited,
    locally_valid as _locally_valid,
    replacements_from_lessons as _replacements_from_lessons,
)


def _local_evidence_enabled() -> bool:
    """Whether retrieve conditions evidence on the query (default) or uses the
    entry's pooled record alone. ``AMFS_LOCAL_EVIDENCE=0`` switches it off."""
    return os.environ.get("AMFS_LOCAL_EVIDENCE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


#: Entries whose local record is read per retrieve: the most relevant ones.
#: Thirty matches the rerank window; the record of the long tail does not
#: change what the agent is shown.
LOCAL_EVIDENCE_HEAD = 30
#: Most below-gate (discredited) entries read near the query per retrieve.
BELOW_GATE_LIMIT = 50
#: Most recent task texts read as the lexical term's background corpus, and
#: how long one read serves an entity. Tasks arrive far slower than retrieves,
#: and a corpus a minute stale weights a query's words the same.
TASK_CORPUS_LIMIT = 200
TASK_CORPUS_TTL_S = 60.0
_task_corpus_cache: dict[tuple[str, str, str], tuple[float, list[str]]] = {}
_TASK_CORPUS_CACHE_MAX = 512


async def _task_corpus(entity_path: str | None) -> list[str]:
    """The entity's recent task texts (see ``keyword_coverage``), read off the
    event loop and held for ``TASK_CORPUS_TTL_S``. Empty for an adapter without
    the read, a store without the column, or an entity with no task history —
    the lexical term then weights by the candidate pool alone."""
    if not entity_path:
        return []
    mem = _get_memory()
    fn = getattr(mem._adapter, "recent_task_texts", None)
    if not callable(fn):
        return []
    # Keyed by tenant as well as namespace: the adapter scopes the read to the
    # request's account through the tenant context, and two accounts naming the
    # same entity path must not read each other's tasks from this cache.
    account = getattr(mem._adapter, "_get_current_account_id", lambda: None)()
    cache_key = (str(account or ""), str(mem.namespace), entity_path)
    now = time.monotonic()
    hit = _task_corpus_cache.get(cache_key)
    if hit is not None and now - hit[0] < TASK_CORPUS_TTL_S:
        return hit[1]
    try:
        texts = await _offload(_db_executor, fn, entity_path, limit=TASK_CORPUS_LIMIT)
    except Exception:  # noqa: BLE001
        # A failed read is not "this entity has no tasks": serve the stale
        # corpus if there is one and leave the cache alone, so the next
        # retrieve tries again instead of weighting by the candidate pool for
        # a minute because of one transient error.
        logger.debug("recent_task_texts failed", exc_info=True)
        return list(hit[1]) if hit is not None else []
    if len(_task_corpus_cache) >= _TASK_CORPUS_CACHE_MAX:
        oldest = min(_task_corpus_cache, key=lambda k: _task_corpus_cache[k][0])
        _task_corpus_cache.pop(oldest, None)
    _task_corpus_cache[cache_key] = (now, list(texts))
    return list(texts)
#: An entry is "about this query" — for the query-scoped regime shift — when
#: its similarity is within this of the best hit's, or it matched on keywords.
LOCAL_SIM_GAP = 0.1


def _rank_anchored() -> bool:
    """Whether trust modulates relevance (default) or is added to it.

    ``AMFS_RANK_ADDITIVE_TRUST=1`` restores the pre-2026-09-18 weighted sum
    without a code deploy. See ``amfs_core.ranking`` for why the default
    changed.
    """
    return os.environ.get("AMFS_RANK_ADDITIVE_TRUST", "").strip().lower() not in (
        "1", "true", "yes", "on",
    )


def _retrieve_min_semantic() -> float:
    """Abstain floor: results whose semantic similarity is below this AND that
    have no keyword match are trimmed as clearly-irrelevant tail (the top
    result is always kept). Env-tunable; conservative default."""
    try:
        return float(os.environ.get("AMFS_RETRIEVE_MIN_SEMANTIC", "0.15"))
    except ValueError:
        return 0.15


def _doc_text_for_rerank(entry: MemoryEntry) -> str:
    """Compact document text for the cross-encoder: key + stringified value,
    bounded so a huge value can't blow up the reranker input."""
    val = entry.value
    if not isinstance(val, str):
        try:
            val = json.dumps(val, default=str)
        except Exception:  # noqa: BLE001
            val = str(val)
    text = f"{entry.key}: {val}"
    return text[:2000]


def _logistic(x: float) -> float:
    """Guarded: math.exp overflows around -745."""
    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, x))))


def _rerank_is_calibrated(scores: list[float]) -> bool:
    """Whether this reranker returns probabilities rather than logits.

    The reranker is injected, so its output range is observed rather than
    contracted. Asked in one place because two callers now depend on the answer
    and they must not disagree: a probability read as a logit, or the reverse,
    inverts both of them.
    """
    return all(0.0 <= s <= 1.0 for s in scores)


#: The cross-encoder's own decision boundary. A candidate it scores above this is
#: one the model puts at better-than-even odds of being relevant, which is the
#: only claim strong enough to overrule the bi-encoder's abstain floor.
RERANK_ENDORSED = 0.5


def _rerank_absolute(scores: list[float]) -> list[float]:
    """The cross-encoder's opinion of each candidate on its own, 0..1.

    The complement of :func:`_normalise_rerank`, and needed because that function
    deliberately answers a different question. Normalisation reports standing
    *within the batch*, so its top member scores high however poor the batch is:
    measured, a candidate at logit -3.0 — the model giving it a 4.7% chance of
    being relevant — normalises to 0.978 when its peers sit at -9, which is the
    highest value in that set. Correct for ranking, where only order matters, and
    useless for "is this relevant at all", where it is off by everything.

    This is the plain logistic, which is the function the model's training
    objective implies, so the result is the probability the model is asserting.
    Batch-independent by construction: no median, no peers.
    """
    if not scores:
        return []
    if _rerank_is_calibrated(scores):
        return list(scores)
    return [_logistic(s) for s in scores]


def _normalise_rerank(scores: list[float]) -> list[float]:
    """Map cross-encoder output onto the 0..1 range the relevance term expects.

    The reranker is injected, so its output range is a matter of observation
    rather than contract. The serving model emits logits — dev returns values
    from about -11 to +7.2 — while some implementations return a probability
    already. Both are accepted: if every score in the batch is inside 0..1 it
    is taken as calibrated, otherwise the batch is squashed with a logistic,
    which is the function the model's training objective implies.

    Not min-max over the batch. Min-max stretches whatever spread happens to be
    present to fill 0..1, so the top result scores 1.0 and the bottom 0.0 no
    matter how close together they really are — amplifying cross-encoder noise
    into a confident-looking ordering. On dev the top two scored 7.2307 and
    7.2206, a distinction the model plainly does not intend to draw, and min-max
    would have turned it into the largest gap in the set.

    Centred on the batch's median, then squashed. The centring is the part that
    matters, and it is here because the plain logistic was measured getting this
    wrong. A logistic is steep only near zero; away from zero it flattens. So
    where the whole batch sat far out on one tail — the cross-encoder confident
    about *every* candidate — a large genuine difference arrived as almost
    nothing: on dev, raw spreads of 3.04 and 6.04 survived as 0.0002 and 0.0047,
    under 0.3% of themselves, and the confidence term then decided a comparison
    relevance had already settled. In one measured case that put a tangential
    entry above one the reranker preferred by 2.73 logits which also carried 12
    validated outcomes, which is the opposite of the intent.

    So the result carries two things, because it has to answer two questions at
    once. ``anchor`` is the plain logistic of the batch's median — *how good is
    this batch at all* — and the centred term is each member's standing among its
    peers, measured where the logistic can still discriminate. Added together and
    clamped, they give a relevance term that keeps an absolutely strong batch high
    while still separating its members.

    Both halves are load-bearing, and dropping either has been tried. Without the
    centred term, differences vanish in the tails, as above. Without the anchor,
    the median maps to exactly 0.5 whatever the batch is worth — and the reranker
    only scores the top ``rerank_top_n``, after which step 8 re-sorts the head
    together with a tail still carrying raw bi-encoder similarity. A uniformly
    strong head would then sit around 0.5 while an unjudged tail entry kept 0.9,
    so the reranker's own favourites would lose to candidates it never saw. The
    anchor is what keeps the two groups on one scale.

    Peer standing is scaled into the room the anchor leaves rather than added and
    clamped, so nothing is thrown away at the edges: a candidate above its median
    moves into the space between the anchor and 1, one below it into the space
    between the anchor and 0. Clamping instead would have cost half of a measured
    2.7-logit gap, and worse, would have flattened the best few of a strong batch
    into an exact tie — the one place the reranker's judgement matters most.

    The median, not the mean, so one far-outlying candidate cannot drag the
    centre off the cluster being compared. Every step is monotone in the score,
    so this can compress differences but never reorder them.

    Worked through: 7.2307 against 7.2206 is worth 0.002, a tie confidence
    settles; 9.5 against 6.8 — measured live, where the flat logistic gave
    0.0015 — is worth 0.59, which no confidence gap overturns; a batch at 9.0
    stays above 0.94 so an unjudged tail at 0.9 does not displace it; and a batch
    at -9 stays below 0.06, correctly losing to a tail the reranker never rejected.
    """
    if not scores:
        return []
    if _rerank_is_calibrated(scores):
        return list(scores)
    ordered = sorted(scores)
    mid, odd = divmod(len(ordered), 2)
    centre = ordered[mid] if odd else (ordered[mid - 1] + ordered[mid]) / 2.0
    anchor = _logistic(centre)
    out: list[float] = []
    for s in scores:
        # Signed standing among peers, in (-0.5, 0.5) and undistorted by where
        # the batch sits, because the logistic sees only the deviation.
        deviation = _logistic(s - centre) - 0.5
        headroom = (1.0 - anchor) if deviation > 0 else anchor
        out.append(anchor + 2.0 * deviation * headroom)
    return out


_immutable_trace_store = None
try:
    from amfs_traces.api import mount_pro_routes
    mount_pro_routes(app)
    logger.info("Pro trace endpoints mounted at /api/v1/pro/traces")

    from amfs_traces.store import PostgresImmutableTraceStore
    from amfs_traces.crypto import seal, get_signing_key, get_signing_key_id
    # The OSS -> immutable mapping lives with the Pro package, shared with its
    # other two seal paths, so what this server seals is what they seal. It used
    # to be inlined here and silently dropped the fields it did not know about.
    from amfs_traces.seal_on_demand import (
        finalize_spans as _pro_finalize_spans,
        immutable_from_oss_trace as _pro_immutable_from_oss_trace,
    )
    _HAS_PRO_TRACES = True
except ImportError:
    _HAS_PRO_TRACES = False

try:
    from amfs_cortex_pro import mount_cortex_pro
    mount_cortex_pro(app)
    logger.info("Pro Cortex endpoints mounted (local)")
except ImportError:
    from amfs_http.pro_proxy import mount_pro_proxy
    mount_pro_proxy(app)

_memory_lock = threading.Lock()


def _get_memory() -> AgentMemory:
    """Return the shared AgentMemory singleton, building it on first call.

    Double-checked under a lock: the sync-bodied routes run on the threadpool,
    so the first requests after a worker starts — a dashboard load fires
    several at once — can arrive here together, and without the lock each
    would build its own ``AgentMemory`` and adapter pool, with every loser
    dropped unclosed. The fast path stays lock-free.
    """
    if _memory is not None:
        return _memory
    with _memory_lock:
        if _memory is not None:
            return _memory
        return _build_memory()


def _build_memory() -> AgentMemory:
    """Construct the shared AgentMemory. Call through ``_get_memory``."""
    global _memory
    if _memory is not None:
        return _memory

    agent_id = os.environ.get("AMFS_AGENT_ID", "http-server")

    http_url = os.environ.get("AMFS_HTTP_URL")
    if http_url:
        if os.environ.get("AMFS_POSTGRES_DSN"):
            logger.warning(
                "Both AMFS_HTTP_URL and AMFS_POSTGRES_DSN are set. "
                "The HTTP adapter takes precedence — direct DB access "
                "is bypassed in favour of the authenticated HTTP API."
            )
        try:
            from amfs_adapter_http import HttpAdapter

            api_key = os.environ.get("AMFS_API_KEY", "")
            logger.info(
                "AMFS HTTP adapter mode — routing through %s", http_url
            )
            adapter = HttpAdapter(base_url=http_url, api_key=api_key)
            _memory = AgentMemory(agent_id=agent_id, adapter=adapter)
            return _memory
        except ImportError:
            logger.warning(
                "AMFS_HTTP_URL is set but amfs-adapter-http is not installed. "
                "Falling back to local adapter. "
                "Install with: pip install amfs-adapter-http"
            )

    config = _resolve_config()

    ttl_interval_str = os.environ.get("AMFS_TTL_SWEEP_INTERVAL")
    ttl_sweep_interval = float(ttl_interval_str) if ttl_interval_str else 300.0

    logger.info("AMFS HTTP server starting — agent_id=%s", agent_id)
    mem = AgentMemory(
        agent_id=agent_id,
        config_path=None,
        adapter=None,
        ttl_sweep_interval=ttl_sweep_interval,
    )

    mem._config = config
    from amfs.factory import create_adapter_from_config

    adapter = create_adapter_from_config(config)
    mem._adapter = adapter
    mem._engine._adapter = adapter
    mem._propagator._adapter = adapter

    # Share the process-wide embedder so mem.write() embeds on the sync
    # fallback path and mem.semantic_search() works. The hot async path embeds
    # explicitly in the write endpoint. Safe no-op when embeddings are disabled.
    embedder = _get_server_embedder()
    if embedder is not None:
        mem._embedder = embedder
        try:
            if hasattr(adapter, "_embedder") and getattr(adapter, "_embedder", None) is None:
                adapter._embedder = embedder
        except Exception:  # noqa: BLE001 - adapter embedding is best-effort
            logger.debug("Could not attach embedder to adapter", exc_info=True)

    _memory = mem
    return _memory


try:
    from amfs_pro_api import mount_pro_api
    mount_pro_api(app, get_memory=_get_memory)
    logger.info("Pro API endpoints mounted (intelligence, extraction)")
except ImportError:
    pass

# Optional retrieval enhancers (proprietary amfs_retrieval, same optional-import
# pattern as the pro blocks above). When the package is present — as in the
# hosted amfs-pro image that serves /api/v1/retrieve — inject a cross-encoder
# reranker + query rewriter into the SINGLE retrieve path so ranking improves
# without a second endpoint (isolation stays enforced in one place). Absent in
# pure-OSS installs, where retrieve stays hybrid-but-unreranked.
try:
    from amfs_retrieval import CrossEncoderReranker, LLMQueryRewriter

    _rerank_on = os.environ.get("AMFS_RERANK", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    # ms-marco-MiniLM-L-6-v2: 0.08GB, apache-2.0 (commercial-safe). The
    # fastembed default (jina v2) is cc-by-nc-4.0 and 1.1GB — unsuitable for a
    # commercial SaaS image. Overridable via AMFS_RERANK_MODEL.
    _rerank_model = os.environ.get("AMFS_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    _reranker = CrossEncoderReranker(model_name=_rerank_model) if _rerank_on else None
    set_retrieval_enhancers(reranker=_reranker, query_rewriter=LLMQueryRewriter())
    logger.info(
        "Retrieval enhancers registered (rerank=%s model=%s, query_rewrite=%s)",
        _rerank_on, _rerank_model, os.environ.get("AMFS_QUERY_REWRITE", "false"),
    )
except ImportError:
    logger.debug("amfs_retrieval not installed — /retrieve stays hybrid without rerank")
except Exception:  # noqa: BLE001 - never block startup on enhancers
    logger.warning("Retrieval enhancer registration failed", exc_info=True)

try:
    from amfs_rooms import mount_rooms
    mount_rooms(app, get_memory=_get_memory)
    logger.info("Room endpoints mounted")
except ImportError:
    pass

try:
    from amfs_managed_models import mount_managed_models
    mount_managed_models(app, get_memory=_get_memory)
    logger.info("Managed Models endpoints mounted")
except ImportError:
    # Said at info, not debug, because the two outcomes are indistinguishable
    # from the outside: every route under /api/v1/models answers 404 either way.
    # An image built without the package is the failure this feature actually
    # shipped with, and the log is where an operator would look for it.
    logger.info(
        "amfs_managed_models not installed — /api/v1/models stays unmounted"
    )

try:
    from amfs_http.openrouter_proxy import mount_openrouter_proxy
    mount_openrouter_proxy(app, get_memory=_get_memory)
except Exception:  # noqa: BLE001 - proxy is optional; never block startup
    logger.debug("OpenRouter proxy not mounted", exc_info=True)

try:
    from amfs_tenant.http_deps import mount_scope_enforcement
    mount_scope_enforcement(app)
except ImportError:
    pass


def _resolve_config() -> AMFSConfig:
    """Resolve AMFS configuration from environment or config files.

    Priority:
    1. AMFS_POSTGRES_DSN env var -> Postgres adapter
    2. AMFS_DATA_DIR env var -> filesystem adapter at that path
    3. amfs.yaml discovery -> load from file
    4. Default -> filesystem adapter at .amfs/
    """
    postgres_dsn = os.environ.get("AMFS_POSTGRES_DSN")
    if postgres_dsn:
        return AMFSConfig(
            namespace="default",
            layers={
                "primary": LayerConfig(
                    adapter="postgres",
                    options={"dsn": postgres_dsn},
                )
            },
        )

    data_dir = os.environ.get("AMFS_DATA_DIR")
    if data_dir:
        return AMFSConfig(
            namespace="default",
            layers={
                "primary": LayerConfig(
                    adapter="filesystem",
                    options={"root": data_dir},
                )
            },
        )

    return load_config_or_default()


def _entry_to_response(entry: MemoryEntry) -> dict[str, Any]:
    """Convert a MemoryEntry to a JSON-safe dict, stripping embeddings."""
    data = entry.model_dump(mode="json")
    data.pop("embedding", None)
    # Properties, so not in the dump; the one word an agent needs next to the
    # confidence number (untested / validated / contested / discredited) and
    # the record behind it: ``{"p": posterior success, "n": outcomes}``. A 0.9
    # over twelve outcomes and a 0.9 over one are different things to act on.
    data["evidence_status"] = entry.evidence_status
    p, n = entry.posterior
    data["posterior"] = {"p": p, "n": n}
    return data


#: Fields a compact payload keeps. Enough to rebuild a MemoryEntry client-side
#: (entity_path, key, value, provenance, confidence, version) plus the evidence
#: an agent acts on; none of the bookkeeping (importance dimensions, integrity
#: fields, tiering, TTL) that made a ten-hit retrieve cost 3-4k tokens.
_COMPACT_ENTRY_FIELDS = (
    "entity_path", "key", "version", "value", "provenance", "confidence",
    "memory_type", "evidence_status", "posterior", "success_count",
    "failure_count", "last_outcome", "discredited_at", "validators",
)
_COMPACT_VALUE_MAX_CHARS = 1200
#: In compact mode the first hits are carried whole (up to the cap above) and
#: the rest as one-liners: an agent acts on the top one or two and skims the
#: others for a contradiction, and a 120-character preview is enough to spot
#: one. Together with the field trim this is what keeps a seven-hit retrieve
#: near what a plain vector store costs in prompt tokens.
_COMPACT_FULL_HITS = 2
_COMPACT_TAIL_CHARS = 120


def _compact_entry_response(entry: MemoryEntry, *, rank: int = 0) -> dict[str, Any]:
    """The fields an agent acts on, and nothing else (``compact=True``).

    ``rank`` is the hit's position; from :data:`_COMPACT_FULL_HITS` on, the
    value is a preview.
    """
    full = _entry_to_response(entry)
    data = {k: full[k] for k in _COMPACT_ENTRY_FIELDS if k in full}
    prov = data.get("provenance") or {}
    data["provenance"] = {
        k: prov.get(k) for k in ("agent_id", "session_id", "written_at") if k in prov
    }
    value = data.get("value")
    cap = _COMPACT_VALUE_MAX_CHARS if rank < _COMPACT_FULL_HITS else _COMPACT_TAIL_CHARS
    if not isinstance(value, str) and rank >= _COMPACT_FULL_HITS:
        value = json.dumps(value, default=str)
    if isinstance(value, str) and len(value) > cap:
        data["value"] = value[:cap] + " …[truncated]"
        data["value_truncated"] = True
    elif value is not data.get("value"):
        data["value"] = value
    return data


def _search_sync(adapter: Any, query: SearchQuery, branch: Any) -> list[MemoryEntry]:
    """``adapter.search`` with or without the branch keyword, whichever it takes."""
    try:
        return adapter.search(query, branch=branch)
    except TypeError:
        return adapter.search(query)


async def _discredited_below_gate(
    entity_path: str,
    *,
    branch: Any,
    seen: set[str],
    vis: Any,
    include_artifacts: bool,
    limit: int,
) -> list[MemoryEntry]:
    """The entity's discredited entries the ranked list did not hold.

    Read only for the regime-shift flag, on every priors call. Two ways the
    ranked list misses the signal: a rule that was validated many times and
    then failed twice has a confidence under the discredit threshold, so a
    retrieve gated at that threshold — the benchmark's setting, and a reasonable
    production one — never saw it; and with no gate at all, the rule that
    stopped working may simply not match this query.

    Not a re-run of the lexical query. The rule that stopped working need not
    share words with this query (it may have matched semantically, or not at
    all), and a rule validated over months is old by write time, so a text
    search sorted by recency and truncated at the pool could miss it. Priors
    are kept per entity and the briefing's section of the same name is read
    over the whole entity, so this is too: every entry on *entity_path* with
    confidence at or under the discredit threshold — a small set, since that is
    what discrediting means — then the discredited ones among them.

    Filtered by the same visibility rules as the ranked list. The rows are not
    returned, but the recommendation they steer is, and a policy decision
    driven by memory the caller cannot see is a leak by another name.
    Best-effort; a store that cannot answer contributes nothing rather than
    failing the retrieve.
    """
    below = SearchQuery(
        entity_path=entity_path,
        min_confidence=0.0,
        max_confidence=DISCREDIT_THRESHOLD,
        limit=limit,
        sort_by="recency",
        depth=3,
        include_artifacts=include_artifacts,
    )
    rows: list[MemoryEntry] = []
    try:
        if _async_adapter is not None:
            rows = await _async_adapter.search(below, branch=branch)
        else:
            adapter = _get_memory()._adapter
            try:
                rows = adapter.search(below, branch=branch)
            except TypeError:
                rows = adapter.search(below)
    except Exception:
        logger.debug("below-gate discredited fetch failed", exc_info=True)
        return []
    if vis is not None and vis.should_filter():
        rows = vis.filter_entries(rows)
    return [
        e for e in rows
        if getattr(e, "discredited_at", None) is not None
        and e.entry_key not in seen
        and not _is_excluded_entity(getattr(e, "entity_path", ""))
        and not _is_synthetic_key(getattr(e, "key", ""))
    ]


def _as_utc(value: Any) -> datetime | None:
    """A datetime or ISO string as an aware UTC datetime; ``None`` otherwise."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _resolved_actions_from_lessons(
    lessons: list[MemoryEntry], contrasts: list[dict[str, Any]]
) -> dict[str, str]:
    """``entry_key -> action_key``: for each entry a contrast lesson names
    under ``avoid``, the action that resolved that task instead.

    The lesson's own ``resolved_action`` where it carries one (written by
    ``amfs_core.evidence.contrast_lesson`` from the record's actions); else
    the priors contrast for the same ``outcome_ref`` — the lesson and the
    contrast are two readings of one outcome, joined on its ref. Neither the
    nearest contrast nor any other is applied to an entry no lesson ties it
    to: the entry keys are the lesson's, the action is the outcome's, and an
    avoided rule that no fail-then-succeed outcome named gets nothing.
    """
    by_ref = {
        str(c.get("outcome_ref")): str(c["resolved_with"])
        for c in contrasts
        if c.get("outcome_ref") and c.get("resolved_with")
    }
    out: dict[str, str] = {}
    for e in lessons:
        value = e.value if isinstance(e.value, dict) else None
        if not value or not isinstance(value.get("avoid"), list):
            continue
        action = value.get("resolved_action") or by_ref.get(str(value.get("outcome_ref")))
        if not action:
            continue
        for spec in value["avoid"]:
            out.setdefault(str(spec), str(action))
    return out


#: How many of the entity's newest outcomes are scanned for the declared
#: situation's own record when the embedding neighbourhood holds none of it.
SITUATION_RECORD_SCAN = 1000


def _priors_for_retrieve(
    *,
    entity_path: str,
    text: str,
    embedder: Any,
    candidate_actions: list[str] | None,
    environment: Mapping[str, Any] | None = None,
    query_vector: list[float] | None = None,
    situation: str | None = None,
) -> dict[str, Any] | None:
    """Action priors for ``entity_path`` on tasks like ``text``.

    Nearest committed outcomes by task embedding when the store can do that,
    re-weighted relative to the nearest one (``neighbourhood_weights``) so the
    priors are about this kind of task and not every task on the entity;
    otherwise the most recent outcomes about the entity. ``None`` when there
    is no outcome record to report — the adapter keeps none (filesystem, S3)
    or nothing has been committed on the entity yet — so the caller sends
    nothing rather than a block whose only content is the candidate list the
    agent itself supplied.

    With a declared *situation*, and outcomes in the neighbourhood that carry
    one, the block is the **situation's own record**: the priors are
    aggregated over the outcomes whose situation is this one
    (``situation_record``: how many), and the rest of the neighbourhood is
    reported separately under ``nearby`` — per situation, what was tried and
    how it went — for the agent to weigh, never to be planned from. Two
    situations that differ by one token embed within 0.02 of each other, so
    text alone pools them; the label is what tells them apart. Outcomes
    without a situation (older clients) leave the block pooled as before.

    Synchronous, and blocking on the sync adapter: callers on the event loop
    run it through ``_offload``. ``query_vector`` is the caller's embedding of
    ``text`` when it has one, so the model is not run a second time.
    *environment* down-weights outcomes recorded under another model /
    runtime / agent version (the rows carry ``session_metadata`` or an
    ``environment`` column when the adapter returns it).
    """
    from amfs_core.actions import (
        PRIORS_K,
        PRIORS_MIN_SIMILARITY,
        aggregate_priors,
        neighbourhood_weights,
    )

    adapter = _get_memory()._adapter
    similar = getattr(adapter, "similar_outcomes", None)
    stats = getattr(adapter, "action_stats", None)
    rows: list[dict[str, Any]] = []
    source = "none"
    if callable(similar) and text.strip() and (query_vector is not None or embedder is not None):
        try:
            vec = query_vector if query_vector is not None else embedder.embed(text[:2000])
            # Over-fetch, then keep the neighbourhood: the floor admits every
            # task on the entity under a retrieval embedder, and the cut that
            # matters is relative to the best match.
            rows = similar(
                entity_path,
                vec,
                k=PRIORS_K * 3,
                min_similarity=PRIORS_MIN_SIMILARITY,
            )
            rows = neighbourhood_weights(rows)[:PRIORS_K]
            source = "similar_outcomes"
        except Exception:  # noqa: BLE001 - priors are best-effort
            logger.debug("similar_outcomes failed", exc_info=True)
            rows = []
    if not rows and callable(stats):
        try:
            rows = stats(entity_path, limit=PRIORS_K * 5)
            source = "action_stats" if rows else source
        except Exception:  # noqa: BLE001
            logger.debug("action_stats failed", exc_info=True)
            rows = []
    mine: list[dict[str, Any]] | None = None
    others: list[dict[str, Any]] = []
    declared = _fold_situation(situation)
    if declared and callable(stats):
        # The situation's own record is an equality, not a neighbourhood.
        # The embedding neighbourhood may hold none of it — a queue whose
        # classes embed far apart returns no neighbour above the floor for a
        # class seen once — and the fallback to the entity's whole record
        # would then serve another class's winner as ``act``. Measured on
        # the ops-queue support demo (2026-09-22): the first Android ticket
        # was told ``act: explain_and_close`` from the webhook class's record
        # and spent its three attempts on the entity's winners. So the exact
        # rows come from the entity's record by situation, whatever the
        # neighbourhood found; the neighbourhood supplies ``nearby``.
        exact_rows = [r for r in rows if _fold_situation(r.get("situation")) == declared]
        if source != "similar_outcomes" or not exact_rows:
            try:
                seen = {r.get("outcome_ref") for r in exact_rows}
                for r in stats(entity_path, limit=SITUATION_RECORD_SCAN):
                    if _fold_situation(r.get("situation")) == declared and r.get("outcome_ref") not in seen:
                        exact_rows.append(r)
                        seen.add(r.get("outcome_ref"))
            except Exception:  # noqa: BLE001
                logger.debug("action_stats by situation failed", exc_info=True)
        labelled = any(r.get("situation") for r in rows) or bool(exact_rows)
        if labelled:
            mine = exact_rows
            others = (
                [r for r in rows if _fold_situation(r.get("situation")) != declared]
                if source == "similar_outcomes" else []
            )
    if not rows and mine is None:
        return None
    if mine is not None:
        block = aggregate_priors(
            mine, candidate_actions=candidate_actions, environment=environment or None,
            situation_exact=True,
        )
        block["situation_record"] = {"situation": str(situation)[:200], "outcomes": len(mine)}
        block["nearby"] = _nearby_record(others, environment=environment or None)
        # The exact record is about this kind of task by construction; the
        # label ``action_stats`` would have it read as the entity-wide pool.
        if source != "similar_outcomes":
            source = "situation_record"
    else:
        block = aggregate_priors(
            rows, candidate_actions=candidate_actions, environment=environment or None
        )
    block["source"] = source
    block["entity_path"] = entity_path
    if source == "similar_outcomes" and rows:
        block["neighbourhood"] = {
            "best_similarity": round(max(float(r.get("task_similarity") or 0.0) for r in rows), 3),
            "outcomes": len(rows),
        }
    return block


def _fold_situation(value: Any) -> str:
    """A situation label as compared: case- and whitespace-folded."""
    return " ".join(str(value or "").split()).casefold()


def _nearby_record(
    rows: list[dict[str, Any]], *, environment: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The neighbourhood outside the declared situation, per situation:
    ``{"outcomes": n, "by_situation": [{"situation", "outcomes", "tried":
    [...]}]}``, nearest situation first, three at most. What the agent sees
    as "on nearby kinds of task"; nothing here is planned from."""
    from amfs_core.actions import aggregate_priors

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(_fold_situation(r.get("situation")) or "(unlabelled)", []).append(r)

    def _nearness(items: list[dict[str, Any]]) -> float:
        return max(float(r.get("task_similarity") or r.get("similarity") or 0.0) for r in items)

    by_situation = []
    for key, items in sorted(groups.items(), key=lambda kv: -_nearness(kv[1]))[:3]:
        label = next((str(r.get("situation")) for r in items if r.get("situation")), key)
        agg = aggregate_priors(items, environment=environment)
        by_situation.append({
            "situation": label[:200],
            "outcomes": len(items),
            "tried": [
                {k: t[k] for k in ("action_key", "won", "lost", "n", "last_3") if k in t}
                for t in agg["tried"][:6]
            ],
        })
    return {"outcomes": len(rows), "by_situation": by_situation}


def _get_visibility_filter(request: Request):
    """Return the UserVisibilityFilter from request.state, or None."""
    return getattr(request.state, "visibility_filter", None)


#: ``request.state`` attribute a layer in front of these routes may set to move
#: a session's *reads* onto a memory branch it never named. The SaaS layer sets
#: it when the calling session is in the canary arm of a live repair canary.
MEMORY_BRANCH_STATE = "memory_branch"
#: ``request.state`` attribute holding attributes the same layer wants on the
#: sealed trace — which canary the session was in and which arm. Merged
#: server-side, after the caller's bag has been validated, so they are exempt
#: from the client cap and cannot be forged from the body.
TRACE_ATTRIBUTES_STATE = "trace_attributes"


def _effective_branch(request: Request | None, branch: str | None) -> str:
    """The branch a read should hit: the caller's when named, else routed, else main.

    A caller that names a branch always gets that branch — an explicit
    ``branch=main`` from a canary session still reads main, which is what a
    repair tool inspecting the baseline needs. A caller that names none reads
    whatever ``request.state.memory_branch`` says, which is how a session the
    SaaS layer put in a canary arm reads the proposal without knowing it, and
    ``main`` when nothing is set — the behaviour every route had before.

    Reads only. Writes and commits never consult this: a canary session reads
    its branch and writes main, because what the agent learns during the test
    is the user's, not the proposal's.
    """
    # ``isinstance`` rather than truthiness: a handler invoked in-process (Pro
    # composes several, and the tests do) receives the ``Query(...)`` sentinel
    # as its default, and that object is truthy without being a branch.
    if isinstance(branch, str) and branch.strip():
        return branch.strip()
    state = getattr(request, "state", None) if request is not None else None
    routed = getattr(state, MEMORY_BRANCH_STATE, None) if state is not None else None
    if isinstance(routed, str) and routed.strip():
        return routed.strip()
    return "main"


#: Attribute keys only the routing layer may set. When it has made a decision
#: for the request (``trace_attributes`` is present, even empty) a client's own
#: claims under these keys are dropped: a session nothing routed must not be
#: able to vote in a canary it was not in.
ROUTED_ONLY_ATTRIBUTES = frozenset({"canary_fix_id", "canary_arm"})


def _routed_trace_attributes(request: Request | None) -> dict[str, Any] | None:
    """Attributes the layer in front of this route wants on the sealed trace.

    ``None`` when no layer made a decision for this request; a dict — possibly
    empty — when one did. Values are coerced to scalars the trace store
    accepts; anything else is dropped rather than failing the commit.
    """
    state = getattr(request, "state", None) if request is not None else None
    raw = getattr(state, TRACE_ATTRIBUTES_STATE, None) if state is not None else None
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key.strip().lower()] = value
    return out


def _merge_routed_attributes(
    request: Request | None, attributes: dict[str, Any] | None
) -> dict[str, Any] | None:
    """*attributes* with the routed stamps on top; ``None`` stays ``None`` when nothing is added.

    The server's values win. A client must not be able to put itself in the
    canary arm by sending ``canary_arm`` in its own bag, and the tally reads
    only these keys to decide which arm a trace belongs to — so once the
    routing layer has spoken for a request, the client's claims under those
    keys are dropped even when the layer's answer was "not routed".
    """
    routed = _routed_trace_attributes(request)
    if routed is None:
        return attributes
    merged = {
        k: v for k, v in (attributes or {}).items()
        if str(k).strip().lower() not in ROUTED_ONLY_ATTRIBUTES
    }
    merged.update(routed)
    if not merged and attributes is None:
        return None
    return merged


def _active_visibility_filter(request: Request):
    """Return the visibility filter when per-user scoping applies, else None.

    None means the caller sees the whole account: either there is no user
    context (plain API key on a single-user / self-hosted install) or the
    user is an account admin.
    """
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        return vis
    return None


def _visibility_scope(request: Request) -> tuple[Any | None, Any | None]:
    """How to apply the caller's visibility to a read: ``(scope, py_filter)``.

    ``scope`` is the rule as a SQL predicate (``amfs_core.scope.SqlScope``)
    when the filter can express itself that way — the hosted
    ``UserVisibilityFilter`` does, via ``sql_predicate()`` — so the read is
    narrowed in the query and a page is a page of visible rows. ``py_filter``
    is the filter object when the rule has to run over loaded entries instead
    (a filter without the hook, or one that declined). Both ``None`` when the
    caller sees the whole account.

    Exactly one of the two is set when scoping applies; a handler that takes
    the SQL route must not also filter in Python, and one that gets
    ``py_filter`` must not page in SQL, or it would page past hidden rows.
    """
    vis = _active_visibility_filter(request)
    if vis is None:
        return None, None
    predicate = getattr(vis, "sql_predicate", None)
    if callable(predicate):
        try:
            scope = predicate()
        except Exception:  # noqa: BLE001 - the Python filter is always correct
            logger.debug("visibility sql_predicate failed; filtering in Python", exc_info=True)
            scope = None
        # A real predicate only: anything else (a filter that declined with
        # None, a stand-in object) means the rule runs over entries.
        if isinstance(scope, SqlScope):
            return scope, None
    return None, vis


def _entries_by_agent(mem: AgentMemory, agent_id: str) -> list[MemoryEntry]:
    """One agent's current entries, narrowed in the query where the adapter
    can (the Postgres adapters take ``agent_id``), over ``list()`` where it
    cannot. Synchronous: call it off the event loop."""
    try:
        return mem._adapter.list(agent_id=agent_id)
    except TypeError:
        return [e for e in mem.list() if e.provenance.agent_id == agent_id]


def _authors_of(mem: AgentMemory, refs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Who wrote each ``(entity_path, key)`` — looked up for just those refs
    where the adapter can (``entry_authors``), from ``list()`` where it
    cannot. The graph views attribute an agent's reads to the entries'
    authors and need the author of the entries it read, not of every entry
    in the namespace. Synchronous: call it off the event loop."""
    if not refs:
        return {}
    lookup = getattr(mem._adapter, "entry_authors", None)
    if callable(lookup):
        return lookup(refs)
    wanted = set(refs)
    return {
        (e.entity_path, e.key): e.provenance.agent_id
        for e in mem.list()
        if (e.entity_path, e.key) in wanted
    }


#: Default page size for ``GET /entries`` when the caller names none, or 0 to
#: keep returning the whole namespace as before (the default). A hosted
#: deployment sets this once its own whole-list consumers page; the response
#: always carries ``total``, so a capped client can tell and page on.
ENTRIES_DEFAULT_LIMIT = int(os.environ.get("AMFS_ENTRIES_DEFAULT_LIMIT", "0") or 0)


def _visible_agent_ids(request: Request) -> set[str] | None:
    """Set of agent_ids the caller may see, or None for unrestricted."""
    vis = _active_visibility_filter(request)
    if vis is None:
        return None
    try:
        return set(vis.get_visible_agent_ids())
    except Exception:
        logger.warning("Failed to resolve visible agents — denying by default", exc_info=True)
        return set()


def _require_agent_visible(request: Request, agent_id: str) -> None:
    """404 when the target agent is hidden from the caller."""
    vis = _active_visibility_filter(request)
    if vis is not None and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")


# ── Keyset pagination and bounded scans ─────────────────────────────────
#
# Paged list routes return the existing list key plus ``next_cursor`` and
# ``has_more``, the shape the rooms endpoints already use. ``limit``/``offset``
# keep working for callers on the old contract; a cursor, when given, wins.


def _check_cursor(cursor: str | None) -> str | None:
    """Reject a malformed cursor at the boundary with a 400, not a 500 later."""
    if not cursor:
        return None
    try:
        decode_cursor(cursor)
    except InvalidCursorError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid cursor: {exc}") from exc
    return cursor


def _page_meta(page: Page[Any]) -> dict[str, Any]:
    return {"next_cursor": page.next_cursor, "has_more": page.has_more}


def _parse_ts(value: str | None) -> datetime | None:
    """ISO-8601 query parameter to an aware datetime; naive input is taken as UTC.

    Adapters compare these against stored timestamps that are always aware,
    and a naive value would raise from inside that comparison rather than
    here, where it can be a 400.
    """
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {value!r}") from exc
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


async def _via_async_or_sync(method: str, *args: Any, **kwargs: Any) -> Any:
    """Call *method* on the async adapter when there is one, else on the sync adapter.

    The async adapter is the request path's; falling back to the sync adapter
    is what keeps the filesystem adapter and the tests working. A failure on
    the async side is logged and retried on the sync side so a transient
    pool problem degrades to a blocking call rather than an error — the same
    posture the read and search routes take.
    """
    if _async_adapter is not None and hasattr(_async_adapter, method):
        try:
            return await getattr(_async_adapter, method)(*args, **kwargs)
        except InvalidCursorError:
            raise
        except Exception:
            logger.warning("async %s failed — falling back to sync adapter", method, exc_info=True)
    return getattr(_get_memory()._adapter, method)(*args, **kwargs)


async def _list_traces_page(
    *,
    limit: int,
    offset: int = 0,
    cursor: str | None = None,
    **filters: Any,
) -> Page[DecisionTrace]:
    """One page of traces, overfetching by one row to learn ``has_more``."""
    rows = await _via_async_or_sync(
        "list_traces", limit=limit + 1, offset=offset, cursor=cursor, **filters
    )
    return page_from_overfetch(
        list(rows),
        limit=limit,
        timestamp=lambda t: t.created_at,
        tiebreak=lambda t: t.id,
    )


async def _list_events_page(
    agent_id: str,
    namespace: str,
    *,
    limit: int,
    offset: int = 0,
    cursor: str | None = None,
    **filters: Any,
) -> Page[Event]:
    rows = await _via_async_or_sync(
        "list_events", agent_id, namespace, limit=limit + 1, offset=offset, cursor=cursor,
        **filters,
    )
    return page_from_overfetch(
        list(rows),
        limit=limit,
        timestamp=lambda e: e.created_at,
        tiebreak=lambda e: e.id,
    )


async def _list_agent_entries_page(
    agent_id: str,
    *,
    limit: int,
    cursor: str | None = None,
    **filters: Any,
) -> Page[MemoryEntry]:
    rows = await _via_async_or_sync(
        "list_entries_for_agent", agent_id, limit=limit + 1, cursor=cursor, **filters
    )
    return page_from_overfetch(
        list(rows),
        limit=limit,
        timestamp=lambda e: e.provenance.written_at,
        tiebreak=entry_tiebreak,
    )


# How many distinct entity paths the memory-graph page will list to name the
# authors of what an agent read. One indexed query per path; an agent that has
# read across more paths than this gets the first N and ``truncated: true``.
_MEMORY_GRAPH_MAX_READ_PATHS = 200


def _bounded_scan(items: list[Any], ceiling: int) -> tuple[list[Any], bool]:
    """Trim a scan to *ceiling* rows and say whether anything was cut.

    Callers fetch ``ceiling + 1`` so the flag is exact: a result exactly at
    the ceiling is complete, one row past it is not.
    """
    return items[:ceiling], len(items) > ceiling


def _require_account_admin(request: Request) -> None:
    """403 for non-admin members of a multi-user account.

    No-op when there is no per-user visibility context (plain API key on a
    single-user or self-hosted install) or when the user is an account admin.
    """
    if _active_visibility_filter(request) is not None:
        raise HTTPException(status_code=403, detail="Requires account admin access")


def _visible_entity_paths(request: Request) -> set[str] | None:
    """Entity paths the caller may see, or None for unrestricted."""
    vis = _active_visibility_filter(request)
    if vis is None:
        return None
    try:
        paths = {e.entity_path for e in vis.filter_entries(_get_memory().list())}
        paths.update(vis.get_room_map().keys())
        return paths
    except Exception:
        logger.warning("Failed to resolve visible entity paths — denying by default", exc_info=True)
        return set()


def _ensure_agent_owner(
    request: Request, agent_id: str, namespace: str = "default"
) -> str | None:
    """Link the agent to the API key's owner user in the Pro agents table.

    This is a no-op when amfs_rooms is not installed or the request
    has no tenant context.  The linkage lets the UserVisibilityFilter
    discover which agents belong to which user.

    Returns the agent's effective owner_user_id after the upsert, or
    None when ownership tracking is unavailable (pure OSS build, no
    tenant context, or an older amfs_rooms without a return value).
    """
    ctx = getattr(request.state, "tenant_ctx", None)
    user_id = getattr(request.state, "user_id", None)
    if not ctx or not user_id:
        return None
    try:
        from amfs_rooms.visibility import ensure_agent_owner

        mem = _get_memory()
        pool = getattr(mem._adapter, "_pool", None)
        if pool is not None:
            owner = ensure_agent_owner(
                pool,
                agent_id=agent_id,
                owner_user_id=user_id,
                account_id=ctx.account_id,
                namespace=namespace,
            )
            return str(owner) if owner is not None else None
    except ImportError:
        pass
    except Exception:
        logger.debug("Failed to ensure agent owner for %s", agent_id, exc_info=True)
    return None


# Server/internal identities are exempt from ownership enforcement: they are
# excluded from per-user visibility anyway and may legitimately appear in
# requests from any user's tooling.
_SYSTEM_AGENT_IDS = {"amfs-server", "system", "amfs"}


def _link_agent_owner_once(request: Request, agent_id: str, namespace: str) -> None:
    """Link agent → owning user and enforce identity ownership.

    Raises 409 when the agent identity is already owned by a DIFFERENT
    user in the account. Without this, a write under someone else's
    identity silently succeeds but is attributed to the other user:
    invisible to its actual author (per-user visibility follows agent
    ownership) and injected into — or even superseding entries in — the
    owner's view. This bites real setups: sticky identities and generic
    agent names ('claude-desktop', 'agent/<os-user>') collide as soon as
    a second user joins the account.

    Successful self-owned links are cached per (agent, ns, user) so the
    hot write path doesn't repeat the DB call. Unlike the _known_agents
    registration cache, this re-runs when the same agent later appears
    under a different (or first) user attribution, so owner linkage is
    never permanently skipped by an early unattributed write.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is None or agent_id in _SYSTEM_AGENT_IDS:
        return
    cache_key = f"{agent_id}:{namespace}:{user_id}"
    if cache_key in _owner_linked_agents:
        return
    owner = _ensure_agent_owner(request, agent_id, namespace)
    if owner is not None and owner != str(user_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Agent identity '{agent_id}' is already registered to a "
                "different user in this account. Set a unique agent identity "
                "(e.g. call amfs_set_identity with a new name) and retry."
            ),
        )
    _owner_linked_agents.add(cache_key)


# ──────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────


def _health_payload() -> dict[str, str]:
    payload = {
        "status": "ok",
        "version": _server_version(),      # deploy/package version — changes per release
        "schema_version": _SCHEMA_VERSION,  # per-entry MemoryEntry schema tag
    }
    if _BUILD_SHA:
        payload["build"] = _BUILD_SHA
    return payload


@app.get("/")
async def root() -> dict[str, str]:
    # A reachable, unauthenticated 200 at the origin root. Generic API gateways
    # and connector validators (e.g. Fly.io Sprites' custom_api "test request")
    # probe base_url/ to confirm the service is live before saving a connector;
    # without this they get a 404 and refuse the connector. Returns the same
    # health payload as /health.
    return _health_payload()


@app.get("/health")
async def health() -> dict[str, str]:
    return _health_payload()


@app.get("/api/v1/health")
async def health_v1() -> dict[str, str]:
    return _health_payload()


@app.get("/api/v1/auth/whoami")
def whoami(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return information about the authenticated caller.

    When Pro middleware is active, this returns account, key type, scopes,
    and rate limit info. In OSS mode, returns basic auth status.
    """
    ctx = getattr(request.state, "tenant_ctx", None)
    user_id = getattr(request.state, "user_id", None)
    vis = _get_visibility_filter(request)

    if ctx is not None:
        result: dict[str, Any] = {
            "authenticated": True,
            "mode": "pro",
            "account_id": str(ctx.account_id),
            "actor_id": str(ctx.actor_id),
            "key_type": ctx.key_type.value if ctx.key_type else None,
            "role": ctx.role.value if ctx.role else None,
            "scopes": [
                {
                    "entity_path_pattern": s.entity_path_pattern,
                    "permission": s.permission.value,
                }
                for s in ctx.scopes
            ],
            "rate_limit_rpm": ctx.rate_limit_rpm,
            "is_admin": ctx.is_admin,
            "user_id": str(user_id) if user_id else None,
            "visibility_filter_active": vis is not None,
            "visibility_filtering": vis is not None and vis.should_filter() if vis else False,
        }
        if vis is not None and vis.should_filter():
            try:
                result["visible_agents"] = sorted(vis.get_visible_agent_ids())
            except Exception:
                result["visible_agents_error"] = True
        return result
    return {
        "authenticated": _auth is not None,
        "mode": "oss",
        "user_id": str(user_id) if user_id else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Entries
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/entry")
async def read_entry_by_query(
    request: Request,
    entity_path: str = Query(...),
    key: str = Query(...),
    branch: str | None = Query(None),
    response: Response = None,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """The same read, with the coordinates where they cannot be confused.

    ``/api/v1/entries/{entity_path:path}/{key}`` cannot express a key that
    contains a slash. The path converter is greedy, so it takes everything up
    to the final segment as the entity path, and a read of
    ``sweep/abc`` / ``active-negotiation/xyz`` arrives as
    ``sweep/abc/active-negotiation`` / ``xyz`` — coordinates nothing was ever
    written at, answered with a confident "not found".

    Slashed keys are not exotic here: 314 of 4,726 entries in production have
    one, written by the negotiation engine and by anything else that namespaces
    its keys. None of them could be read back. The OpenAI ``fetch`` tool is the
    sharpest edge of that, because it exists to resolve an id that ``search``
    just handed out — so search would offer an entry and fetch would deny it
    existed.

    The old route stays exactly as it was. It is unambiguous whenever the key
    has no slash, which is most of the time, and rewriting every caller to gain
    nothing is a worse trade than leaving them alone.
    """
    return await _read_entry(request, entity_path, key, branch, response)


@app.get("/api/v1/entries/{entity_path:path}/{key}")
async def read_entry(
    request: Request,
    entity_path: str,
    key: str,
    branch: str | None = Query(None),
    response: Response = None,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    return await _read_entry(request, entity_path, key, branch, response)


async def _read_entry(
    request: Request,
    entity_path: str,
    key: str,
    branch: str | None,
    response: Response | None = None,
) -> dict[str, Any]:
    branch = _effective_branch(request, branch)
    mem = _get_memory()
    credited = False
    if _async_adapter is not None:
        try:
            entry = await _async_adapter.read(entity_path, key, branch=branch)
        except Exception:
            logger.warning("Async read failed for %s/%s — falling back to sync", entity_path, key, exc_info=True)
            entry = None
        if entry is None:
            entry = mem.read(entity_path, key, branch=branch)
            if entry is not None:
                logger.warning(
                    "Async adapter missed entry %s/%s but sync found it — RLS context mismatch",
                    entity_path, key,
                )
        if entry is not None:
            asyncio.create_task(_async_adapter.increment_recall_count(entity_path, key, branch=branch))
            credited = True
    else:
        entry = mem.read(entity_path, key, branch=branch)
    if entry is None:
        return {"status": "not_found", "entity_path": entity_path, "key": key}

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_entry_visible(entry):
        return {"status": "not_found", "entity_path": entity_path, "key": key}

    # After the visibility check, never before it. The recall bump above is
    # issued on the entry as fetched, but the block carries the author's agent id
    # for the cross-surface claim — attached earlier, a read of an entry this
    # caller may not see would answer "not found" while the header named who
    # wrote it. The bump is the only thing that legitimately precedes the check,
    # because it records that the row was touched and reveals nothing.
    if credited:
        _attach_reuse_value(
            response, request, credited=entry, hits=1, surface="read", branch=branch
        )
    return _entry_to_response(entry)


@app.get("/api/v1/quality/{entity_path:path}/{key}")
def entry_quality(
    request: Request,
    entity_path: str,
    key: str,
    branch: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Compute a quality report for a stored entry on demand."""
    branch = _effective_branch(request, branch)
    mem = _get_memory()
    # Read straight from the adapter — this is an internal/dashboard inspection,
    # not an agent recall, so it must NOT increment recall_count (doing so
    # inflates reuse metrics every time a topic/agent page is viewed).
    entry = mem._adapter.read(entity_path, key, branch=branch)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    vis = _active_visibility_filter(request)
    if vis is not None and not vis.is_entry_visible(entry):
        raise HTTPException(status_code=404, detail="Entry not found")

    try:
        existing_entries = mem.list(entity_path)
        if vis is not None:
            existing_entries = vis.filter_entries(existing_entries)
        existing_keys = [e.key for e in existing_entries if e.key != key]
    except Exception:
        existing_keys = []

    evaluator = HeuristicQualityEvaluator()
    mt = entry.memory_type.value if hasattr(entry.memory_type, "value") else str(entry.memory_type)
    report = evaluator.evaluate(
        entry.value,
        entity_path=entity_path,
        key=key,
        confidence=entry.confidence,
        memory_type=mt,
        pattern_refs=list(entry.provenance.pattern_refs),
        existing_keys=existing_keys,
    )
    return {
        "entity_path": entity_path,
        "key": key,
        "quality": report.model_dump(mode="json"),
    }


async def _scope_block(
    entity_path: str,
    key: str,
    *,
    branch: str = "main",
    request: Request | None = None,
    agent_id: str | None = None,
) -> dict[str, Any] | None:
    """What is already stored beside a write, for handing back to the agent.

    The factual half of the read gap. An agent writes far more than it reads, and
    the cheapest thing that changes that is telling it, at the moment it writes,
    that "six entries are already here and four have never been read back" — a
    statement about its own store, carrying no instruction. Returned as data in a
    tool result, so it reaches every client rather than only the ones that
    support hooks.

    Computed here rather than by the caller, and that is the whole design. The
    gateway used to ask separately, which is a second ``GET /api/v1/entries`` and
    therefore a third billed op on every write — enough of a cost that the
    feature shipped switched off and stayed off. Inline it is one aggregate in a
    request that was already happening, so nothing new is metered and the block
    can simply be on.

    ``None`` when there are no neighbours: a scope containing only the entry just
    written has nothing to report, and an empty block would read as a finding.

    Two filters apply, and they answer different questions. *agent_id* goes to
    the adapter, which counts an entry only if it is shared or this agent wrote
    it — the rule ``AgentMemory.list`` enforces, which an aggregate that skips
    ``list`` would otherwise drop. The per-user visibility filter below then
    narrows the key sample further where an account separates its users. The
    first cannot be replaced by the second: it is inactive on most deployments,
    and only ever touched the sample, never the counts.
    """
    try:
        adapter = _async_adapter if _async_adapter is not None else _get_memory()._adapter
        counts = adapter.scope_counts(
            entity_path, exclude_key=key, branch=branch, agent_id=agent_id
        )
        if inspect.isawaitable(counts):
            counts = await counts
    except Exception:  # noqa: BLE001 - a write must not fail on its own footnote
        logger.debug("scope block failed for %s", entity_path, exc_info=True)
        return None

    if not counts or not counts.get("existing_entries"):
        return None

    # Visibility is applied to the key sample, which is the only part that names
    # anything. The counts describe the namespace the caller just wrote into.
    vis = _get_visibility_filter(request) if request is not None else None
    if vis is not None and vis.should_filter():
        try:
            listed = await _offload(
                _db_executor, _get_memory()._adapter.list, entity_path, branch=branch
            )
            visible = {e.key for e in vis.filter_entries(listed)}
            counts = {**counts, "keys": [k for k in counts.get("keys", []) if k in visible]}
        except Exception:  # noqa: BLE001 - drop the sample rather than leak it
            counts = {**counts, "keys": []}
    return counts


@app.post("/api/v1/entries")
async def write_entry(
    req: WriteRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    type_map = {m.value: m for m in MemoryType}
    mt = type_map.get(req.memory_type.lower(), MemoryType.FACT)

    # The caller's identity lives on a per-request handle. This block awaits
    # (ensure_agent, the write-time embedding, the async write), and a swap of
    # the shared tagger restored in a ``finally`` is exactly what stamped
    # concurrent writes with each other's agent — see ``AgentMemory.as_agent``.
    handle = mem
    if req.agent_id or req.session_id:
        handle = mem.as_agent(req.agent_id or mem.agent_id)
        if req.session_id:
            handle._tagger.session_id = req.session_id

    _used_async = False
    if _async_adapter is not None:
        _agent_ns = _async_adapter._namespace
        if req.agent_id:
            _agent_cache_key = f"{req.agent_id}:{_agent_ns}"
            if _agent_cache_key not in _known_agents:
                try:
                    await _async_adapter.ensure_agent(req.agent_id, _agent_ns)
                    _known_agents.add(_agent_cache_key)
                except Exception:
                    pass
            _link_agent_owner_once(request, req.agent_id, _agent_ns)

        from amfs_core.content import embedding_input
        from amfs_core.models import Provenance
        provenance = handle._tagger.tag(pattern_refs=req.pattern_refs or None)
        # Classify here so the flag is set before the inline-built entry hits
        # the async adapter, and embed a clean descriptor for artifacts.
        _is_artifact, _embed_text = embedding_input(req.key, req.value)
        entry_obj = MemoryEntry(
            entity_path=req.entity_path,
            key=req.key,
            version=1,
            value=req.value,
            provenance=provenance,
            confidence=req.confidence,
            memory_type=mt,
            shared=req.shared,
            branch=req.branch,
            is_artifact=_is_artifact,
        )
        # Write-time embedding for semantic retrieval. The async adapter
        # persists entry.embedding when the pgvector column exists; without
        # this the hot write path stores no vector (embeddings never land).
        # Crash-safe: a failure here just stores the entry without a vector.
        _embedder = _get_server_embedder()
        if _embedder is not None:
            try:
                entry_obj = entry_obj.model_copy(
                    update={
                        "embedding": await _offload(
                            _model_executor, _embedder.embed, _embed_text
                        )
                    }
                )
            except Exception:  # noqa: BLE001 - never fail a write on embedding
                logger.warning(
                    "write-time embedding failed for %s/%s — storing without vector",
                    req.entity_path, req.key, exc_info=True,
                )
        try:
            entry = await _async_adapter.write(entry_obj)
            _used_async = True
        except Exception:
            logger.warning(
                "Async write failed for %s/%s — falling back to sync adapter",
                req.entity_path, req.key, exc_info=True,
            )
            entry = handle.write(
                req.entity_path,
                req.key,
                req.value,
                confidence=req.confidence,
                pattern_refs=req.pattern_refs or None,
                memory_type=mt,
                shared=req.shared,
                branch=req.branch,
            )
    if not _used_async and _async_adapter is None:
        if req.agent_id:
            _agent_cache_key = f"{req.agent_id}:{mem.namespace}"
            if _agent_cache_key not in _known_agents:
                try:
                    mem._adapter.ensure_agent(req.agent_id, mem.namespace)
                    _known_agents.add(_agent_cache_key)
                except Exception:
                    pass
            _link_agent_owner_once(request, req.agent_id, mem.namespace)
        entry = handle.write(
            req.entity_path,
            req.key,
            req.value,
            confidence=req.confidence,
            pattern_refs=req.pattern_refs or None,
            memory_type=mt,
            shared=req.shared,
            branch=req.branch,
        )
    _sse_manager.broadcast(entry)

    _resource = f"{req.entity_path}/{req.key}"
    _ip = request.client.host if request.client else None
    _agent = entry.provenance.agent_id
    _ek = f"{entry.entity_path}/{entry.key}"
    _ns = _async_adapter._namespace if _async_adapter else mem.namespace
    _branch = entry.branch or "main"
    _confidence = entry.confidence

    if _async_adapter is not None:
        async def _bg_async_side_effects() -> None:
            try:
                await _async_adapter.log_event(Event(
                    namespace=_ns,
                    agent_id=_agent,
                    branch=_branch,
                    event_type=EventType.WRITE,
                    summary=f"Wrote {req.entity_path}/{req.key} v{entry.version}",
                    details={
                        "entity_path": req.entity_path,
                        "key": req.key,
                        "version": entry.version,
                        "confidence": _confidence,
                        "memory_type": mt.value,
                        "shared": req.shared,
                    },
                ))
            except Exception:
                logger.debug("bg: Failed to log write event", exc_info=True)
            try:
                await _async_adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=_agent,
                        source_type="agent",
                        relation="wrote",
                        target_entity=_ek,
                        target_type="entry",
                        confidence=_confidence,
                        provenance={"agent_id": _agent, "trigger": "write"},
                    ),
                    namespace=_ns,
                    branch=_branch,
                )
            except Exception:
                logger.debug("bg: Failed to materialize wrote edge", exc_info=True)

        asyncio.create_task(_bg_async_side_effects())
    else:
        _tenant_account_id: str | None = None
        _tenant_team_id: str | None = None
        _tenant_is_admin: bool = False
        try:
            from amfs_postgres.tenant_context import (
                get_request_tenant_account_id,
                get_request_tenant_team_id,
                get_request_is_account_admin,
            )
            _tenant_account_id = get_request_tenant_account_id()
            _tenant_team_id = get_request_tenant_team_id()
            _tenant_is_admin = get_request_is_account_admin()
        except ImportError:
            pass

        def _bg_write_side_effects() -> None:
            try:
                from amfs_postgres.tenant_context import (
                    set_tls_tenant_account_id,
                    set_tls_tenant_team_id,
                    set_tls_is_account_admin,
                    clear_tls_tenant_account_id,
                    clear_tls_tenant_team_id,
                    clear_tls_is_account_admin,
                )
                set_tls_tenant_account_id(_tenant_account_id)
                set_tls_tenant_team_id(_tenant_team_id)
                set_tls_is_account_admin(_tenant_is_admin)
            except ImportError:
                pass

            try:
                _audit_log("memory.write", resource=_resource, ip_address=_ip)
            except Exception:
                logger.debug("bg: Failed to write audit log", exc_info=True)
            try:
                mem._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=_agent,
                        source_type="agent",
                        relation="wrote",
                        target_entity=_ek,
                        target_type="entry",
                        confidence=_confidence,
                        provenance={"agent_id": _agent, "trigger": "write"},
                    ),
                    namespace=_ns,
                    branch=_branch,
                )
            except Exception:
                logger.debug("bg: Failed to materialize wrote edge", exc_info=True)
            finally:
                try:
                    from amfs_postgres.tenant_context import (
                        clear_tls_tenant_account_id,
                        clear_tls_tenant_team_id,
                        clear_tls_is_account_admin,
                    )
                    clear_tls_tenant_account_id()
                    clear_tls_tenant_team_id()
                    clear_tls_is_account_admin()
                except ImportError:
                    pass

        _bg_executor.submit(_bg_write_side_effects)

    out = _entry_to_response(entry)
    if req.include_scope:
        # Taken from the entry just written rather than the request or the
        # tagger: the tagger is restored to its previous identity by this point,
        # and req.agent_id is optional, while provenance records who the write
        # was actually attributed to. That is the identity whose private
        # neighbours may be counted.
        scope = await _scope_block(
            req.entity_path,
            req.key,
            branch=_branch,
            request=request,
            agent_id=getattr(getattr(entry, "provenance", None), "agent_id", None),
        )
        if scope is not None:
            out["scope"] = scope
    return out


@app.get("/api/v1/entries")
async def list_entries(
    request: Request,
    entity_path: str | None = Query(None),
    branch: str | None = Query(None),
    include_superseded: bool = Query(False),
    limit: int | None = Query(None, ge=1, le=10_000),
    offset: int = Query(0, ge=0),
    sort: str | None = Query(None, pattern="^(written_at|recall_count)$"),
    fields: str | None = Query(None, pattern="^meta$"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    try:
        from amfs_postgres.tenant_context import get_request_tenant_account_id
        _tls_acct = get_request_tenant_account_id()
    except ImportError:
        _tls_acct = "NO_MODULE"
    _state_acct = getattr(request.state, "account_id", None)
    _state_user = getattr(request.state, "user_id", None)
    _has_ctx = getattr(request.state, "tenant_ctx", None) is not None
    logger.warning(
        "[TLS-DIAG] /entries tls_account=%s state_account=%s state_user=%s has_tenant_ctx=%s",
        _tls_acct, _state_acct, _state_user, _has_ctx,
    )
    branch = _effective_branch(request, branch)
    mem = _get_memory()
    if limit is None and ENTRIES_DEFAULT_LIMIT > 0:
        limit = ENTRIES_DEFAULT_LIMIT
    scope, py_vis = _visibility_scope(request)
    # Only the Postgres adapters page in SQL; the filesystem adapter's list()
    # takes none of the keyword arguments and is paged below as before.
    sql_paged = _async_adapter is not None or hasattr(mem._adapter, "count_entries")

    if py_vis is None and sql_paged:
        # Visibility, order and page all in the query: the database returns
        # the page, not the namespace. Before this, an account-wide listing
        # loaded every current entry (100K on the largest account), filtered
        # and sorted them in Python on the event loop, and sliced the page off
        # the end — for a caller asking for 50 rows.
        # The sync adapter is reached through AgentMemory.list(), which hides
        # other agents' private entries; the async adapter never did. Each
        # path keeps the behaviour it had, as a predicate rather than a pass.
        sync_scope = SqlScope.all_of(
            scope, SqlScope("shared OR agent_id = %s", (mem.agent_id,))
        )
        page_kw: dict[str, Any] = {
            "branch": branch,
            "include_superseded": include_superseded,
            "scope": scope,
            "order_by": sort,
            "limit": limit,
            "offset": offset,
        }
        sync_kw = {**page_kw, "scope": sync_scope}
        count_keys = ("branch", "include_superseded", "scope")
        sync_list = functools.partial(mem._adapter.list, entity_path, **sync_kw)
        try:
            if _async_adapter is not None:
                entries = await _async_adapter.list(entity_path, **page_kw)
                count_kw = {k: page_kw[k] for k in count_keys}
            else:
                entries = await _offload(_db_executor, sync_list)
                count_kw = {k: sync_kw[k] for k in count_keys}
        except Exception:
            logger.warning(
                "Paged list failed for %s — falling back to sync", entity_path, exc_info=True
            )
            entries = []
            count_kw = {k: sync_kw[k] for k in count_keys}
        recovered_by_sync = False
        if not entries and offset == 0 and _async_adapter is not None:
            # The async pool once lost its tenant context and answered every
            # read with nothing; cheap to rule out on an empty first page.
            sync_entries = await _offload(_db_executor, sync_list)
            if sync_entries:
                logger.warning(
                    "Async adapter returned 0 entries for %s but sync found %d — RLS mismatch",
                    entity_path, len(sync_entries),
                )
                entries = sync_entries
                recovered_by_sync = True
        if limit is None and offset == 0:
            total = len(entries)
        elif _async_adapter is not None and not recovered_by_sync:
            total = await _async_adapter.count_entries(entity_path, **count_kw)
        else:
            # The page came from the sync adapter, so the count must too: the
            # async pool that returned nothing would count nothing, and the
            # sync page was scoped by sync_scope, not by count_kw's scope.
            if recovered_by_sync:
                count_kw = {k: sync_kw[k] for k in count_keys}
            total = await _offload(
                _db_executor, mem._adapter.count_entries, entity_path, **count_kw
            )
        logger.warning(
            "[ENTRIES] entity_path=%s sql_paged rows=%d total=%d scoped=%s sort=%s "
            "limit=%s offset=%d",
            entity_path, len(entries), total, scope is not None, sort, limit, offset,
        )
    else:
        # A filter that can only run over loaded entries, or an adapter that
        # cannot page: load off the event loop, then filter, sort and page
        # here — in that order, so a caller can never page past entries it
        # is not allowed to see.
        load_all = functools.partial(
            mem.list, entity_path, branch=branch, include_superseded=include_superseded
        )
        if _async_adapter is not None:
            try:
                entries = await _async_adapter.list(
                    entity_path, branch=branch, include_superseded=include_superseded
                )
            except Exception:
                logger.warning(
                    "Async list failed for %s — falling back to sync", entity_path, exc_info=True
                )
                entries = []
            if not entries:
                sync_entries = await _offload(_db_executor, load_all)
                if sync_entries:
                    logger.warning(
                        "Async adapter returned 0 entries for %s but sync found %d — RLS mismatch",
                        entity_path, len(sync_entries),
                    )
                    entries = sync_entries
        else:
            entries = await _offload(_db_executor, load_all)
        total_before = len(entries)
        if py_vis is not None:
            entries = await _offload(_db_executor, py_vis.filter_entries, entries)
        logger.warning(
            "[ENTRIES] entity_path=%s mem.list=%d after_filter=%d filtered=%s",
            entity_path, total_before, len(entries), py_vis is not None,
        )

        if sort == "written_at":
            entries = sorted(entries, key=lambda e: e.provenance.written_at, reverse=True)
        elif sort == "recall_count":
            entries = sorted(entries, key=lambda e: e.recall_count, reverse=True)

        total = len(entries)
        if offset:
            entries = entries[offset:]
        if limit is not None:
            entries = entries[:limit]

    payload = [_entry_to_response(e) for e in entries]
    if fields == "meta":
        for item in payload:
            item.pop("value", None)
            item.pop("artifact_refs", None)

    return {"entries": payload, "total": total}


@app.get("/api/v1/entities")
async def list_entity_summaries(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Per-entity aggregates without entry values — a few KB instead of the
    multi-MB /entries payload. Dashboards should prefer this endpoint."""
    mem = _get_memory()
    adapter = mem._adapter

    scope, py_vis = _visibility_scope(request)

    def _summaries() -> list[dict[str, Any]]:
        """Off the event loop: one GROUP BY where the adapter and the
        visibility rule allow it, a load-and-reduce otherwise."""
        if py_vis is None:
            try:
                # The visibility rule travels into the GROUP BY as a
                # predicate, so a per-user dashboard gets the same one query
                # an admin does. Only passed when there is one: the ABC's
                # default implementation does not take it.
                if scope is None:
                    return adapter.entity_summaries()
                return adapter.entity_summaries(scope=scope)
            except TypeError:
                # An adapter without SQL scoping; reduce over its rows.
                pass
        from amfs_core.aggregates import entity_summaries_from_entries

        entries = mem.list()
        vis = py_vis if py_vis is not None else _active_visibility_filter(request)
        if vis is not None:
            entries = vis.filter_entries(entries)
        return entity_summaries_from_entries(entries)

    summaries = await _offload(_db_executor, _summaries)
    return json.loads(json.dumps({"entities": summaries}, default=str))


@app.post("/api/v1/aggregate")
def aggregate_entries_endpoint(
    request: Request,
    req: AggregateRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Compute one aggregate server-side over the records under an entity path.

    Returns only the computed result, not the raw records — so an agent can get
    a sum/mean/group-by across thousands of records without pulling any of them
    into context. The visibility filter is applied BEFORE reducing (same order
    as /api/v1/entities), so a caller can never aggregate over entries they
    can't read.
    """
    from amfs_core.aggregates import AGGREGATE_OPS, aggregate_entries

    if req.op not in AGGREGATE_OPS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown op {req.op!r}; expected one of {list(AGGREGATE_OPS)}",
        )
    if req.op != "count" and not req.field:
        raise HTTPException(
            status_code=400, detail=f"op {req.op!r} requires a 'field'"
        )

    mem = _get_memory()
    entries = mem.list(req.entity_path, branch=_effective_branch(request, req.branch))

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        entries = vis.filter_entries(entries)

    try:
        result = aggregate_entries(
            entries,
            op=req.op,
            field=req.field,
            group_by=req.group_by,
            row_path=req.row_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["entity_path"] = req.entity_path
    result["entries_scanned"] = len(entries)
    return json.loads(json.dumps(result, default=str))


# ──────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────


def _attach_reuse_value(
    response: Response | None,
    request: Request,
    *,
    credited: MemoryEntry | None,
    hits: int,
    surface: str | None = None,
    branch: str = "main",
) -> None:
    """Compute the reuse block for a credited read, return it and persist it.

    Called at the point ``recall_count`` is bumped, which is the only place that
    already knows which entry the reuse is being credited to. Everything the
    block needs is in hand there: the entry's own content size, its stored
    recall count before this reuse, who wrote it, and — from the header the
    gateway has always sent — who is reading it now.

    The row written to ``amfs_reuse_events`` carries the same estimate as the
    block, taken from the block rather than recomputed, so a figure in a weekly
    digest cannot disagree with the figure the user was shown in chat. The block
    answers "what did memory just do for you"; the row is what lets anyone ask
    that later, when the session it happened in is long gone.

    Best-effort in the same sense the recall bump above it is: this is reporting,
    and a defect in it must never change the answer the caller came for.
    """
    if credited is None or hits <= 0:
        return
    try:
        written_by = _known_agent_id(
            getattr(getattr(credited, "provenance", None), "agent_id", None)
        )
        reused_by = _known_agent_id(request.headers.get("x-amfs-agent-id"))
        content_chars = entry_content_chars(credited)
        block = reuse_value_block(
            hits=hits,
            content_chars=content_chars,
            reused_before=getattr(credited, "recall_count", 0) or 0,
            written_by=written_by,
            reused_by=reused_by,
            # Which row the credit landed on, so a caller answering with one
            # memory can check the block is about that memory. recall and
            # read_from credit the current version and may then answer with an
            # older one from history.
            credited={
                "entity_path": credited.entity_path,
                "key": credited.key,
                "version": getattr(credited, "version", None),
            },
        )
        if not block:
            return
        # A caller invoking the handler in-process has no response to decorate,
        # and the reuse still happened, so the row is written either way.
        if response is not None:
            # Separators without spaces: a header value is not read by a human and
            # the default ", " padding is wasted bytes on every read response.
            response.headers[REUSE_VALUE_HEADER] = json.dumps(
                block, separators=(",", ":"), default=str
            )
        _persist_reuse_event(
            credited,
            # The row stores the raw integer, and the block carries the same
            # figure formatted for display ("~1.2K"). Both come from
            # recall_tokens_for_chars with the same inputs, so there is still one
            # source for the number — passing the block's own field instead would
            # store a string that int() rejects, and since the write swallows its
            # errors, that dropped every row in silence.
            est_tokens_saved=recall_tokens_for_chars(content_chars, hits=hits),
            written_by=written_by,
            reused_by=reused_by,
            surface=surface,
            branch=branch,
        )
    except Exception:  # noqa: BLE001 - reporting must not break the read
        logger.debug("reuse value block failed", exc_info=True)


def _known_agent_id(value: str | None) -> str | None:
    """An agent id only when one was actually supplied.

    ``request.headers.get`` yields ``""`` for a header that is present and empty,
    and an empty string is not NULL, so it would satisfy ``reused_by IS NOT NULL``
    in the summary and be counted as a *different* agent reusing the memory. The
    block already treats it as unknown, so without this the stored row and the
    line the user was shown disagree about the strongest claim either can make.
    """
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _persist_reuse_event(
    credited: MemoryEntry,
    *,
    est_tokens_saved: int,
    written_by: str | None,
    reused_by: str | None,
    surface: str | None,
    branch: str,
) -> None:
    """Keep the reuse the block just described, so it outlives the session.

    Scheduled rather than awaited, like the recall bump it accompanies: the read
    has already been answered and nothing about it should wait on bookkeeping.
    Silent when there is no async adapter (an in-process caller, or a filesystem
    backend) and when there is no running loop, because both mean there is
    nowhere to write and neither is an error.
    """
    recorder = getattr(_async_adapter, "record_reuse_event", None)
    if recorder is None:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    asyncio.create_task(
        recorder(
            credited.entity_path,
            credited.key,
            branch=branch,
            entry_version=getattr(credited, "version", None),
            written_by=written_by,
            reused_by=reused_by,
            est_tokens_saved=est_tokens_saved,
            surface=surface,
        )
    )


@app.post("/api/v1/search")
async def search_entries(
    request: Request,
    req: SearchRequest,
    # Injected by FastAPI on the type despite the default, which is what keeps
    # this callable directly with no knowledge of it: the reuse header is
    # reporting, and a caller invoking the handler in-process — the retrieval
    # tests, and anything Pro composes — should not have to supply a response
    # object to ask a question.
    response: Response = None,
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    branch = _effective_branch(request, getattr(req, "branch", None))
    sq = SearchQuery(
        query=req.query,
        entity_path=req.entity_path,
        min_confidence=req.min_confidence,
        max_confidence=req.max_confidence,
        agent_id=req.agent_id,
        since=req.since,
        pattern_ref=req.pattern_ref,
        sort_by=req.sort_by,
        limit=req.limit,
        depth=req.depth,
        include_artifacts=req.include_artifacts,
        include_descendants=req.include_descendants,
    )
    mem = _get_memory()
    if _async_adapter is not None:
        try:
            results = await _async_adapter.search(sq, branch=branch)
        except Exception:
            logger.warning("Async search failed — falling back to sync", exc_info=True)
            results = []
        if not results:
            # Runs on every search that finds nothing, so it goes off the loop
            # like the primary path did.
            sync_results = await _offload(_db_executor, _sync_search, mem._adapter, sq, branch)
            if sync_results:
                logger.warning(
                    "Async search returned 0 results but sync found %d — RLS context mismatch",
                    len(sync_results),
                )
                results = sync_results
    else:
        results = await _offload(_db_executor, _sync_search, mem._adapter, sq, branch)

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        results = vis.filter_entries(results)

    # Reuse accounting. `amfs_search` is the read surface some agent profiles
    # (e.g. the Base44 builder profile) expose instead of /retrieve, so a
    # text-driven search is a real recall — but historically it incremented no
    # counter, leaving reuse metrics (memories reused / rework avoided) reading
    # 0 even when memory was clearly used. Only credit *query-driven* searches
    # (recall intent), not pure filter/browse (agent_id/entity_path listings),
    # and only the TOP hit: the rest of a ranked list is a candidate set the
    # agent scrolls past, and crediting three per query inflated reuse roughly
    # threefold (one real session: 4 lookups booked as 16 reuses).
    # Skip system/bench scratch namespaces. Best-effort: never let accounting
    # failure affect the response.
    if req.query and req.query.strip():
        credited = 0
        credited_entry = None
        for entry in results:
            if credited >= REUSE_CREDIT_K:
                break
            # Scratch namespaces don't consume the credit: a telemetry row
            # ranking first would otherwise silently swallow the whole budget.
            if entry.entity_path.startswith(("_system/", "bench/", "bench-")):
                continue
            credited += 1
            if credited_entry is None:
                credited_entry = entry
            try:
                if _async_adapter is not None:
                    await _async_adapter.increment_recall_count(
                        entry.entity_path, entry.key, branch=branch
                    )
                else:
                    _get_memory()._adapter.increment_recall_count(
                        entry.entity_path, entry.key, branch=branch
                    )
            except Exception:  # noqa: BLE001 - reuse accounting is best-effort
                logger.debug("search recall bump failed", exc_info=True)
        _attach_reuse_value(
            response,
            request,
            credited=credited_entry,
            hits=credited,
            surface="search",
            branch=branch,
        )

    return [_entry_to_response(e) for e in results]


@app.post("/api/v1/retrieve")
async def retrieve_entries(
    request: Request,
    req: RetrieveRequest,
    # See search_entries: injected on the type, defaulted so the handler stays
    # callable in-process without one.
    response: Response = None,
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Semantic (meaning-based) retrieval.

    Ranks entries by embedding similarity to the query, blended with recency
    and confidence, so plain-language queries match paraphrased memories.
    Account isolation comes from the RLS-scoped async adapter; user/room
    visibility is then enforced by UserVisibilityFilter (same as /search).

    Hybrid by construction: candidates are the UNION of pgvector semantic
    neighbours and OR-tsquery lexical matches (not a zero-hit-only fallback),
    so a paraphrase match and an exact-term match both surface. Temporal intent
    ("yesterday", "last week") is parsed out of the query into a recency signal
    instead of polluting the embedded text. Account isolation (RLS) + per-user/
    room visibility (UserVisibilityFilter) are applied ONCE over the merged set.
    A Pro-injected cross-encoder reranker and query rewriter refine the top
    candidates when present. Degrades gracefully to lexical-only when the
    embedder or pgvector column is unavailable.
    """
    from datetime import datetime as _dt, timezone as _tz

    from amfs_core.content import ARTIFACT_PENALTY, PROCEDURE_BOOST, classify_artifact
    from amfs_core.query_norm import normalize_temporal

    branch = _effective_branch(request, req.branch)
    vis = _get_visibility_filter(request)
    embedder = _get_server_embedder()

    # 1. Split temporal intent out of the query so it becomes a recency signal
    #    rather than embedded/lexical noise.
    tnorm = normalize_temporal(req.query)
    topical = tnorm.topical
    recency_weight = req.recency_weight * tnorm.recency_weight_boost

    # 2. Query variants: the Pro-injected rewriter adds paraphrases/HyDE; in OSS
    #    this is just the topical query.
    queries: list[str] = [topical]
    rewriter = _retrieval_query_rewriter
    if rewriter is not None:
        try:
            for q in rewriter.expand(topical):
                if q and q not in queries:
                    queries.append(q)
        except Exception:  # noqa: BLE001 - rewrite is best-effort
            logger.debug("query rewrite failed", exc_info=True)

    # Over-fetch a wide candidate pool so the blend + visibility filtering have
    # headroom before trimming to `limit`.
    pool = max(req.limit * 10, 150)

    # entry_key -> {"entry", "sim", "keyword"}
    candidates: dict[str, dict[str, Any]] = {}

    # 3a. Semantic channel (per query variant), keep best similarity per entry.
    #     Each variant is embedded once, off the event loop, and the vector is
    #     reused by the below-gate read further down.
    query_vectors: dict[str, list[float]] = {}
    if embedder is not None and _async_adapter is not None:
        for qtext in queries:
            # Best-effort: an embedder that cannot be driven from here (a test
            # stub, an unexpected model failure) leaves the adapter to embed
            # the text itself, exactly as it did before the vector was shared.
            if callable(getattr(embedder, "embed", None)):
                try:
                    query_vectors[qtext] = await _offload(_model_executor, embedder.embed, qtext)
                except Exception:  # noqa: BLE001
                    logger.debug("query embedding failed — adapter will embed", exc_info=True)
            sq = SemanticQuery(
                text=qtext,
                entity_path=req.entity_path,
                min_confidence=req.min_confidence,
                limit=pool,
                embedding=query_vectors.get(qtext),
            )
            try:
                pairs = await _async_adapter.semantic_search(sq, embedder, branch=branch)
            except Exception:
                logger.warning("semantic_search failed — continuing with lexical", exc_info=True)
                pairs = []
            for entry, sim in pairs:
                slot = candidates.get(entry.entry_key)
                if slot is None:
                    candidates[entry.entry_key] = {"entry": entry, "sim": sim, "keyword": 0.0}
                elif sim > slot["sim"]:
                    slot["sim"] = sim

    # 3b. Lexical channel (first-class, always run — OR-tsquery recall).
    sq_lex = SearchQuery(
        query=topical,
        entity_path=req.entity_path,
        min_confidence=req.min_confidence,
        limit=pool,
        sort_by="confidence",
        depth=3,
        include_artifacts=req.include_artifacts,
    )
    lex_entries: list[MemoryEntry] = []
    if _async_adapter is not None:
        try:
            lex_entries = await _async_adapter.search(sq_lex, branch=branch)
        except Exception:
            logger.debug("async lexical search failed", exc_info=True)
            lex_entries = []
    for entry in lex_entries:
        slot = candidates.get(entry.entry_key)
        if slot is None:
            candidates[entry.entry_key] = {"entry": entry, "sim": 0.0, "keyword": 1.0}
        else:
            slot["keyword"] = 1.0

    # 3c. Sync fallback only if nothing came back from the async paths (e.g.
    #     non-postgres deployment or embedder+async both unavailable).
    if not candidates:
        mem = _get_memory()
        try:
            results = await _offload(_db_executor, _sync_search, mem._adapter, sq_lex, branch)
        except Exception:
            results = []
        for entry in results:
            candidates.setdefault(
                entry.entry_key, {"entry": entry, "sim": 0.0, "keyword": 1.0}
            )

    # 3d. Below-gate read, scoped to this query. A discredited entry sits under
    #     the discredit threshold by definition, so a caller gating at or above
    #     it (the benchmark's setting, and a reasonable production one) never
    #     sees the rule that stopped working — nor, until now, the rule that
    #     stopped working *elsewhere* but still works for tasks like this one.
    #     Read semantically with the query vector already in hand, so each row
    #     arrives with its similarity: only the discredited ones are kept, and
    #     only as candidates for the avoid list, the local-evidence rescue in
    #     6b, and the query-scoped regime-shift reading in 12. Entries that are
    #     merely low-confidence stay hidden, as the caller's gate asks.
    below_gate: dict[str, dict[str, Any]] = {}
    if req.entity_path and req.min_confidence > 0.0:
        gate_ceiling = min(req.min_confidence, DISCREDIT_THRESHOLD)
        # Semantic, with the vector already in hand.
        if _async_adapter is not None and embedder is not None and topical in query_vectors:
            try:
                pairs = await _async_adapter.semantic_search(
                    SemanticQuery(
                        text=topical,
                        entity_path=req.entity_path,
                        min_confidence=0.0,
                        max_confidence=gate_ceiling,
                        limit=BELOW_GATE_LIMIT,
                        embedding=query_vectors[topical],
                    ),
                    embedder,
                    branch=branch,
                )
            except Exception:  # noqa: BLE001 - best-effort
                logger.debug("below-gate semantic read failed", exc_info=True)
                pairs = []
            for entry, sim in pairs:
                if getattr(entry, "discredited_at", None) is None or entry.entry_key in candidates:
                    continue
                below_gate[entry.entry_key] = {"entry": entry, "sim": sim, "keyword": 0.0}
        # Lexical, always: the channel the ranked list itself always runs, and
        # the only one a store without vectors has.
        below_lex = SearchQuery(
            query=topical,
            entity_path=req.entity_path,
            min_confidence=0.0,
            max_confidence=gate_ceiling,
            limit=BELOW_GATE_LIMIT,
            sort_by="recency",
            depth=3,
            include_artifacts=req.include_artifacts,
        )
        lex_below: list[MemoryEntry] = []
        try:
            if _async_adapter is not None:
                lex_below = await _async_adapter.search(below_lex, branch=branch)
            else:
                lex_below = await _offload(
                    _db_executor, _search_sync, _get_memory()._adapter, below_lex, branch
                )
        except Exception:  # noqa: BLE001
            logger.debug("below-gate lexical read failed", exc_info=True)
        for entry in lex_below:
            if getattr(entry, "discredited_at", None) is None or entry.entry_key in candidates:
                continue
            slot = below_gate.get(entry.entry_key)
            if slot is None:
                below_gate[entry.entry_key] = {"entry": entry, "sim": 0.0, "keyword": 1.0}
            else:
                slot["keyword"] = 1.0
        candidates.update(below_gate)

    # 4. Drop benchmark/system scratch namespaces from user recall, and the
    #    system-written contrast lessons: those are consumed by the briefing
    #    (folded into the discredited section as "replaced by") and are not
    #    knowledge an agent should read or be credited for. The lessons this
    #    query surfaced are kept aside for the avoid list, which names what
    #    replaced each avoided entry the same way the briefing does.
    lessons: list[MemoryEntry] = [
        v["entry"] for v in candidates.values()
        if _is_synthetic_key(getattr(v["entry"], "key", ""))
        and not _is_excluded_entity(getattr(v["entry"], "entity_path", ""))
    ]
    candidates = {
        k: v
        for k, v in candidates.items()
        if not _is_excluded_entity(getattr(v["entry"], "entity_path", ""))
        and not _is_synthetic_key(getattr(v["entry"], "key", ""))
    }

    # 5. Visibility (account RLS already scoped the fetch; this adds per-user +
    #    room scoping) applied ONCE over the full merged set so lexical-only
    #    hits are filtered exactly like semantic ones — no leak path. The
    #    lessons kept aside are filtered the same way: a replacement link
    #    read from a lesson the caller cannot see is a leak by another name.
    if vis is not None and vis.should_filter():
        allowed = {e.entry_key for e in vis.filter_entries([v["entry"] for v in candidates.values()])}
        candidates = {k: v for k, v in candidates.items() if k in allowed}
        lessons = vis.filter_entries(lessons) if lessons else lessons

    # 6. Artifact awareness (column authoritative once backfilled; classify on
    #    the fly otherwise so demotion works immediately).
    col_ready = getattr(_async_adapter, "_has_is_artifact_col", False)

    def _is_artifact(e: MemoryEntry) -> bool:
        if col_ready:
            return bool(e.is_artifact)
        return classify_artifact(e.key, e.value)

    if not req.include_artifacts:
        candidates = {k: v for k, v in candidates.items() if not _is_artifact(v["entry"])}

    # 6a. Graded lexical term. The channels above mark a candidate 1.0 for
    #     matching *any* query word, which under an OR-combined full-text query
    #     is nearly every candidate; the blend then has no lexical signal at
    #     all. ``keyword_coverage`` reads the pool once and scores each entry by
    #     the rare query terms it carries — the service name, the error code —
    #     so the entry the query is about outranks one that shares its
    #     phrasing. "Rare" is judged against the entity's past tasks where it
    #     can be: the words every task here shares are template, the words
    #     that vary are the subject, and only the task history tells them
    #     apart (``keyword_coverage`` for the measurement). Graded over the
    #     lexical channel's own hits only: an entry the channel did not return
    #     keeps 0, as before, rather than being handed lexical relevance for
    #     an incidental word. Under the additive (rollback) form the flag
    #     stays binary, so the switch restores the old ranking exactly.
    anchored = _rank_anchored()
    lexical_hits = {k: v for k, v in candidates.items() if v["keyword"] > 0.0}
    if anchored and lexical_hits and topical.strip():
        coverage = _keyword_coverage(
            topical,
            {k: _entry_text(v["entry"].key, v["entry"].value) for k, v in lexical_hits.items()},
            background=await _task_corpus(req.entity_path),
        )
        for k, v in lexical_hits.items():
            v["keyword"] = float(coverage.get(k, 0.0))

    # 6b. Discredited entries: a failure left them under the discredit
    #     threshold and no success has lifted them since. Out of the ranked
    #     list unless asked for; kept aside so ``include_avoid`` can hand them
    #     back flagged, because "this is what stopped working" is an answer.
    #     Before the split, the local record: for the most relevant candidates,
    #     what happened on tasks like this one. An entry's pooled evidence sums
    #     every outcome it was credited with whatever the task; a rule that is
    #     right for one class of task and wrong for another reads ``contested``
    #     to both. The outcome rows carry the task embedding, so the record can
    #     be conditioned on the query — and a discredited rule that still works
    #     for tasks like this one is kept, labelled ``contested``, rather than
    #     hidden from the one class that needs it.
    #
    #     The same record, read the other way: a *validated* rule whose two
    #     most recent outcomes on tasks like this one were failures has
    #     stopped working for this class of task, whatever its pooled label
    #     says (``locally_discredited``). It leaves the ranked list for the
    #     avoid list, with what replaced it. Grid v4 measured the alternative
    #     — the rule kept its ``validated`` label and its rank through 4-7
    #     failures on the same quirk, because the pooled record was long and
    #     the failures few.
    local_evidence: dict[str, dict[str, Any]] = {}
    if _local_evidence_enabled() and topical in query_vectors and candidates:
        near_fn = getattr(_get_memory()._adapter, "evidence_near", None)
        if callable(near_fn):
            from amfs_core.actions import PRIORS_MIN_SIMILARITY as _PRIORS_MIN_SIM

            by_relevance = sorted(
                candidates.values(), key=lambda v: (v["sim"], v["keyword"]), reverse=True
            )
            head_keys = [v["entry"].entry_key for v in by_relevance[:LOCAL_EVIDENCE_HEAD]]
            try:
                # The priors' absolute floor, under the relative weighting the
                # adapter applies: an outcome on an unrelated task is not local
                # evidence however alone it is.
                local_evidence = await _offload(
                    _db_executor,
                    functools.partial(near_fn, min_similarity=_PRIORS_MIN_SIM),
                    head_keys,
                    query_vectors[topical],
                )
            except Exception:  # noqa: BLE001 - the pooled record stands
                logger.debug("evidence_near failed", exc_info=True)
                local_evidence = {}

    avoided: list[MemoryEntry] = []
    # entry_key -> (similarity, keyword) for the query-scoped shift reading.
    avoided_match: dict[str, tuple[float, float]] = {}
    rescued: set[str] = set()
    # Validated by the pooled record, failed on the last two tasks like this.
    locally_discredited: set[str] = set()
    if not req.include_discredited:
        kept_candidates: dict[str, dict[str, Any]] = {}
        for k, v in candidates.items():
            if getattr(v["entry"], "discredited_at", None) is not None:
                if _locally_valid(local_evidence.get(k)):
                    rescued.add(k)
                    kept_candidates[k] = v
                elif v["sim"] > 0.0 or v["keyword"] > 0.0:
                    avoided.append(v["entry"])
                    avoided_match[k] = (float(v["sim"]), float(v["keyword"]))
            elif _locally_discredited(local_evidence.get(k)):
                locally_discredited.add(k)
                avoided.append(v["entry"])
                avoided_match[k] = (float(v["sim"]), float(v["keyword"]))
            else:
                kept_candidates[k] = v
        candidates = kept_candidates
    else:
        # Nothing hidden, so nothing to rescue; the below-gate rows join the
        # ranked list like any other candidate.
        pass

    # 7. Blend: relevance (semantic + keyword), modulated by trust (confidence
    #    + evidence) and recency. See ``amfs_core.ranking`` for the form and
    #    the measurement behind it.
    now = _dt.now(_tz.utc)
    half_life = 30.0
    keyword_weight = 0.15
    evidence_weight = req.evidence_weight

    def _is_procedure(e: MemoryEntry) -> bool:
        mt = getattr(e, "memory_type", None)
        return str(getattr(mt, "value", mt)) == MemoryType.PROCEDURE.value

    def _composite(
        relevance: float, recency: float, conf: float, keyword: float, artifact: bool,
        evidence: float = 0.0, procedure: bool = False,
    ) -> float:
        """The composite score, in one place because step 8 recomputes it.

        *relevance* is whichever estimate of "does this answer the query" we
        currently trust: the bi-encoder similarity here, the normalised
        cross-encoder score once the reranker has spoken. Everything else is
        held constant between the two, which is the point — the reranker is a
        better relevance term, not a licence to discard confidence.

        *evidence* is the outcome record in one signed number
        (``amfs_core.evidence.evidence_signal``). Confidence already moves with
        outcomes; this term is what separates an author's untested 0.9 from a
        0.9 that has been confirmed a dozen times, and what pushes an entry
        with a mixed record below both.

        *procedure* applies ``PROCEDURE_BOOST``: at equal relevance, how to do
        the task ranks above a fact about it.
        """
        score = composite_score(
            relevance=relevance,
            recency=recency,
            confidence=conf,
            evidence=evidence,
            keyword=keyword,
            semantic_weight=req.semantic_weight,
            recency_weight=recency_weight,
            confidence_weight=req.confidence_weight,
            keyword_weight=keyword_weight,
            evidence_weight=evidence_weight,
            anchored=anchored,
        )
        if artifact:
            score *= ARTIFACT_PENALTY
        if procedure:
            score *= PROCEDURE_BOOST
        return score

    # Environment scoping: a procedure whose stated environment preconditions
    # contradict the asking run (``{"runtime": "python3.12"}`` against a
    # python3.9 run) is a way of doing the task somewhere else. It is dropped
    # from the hits and named in ``_meta.not_applicable`` so the agent knows a
    # way exists. Without an environment on the request nothing is dropped.
    environment = {k: v for k, v in (req.environment or {}).items() if v}
    not_applicable: list[dict[str, Any]] = []
    if environment:
        from amfs_core.models import preconditions_status as _preconditions_status

        kept_env: dict[Any, Any] = {}
        for ck, slot in candidates.items():
            entry = slot["entry"]
            if _is_procedure(entry):
                status, detail = _preconditions_status(entry.value, environment)
                if status == "not_applicable":
                    not_applicable.append({
                        "entity_path": entry.entity_path, "key": entry.key, "why": detail,
                    })
                    continue
            kept_env[ck] = slot
        candidates = kept_env

    scored: list[tuple[MemoryEntry, float, dict[str, Any]]] = []
    for slot in candidates.values():
        entry = slot["entry"]
        sim = float(slot["sim"])
        keyword = float(slot["keyword"])
        written = getattr(entry.provenance, "written_at", None)
        if written is not None:
            if written.tzinfo is None:
                written = written.replace(tzinfo=_tz.utc)
            age_days = max(0.0, (now - written).total_seconds() / 86400.0)
            recency = 0.5 ** (age_days / half_life)
        else:
            recency = 0.0
        conf = float(entry.confidence)
        artifact = _is_artifact(entry)
        procedure = _is_procedure(entry)
        pooled = _evidence_signal(entry)
        local = local_evidence.get(entry.entry_key)
        evidence, local_w = _blend_local_evidence(pooled, local)
        status = entry.evidence_status
        if entry.entry_key in rescued:
            # Discredited everywhere, working here: the label the pooled
            # record would give a mixed history, and the one the agent should
            # read as "check before you lean on it".
            status = "contested"
            conf = max(conf, DISCREDIT_THRESHOLD)
        # Components are kept unrounded so step 8 can rebuild the score
        # exactly; rounding happens once, on the way out.
        bd: dict[str, Any] = {
            "semantic": sim,
            "relevance": req.semantic_weight * sim + keyword_weight * keyword,
            "recency": recency,
            "confidence": conf,
            "keyword": keyword,
            "evidence": evidence,
            "evidence_status": status,
            "is_artifact": artifact,
            "is_procedure": procedure,
        }
        if local_w > 0.0 and local is not None:
            bd["evidence_local"] = {
                "success": round(float(local.get("success", 0.0)), 3),
                "failure": round(float(local.get("failure", 0.0)), 3),
                "n": int(local.get("n", 0)),
                "weight": round(local_w, 3),
            }
        scored.append((
            entry,
            _composite(sim, recency, conf, keyword, artifact, evidence, procedure),
            bd,
        ))

    scored.sort(key=lambda t: t[1], reverse=True)

    # 8. Cross-encoder rerank (Pro-injected) over the top-N: the cross-encoder
    #    replaces the *relevance term* of the composite, not the composite.
    #
    #    It used to replace the whole score, and that silently switched off
    #    continual learning. Every hosted read goes through here and real
    #    result sets are far smaller than rerank_top_n, so in practice ranking
    #    was cross-encoder relevance and nothing else: confidence, recency and
    #    the reinforcement behind them counted for zero. Measured on dev before
    #    this change — an entry discredited by eight critical_failure outcomes,
    #    confidence collapsed 0.9834 -> 0.268, still ranked first, ahead of a
    #    near-identical entry validated twelve times at confidence 1.0. Two
    #    entries with identical text, one at confidence 0.5 and one at 1.0,
    #    ranked with the 0.5 one first. Outcomes were recorded faithfully and
    #    then ignored at the only point where they could change behaviour.
    reranker = _retrieval_reranker
    rerank_top_n = 30
    if reranker is not None and getattr(reranker, "available", False) and scored:
        head = scored[:rerank_top_n]
        try:
            # Cross-encoder inference over up to thirty documents: the single
            # most expensive thing on the read path, and it ran on the event
            # loop until 2026-09-18.
            rr_scores = await _offload(
                _model_executor,
                reranker.rerank,
                topical,
                [_doc_text_for_rerank(e) for e, _, _ in head],
            )
        except Exception:  # noqa: BLE001 - rerank is best-effort
            logger.debug("rerank failed", exc_info=True)
            rr_scores = None
        if rr_scores and len(rr_scores) == len(head):
            raw = [float(rs) for rs in rr_scores]
            normalised = _normalise_rerank(raw)
            # Standing within the batch decides the ranking; the model's own
            # opinion of the candidate decides whether step 9 may discard it. Two
            # questions, two numbers, computed here where the batch is in hand.
            absolute = _rerank_absolute(raw)
            reranked = [
                (
                    entry,
                    _composite(
                        norm, bd["recency"], bd["confidence"], bd["keyword"],
                        bd["is_artifact"], bd.get("evidence", 0.0),
                        bd.get("is_procedure", False),
                    ),
                    {**bd, "rerank": rs, "rerank_normalised": norm,
                     "rerank_absolute": absolute_score,
                     "relevance": req.semantic_weight * norm + keyword_weight * bd["keyword"]},
                )
                for (entry, _, bd), rs, norm, absolute_score in zip(
                    head, raw, normalised, absolute
                )
            ]
            # The whole list, not head-then-tail: re-scoring the head can move a
            # member of it below an entry the reranker never saw, and stitching
            # the two halves back together in order would pin it above anyway.
            scored = reranked + scored[rerank_top_n:]
            scored.sort(key=lambda t: t[1], reverse=True)

    # 9. Abstain floor: trim clearly-irrelevant tail (low relevance AND no
    #    keyword match), but never drop the single best result.
    #
    #    An entry the cross-encoder positively endorses is exempt, because this
    #    step was otherwise undoing step 8. While step 8 *replaced* the score with
    #    the rerank, the cross-encoder's favourite was rank one by construction,
    #    and rank one is kept unconditionally. Now that the rerank only sets the
    #    relevance term, confidence and recency can put that favourite second —
    #    and an entry the reranker rescued is precisely the one whose bi-encoder
    #    score is low, since rescuing those is what a reranker is for. So judging
    #    on ``semantic`` alone deletes the judgement the reranker was added to
    #    make, and can delete an entry reinforcement had promoted.
    #
    #    Endorsement is the *absolute* score and not the normalised one, which is
    #    the distinction that earns the two fields. The normalised value reports
    #    standing within the batch, so its best member scores high however poor
    #    the batch is: measured, a candidate at logit -3.0 — a 4.7% chance of
    #    relevance by the model's own reckoning — normalises to 0.978 against
    #    peers at -9. Exempting on that would keep junk this floor exists to trim,
    #    and would do it hardest in the case abstention is for, where nothing in
    #    the batch is any good.
    floor = _retrieve_min_semantic()
    if floor > 0 and len(scored) > 1:
        kept = [scored[0]]
        for entry, score, bd in scored[1:]:
            endorsed = float(bd.get("rerank_absolute") or 0.0) >= RERANK_ENDORSED
            if (
                float(bd.get("semantic") or 0.0) < floor
                and not bd.get("keyword")
                and not endorsed
            ):
                continue
            kept.append((entry, score, bd))
        scored = kept

    # 10. Reuse accounting. Semantic retrieval is the primary way memories
    #     (e.g. browser-extension clips) get surfaced and used, but it
    #     historically incremented no counter — so value metrics (rework
    #     avoided / memories reused) read 0 even when memory was clearly reused.
    #     Bump recall_count for the TOP hit only. Reuse should count what the
    #     agent took, not what it was shown: the rest of a ranked list is a
    #     candidate set, and crediting the top three booked reuse that never
    #     happened (a real session: 4 lookups, 16 credited reuses, 1 that
    #     changed the agent's behavior). Best-effort: never let accounting
    #     failure affect the response.
    credited_entry = None
    credited_hits = 0
    for entry, _score, _bd in scored[:REUSE_CREDIT_K]:
        credited_hits += 1
        if credited_entry is None:
            credited_entry = entry
        try:
            if _async_adapter is not None:
                await _async_adapter.increment_recall_count(
                    entry.entity_path, entry.key, branch=branch
                )
            else:
                _get_memory()._adapter.increment_recall_count(
                    entry.entity_path, entry.key, branch=branch
                )
        except Exception:  # noqa: BLE001 - reuse accounting is best-effort
            logger.debug("retrieve recall bump failed", exc_info=True)
    _attach_reuse_value(
        response,
        request,
        credited=credited_entry,
        hits=credited_hits,
        surface="retrieve",
        branch=branch,
    )

    head = scored[: req.limit]
    # 11. Evidence-aware k. When the record has confirmed the top hit and it is
    #     clearly ahead, the alternatives are noise in the prompt: keep the ones
    #     within reach of it and drop the rest. Never below one result.
    if req.adaptive_k and head and head[0][0].evidence_status == "validated":
        top_score = head[0][1]
        if anchored:
            # Under the anchored blend the score is relevance scaled by trust,
            # so "within reach" is read on the two terms it is made of: a
            # validated peer stays if its score is close; an entry the record
            # has not confirmed stays if it is *more* relevant than the leader,
            # or carries a rare query term the leader lacks (its lexical
            # coverage ahead by ADAPTIVE_K_KEYWORD_GAP) — the cases where the
            # leader's record, not its topic, put it first, and the agent
            # should still see what the query was actually about. A rescued
            # peer — discredited by the pooled record, confirmed on tasks
            # like this one — is read as validated here: the record that
            # kept it is the local one.
            top_rel = float(head[0][2].get("relevance") or 0.0)
            top_kw = float(head[0][2].get("keyword") or 0.0)
            head = [head[0]] + [
                t for t in head[1:]
                if (
                    (t[0].evidence_status == "validated" or t[0].entry_key in rescued)
                    and t[1] >= top_score * ADAPTIVE_K_KEEP_RATIO
                )
                or float(t[2].get("relevance") or 0.0) > top_rel
                or float(t[2].get("keyword") or 0.0) >= top_kw + ADAPTIVE_K_KEYWORD_GAP
            ]
        else:
            head = [t for t in head if t[1] >= top_score * ADAPTIVE_K_KEEP_RATIO] or head[:1]

    # 12. Action priors and a recommendation, as one trailing element the client
    #     asked for. What the entries cannot say — "tried here and failed",
    #     "nobody has tried X" — comes from the outcome record, not from memory.
    #     Computed before the hits are rendered: the query-scoped shift it
    #     reads also tightens the hit list (11b), and the contrast it finds
    #     names what to do instead of an avoided entry.
    meta: dict[str, Any] | None = None
    priors: dict[str, Any] | None = None
    shifted_local = False
    if req.include_priors and req.entity_path:
        from amfs_core.actions import recommend as _recommend

        priors_text = req.situation or topical
        priors_vec = query_vectors.get(priors_text)
        if priors_vec is None and embedder is not None and callable(getattr(embedder, "embed", None)):
            try:
                priors_vec = await _offload(_model_executor, embedder.embed, priors_text[:2000])
            except Exception:  # noqa: BLE001
                priors_vec = None
        priors = await _offload(
            _db_executor,
            _priors_for_retrieve,
            entity_path=req.entity_path,
            text=priors_text,
            embedder=embedder,
            candidate_actions=req.candidate_actions,
            environment=environment,
            query_vector=priors_vec,
            situation=req.situation,
        )
        top = head[0][0] if head else None
        top_bd = head[0][2] if head else {}
        # A rescued top hit is discredited by the pooled record and working
        # on tasks like this one. The recommendation reads the local verdict
        # — the status step 7 rendered ("contested"), no recent failure, no
        # shift — or the rescue would be undone here by an ``explore`` for
        # exactly the class of task the rule still works on.
        top_rescued = bool(top is not None and top.entry_key in rescued)
        recent_failure = bool(
            top is not None
            and not top_rescued
            and top.last_outcome is not None
            and not _evidence_is_success(top.last_outcome)
        )
        # A long-validated rule whose record no longer supports acting on it is
        # the retrieve-time reading of a regime shift; the briefing's section of
        # the same name applies the same predicate over the whole entity, this
        # over the hits. Read over the head *and* the discredited entries kept
        # aside: a rule that was validated eight times and then discredited by
        # two failures has left the ranking, and it is the clearest case there
        # is. The first-strike case — one failure, still ``validated`` — is not
        # a shift, or the recommendation would skip a winning action on the
        # same failure the label forgives.
        #
        # The candidate fetch honours min_confidence, and a discredited rule sits
        # below the discredit threshold by definition — so with the gate at or
        # above it (the benchmark's setting) the rows this reads never arrived.
        # Fetched here without the gate, for this reading only: the ranked list
        # is unchanged.
        # Always, not only when a confidence gate is set: the read is
        # entity-wide, so it also brings in the rule that stopped working but
        # shares no words with this query — which the ranked list never held
        # whatever the gate.
        #
        # Two readings, two uses. The *entity-wide* one — head, avoided, and
        # every discredited entry on the entity — is reported in ``_meta`` for
        # the briefing-style "something on this entity changed" signal. The
        # *query-scoped* one steers the recommendation: only the entries this
        # query is about (a similarity within LOCAL_SIM_GAP of the best hit,
        # or a keyword match), so a rule that stopped working for one class of
        # task does not send every other class to explore past memory that is
        # still right for it. Grid v3 measured that mistake at 14% success on
        # the explores it produced.
        shift_pool: list[MemoryEntry] = [e for e, _, _ in head] + list(avoided)
        seen_keys = {e.entry_key for e in shift_pool}
        shift_pool.extend(
            await _discredited_below_gate(
                req.entity_path,
                branch=branch,
                seen=seen_keys,
                vis=vis,
                include_artifacts=req.include_artifacts,
                limit=pool,
            )
        )
        shifted_entries = [e for e in shift_pool if _regime_shifted(e, now=now)]
        shifted = bool(shifted_entries)

        def _shift_at(entries: list[MemoryEntry]) -> datetime | None:
            return max(
                (
                    at if at.tzinfo else at.replace(tzinfo=UTC)
                    for at in (e.last_outcome_at for e in entries)
                    if at is not None
                ),
                default=None,
            )

        best_sim = max((float(bd.get("semantic") or 0.0) for _, _, bd in head), default=0.0)
        local_floor = max(0.0, best_sim - LOCAL_SIM_GAP)

        def _about_this_query(entry: MemoryEntry, sim: float, keyword: float) -> bool:
            return keyword > 0.0 or (sim > 0.0 and sim >= local_floor)

        local_pool: list[MemoryEntry] = [
            e for e, _, bd in head
            if _about_this_query(e, float(bd.get("semantic") or 0.0), float(bd.get("keyword") or 0.0))
        ]
        local_pool.extend(
            e for e in avoided
            if _about_this_query(e, *avoided_match.get(e.entry_key, (0.0, 0.0)))
        )
        shifted_local_entries = [
            e for e in local_pool
            if e.entry_key not in rescued and _regime_shifted(e, now=now)
        ]
        shifted_local = bool(shifted_local_entries)
        top_shifted = bool(
            top is not None and not top_rescued and _regime_shifted(top, now=now)
        )
        hit_statuses = [str(e.evidence_status) for e, _, _ in head if e.evidence_status]
        # The situation-exact record: the lessons among the hits that are
        # about this task (their situation compared to the declared one, else
        # found in the task text). They re-order the plan — priors are pooled
        # over a neighbourhood that can hold two classes with opposite
        # answers; a lesson names one class.
        from amfs_core.lessons import applicable_claims as _applicable_claims
        from amfs_core.lessons import lesson_of as _lesson_of

        # Not the avoided ones: an avoided lesson is one retrieve hid because
        # it failed on this kind of task, and its status need not say
        # ``discredited`` yet — a locally falsified "worked" claim must not
        # promote the action it was falsified on.
        avoided_keys = {e.entry_key for e in avoided}
        lesson_rows = []
        for e in local_pool:
            if e.entry_key in avoided_keys:
                continue
            lesson = _lesson_of(e.value)
            if lesson is not None:
                lesson_rows.append(dict(lesson, evidence_status=e.evidence_status))
        claims = _applicable_claims(lesson_rows, req.query, declared=req.situation)
        recommendation = _recommend(
            priors,
            agent_id=req.agent_id or "",
            candidate_actions=req.candidate_actions,
            lessons=claims or None,
            top_hit_status=(
                str(top_bd.get("evidence_status") or top.evidence_status)
                if top is not None else None
            ),
            top_hit_recent_failure=recent_failure,
            top_hit_shifted=top_shifted,
            regime_shift=shifted_local,
            regime_shift_at=_shift_at(shifted_local_entries),
            abstain=req.abstain,
            hit_statuses=hit_statuses,
            # ``action_stats`` is every outcome on the entity, no similarity:
            # a record that can name a winner but is not about this kind of
            # task, so a shift read over it does not send the agent exploring.
            priors_are_local=(priors or {}).get("source") != "action_stats",
        )
        if (
            priors is not None or recommendation is not None or shifted
            or req.abstain or not_applicable
        ):
            from amfs_core.actions import guidance_strength as _guidance_strength

            meta = {
                "_meta": True,
                "priors": priors,
                "recommendation": recommendation,
                "regime_shift": shifted,
                "regime_shift_scope": (
                    "query" if shifted_local else ("entity" if shifted else None)
                ),
                # Rated over the query-scoped shift, like the recommendation:
                # a rule that stopped working for another class of task does
                # not thin the guidance for this one.
                "guidance_strength": _guidance_strength(
                    priors, hit_statuses, regime_shift=shifted_local
                ),
            }
            if not_applicable:
                meta["not_applicable"] = not_applicable
    if meta is None and not_applicable:
        # No priors asked for, but the environment dropped a procedure: say so
        # in the same trailing element, so the client learns a way exists.
        meta = {"_meta": True, "not_applicable": not_applicable}

    # 11b. Under a query-scoped shift — a rule the query is about has stopped
    #      working — the alternatives to the leader that the record has
    #      already marked against on tasks like this are noise with a cost:
    #      grid v4 found 5.4 discredited-or-contested rows in the context of
    #      failing episodes against 3.0 in successes. Drop the non-leading
    #      hits whose status is ``contested`` or whose local record on this
    #      kind of task is more failure than success. Never below one hit,
    #      and the leader itself is never dropped here: what to do about the
    #      leader is the recommendation's call. A rescued hit is never weak:
    #      it carries the ``contested`` label too, but it is in the list
    #      because its local record says it works here — the opposite of
    #      what the label means on a pooled record.
    if req.adaptive_k and shifted_local and len(head) > 1:
        def _weak_under_shift(t: tuple[MemoryEntry, float, dict[str, Any]]) -> bool:
            if t[0].entry_key in rescued:
                return False
            bd = t[2]
            if str(bd.get("evidence_status") or "") == "contested":
                return True
            local = bd.get("evidence_local") or {}
            return float(local.get("failure") or 0.0) > float(local.get("success") or 0.0)

        head = [head[0]] + [t for t in head[1:] if not _weak_under_shift(t)]

    # 13. Render. Hits first, then the avoid list, then the trailing meta
    #     element (clients lift it out by its ``_meta`` flag).
    render = _compact_entry_response if req.compact else _entry_to_response
    out: list[dict[str, Any]] = []
    for rank, (entry, score, breakdown) in enumerate(head):
        data = _compact_entry_response(entry, rank=rank) if req.compact else render(entry)
        data["_score"] = round(score, 4)
        if entry.entry_key in rescued:
            # The label the ranking used, not the pooled one the entry carries:
            # to this query the rule is contested, not discredited, and the
            # agent reads the top-level field.
            data["evidence_status"] = breakdown.get("evidence_status")
            data["_rescued"] = True
        if req.compact:
            data["_breakdown"] = {"evidence_status": breakdown.get("evidence_status")}
        else:
            data["_breakdown"] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in breakdown.items()
            }
        out.append(data)
    if req.include_avoid and avoided:
        avoided.sort(key=lambda e: (e.last_outcome_at or e.provenance.written_at), reverse=True)
        # What replaced each avoided entry, per entry: the entries a contrast
        # lesson on this query says the task was resolved with (same reading
        # as the briefing's ``discredited[].replaced_by``), and the action
        # that resolved it — the lesson's own ``resolved_action`` where it
        # carries one, else the priors contrast for the same outcome. A
        # contrast the lessons do not tie to this entry says nothing about
        # it: two avoided rules must not both claim the one fix.
        replaced_by = _replacements_from_lessons(lessons) if lessons else {}
        resolved_action = _resolved_actions_from_lessons(
            lessons, (priors or {}).get("contrasts") or []
        )
        for entry in avoided[:AVOID_LIST_MAX]:
            data = (
                _compact_entry_response(entry, rank=_COMPACT_FULL_HITS) if req.compact
                else render(entry)
            )
            data["_score"] = 0.0
            data["_avoid"] = True
            local_only = entry.entry_key in locally_discredited
            replacements = list(replaced_by.get(entry.entry_key) or [])
            action = resolved_action.get(entry.entry_key)
            data["_breakdown"] = {
                "evidence": -1.0,
                "evidence_status": "discredited",
                "failure_count": entry.failure_count,
                "last_outcome": entry.last_outcome,
                "locally_discredited": local_only,
                "replaced_by": replacements,
                "resolved_with_action": action,
            }
            if req.compact:
                # An avoid row's work is to name what stopped working and what
                # replaced it, not to restate the rule: the value it carried
                # was, in grid v4, the text failing agents were still acting
                # on. Compact mode already cut it to a preview; a one-liner
                # that says so is the whole message. The field stays a string
                # so every client parses the row as an entry.
                #
                # The date is the last *failure*: for a locally discredited
                # rule the newest nearby outcome (a failure by construction —
                # the entry's own ``last_outcome_at`` may be a later success
                # on another class of task); otherwise the entry's last
                # outcome when that was a failure, else nothing.
                when = None
                if local_only:
                    recent = (local_evidence.get(entry.entry_key) or {}).get("recent") or []
                    when = _as_utc(recent[0].get("committed_at")) if recent else None
                elif entry.last_outcome is not None and not _evidence_is_success(entry.last_outcome):
                    when = entry.last_outcome_at
                parts = [
                    "stopped working on tasks like this" if local_only else "discredited",
                    f"last failure {when.date().isoformat()}" if when else "",
                ]
                if replacements:
                    parts.append("replaced by " + ", ".join(
                        r.rsplit("/", 1)[-1] for r in replacements[:3]
                    ))
                if action:
                    parts.append(f"resolved instead with {action}")
                data["value"] = "; ".join(p for p in parts if p)
                data.pop("value_truncated", None)
            out.append(data)
    if meta is not None:
        out.append(meta)
    return out


# ──────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/stats")
async def get_stats(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    scope, py_vis = _visibility_scope(request)

    def _stats() -> dict[str, Any]:
        """Off the event loop. The scoped shapes must be a SUPERSET of the
        MemoryStats shape, because the client parses /stats via
        MemoryStats.model_validate — any missing field silently defaults
        (confidence→0.0, outcome→0) and any mis-named key (e.g.
        "oldest_entry" vs "oldest_entry_at") is dropped to None. Both scoped
        routes below produce the same keys as the unscoped aggregate."""
        if py_vis is None:
            try:
                # The visibility rule as a predicate in the aggregate: a
                # per-user dashboard gets the same query an admin does. Only
                # passed when there is one; the ABC default takes no scope.
                if scope is None:
                    return mem._adapter.stats_extended()
                return mem._adapter.stats_extended(scope=scope)
            except TypeError:
                pass  # an adapter without SQL scoping; reduce over its rows
        # A filter that can only run over loaded entries: load, filter,
        # aggregate with the same shared helper the adapter defaults use.
        from amfs_core.aggregates import extended_stats_from_entries

        entries = mem.list()
        vis = py_vis if py_vis is not None else _active_visibility_filter(request)
        if vis is not None:
            entries = vis.filter_entries(entries)
        return extended_stats_from_entries(entries)

    stats = await _offload(_db_executor, _stats)
    return json.loads(json.dumps(stats, default=str))


# ──────────────────────────────────────────────────────────────────────
# Integrity verification
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/verify")
def verify_integrity(
    request: Request,
    body: dict[str, Any] = {},
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    entity_path = body.get("entity_path")
    # Integrity verification walks the full hash chain, which necessarily
    # spans entries the caller may not be allowed to see. Restrict it to
    # account admins / unrestricted callers.
    if _active_visibility_filter(request) is not None:
        raise HTTPException(
            status_code=403,
            detail="Integrity verification requires account admin access",
        )
    return mem.verify(entity_path)


# ──────────────────────────────────────────────────────────────────────
# Atomic commits
# ──────────────────────────────────────────────────────────────────────


#: Per-write options this endpoint forwards to the transaction.
#:
#: An allow-list, so a key a caller invents cannot reach ``tx.write`` as an
#: unexpected keyword and turn a batch into a 500.
_COMMIT_WRITE_OPTIONS = ("confidence", "memory_type", "pattern_refs", "shared")


@app.post("/api/v1/commits")
def create_commit(
    body: dict[str, Any],
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run a transaction here, rather than assembling one on the client.

    A client-side transaction over HTTP wrote each entry with its own request
    and then called ``save_commit``, which the HTTP adapter has never
    implemented — there is no endpoint that accepts a commit somebody else
    minted. So the id came back to the caller and the commit itself was
    dropped, and nothing tied the entries together afterwards.

    Doing it here is what makes the group a single round trip that either
    reaches the server or does not, instead of n requests from a client that
    can half-succeed and no way for the caller to find out which half.

    It is not yet one database transaction, and the difference matters. The
    Postgres adapter does not override ``write_batch``, so it inherits the
    default that writes each entry on its own connection, and ``save_commit``
    runs on a further one after those have committed. A failure part-way
    leaves the earlier entries written. What is fixed here is the client-side
    fan-out; what remains is the storage-side one.

    The per-write options are the reason this was not simply switched over
    earlier. This endpoint used to forward only path, key and value, so moving
    a batch here would have silently discarded confidence, memory type,
    pattern refs and sharing on every entry — writing the right entries with
    the wrong metadata, which is worse than not writing them.
    """
    from amfs_core.models import MemoryType

    mem = _get_memory()
    writes = body.get("writes", [])
    message = body.get("message", "")
    if not writes:
        # Refused rather than accepted as a no-op, because the two are
        # indistinguishable to the caller otherwise. An empty batch never
        # reaches flush, so the response carries no commit — which is the same
        # response a server too old to mint one sends back. The client reads
        # that as "committed, details unavailable" and moves on, when in fact
        # nothing was written at all.
        raise HTTPException(
            status_code=422, detail="writes is empty — nothing to commit."
        )

    # Whose writes these are. Running the transaction here moved the work off
    # the client and took the caller's name off it with it: everything went in
    # as this process's own agent, because that is who _get_memory() is. The
    # entries were credited to the server and so was the commit — and since
    # GET /api/v1/commits hides commits whose author the caller cannot see, the
    # caller could not then see the commit they had just made. amfs_commit_log
    # answered zero for an account with commits sitting in it.
    #
    # Swapping the tagger is how POST /api/v1/entries has always done this.
    # AgentMemory.agent_id reads through to the tagger, so one swap covers both
    # the entries' provenance and the commit's author.
    #
    # The header is a fallback for a client that does not send the field yet:
    # the MCP gateway has always set X-AMFS-Agent-Id on every request, so
    # reading it means a gateway already in production is attributed correctly
    # from the moment this deploys. The body wins when both are present.
    agent_id = body.get("agent_id") or request.headers.get("x-amfs-agent-id")
    session_id = body.get("session_id")

    # A per-request handle carries the caller's identity (``AgentMemory.as_agent``).
    # Swapping the shared tagger and restoring it in a ``finally`` was atomic
    # only while this body ran on the event loop; on the threadpool two commits
    # would stamp each other's writes.
    handle = mem
    if agent_id or session_id:
        handle = mem.as_agent(agent_id or mem.agent_id)
        if session_id:
            handle._tagger.session_id = session_id

    if agent_id:
        try:
            # The gateway ensures its agent when the session opens, so this
            # is normally a no-op. It is here for a client committing as an
            # agent this server has not seen, where the entry insert would
            # otherwise fail on a name nothing has registered.
            mem._adapter.ensure_agent(agent_id, mem.namespace)
        except Exception:  # noqa: BLE001
            logger.warning(
                "ensure_agent failed for %s — committing anyway", agent_id,
                exc_info=True,
            )
        _link_agent_owner_once(request, agent_id, mem.namespace)

    with handle.transaction(message) as tx:
        for w in writes:
            options = {k: w[k] for k in _COMMIT_WRITE_OPTIONS if k in w}
            # Arrives over the wire as a string, and the write path wants
            # the enum. An unknown one is refused rather than dropped:
            # writing the entry with a default type would put the wrong
            # metadata on the right memory, and nothing later can tell that
            # happened.
            if "memory_type" in options:
                raw = str(options["memory_type"])
                try:
                    options["memory_type"] = MemoryType(raw)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Invalid memory_type {raw!r}. Valid: "
                            + ", ".join(m.value for m in MemoryType)
                        ),
                    ) from exc
            tx.write(w["entity_path"], w["key"], w.get("value"), **options)

    commit = tx.commit
    return {
        "commit_id": commit.id if commit else None,
        "message": message,
        "entries_written": len(tx.entries),
        # The whole commit, so a caller does not have to fetch what was just
        # created to learn the versions its entries landed on.
        "commit": json.loads(json.dumps(commit.model_dump(mode="json"), default=str))
        if commit else None,
    }


@app.get("/api/v1/commits")
def list_commits(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    branch: str | None = Query(None),
    namespace: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Commits, newest first, filtered before the limit is applied.

    ``branch`` and ``namespace`` are new, and the reason is a real bug rather
    than completeness. Without them the caller can only take the newest N
    account-wide and filter what it wanted out of the page it was given, so a
    page full of another branch's commits comes back empty even though the
    branch it asked about has plenty.

    That is not hypothetical: ``TransactionBuffer.flush`` asks for exactly one
    commit to use as the new commit's parent. One commit on the wrong branch
    means no parent, which means every commit over HTTP is a root, no two
    commits share an ancestor, and ``common_ancestor`` answers "none" forever —
    while looking exactly like an honest answer about unrelated history.

    Both default to None rather than to "main" and "default", so a caller that
    does not pass them keeps the behaviour it had: whatever branch and
    namespace this server's own memory is configured for.
    """
    mem = _get_memory()
    if branch is None and namespace is None:
        commits = mem.commit_log(limit=limit)
    else:
        # Straight to the adapter, because commit_log deliberately takes
        # neither — it reads the server's own branch and namespace, which are
        # not the caller's to begin with. The route already reaches for
        # _adapter for stats_extended.
        commits = mem._adapter.list_commits(
            branch=branch or mem._branch,
            limit=limit,
            namespace=namespace or mem._config.namespace,
        )
    allowed = _visible_agent_ids(request)
    if allowed is not None:
        commits = [c for c in commits if c.author_agent_id in allowed]
    return {
        "commits": [json.loads(json.dumps(c.model_dump(mode="json"), default=str)) for c in commits],
        "count": len(commits),
    }


@app.get("/api/v1/commits/{commit_id}")
def get_commit(
    request: Request,
    commit_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    commit = mem.get_commit(commit_id)
    allowed = _visible_agent_ids(request)
    if commit is not None and allowed is not None and commit.author_agent_id not in allowed:
        commit = None
    if commit is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Commit not found")
    return json.loads(json.dumps(commit.model_dump(mode="json"), default=str))


@app.post("/api/v1/merge-base")
def merge_base(
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    ancestor = mem.common_ancestor(body["commit_a"], body["commit_b"])
    return {
        "ancestor_commit_id": ancestor,
        "commit_a": body["commit_a"],
        "commit_b": body["commit_b"],
    }


# ──────────────────────────────────────────────────────────────────────
# Agent binding
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents/{agent_id:path}/profile")
async def get_agent_profile(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return an agent's profile, stats, and registration info.

    When no explicit profile has been registered, synthesises one from the
    agent's actual activity — entity paths become auto-inferred capabilities
    and memory-type distribution is included.
    """
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()

    def _profile_reads() -> tuple[Any, list[MemoryEntry], int, int]:
        """Every synchronous read this page needs, off the event loop."""
        agent = mem._adapter.get_agent(agent_id, namespace=mem.namespace)
        entries = [
            e for e in _entries_by_agent(mem, agent_id)
            if not e.entity_path.startswith("_system/")
        ]
        # Both numbers are aggregates the adapter computes where the traces
        # live; this route used to pull up to 10,000 full traces to count them.
        trace_count = mem._adapter.count_traces(agent_id=agent_id)
        total_reads = sum(
            sum(keys.values()) for keys in mem._adapter.trace_read_counts(agent_id).values()
        )
        return agent, entries, trace_count, total_reads

    agent, entries, trace_count, total_reads = await _offload(_db_executor, _profile_reads)
    entities_touched = {e.entity_path for e in entries}
    last_active = max(
        (e.provenance.written_at for e in entries),
        default=None,
    )

    result: dict[str, Any] = {
        "agentId": agent_id,
        "entriesWritten": len(entries),
        "entitiesTouched": len(entities_touched),
        "entityPaths": sorted(entities_touched),
        "totalReads": total_reads,
        "decisionTraces": trace_count,
        "lastActive": last_active.isoformat() if last_active else None,
    }

    if agent:
        profile = agent.profile
        capabilities = agent.capabilities
        contracts = agent.contracts

        if not profile and not capabilities and entries:
            profile, capabilities = _synthesize_agent_profile(
                agent_id, entries, trace_count,
            )

        result["profile"] = profile.model_dump() if profile else {}
        result["capabilities"] = [c.model_dump() for c in capabilities]
        result["contracts"] = [c.model_dump() for c in contracts]
        result["displayName"] = agent.display_name
        result["createdAt"] = agent.created_at.isoformat() if agent.created_at else None
    return result


def _synthesize_agent_profile(
    agent_id: str,
    entries: list,
    traces: list | int,
) -> tuple:
    """Build an AgentProfile and capabilities from observed activity."""
    from amfs_core.models import AgentProfile, AgentCapability, MemoryType
    from collections import Counter

    entities_touched = {e.entity_path for e in entries}
    memory_types: Counter[str] = Counter()
    key_prefixes: Counter[str] = Counter()
    for e in entries:
        mt = e.memory_type if hasattr(e, "memory_type") and e.memory_type else MemoryType.FACT
        memory_types[mt.value if hasattr(mt, "value") else str(mt)] += 1
        prefix = e.key.split("-")[0] if "-" in e.key else e.key
        key_prefixes[prefix] += 1

    top_prefixes = [p for p, _ in key_prefixes.most_common(5)]
    type_summary = ", ".join(
        f"{count} {mtype}" for mtype, count in memory_types.most_common()
    )
    desc_parts = []
    if type_summary:
        desc_parts.append(f"Writes: {type_summary}.")
    if entities_touched:
        desc_parts.append(
            f"Active across {len(entities_touched)} "
            f"entit{'y' if len(entities_touched) == 1 else 'ies'}."
        )
    # Callers pass the count; a list is still accepted for older call sites.
    trace_count = traces if isinstance(traces, int) else len(traces)
    if trace_count:
        desc_parts.append(f"{trace_count} decision trace(s) recorded.")

    profile = AgentProfile(
        description=" ".join(desc_parts),
        auto_context_paths=sorted(entities_touched)[:10],
        tags=_infer_tags(agent_id, top_prefixes, memory_types),
    )

    capabilities = []
    entity_groups: dict[str, list[str]] = {}
    for ep in sorted(entities_touched):
        group = ep.split("/")[0] if "/" in ep else ep
        entity_groups.setdefault(group, []).append(ep)

    for group, paths in entity_groups.items():
        capabilities.append(AgentCapability(
            name=group,
            description=f"Works on {len(paths)} entit{'y' if len(paths) == 1 else 'ies'} under {group}/",
            entity_paths=paths[:10],
        ))

    return profile, capabilities


def _infer_tags(
    agent_id: str,
    key_prefixes: list[str],
    memory_types: "Counter",
) -> list[str]:
    """Derive a small set of tags from the agent's activity."""
    tags: list[str] = []
    prefix_tag_map = {
        "task": "task-executor",
        "pattern": "pattern-detector",
        "risk": "risk-assessor",
        "decision": "decision-maker",
        "action": "action-logger",
    }
    for prefix in key_prefixes:
        if prefix in prefix_tag_map and prefix_tag_map[prefix] not in tags:
            tags.append(prefix_tag_map[prefix])

    if memory_types.get("belief", 0) > memory_types.get("fact", 0):
        tags.append("hypothesis-driven")
    if memory_types.get("experience", 0) > 0:
        tags.append("experiential")

    return tags[:5]


@app.put("/api/v1/agents/{agent_id:path}/profile")
def update_agent_profile(
    request: Request,
    agent_id: str,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from amfs_core.models import AgentProfile

    mem = _get_memory()
    profile = AgentProfile.model_validate(body)
    # Link the agent to the API key's owner so it shows on the user's dashboard
    # immediately — set_identity announces through this endpoint before the
    # agent has written any memory, so the write-path owner link never fires.
    # Runs BEFORE the profile update: a colliding identity (owned by another
    # user in the account) must 409 here without touching the agent record.
    _link_agent_owner_once(request, agent_id, mem.namespace)
    agent = mem._adapter.update_agent_profile(agent_id, profile)
    return json.loads(json.dumps(agent.model_dump(mode="json"), default=str))


@app.put("/api/v1/agents/{agent_id:path}/capabilities")
def update_agent_capabilities(
    request: Request,
    agent_id: str,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from amfs_core.models import AgentCapability

    mem = _get_memory()
    _link_agent_owner_once(request, agent_id, mem.namespace)
    capabilities = [AgentCapability.model_validate(c) for c in body.get("capabilities", [])]
    agent = mem._adapter.update_agent_capabilities(agent_id, capabilities)
    return json.loads(json.dumps(agent.model_dump(mode="json"), default=str))


@app.put("/api/v1/agents/{agent_id:path}/contracts")
def update_agent_contracts(
    request: Request,
    agent_id: str,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    from amfs_core.models import MemoryContract

    mem = _get_memory()
    _link_agent_owner_once(request, agent_id, mem.namespace)
    contracts = [MemoryContract.model_validate(c) for c in body.get("contracts", [])]
    agent = mem._adapter.update_agent_contracts(agent_id, contracts)
    return json.loads(json.dumps(agent.model_dump(mode="json"), default=str))


@app.get("/api/v1/agents/discover")
def discover_agents(
    request: Request,
    capability: str | None = Query(None),
    entity_path: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    agents = mem.discover_agents(capability=capability, entity_path=entity_path)

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        agents = [a for a in agents if vis.is_agent_visible(a.agent_id)]

    return {
        "agents": [json.loads(json.dumps(a.model_dump(mode="json"), default=str)) for a in agents],
        "count": len(agents),
    }


# ──────────────────────────────────────────────────────────────────────
# Diff & patch
# ──────────────────────────────────────────────────────────────────────


def _require_entry_visible(request: Request, entity_path: str, key: str) -> None:
    """404 when the current entry exists but is hidden from the caller."""
    vis = _active_visibility_filter(request)
    if vis is None:
        return
    entry = _get_memory()._adapter.read(entity_path, key)
    if entry is not None and not vis.is_entry_visible(entry):
        raise HTTPException(status_code=404, detail="Entry not found")


@app.post("/api/v1/diff")
def compute_diff(
    request: Request,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    _require_entry_visible(request, body["entity_path"], body["key"])
    return mem.diff(
        body["entity_path"],
        body["key"],
        body.get("old_version"),
    )


@app.post("/api/v1/patches")
def create_patch(
    request: Request,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    _require_entry_visible(request, body["entity_path"], body["key"])
    return mem.create_patch(
        body["entity_path"],
        body["key"],
        body.get("source_version"),
    )


# ──────────────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────────────


async def _history_payload(
    request: Request,
    entity_path: str,
    key: str,
    since: str | None,
    until: str | None,
) -> dict[str, Any]:
    mem = _get_memory()
    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None

    versions = mem.history(entity_path, key, since=since_dt, until=until_dt)
    vis = _active_visibility_filter(request)
    if vis is not None:
        versions = vis.filter_entries(versions)
    return {
        "entity_path": entity_path,
        "key": key,
        "version_count": len(versions),
        "versions": [_entry_to_response(e) for e in versions],
    }


@app.get("/api/v1/history")
async def get_history_by_query(
    request: Request,
    entity_path: str = Query(...),
    key: str = Query(...),
    since: str | None = Query(None),
    until: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """History with the coordinates where they cannot be confused.

    The same greedy-path problem ``/api/v1/entry`` was added for: a key
    containing a slash cannot be expressed as a path segment, so a history
    lookup for one silently reported no versions. Reading an entry back was
    fixed first because it was the louder failure; this is the same defect on
    the same coordinates, and the version chain is what the lineage panel is
    built on.
    """
    return await _history_payload(request, entity_path, key, since, until)


@app.get("/api/v1/history/{entity_path:path}/{key}")
async def get_history(
    request: Request,
    entity_path: str,
    key: str,
    since: str | None = Query(None),
    until: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Unchanged, and unambiguous whenever the key has no slash.

    Kept as it is for the callers already on it, exactly as the entries route
    was: rewriting every one of them to gain nothing on the common case is the
    worse trade.
    """
    return await _history_payload(request, entity_path, key, since, until)


# ──────────────────────────────────────────────────────────────────────
# Outcomes
# ──────────────────────────────────────────────────────────────────────

_OUTCOME_TYPE_MAP = {
    "success": OutcomeType.SUCCESS,
    "minor_failure": OutcomeType.MINOR_FAILURE,
    "failure": OutcomeType.FAILURE,
    "critical_failure": OutcomeType.CRITICAL_FAILURE,
    # Legacy values (backward compatible)
    "clean_deploy": OutcomeType.CLEAN_DEPLOY,
    "regression": OutcomeType.REGRESSION,
    "p2_incident": OutcomeType.P2_INCIDENT,
    "p1_incident": OutcomeType.P1_INCIDENT,
}


# The trace store holds one psycopg connection for the life of the process
# and a psycopg connection is not safe for concurrent use. Every seal, from
# whichever thread, goes through this lock; it also guards the sequence
# counter below and the reconnect.
_seal_lock = threading.RLock()


def _store_connection_closed(store: Any) -> bool:
    """Whether the store's connection is known to be gone.

    Cloud SQL closed the store's connection under load on 2026-09-18 (its
    idle timeout, a failover, or the ``client_connection_check_interval``
    reaper — the effect is the same) and every seal on the process failed
    with ``the connection is closed`` from then on, because nothing ever
    looked. A closed connection reports it; a broken-but-open one is caught
    by the retry in :func:`_auto_seal_trace`.
    """
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    try:
        return bool(getattr(conn, "closed", False)) or bool(getattr(conn, "broken", False))
    except Exception:  # noqa: BLE001 - a fake connection in tests
        return False


def _reset_immutable_store() -> None:
    """Drop the cached store so the next call reconnects."""
    global _immutable_trace_store
    with _seal_lock:
        store = _immutable_trace_store
        _immutable_trace_store = None
        conn = getattr(store, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def _get_immutable_store():
    """Lazily create the immutable trace store (Pro only).

    Rebuilt when the cached store's connection has closed under it. The
    schema is not re-applied on reconnect: it was applied when the process
    first opened the store, and re-running the DDL takes locks a busy
    instance should not be asking for on the request path.
    """
    global _immutable_trace_store
    with _seal_lock:
        if _immutable_trace_store is not None:
            if not _store_connection_closed(_immutable_trace_store):
                return _immutable_trace_store
            logger.warning("Immutable trace store connection closed — reconnecting")
            _reset_immutable_store()
            reconnect = True
        else:
            reconnect = False
        if not _HAS_PRO_TRACES:
            return None
        dsn = os.environ.get("AMFS_POSTGRES_DSN")
        if not dsn:
            return None
        try:
            import psycopg
            from psycopg.rows import dict_row
            conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
            _immutable_trace_store = PostgresImmutableTraceStore(
                conn, auto_schema=not reconnect
            )
            logger.info("Immutable trace store %s", "reconnected" if reconnect else "initialized")
            return _immutable_trace_store
        except Exception:
            logger.debug("Failed to init immutable trace store", exc_info=True)
            return None


_seal_sequence: dict[str, int] = {}

def _scan_captured_text(
    mem: AgentMemory,
    text: str | None,
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """Redact secrets from captured text. Thin wrapper over the shared scanner.

    The implementation lives in :mod:`amfs_core.capture` so the SDK and MCP paths
    apply exactly the same rules; this only supplies the identity and adapter from
    the request's memory handle.

    The identity defaults to the handle's, which is the server's shared singleton
    on this process. Callers that already know whose text this is should pass it:
    on ``POST /api/v1/traces`` the trace carries its own agent, and the singleton's
    is the server default rather than the client's.
    """
    return scan_captured_text(
        text,
        adapter=getattr(mem, "_adapter", None),
        agent_id=agent_id if agent_id is not None else mem.agent_id,
        session_id=session_id if session_id is not None else mem.session_id,
    )


def _scan_captured_actions(
    mem: AgentMemory,
    actions: list[Any],
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> list[Any]:
    """Clear secrets from recorded actions, dropping any that cannot be cleared.

    Companion to :func:`_scan_captured_text` with the same identity handling. An
    action survives only if every one of its arguments clears the gate, so a
    dropped action leaves no partial example behind. Its result is scanned too but
    only emptied on a block, since the training target is the tool and its
    arguments and a result-less action is still a usable example.
    """
    scan_identity = {
        "adapter": getattr(mem, "_adapter", None),
        "agent_id": agent_id if agent_id is not None else mem.agent_id,
        "session_id": session_id if session_id is not None else mem.session_id,
    }
    kept = []
    for action in actions:
        arguments = scan_captured_arguments(action.arguments, **scan_identity)
        if arguments is None:
            continue
        summary = scan_captured_text(action.result_summary, **scan_identity)
        kept.append(action.model_copy(update={
            "arguments": arguments,
            "result_summary": summary or "",
        }))
    return kept


def _auto_seal_trace(
    mem: AgentMemory,
    oss_trace: Any | None = None,
    *,
    session_metadata: Any | None = None,
) -> str | None:
    """If Pro traces are available, seal an OSS trace as immutable.

    ``oss_trace`` defaults to the trace ``mem`` just committed. ``POST
    /api/v1/traces`` passes the trace it persisted instead, together with the
    request body's raw ``session_metadata``: validating the body into the OSS
    model discards keys the model does not declare, and the Pro recorder's spans
    and LLM calls travel in exactly such keys.

    The OSS -> immutable mapping is the Pro package's, not this file's. The copy
    that lived here mapped only the fields it knew about, so every field added
    to the sealed record afterwards — ``llm_calls`` first — was silently dropped
    on this path while the other two seal paths carried it.
    """
    if not _HAS_PRO_TRACES:
        return None
    if oss_trace is None:
        oss_trace = getattr(mem, "_last_trace", None)
    if oss_trace is None:
        return None

    def _is_connection_error(exc: BaseException) -> bool:
        try:
            import psycopg
            return isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError))
        except ImportError:  # pragma: no cover - psycopg absent means no store
            return False

    # One retry, and only for a dead connection: the first attempt may be the
    # one that discovers the store's connection went away since the last seal.
    # Anything else fails once, as before. The trace is not lost — the OSS
    # ``decision_traces`` row was written by the commit; only the immutable
    # copy is missing, and the warning names the outcome so it can be found.
    for attempt in (1, 2):
        store = _get_immutable_store()
        if store is None:
            return None
        try:
            with _seal_lock:
                return _seal_with_store(
                    store, mem, oss_trace, session_metadata=session_metadata
                )
        except Exception as exc:  # noqa: BLE001
            if attempt == 1 and _is_connection_error(exc):
                logger.warning(
                    "Trace store connection failed on seal — reconnecting once",
                    exc_info=True,
                )
                _reset_immutable_store()
                continue
            logger.warning(
                "Failed to auto-seal immutable trace for outcome %s",
                getattr(oss_trace, "outcome_ref", None), exc_info=True,
            )
            return None
    return None


def _seal_with_store(
    store: Any,
    mem: AgentMemory,
    oss_trace: Any,
    *,
    session_metadata: Any | None,
) -> str:
    """The seal itself. Caller holds ``_seal_lock`` and handles failure."""
    from uuid import UUID as _UUID

    now = datetime.now(timezone.utc)
    # The trace's own session when it has one: a trace posted by a remote
    # client belongs to that client's session, not to the server handle's.
    session_id = getattr(oss_trace, "session_id", None) or mem.session_id
    seq = _seal_sequence.get(session_id, 0)
    parent_hash = store.get_latest_hash(session_id)

    account_id = None
    try:
        from amfs_postgres.tenant_context import get_request_tenant_account_id
        tid = get_request_tenant_account_id()
        if tid:
            account_id = _UUID(tid)
    except (ImportError, ValueError):
        pass

    imm = _pro_immutable_from_oss_trace(
        oss_trace,
        session_id=session_id,
        sequence_number=seq,
        account_id=account_id,
        # From the trace, not from ``mem``. ``mem`` is shared by every
        # request: the caller's agent is written onto its tagger for the
        # duration of the commit and restored in a ``finally``, and this runs
        # after that restore — so reading it here sealed every trace under
        # the server's own default agent. Tuning datasets are built from the
        # sealed traces and selected by agent, so the loss is silent: the
        # model trains on an empty set. The trace was built while the tagger
        # still pointed at the caller.
        agent_id=getattr(oss_trace, "agent_id", None) or mem.agent_id,
        created_at=now,
        session_metadata=session_metadata,
    )
    imm = _pro_finalize_spans(imm)
    sealed = seal(
        imm,
        get_signing_key(),
        parent_hash=parent_hash,
        sequence_number=seq,
        signing_key_id=get_signing_key_id(),
    )
    saved = store.save(sealed)
    _seal_sequence[session_id] = seq + 1
    logger.info("Auto-sealed immutable trace %s for outcome %s",
                 saved.id, getattr(oss_trace, "outcome_ref", None))
    return str(saved.id)


# The longest slice of a request used to look for matching memory. task_input is
# capped at MAX_CAPTURED_CHARS (200k), and embedding a novel to count how many
# entries relate to it would cost more than the commit it rides on. The opening
# of a request carries what it is about; the rest is detail.
#: ``adaptive_k``: results scoring below this fraction of a validated top hit
#: are dropped. 0.85 keeps near-ties (two confirmed approaches) and drops the
#: long tail of alternatives the record has said nothing about.
ADAPTIVE_K_KEEP_RATIO = 0.85
#: ``adaptive_k`` under the anchored blend: an entry the record has not
#: confirmed also stays behind a validated leader when its graded lexical
#: coverage exceeds the leader's by this much — it carries a rare query term
#: (the service name, the error code) the leader does not. Read on relevance
#: alone the rule pruned the untested runbook for the task's own service
#: behind a validated note about another service written in the query's
#: phrasing: a hair less relevant, and gone from a one-row list. Three
#: equally relevant fixes for the same symptom, one of them validated, still
#: collapse to the validated one, which is what the option is for.
ADAPTIVE_K_KEYWORD_GAP = 0.2
#: Most discredited entries appended for ``include_avoid``.
AVOID_LIST_MAX = 3
_GAP_QUERY_CHARS = 2_000
#: An agent that failed more times than this in one task has a problem this
#: endpoint cannot label; the cap keeps a runaway loop from posting a megabyte
#: of attempts into one outcome row.
_MAX_ATTEMPTS_PER_OUTCOME = 50

# Named rather than inlined because it is the one number a reader will want to
# argue with: enough keys to act on, few enough that the block stays a summary.
_GAP_SAMPLE = 3
#: Seconds the commit response waits for the gap report. It is the least
#: valuable thing on the response and runs two 150-row searches plus an
#: embedding; under load those were a visible share of commit latency. Past
#: the budget the commit returns without it. ``AMFS_MEMORY_GAP_TIMEOUT``
#: overrides; ``0`` turns the report off.
_GAP_TIMEOUT_S = float(os.environ.get("AMFS_MEMORY_GAP_TIMEOUT", "2.0") or 0.0)


async def _memories_matching_task(
    task_input: str,
    *,
    request: Request,
    branch: str = "main",
) -> list[str]:
    """Entry keys that would surface for ``task_input`` — without reading them.

    This is the candidate half of ``/api/v1/retrieve``: the semantic and lexical
    channels, the excluded-namespace drop, and the visibility filter. It is
    deliberately not the other half. No blend, no rerank, no ``limit`` trim, and
    above all **no ``recall_count`` bump** — this runs to measure what the agent
    could have consulted, and a measurement that credits reuse inflates the very
    number it exists to report. ``AgentMemory.explain`` refuses the same trap for
    the same reason, serving from read-time snapshots rather than re-reading.

    Where it does stay in step with retrieve is relevance: the predicate below is
    retrieve's own abstain rule — a real semantic hit, or a lexical one — so an
    entry counted here is one the agent would have been shown had it asked. That
    matters because the agent can check this claim by running ``retrieve`` on the
    same text, and a number it cannot reproduce is worse than no number.

    Runs in-process against the async adapter and the local ONNX embedder, so it
    costs no metered operation and no HTTP hop. Returns keys ordered strongest
    first, semantic hits ahead of lexical-only ones.
    """
    from amfs_core.query_norm import normalize_temporal

    if _async_adapter is None:
        return []
    # Truncated before normalising so the regex pass is not run over 200k of text.
    text = task_input.strip()[:_GAP_QUERY_CHARS]
    if not text:
        return []
    # Retrieve searches only the topical remainder, so this must too, or the count
    # is one the agent cannot reproduce with ``amfs_retrieve`` on the same text —
    # which the note below invites it to do. ``topical`` falls back to the original
    # when stripping would empty it, so a request that is only a date still
    # searches for something.
    query = normalize_temporal(text).topical

    embedder = _get_server_embedder()
    # entry_key -> {"entry", "sim", "keyword"}, the same slot shape retrieve uses.
    candidates: dict[str, dict[str, Any]] = {}

    if embedder is not None:
        try:
            pairs = await _async_adapter.semantic_search(
                SemanticQuery(text=query, limit=150), embedder, branch=branch
            )
        except Exception:  # noqa: BLE001 - a gap report never costs a commit
            logger.debug("gap semantic_search failed", exc_info=True)
            pairs = []
        for entry, sim in pairs:
            slot = candidates.get(entry.entry_key)
            if slot is None:
                candidates[entry.entry_key] = {"entry": entry, "sim": sim, "keyword": 0.0}
            elif sim > slot["sim"]:
                slot["sim"] = sim

    # Always run, exactly as retrieve does, and load-bearing here beyond parity:
    # the semantic channel requires ``embedding IS NOT NULL``, and outcome
    # propagation has historically stripped embeddings from precisely the entries
    # that outcomes validated. Without the lexical channel the report would be
    # blindest about the memories that earned their confidence.
    try:
        lex = await _async_adapter.search(
            SearchQuery(query=query, limit=150, sort_by="confidence", depth=3),
            branch=branch,
        )
    except Exception:  # noqa: BLE001
        logger.debug("gap lexical search failed", exc_info=True)
        lex = []
    for entry in lex:
        slot = candidates.get(entry.entry_key)
        if slot is None:
            candidates[entry.entry_key] = {"entry": entry, "sim": 0.0, "keyword": 1.0}
        else:
            slot["keyword"] = 1.0

    candidates = {
        k: v
        for k, v in candidates.items()
        if not _is_excluded_entity(getattr(v["entry"], "entity_path", ""))
        and not _is_synthetic_key(getattr(v["entry"], "key", ""))
    }

    # Applied over the merged set, once, for the same reason retrieve does it
    # there: a lexical-only hit must be filtered exactly like a semantic one or
    # the count itself becomes a leak path.
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        allowed = {
            e.entry_key for e in vis.filter_entries([v["entry"] for v in candidates.values()])
        }
        candidates = {k: v for k, v in candidates.items() if k in allowed}

    floor = _retrieve_min_semantic()
    matched = [
        (k, v["sim"])
        for k, v in candidates.items()
        if v["sim"] >= floor or v["keyword"]
    ]
    matched.sort(key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in matched]


async def _memory_gap(req: OutcomeRequest, *, request: Request) -> dict[str, Any] | None:
    """Report the memory this task matched against the memory it drew on.

    The loop this product claims only closes if storing a memory leads to using
    one, and nothing in a session tells the agent it skipped something. The
    commit is where that can still be said: the request is in hand, so what
    *would* have matched is computable, and the trace already knows what was
    linked to the outcome.

    On wording, which is the whole risk here. ``causal_entry_keys`` holds entries
    that were explicitly recorded as read — direct reads, and retrieve's top hit.
    A briefing records none, and neither does a bare search. So an entry that is
    matched-but-unlinked is **not** an entry the agent ignored, and this must
    never say it was: a session that opened with a briefing and used a dozen
    entries well would be the one accused. It reports linkage, which is what it
    can prove, and names both readings in the note so the agent can tell which
    applies to it.

    Returns ``None`` when there is nothing worth saying, so a commit that matched
    nothing stays quiet rather than reporting a zero.
    """
    if not (req.task_input or "").strip():
        return None
    matched = await _memories_matching_task(req.task_input or "", request=request)
    if not matched:
        return None

    linked = set(req.causal_entry_keys or [])
    unlinked = [k for k in matched if k not in linked]
    linked_count = len(matched) - len(unlinked)
    gap: dict[str, Any] = {
        "matched": len(matched),
        "linked_to_outcome": linked_count,
    }
    if not unlinked:
        return gap
    # Count and sample are separate fields so no wording can pass a capped list
    # off as a complete one. This block's only job is to be believed.
    gap["unlinked"] = len(unlinked)
    gap["unlinked_sample"] = unlinked[:_GAP_SAMPLE]
    plural = "memory" if len(matched) == 1 else "memories"
    if linked_count:
        shown = (
            "the strongest are in unlinked_sample"
            if len(unlinked) > _GAP_SAMPLE
            else "they are in unlinked_sample"
        )
        gap["note"] = (
            f"{len(matched)} stored {plural} match this task and "
            f"{linked_count} are linked to this outcome. Of the "
            f"{len(unlinked)} that are not, {shown} — worth a look if this task "
            "comes round again."
        )
    else:
        gap["note"] = (
            f"{len(matched)} stored {plural} match this task and none are linked "
            "to this outcome. If you did not consult them, amfs_retrieve on the "
            "task text will surface them. If you did — through a briefing or a "
            "search, which record no read — reading the entry directly is what "
            "records the link."
        )
    return gap


@app.post("/api/v1/outcomes")
async def commit_outcome(
    req: OutcomeRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()

    otype = _OUTCOME_TYPE_MAP.get(req.outcome_type.lower())
    if otype is None:
        valid = ", ".join(_OUTCOME_TYPE_MAP.keys())
        return {"error": f"Invalid outcome_type '{req.outcome_type}'. Must be one of: {valid}"}

    # The remote session's attribute bag and LLM calls, passed explicitly for
    # the same reason ``tool_calls`` is: ``mem`` is shared, so nothing may be
    # buffered on it between requests. Attributes are validated by
    # ``commit_outcome``; a bad bag is a 422 here rather than a lost commit.
    client_meta = req.session_metadata or {}
    try:
        client_attributes = validate_session_attributes(client_meta.get("attributes"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"session_metadata.attributes: {exc}"
        ) from exc
    # After validation, never before: the routing layer's stamps (which canary
    # this session was in, and which arm) are the server's, so they are exempt
    # from the client's key cap and overwrite anything the body claimed.
    client_attributes = _merge_routed_attributes(request, client_attributes) or {}
    client_llm_calls = client_meta.get("llm_calls")
    if not isinstance(client_llm_calls, list):
        client_llm_calls = []
    # Attempts may arrive on the body or, from an older SDK, only inside the
    # session metadata; the body wins. Validated so a malformed attempt is a
    # 422 rather than a trigger error mid-commit.
    raw_attempts = req.attempts or client_meta.get("attempts") or []
    if not isinstance(raw_attempts, list):
        raise HTTPException(status_code=422, detail="attempts must be a list")
    try:
        attempts = [AttemptRecord.model_validate(a) for a in raw_attempts]
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"attempts: {exc}") from exc
    if len(attempts) > _MAX_ATTEMPTS_PER_OUTCOME:
        raise HTTPException(
            status_code=422,
            detail=f"attempts: at most {_MAX_ATTEMPTS_PER_OUTCOME} per outcome",
        )
    final_action_index = req.final_action_index
    if final_action_index is None:
        raw_fai = (client_meta.get("attributes") or {}).get("final_action_index") \
            if isinstance(client_meta.get("attributes"), dict) else None
        if isinstance(raw_fai, int) and not isinstance(raw_fai, bool):
            final_action_index = raw_fai
    if final_action_index is not None and (
        final_action_index < 0 or final_action_index >= max(1, len(req.tool_calls))
    ):
        raise HTTPException(
            status_code=422, detail="final_action_index must index into tool_calls"
        )
    commit_kwargs: dict[str, Any] = dict(
        causal_entry_keys=req.causal_entry_keys,
        causal_confidence=req.causal_confidence,
        attempts=attempts,
        final_action_index=final_action_index,
        # The client's read versions, never the server's shared tracker.
        causal_entry_versions=req.causal_entry_versions or {},
        # Action-level learning. Derived from the request's own tool calls
        # and attempts when the client did not send them, never from the
        # shared tracker; the keys are scanned inside commit_outcome.
        actions_taken=(
            req.actions_taken
            if req.actions_taken is not None
            else derive_actions_taken(
                req.tool_calls,
                [a.model_dump(mode="json") for a in attempts],
                final_action_index if final_action_index is not None
                else (len(req.tool_calls) - 1 if req.tool_calls else None),
                otype.value,
            )
        ),
        entity_path=req.entity_path,
        entity_paths=req.entity_paths,
        situation=req.situation[:200] if req.situation else None,
        # Not scanned here: commit_outcome scans at trace construction, so
        # every caller gets it. Scanning again would be harmless but would
        # imply this endpoint is where the guarantee lives, which is the
        # assumption that left the MCP path unscanned.
        task_input=req.task_input,
        response_text=req.response_text,
        # Passed explicitly, and never omitted: ``mem`` is shared across
        # requests, so letting this fall through to its tracker would attribute
        # whatever actions happen to be buffered there to this caller.
        tool_calls=req.tool_calls,
        attributes=client_attributes or None,
        llm_calls=client_llm_calls or None,
        # The same declaration that suppresses the seal below, applied one
        # layer deeper. The trace this would write is assembled on the shared
        # handle, so it carried this process's session and whatever the last
        # request left on its tracker; the caller's own trace arrives on
        # ``/traces`` moments later. Persisting both left two decision_traces
        # rows per outcome, indistinguishable by agent because the tagger is
        # pointed at the caller for exactly this block — so every count and
        # ratio taken over that table was measured against a population
        # roughly twice its true size, half of it untrue.
        persist_trace=not req.trace_follows,
    )

    # The commit and the seal run on the DB executor rather than inline: the
    # sync adapter's transaction, the outcome embedding, the trace insert and
    # the immutable seal together held the event loop for seconds under load,
    # and every other request on the instance waited behind them.
    #
    # They run on a per-request handle (``as_agent``), not the shared one.
    # The commit reads the handle's identity throughout and leaves the trace
    # it built on the handle for the seal to pick up; now that both sides of
    # that hand-off are awaited, swapping the shared tagger and restoring it
    # in a ``finally`` would stamp interleaved commits and writes with each
    # other's agent — the corruption ``as_agent`` exists to end. The clone
    # shares the adapter and nothing mutable, so commits no longer serialize
    # on the process.
    if req.agent_id:
        # Ownership guard first: a 409 for a foreign-owned identity ends the
        # request before anything is written as that agent.
        _link_agent_owner_once(request, req.agent_id, mem.namespace)
        try:
            await _offload(_db_executor, mem._adapter.ensure_agent, req.agent_id, mem.namespace)
        except Exception:
            pass
    # With no causal keys on the body the SDK falls back to the handle's read
    # tracker; the clone's resolves to this request's scope exactly as the
    # shared one does (``read_tracker_scope`` middleware), so that path is
    # unchanged by the clone.
    handle = mem.as_agent(req.agent_id or mem.agent_id)
    entries = await _offload(
        _db_executor, handle.commit_outcome, req.outcome_ref, otype, **commit_kwargs
    )
    # Skipped when the caller's own trace is on its way: the caller's actions
    # and attribute bag were passed in explicitly above, but the causal
    # entries, query events and session window on this handle are only what
    # this request supplied. Sealing it as well as the caller's left two
    # traces per outcome — doubling every count and average taken over them.
    immutable_trace_id = (
        None if req.trace_follows
        else await _offload(_db_executor, _auto_seal_trace, handle)
    )

    _ip = request.client.host if request.client else None
    _bg_executor.submit(
        contextvars.copy_context().run,
        functools.partial(
            _audit_log, "outcome.commit", resource=req.outcome_ref, ip_address=_ip
        ),
    )

    result: dict[str, Any] = {
        "outcome_ref": req.outcome_ref,
        "outcome_type": req.outcome_type,
        "affected_entries": len(entries),
        "entries": [_entry_to_response(e) for e in entries],
    }
    if immutable_trace_id:
        result["immutable_trace_id"] = immutable_trace_id
    # Fail-open, and not as a formality: this is the least valuable thing on the
    # response and the commit is the most valuable thing in the session, so the
    # order of those two should be visible in the code. The same reasoning put
    # the loose annotation on the MCP ``actions`` parameter — a defect in the
    # reporting half must never cost the seal.
    gap = None
    if _GAP_TIMEOUT_S > 0:
        try:
            gap = await asyncio.wait_for(_memory_gap(req, request=request), timeout=_GAP_TIMEOUT_S)
        except TimeoutError:
            logger.debug("memory gap report skipped: over %.1fs budget", _GAP_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            logger.debug("memory gap report failed", exc_info=True)
    if gap is not None:
        result["memory_gap"] = gap
    return result


@app.get("/api/v1/outcomes")
def list_outcomes(
    request: Request,
    entity_path: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(100),
    outcome_ref: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    since_dt = datetime.fromisoformat(since) if since else None
    records = mem._adapter.list_outcomes(
        entity_path=entity_path,
        since=since_dt,
        limit=limit,
        outcome_ref=outcome_ref,
    )
    allowed = _visible_agent_ids(request)
    if allowed is not None:
        records = [r for r in records if r.agent_id in allowed]
    return {"outcomes": [r.model_dump(mode="json") for r in records]}


# ──────────────────────────────────────────────────────────────────────
# Context & Explain
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/context")
def record_context(
    req: ContextRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    mem.record_context(req.label, req.summary, source=req.source)
    return {"recorded": req.label, "source": req.source}


@app.get("/api/v1/explain")
def explain(
    request: Request,
    outcome_ref: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    mem = _get_memory()
    allowed = _visible_agent_ids(request)
    if allowed is not None:
        # Restricted members may only explain outcomes committed by agents
        # they can see; the ref-less session explain spans the whole account.
        if outcome_ref is None:
            raise HTTPException(status_code=403, detail="outcome_ref is required")
        # Filtered where the outcomes live rather than paged through all of
        # them here; an outcome_ref is not unique, so the newest record wins.
        records = mem._adapter.list_outcomes(outcome_ref=outcome_ref, limit=1)
        record = records[0] if records else None
        if record is None or record.agent_id not in allowed:
            raise HTTPException(status_code=404, detail="Outcome not found")
    return mem.explain(outcome_ref)


# ──────────────────────────────────────────────────────────────────────
# Decision Traces
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/traces")
async def list_traces(
    request: Request,
    entity_path: str | None = Query(None),
    agent_id: str | None = Query(None),
    outcome_type: str | None = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
    cursor: str | None = Query(None),
    since: str | None = Query(None),
    until: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Decision traces, newest first, keyset-paginated.

    Follow ``next_cursor`` while ``has_more`` is true. ``offset`` is honoured
    only when no cursor is given. The page's cursor points at the last row
    read, so a caller whose visibility hides some agents still advances past
    them rather than seeing the same rows again.

    ``since`` (inclusive) and ``until`` (exclusive) bound ``created_at`` in
    the query itself, so a window that excludes the newest traces still
    returns the older matches instead of an empty first page. Both go through
    :func:`_parse_ts` like the sibling list routes: naive input is taken as
    UTC rather than compared naive against aware timestamps.
    """
    limit = clamp_limit(limit)
    page = await _list_traces_page(
        entity_path=entity_path,
        agent_id=agent_id,
        outcome_type=outcome_type,
        limit=limit,
        offset=offset,
        cursor=_check_cursor(cursor),
        since=_parse_ts(since),
        until=_parse_ts(until),
    )
    traces = page.items
    allowed = _visible_agent_ids(request)
    if allowed is not None:
        traces = [t for t in traces if t.agent_id in allowed]
    return {
        "traces": [t.model_dump(mode="json") for t in traces],
        **_page_meta(page),
    }


@app.post("/api/v1/traces")
async def save_trace(
    req: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Persist a decision trace (used by HttpAdapter.save_trace)."""
    body = await req.json()
    trace = DecisionTrace.model_validate(body)
    mem = _get_memory()
    if trace.agent_id:
        _link_agent_owner_once(req, trace.agent_id, mem.namespace)
    # HttpAdapter posts the whole trace here rather than through /outcomes, so
    # this is the second entry point captured text can arrive on and it needs
    # the same scan before anything is written. Unconditionally, with no truthiness
    # guard: an empty string skipped the scan and was stored as "" while every
    # other path normalises absent capture to None.
    #
    # Scanned under the *trace's* identity, not the server singleton's. No rule
    # reads it today — SafetyGate matches on content and on the fixed capture
    # namespace — so this changes no redaction now. It is the identity the
    # provenance should have carried all along, and getting it right here means a
    # later per-agent policy applies to the agent that produced the text rather
    # than to whichever identity this process happens to hold.
    if trace.task_input is not None or trace.response_text is not None:
        trace = trace.model_copy(update={
            "task_input": _scan_captured_text(
                mem,
                trace.task_input,
                agent_id=trace.agent_id,
                session_id=trace.session_id,
            ),
            "response_text": _scan_captured_text(
                mem,
                trace.response_text,
                agent_id=trace.agent_id,
                session_id=trace.session_id,
            ),
        })
    # Action arguments are caller-supplied on the same footing as the capture above
    # and get the same treatment, under the same identity. An action whose
    # arguments cannot be cleared is dropped rather than stored partially.
    if trace.tool_calls:
        trace = trace.model_copy(update={
            "tool_calls": _scan_captured_actions(
                mem,
                trace.tool_calls,
                agent_id=trace.agent_id,
                session_id=trace.session_id,
            ),
        })
    raw_meta = body.get("session_metadata") if isinstance(body, dict) else None
    # The routing layer's stamps, on the same footing as in /outcomes: this is
    # the other request a routed session's trace can arrive on, and a canary
    # arm that is stamped on one path and not the other is a tally that counts
    # SDK sessions in neither arm. Written onto the trace that is saved and onto
    # the raw metadata that is sealed, since the seal prefers the raw body when
    # there is one and the trace's own metadata when there is not. Through the
    # same merge as /outcomes, so a decided-but-unrouted request (``{}``) drops
    # the client's own canary claims here too — this is the path every
    # HttpAdapter commit seals on, since ``trace_follows`` skips the other.
    if _routed_trace_attributes(req) is not None:
        meta = trace.session_metadata or SessionMetadata()
        existing = getattr(meta, "attributes", None)
        merged = _merge_routed_attributes(
            req, dict(existing) if isinstance(existing, dict) else {}
        )
        trace = trace.model_copy(
            update={"session_metadata": meta.model_copy(update={"attributes": merged or {}})}
        )
        if isinstance(raw_meta, dict):
            raw_attrs = raw_meta.get("attributes")
            raw_meta = {
                **raw_meta,
                "attributes": _merge_routed_attributes(
                    req, dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
                ) or {},
            }
    # Both DB round trips off the event loop; the seal takes its own lock on
    # the trace store, and nothing here reads the shared handle's tracker, so
    # unlike /outcomes this path needs no serialisation of its own.
    saved = await _offload(_db_executor, mem._adapter.save_trace, trace)
    # Sealed like a trace committed through /outcomes. Until this call, a trace
    # arriving here — which is every trace an HttpAdapter client commits — was
    # never sealed, so it had no immutable copy at all. The saved trace is what
    # is sealed, so the immutable copy carries the persisted id; the raw body is
    # passed alongside because ``model_validate`` above dropped the
    # ``session_metadata`` keys the Pro recorder's spans travel in.
    immutable_trace_id = await _offload(
        _db_executor, _auto_seal_trace, mem, saved, session_metadata=raw_meta
    )
    result = saved.model_dump(mode="json")
    if immutable_trace_id:
        result["immutable_trace_id"] = immutable_trace_id
    return result


# NOTE: must be registered before /api/v1/traces/{trace_id} so "share-stats"
# isn't captured as a trace_id.
@app.get("/api/v1/traces/share-stats")
def get_share_stats(
    request: Request,
    since: datetime | None = Query(None),
    pair_limit: int = Query(20, ge=1, le=100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Cross-agent knowledge-share totals and top reader/author pairs,
    aggregated server-side so full traces never cross the wire."""
    mem = _get_memory()

    vis = _get_visibility_filter(request)
    agent_ids: list[str] | None = None
    if vis is not None and vis.should_filter():
        agent_ids = sorted(vis.get_visible_agent_ids())

    stats = mem._adapter.share_stats(
        since=since, pair_limit=pair_limit, agent_ids=agent_ids
    )
    return json.loads(json.dumps(stats, default=str))


@app.get("/api/v1/traces/{trace_id}")
async def get_trace(
    request: Request,
    trace_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    trace = await _via_async_or_sync("get_trace", trace_id)
    allowed = _visible_agent_ids(request)
    if trace is not None and allowed is not None and trace.agent_id not in allowed:
        trace = None
    if trace is None:
        return JSONResponse({"error": "Trace not found"}, status_code=404)
    return trace.model_dump(mode="json")


@app.post("/api/v1/traces/{trace_id}/explain")
def explain_trace(
    request: Request,
    trace_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    api_key = os.environ.get("AMFS_LLM_API_KEY", "")
    if not api_key:
        return JSONResponse(
            {"error": "LLM not configured. Set AMFS_LLM_API_KEY to enable AI explanations."},
            status_code=503,
        )

    mem = _get_memory()
    trace = mem._adapter.get_trace(trace_id)
    allowed = _visible_agent_ids(request)
    if trace is not None and allowed is not None and trace.agent_id not in allowed:
        trace = None
    if trace is None:
        return JSONResponse({"error": "Trace not found"}, status_code=404)

    provider = os.environ.get("AMFS_LLM_PROVIDER", "openai")
    model = os.environ.get("AMFS_LLM_MODEL", "gpt-4o-mini")

    td = trace.model_dump(mode="json")

    entries_desc = []
    for e in td.get("causal_entries", []):
        desc = f"- {e['entity_path']}/{e['key']} (v{e['version']}, confidence: {e['confidence']:.0%})"
        if e.get("value"):
            desc += f"\n  Value: {json.dumps(e['value'], default=str)}"
        if e.get("memory_type"):
            desc += f"\n  Type: {e['memory_type']}"
        if e.get("written_by"):
            desc += f"\n  Written by: {e['written_by']}"
        entries_desc.append(desc)

    contexts_desc = []
    for c in td.get("external_contexts", []):
        desc = f"- {c['label']}: {c['summary']}"
        if c.get("source"):
            desc += f" (source: {c['source']})"
        contexts_desc.append(desc)

    queries_desc = []
    for q in td.get("query_events", []):
        desc = f"- {q['operation']}({json.dumps(q.get('parameters', {}))}) → {q.get('result_count', 0)} results"
        if q.get("duration_ms"):
            desc += f" in {q['duration_ms']:.1f}ms"
        queries_desc.append(desc)

    errors_desc = []
    for e in td.get("error_events", []):
        errors_desc.append(f"- [{e['operation']}] {e['error_type']}: {e['message']}")

    diff_desc = ""
    sd = td.get("state_diff")
    if sd:
        diff_desc = f"Entries created: {sd.get('entries_created', 0)}, updated: {sd.get('entries_updated', 0)}"
        for cc in sd.get("confidence_changes", []):
            diff_desc += f"\n  {cc['entity_path']}/{cc['key']}: {cc['before']:.0%} → {cc['after']:.0%}"

    duration_str = ""
    if td.get("session_duration_ms"):
        mins = td["session_duration_ms"] / 60000
        duration_str = f"{mins:.0f} minutes" if mins >= 1 else f"{td['session_duration_ms']:.0f}ms"

    prompt = f"""You are analyzing an AI agent's decision trace from AMFS (Agent Memory File System). Your job is to explain what happened in clear, actionable language that helps a human understand the agent's reasoning and the impact of its decision.

DECISION TRACE DATA:
- Agent: {td.get('agent_id')}
- Outcome Reference: {td.get('outcome_ref', 'None')}
- Outcome Type: {td.get('outcome_type', 'None')}
- Decision Summary: {td.get('decision_summary', 'No summary provided')}
- Session Duration: {duration_str or 'Unknown'}

MEMORY ENTRIES READ (what the agent knew):
{chr(10).join(entries_desc) if entries_desc else 'None'}

EXTERNAL SOURCES CONSULTED:
{chr(10).join(contexts_desc) if contexts_desc else 'None'}

SEARCHES PERFORMED:
{chr(10).join(queries_desc) if queries_desc else 'None'}

ERRORS ENCOUNTERED:
{chr(10).join(errors_desc) if errors_desc else 'None'}

STATE CHANGES:
{diff_desc or 'None'}

Respond with a JSON object containing these fields:
- "narrative": A 2-3 paragraph human-readable story explaining what happened. Start with what the agent was trying to do, then what information it gathered and from where, then what decision it made and why, and finally what the outcome was. Use specific data values from the trace. Write as if explaining to a team lead.
- "key_findings": An array of 3-5 bullet points highlighting the most important facts that influenced the decision. Each should be a complete sentence.
- "risk_assessment": A paragraph assessing risks. For incidents/regressions, explain what went wrong. For clean deploys, explain what risks were mitigated and what could still go wrong.
- "confidence_analysis": A paragraph explaining why the confidence levels are what they are, referencing specific before/after changes if available.
- "recommendations": An array of 2-4 actionable recommendations for what should happen next based on this decision trace.

Return ONLY valid JSON, no markdown formatting."""

    try:
        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                return JSONResponse(
                    {"error": "openai package not installed. Run: pip install openai"},
                    status_code=503,
                )
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            result_text = response.choices[0].message.content or "{}"
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                return JSONResponse(
                    {"error": "anthropic package not installed. Run: pip install anthropic"},
                    status_code=503,
                )
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt + "\n\nRespond with JSON only."}],
            )
            result_text = response.content[0].text
        else:
            return JSONResponse({"error": f"Unsupported LLM provider: {provider}"}, status_code=400)

        explanation = json.loads(result_text)

        for field in ["narrative", "key_findings", "risk_assessment", "confidence_analysis", "recommendations"]:
            if field not in explanation:
                explanation[field] = [] if field in ("key_findings", "recommendations") else ""

        return {"explanation": explanation, "model": model, "provider": provider}

    except json.JSONDecodeError:
        return JSONResponse({"error": "LLM returned invalid JSON", "raw": result_text[:500]}, status_code=502)
    except Exception as exc:
        logger.exception("LLM explain failed")
        return JSONResponse({"error": f"LLM call failed: {exc}"}, status_code=502)


# ──────────────────────────────────────────────────────────────────────
# Admin — Usage
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/usage")
def get_usage(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    # Account-wide usage/billing telemetry is admin-only. Non-admin members
    # get a zeroed payload (rather than 403) so the settings page still
    # renders without exposing the owner's data.
    if _active_visibility_filter(request) is not None:
        return {
            "requestsToday": 0,
            "requestsThisMonth": 0,
            "peakRpm": 0,
            "avgLatencyMs": 0,
            "quotas": [],
            "topAgents": [],
            "topEntities": [],
        }
    mem = _get_memory()
    st = mem.stats()
    # Counted in SQL. This used to be len() over a list capped at 10,000, so
    # every account past that reported exactly 10,000 decision traces.
    outcome_count = mem._adapter.count_outcomes()

    top_agents = sorted(st.agents.items(), key=lambda x: x[1], reverse=True)[:10]
    top_entities = sorted(st.entities.items(), key=lambda x: x[1], reverse=True)[:10]

    api_key_count = 0
    requests_today = 0
    requests_this_month = 0
    pool = _get_db_pool()
    ns = _get_namespace()

    if pool is not None:
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT count(*) AS cnt FROM amfs_api_keys WHERE namespace = %s AND active = true",
                        (ns,),
                    )
                    row = cur.fetchone()
                    api_key_count = row["cnt"] if row else 0

                    cur.execute(
                        """SELECT count(*) AS cnt FROM amfs_audit_log
                           WHERE namespace = %s
                             AND created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')""",
                        (ns,),
                    )
                    row = cur.fetchone()
                    requests_today = row["cnt"] if row else 0

                    cur.execute(
                        """SELECT count(*) AS cnt FROM amfs_audit_log
                           WHERE namespace = %s
                             AND created_at >= date_trunc('month', now() AT TIME ZONE 'UTC')""",
                        (ns,),
                    )
                    row = cur.fetchone()
                    requests_this_month = row["cnt"] if row else 0
        except Exception:
            pass

    return {
        "requestsToday": requests_today,
        "requestsThisMonth": requests_this_month,
        "peakRpm": 0,
        "avgLatencyMs": 0,
        "quotas": [
            {"label": "Memory entries", "current": st.total_entries, "limit": 0},
            {"label": "Decision traces", "current": outcome_count, "limit": 0},
            {"label": "API keys", "current": api_key_count, "limit": 0},
            {"label": "Users", "current": st.total_agents, "limit": 0},
        ],
        "topAgents": [
            {"agentId": aid, "requests": count} for aid, count in top_agents
        ],
        "topEntities": [
            {"entityPath": ep, "reads": 0, "writes": count}
            for ep, count in top_entities
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents")
async def list_agents(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all known agents with entry counts and last activity."""
    try:
        from amfs_postgres.tenant_context import get_request_tenant_account_id
        _tls_acct = get_request_tenant_account_id()
    except ImportError:
        _tls_acct = "NO_MODULE"
    _state_acct = getattr(request.state, "account_id", None)
    _state_user = getattr(request.state, "user_id", None)
    _has_ctx = getattr(request.state, "tenant_ctx", None) is not None
    logger.warning(
        "[TLS-DIAG] /agents tls_account=%s state_account=%s state_user=%s has_tenant_ctx=%s",
        _tls_acct, _state_acct, _state_user, _has_ctx,
    )
    mem = _get_memory()
    vis = _active_visibility_filter(request)
    # A non-admin sees only agents they own, and an agent's own entries are
    # always visible to its owner — so the entry-level visibility pass the
    # handler used to run reduces, for this listing, to "these agent ids".
    own: set[str] | None = set(vis.get_user_agents()) if vis is not None else None

    agent_data: dict[str, dict[str, Any]] = {}
    summarise = getattr(mem._adapter, "agent_summaries", None)
    if callable(summarise):
        # One GROUP BY instead of loading every entry in the account into
        # Python to count them — 100K rows on the largest account, on the
        # event loop, for a page that shows a few dozen numbers.
        rows = await _offload(
            _db_executor, summarise, agent_ids=sorted(own) if own is not None else None
        )
        for r in rows:
            agent_data[r["agent_id"]] = {
                "agent_id": r["agent_id"],
                "entries_written": r["entries_written"],
                "entities_touched": r["entities_touched"],
                "last_active": r["last_active"],
                "first_seen": r["first_seen"],
            }
    else:
        entries = await _offload(_db_executor, mem.list)
        if vis is not None:
            entries = vis.filter_entries(entries)
        for e in entries:
            if e.entity_path.startswith("_system/"):
                continue
            aid = e.provenance.agent_id
            if aid not in agent_data:
                agent_data[aid] = {
                    "agent_id": aid,
                    "entries_written": 0,
                    "entities_touched": 0,
                    "_entities": set(),
                    "last_active": e.provenance.written_at,
                    "first_seen": e.provenance.written_at,
                }
            agent_data[aid]["entries_written"] += 1
            agent_data[aid]["_entities"].add(e.entity_path)
            if e.provenance.written_at > agent_data[aid]["last_active"]:
                agent_data[aid]["last_active"] = e.provenance.written_at
            if e.provenance.written_at and (
                agent_data[aid]["first_seen"] is None
                or e.provenance.written_at < agent_data[aid]["first_seen"]
            ):
                agent_data[aid]["first_seen"] = e.provenance.written_at
        for d in agent_data.values():
            d["entities_touched"] = len(d.pop("_entities"))
        if own is not None:
            agent_data = {aid: d for aid, d in agent_data.items() if aid in own}

    if own is not None:
        # Include owner-linked agents that have written zero entries (e.g. an
        # agent that called set_identity over MCP but hasn't written memory
        # yet). Without this they never appear on the dashboard.
        for aid in own:
            if aid not in agent_data:
                agent_data[aid] = {
                    "agent_id": aid,
                    "entries_written": 0,
                    "entities_touched": 0,
                    "last_active": None,
                    "first_seen": None,
                }
    logger.warning(
        "[AGENTS] agents=%d scoped=%s sql=%s",
        len(agent_data), own is not None, callable(summarise),
    )

    known_agent_ids = list(agent_data.keys())

    def _registration_and_descriptions() -> tuple[dict[str, Any], dict[str, Any]]:
        """Two small reads, off the event loop together."""
        registration: dict[str, dict[str, Any]] = {}
        if known_agent_ids:
            try:
                from amfs_postgres.adapter import PostgresAdapter
                adapter = mem._adapter
                if isinstance(adapter, PostgresAdapter):
                    with adapter._pool.connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT agent_id, created_at, last_active_at, profile "
                                "FROM amfs_agents "
                                "WHERE namespace = %s AND agent_id = ANY(%s)",
                                [adapter._namespace, known_agent_ids],
                            )
                            for row in cur.fetchall():
                                registration[row["agent_id"]] = {
                                    "created_at": row["created_at"],
                                    "last_active_at": row.get("last_active_at"),
                                    "profile": row.get("profile"),
                                }
            except (ImportError, Exception):
                pass
        descriptions: dict[str, dict[str, Any]] = {}
        try:
            for de in mem.list("_system/agents"):
                val = de.value if isinstance(de.value, dict) else {}
                descriptions[de.key] = {
                    "description": val.get("description", ""),
                    "platform": val.get("platform", ""),
                }
        except Exception:
            pass
        return registration, descriptions

    agent_registration, agent_descriptions = await _offload(
        _db_executor, _registration_and_descriptions
    )

    agents = []
    for ad in sorted(agent_data.values(), key=lambda x: x["entries_written"], reverse=True):
        desc_info = agent_descriptions.get(ad["agent_id"], {})
        reg = agent_registration.get(ad["agent_id"], {})
        created = reg.get("created_at") or ad.get("first_seen")
        # Zero-write agents have no entry-derived activity; fall back to the
        # registration row (set via set_identity / profile update).
        last_active = ad["last_active"] or reg.get("last_active_at")
        description = desc_info.get("description", "")
        platform = desc_info.get("platform", "")
        prof = reg.get("profile")
        if isinstance(prof, dict):
            if not description:
                description = prof.get("description", "") or ""
            if not platform:
                sm = prof.get("session_metadata")
                if isinstance(sm, dict):
                    platform = sm.get("platform", "") or ""
        agents.append({
            "agentId": ad["agent_id"],
            "entriesWritten": ad["entries_written"],
            "entitiesTouched": ad["entities_touched"],
            "lastActive": last_active.isoformat() if last_active else None,
            "createdAt": created.isoformat() if created else None,
            "description": description,
            "platform": platform,
        })
    return {"agents": agents}


@app.get("/api/v1/reuse")
def reuse_summary(
    request: Request,
    days: int = 7,
    limit: int = 10,
    agent: str | None = None,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Reuse over a window: how much, of what, by which agents, and across which.

    The read side of ``amfs_reuse_events``. It exists so that seeing what memory
    did for you does not depend on an agent choosing to mention it in chat — the
    same failure mode as an always-applied rule being ignored. A dashboard panel
    and a weekly digest can both answer from here, days after the session ended.

    Defaults to a week because that is the digest's window; ``days`` is clamped so
    a hand-written URL cannot ask for an unbounded scan.

    Scoped to the agents the caller may see, like ``/stats`` and ``/agents``. RLS
    keeps accounts apart, but within one account a non-admin user sees only some
    agents, and these rows name entity paths, keys and agent ids.

    ``agent`` narrows the window to one agent as READER, for a page about that
    agent. It narrows on top of the visibility scope and never widens it. Doing it
    here rather than letting the caller filter matters: the lists below are cut to
    ``limit`` by reuse volume first, so a caller keeping the rows that name its
    agent would silently lose a quiet agent's reuse and could not tell that apart
    from the agent having none.
    """
    days = max(1, min(int(days or 7), 365))
    limit = max(1, min(int(limit or 10), 100))
    since = datetime.now(UTC) - timedelta(days=days)

    adapter = _get_memory()._adapter
    summarise = getattr(adapter, "reuse_summary", None)
    if summarise is None:
        # A filesystem backend keeps no events. Saying so beats a 500 and beats
        # an empty body that reads as "no reuse happened".
        return {
            "since": since.isoformat(),
            "days": days,
            "available": False,
            "reason": "reuse events need the Postgres adapter",
        }
    agent = agent or None
    kwargs: dict[str, Any] = {
        "since": since,
        "limit": limit,
        "visible_agents": _visible_agent_ids(request),
    }
    if agent is not None:
        # This server carries no dependency on the adapter package — it duck-types
        # whatever backend it was handed — so an adapter predating the parameter is
        # a real deployment, not a hypothetical. Passing the keyword blindly would
        # raise TypeError and take out the whole endpoint, including the
        # account-wide answer that still works. Refusing just the narrowed question
        # keeps a stale pairing degraded rather than broken.
        if "agent" not in inspect.signature(summarise).parameters:
            return {
                "since": since.isoformat(),
                "days": days,
                "available": False,
                "agent": agent,
                "reason": "this backend cannot scope reuse to one agent",
            }
        kwargs["agent"] = agent

    summary = summarise(**kwargs)
    return {
        "since": since.isoformat(),
        "days": days,
        "available": True,
        "agent": agent,
        **summary,
    }


@app.get("/api/v1/agents/{agent_id:path}/memory-graph")
def agent_memory_graph(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """All entries written by or read by this agent, grouped by entity."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    # Acting AS this agent, not as the server. ``search`` and ``list`` keep an
    # entry only if it is shared or the acting identity's own, so asking the
    # server's handle produced a page that could not see the agent's PRIVATE
    # entries at all: an agent whose every entry was private showed 0 memories
    # and 0 topics while its card, counted by SQL with no such filter,
    # correctly said 4 and 1.
    handle = mem.as_agent(agent_id)
    ceiling = max_scan_rows()

    def _scoped(rows: list) -> list:
        # The same per-user filter every other entry-returning route applies,
        # and this one did not: agent ids are matched as bare strings within an
        # account, so where two people share an account and their agents share a
        # default name, each was shown the other's entries.
        if vis is not None and vis.should_filter():
            return vis.filter_entries(rows)
        return rows

    # This agent's own entries, filtered by author in SQL. This used to be
    # ``handle.list()`` — every current entry in the namespace, 400k rows on a
    # large account, 208 s in production — filtered down to the agent's 69 in
    # Python afterwards. Fetched one past the ceiling so the flag is exact,
    # and the flag is read off the RAW rows: the per-user filter below runs
    # after the limit, so where two users' agents share a name, a page at the
    # ceiling may hold rows this caller cannot see in place of ones they can.
    # The response cannot recover those, but it must not call itself complete.
    own_raw, own_truncated = _bounded_scan(
        handle.search(agent_id=agent_id, limit=ceiling + 1), ceiling,
    )
    own_entries = _scoped(own_raw)
    # Read counts and the trace count are aggregated by the adapter; this
    # used to pull up to 10,000 full traces to tally causal_entries here.
    trace_count = mem._adapter.count_traces(agent_id=agent_id)
    read_entities: dict[str, dict[str, int]] = {
        ep: dict(keys) for ep, keys in mem._adapter.trace_read_counts(agent_id).items()
    }

    written_by_agent = [
        e for e in own_entries if not e.entity_path.startswith("_system/")
    ]
    entities_written: dict[str, list[dict]] = {}
    for e in written_by_agent:
        ep = e.entity_path
        if ep not in entities_written:
            entities_written[ep] = []
        entities_written[ep].append({
            "key": e.key,
            "version": e.version,
            "confidence": e.confidence,
            "memoryType": e.memory_type.value if hasattr(e.memory_type, "value") else str(e.memory_type),
            "writtenAt": e.provenance.written_at.isoformat(),
            "recallCount": getattr(e, "recall_count", 0),
        })

    # How many times this agent's own saved knowledge was recalled. This is the
    # reliable reuse signal (recall_count is incremented on every read), unlike
    # timeline read events which are not recorded on every deployment.
    from amfs_core.aggregates import recall_tokens_saved

    total_recalls = sum(getattr(e, "recall_count", 0) for e in written_by_agent)
    recalled_tokens_saved = sum(recall_tokens_saved(e) for e in written_by_agent)

    # Supplement the trace read counts with READ events from the timeline
    # (persisted independently). Bounded by AMFS_MAX_SCAN_ROWS per event
    # type, and the response says when the bound was hit.
    total_read_events = 0
    read_events_truncated = False
    try:
        read_events, cut_a = _bounded_scan(
            mem._adapter.list_events(
                agent_id, mem.namespace, event_type="read", limit=ceiling + 1,
            ),
            ceiling,
        )
        cross_read_events, cut_b = _bounded_scan(
            mem._adapter.list_events(
                agent_id, mem.namespace, event_type="cross_agent_read", limit=ceiling + 1,
            ),
            ceiling,
        )
        read_events_truncated = cut_a or cut_b
        for ev in read_events + cross_read_events:
            ep = ev.details.get("entity_path", "")
            key = ev.details.get("key", "")
            if ep and key:
                if ep not in read_entities:
                    read_entities[ep] = {}
                read_entities[ep][key] = read_entities[ep].get(key, 0) + 1
        total_read_events = len(read_events) + len(cross_read_events)
    except Exception:
        pass

    # Who wrote what this agent read. Only the entity paths it actually read
    # are listed — a few dozen at most, each an indexed lookup — rather than
    # the whole namespace. Bounded so an agent that has read everywhere does
    # not turn this back into the scan it replaces; the flag says when it hit.
    entry_authors: dict[str, str] = {}
    for e in own_entries:
        entry_authors[f"{e.entity_path}/{e.key}"] = e.provenance.agent_id
    read_paths = sorted(read_entities)
    authors_truncated = len(read_paths) > _MEMORY_GRAPH_MAX_READ_PATHS
    for ep in read_paths[:_MEMORY_GRAPH_MAX_READ_PATHS]:
        try:
            for e in _scoped(handle.list(ep)):
                entry_authors.setdefault(f"{ep}/{e.key}", e.provenance.agent_id)
        except Exception:
            logger.debug("memory-graph: could not list %s for authors", ep, exc_info=True)

    cross_agent_reads: dict[str, list[dict]] = {}
    for ep, keys in read_entities.items():
        for key, count in keys.items():
            author = entry_authors.get(f"{ep}/{key}")
            if author and author != agent_id:
                if author not in cross_agent_reads:
                    cross_agent_reads[author] = []
                cross_agent_reads[author].append({
                    "entityPath": ep,
                    "key": key,
                    "readCount": count,
                })

    nodes = []
    for ep in sorted(set(list(entities_written.keys()) + list(read_entities.keys()))):
        nodes.append({
            "entityPath": ep,
            "writtenEntries": entities_written.get(ep, []),
            "readCounts": read_entities.get(ep, {}),
        })

    return {
        "agentId": agent_id,
        "nodes": nodes,
        "traceCount": trace_count,
        "totalWritten": len(written_by_agent),
        "totalReads": total_read_events,
        "totalRecalls": total_recalls,
        "recalledTokensSaved": recalled_tokens_saved,
        "crossAgentReads": cross_agent_reads,
        "truncated": read_events_truncated or own_truncated or authors_truncated,
    }


@app.get("/api/v1/agents/{agent_id:path}/cross-agent-reads")
async def agent_cross_reads(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Which other agents' memory this agent has read."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()

    def _reads_and_authors() -> tuple[dict[str, dict[str, int]], dict[tuple[str, str], str]]:
        # Aggregated by the adapter; previously up to 10,000 full traces were
        # fetched to tally their causal_entries here.
        read_entities = mem._adapter.trace_read_counts(agent_id)
        refs = [(ep, key) for ep, keys in read_entities.items() for key in keys]
        return read_entities, _authors_of(mem, refs)

    read_entities, entry_authors = await _offload(_db_executor, _reads_and_authors)

    cross_reads: dict[str, list[dict[str, Any]]] = {}
    for ep, keys in read_entities.items():
        for key, count in keys.items():
            author = entry_authors.get((ep, key))
            if author and author != agent_id:
                if author not in cross_reads:
                    cross_reads[author] = []
                cross_reads[author].append({
                    "entityPath": ep,
                    "key": key,
                    "readCount": count,
                })

    return {
        "agentId": agent_id,
        "readsFrom": cross_reads,
        "agentsReadFrom": list(cross_reads.keys()),
        "totalCrossAgentReads": sum(
            r["readCount"] for reads in cross_reads.values() for r in reads
        ),
    }


@app.get("/api/v1/agents/{agent_id:path}/recall/{entity_path:path}/{key}")
def agent_recall(
    request: Request,
    agent_id: str,
    entity_path: str,
    key: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Recall an agent's own memory for a key (agent-scoped read).

    Unlike the generic read endpoint, this returns only entries written
    by the specified agent — what that agent's brain actually knows.
    """
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    # Acting as the agent through a per-request handle, not by swapping the
    # shared tagger: these routes run on the threadpool now, and a swap there
    # is a race (``AgentMemory.as_agent``).
    entry = _get_memory().as_agent(agent_id).recall(entity_path, key)
    if entry is None:
        return {"status": "not_found", "agentId": agent_id,
                "entityPath": entity_path, "key": key}
    return _entry_to_response(entry)


@app.get("/api/v1/agents/{agent_id:path}/entries")
def agent_entries(
    request: Request,
    agent_id: str,
    entity_path: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all entries written by a specific agent."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    entries = mem.search(entity_path=entity_path, agent_id=agent_id)
    return {
        "agentId": agent_id,
        "count": len(entries),
        "entries": [_entry_to_response(e) for e in entries],
    }


@app.get("/api/v1/agents/{agent_id:path}/read-from/{source_agent_id}/{entity_path:path}/{key}")
def agent_read_from(
    request: Request,
    agent_id: str,
    source_agent_id: str,
    entity_path: str,
    key: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Read a specific key from another agent's memory (cross-agent read)."""
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(source_agent_id):
        return {"status": "not_found", "sourceAgentId": source_agent_id,
                "entityPath": entity_path, "key": key}

    mem = _get_memory()
    # Route through read_from rather than repeating its search inline. The
    # difference is everything this endpoint exists to record: the two
    # CROSS_AGENT_READ events, the read tracker entry that puts the read in the
    # session's causal chain, and the learned_from edge. Doing the search here
    # returned the same entry while leaving no trace that one agent had learned
    # from another — and that edge is what the authority ranking is built on,
    # so a cross-agent read over HTTP never counted for anything.
    #
    # A per-request handle attributes the read to the caller (see agent_recall).
    entry = mem.as_agent(agent_id).read_from(source_agent_id, entity_path, key)

    if entry is None:
        return {"status": "not_found", "sourceAgentId": source_agent_id,
                "entityPath": entity_path, "key": key}
    return _entry_to_response(entry)


_ACTIVITY_SOURCES = ("w", "o", "e")  # writes, outcomes, events


def _encode_activity_cursor(positions: dict[str, str | None]) -> str | None:
    """A composite cursor over the three activity sources.

    The activity feed merges three independently-ordered streams, so a
    single ``(timestamp, id)`` position cannot resume it exactly: two
    sources can share a timestamp, and dropping to a strict ``<`` on the
    timestamp alone would skip rows. Each source keeps its own keyset
    position instead, and the composite is just the three of them.
    """
    if all(v is None for v in positions.values()):
        return None
    return encode_cursor(datetime.now(UTC), {k: positions.get(k) for k in _ACTIVITY_SOURCES})


def _decode_activity_cursor(cursor: str | None) -> dict[str, str | None]:
    if not cursor:
        return {k: None for k in _ACTIVITY_SOURCES}
    _, positions = decode_cursor(cursor)
    if not isinstance(positions, dict):
        raise InvalidCursorError("cursor does not belong to the activity feed")
    return {k: positions.get(k) for k in _ACTIVITY_SOURCES}


@app.get("/api/v1/agents/{agent_id:path}/activity")
async def agent_activity(
    request: Request,
    agent_id: str,
    limit: int = Query(100, ge=1),
    cursor: str | None = Query(None),
    since: str | None = Query(None),
    until: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Timeline of writes, outcomes, reads, and other events for this agent.

    Each of the three sources — the agent's current entries, its traces, and
    its timeline events — is filtered, ordered and paged by the adapter, so
    this route reads at most ``3 * (limit + 1)`` rows however much history the
    agent has. It used to list every entry in the namespace and filter here.

    Keyset-paginated with a composite cursor: follow ``next_cursor`` while
    ``has_more`` is true. ``since``/``until`` bound every source by time.
    """
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    limit = clamp_limit(limit)
    try:
        positions = _decode_activity_cursor(cursor)
        for pos in positions.values():
            _check_cursor(pos)
    except InvalidCursorError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid cursor: {exc}") from exc
    since_dt = _parse_ts(since)
    until_dt = _parse_ts(until)

    mem = _get_memory()
    entries_page = await _list_agent_entries_page(
        agent_id, limit=limit, cursor=positions["w"], since=since_dt, until=until_dt,
    )
    traces_page = await _list_traces_page(
        agent_id=agent_id, limit=limit, cursor=positions["o"], since=since_dt, until=until_dt,
    )
    events_page = await _list_events_page(
        agent_id, mem.namespace, limit=limit, cursor=positions["e"],
        since=since_dt, until=until_dt,
    )

    # Every candidate row carries the cursor its source would resume from if
    # this row were the last one taken, so the merge can advance each source
    # independently and exactly.
    candidates: list[tuple[datetime, str, dict[str, Any], str]] = []
    for e in entries_page.items:
        candidates.append((
            e.provenance.written_at, "w",
            {
                "type": "write",
                "entityPath": e.entity_path,
                "key": e.key,
                "version": e.version,
                "confidence": e.confidence,
                "timestamp": e.provenance.written_at.isoformat(),
            },
            encode_cursor(e.provenance.written_at, entry_tiebreak(e)),
        ))
    for t in traces_page.items:
        # Traces without an outcome_ref are not shown, but they still
        # advance the trace cursor so paging never stalls on a run of them.
        item = {
            "type": "outcome",
            "outcomeRef": t.outcome_ref,
            "outcomeType": t.outcome_type,
            "causalEntryCount": len(t.causal_entries),
            "timestamp": t.created_at.isoformat(),
        } if t.outcome_ref else None
        candidates.append((t.created_at, "o", item, encode_cursor(t.created_at, t.id)))
    for evt in events_page.items:
        candidates.append((
            evt.created_at, "e",
            {
                "type": (
                    evt.event_type.value
                    if hasattr(evt.event_type, "value")
                    else str(evt.event_type)
                ),
                "summary": evt.summary,
                "details": evt.details,
                "actorAgentId": evt.actor_agent_id,
                "timestamp": evt.created_at.isoformat(),
            },
            encode_cursor(evt.created_at, evt.id),
        ))

    def _sort_ts(ts: datetime) -> datetime:
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    candidates.sort(key=lambda c: (_sort_ts(c[0]), c[1]), reverse=True)

    timeline: list[dict[str, Any]] = []
    next_positions: dict[str, str | None] = dict(positions)
    consumed = 0
    for _, source, item, pos in candidates:
        if len(timeline) >= limit:
            break
        next_positions[source] = pos
        consumed += 1
        if item is not None:
            timeline.append(item)

    leftover = len(candidates) - consumed
    has_more = leftover > 0 or entries_page.has_more or traces_page.has_more or events_page.has_more
    return {
        "agentId": agent_id,
        "timeline": timeline,
        "next_cursor": _encode_activity_cursor(next_positions) if has_more else None,
        "has_more": has_more,
    }


@app.get("/api/v1/agents/{agent_id:path}/timeline")
async def agent_timeline(
    request: Request,
    agent_id: str,
    event_type: str | None = Query(None),
    since: str | None = Query(None),
    until: str | None = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
    cursor: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Git-like event log for an agent — every write, outcome, and
    cross-agent read is recorded as an event on the agent's timeline.

    Keyset-paginated: follow ``next_cursor`` while ``has_more`` is true.
    ``offset`` is honoured only when no cursor is given.
    """
    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter() and not vis.is_agent_visible(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    mem = _get_memory()
    since_dt = _parse_ts(since)
    until_dt = _parse_ts(until)
    limit = clamp_limit(limit)
    page = await _list_events_page(
        agent_id, mem.namespace,
        event_type=event_type,
        since=since_dt, until=until_dt,
        limit=limit, offset=offset, cursor=_check_cursor(cursor),
    )
    return {
        "agentId": agent_id,
        "events": [e.model_dump(mode="json") for e in page.items],
        "count": len(page.items),
        **_page_meta(page),
    }


class LogEventRequest(BaseModel):
    agent_id: str
    event_type: str
    summary: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    actor_agent_id: str | None = None
    branch: str = "main"


@app.post("/api/v1/timeline/events")
def log_timeline_event(
    request: Request,
    body: LogEventRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Log a timeline event for an agent."""
    mem = _get_memory()
    if body.agent_id:
        _link_agent_owner_once(request, body.agent_id, mem.namespace)
    try:
        event_type_enum = EventType(body.event_type)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown event_type: {body.event_type}"},
        )
    event = Event(
        namespace=mem.namespace,
        agent_id=body.agent_id,
        branch=body.branch,
        event_type=event_type_enum,
        summary=body.summary,
        details=body.details,
        actor_agent_id=body.actor_agent_id,
    )
    # WRITE events are authoritatively logged by the write handler
    # (POST /api/v1/entries). HTTP-backed SDK clients ALSO emit a WRITE event
    # here from their background log path, which produced two identical WRITE
    # events per write (the "double-write" bug). The server is the single
    # source of truth for WRITE timeline events, so drop client-originated
    # ones here. This is version-agnostic: it fixes every client regardless of
    # which SDK version they run via `uvx`, without waiting on a PyPI release.
    # We still return a well-formed (but unpersisted) event so clients that
    # parse the response don't error.
    if event_type_enum is EventType.WRITE:
        return event.model_dump(mode="json")
    saved = mem._adapter.log_event(event)
    return saved.model_dump(mode="json")


class UpsertGraphEdgeRequest(BaseModel):
    source_entity: str
    source_type: str = "agent"
    relation: str
    target_entity: str
    target_type: str = "agent"
    provenance: dict[str, Any] = Field(default_factory=dict)
    branch: str = "main"


@app.post("/api/v1/graph/edges")
def upsert_graph_edge_endpoint(
    body: UpsertGraphEdgeRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Create or update a knowledge graph edge."""
    mem = _get_memory()
    edge = GraphEdge(
        source_entity=body.source_entity,
        source_type=body.source_type,
        relation=body.relation,
        target_entity=body.target_entity,
        target_type=body.target_type,
        provenance=body.provenance,
    )
    mem._adapter.upsert_graph_edge(edge, namespace=mem.namespace, branch=body.branch)
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────
# Agent Groups & Enriched Agents
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents/enriched")
def list_agents_enriched(
    request: Request,
    histograms: bool = True,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return enriched agent info with activity histograms.

    Pass histograms=false to skip the per-agent activity histogram queries —
    summary consumers (e.g. the dashboard overview's agent count) only need
    the roster, and the histograms cost up to 100 extra queries per call.
    """
    mem = _get_memory()
    agents = mem._adapter.list_agents_enriched(namespace=mem.namespace)
    allowed = _visible_agent_ids(request)
    if allowed is not None:
        agents = [
            a for a in agents
            if (a.get("agent_id") or a.get("agentId", "")) in allowed
        ]
    if histograms:
        for agent in agents[:100]:
            aid = agent.get("agent_id") or agent.get("agentId", "")
            if aid:
                agent["activity_histogram"] = mem._adapter.get_agent_activity_histogram(
                    aid, days=7, namespace=mem.namespace,
                )
    return {"agents": agents}


@app.get("/api/v1/agent-groups")
def list_agent_groups(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all agent groups."""
    mem = _get_memory()
    groups = mem._adapter.list_agent_groups(namespace=mem.namespace)
    allowed = _visible_agent_ids(request)
    if allowed is not None:
        scoped = []
        for g in groups:
            visible_members = [a for a in g.agent_ids if a in allowed]
            if not visible_members:
                continue
            g = g.model_copy(update={
                "agent_ids": visible_members,
                "member_count": len(visible_members),
            })
            scoped.append(g)
        groups = scoped
    return {"groups": [g.model_dump(mode="json") for g in groups]}


@app.post("/api/v1/agent-groups", status_code=201)
async def create_agent_group_endpoint(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Create a new agent group."""
    body = await request.json()
    name = body.get("name")
    if not name:
        return JSONResponse(
            status_code=400,
            content={"error": "name is required"},
        )
    mem = _get_memory()
    group = AgentGroup(
        namespace=mem.namespace,
        name=name,
        description=body.get("description", ""),
        color=body.get("color"),
        icon=body.get("icon"),
        position=body.get("position", 0.0),
        auto_generated=body.get("autoGenerated", False),
        source_cluster_id=body.get("sourceClusterId"),
    )
    created = mem._adapter.create_agent_group(group, namespace=mem.namespace)
    return created.model_dump(mode="json")


@app.put("/api/v1/agent-groups/{group_id}")
async def update_agent_group_endpoint(
    group_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Update an existing agent group."""
    body = await request.json()
    mem = _get_memory()
    kwargs: dict[str, Any] = {}
    for field in ("name", "description", "color", "icon", "position"):
        if field in body and body[field] is not None:
            kwargs[field] = body[field]
    updated = mem._adapter.update_agent_group(
        group_id, namespace=mem.namespace, **kwargs,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Group not found")
    return updated.model_dump(mode="json")


@app.delete("/api/v1/agent-groups/{group_id}")
def delete_agent_group_endpoint(
    group_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete an agent group."""
    mem = _get_memory()
    deleted = mem._adapter.delete_agent_group(group_id, namespace=mem.namespace)
    return {"deleted": deleted}


@app.post("/api/v1/agent-groups/{group_id}/members")
async def add_group_members(
    group_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Add agents to a group."""
    body = await request.json()
    agent_ids = body.get("agent_ids")
    if not agent_ids or not isinstance(agent_ids, list):
        return JSONResponse(
            status_code=400,
            content={"error": "agent_ids list is required"},
        )
    allowed = _visible_agent_ids(request)
    if allowed is not None and any(a not in allowed for a in agent_ids):
        raise HTTPException(status_code=403, detail="Cannot group agents you do not own")
    mem = _get_memory()
    count = mem._adapter.add_agents_to_group(
        group_id, agent_ids, namespace=mem.namespace,
    )
    return {"added": count}


@app.delete("/api/v1/agent-groups/{group_id}/members")
async def remove_group_members(
    group_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Remove agents from a group."""
    body = await request.json()
    agent_ids = body.get("agent_ids")
    if not agent_ids or not isinstance(agent_ids, list):
        return JSONResponse(
            status_code=400,
            content={"error": "agent_ids list is required"},
        )
    mem = _get_memory()
    count = mem._adapter.remove_agents_from_group(
        group_id, agent_ids, namespace=mem.namespace,
    )
    return {"removed": count}


@app.put("/api/v1/agent-groups/reorder")
async def reorder_agent_groups_endpoint(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Reorder agent groups by setting new positions."""
    body = await request.json()
    positions = body.get("positions")
    if not positions or not isinstance(positions, list):
        return JSONResponse(
            status_code=400,
            content={"error": "positions list is required"},
        )
    tuples = [(p["group_id"], p["position"]) for p in positions]
    mem = _get_memory()
    mem._adapter.reorder_agent_groups(tuples, namespace=mem.namespace)
    return {"ok": True}


@app.get("/api/v1/agent-groups/suggestions")
def agent_group_suggestions(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return cluster-based group suggestions, excluding dismissed ones."""
    from amfs_core.models import DigestType

    mem = _get_memory()
    adapter = mem._adapter
    digest = adapter.get_digest(
        DigestType.AGENT_CLUSTERS,
        f"account:{mem.namespace}",
        namespace=mem.namespace,
    )
    if not digest:
        return {"suggestions": []}

    try:
        dismissed = set(adapter.list_dismissed_cluster_ids(mem.namespace))
    except Exception:
        dismissed = set()

    clusters = digest.summary.get("clusters", [])
    suggestions = [c for c in clusters if c.get("cluster_id") not in dismissed]
    allowed = _visible_agent_ids(request)
    if allowed is not None:
        scoped = []
        for c in suggestions:
            members = [a for a in (c.get("agent_ids") or []) if a in allowed]
            if len(members) < 2:
                continue
            scoped.append({**c, "agent_ids": members})
        suggestions = scoped
    return {"suggestions": suggestions}


@app.post("/api/v1/agent-groups/suggestions/{cluster_id}/accept", status_code=201)
def accept_cluster_suggestion(
    cluster_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Accept a cluster suggestion — create a group from it."""
    from amfs_core.models import DigestType

    mem = _get_memory()
    adapter = mem._adapter
    digest = adapter.get_digest(
        DigestType.AGENT_CLUSTERS,
        f"account:{mem.namespace}",
        namespace=mem.namespace,
    )
    if not digest:
        raise HTTPException(status_code=404, detail="No cluster digest found")

    clusters = digest.summary.get("clusters", [])
    cluster = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    group = AgentGroup(
        namespace=mem.namespace,
        name=cluster.get("suggested_name", cluster_id),
        description=cluster.get("rationale", ""),
        auto_generated=True,
        source_cluster_id=cluster_id,
    )
    created = adapter.create_agent_group(group, namespace=mem.namespace)

    agent_ids = cluster.get("agents", [])
    if agent_ids:
        adapter.add_agents_to_group(created.id, agent_ids, namespace=mem.namespace)

    return created.model_dump(mode="json")


@app.post("/api/v1/agent-groups/suggestions/{cluster_id}/dismiss")
def dismiss_cluster_suggestion_endpoint(
    cluster_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Dismiss a cluster suggestion so it no longer appears."""
    mem = _get_memory()
    mem._adapter.dismiss_cluster_suggestion(cluster_id, mem.namespace)
    return {"dismissed": True}


# The cluster compile walks every agent in the namespace and is pure Python
# CPU — minutes on an account with thousands of agents. The dashboard's Agents
# page POSTs it on every mount, so without a guard two tabs or one reload queue
# a second full compile behind the first while the process is pinned at one
# core. One compile at a time per process, and a compile that finished within
# the window is reused rather than repeated; the caller's ``ok`` is unchanged
# because it only ever waited for the suggestions to be refreshed.
_CLUSTER_RECOMPUTE_MIN_INTERVAL_S = float(
    os.environ.get("AMFS_CLUSTER_RECOMPUTE_MIN_INTERVAL_S", "300"),
)
_cluster_recompute_lock = threading.Lock()
_cluster_recompute_last: dict[str, float] = {}


def _recompute_window_key(request: Request, namespace: str) -> str:
    """Who the reuse window belongs to.

    A multi-tenant deployment runs every account under one namespace and keeps
    them apart with row-level security, so the namespace alone would let the
    first account's compile mark every other account on the instance as fresh.
    The account the compile will actually run under is the one on the RLS
    context — the same ContextVar the Postgres connection reads — so that is
    the key, whichever middleware branch authenticated the request.
    ``request.state.account_id`` is the fallback for a deployment whose
    middleware sets the request state but not the RLS context; a single-tenant
    server has neither and the namespace is the whole story.
    """
    account_id: Any = None
    try:
        from amfs_postgres.tenant_context import get_request_tenant_account_id

        account_id = get_request_tenant_account_id()
    except ImportError:
        pass
    if not account_id:
        account_id = getattr(request.state, "account_id", None)
    return f"{account_id or ''}:{namespace}"


@app.post("/api/v1/agent-groups/recompute")
def recompute_clusters(
    request: Request,
    force: bool = False,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Trigger recomputation of agent clusters.

    Runs on the threadpool (plain ``def``): the compile is CPU-bound Python and
    used to freeze the event loop for its whole duration. ``force=true`` skips
    the reuse window but still waits its turn behind a compile in flight. The
    window is per tenant account; the lock is per process, because the compile
    is what pins the core.
    """
    try:
        from amfs_cortex.compiler import DigestCompiler
        from amfs_postgres.adapter import PostgresAdapter
    except ImportError:
        return JSONResponse(
            status_code=400,
            content={"error": "amfs-cortex package is required for recomputation"},
        )
    mem = _get_memory()
    adapter = mem._adapter
    if not isinstance(adapter, PostgresAdapter):
        return JSONResponse(
            status_code=400,
            content={"error": "Cluster recomputation requires Postgres adapter"},
        )
    ns = mem.namespace
    window_key = _recompute_window_key(request, ns)
    with _cluster_recompute_lock:
        last = _cluster_recompute_last.get(window_key)
        if (
            not force
            and last is not None
            and time.monotonic() - last < _CLUSTER_RECOMPUTE_MIN_INTERVAL_S
        ):
            return {"ok": True, "skipped": "recent"}
        DigestCompiler(adapter=adapter, namespace=ns).compile(f"cluster:account:{ns}")
        _cluster_recompute_last[window_key] = time.monotonic()
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────
# Agent Snapshots
# ──────────────────────────────────────────────────────────────────────


SNAPSHOT_ENTITY = "_system/agent-snapshots"


@app.post("/api/v1/agents/{agent_id:path}/snapshots")
def create_snapshot(
    request: Request,
    agent_id: str,
    req: CreateSnapshotRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Create a named snapshot of an agent's brain state."""
    _require_agent_visible(request, agent_id)
    mem = _get_memory()
    snapshot_id = f"snap-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    snapshot_value = {
        "id": snapshot_id,
        "agent_id": agent_id,
        "name": req.name,
        "description": req.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": req.snapshot_data.get("stats", {}),
        "data": req.snapshot_data,
    }

    mem.as_agent(agent_id).write(
        SNAPSHOT_ENTITY,
        f"{agent_id}/{snapshot_id}",
        snapshot_value,
        confidence=1.0,
    )

    try:
        mem._adapter.log_event(Event(
            namespace=mem.namespace,
            agent_id=agent_id,
            branch="main",
            event_type=EventType.SNAPSHOT_TAKEN,
            summary=f"Snapshot '{req.name}' taken",
            details={
                "snapshot_id": snapshot_id,
                "name": req.name,
                "description": req.description,
                **snapshot_value.get("stats", {}),
            },
        ))
    except Exception:
        logger.debug("Failed to log snapshot event", exc_info=True)

    return {
        "id": snapshot_id,
        "name": req.name,
        "agent_id": agent_id,
        "created_at": snapshot_value["created_at"],
    }


@app.get("/api/v1/agents/{agent_id:path}/snapshots")
def list_snapshots(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List all snapshots for an agent."""
    _require_agent_visible(request, agent_id)
    mem = _get_memory()
    entries = mem.list(entity_path=SNAPSHOT_ENTITY)
    prefix = f"{agent_id}/"
    snapshots = []
    for e in entries:
        if not e.key.startswith(prefix):
            continue
        val = e.value if isinstance(e.value, dict) else {}
        snapshots.append({
            "id": val.get("id", e.key.split("/")[-1]),
            "name": val.get("name", e.key),
            "description": val.get("description", ""),
            "agent_id": agent_id,
            "created_at": val.get("created_at", e.provenance.written_at.isoformat()),
            "stats": val.get("stats", {}),
        })
    snapshots.sort(key=lambda s: s["created_at"], reverse=True)
    return {"snapshots": snapshots, "count": len(snapshots)}


@app.get("/api/v1/agents/{agent_id:path}/snapshots/{snapshot_id}")
def get_snapshot(
    request: Request,
    agent_id: str,
    snapshot_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get the full data for a specific snapshot."""
    _require_agent_visible(request, agent_id)
    mem = _get_memory()
    key = f"{agent_id}/{snapshot_id}"
    entry = mem.recall(SNAPSHOT_ENTITY, key)
    if entry is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    val = entry.value if isinstance(entry.value, dict) else {}
    return val


@app.delete("/api/v1/agents/{agent_id:path}/snapshots/{snapshot_id}")
def delete_snapshot(
    request: Request,
    agent_id: str,
    snapshot_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Delete a snapshot."""
    _require_agent_visible(request, agent_id)
    mem = _get_memory()
    key = f"{agent_id}/{snapshot_id}"
    entry = mem.recall(SNAPSHOT_ENTITY, key)
    if entry is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    mem.write(SNAPSHOT_ENTITY, key, None, confidence=0.0)
    return {"deleted": True, "id": snapshot_id}


@app.post("/api/v1/agents/{agent_id:path}/snapshots/{snapshot_id}/recover")
def recover_snapshot(
    request: Request,
    agent_id: str,
    snapshot_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Recover an agent's memory to the state captured in a snapshot."""
    _require_agent_visible(request, agent_id)
    mem = _get_memory()
    key = f"{agent_id}/{snapshot_id}"
    entry = mem.recall(SNAPSHOT_ENTITY, key)
    if entry is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    val = entry.value if isinstance(entry.value, dict) else {}
    created_at = val.get("created_at")
    if not created_at:
        raise HTTPException(status_code=400, detail="Snapshot missing timestamp")

    timestamp = datetime.fromisoformat(created_at)
    count = mem._adapter.rollback_to_timestamp(
        agent_id, "main", timestamp, mem.namespace,
    )

    try:
        mem._adapter.log_event(Event(
            namespace=mem.namespace,
            agent_id=agent_id,
            branch="main",
            event_type=EventType.SNAPSHOT_RECOVERED,
            summary=f"Recovered from snapshot '{val.get('name', snapshot_id)}'",
            details={
                "snapshot_id": snapshot_id,
                "name": val.get("name", ""),
                "recovered_to": created_at,
                "entries_restored": count,
            },
        ))
    except Exception:
        logger.debug("Failed to log recovery event", exc_info=True)

    return {
        "recovered": True,
        "snapshot_id": snapshot_id,
        "entries_restored": count,
        "recovered_to": created_at,
    }


# ──────────────────────────────────────────────────────────────────────
# Rollback
# ──────────────────────────────────────────────────────────────────────


@app.post("/api/v1/rollback")
def rollback(
    request: Request,
    body: dict[str, Any],
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Rollback an agent's memory to a specific event or timestamp.

    Accepts either ``target_event_id`` (looks up the event to get its
    timestamp and agent) or ``target_timestamp`` + ``agent_id``.
    """
    mem = _get_memory()
    target_event_id = body.get("target_event_id")
    target_timestamp = body.get("target_timestamp")
    agent_id = body.get("agent_id")
    if agent_id:
        _require_agent_visible(request, agent_id)

    if target_event_id:
        event = mem._adapter.get_event(target_event_id, mem.namespace)
        if event is None and agent_id:
            # Fallback for adapters without get_event: a bounded scan of the
            # agent's timeline, newest first, capped by AMFS_MAX_SCAN_ROWS.
            for e in mem._adapter.list_events(agent_id, mem.namespace, limit=max_scan_rows()):
                if e.id == target_event_id:
                    event = e
                    break
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        timestamp = event.created_at
        agent_id = agent_id or event.agent_id
        _require_agent_visible(request, agent_id)
    elif target_timestamp:
        if not agent_id:
            raise HTTPException(
                status_code=400,
                detail="agent_id is required when using target_timestamp",
            )
        timestamp = datetime.fromisoformat(target_timestamp)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide target_event_id or target_timestamp",
        )

    count = mem._adapter.rollback_to_timestamp(
        agent_id, "main", timestamp, mem.namespace,
    )

    try:
        mem._adapter.log_event(Event(
            namespace=mem.namespace,
            agent_id=agent_id,
            branch="main",
            event_type=EventType.ROLLBACK,
            summary=f"Rolled back to {timestamp.isoformat()}",
            details={
                "rolled_back_to": timestamp.isoformat(),
                "entries_restored": count,
                "source_event_id": target_event_id,
            },
        ))
    except Exception:
        logger.debug("Failed to log rollback event", exc_info=True)

    return {
        "entries_restored": count,
        "rolled_back_to": timestamp.isoformat(),
        "agent_id": agent_id,
    }


# ──────────────────────────────────────────────────────────────────────
# Agent-scoped branches & pull requests
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/agents/{agent_id:path}/branches")
def agent_branches(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Branches created or merged by this agent."""
    _require_agent_visible(request, agent_id)
    mem = _get_memory()
    try:
        all_branches = mem.list_branches()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        return {"agentId": agent_id, "branches": [], "count": 0}
    scoped = [
        b for b in all_branches
        if getattr(b, "created_by", None) == agent_id
        or getattr(b, "merged_by", None) == agent_id
    ]
    return {
        "agentId": agent_id,
        "branches": [b.model_dump(mode="json") if hasattr(b, "model_dump") else b for b in scoped],
        "count": len(scoped),
    }


@app.get("/api/v1/agents/{agent_id:path}/pull-requests")
def agent_pull_requests(
    request: Request,
    agent_id: str,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Pull requests created, merged, or closed by this agent."""
    _require_agent_visible(request, agent_id)
    mem = _get_memory()
    try:
        all_prs = mem.list_pull_requests()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        return {"agentId": agent_id, "pullRequests": [], "count": 0}
    scoped = [
        pr for pr in all_prs
        if getattr(pr, "created_by", None) == agent_id
        or getattr(pr, "merged_by", None) == agent_id
        or getattr(pr, "closed_by", None) == agent_id
    ]
    return {
        "agentId": agent_id,
        "pullRequests": [pr.model_dump(mode="json") if hasattr(pr, "model_dump") else pr for pr in scoped],
        "count": len(scoped),
    }


# ──────────────────────────────────────────────────────────────────────
# Pro Branching Plugin (amfs_branching — proprietary)
# ──────────────────────────────────────────────────────────────────────

try:
    from amfs_branching import mount_branching_routes  # type: ignore[import-not-found]
    mount_branching_routes(app, get_memory=_get_memory)
    logger.info("Pro branching routes mounted")
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────
# Control Plane Plugin (amfs_control_plane — proprietary SaaS billing)
# ──────────────────────────────────────────────────────────────────────

try:
    from amfs_control_plane import mount_control_plane  # type: ignore[import-not-found]
    mount_control_plane(app)
    logger.info("Control-plane routes mounted (auth, billing, Stripe webhook)")
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────
# Admin — API Keys
# ──────────────────────────────────────────────────────────────────────


def _get_db_pool():
    """Return the underlying database pool if using the Postgres adapter."""
    mem = _get_memory()
    adapter = mem._adapter
    if hasattr(adapter, "_pool"):
        return adapter._pool
    return None


def _get_namespace() -> str:
    """Return the namespace for the current adapter.

    All admin queries MUST use this to scope data to the correct tenant.
    """
    mem = _get_memory()
    adapter = mem._adapter
    if hasattr(adapter, "_namespace"):
        return adapter._namespace
    return "default"


def _audit_log(
    action: str,
    *,
    resource: str | None = None,
    actor_type: str = "api_key",
    actor_name: str = "api",
    ip_address: str | None = None,
) -> None:
    """Write an entry to the audit log. Silently no-ops without Postgres."""
    pool = _get_db_pool()
    if pool is None:
        return
    ns = _get_namespace()
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO amfs_audit_log
                           (namespace, actor_type, actor_name, action, resource, ip_address)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (ns, actor_type, actor_name, action, resource, ip_address),
                )
    except Exception:
        logger.debug("Failed to write audit log", exc_info=True)


@app.get("/api/v1/admin/api-keys")
def list_api_keys(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"keys": []}
    ns = _get_namespace()
    # Non-admin members only see keys they created themselves (they still
    # need this endpoint for their own setup/settings page).
    restricted_user_id = None
    if _active_visibility_filter(request) is not None:
        restricted_user_id = getattr(request.state, "user_id", None)
        if restricted_user_id is None:
            return {"keys": []}
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if restricted_user_id is not None:
                cur.execute(
                    """SELECT id, name, prefix, key_type, active, scopes,
                              rate_limit_rpm, last_used, created_at, expires_at
                       FROM amfs_api_keys
                       WHERE namespace = %s AND created_by = %s
                       ORDER BY created_at DESC""",
                    (ns, str(restricted_user_id)),
                )
            else:
                cur.execute(
                    """SELECT id, name, prefix, key_type, active, scopes,
                              rate_limit_rpm, last_used, created_at, expires_at
                       FROM amfs_api_keys
                       WHERE namespace = %s
                       ORDER BY created_at DESC""",
                    (ns,),
                )
            rows = cur.fetchall()
    keys = []
    for row in rows:
        scopes = row["scopes"] or []
        if isinstance(scopes, str):
            scopes = json.loads(scopes)
        keys.append({
            "id": str(row["id"]),
            "name": row["name"],
            "prefix": row["prefix"],
            "keyType": row["key_type"],
            "active": row["active"],
            "scopes": scopes,
            "rateLimitRpm": row["rate_limit_rpm"],
            "lastUsed": row["last_used"].isoformat() if row["last_used"] else None,
            "createdAt": row["created_at"].isoformat(),
            "expiresAt": row["expires_at"].isoformat() if row["expires_at"] else None,
        })
    return {"keys": keys}


@app.post("/api/v1/admin/api-keys")
def create_api_key(
    req: CreateAPIKeyRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "API key management requires a Postgres backend"}

    raw_key = f"amfs_{secrets.token_urlsafe(32)}"
    prefix = raw_key[:12]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    ns = _get_namespace()

    user_id = getattr(request.state, "user_id", None)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO amfs_api_keys
                       (namespace, name, key_hash, prefix, key_type, scopes, rate_limit_rpm, expires_at, created_by)
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                   RETURNING id, created_at""",
                (
                    ns,
                    req.name,
                    key_hash,
                    prefix,
                    req.key_type,
                    json.dumps(req.scopes),
                    req.rate_limit_rpm,
                    req.expires_at,
                    str(user_id) if user_id else None,
                ),
            )
            row = cur.fetchone()

    _audit_log(
        "api_key.create",
        resource=req.name,
        ip_address=request.client.host if request.client else None,
    )

    return {
        "id": str(row["id"]),
        "name": req.name,
        "key": raw_key,
        "prefix": prefix,
        "keyType": req.key_type,
        "createdAt": row["created_at"].isoformat(),
    }


@app.delete("/api/v1/admin/api-keys/{key_id}")
def revoke_api_key(
    key_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    pool = _get_db_pool()
    if pool is None:
        return {"error": "API key management requires a Postgres backend"}
    ns = _get_namespace()
    restricted_user_id = None
    if _active_visibility_filter(request) is not None:
        restricted_user_id = getattr(request.state, "user_id", None)
        if restricted_user_id is None:
            return {"error": "Key not found"}
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if restricted_user_id is not None:
                cur.execute(
                    """UPDATE amfs_api_keys SET active = FALSE
                       WHERE id = %s::uuid AND namespace = %s AND created_by = %s
                       RETURNING id, name""",
                    (key_id, ns, str(restricted_user_id)),
                )
            else:
                cur.execute(
                    """UPDATE amfs_api_keys SET active = FALSE
                       WHERE id = %s::uuid AND namespace = %s
                       RETURNING id, name""",
                    (key_id, ns),
                )
            row = cur.fetchone()
    if row is None:
        return {"error": "Key not found"}
    _audit_log(
        "api_key.revoke",
        resource=row.get("name", key_id),
        ip_address=request.client.host if request.client else None,
    )
    return {"revoked": str(row["id"])}


# ──────────────────────────────────────────────────────────────────────
# Admin — Audit Log
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/audit")
def list_audit_log(
    request: Request,
    action: str | None = Query(None),
    limit: int = Query(200),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    # Account-wide audit log is admin-only; non-admin members get an empty
    # list so the settings page renders without exposing other users' actions.
    if _active_visibility_filter(request) is not None:
        return {"entries": []}
    pool = _get_db_pool()
    if pool is None:
        return {"entries": []}

    ns = _get_namespace()
    conditions = ["namespace = %s"]
    params: list[Any] = [ns]

    if action is not None and action != "all":
        conditions.append("action = %s")
        params.append(action)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, actor_type, actor_name, action, resource,
               ip_address, created_at
        FROM amfs_audit_log
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    entries = []
    for row in rows:
        entries.append({
            "id": str(row["id"]),
            "actorType": row["actor_type"],
            "actorName": row["actor_name"],
            "action": row["action"],
            "resource": row["resource"],
            "ipAddress": row["ip_address"],
            "createdAt": row["created_at"].isoformat(),
        })
    return {"entries": entries}


# ──────────────────────────────────────────────────────────────────────
# Patterns — OSS: list pattern_refs used across entries
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/patterns")
def list_patterns(
    request: Request,
    entity_path: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List unique pattern_refs used across memory entries with usage counts."""
    mem = _get_memory()
    entries = mem.list(entity_path)
    vis = _active_visibility_filter(request)
    if vis is not None:
        entries = vis.filter_entries(entries)

    pattern_counts: dict[str, int] = {}
    pattern_entities: dict[str, set[str]] = {}
    for entry in entries:
        for ref in entry.provenance.pattern_refs:
            pattern_counts[ref] = pattern_counts.get(ref, 0) + 1
            if ref not in pattern_entities:
                pattern_entities[ref] = set()
            pattern_entities[ref].add(entry.entity_path)

    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {
        "patterns": [
            {
                "pattern_ref": ref,
                "usage_count": count,
                "entity_paths": sorted(pattern_entities[ref]),
            }
            for ref, count in sorted_patterns
        ],
        "total": len(pattern_counts),
    }


# ──────────────────────────────────────────────────────────────────────
# Admin — Teams (Pro)
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/teams")
def list_teams(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    # Team management is admin-only; non-admin members get an empty list.
    if _active_visibility_filter(request) is not None:
        return {"teams": []}
    pool = _get_db_pool()
    if pool is None:
        return {"teams": []}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.id, t.name, t.slug, t.description,
                          t.created_at, t.updated_at,
                          COUNT(m.id) AS member_count
                   FROM amfs_teams t
                   LEFT JOIN amfs_team_members m
                       ON m.team_id = t.id AND m.removed_at IS NULL
                   WHERE t.namespace = %s
                   GROUP BY t.id
                   ORDER BY t.created_at DESC""",
                (ns,),
            )
            rows = cur.fetchall()
    return {
        "teams": [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "slug": row["slug"],
                "description": row["description"],
                "memberCount": row["member_count"],
                "createdAt": row["created_at"].isoformat(),
                "updatedAt": row["updated_at"].isoformat(),
            }
            for row in rows
        ]
    }


@app.post("/api/v1/admin/teams")
def create_team(
    req: CreateTeamRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO amfs_teams (namespace, name, slug, description)
                   VALUES (%s, %s, %s, %s)
                   RETURNING id, created_at, updated_at""",
                (ns, req.name, req.slug, req.description),
            )
            row = cur.fetchone()
    _audit_log(
        "team.create",
        resource=req.slug,
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "name": req.name,
        "slug": req.slug,
        "description": req.description,
        "createdAt": row["created_at"].isoformat(),
        "updatedAt": row["updated_at"].isoformat(),
    }


@app.patch("/api/v1/admin/teams/{team_id}")
def update_team(
    team_id: str,
    req: UpdateTeamRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}

    updates: list[str] = []
    params: list[Any] = []
    if req.name is not None:
        updates.append("name = %s")
        params.append(req.name)
    if req.description is not None:
        updates.append("description = %s")
        params.append(req.description)

    if not updates:
        return {"error": "No fields to update"}

    updates.append("updated_at = NOW()")
    ns = _get_namespace()
    params.extend([team_id, ns])

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE amfs_teams SET {', '.join(updates)}
                    WHERE id = %s::uuid AND namespace = %s
                    RETURNING id, name, slug, description, created_at, updated_at""",
                params,
            )
            row = cur.fetchone()

    if row is None:
        return {"error": "Team not found"}
    _audit_log(
        "team.update",
        resource=str(row["slug"]),
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "slug": row["slug"],
        "description": row["description"],
        "createdAt": row["created_at"].isoformat(),
        "updatedAt": row["updated_at"].isoformat(),
    }


@app.delete("/api/v1/admin/teams/{team_id}")
def delete_team(
    team_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM amfs_teams
                   WHERE id = %s::uuid AND namespace = %s
                   RETURNING id, slug""",
                (team_id, ns),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Team not found"}
    _audit_log(
        "team.delete",
        resource=row.get("slug", team_id),
        ip_address=request.client.host if request.client else None,
    )
    return {"deleted": str(row["id"])}


# ──────────────────────────────────────────────────────────────────────
# Admin — Team Members (Pro)
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/teams/{team_id}/members")
def list_team_members(
    request: Request,
    team_id: str,
    include_removed: bool = Query(False),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    # Member emails/roles are admin-only.
    if _active_visibility_filter(request) is not None:
        return {"members": []}
    pool = _get_db_pool()
    if pool is None:
        return {"members": []}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if include_removed:
                cur.execute(
                    """SELECT id, email, display_name, role,
                              invited_at, accepted_at, created_at,
                              removed_at, removed_by
                       FROM amfs_team_members
                       WHERE team_id = %s::uuid AND namespace = %s
                       ORDER BY created_at""",
                    (team_id, ns),
                )
            else:
                cur.execute(
                    """SELECT id, email, display_name, role,
                              invited_at, accepted_at, created_at,
                              removed_at, removed_by
                       FROM amfs_team_members
                       WHERE team_id = %s::uuid AND namespace = %s
                         AND removed_at IS NULL
                       ORDER BY created_at""",
                    (team_id, ns),
                )
            rows = cur.fetchall()
    return {
        "members": [
            {
                "id": str(row["id"]),
                "email": row["email"],
                "displayName": row["display_name"],
                "role": row["role"],
                "invitedAt": row["invited_at"].isoformat(),
                "acceptedAt": row["accepted_at"].isoformat() if row["accepted_at"] else None,
                "createdAt": row["created_at"].isoformat(),
                "removedAt": row["removed_at"].isoformat() if row.get("removed_at") else None,
                "removedBy": row.get("removed_by"),
            }
            for row in rows
        ]
    }


@app.post("/api/v1/admin/teams/{team_id}/members")
def add_team_member(
    team_id: str,
    req: AddTeamMemberRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # If this email was previously removed from this team, reinstate instead of duplicating
            cur.execute(
                """SELECT id FROM amfs_team_members
                   WHERE team_id = %s::uuid AND email = %s AND namespace = %s
                     AND removed_at IS NOT NULL""",
                (team_id, req.email, ns),
            )
            existing_removed = cur.fetchone()
            if existing_removed:
                cur.execute(
                    """UPDATE amfs_team_members
                       SET removed_at = NULL, removed_by = NULL,
                           role = %s, display_name = %s
                       WHERE id = %s::uuid
                       RETURNING id, invited_at, created_at""",
                    (req.role, req.display_name, existing_removed["id"]),
                )
                row = cur.fetchone()
            else:
                cur.execute(
                    """INSERT INTO amfs_team_members
                       (namespace, team_id, email, display_name, role)
                       VALUES (%s, %s::uuid, %s, %s, %s)
                       RETURNING id, invited_at, created_at""",
                    (ns, team_id, req.email, req.display_name, req.role),
                )
                row = cur.fetchone()
    _audit_log(
        "team.member.add",
        resource=f"{team_id}/{req.email}",
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "teamId": team_id,
        "email": req.email,
        "displayName": req.display_name,
        "role": req.role,
        "invitedAt": row["invited_at"].isoformat(),
        "createdAt": row["created_at"].isoformat(),
    }


@app.patch("/api/v1/admin/teams/{team_id}/members/{member_id}")
def update_team_member(
    team_id: str,
    member_id: str,
    req: UpdateTeamMemberRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}

    updates: list[str] = []
    params: list[Any] = []
    if req.role is not None:
        updates.append("role = %s")
        params.append(req.role)
    if req.display_name is not None:
        updates.append("display_name = %s")
        params.append(req.display_name)

    if not updates:
        return {"error": "No fields to update"}

    ns = _get_namespace()
    params.extend([member_id, team_id, ns])

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE amfs_team_members SET {', '.join(updates)}
                    WHERE id = %s::uuid AND team_id = %s::uuid AND namespace = %s
                    RETURNING id, email, display_name, role, invited_at, accepted_at, created_at""",
                params,
            )
            row = cur.fetchone()

    if row is None:
        return {"error": "Member not found"}
    _audit_log(
        "team.member.update",
        resource=f"{team_id}/{row['email']}",
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "email": row["email"],
        "displayName": row["display_name"],
        "role": row["role"],
        "invitedAt": row["invited_at"].isoformat(),
        "acceptedAt": row["accepted_at"].isoformat() if row["accepted_at"] else None,
        "createdAt": row["created_at"].isoformat(),
    }


@app.delete("/api/v1/admin/teams/{team_id}/members/{member_id}")
def remove_team_member(
    team_id: str,
    member_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    removed_by = request.headers.get("X-AMFS-Dashboard-Actor", "api")
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE amfs_team_members
                   SET removed_at = NOW(), removed_by = %s
                   WHERE id = %s::uuid AND team_id = %s::uuid
                     AND namespace = %s AND removed_at IS NULL
                   RETURNING id, email""",
                (removed_by, member_id, team_id, ns),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Member not found"}
    _audit_log(
        "team.member.remove",
        resource=f"{team_id}/{row['email']}",
        ip_address=request.client.host if request.client else None,
    )
    return {"deleted": str(row["id"])}


@app.post("/api/v1/admin/teams/{team_id}/members/{member_id}/reinstate")
def reinstate_team_member(
    team_id: str,
    member_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Re-activate a previously removed team member."""
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Team management requires a Postgres backend"}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE amfs_team_members
                   SET removed_at = NULL, removed_by = NULL
                   WHERE id = %s::uuid AND team_id = %s::uuid
                     AND namespace = %s AND removed_at IS NOT NULL
                   RETURNING id, email, display_name, role,
                             invited_at, accepted_at, created_at""",
                (member_id, team_id, ns),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Removed member not found"}
    _audit_log(
        "team.member.reinstate",
        resource=f"{team_id}/{row['email']}",
        ip_address=request.client.host if request.client else None,
    )
    return {
        "id": str(row["id"]),
        "email": row["email"],
        "displayName": row["display_name"],
        "role": row["role"],
        "invitedAt": row["invited_at"].isoformat(),
        "acceptedAt": row["accepted_at"].isoformat() if row["accepted_at"] else None,
        "createdAt": row["created_at"].isoformat(),
        "reinstated": True,
    }


@app.get("/api/v1/admin/members/check-email")
def check_member_email(
    request: Request,
    email: str = Query(...),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Check if an email is associated with any active or removed team memberships.

    Called by the dashboard OAuth flow to determine if a returning user
    should be blocked (removed) or allowed through.

    Returns status: "active", "removed", or "not_found".
    """
    _require_account_admin(request)
    pool = _get_db_pool()
    if pool is None:
        return {"email": email, "status": "unknown", "memberships": []}
    ns = _get_namespace()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT m.id, m.team_id, t.name AS team_name, m.role,
                          m.removed_at, m.removed_by,
                          m.created_at, m.accepted_at
                   FROM amfs_team_members m
                   JOIN amfs_teams t ON t.id = m.team_id
                   WHERE m.email = %s AND m.namespace = %s
                   ORDER BY m.removed_at NULLS FIRST""",
                (email, ns),
            )
            rows = cur.fetchall()
    if not rows:
        return {"email": email, "status": "not_found", "memberships": []}
    active = [r for r in rows if r["removed_at"] is None]
    removed = [r for r in rows if r["removed_at"] is not None]
    if active:
        status = "active"
    elif removed:
        status = "removed"
    else:
        status = "not_found"
    return {
        "email": email,
        "status": status,
        "memberships": [
            {
                "id": str(r["id"]),
                "teamId": str(r["team_id"]),
                "teamName": r["team_name"],
                "role": r["role"],
                "removedAt": r["removed_at"].isoformat() if r["removed_at"] else None,
                "removedBy": r["removed_by"],
                "createdAt": r["created_at"].isoformat(),
                "acceptedAt": r["accepted_at"].isoformat() if r["accepted_at"] else None,
            }
            for r in rows
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Admin — Pattern Detection (Pro)
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/patterns")
def list_detected_patterns(
    pattern_type: str | None = Query(None),
    category: str | None = Query(None),
    severity: str | None = Query(None),
    resolved: bool | None = Query(None),
    agent_id: str | None = Query(None),
    limit: int = Query(100),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List previously detected patterns from the database."""
    try:
        from amfs_patterns.detector import PATTERN_CATEGORIES, PATTERN_METADATA
    except ImportError:
        PATTERN_CATEGORIES = {}
        PATTERN_METADATA = {}

    pool = _get_db_pool()
    if pool is None:
        return {"patterns": [], "categories": PATTERN_CATEGORIES, "metadata": PATTERN_METADATA}

    ns = _get_namespace()
    conditions = ["namespace = %s"]
    params: list[Any] = [ns]

    if pattern_type is not None:
        conditions.append("pattern_type = %s")
        params.append(pattern_type)
    if category is not None:
        conditions.append("category = %s")
        params.append(category)
    if severity is not None:
        conditions.append("severity = %s")
        params.append(severity)
    if resolved is not None:
        conditions.append("resolved = %s")
        params.append(resolved)
    if agent_id is not None:
        conditions.append("(details->>'agent' = %s OR details->>'agent_a' = %s OR details->>'agent_b' = %s)")
        params.extend([agent_id, agent_id, agent_id])

    where = " AND ".join(conditions)
    sql = f"""
        SELECT id, pattern_type, severity, entity_path, description,
               details, resolved, detected_at, resolved_at,
               COALESCE(category, 'collaboration') as category
        FROM amfs_detected_patterns
        WHERE {where}
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
            detected_at DESC
        LIMIT %s
    """
    params.append(limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return {
        "patterns": [
            {
                "id": str(row["id"]),
                "patternType": row["pattern_type"],
                "severity": row["severity"],
                "category": row["category"],
                "entityPath": row["entity_path"],
                "description": row["description"],
                "details": row["details"] or {},
                "resolved": row["resolved"],
                "detectedAt": row["detected_at"].isoformat(),
                "resolvedAt": row["resolved_at"].isoformat() if row["resolved_at"] else None,
            }
            for row in rows
        ],
        "categories": PATTERN_CATEGORIES,
        "metadata": PATTERN_METADATA,
    }


@app.post("/api/v1/admin/patterns/scan")
def run_pattern_scan(
    req: RunPatternDetectionRequest,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run the collaboration-aware pattern detector and persist results."""
    try:
        from amfs_patterns import PatternDetector
        from amfs_patterns.detector import PATTERN_CATEGORIES, PATTERN_METADATA
    except ImportError:
        return {"error": "amfs-patterns package not installed"}

    mem = _get_memory()
    entries = mem.list(req.entity_path)
    # Pattern detection is a whole-history analysis, so it reads a bounded
    # window of outcomes (AMFS_MAX_SCAN_ROWS, newest first) and reports when
    # the window was not the whole history.
    scan_ceiling = max_scan_rows()
    outcomes, outcomes_truncated = _bounded_scan(
        mem._adapter.list_outcomes(entity_path=req.entity_path, limit=scan_ceiling + 1),
        scan_ceiling,
    )

    if req.agent_id:
        entries = [e for e in entries if e.provenance.agent_id == req.agent_id]

    branches: list[Any] = []
    pull_requests: list[Any] = []
    try:
        branches = mem.list_branches()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        pass
    try:
        pull_requests = mem.list_pull_requests()  # type: ignore[attr-defined]
    except (AttributeError, Exception):
        pass

    detector = PatternDetector(
        stale_days=req.stale_days,
        orphan_days=req.orphan_days,
        pr_stale_days=req.pr_stale_days,
        similarity_threshold=req.similarity_threshold,
        incident_threshold=req.incident_threshold,
    )
    report = detector.analyze(
        entries,
        outcome_data=outcomes,
        branches=branches,
        pull_requests=pull_requests,
    )

    pool = _get_db_pool()
    persisted = 0
    if pool is not None:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM amfs_detected_patterns WHERE resolved = FALSE")
                for p in report.patterns:
                    cur.execute(
                        """INSERT INTO amfs_detected_patterns
                               (pattern_type, severity, category, entity_path,
                                description, details, detected_at)
                           VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)""",
                        (
                            p.pattern_type,
                            p.severity,
                            p.category,
                            p.entity_path,
                            p.description,
                            json.dumps(p.details, default=str),
                            p.detected_at,
                        ),
                    )
                    persisted += 1

    _audit_log(
        "patterns.scan",
        resource=req.entity_path or "*",
        ip_address=request.client.host if request.client else None,
    )

    return {
        "scannedEntries": report.scanned_entries,
        "scannedOutcomes": report.scanned_outcomes,
        "outcomesTruncated": outcomes_truncated,
        "scanDurationMs": round(report.scan_duration_ms, 2),
        "patternsFound": len(report.patterns),
        "patternsPersisted": persisted,
        "patterns": [
            {
                "patternType": p.pattern_type,
                "severity": p.severity,
                "category": p.category,
                "entityPath": p.entity_path,
                "description": p.description,
                "details": p.details,
                "detectedAt": p.detected_at.isoformat(),
            }
            for p in report.patterns
        ],
        "categories": PATTERN_CATEGORIES,
        "metadata": PATTERN_METADATA,
    }


@app.patch("/api/v1/admin/patterns/{pattern_id}/resolve")
def resolve_pattern(
    pattern_id: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Mark a detected pattern as resolved."""
    pool = _get_db_pool()
    if pool is None:
        return {"error": "Pattern management requires a Postgres backend"}
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE amfs_detected_patterns
                   SET resolved = TRUE, resolved_at = NOW()
                   WHERE id = %s::uuid
                   RETURNING id, pattern_type, entity_path""",
                (pattern_id,),
            )
            row = cur.fetchone()
    if row is None:
        return {"error": "Pattern not found"}
    _audit_log(
        "patterns.resolve",
        resource=pattern_id,
        ip_address=request.client.host if request.client else None,
    )
    return {"resolved": str(row["id"]), "patternType": row["pattern_type"]}


# ──────────────────────────────────────────────────────────────────────
# Pro — Expertise Heatmap
# ──────────────────────────────────────────────────────────────────────


def _filter_graph_edges(request: Request, edges: list) -> list:
    """Restrict knowledge-graph edges for non-admin members.

    An edge is visible when it is attributable to one of the caller's
    agents (provenance agent_id or an agent-typed endpoint) or lives on a
    shared room entity path. Unattributable edges are hidden by default.
    """
    vis = _active_visibility_filter(request)
    if vis is None:
        return edges
    allowed = _visible_agent_ids(request) or set()
    try:
        room_paths = set(vis.get_room_map().keys())
    except Exception:
        room_paths = set()

    def _edge_visible(e: Any) -> bool:
        prov = getattr(e, "provenance", None) or {}
        if isinstance(prov, dict):
            if prov.get("agent_id") in allowed:
                return True
            ep = prov.get("entity_path")
            if ep and ep in room_paths:
                return True
        if getattr(e, "source_type", None) == "agent" and e.source_entity in allowed:
            return True
        if getattr(e, "target_type", None) == "agent" and e.target_entity in allowed:
            return True
        return False

    return [e for e in edges if _edge_visible(e)]


@app.get("/api/v1/pro/graph/neighbors")
def graph_neighbors(
    request: Request,
    entity: str = Query(...),
    relation: str | None = Query(None),
    direction: str = Query("both"),
    min_confidence: float = Query(0.0),
    depth: int = Query(1, ge=1, le=5),
    limit: int = Query(200, ge=1, le=1000),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Traverse the knowledge graph from an entity."""
    mem = _get_memory()
    try:
        edges = mem.graph_neighbors(
            entity,
            relation=relation,
            direction=direction,
            min_confidence=min_confidence,
            depth=depth,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("graph_neighbors failed for %s: %s", entity, exc)
        return JSONResponse(
            {"entity": entity, "edges": [], "count": 0, "error": str(exc)},
            status_code=200,
        )
    edges = _filter_graph_edges(request, edges)
    return {
        "entity": entity,
        "edges": [e.model_dump(mode="json") for e in edges],
        "count": len(edges),
    }


@app.get("/api/v1/pro/graph/expertise")
async def expertise_graph(
    request: Request,
    agent_id: str | None = Query(None),
    limit_agents: int = Query(30, ge=1, le=200),
    limit_entities: int = Query(30, ge=1, le=200),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Build an agent×entity expertise heatmap.

    Returns a list of agents, entities, and cells with scores derived
    from write counts. When ``agent_id`` is provided, results are scoped
    to that single agent.
    """
    mem = _get_memory()

    # The set of authors the cells may name: the caller's own agents under
    # per-user scoping (an agent's entries are always visible to its owner,
    # so the entry-level pass reduces to this), narrowed to one when asked.
    vis = _active_visibility_filter(request)
    agent_ids: list[str] | None = None
    if vis is not None:
        visible = set(vis.get_user_agents())
        agent_ids = sorted(visible & {agent_id}) if agent_id else sorted(visible)
        if not agent_ids:
            return {"agents": [], "entities": [], "cells": []}
    elif agent_id:
        agent_ids = [agent_id]

    # Write counts per (agent, entity) come from one GROUP BY — the same
    # aggregate the authority ranking uses — instead of every entry in the
    # namespace loaded to be counted. Unscoped, it leaves out the _system/
    # and benchmark rows, as /entities and /stats already do.
    rows = await _offload(_db_executor, mem._adapter.agent_entity_stats, agent_ids=agent_ids)

    agent_entity_weights: dict[str, dict[str, int]] = {}
    agent_totals: dict[str, int] = {}
    entity_totals: dict[str, int] = {}

    for r in rows:
        aid, ep, n = r["agent_id"], r["entity_path"], int(r["entry_count"])
        agent_totals[aid] = agent_totals.get(aid, 0) + n
        entity_totals[ep] = entity_totals.get(ep, 0) + n
        agent_entity_weights.setdefault(aid, {})[ep] = n

    top_agents = [
        a for a, _ in sorted(agent_totals.items(), key=lambda x: x[1], reverse=True)
    ][:limit_agents]
    top_agent_set = set(top_agents)

    relevant_entities: set[str] = set()
    for aid in top_agent_set:
        relevant_entities.update(agent_entity_weights.get(aid, {}).keys())
    top_entities = [
        ep
        for ep, _ in sorted(
            [(ep, entity_totals.get(ep, 0)) for ep in relevant_entities],
            key=lambda x: x[1],
            reverse=True,
        )
    ][:limit_entities]
    top_entity_set = set(top_entities)

    cells: list[dict[str, Any]] = []
    for aid in top_agents:
        for ep, weight in agent_entity_weights.get(aid, {}).items():
            if ep in top_entity_set:
                cells.append({
                    "agent": aid,
                    "entity": ep,
                    "score": weight,
                    "relations": ["writes"],
                })

    return {
        "agents": top_agents,
        "entities": top_entities,
        "cells": cells,
    }


@app.post("/api/v1/pro/graph/backfill")
def graph_backfill(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Backfill knowledge graph edges from existing entries and outcomes.

    Materializes edges from: pattern_refs → 'references' edges,
    outcome causal chains → 'informed' + 'read' edges,
    agent writes → 'wrote' edges.
    """
    mem = _get_memory()
    entries = mem.list()
    try:
        outcomes = mem._adapter.list_outcomes() if hasattr(mem._adapter, "list_outcomes") else []
    except Exception:
        outcomes = []

    created = 0
    errors = 0

    for e in entries:
        aid = e.provenance.agent_id
        ep = e.entity_path
        ek = f"{ep}/{e.key}"

        try:
            mem._adapter.upsert_graph_edge(
                GraphEdge(
                    source_entity=aid,
                    source_type="agent",
                    relation="wrote",
                    target_entity=ek,
                    target_type="entry",
                    confidence=e.confidence,
                    provenance={"agent_id": aid, "trigger": "backfill"},
                ),
                namespace=mem.namespace,
                branch=e.branch or "main",
            )
            created += 1
        except Exception:
            errors += 1

        for ref in (e.provenance.pattern_refs or []):
            try:
                mem._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=ek,
                        source_type="entry",
                        relation="references",
                        target_entity=ref,
                        target_type="entry",
                        provenance={"agent_id": aid, "trigger": "backfill"},
                    ),
                    namespace=mem.namespace,
                    branch=e.branch or "main",
                )
                created += 1
            except Exception:
                errors += 1

    for o in outcomes:
        otype = getattr(o, "outcome_type", None)
        edge_conf = 1.0 if otype and otype.value == "success" else 0.7
        aid = getattr(o, "agent_id", "unknown")
        for ek in getattr(o, "causal_entry_keys", []):
            try:
                mem._adapter.upsert_graph_edge(
                    GraphEdge(
                        source_entity=ek,
                        source_type="entry",
                        relation="informed",
                        target_entity=o.outcome_ref,
                        target_type="outcome",
                        confidence=edge_conf,
                        provenance={"agent_id": aid, "trigger": "backfill"},
                    ),
                    namespace=mem.namespace,
                    branch="main",
                )
                created += 1
            except Exception:
                errors += 1

    return {"edges_created": created, "errors": errors, "entries_scanned": len(entries), "outcomes_scanned": len(outcomes)}


# ──────────────────────────────────────────────────────────────────────
# Pro — HMO Memory Tiers
# ──────────────────────────────────────────────────────────────────────


def _compute_tiers(entries: list) -> tuple[dict[str, int], dict[str, float]]:
    """Score entries and assign HMO tiers (Hot/Warm/Archive)."""
    from amfs_core.tiering import PriorityScorer, TierAssigner

    scorer = PriorityScorer()
    assigner = TierAssigner()
    return assigner.assign_with_scores(entries, scorer)


def _tiered_entries(
    mem: AgentMemory, vis: Any | None, agent_id: str | None
) -> tuple[list[MemoryEntry], tuple[dict[str, int], dict[str, float]]]:
    """The entries a tiers page scores, and their tiers and scores.

    The tiering is a Python scorer over the whole visible set, so the set has
    to be loaded; this loads only the named agent's rows when there is one
    and runs the load, the visibility pass and the scoring together off the
    event loop. Synchronous: call it via ``_offload``.
    """
    entries = _entries_by_agent(mem, agent_id) if agent_id else mem.list()
    if vis is not None:
        entries = vis.filter_entries(entries)
    return entries, _compute_tiers(entries)


@app.get("/api/v1/pro/tiers/distribution")
async def tiers_distribution(
    request: Request,
    agent_id: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return HMO tier distribution (Hot / Warm / Archive)."""
    entries, (tiers, scores) = await _offload(
        _db_executor, _tiered_entries, _get_memory(), _active_visibility_filter(request), agent_id
    )

    hot = warm = archive = scored_count = 0
    score_sum = 0.0
    for key, tier in tiers.items():
        if tier == 1:
            hot += 1
        elif tier == 2:
            warm += 1
        else:
            archive += 1
        s = scores.get(key)
        if s is not None:
            scored_count += 1
            score_sum += s

    return {
        "total": len(entries),
        "hot": hot,
        "warm": warm,
        "archive": archive,
        "scored": scored_count,
        "avg_priority_score": round(score_sum / scored_count, 6) if scored_count else None,
    }


@app.get("/api/v1/pro/tiers/entries")
async def tiers_entries(
    request: Request,
    tier: int = Query(..., ge=1, le=3),
    limit: int = Query(50, ge=1, le=500),
    agent_id: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return entries for a given HMO tier as a flat array."""
    entries, (tier_map, score_map) = await _offload(
        _db_executor, _tiered_entries, _get_memory(), _active_visibility_filter(request), agent_id
    )

    filtered = [e for e in entries if tier_map.get(e.entry_key) == tier]
    filtered.sort(key=lambda e: score_map.get(e.entry_key, 0.0), reverse=True)

    return [
        {
            "entity_path": e.entity_path,
            "key": e.key,
            "confidence": e.confidence,
            "tier": tier,
            "priority_score": round(score_map.get(e.entry_key, 0.0), 6),
            "recall_count": getattr(e, "recall_count", 0),
            "importance_score": getattr(e, "importance_score", None),
            "written_at": e.provenance.written_at.isoformat(),
            "agent_id": e.provenance.agent_id,
        }
        for e in filtered[:limit]
    ]


# ──────────────────────────────────────────────────────────────────────
# SSE Stream
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/stream")
async def stream(
    request: Request,
    entity_path: str = Query("*"),
    _auth: str | None = Depends(verify_api_key),
) -> EventSourceResponse:
    # Non-admin members only receive write events for entries they are
    # allowed to see; everything else is dropped before it hits the wire.
    vis = _active_visibility_filter(request)
    predicate = vis.is_entry_visible if vis is not None else None
    return EventSourceResponse(
        _sse_manager.event_generator(entity_path, predicate=predicate)
    )


# ──────────────────────────────────────────────────────────────────────
# System Config
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/admin/config")
def get_system_config(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return system configuration and active Pro module status."""
    mem = _get_memory()
    adapter_type = type(mem._adapter).__name__

    pro_modules: dict[str, bool] = {}
    for mod_name in ("amfs_traces", "amfs_pro_api", "amfs_critic", "amfs_distiller",
                     "amfs_retrieval", "amfs_ml", "amfs_safety", "amfs_extraction",
                     "amfs_pro_connectors"):
        try:
            __import__(mod_name)
            pro_modules[mod_name] = True
        except ImportError:
            pro_modules[mod_name] = False

    return {
        "adapter": adapter_type,
        "namespace": getattr(mem, "_namespace", os.environ.get("AMFS_NAMESPACE", "default")),
        "agent_id": mem.agent_id,
        "session_id": mem.session_id,
        "postgres_configured": bool(os.environ.get("AMFS_POSTGRES_DSN")),
        "llm_configured": bool(os.environ.get("AMFS_LLM_API_KEY")),
        "extraction_enabled": os.environ.get("AMFS_AUTO_EXTRACT", "").lower() == "true",
        "otel_enabled": bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")),
        "pro_modules": pro_modules,
    }


# ──────────────────────────────────────────────────────────────────────
# Connectors / Webhook Ingestion
# ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/connectors")
def list_connectors(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List available and installed connectors."""
    try:
        from amfs_connectors import ConnectorRegistry

        registry = ConnectorRegistry()
        return {
            "connectors": registry.list_installed(),
            "total": len(registry.list_available()),
        }
    except ImportError:
        return {"connectors": [], "total": 0}


# ──────────────────────────────────────────────────────────────────────
# Memory Cortex (Briefing + Status)
# ──────────────────────────────────────────────────────────────────────


def _is_agent_visible_for_entity(
    agent_id: str,
    entity_path: str,
    user_agents: set[str],
    room_map: dict[str, set[str]],
) -> bool:
    """Check if an agent is visible in the context of a specific entity_path.

    Mirrors the logic of UserVisibilityFilter.is_entry_visible but works
    with raw agent_id + entity_path instead of requiring a full entry object.
    """
    if agent_id in user_agents:
        return True
    room_members = room_map.get(entity_path)
    if room_members:
        if agent_id in room_members:
            return True
        if agent_id in ("amfs-server", "system", "amfs"):
            return True
    return False


def _filter_briefing_digests(vis: Any, digests: list) -> list:
    """Filter briefing digests and their hot_context through visibility.

    Ensures briefing only surfaces data the user could also access via
    amfs_read / amfs_search, maintaining consistency across all endpoints.
    """
    user_agents = vis.get_user_agents()
    room_map = vis.get_room_map()
    filtered = []

    for digest in digests:
        scope = digest.scope
        source_agents = digest.source_agents

        if "hot_context" in digest.summary:
            digest.summary["hot_context"] = [
                h for h in digest.summary["hot_context"]
                if _is_agent_visible_for_entity(
                    h.get("agent", ""), scope, user_agents, room_map,
                )
            ]

        # who_to_ask names agents by id and tells the caller to read from them.
        # Unfiltered, it would both recommend a read that read_from then denies
        # and disclose that an agent the caller cannot see exists — so this is
        # a tenant-isolation control, not presentation.
        if "who_to_ask" in digest.summary:
            visible_authors = [
                w for w in digest.summary["who_to_ask"]
                if _is_agent_visible_for_entity(
                    w.get("agent_id", ""), scope, user_agents, room_map,
                )
            ]
            if visible_authors:
                digest.summary["who_to_ask"] = visible_authors
            else:
                digest.summary.pop("who_to_ask")

        # schema_profile / materialized_aggregates are computed over EVERY entry
        # in the entity, so their sums, ranges and enum values can disclose data
        # authored by agents this caller cannot see — the same data amfs_read /
        # amfs_search / amfs_aggregate would refuse them. The digest is compiled
        # once and served to callers of differing visibility, so it cannot be
        # re-scoped per caller here; keep the rollups only when the caller can
        # actually reach everything they summarise. A room member reaches every
        # entry on the topic; otherwise every source agent must be visible.
        if "schema_profile" in digest.summary or "materialized_aggregates" in digest.summary:
            room_ok = scope in room_map
            # webhook/{source} and external/{source} are ingestion sources, not
            # agents whose memory the caller might be barred from — the digest
            # expands every external writer into both. They can never satisfy
            # _is_agent_visible_for_entity, so counting them meant any entity that
            # ingested a single external event failed the all()-visible check and
            # lost its rollups, even though the caller can already read those
            # entries via amfs_search / amfs_aggregate. Gate on the real agents
            # only; the synthetic sources neither grant nor deny visibility.
            real_agents = [
                a for a in source_agents
                if not a.startswith(("webhook/", "external/"))
            ]
            all_agents_visible = bool(real_agents) and all(
                _is_agent_visible_for_entity(a, scope, user_agents, room_map)
                for a in real_agents
            )
            if not (room_ok or all_agents_visible):
                digest.summary.pop("schema_profile", None)
                digest.summary.pop("materialized_aggregates", None)

        if source_agents:
            has_visible = any(
                _is_agent_visible_for_entity(a, scope, user_agents, room_map)
                for a in source_agents
            )
            if not has_visible:
                continue
        else:
            has_room_access = scope in room_map
            has_visible_hot = bool(digest.summary.get("hot_context"))
            # A surviving who_to_ask names only agents already cleared above,
            # so keeping the digest for its sake discloses nothing further —
            # and dropping it would throw away the routing this caller is
            # allowed to act on, which is the whole point of the block.
            has_visible_ask = bool(digest.summary.get("who_to_ask"))
            if not has_room_access and not has_visible_hot and not has_visible_ask:
                continue

        filtered.append(digest)

    return filtered


async def _credit_briefing_reuse(
    response: Response | None,
    request: Request,
    digests: list[Any],
    *,
    branch: str = "main",
) -> None:
    """Book reuse for the memories a briefing hands over verbatim.

    A briefing reported ``recall_count`` without ever incrementing it, and the
    agent that follows the documented workflow — brief first, then work — was
    the one penalised for it: every entry it was briefed on stayed at zero
    forever, it saw no value line for a memory that had just done its job, and
    the write-only ratio counted briefed knowledge as never read.

    Only ``hot_context`` is credited, because that is the part of a digest whose
    entry text is passed through as-is. The compiled narrative and key facts are
    a synthesis *about* entries; the agent reads those, not them.

    Capped at ``REUSE_CREDIT_K``, the same cap retrieve uses, for the same
    reason and with a sharper precedent behind it: crediting every entry on a
    read path once inflated a nine-entry topic to 395 recalls, because a
    dashboard page view counted as reuse of everything on it. So this credits
    what the agent most likely acted on rather than everything it was shown, and
    is deliberately conservative — an entry surfaced further down a briefing
    still books nothing.
    """
    surfaced: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for d in digests:
        summary = getattr(d, "summary", None) or {}
        if not isinstance(summary, dict):
            continue
        for item in summary.get("hot_context") or []:
            if not isinstance(item, dict):
                continue
            ref = (item.get("entity_path"), item.get("key"))
            if ref[0] and ref[1] and ref not in seen:
                seen.add(ref)
                surfaced.append((str(ref[0]), str(ref[1])))

    credited_entry: MemoryEntry | None = None
    credited_hits = 0
    for entity_path, key in surfaced[:REUSE_CREDIT_K]:
        # Read before the bump, and at the adapter rather than through the
        # engine. Before, so the block reports the recall count as it stood
        # *prior* to this reuse, the way every other credited read does. At the
        # adapter, because engine.read would itself count as a recall and the
        # lookup that reports a read must not be one — the distinction amfs#257
        # was opened to fix.
        entry = None
        try:
            if _async_adapter is not None:
                entry = await _async_adapter.read(entity_path, key, branch=branch)
            else:
                entry = await _offload(
                    _db_executor, _get_memory()._adapter.read, entity_path, key, branch=branch
                )
        except Exception:  # noqa: BLE001 - the value block is reporting, not the answer
            logger.debug("briefing reuse lookup failed", exc_info=True)

        try:
            if _async_adapter is not None:
                await _async_adapter.increment_recall_count(entity_path, key, branch=branch)
            else:
                _get_memory()._adapter.increment_recall_count(entity_path, key, branch=branch)
        except Exception:  # noqa: BLE001 - reuse accounting is best-effort
            logger.debug("briefing recall bump failed", exc_info=True)
            continue

        credited_hits += 1
        if credited_entry is None:
            credited_entry = entry

    # Same call, same place, as every other credited read: bumping recall_count
    # without this wrote the count but no amfs_reuse_events row and no
    # X-SenseLab-Value header, so the agent that briefed first still saw no
    # value line and the event table drifted out of step with the counter.
    _attach_reuse_value(
        response,
        request,
        credited=credited_entry,
        hits=credited_hits,
        surface="briefing",
        branch=branch,
    )


@app.get("/api/v1/briefing")
async def get_briefing(
    request: Request,
    entity_path: str | None = Query(None),
    agent_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    credit_reuse: bool = Query(False),
    compact: bool = Query(False),
    since: datetime | None = Query(None),
    branch: str | None = Query(None),
    env_model: str | None = Query(None),
    env_agent_version: str | None = Query(None),
    env_runtime: str | None = Query(None),
    env_platform: str | None = Query(None),
    # See retrieve_entries: injected on the type, defaulted so the handler stays
    # callable in-process without one.
    response: Response = None,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get a ranked briefing of compiled knowledge digests.

    *compact* returns only the lead entity digest with its hot context and
    evidence sections, narrative trimmed. *since* trims the list sections to
    what changed after that moment — the delta since the last briefing.
    *branch* reads the hot context and evidence sections from that memory
    branch instead of ``main`` — how a repair branch or a canary is briefed
    before it is merged. The ``env_*`` parameters describe the asking run;
    with them each procedure in the lead digest is marked applicable or not
    against its environment preconditions.

    *credit_reuse* books the briefing as a real read of the knowledge it
    surfaces. It is off by default and has to be asked for, because the same
    endpoint serves an agent about to act on a briefing and a dashboard panel
    rendering one for a human to look at — and only the caller can tell those
    apart. Defaulting it on would make every page view count as reuse, which is
    the shape of a bug this codebase has already had.
    """
    mem = _get_memory()
    briefing_kwargs: dict[str, Any] = {
        "entity_path": entity_path,
        "agent_id": agent_id,
        "limit": limit,
        "compact": compact,
    }
    if since is not None:
        briefing_kwargs["since"] = since
    environment = {
        k: v for k, v in (
            ("model", env_model), ("agent_version", env_agent_version),
            ("runtime", env_runtime), ("platform", env_platform),
        ) if v
    }
    if environment:
        briefing_kwargs["environment"] = environment
    # Resolved through the routing hook like every other read, then passed only
    # when it is not main so a server-side ``briefing`` that predates the
    # keyword keeps working (``mem.briefing`` here is always current, but the
    # in-process callers in Pro compose this handler with their own memory).
    branch = _effective_branch(request, branch)
    if branch != "main":
        briefing_kwargs["branch"] = branch
    digests = mem.briefing(**briefing_kwargs)

    vis = _get_visibility_filter(request)
    if vis is not None and vis.should_filter():
        digests = _filter_briefing_digests(vis, digests)

    # After visibility filtering, never before: an entry the caller may not see
    # must not be credited to them either.
    if credit_reuse:
        await _credit_briefing_reuse(response, request, digests, branch=branch)

    return {
        "digests": [d.model_dump(mode="json") for d in digests],
        "total": len(digests),
    }


@app.get("/api/v1/cortex/status")
def cortex_status(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get Cortex worker status and digest statistics."""
    mem = _get_memory()
    try:
        digests = mem.briefing(limit=0)
    except Exception:
        digests = []

    try:
        from amfs_postgres.adapter import PostgresAdapter

        adapter = mem._adapter
        if isinstance(adapter, PostgresAdapter):
            all_digests = adapter.list_digests()

            vis = _get_visibility_filter(request)
            if vis is not None and vis.should_filter():
                all_digests = _filter_briefing_digests(vis, all_digests)

            return {
                "status": "active" if all_digests else "idle",
                "digest_count": len(all_digests),
                "digest_types": {
                    "entity": sum(1 for d in all_digests if d.digest_type.value == "entity"),
                    "agent_brief": sum(1 for d in all_digests if d.digest_type.value == "agent_brief"),
                    "source": sum(1 for d in all_digests if d.digest_type.value == "source"),
                },
            }
    except ImportError:
        pass
    except Exception:
        logger.exception("Failed to fetch Cortex status")

    return {"status": "unavailable", "digest_count": 0}


@app.get("/api/v1/cortex/digests")
def list_cortex_digests(
    request: Request,
    digest_type: str | None = Query(None),
    scope: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List compiled digests, optionally filtered by type and scope."""
    try:
        from amfs_postgres.adapter import PostgresAdapter
        from amfs_core.models import DigestType

        mem = _get_memory()
        adapter = mem._adapter
        if isinstance(adapter, PostgresAdapter):
            dt = DigestType(digest_type) if digest_type else None
            namespace = getattr(adapter, "_namespace", "default")
            digests = adapter.list_digests(digest_type=dt, namespace=namespace)
            if scope:
                digests = [d for d in digests if d.scope == scope]

            vis = _get_visibility_filter(request)
            if vis is not None and vis.should_filter():
                digests = _filter_briefing_digests(vis, digests)

            return {
                "digests": [d.model_dump(mode="json") for d in digests],
                "total": len(digests),
            }
    except (ImportError, ValueError):
        pass
    except Exception:
        logger.exception("Failed to list cortex digests")

    return {"digests": [], "total": 0}


@app.get("/api/v1/cortex/activity")
def cortex_activity(
    request: Request,
    limit: int = Query(default=50, le=200),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get recent Cortex compilation and event activity."""
    # The worker activity log spans the whole account — admin-only view.
    if _active_visibility_filter(request) is not None:
        return {"events": [], "total": 0, "throughput": [], "stats": None}
    if _cortex_worker:
        log = _cortex_worker.activity_log
        recent = log[-limit:] if len(log) > limit else log
        return {
            "events": list(reversed(recent)),
            "total": len(log),
            "throughput": _cortex_worker.throughput,
            "stats": _cortex_worker.stats,
        }
    return {"events": [], "total": 0, "throughput": [], "stats": None}


@app.post("/api/v1/cortex/recompile")
def cortex_recompile(
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Trigger a full digest recompilation."""
    if not _cortex_worker:
        raise HTTPException(status_code=503, detail="Cortex worker not running")
    count = _cortex_worker._compiler.recompile_all()
    return {"recompiled": count}


# ------------------------------------------------------------------
# Consolidation (Cortex compaction) endpoints
# ------------------------------------------------------------------


@app.post("/api/v1/cortex/consolidate")
def run_consolidation(
    request: Request,
    body: dict[str, Any] | None = None,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Run Tier A (auto-safe) consolidation.

    Optionally accepts ``entity_path`` to consolidate a single entity.
    """
    from amfs_cortex.consolidator import ConsolidationStrategy

    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem.namespace
    branch = (body or {}).get("branch", "main")
    entity_path = (body or {}).get("entity_path")

    visible_paths = _visible_entity_paths(request)
    if visible_paths is not None:
        # Restricted members may only consolidate entities they can see;
        # account-wide consolidation is admin-only.
        if not entity_path or entity_path not in visible_paths:
            raise HTTPException(
                status_code=403,
                detail="Consolidation requires a visible entity_path",
            )

    strategy = ConsolidationStrategy(adapter, namespace=namespace)
    if entity_path:
        report = strategy.run_entity(entity_path, branch=branch)
    else:
        report = strategy.run(branch=branch)

    return report.model_dump()


@app.get("/api/v1/cortex/consolidation/candidates")
def list_consolidation_candidates(
    request: Request,
    entity_path: str = Query(...),
    branch: str = Query("main"),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List Tier B consolidation candidates (proposals) for an entity."""
    from amfs_cortex.consolidator import ConsolidationStrategy

    visible_paths = _visible_entity_paths(request)
    if visible_paths is not None and entity_path not in visible_paths:
        return {"entity_path": entity_path, "proposals": []}

    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem.namespace

    strategy = ConsolidationStrategy(adapter, namespace=namespace)
    proposals = strategy.find_consolidation_candidates(entity_path, branch=branch)

    return {
        "entity_path": entity_path,
        "proposals": [p.model_dump() for p in proposals],
    }


@app.get("/api/v1/cortex/consolidation/proposals")
def list_consolidation_proposals(
    request: Request,
    entity_path: str | None = Query(None),
    status: str | None = Query(None),
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """List consolidation proposals from persisted branches.

    Scans branches matching ``cortex/consolidation/`` and returns
    structured proposal metadata extracted from each branch's diff.
    """
    visible_paths = _visible_entity_paths(request)
    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem.namespace

    try:
        branches = adapter.list_branches(namespace=namespace, status="active" if status != "all" else None)
    except Exception:
        return {"proposals": [], "total": 0}

    consolidation_branches = [
        b for b in branches if b.name.startswith("cortex/consolidation/")
    ]
    if entity_path:
        consolidation_branches = [
            b for b in consolidation_branches
            if b.name.startswith(f"cortex/consolidation/{entity_path}/")
        ]

    if status and status in ("approved", "rejected"):
        branch_status = "merged" if status == "approved" else "closed"
        branches_all = adapter.list_branches(namespace=namespace, status=branch_status)
        extra = [b for b in branches_all if b.name.startswith("cortex/consolidation/")]
        if entity_path:
            extra = [b for b in extra if b.name.startswith(f"cortex/consolidation/{entity_path}/")]
        consolidation_branches.extend(extra)

    all_proposals = []
    for b in consolidation_branches:
        parts = b.name.split("/")
        ep = "/".join(parts[2:-1]) if len(parts) > 3 else parts[2] if len(parts) >= 3 else "unknown"

        if visible_paths is not None and ep not in visible_paths:
            continue

        branch_status_map = {"active": "pending", "merged": "approved", "closed": "rejected"}
        prop_status = branch_status_map.get(b.status.value if hasattr(b.status, "value") else str(b.status), "pending")

        if status and status != "all" and prop_status != status:
            continue

        try:
            diff = adapter.diff_branch(b.name, namespace=namespace)
            entry_keys = [f"{d.entity_path}/{d.key}" for d in diff.entries] if hasattr(diff, "entries") else []
        except Exception:
            diff = None
            entry_keys = []

        proposed_value = None
        proposed_confidence = 0.0
        if diff and hasattr(diff, "entries") and diff.entries:
            first = diff.entries[0]
            proposed_value = first.branch_value if hasattr(first, "branch_value") else None
            proposed_confidence = 0.8

        all_proposals.append({
            "id": b.id or b.name,
            "entity_path": ep,
            "branch_name": b.name,
            "strategy": b.description.split(":")[0].strip().lower().replace(" ", "_") if b.description and ":" in b.description else "consolidation",
            "risk_tier": "review_required",
            "source_entry_keys": entry_keys,
            "proposed_value": proposed_value,
            "proposed_confidence": proposed_confidence,
            "compression_ratio": max(len(entry_keys), 1),
            "rationale": b.description or "Consolidation proposal",
            "status": prop_status,
            "created_at": (b.created_at or b.branched_at).isoformat() if (b.created_at or b.branched_at) else None,
            "reviewed_by": b.merged_by,
            "reviewed_at": b.merged_at.isoformat() if b.merged_at else None,
        })

    return {
        "proposals": all_proposals,
        "total": len(all_proposals),
    }


@app.get("/api/v1/cortex/consolidation/status")
def consolidation_status(
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Get consolidation health metrics for the dashboard."""
    mem = _get_memory()
    adapter = mem._adapter
    namespace = mem.namespace
    vis = _active_visibility_filter(request)

    consolidation_runs = 0
    total_auto_archived = 0
    # Worker counters are account-global — only exposed to admins.
    if _cortex_worker and vis is None:
        consolidation_runs = _cortex_worker._consolidation_runs
        for entry in _cortex_worker._activity_log:
            if entry.get("type") == "consolidation_run":
                total_auto_archived += entry.get("auto_archived", 0)

    try:
        branches = adapter.list_branches(namespace=namespace, status="active")
        pending_branches = [
            b for b in branches if b.name.startswith("cortex/consolidation/")
        ]
        if vis is not None:
            visible_paths = _visible_entity_paths(request) or set()
            pending_branches = [
                b for b in pending_branches
                if "/".join(b.name.split("/")[2:-1]) in visible_paths
            ]
    except Exception:
        pending_branches = []

    entities_ready: list[str] = []
    try:
        entries = adapter.list()
        if vis is not None:
            entries = vis.filter_entries(entries)
        entity_counts: dict[str, int] = {}
        for e in entries:
            entity_counts[e.entity_path] = entity_counts.get(e.entity_path, 0) + 1
        for ep, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            if count >= 10:
                entities_ready.append(ep)
            if len(entities_ready) >= 10:
                break
    except Exception:
        pass

    return {
        "consolidation_runs": consolidation_runs,
        "auto_archived": total_auto_archived,
        "pending_proposals": len(pending_branches),
        "entities_ready": entities_ready,
    }


@app.post("/api/v1/webhooks/{connector_name}")
async def ingest_webhook(
    connector_name: str,
    request: Request,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Receive and process a webhook event through the connector framework."""
    try:
        from amfs_connectors import WebhookIngester, WebhookConfig
    except ImportError:
        raise HTTPException(status_code=501, detail="Connector framework not installed")

    body = await request.body()
    headers = dict(request.headers)

    mem = _get_memory()

    secret_env = f"AMFS_CONNECTOR_{connector_name.upper().replace('-', '_')}_SECRET"
    secret = os.environ.get(secret_env)

    config = WebhookConfig(
        name=connector_name,
        connector_type="webhook",
        entity_path=connector_name,
        secret=secret,
    )
    ingester = WebhookIngester(config, memory=mem)

    try:
        from amfs_connectors import ConnectorRegistry

        registry = ConnectorRegistry()
        connector = registry.get(connector_name)
        if connector:
            ingester.register_transform("*", connector.transform)
    except Exception:
        pass

    event_type = headers.get("x-event-type", "generic")
    event_id = headers.get("x-event-id")

    results = ingester.ingest(
        body,
        headers,
        source=connector_name,
        event_type=event_type,
        event_id=event_id,
    )

    persisted = 0
    as_webhook = mem.as_agent(f"webhook/{connector_name}")
    for r in results:
        if r.success and r.action == "write":
            entry = as_webhook.write(
                r.entity_path,
                r.key,
                r.details,
                confidence=1.0,
                memory_type=MemoryType.EXPERIENCE,
            )
            _sse_manager.broadcast(entry)
            persisted += 1

    return {
        "results": [r.model_dump(mode="json") for r in results],
        "total": len(results),
        "persisted": persisted,
    }


@app.post("/api/v1/events")
def ingest_event(
    body: EventRequest,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Ingest an event directly into the shared memory pool.

    Simple alternative to the webhook/connector framework for apps that
    just want to push context into AMFS.
    """
    mem = _get_memory()

    entry = mem.as_agent(f"external/{body.source}").write(
        body.entity_path,
        body.key,
        body.value,
        confidence=1.0,
        memory_type=MemoryType.EXPERIENCE,
    )
    _sse_manager.broadcast(entry)

    return {
        "status": "ok",
        "entity_path": body.entity_path,
        "key": body.key,
        "source": body.source,
        "agent_id": f"external/{body.source}",
    }


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

# With more than one worker, uvicorn runs a supervisor that pings every worker
# twice a second and SIGKILLs any that has not answered within
# ``timeout_worker_healthcheck`` — 5 seconds by default — then spawns a fresh
# one and logs the same "Child process [pid] died" it would for a real crash.
# The pong comes from a thread the worker starts on entry, so it needs the GIL,
# and a fresh worker does not have one to give: spawn re-imports this module
# and its C extensions and the lifespan loads the embedding models, all of it
# CPU-bound, all of it competing with the worker next door that is serving
# traffic. On a 2-vCPU Cloud Run instance that reliably took longer than 5
# seconds, so one slot per instance was killed every 8 seconds forever and
# never once logged "Started server process" (2026-09-21, 36 kills a minute
# across the fleet, every request that landed on the dying slot truncated).
# Two minutes is generous room to start and rides out any GIL hold a request
# could plausibly cause, while a worker that is genuinely wedged still gets
# replaced. A single worker has no supervisor and is unaffected either way.
DEFAULT_WORKER_HEALTHCHECK_S = 120


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amfs-http",
        description="AMFS HTTP/REST API server with SSE support",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AMFS_HTTP_HOST", "0.0.0.0"),
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.environ.get("AMFS_HTTP_PORT", "8741")),
        help="Port to bind (default: 8741)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--with-cortex",
        action="store_true",
        default=False,
        help="Run embedded Cortex worker in-process (for single-instance deployments)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("AMFS_HTTP_WORKERS", "1")),
        help="Number of uvicorn worker processes (default: 1, env: AMFS_HTTP_WORKERS)",
    )
    parser.add_argument(
        "--worker-healthcheck-timeout",
        type=int,
        default=int(os.environ.get("AMFS_HTTP_WORKER_HEALTHCHECK_S", str(DEFAULT_WORKER_HEALTHCHECK_S))),
        help=(
            "Seconds a worker may go without answering the supervisor's ping before "
            "it is killed and replaced; only meaningful with --workers > 1 "
            f"(default: {DEFAULT_WORKER_HEALTHCHECK_S}, env: AMFS_HTTP_WORKER_HEALTHCHECK_S)"
        ),
    )
    return parser.parse_args()


_cortex_worker = None


def _make_tenant_provider(dsn: str):
    """Build a tenant_provider callback for the Cortex worker.

    Returns a callable that queries the ``accounts`` table (Pro/SaaS only)
    for all tenant UUIDs.  Falls back to ``[None]`` for OSS deployments
    that don't have the ``accounts`` table.
    """

    def _provider() -> list:
        try:
            import psycopg

            with psycopg.connect(dsn, autocommit=True) as conn:
                rows = conn.execute(
                    "SELECT id::text FROM accounts"
                ).fetchall()
                if rows:
                    return [r[0] for r in rows]
        except Exception:
            pass
        return [None]

    return _provider


def _start_embedded_cortex() -> None:
    """Start the embedded Cortex worker in this process, if it can be.

    Called from the app lifespan, so it runs once per serving process — see
    ``_lifespan`` for why not from ``main()``. Idempotent: a second call in the
    same process (a test re-entering the lifespan) is a no-op.
    """
    global _cortex_worker
    if _cortex_worker is not None:
        return
    dsn = os.environ.get("AMFS_POSTGRES_DSN")
    if not dsn:
        logger.warning("AMFS_WITH_CORTEX requires AMFS_POSTGRES_DSN")
        return

    try:
        from amfs_postgres.adapter import PostgresAdapter
        from amfs_cortex.compiler import DigestCompiler
        from amfs_cortex.worker import CortexWorker

        namespace = os.environ.get("AMFS_NAMESPACE", "default")
        # One background thread compiling digests on a timer, sharing a
        # process with the request path. Sized for that rather than left
        # on the pool default, which would have it hold as many
        # connections as the endpoints do while using one at a time —
        # multiplied by every process the service scales out to.
        adapter = PostgresAdapter(
            dsn=dsn, namespace=namespace, min_pool_size=1, max_pool_size=2
        )

        strategies = []
        try:
            from amfs_cortex_pro import get_pro_strategies
            strategies = get_pro_strategies()
            logger.info("Pro compilation strategies loaded")
        except ImportError:
            pass

        compiler = DigestCompiler(
            adapter=adapter,
            strategies=strategies or None,
            namespace=namespace,
        )
        tenant_provider = _make_tenant_provider(dsn)
        # The catch-up scan runs in every serving process (no advisory lock),
        # so its interval sets the fleet-wide scan rate: 36 processes at the
        # 300 s default is one tenant-wide scan every ~8 s. A deployment that
        # runs several workers per instance should raise this in proportion
        # to keep the rate where it was. The scan is a GROUP BY since the
        # adapter grew list_scopes(), but the knob stays: 0 disables it on
        # deployments where the event path is trusted to compile every scope.
        catchup_s = float(os.environ.get("AMFS_CORTEX_CATCHUP_INTERVAL_S", "300"))
        worker = CortexWorker(
            dsn=dsn,
            compiler=compiler,
            use_advisory_lock=False,
            catchup_interval_s=catchup_s,
            tenant_provider=tenant_provider,
        )

        try:
            from amfs_cortex_pro import get_outcome_wiring, HotContextTracker
            wiring = get_outcome_wiring(adapter, namespace)
            if wiring:
                worker._outcome_wiring = wiring
                logger.info("Outcome wiring attached to embedded Cortex worker")
            tracker = HotContextTracker()
            worker._hot_tracker = tracker
            logger.info("Hot context tracker attached to embedded Cortex worker")
        except ImportError:
            pass

        from amfs_http.pro_proxy import create_forwarder
        forwarder = create_forwarder()
        if forwarder:
            worker._pro_forwarder = forwarder

        t = threading.Thread(target=worker.run, daemon=True, name="cortex-embedded")
        t.start()
        _cortex_worker = worker
        logger.info("Embedded Cortex worker started (pid=%d)", os.getpid())
    except ImportError:
        logger.warning("AMFS_WITH_CORTEX requires amfs-cortex package")
    except Exception:
        logger.exception("Failed to start embedded Cortex worker — server will run without Cortex")


def main() -> None:
    """Run the AMFS HTTP server via uvicorn."""
    args = _parse_args()

    # The flag becomes an environment variable rather than a start here, so
    # that every uvicorn worker process — which inherits the environment but
    # not this function's locals — starts its own Cortex from the lifespan.
    # Setting the variable directly (docker-compose, a systemd unit) works the
    # same without the flag.
    if args.with_cortex:
        os.environ["AMFS_WITH_CORTEX"] = "1"

    workers = args.workers
    if args.reload and workers > 1:
        logger.warning("--reload is incompatible with --workers > 1; forcing workers=1")
        workers = 1

    logger.info(
        "Starting AMFS HTTP server on %s:%d (workers=%d, worker healthcheck=%ds)",
        args.host, args.port, workers, args.worker_healthcheck_timeout,
    )
    uvicorn.run(
        "amfs_http.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        timeout_worker_healthcheck=args.worker_healthcheck_timeout,
    )


if __name__ == "__main__":
    main()
