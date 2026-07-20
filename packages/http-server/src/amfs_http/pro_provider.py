"""Optional Pro retrieval/sealing provider for the OpenRouter proxy.

The OSS proxy retrieves memory with the plain SDK ``search`` (full-text / list).
When the Pro ``amfs_retrieval`` package is installed, this module upgrades that to
the ``MultiStrategyRetriever`` (adaptive-weighted RRF + optional cross-encoder
rerank over a real default embedder), which is what wins the precision/adversarial
benchmarks. When ``amfs_traces`` is installed and sealing is enabled, write-backs
can be sealed into a signed ``ImmutableDecisionTrace``.

Everything here is soft/optional and fail-open:
- OSS-only deployments (no Pro packages) get ``None`` and the proxy falls back to
  SDK search — no error, no behaviour change.
- Any construction/among failure degrades to the OSS path.

Env:
- AMFS_PROXY_PRO_RETRIEVAL   "false" to force the OSS search path (default true)
- AMFS_PROXY_PRO_RERANK      "false" to skip reranking entirely (default true)
- AMFS_LLM_RERANK            "true" to use the LLM listwise reranker (Pro tier) —
                             the lever that reaches >90% adversarial Hit@1. Costs
                             one LLM call per retrieval, so it is opt-in. Needs
                             AMFS_LLM_RERANK_BASE_URL/API_KEY (e.g. OpenRouter).
                             When unset/unavailable, falls back to the offline
                             cross-encoder.

NOTE(integration-test): the Pro retrieval path requires the amfs-internal packages
and a populated adapter; it is exercised in the amfs-internal retrieval eval, not
in OSS unit CI. The unit test here only covers graceful absence + value extraction.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PRO_RETRIEVAL = os.environ.get("AMFS_PROXY_PRO_RETRIEVAL", "true").lower() == "true"
PRO_RERANK = os.environ.get("AMFS_PROXY_PRO_RERANK", "true").lower() == "true"

_cache: dict[str, Any] = {}


def _build_reranker() -> Any | None:
    """Build the strongest available reranker.

    Prefers the LLM listwise reranker (Pro tier) when ``AMFS_LLM_RERANK=true`` and
    an OpenAI-compatible client/key are configured — it resolves confusable facts
    under adversarial paraphrase and is the lever for top-1 accuracy, at the cost
    of one LLM call per retrieval. Otherwise falls back to the offline
    ``CrossEncoderReranker`` (single-digit ms, zero egress). Returns ``None`` only
    if neither is importable.
    """
    try:
        from amfs_retrieval import LLMReranker

        llm = LLMReranker()  # reads AMFS_LLM_RERANK* env
        if llm.available:
            logger.info("pro_provider: LLM listwise reranker enabled for proxy")
            return llm
    except Exception:  # noqa: BLE001 - LLM rerank optional
        logger.debug("pro_provider: LLM reranker unavailable", exc_info=True)
    try:
        from amfs_retrieval import CrossEncoderReranker

        return CrossEncoderReranker()
    except Exception:  # noqa: BLE001 - rerank optional
        logger.debug("pro_provider: reranker unavailable", exc_info=True)
        return None


def build_pro_retriever(adapter: Any) -> Any | None:
    """Build a Pro MultiStrategyRetriever over ``adapter``, or None if unavailable.

    Cached on the adapter identity so the embedder/reranker are loaded once.
    """
    if not PRO_RETRIEVAL or adapter is None:
        return None
    cache_key = f"retriever:{id(adapter)}"
    if cache_key in _cache:
        return _cache[cache_key]
    try:
        from amfs_retrieval import MultiStrategyRetriever, create_pro_embedder

        reranker = _build_reranker() if PRO_RERANK else None

        retriever = MultiStrategyRetriever(
            adapter, embedder=create_pro_embedder(), reranker=reranker
        )
        _cache[cache_key] = retriever
        logger.info("pro_provider: Pro MultiStrategyRetriever enabled for proxy")
        return retriever
    except Exception:  # noqa: BLE001 - Pro package absent or failed to build
        logger.debug("pro_provider: Pro retrieval unavailable", exc_info=True)
        _cache[cache_key] = None
        return None


def _result_value(result: Any) -> Any:
    """Extract the stored value from a RetrievalResult-like object or dict."""
    entry = getattr(result, "entry", None)
    if entry is None and isinstance(result, dict):
        entry = result.get("entry", result)
    if isinstance(entry, dict):
        return entry.get("value")
    return getattr(entry, "value", None)


def pro_retrieve_values(
    retriever: Any, scope: str | None, query: str, limit: int
) -> list[Any] | None:
    """Run Pro retrieval and return raw stored values (fail-open -> None)."""
    if retriever is None or not query:
        return None
    try:
        results = retriever.retrieve(query, entity_path=scope, limit=limit)
    except Exception:  # noqa: BLE001
        logger.debug("pro_provider: retrieve failed", exc_info=True)
        return None
    values = [v for v in (_result_value(r) for r in results or []) if v is not None]
    return values or None
