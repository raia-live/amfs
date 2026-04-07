"""Tests for Gap C: search pipeline, embedding version, semantic recall scoring."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from amfs_core.embedder import EmbedderABC, cosine_similarity
from amfs_core.models import RecallConfig, ScoredEntry, SearchQuery
from amfs.memory import AgentMemory
from amfs_filesystem.adapter import FilesystemAdapter


class DeterministicEmbedder(EmbedderABC):
    """Hash-based embedder that produces distinct vectors for distinct inputs."""

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255.0 for b in h[:16]]


def _make_memory(tmp_path: Path, embedder: EmbedderABC | None = None) -> AgentMemory:
    adapter = FilesystemAdapter(root=tmp_path / ".amfs")
    return AgentMemory(
        agent_id="test-agent",
        adapter=adapter,
        embedder=embedder,
    )


class TestSearchQueryModel:
    def test_query_field_present(self):
        sq = SearchQuery(query="hello world")
        assert sq.query == "hello world"

    def test_query_field_defaults_none(self):
        sq = SearchQuery()
        assert sq.query is None

    def test_query_round_trips(self):
        sq = SearchQuery(query="test", entity_path="svc/a", min_confidence=0.5)
        dumped = sq.model_dump()
        restored = SearchQuery(**dumped)
        assert restored.query == "test"
        assert restored.entity_path == "svc/a"


class TestEmbeddingVersion:
    def test_embedding_write_preserves_version(self, tmp_path: Path):
        """Embedding update must not reset version to 1."""
        embedder = DeterministicEmbedder()
        mem = _make_memory(tmp_path, embedder=embedder)

        first = mem.write("svc/auth", "pattern-retry", "retry with backoff")
        assert first.version == 1

        second = mem.write("svc/auth", "pattern-retry", "retry with jittered backoff")
        assert second.version == 2

        current = mem.read("svc/auth", "pattern-retry")
        assert current is not None
        assert current.version == 2
        assert current.embedding is not None


class TestRecallCosineScoring:
    def test_semantic_scores_vary_with_similarity(self, tmp_path: Path):
        """Semantic breakdown must not be constant 1.0 for different entries."""
        embedder = DeterministicEmbedder()
        mem = _make_memory(tmp_path, embedder=embedder)

        mem.write("svc/auth", "task-login", "implement OAuth2 login flow")
        mem.write("svc/auth", "task-cache", "add Redis cache layer for sessions")
        mem.write("svc/payments", "task-checkout", "stripe checkout integration")

        results = mem.search(
            query="OAuth authentication",
            recall_config=RecallConfig(),
            limit=10,
        )

        assert len(results) == 3
        assert all(isinstance(r, ScoredEntry) for r in results)

        semantic_scores = [r.breakdown["semantic"] for r in results]
        assert len(set(round(s, 6) for s in semantic_scores)) > 1, (
            f"Semantic scores should vary but got {semantic_scores}"
        )

    def test_no_embedder_semantic_score_zero(self, tmp_path: Path):
        """Without embedder, semantic component must be 0.0."""
        mem = _make_memory(tmp_path, embedder=None)
        mem.write("svc/a", "k1", "value one")

        results = mem.search(
            query="value",
            recall_config=RecallConfig(),
            limit=10,
        )

        assert len(results) == 1
        assert results[0].breakdown["semantic"] == 0.0

    def test_query_threads_to_search_query(self, tmp_path: Path):
        """The query parameter must be passed into SearchQuery for adapters."""
        mem = _make_memory(tmp_path)
        mem.write("svc/a", "k1", "hello world")

        results = mem.search(query="hello", limit=10)
        assert len(results) >= 1


class TestRecallConfigDefaults:
    def test_no_embedder_still_ranks_by_recency_and_confidence(self, tmp_path: Path):
        mem = _make_memory(tmp_path, embedder=None)

        mem.write("svc/a", "old", "old entry", confidence=0.5)
        mem.write("svc/a", "new", "new entry", confidence=0.9)

        results = mem.search(
            query="entry",
            recall_config=RecallConfig(),
            limit=10,
        )

        assert len(results) == 2
        assert results[0].entry.key == "new"
        for r in results:
            assert r.breakdown["semantic"] == 0.0
            assert r.breakdown["recency"] > 0
            assert r.breakdown["confidence"] > 0
