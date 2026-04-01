"""Semantic Search — find memories by meaning, not just exact keys.

Shows how to plug in any embedding model (OpenAI, Cohere, sentence-transformers)
via the EmbedderABC interface.

    uv run python examples/semantic_search.py
"""

from __future__ import annotations

import hashlib
from typing import Any

from amfs import AgentMemory, EmbedderABC


class DemoEmbedder(EmbedderABC):
    """Toy embedder that produces deterministic pseudo-embeddings.

    In production, replace with a real embedder::

        class OpenAIEmbedder(EmbedderABC):
            def embed(self, text: str) -> list[float]:
                resp = openai.embeddings.create(input=text, model="text-embedding-3-small")
                return resp.data[0].embedding
    """

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255.0 for b in h[:32]]


def main() -> None:
    embedder = DemoEmbedder()

    with AgentMemory(agent_id="search-agent", embedder=embedder) as mem:
        # Write entries — embeddings are auto-computed on write
        mem.write("checkout", "retry-pattern",
                  {"max_retries": 3, "strategy": "exponential backoff"})
        mem.write("checkout", "timeout-handling",
                  {"connect_timeout": 3000, "read_timeout": 10000})
        mem.write("payment", "circuit-breaker",
                  {"threshold": 5, "description": "trips after 5 failures"})
        mem.write("auth", "rate-limiting",
                  {"rpm": 100, "strategy": "sliding window"})

        # Semantic search — find by meaning
        print("Searching for: 'error recovery patterns'")
        results = mem.semantic_search("error recovery patterns", limit=3)
        for entry, similarity in results:
            print(f"  {entry.entry_key} (similarity={similarity:.4f})")
            print(f"    value={entry.value}")
        print()

        print("Searching for: 'request limits'")
        results = mem.semantic_search("request limits", limit=3)
        for entry, similarity in results:
            print(f"  {entry.entry_key} (similarity={similarity:.4f})")

    print()
    print("NOTE: This uses a toy embedder for demo purposes.")
    print("For real semantic search, plug in OpenAI, Cohere, or")
    print("sentence-transformers via the EmbedderABC interface.")


if __name__ == "__main__":
    main()
