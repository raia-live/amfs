"""Ingestion-latency probe for the performance comparison.

Measures two things per memory system, with NO LLM calls:
  write_latency_ms      - wall time of the write/seed call itself.
  time_to_searchable_s  - wall time from issuing the write until the fact is
                          actually returned by a query (the number that bites a
                          write-then-read agent on asynchronous platforms).

Local systems (vector store, SenseLab filesystem) are synchronous, so the two
are essentially equal. Hosted platforms ingest asynchronously, so
time-to-searchable is dominated by their indexing pipeline.
"""

from __future__ import annotations

import json
import statistics as stats
import time
import uuid
from pathlib import Path

from pro_benchmark import (
    DATASET, HERE, FastEmbedEmbedder, VectorStore, Mem0Backend,
    SupermemoryBackend, AmfsProBackend,
)

N_FACTS = 6
POLL_S = 0.5
TIMEOUT_S = 40.0


def probe(backend, is_local: bool) -> dict:
    writes, tts = [], []
    for fact, _, _ in DATASET[:N_FACTS]:
        scope = f"ingest-{backend.name}-{uuid.uuid4().hex[:6]}"
        t0 = time.time()
        backend.seed(scope, fact)
        writes.append((time.time() - t0) * 1000.0)

        # query with the fact text so a match is guaranteed once indexed
        found_at = None
        deadline = t0 + TIMEOUT_S
        while time.time() < deadline:
            try:
                if is_local and hasattr(backend, "retrieve_gated"):
                    hit = bool(backend.retrieve_gated(scope, fact)[0])
                else:
                    hit = bool(backend.retrieve(scope, fact))
            except Exception:
                hit = False
            if hit:
                found_at = time.time() - t0
                break
            if is_local:
                # synchronous: one miss means it will never appear; avoid busy wait
                found_at = time.time() - t0
                break
            time.sleep(POLL_S)
        tts.append(found_at if found_at is not None else TIMEOUT_S)
        print(f"  {backend.name}: write {writes[-1]:.1f} ms, searchable {tts[-1]:.2f} s")

    return {
        "write_latency_ms_p50": round(stats.median(writes), 1),
        "write_latency_ms_max": round(max(writes), 1),
        "time_to_searchable_s_p50": round(stats.median(tts), 2),
        "time_to_searchable_s_max": round(max(tts), 2),
        "n": len(tts),
    }


def main() -> None:
    emb = FastEmbedEmbedder()
    out = {}
    print("vector...");      out["vector"] = probe(VectorStore(emb), is_local=True)
    print("amfs-pro...");    out["amfs-pro"] = probe(AmfsProBackend(emb, HERE / "pro_audit.jsonl"), is_local=True)
    print("mem0...");        out["mem0"] = probe(Mem0Backend(), is_local=False)
    print("supermemory..."); out["supermemory"] = probe(SupermemoryBackend(), is_local=False)

    Path(HERE / "ingestion.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
