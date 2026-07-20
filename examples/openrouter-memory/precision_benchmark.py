"""Retrieval precision under contention + adversarial (paraphrased) robustness.

Unlike the recall study (one fact per scope), here ALL facts share one scope, so
the retriever must discriminate the right fact from confusable siblings (five
database engines, five on-call people, four regions, four ports, ...). We ask
each fact two ways:

  direct       - names the entity, close to stored wording.
  adversarial  - paraphrased, no lexical overlap with the stored fact or answer.

Metric (no reader LLM; this measures the retriever itself): for each query, is the
correct fact in the retrieved top-k?
  hit@1  - the top-ranked item is the correct fact.
  hit@3  - the correct fact is anywhere in the top 3.
  MRR    - mean reciprocal rank of the correct fact within the top 3.

"Correct fact retrieved" = a retrieved item contains all of the fact's distinctive
answer tokens (case-insensitive), which are unique across the dataset.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from pro_benchmark import (
    HERE, FastEmbedEmbedder, VectorStore, Mem0Backend, SupermemoryBackend, AmfsProBackend,
)

# id, fact, direct question, adversarial (paraphrased) question, distinctive tokens
DATASET = [
    ("db-billing", "The billing service runs on PostgreSQL 15.",
     "What database engine does the billing service use?",
     "For the component that issues invoices and charges cards, which datastore technology are we standardized on?",
     ["postgresql"]),
    ("db-analytics", "The analytics service runs on ClickHouse 23.8.",
     "Which database backs the analytics service?",
     "Our reporting and aggregation workloads sit on top of which columnar store?",
     ["clickhouse"]),
    ("db-auth", "The auth service runs on MySQL 8.0.",
     "What database does the auth service use?",
     "Sign-in and identity records are persisted in which relational engine?",
     ["mysql"]),
    ("db-search", "The search service runs on Elasticsearch 8.11.",
     "Which engine powers the search service?",
     "Full-text lookups across the catalog are served from which cluster technology?",
     ["elasticsearch"]),
    ("db-cache", "The cache tier runs on Redis 7.2.",
     "What technology is the cache tier built on?",
     "Ephemeral hot key-value data is held in which in-memory store?",
     ["redis"]),

    ("oncall-payments", "Dana Okafor is the on-call engineer for the payments team.",
     "Who is on call for the payments team?",
     "If the charge pipeline pages someone tonight, which engineer picks it up?",
     ["okafor"]),
    ("oncall-infra", "Miguel Santos is the on-call engineer for the infra team.",
     "Who is on call for the infra team?",
     "When the clusters wake somebody at 3am, who answers for platform?",
     ["santos"]),
    ("oncall-mobile", "Priya Nair is the on-call engineer for the mobile team.",
     "Who is on call for the mobile team?",
     "Which engineer owns alerts coming from the iOS and Android apps?",
     ["nair"]),
    ("oncall-data", "Tomas Berg is the on-call engineer for the data team.",
     "Who is on call for the data team?",
     "If the nightly ETL jobs fail, who is the designated responder?",
     ["berg"]),
    ("oncall-security", "Aisha Khan is the on-call engineer for the security team.",
     "Who is on call for the security team?",
     "Which engineer responds first to intrusion and breach alerts?",
     ["khan"]),

    ("region-prod", "Production runs in us-east-1.",
     "Which region is production in?",
     "Where is our live customer-facing traffic hosted geographically?",
     ["us-east-1"]),
    ("region-staging", "Staging runs in eu-west-1.",
     "Which region is staging in?",
     "Our pre-release environment is pinned to which cloud locale?",
     ["eu-west-1"]),
    ("region-dr", "Disaster recovery runs in us-west-2.",
     "Which region hosts disaster recovery?",
     "If the primary site is lost, failover traffic is directed to which region?",
     ["us-west-2"]),
    ("region-dev", "The dev sandbox runs in ap-south-1.",
     "Which region is the dev sandbox in?",
     "Throwaway engineering experiments are provisioned in which locale?",
     ["ap-south-1"]),

    ("port-gateway", "The API gateway listens on port 8443.",
     "What port does the API gateway use?",
     "Inbound TLS for the edge entrypoint terminates on which port number?",
     ["8443"]),
    ("port-metrics", "The metrics endpoint listens on port 9090.",
     "What port serves metrics?",
     "The Prometheus scrape target is exposed on which port?",
     ["9090"]),
    ("port-admin", "The admin console listens on port 8081.",
     "What port is the admin console on?",
     "The back-office operator UI is bound to which port?",
     ["8081"]),
    ("port-grpc", "The gRPC service listens on port 50051.",
     "What port does the gRPC service use?",
     "Internal binary RPC traffic is accepted on which port?",
     ["50051"]),

    ("sla-sev1", "SEV-1 incidents must be acknowledged within 15 minutes.",
     "How fast must a SEV-1 be acknowledged?",
     "For the highest-severity outage, what is the maximum time before someone must claim it?",
     ["15", "minute"]),
    ("sla-sev2", "SEV-2 incidents must be acknowledged within 60 minutes.",
     "How fast must a SEV-2 be acknowledged?",
     "For a moderate-severity incident, how long before it must be picked up?",
     ["60", "minute"]),
    ("rotate-keys", "Production API keys are rotated every 90 days.",
     "How often are production API keys rotated?",
     "What is the maximum lifetime of a live API credential before replacement?",
     ["90", "day"]),
    ("backup-retention", "Database backups are retained for 35 days.",
     "How long are database backups kept?",
     "How far back in time can we restore data from cold storage?",
     ["35", "day"]),

    ("vendor-cdn", "Static assets are served through Fastly.",
     "Which CDN do we use?",
     "Our edge caching for images and scripts is provided by which vendor?",
     ["fastly"]),
    ("vendor-payments", "Card payments are processed through Stripe.",
     "Which payment processor do we use?",
     "Credit-card charges are settled via which third party?",
     ["stripe"]),
    ("vendor-email", "Transactional email is sent through Postmark.",
     "Which service sends our transactional email?",
     "Receipt and password-reset messages are delivered by which provider?",
     ["postmark"]),
    ("vendor-errors", "Application errors are tracked in Sentry.",
     "Which tool do we use for error tracking?",
     "Exception stack traces from services are aggregated in which platform?",
     ["sentry"]),

    ("cust-largest", "Our largest customer is Northwind Traders.",
     "Who is our largest customer?",
     "Which account represents the biggest share of our revenue?",
     ["northwind"]),
    ("cust-renewal", "The Northwind contract renews in March 2027.",
     "When does the Northwind contract renew?",
     "When is our biggest account next up for re-signing?",
     ["march", "2027"]),
    ("deploy-day", "We deploy to production only on Tuesdays.",
     "Which day do we deploy to production?",
     "On what weekday are code releases to the live environment allowed?",
     ["tuesday"]),
    ("merge-approvals", "Merging to main requires 3 approvals.",
     "How many approvals are required to merge to main?",
     "Before code can land on the trunk branch, how many sign-offs are needed?",
     ["3", "approval"]),
]

TOPK = 3


def matches(text: str, tokens: list[str]) -> bool:
    t = (text or "").lower()
    return all(tok.lower() in t for tok in tokens)


def rank_of(retrieved: list[str], tokens: list[str]) -> int | None:
    for i, r in enumerate(retrieved):
        if matches(r, tokens):
            return i + 1
    return None


def score(backend, scope: str, use_adv: bool) -> dict:
    hit1 = hit3 = 0
    rr = 0.0
    for _id, fact, qd, qa, tokens in DATASET:
        query = qa if use_adv else qd
        try:
            if hasattr(backend, "retrieve_ranked"):
                retrieved = backend.retrieve_ranked(scope, query, k=TOPK)
            else:
                retrieved = backend.retrieve(scope, query)[:TOPK]
        except Exception:
            retrieved = []
        r = rank_of(retrieved, tokens)
        if r == 1:
            hit1 += 1
        if r is not None and r <= 3:
            hit3 += 1
            rr += 1.0 / r
    n = len(DATASET)
    return {
        "hit@1": round(100 * hit1 / n, 1),
        "hit@3": round(100 * hit3 / n, 1),
        "mrr": round(rr / n, 3),
        "n": n,
    }


def warm(backend, scope: str, cap_s: float = 150.0) -> int:
    """Poll hosted ingestion by counting how many facts are searchable; bounded."""
    t0 = time.time()
    last = -1
    while time.time() - t0 < cap_s:
        cov = 0
        for _id, fact, qd, qa, tokens in DATASET:
            try:
                if rank_of(backend.retrieve(scope, qd), tokens):
                    cov += 1
            except Exception:
                pass
        print(f"   {backend.name} coverage {cov}/{len(DATASET)} at {time.time()-t0:.0f}s")
        if cov == len(DATASET) or cov == last:
            return cov
        last = cov
        time.sleep(10)
    return last


SEMANTIC_WEIGHTS = {"semantic_weight": 1.0, "keyword_weight": 0.0,
                    "temporal_weight": 0.0, "confidence_weight": 0.0}


def main() -> None:
    emb = FastEmbedEmbedder()
    run = uuid.uuid4().hex[:6]

    # Two SenseLab reranker tiers on top of adaptive-fusion retrieval:
    #   xenc - local cross-encoder (fastembed, offline, single-digit ms)
    #   llm  - listwise LLM reranker (Pro tier) that resolves confusable facts
    #          under adversarial paraphrase; the lever for top-1 accuracy.
    from amfs_retrieval import CrossEncoderReranker, LLMReranker
    xenc = CrossEncoderReranker()
    llm = LLMReranker(
        enabled=True,
        base_url=os.environ.get("AMFS_LLM_RERANK_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ.get("AMFS_LLM_RERANK_API_KEY", os.environ["OPENROUTER_API_KEY"]),
        model=os.environ.get("AMFS_LLM_RERANK_MODEL", "openai/gpt-4o-mini"),
    )

    backends = {
        "vector": (VectorStore(emb), True),
        "senselab-semantic": (AmfsProBackend(emb, HERE / "pro_audit.jsonl",
                                    retriever_weights=SEMANTIC_WEIGHTS,
                                    namespace="prec_sem"), True),
        "senselab-xenc": (AmfsProBackend(emb, HERE / "pro_audit.jsonl",
                                    namespace="prec_xenc", reranker=xenc), True),
        "senselab-llm": (AmfsProBackend(emb, HERE / "pro_audit.jsonl",
                                    namespace="prec_llm", reranker=llm), True),
        "mem0": (Mem0Backend(), False),
        "supermemory": (SupermemoryBackend(), False),
    }

    out = {}
    for name, (be, is_local) in backends.items():
        scope = f"prec-{name}-{run}"
        print(f"seeding {name} ({len(DATASET)} facts into one scope)...")
        for _id, fact, *_ in DATASET:
            try:
                be.seed(scope, fact)
            except Exception as e:
                print(f"   seed fail {name}: {e}")
        cov = len(DATASET)
        if not is_local:
            print(f"warming {name}...")
            cov = warm(be, scope)
        direct = score(be, scope, use_adv=False)
        adv = score(be, scope, use_adv=True)
        out[name] = {"coverage": cov, "direct": direct, "adversarial": adv}
        print(f"  {name}: direct {direct}  adversarial {adv}  coverage {cov}/{len(DATASET)}")

    out["_meta"] = {"facts": len(DATASET), "top_k": TOPK,
                    "clusters": "5 db, 5 on-call, 4 region, 4 port, 4 sla, 4 vendor, 4 misc"}
    Path(HERE / "precision_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote precision_results.json")


if __name__ == "__main__":
    main()
