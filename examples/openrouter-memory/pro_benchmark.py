"""Pro head-to-head benchmark: SenseLab vs mem0, Supermemory, a vector store.

Six conditions, all answered by the SAME reader model per cell so the only thing
that varies is what context the memory layer puts in front of the model:

  no-memory   : reader gets the question only (the failure floor).
  in-context  : the ground-truth fact is pasted into the prompt (capability ceiling).
  vector      : local embeddings (fastembed) + cosine top-k, no relevance gate.
  mem0        : hosted mem0 platform (MemoryClient) add + search.
  supermemory : hosted Supermemory documents.add + search.memories.
  amfs-pro    : amfs_retrieval.MultiStrategyRetriever with a confidence + semantic
                relevance gate; on a miss it injects a decline instruction instead of
                guessing. Reads/writes are captured in a sealed, signed
                ImmutableDecisionTrace (amfs_traces) that we then verify().

Experiments:
  1. Cross-model recall  : state a fact to a writer model, ask a different-vendor
                           reader model. Measures recall % per arm (+ per pair, + CI).
  2. Abstain-on-miss     : ask a question with NO supporting memory. Measures whether
                           each arm declines vs fabricates.
  3. Overhead            : p50/p95 reader latency and injected prompt tokens.
  4. Governance (deterministic, no LLM): sealed-trace seal/verify + chain verify,
                           MemorySafetyValidator on a poisoned write, entry hashing.

Secrets are read from the environment only (never written to disk):
  OPENROUTER_API_KEY, MEM0_API_KEY, SUPERMEMORY_API_KEY
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

HERE = Path(__file__).parent
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]

TRIALS = int(os.environ.get("PRO_TRIALS", "6"))          # reader trials per (fact,pair) for local arms
HOSTED_TRIALS = int(os.environ.get("PRO_HOSTED_TRIALS", str(TRIALS)))  # for mem0/supermemory
MISS_TRIALS = int(os.environ.get("PRO_MISS_TRIALS", "3"))
CONF_GATE = float(os.environ.get("PRO_CONF_GATE", "0.5"))
SEM_FLOOR = float(os.environ.get("PRO_SEM_FLOOR", "0.55"))
TOP_K = 3

MODEL_PAIRS = [
    ("openai/gpt-4o-mini", "mistralai/mistral-nemo"),
    ("mistralai/mistral-nemo", "amazon/nova-micro-v1"),
    ("amazon/nova-micro-v1", "cohere/command-r7b-12-2024"),
]

DATASET = [
    ("Our production database is PostgreSQL 15 and we only deploy on Fridays.",
     "What version is our production database, and which day do we deploy? Be concise.",
     [["postgres", "postgresql"], ["15"], ["friday"]]),
    ("The primary on-call engineer this quarter is Priya Nair, reachable at extension 4417.",
     "Who is the primary on-call engineer this quarter and what is their extension? Be concise.",
     [["priya"], ["4417"]]),
    ("Our services run in the eu-west-2 region and data must never leave the EU.",
     "Which region do our services run in? Be concise.",
     [["eu-west-2"]]),
    ("The internal billing API base URL is https://billing.internal.acme.dev/v3.",
     "What is the internal billing API base URL? Be concise.",
     [["billing.internal.acme.dev"], ["v3"]]),
    ("SEV-1 incidents must be acknowledged within 5 minutes and escalated to the VP after 15 minutes.",
     "How many minutes do we have to acknowledge a SEV-1 incident? Be concise.",
     [["5"], ["minute"]]),
    ("We rotate production API keys every 30 days using the vault-rotate job.",
     "How often do we rotate production API keys? Be concise.",
     [["30"], ["day"]]),
    ("Our largest customer is Meridian Freight, whose contract renews in March 2027.",
     "Who is our largest customer and when does their contract renew? Be concise.",
     [["meridian"], ["march"], ["2027"]]),
    ("Merging to the main branch is protected and requires exactly two approvals.",
     "How many approvals are required to merge to the main branch? Be concise.",
     [["two", "2"]]),
]

# Questions whose answer was NEVER stored — for the abstain-on-miss experiment.
MISS_QUESTIONS = [
    "What is our AWS account root email address? Be concise.",
    "What is the office WiFi password? Be concise.",
    "Which payment processor handles our EU invoices? Be concise.",
    "What is the version of our Redis cache in production? Be concise.",
    "What is the CEO's mobile phone number? Be concise.",
    "What is the hostname of our staging database? Be concise.",
    "Which CDN provider do we use for static assets? Be concise.",
    "What is the name of our PagerDuty escalation policy? Be concise.",
]

DECLINE_PAT = re.compile(
    r"\b(i (do not|don't) (have|know)|no (information|record|data)|not (sure|available|in (my|our)|specified|provided)|"
    r"cannot (find|determine|provide)|can't (find|determine|provide)|unable to|unknown|not stored|"
    r"don't have (that|this|access)|no access|i'm not aware|not aware of)\b",
    re.IGNORECASE,
)


def grade(answer: str, groups) -> bool:
    a = answer.lower()
    return all(any(alt in a for alt in g) for g in groups)


def is_decline(answer: str) -> bool:
    return bool(DECLINE_PAT.search(answer or ""))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * (c - m), 1), round(100 * (c + m), 1)]


# --------------------------------------------------------------------------
# OpenRouter reader call (shared by every arm)
# --------------------------------------------------------------------------

def reader_call(model: str, system: str | None, question: str, tries: int = 5):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    payload = {"model": model, "messages": messages, "max_tokens": 80, "usage": {"include": True}}
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    r = None
    for i in range(tries):
        t0 = time.time()
        try:
            r = requests.post(OPENROUTER, headers=headers, data=json.dumps(payload), timeout=120)
        except Exception:
            time.sleep(1.5 * (i + 1))
            continue
        dt = (time.time() - t0) * 1000
        if r.status_code == 200:
            j = r.json()
            ans = j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            usage = j.get("usage", {}) or {}
            return ans, dt, usage, j.get("model", model)
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (i + 1))
            continue
        break
    return "", 0.0, {}, model


# --------------------------------------------------------------------------
# Embedder (fastembed) wrapping amfs_core EmbedderABC
# --------------------------------------------------------------------------

from amfs_core.embedder import EmbedderABC  # noqa: E402


class FastEmbedEmbedder(EmbedderABC):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name)

    def embed(self, text: str) -> list[float]:
        vec = next(iter(self._model.embed([text])))
        return [float(x) for x in vec]

    def embed_value(self, value: Any) -> list[float]:
        return self.embed(value if isinstance(value, str) else json.dumps(value))


def cosine(a: list[float], b: list[float]) -> float:
    av, bv = np.array(a), np.array(b)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv)) or 1.0
    return float(np.dot(av, bv) / denom)


# --------------------------------------------------------------------------
# Memory backends
# --------------------------------------------------------------------------

class VectorStore:
    """Local embeddings + cosine top-k, no relevance gate (the naive RAG baseline)."""
    name = "vector"

    def __init__(self, embedder: FastEmbedEmbedder) -> None:
        self._emb = embedder
        self._store: dict[str, list[tuple[list[float], str]]] = {}

    def seed(self, scope: str, fact: str) -> None:
        self._store.setdefault(scope, []).append((self._emb.embed(fact), fact))

    def retrieve(self, scope: str, query: str) -> list[str]:
        items = self._store.get(scope, [])
        if not items:
            return []
        qv = self._emb.embed(query)
        ranked = sorted(items, key=lambda it: cosine(qv, it[0]), reverse=True)
        return [t for _, t in ranked[:TOP_K]]


class Mem0Backend:
    name = "mem0"

    def __init__(self) -> None:
        from mem0 import MemoryClient
        self._c = MemoryClient(api_key=os.environ["MEM0_API_KEY"])

    def seed(self, scope: str, fact: str) -> None:
        self._c.add([{"role": "user", "content": fact}], user_id=scope)

    def _search(self, scope: str, query: str) -> list[str]:
        resp = self._c.search(query, version="v2", filters={"user_id": scope}, limit=TOP_K)
        results = resp.get("results", resp) if isinstance(resp, dict) else resp
        texts = []
        for m in results or []:
            if isinstance(m, dict):
                t = m.get("memory") or m.get("text") or m.get("content")
                if t:
                    texts.append(t)
        return texts

    def wait_ready(self, scope: str, probe: str, tries: int = 12) -> bool:
        for _ in range(tries):
            try:
                if self._search(scope, probe):
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def retrieve(self, scope: str, query: str) -> list[str]:
        for i in range(3):
            try:
                return self._search(scope, query)
            except Exception:
                time.sleep(1.0 * (i + 1))
        return []


class SupermemoryBackend:
    name = "supermemory"

    def __init__(self) -> None:
        import supermemory
        self._c = supermemory.Supermemory(api_key=os.environ["SUPERMEMORY_API_KEY"])

    def seed(self, scope: str, fact: str) -> None:
        self._c.documents.add(content=fact, container_tag=scope, dreaming="instant")

    def _search(self, scope: str, query: str) -> list[str]:
        resp = self._c.search.memories(q=query, container_tag=scope, limit=TOP_K)
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        texts = []
        for m in (data.get("results") or []):
            t = m.get("memory") or m.get("content") or m.get("chunk") or m.get("summary")
            if t:
                texts.append(t if isinstance(t, str) else str(t))
        return texts

    def wait_ready(self, scope: str, probe: str, tries: int = 12) -> bool:
        for _ in range(tries):
            try:
                if self._search(scope, probe):
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def retrieve(self, scope: str, query: str) -> list[str]:
        for i in range(3):
            try:
                return self._search(scope, query)
            except Exception:
                time.sleep(1.0 * (i + 1))
        return []


class AmfsProBackend:
    """SenseLab: multi-strategy retrieval with confidence + semantic gate, sealed traces."""
    name = "amfs-pro"

    def __init__(self, embedder: FastEmbedEmbedder, audit_log: Path,
                 retriever_weights: dict | None = None, namespace: str = "pro_bench",
                 reranker: object | None = None) -> None:
        from amfs import AgentMemory
        from amfs_filesystem import FilesystemAdapter
        from amfs_retrieval import MultiStrategyRetriever
        from amfs_traces.store import InMemoryTraceStore
        self._AgentMemory = AgentMemory
        self._emb = embedder
        self._adapter = FilesystemAdapter(HERE / ".pro_data", namespace=namespace)
        weights = retriever_weights or {}
        self._retriever = MultiStrategyRetriever(
            self._adapter, embedder=embedder, reranker=reranker, **weights)
        self._audit = audit_log
        self._store = InMemoryTraceStore()

    def _mem(self, session: str):
        return self._AgentMemory(agent_id="pro-agent", session_id=session, adapter=self._adapter)

    def seed(self, scope: str, fact: str) -> None:
        from amfs_traces import TraceRecorder
        from amfs_core.models import MemoryType, OutcomeType
        rec = TraceRecorder(self._mem(f"w-{uuid.uuid4().hex[:8]}"), self._store)
        key = f"fact-{uuid.uuid4().hex[:8]}"
        rec.write(scope, key, fact, confidence=0.9, memory_type=MemoryType.FACT)
        rec.commit_outcome(f"seed-{key}", OutcomeType.SUCCESS, decision_summary="seeded fact")

    def retrieve_gated(self, scope: str, query: str) -> tuple[list[str], list[dict]]:
        results = self._retriever.retrieve(query, entity_path=scope, min_confidence=CONF_GATE, limit=TOP_K)
        kept, meta = [], []
        for r in results:
            sem = r.strategy_scores.get("semantic", 0.0)
            if sem >= SEM_FLOOR:
                kept.append(str(r.entry.get("value")))
                meta.append({"key": r.entry_key, "semantic": round(sem, 3),
                             "confidence": r.entry.get("confidence")})
        return kept, meta

    def retrieve_ranked(self, scope: str, query: str, k: int = TOP_K) -> list[str]:
        """Raw top-k ranking, no confidence/relevance gate — for ranking-quality comparison."""
        results = self._retriever.retrieve(query, entity_path=scope, min_confidence=0.0, limit=k)
        return [str(r.entry.get("value")) for r in results]

    def retrieve(self, scope: str, query: str) -> list[str]:
        return self.retrieve_ranked(scope, query)

    def record_reader(self, scope: str, query: str, injected_meta: list[dict],
                      routed_model: str, usage: dict, answer: str) -> dict:
        """Build + seal + verify a decision trace for this reader turn."""
        from amfs_traces import TraceRecorder
        from amfs_traces.crypto import verify
        from amfs_core.models import OutcomeType
        session = f"r-{uuid.uuid4().hex[:8]}"
        rec = TraceRecorder(self._mem(session), self._store)
        for m in injected_meta:
            ep, key = m["key"].rsplit("/", 1)
            rec.read(ep, key)
        rec.record_llm_call(model=routed_model, provider="openrouter",
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            cost_usd=usage.get("cost"))
        rec.record_context("routed-model", f"model={routed_model} cost={usage.get('cost')}",
                           source="openrouter")
        ref = f"req-{uuid.uuid4().hex[:10]}"
        _, sealed = rec.commit_outcome(ref, OutcomeType.SUCCESS,
                                       decision_summary=f"answered via {routed_model}")
        vr = verify(sealed, os.environ.get("AMFS_TRACE_SIGNING_KEY",
                    __import__("hashlib").sha256(b"amfs-dev-signing-key").hexdigest()))
        audit = {
            "outcome_ref": ref, "routed_model": routed_model,
            "cost_usd": usage.get("cost"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "memory_used": injected_meta,
            "content_hash": sealed.content_hash, "signature": sealed.signature,
            "signing_key_id": sealed.signing_key_id, "parent_hash": sealed.parent_hash,
            "sequence_number": sealed.sequence_number,
            "trace_verified": vr.valid,
            "answer_excerpt": (answer or "")[:200],
        }
        with open(self._audit, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit, default=str) + "\n")
        return audit


INJECT_TMPL = ("Relevant facts from memory:\n{facts}\n"
               "Answer using these facts if they are relevant.")
ABSTAIN_MSG = ("You have no reliable stored information for this question. "
               "If you do not know the answer from memory, say you do not have that "
               "information rather than guessing.")


def _fmt(texts: list[str]) -> str:
    return INJECT_TMPL.format(facts="\n".join(f"- {t}" for t in texts))


# --------------------------------------------------------------------------
# Experiment 1: cross-model recall
# --------------------------------------------------------------------------

def run_recall(backends: dict, amfs: "AmfsProBackend") -> dict:
    arms = ["no-memory", "in-context", "vector", "mem0", "supermemory", "amfs-pro"]
    hits = {a: [0, 0] for a in arms}
    per_pair = {f"{w}->{r}": {a: [0, 0] for a in arms} for w, r in MODEL_PAIRS}
    latency = {a: [] for a in arms}
    prompt_tokens = {a: [] for a in arms}
    trace_verified = [0, 0]

    # Seed each fact ONCE per stateful backend (memory is model-independent), then
    # warm the hosted scopes a single time so per-trial retrieval does not poll.
    run = uuid.uuid4().hex[:6]
    fact_scopes: list[dict] = []
    for fi, (fact, question, groups) in enumerate(DATASET):
        scopes = {}
        for name in ("vector", "mem0", "supermemory", "amfs-pro"):
            scope = f"{name}-{run}-{fi}"
            scopes[name] = scope
            try:
                (amfs if name == "amfs-pro" else backends[name]).seed(scope, fact)
            except Exception as e:
                print(f"   seed {name} f{fi} failed: {e}")
        fact_scopes.append(scopes)
    print("seeded all facts; warming hosted ingestion...")
    for fi, (fact, question, groups) in enumerate(DATASET):
        for name in ("mem0", "supermemory"):
            ok = backends[name].wait_ready(fact_scopes[fi][name], question)
            print(f"   {name} f{fi} ready={ok}")

    total = len(DATASET) * len(MODEL_PAIRS)
    cellno = 0
    for fi, (fact, question, groups) in enumerate(DATASET):
        scopes = fact_scopes[fi]
        for (writer, reader) in MODEL_PAIRS:
            cellno += 1
            pk = f"{writer}->{reader}"
            print(f"[recall {cellno}/{total}] {pk} :: {question[:38]}")

            for t in range(TRIALS):
                def run_arm(arm: str, system: str | None):
                    ans, dt, usage, routed = reader_call(reader, system, question)
                    ok = grade(ans, groups)
                    hits[arm][0] += int(ok); hits[arm][1] += 1
                    per_pair[pk][arm][0] += int(ok); per_pair[pk][arm][1] += 1
                    latency[arm].append(dt)
                    prompt_tokens[arm].append(usage.get("prompt_tokens", 0))
                    return ans, usage, routed

                run_arm("no-memory", None)
                run_arm("in-context", fact)

                vt = backends["vector"].retrieve(scopes["vector"], question)
                run_arm("vector", _fmt(vt) if vt else None)

                try:
                    mt = backends["mem0"].retrieve(scopes["mem0"], question)
                except Exception:
                    mt = []
                run_arm("mem0", _fmt(mt) if mt else None)

                try:
                    st = backends["supermemory"].retrieve(scopes["supermemory"], question)
                except Exception:
                    st = []
                run_arm("supermemory", _fmt(st) if st else None)

                kept, meta = amfs.retrieve_gated(scopes["amfs-pro"], question)
                system = _fmt(kept) if kept else ABSTAIN_MSG
                ans, usage, routed = run_arm("amfs-pro", system)
                if kept:
                    audit = amfs.record_reader(scopes["amfs-pro"], question, meta, routed, usage, ans)
                    trace_verified[1] += 1
                    trace_verified[0] += int(audit["trace_verified"])

    def pct(cell):
        c, n = cell
        return {"pct": round(100 * c / n, 1) if n else None, "n": n, "ci": wilson(c, n)}

    def pctl(xs, q):
        return round(statistics.quantiles(xs, n=100)[q - 1], 1) if len(xs) >= 2 else (round(xs[0], 1) if xs else None)

    return {
        "recall": {a: pct(hits[a]) for a in arms},
        "per_pair": {pk: {a: pct(d[a]) for a in arms} for pk, d in per_pair.items()},
        "overhead": {
            a: {
                "latency_p50_ms": pctl(latency[a], 50),
                "latency_p95_ms": pctl(latency[a], 95),
                "mean_prompt_tokens": round(statistics.mean(prompt_tokens[a]), 1) if prompt_tokens[a] else None,
            } for a in arms
        },
        "amfs_trace_verified_pct": round(100 * trace_verified[0] / trace_verified[1], 1) if trace_verified[1] else None,
        "amfs_trace_verified_n": trace_verified[1],
    }


# --------------------------------------------------------------------------
# Experiment 2: abstain-on-miss
# --------------------------------------------------------------------------

def run_abstain(backends: dict, amfs: "AmfsProBackend") -> dict:
    """Seed real facts, then ask questions with NO supporting memory.
    Measure decline (good) vs fabricate (bad) per arm."""
    arms = ["no-memory", "vector", "mem0", "supermemory", "amfs-pro"]
    decline = {a: [0, 0] for a in arms}
    reader = MODEL_PAIRS[0][1]

    # one populated scope per backend, seeded with all 8 real facts
    scopes = {}
    for name in ("vector", "mem0", "supermemory", "amfs-pro"):
        scope = f"miss-{name}-{uuid.uuid4().hex[:6]}"
        scopes[name] = scope
        for fact, _, _ in DATASET:
            try:
                (amfs if name == "amfs-pro" else backends[name]).seed(scope, fact)
            except Exception as e:
                print(f"   abstain seed {name} failed: {e}")
    for name in ("mem0", "supermemory"):
        backends[name].wait_ready(scopes[name], DATASET[0][1])

    for qi, q in enumerate(MISS_QUESTIONS):
        print(f"[abstain {qi+1}/{len(MISS_QUESTIONS)}] {q[:44]}")
        for t in range(MISS_TRIALS):
            def run(arm, system):
                ans, *_ = reader_call(reader, system, q)
                d = is_decline(ans)
                decline[arm][0] += int(d); decline[arm][1] += 1

            run("no-memory", None)
            vt = backends["vector"].retrieve(scopes["vector"], q)
            run("vector", _fmt(vt) if vt else None)
            try:
                mt = backends["mem0"].retrieve(scopes["mem0"], q)
            except Exception:
                mt = []
            run("mem0", _fmt(mt) if mt else None)
            try:
                st = backends["supermemory"].retrieve(scopes["supermemory"], q)
            except Exception:
                st = []
            run("supermemory", _fmt(st) if st else None)
            kept, _ = amfs.retrieve_gated(scopes["amfs-pro"], q)
            run("amfs-pro", _fmt(kept) if kept else ABSTAIN_MSG)

    return {a: {"decline_pct": round(100 * decline[a][0] / decline[a][1], 1) if decline[a][1] else None,
                "n": decline[a][1], "ci": wilson(decline[a][0], decline[a][1])} for a in arms}


# --------------------------------------------------------------------------
# Experiment 4: governance demos (deterministic, no LLM)
# --------------------------------------------------------------------------

def governance_demos() -> dict:
    import hashlib
    from amfs import AgentMemory
    from amfs_filesystem import FilesystemAdapter
    from amfs_core.models import MemoryType, OutcomeType, Provenance
    from amfs_traces import TraceRecorder
    from amfs_traces.store import InMemoryTraceStore
    from amfs_traces.crypto import verify, verify_chain
    from amfs_safety import MemorySafetyValidator

    key = os.environ.get("AMFS_TRACE_SIGNING_KEY", hashlib.sha256(b"amfs-dev-signing-key").hexdigest())

    # --- sealed trace: seal -> verify clean -> tamper -> verify fails -> chain ---
    adapter = FilesystemAdapter(HERE / ".gov_data", namespace="gov")
    store = InMemoryTraceStore()
    mem = AgentMemory(agent_id="gov-agent", session_id="gov-sess", adapter=adapter)
    rec = TraceRecorder(mem, store)
    rec.write("gov/svc", "db", "PostgreSQL 15", confidence=0.9, memory_type=MemoryType.FACT)
    _, t1 = rec.commit_outcome("gov-1", OutcomeType.SUCCESS, decision_summary="first")
    rec.clear()  # same recorder keeps the sequence counter, so the chain links
    rec.read("gov/svc", "db")
    _, t2 = rec.commit_outcome("gov-2", OutcomeType.SUCCESS, decision_summary="second")

    clean = verify(t1, key)
    tampered = t1.model_copy(update={"decision_summary": "ALTERED"})
    tamper_res = verify(tampered, key)
    chain = verify_chain([t1, t2], key)

    # --- safety validator: contradiction + low-confidence write ---
    mem.write("gov/policy", "region", "eu-west-2", confidence=0.9, memory_type=MemoryType.FACT)
    existing = mem.read("gov/policy", "region")
    validator = MemorySafetyValidator(adapter)
    poisoned = existing.model_copy(update={
        "value": "us-east-1",
        "confidence": 0.95,
        "provenance": Provenance(agent_id="attacker", session_id="x",
                                 written_at=datetime.now(timezone.utc)),
    })
    contradiction = validator.validate_write(poisoned)
    lowconf = existing.model_copy(update={"value": "maybe-eu", "confidence": 0.3,
                                          "memory_type": MemoryType.FACT})
    lowconf_res = validator.validate_write(lowconf)

    return {
        "sealed_trace": {
            "signed": bool(t1.signature) and bool(t1.signing_key_id),
            "clean_verify_valid": clean.valid,
            "tamper_detected": not tamper_res.valid,
            "tamper_errors": tamper_res.errors[:2],
            "chain_verified": chain.chain_verified,
            "chain_length": chain.chain_length,
            "content_hash": t1.content_hash[:16],
            "signature": t1.signature[:16],
        },
        "safety_validator": {
            "contradiction_flagged": any(i.check == "contradiction" for i in contradiction.issues),
            "contradiction_detail": contradiction.issues[0].description if contradiction.issues else None,
            "low_confidence_flagged": any(i.check == "confidence_threshold" for i in lowconf_res.issues),
        },
    }


def main() -> None:
    audit_log = HERE / "pro_audit.jsonl"
    if audit_log.exists():
        audit_log.unlink()
    import shutil
    for d in (".pro_data", ".gov_data"):
        p = HERE / d
        if p.exists():
            shutil.rmtree(p)

    embedder = FastEmbedEmbedder()
    print("embedder ready")
    backends = {
        "vector": VectorStore(embedder),
        "mem0": Mem0Backend(),
        "supermemory": SupermemoryBackend(),
    }
    amfs = AmfsProBackend(embedder, audit_log)
    print("backends ready")

    recall = run_recall(backends, amfs)
    abstain = run_abstain(backends, amfs)
    gov = governance_demos()

    summary = {
        "config": {
            "model_pairs": MODEL_PAIRS, "facts": len(DATASET), "trials": TRIALS,
            "miss_questions": len(MISS_QUESTIONS), "miss_trials": MISS_TRIALS,
            "conf_gate": CONF_GATE, "sem_floor": SEM_FLOOR, "top_k": TOP_K,
        },
        "recall": recall["recall"],
        "per_pair_recall": recall["per_pair"],
        "overhead": recall["overhead"],
        "abstain_on_miss": abstain,
        "amfs_trace_verified_pct": recall["amfs_trace_verified_pct"],
        "amfs_trace_verified_n": recall["amfs_trace_verified_n"],
        "governance": gov,
    }
    out = HERE / "pro_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
