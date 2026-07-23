"""Experiment battery for the whitepaper.

Quantifies the problems a memory + governance/audit proxy solves, across multiple
facts, cross-vendor model pairs, and repeated trials. Three arms:

  no-memory  : ask the reader model the question with no context (the pain).
  in-context : prepend the fact to the reader prompt (control -> proves the model
               CAN use the fact; isolates *memory transfer* as the only variable).
  proxy      : state the fact to a writer model via the AMFS proxy, then ask the
               reader model via the proxy (memory follows across the vendor switch).

Metrics:
  - cross-model recall success rate per arm (and per model pair)
  - audit reconstruction coverage (proxy): can we recover memory(version,
    confidence, content_hash) + routed model + cost + tokens for each answer
  - model-attribution coverage
  - injected-token + latency overhead
  - integrity / tamper-evidence (separate deterministic test via amfs_core)

Prereqs:
  export OPENROUTER_API_KEY=sk-or-...
  python proxy.py           # 127.0.0.1:8088
  python paper_benchmark.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from pathlib import Path

import requests

PROXY = os.environ.get("AMFS_PROXY_URL", "http://127.0.0.1:8088")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]
HERE = Path(__file__).parent
AUDIT_LOG = HERE / "audit.jsonl"

TRIALS = int(os.environ.get("PAPER_TRIALS", "2"))

# writer != reader, cross-vendor
MODEL_PAIRS = [
    ("openai/gpt-4o-mini", "mistralai/mistral-nemo"),
    ("mistralai/mistral-nemo", "amazon/nova-micro-v1"),
    ("amazon/nova-micro-v1", "cohere/command-r7b-12-2024"),
]

# (fact stated in session 1, question asked in session 2, required-keyword groups)
DATASET = [
    ("Remember for later: our production database is PostgreSQL 15 and we only deploy on Fridays.",
     "What version is our production database, and which day do we deploy? Be concise.",
     [["postgres", "postgresql"], ["15"], ["friday"]]),
    ("Remember for later: the primary on-call engineer this quarter is Priya Nair, reachable at extension 4417.",
     "Who is the primary on-call engineer this quarter and what is their extension? Be concise.",
     [["priya"], ["4417"]]),
    ("Remember for later: our services run in the eu-west-2 region and data must never leave the EU.",
     "Which region do our services run in? Be concise.",
     [["eu-west-2"]]),
    ("Remember for later: the internal billing API base URL is https://billing.internal.acme.dev/v3.",
     "What is the internal billing API base URL? Be concise.",
     [["billing.internal.acme.dev"], ["v3"]]),
    ("Remember for later: SEV-1 incidents must be acknowledged within 5 minutes and escalated to the VP after 15 minutes.",
     "How many minutes do we have to acknowledge a SEV-1 incident? Be concise.",
     [["5"], ["minute"]]),
    ("Remember for later: we rotate production API keys every 30 days using the vault-rotate job.",
     "How often do we rotate production API keys? Be concise.",
     [["30"], ["day"]]),
    ("Remember for later: our largest customer is Meridian Freight, whose contract renews in March 2027.",
     "Who is our largest customer and when does their contract renew? Be concise.",
     [["meridian"], ["march"], ["2027"]]),
    ("Remember for later: merging to the main branch is protected and requires exactly two approvals.",
     "How many approvals are required to merge to the main branch? Be concise.",
     [["two", "2"]]),
]


def _grade(answer: str, groups) -> bool:
    a = answer.lower()
    return all(any(alt in a for alt in g) for g in groups)


def _call(url: str, payload: dict, headers: dict, tries: int = 4):
    for i in range(tries):
        t0 = time.time()
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        dt = (time.time() - t0) * 1000
        if r.status_code == 200:
            return r, dt
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (i + 1))
            continue
        return r, dt
    return r, dt


def _direct(model: str, messages: list[dict]):
    r, dt = _call(OPENROUTER, {"model": model, "messages": messages, "max_tokens": 80,
                               "usage": {"include": True}},
                  {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    j = r.json() if r.status_code == 200 else {}
    ans = j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    usage = j.get("usage", {}) or {}
    return ans, dt, usage


def _proxy(model: str, messages: list[dict], agent: str, entity: str):
    r, dt = _call(f"{PROXY}/v1/chat/completions",
                  {"model": model, "messages": messages, "max_tokens": 80},
                  {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                   "X-AMFS-Agent": agent, "X-AMFS-Entity": entity})
    j = r.json() if r.status_code == 200 else {}
    ans = j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return ans, dt, r.headers


def integrity_test() -> dict:
    """Deterministic tamper-evidence test using amfs_core directly."""
    from datetime import datetime, timezone

    from amfs_core.hashing import content_hash, verify_entries
    from amfs_core.models import MemoryEntry, Provenance

    prov = Provenance(agent_id="a", session_id="s", written_at=datetime.now(timezone.utc))
    val = "our production database is PostgreSQL 15"
    h = content_hash(val)
    good = MemoryEntry(entity_path="paper/integrity", key="fact", version=1,
                       value=val, provenance=prov,
                       content_hash=h, integrity_chain=h)
    rep_before = verify_entries([good])

    tampered = good.model_copy(update={"value": "our production database is MySQL 5.7"})
    rep_after = verify_entries([tampered])

    return {
        "clean_valid": rep_before.valid == 1 and not rep_before.corrupted,
        "tamper_detected": len(rep_after.corrupted) == 1,
        "corrupted_report": rep_after.corrupted,
    }


def main() -> None:
    try:
        requests.get(f"{PROXY}/health", timeout=5)
    except Exception:
        raise SystemExit(f"Proxy not reachable at {PROXY}. Start it: python proxy.py")

    # fresh audit log for this experiment
    if AUDIT_LOG.exists():
        AUDIT_LOG.unlink()

    arms = {"no-memory": [], "in-context": [], "proxy": []}
    proxy_reader_refs = []          # (outcome_ref, entity)
    overhead = {"no_memory_prompt_tokens": [], "proxy_prompt_tokens": [],
                "no_memory_latency": [], "proxy_reader_latency": []}
    per_pair = {f"{w}->{r}": {"no-memory": [0, 0], "in-context": [0, 0], "proxy": [0, 0]}
                for (w, r) in MODEL_PAIRS}

    n_cells = len(DATASET) * len(MODEL_PAIRS) * TRIALS
    cell = 0
    for fact, question, groups in DATASET:
        for (writer, reader) in MODEL_PAIRS:
            pair_key = f"{writer}->{reader}"
            for t in range(TRIALS):
                cell += 1
                print(f"[{cell}/{n_cells}] {pair_key} :: {question[:40]}...")

                # ARM no-memory: reader gets question only
                ans, dt, usage = _direct(reader, [{"role": "user", "content": question}])
                ok = _grade(ans, groups)
                arms["no-memory"].append(ok)
                per_pair[pair_key]["no-memory"][0] += int(ok)
                per_pair[pair_key]["no-memory"][1] += 1
                overhead["no_memory_prompt_tokens"].append(usage.get("prompt_tokens", 0))
                overhead["no_memory_latency"].append(dt)

                # ARM in-context: fact prepended to reader prompt (control)
                ans, _, _ = _direct(reader, [{"role": "user", "content": fact + "\n\n" + question}])
                ok = _grade(ans, groups)
                arms["in-context"].append(ok)
                per_pair[pair_key]["in-context"][0] += int(ok)
                per_pair[pair_key]["in-context"][1] += 1

                # ARM proxy: writer states fact, reader asks — via proxy
                entity = f"paper/{uuid.uuid4().hex[:10]}"
                agent = "paper-agent"
                _proxy(writer, [{"role": "user", "content": fact}], agent, entity)
                ans, dt, headers = _proxy(reader, [{"role": "user", "content": question}], agent, entity)
                ok = _grade(ans, groups)
                arms["proxy"].append(ok)
                per_pair[pair_key]["proxy"][0] += int(ok)
                per_pair[pair_key]["proxy"][1] += 1
                overhead["proxy_reader_latency"].append(dt)
                ref = headers.get("X-AMFS-Outcome-Ref")
                if ref:
                    proxy_reader_refs.append((ref, entity))

    # ---- audit coverage from audit.jsonl ----
    records = {}
    if AUDIT_LOG.exists():
        for line in AUDIT_LOG.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                records[rec["outcome_ref"]] = rec

    full_audit, attributed = 0, 0
    for ref, _entity in proxy_reader_refs:
        rec = records.get(ref)
        if not rec:
            continue
        ce = rec.get("causal_entries", [])
        has_mem = any(e.get("content_hash") and e.get("version") and e.get("confidence") is not None for e in ce)
        has_model = bool(rec.get("routed_model"))
        has_cost = rec.get("cost_usd") is not None
        has_tokens = rec.get("prompt_tokens") is not None and rec.get("completion_tokens") is not None
        if has_mem and has_model and has_cost and has_tokens:
            full_audit += 1
        if has_model:
            attributed += 1
        if rec.get("prompt_tokens") is not None:
            overhead["proxy_prompt_tokens"].append(rec["prompt_tokens"])

    def rate(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    n_proxy_reads = len(proxy_reader_refs)
    summary = {
        "config": {"facts": len(DATASET), "model_pairs": MODEL_PAIRS, "trials": TRIALS,
                   "cells": n_cells},
        "recall_success_pct": {arm: rate(v) for arm, v in arms.items()},
        "n_per_arm": {arm: len(v) for arm, v in arms.items()},
        "audit": {
            "proxy_reader_calls": n_proxy_reads,
            "full_reconstruction_pct": round(100 * full_audit / n_proxy_reads, 1) if n_proxy_reads else None,
            "model_attribution_pct": round(100 * attributed / n_proxy_reads, 1) if n_proxy_reads else None,
        },
        "overhead": {
            "mean_baseline_prompt_tokens": round(statistics.mean(overhead["no_memory_prompt_tokens"]), 1) if overhead["no_memory_prompt_tokens"] else None,
            "mean_proxy_prompt_tokens": round(statistics.mean(overhead["proxy_prompt_tokens"]), 1) if overhead["proxy_prompt_tokens"] else None,
            "mean_no_memory_latency_ms": round(statistics.mean(overhead["no_memory_latency"]), 1) if overhead["no_memory_latency"] else None,
            "mean_proxy_reader_latency_ms": round(statistics.mean(overhead["proxy_reader_latency"]), 1) if overhead["proxy_reader_latency"] else None,
        },
        "per_pair_recall": {
            pk: {arm: (round(100 * c / n, 1) if n else None) for arm, (c, n) in d.items()}
            for pk, d in per_pair.items()
        },
        "integrity": integrity_test(),
    }

    out = HERE / "paper_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
