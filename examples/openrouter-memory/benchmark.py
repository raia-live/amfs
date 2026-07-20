"""Benchmark: does memory follow the agent across an OpenRouter model switch,
and can we AUDIT which memory + which model produced each answer?

Scenario (single agent + entity):
  Session 1 -> state a durable fact, routed to MODEL_A (cheap).
  Session 2 -> fresh context, forced to MODEL_B (different vendor), ask a
               question that requires the fact from session 1.

Arms:
  no-memory : call OpenRouter directly (the pain, quantified).
  proxy     : call through the AMFS memory proxy (the candidate).

We measure the incumbents' table stakes (did the fact survive the switch,
latency, cost) AND the AMFS differentiator: reconstruct the decision trace
"answer -> memory used -> routed model" from audit.jsonl.

Prereqs:
  export OPENROUTER_API_KEY=sk-or-...
  python proxy.py            # in another terminal (127.0.0.1:8088)
  python benchmark.py
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import requests

PROXY = os.environ.get("AMFS_PROXY_URL", "http://127.0.0.1:8088")
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]
AUDIT_LOG = Path(__file__).parent / "audit.jsonl"

MODEL_A = "openai/gpt-4o-mini"       # session 1 (states the fact)
MODEL_B = "mistralai/mistral-nemo"   # session 2 (different vendor, answers)

FACT = (
    "Remember this for later: our production database is PostgreSQL 15, "
    "and we only deploy on Fridays."
)
QUESTION = "What version is our production database, and which day do we deploy? Answer concisely."
# Grading keywords the answer must contain to count as 'remembered'.
MUST_CONTAIN = [["postgres", "postgresql"], ["15"], ["friday"]]


def _grade(answer: str) -> bool:
    a = answer.lower()
    return all(any(alt in a for alt in group) for group in MUST_CONTAIN)


def _direct(model: str, messages: list[dict]) -> tuple[str, float, dict]:
    t0 = time.time()
    r = requests.post(
        OPENROUTER,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        data=json.dumps({"model": model, "messages": messages, "usage": {"include": True}}),
        timeout=120,
    )
    dt = (time.time() - t0) * 1000
    j = r.json()
    ans = j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return ans, dt, j


def _via_proxy(model: str, messages: list[dict], agent: str, entity: str) -> tuple[str, float, dict]:
    t0 = time.time()
    r = requests.post(
        f"{PROXY}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "X-AMFS-Agent": agent,
            "X-AMFS-Entity": entity,
        },
        data=json.dumps({"model": model, "messages": messages}),
        timeout=120,
    )
    dt = (time.time() - t0) * 1000
    j = r.json()
    ans = j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return ans, dt, {"headers": r.headers, "body": j}  # keep case-insensitive headers


def run_arm_no_memory() -> dict:
    print("\n=== ARM: no-memory (direct OpenRouter, no external memory) ===")
    # Session 1: state the fact (model A). Session 2 is a *fresh* context (model B).
    _direct(MODEL_A, [{"role": "user", "content": FACT}])
    ans, dt, j = _direct(MODEL_B, [{"role": "user", "content": QUESTION}])
    ok = _grade(ans)
    print(f"  session2 model={j.get('model')}  remembered={ok}")
    print(f"  answer: {ans.strip()[:160]}")
    return {"arm": "no-memory", "remembered": ok, "latency_ms": round(dt, 1),
            "routed_model": j.get("model"), "answer": ans.strip()}


def run_arm_proxy() -> dict:
    print("\n=== ARM: proxy (AMFS memory follows across the model switch) ===")
    agent = "bench-agent"
    entity = f"bench/run-{uuid.uuid4().hex[:8]}"  # isolate each run
    # Session 1: state the fact via MODEL_A -> proxy stores it.
    _via_proxy(MODEL_A, [{"role": "user", "content": FACT}], agent, entity)
    # Session 2: fresh context, forced to MODEL_B -> proxy injects the stored fact.
    ans, dt, meta = _via_proxy(MODEL_B, [{"role": "user", "content": QUESTION}], agent, entity)
    ok = _grade(ans)
    h = meta["headers"]
    print(f"  session2 requested={MODEL_B}  routed={h.get('X-AMFS-Routed-Model')}  "
          f"injected={h.get('X-AMFS-Memory-Injected')}  remembered={ok}")
    print(f"  answer: {ans.strip()[:160]}")
    return {"arm": "proxy", "remembered": ok, "latency_ms": round(dt, 1),
            "routed_model": h.get("X-AMFS-Routed-Model"),
            "memory_injected": h.get("X-AMFS-Memory-Injected"),
            "outcome_ref": h.get("X-AMFS-Outcome-Ref"),
            "entity": entity, "answer": ans.strip()}


def show_audit_trail(entity: str, outcome_ref: str) -> None:
    """The differentiator: reconstruct 'answer -> memory used -> routed model'."""
    print("\n=== AUDIT TRAIL (AMFS decision trace — the differentiator) ===")
    if not AUDIT_LOG.exists():
        print("  (no audit.jsonl found)")
        return
    records = [json.loads(l) for l in AUDIT_LOG.read_text().splitlines() if l.strip()]
    rec = next((r for r in records if r.get("outcome_ref") == outcome_ref), None)
    if not rec:
        print("  (outcome not found in audit log)")
        return
    print(f"  outcome_ref : {rec['outcome_ref']}")
    print(f"  routed_model: {rec['routed_model']}  (requested: {rec['requested_model']})")
    print(f"  cost_usd    : {rec['cost_usd']}   latency_ms: {rec['latency_ms']}")
    print(f"  memory used in this answer:")
    for e in rec.get("causal_entries", []):
        print(f"    - {e.get('entity_path')}/{e.get('key')} v{e.get('version')} "
              f"(conf {e.get('confidence')}): {str(e.get('value'))[:80]}")
    for c in rec.get("external_contexts", []):
        print(f"  context     : [{c.get('label')}] {c.get('summary')}")
    print("\n  => We can prove exactly which remembered fact and which routed model")
    print("     produced this answer. This is what generic memory proxies do not expose.")


def main() -> None:
    try:
        requests.get(f"{PROXY}/health", timeout=5)
    except Exception:
        raise SystemExit(f"Proxy not reachable at {PROXY}. Start it: python proxy.py")

    print(f"MODEL_A (session 1): {MODEL_A}")
    print(f"MODEL_B (session 2): {MODEL_B}")

    no_mem = run_arm_no_memory()
    proxy = run_arm_proxy()
    if proxy.get("outcome_ref"):
        show_audit_trail(proxy["entity"], proxy["outcome_ref"])

    print("\n=== SUMMARY ===")
    for r in (no_mem, proxy):
        print(f"  {r['arm']:10} remembered={r['remembered']!s:5} "
              f"routed={r['routed_model']} latency={r['latency_ms']}ms")

    results = {"model_a": MODEL_A, "model_b": MODEL_B, "no_memory": no_mem, "proxy": proxy}
    out = Path(__file__).parent / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
