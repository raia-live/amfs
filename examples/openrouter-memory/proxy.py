"""Memory-augmenting OpenRouter proxy (validation spike — shape C).

An OpenAI-compatible ``POST /v1/chat/completions`` endpoint. Point your client's
``base_url`` at this proxy instead of OpenRouter and memory "follows" the agent
across whatever model OpenRouter routes to.

This spike exists to test AMFS's *differentiator* vs. incumbents (Supermemory,
mem0): not generic recall, but an auditable, versioned decision trace that records
**which remembered facts + which routed model produced each answer**.

Scoping is via explicit headers (design decision):
    X-AMFS-Agent    agent identity          (default: "default-agent")
    X-AMFS-Entity   memory scope / path     (default: "openrouter/session")

The OpenRouter key is taken from the inbound ``Authorization: Bearer ...`` header
if present, otherwise from the ``OPENROUTER_API_KEY`` env var. It is never written
to disk.

Run:
    OPENROUTER_API_KEY=sk-or-... python proxy.py     # serves on 127.0.0.1:8088
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from amfs import AgentMemory, MemoryType, OutcomeType
from amfs_filesystem import FilesystemAdapter

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DATA_DIR = Path(os.environ.get("AMFS_PROXY_DATA_DIR", Path(__file__).parent / ".data"))
AUDIT_LOG = Path(os.environ.get("AMFS_PROXY_AUDIT_LOG", Path(__file__).parent / "audit.jsonl"))
MAX_INJECT = int(os.environ.get("AMFS_PROXY_MAX_INJECT", "5"))

app = FastAPI(title="AMFS memory-augmenting OpenRouter proxy")

# One shared adapter/root for the whole proxy; AgentMemory is created per-request
# so each call gets its own causal read-tracker (its own session/trace).
_adapter = FilesystemAdapter(DATA_DIR, namespace="openrouter_proxy")


def _memory(agent_id: str) -> AgentMemory:
    return AgentMemory(agent_id=agent_id, adapter=_adapter)


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):  # multimodal content parts
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return str(content)
    return ""


def _rank_memories(entries: list, query: str, k: int) -> list:
    """Cheap lexical ranking so the spike needs no embedder.

    Overlap of lowercased word sets between the query and each stored value.
    """
    q_words = {w.strip(".,!?;:").lower() for w in query.split() if len(w) > 2}
    scored = []
    for e in entries:
        val = e.value if isinstance(e.value, str) else json.dumps(e.value)
        v_words = {w.strip(".,!?;:").lower() for w in val.split() if len(w) > 2}
        overlap = len(q_words & v_words)
        scored.append((overlap, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    # keep positive-overlap first, then fall back to most recent to seed context
    hits = [e for score, e in scored if score > 0][:k]
    if not hits:
        hits = [e for _, e in scored][:k]
    return hits


def _looks_like_statement(text: str) -> bool:
    """Naive write-back heuristic. Store declarative statements, skip questions."""
    t = text.strip()
    return len(t) > 15 and not t.endswith("?")


def _forward_to_openrouter(api_key: str, payload: dict, stream: bool):
    return requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/raia-live/amfs",
            "X-Title": "AMFS memory proxy spike",
        },
        data=json.dumps(payload),
        stream=stream,
        timeout=120,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    started = time.time()
    body = await request.json()

    agent_id = request.headers.get("X-AMFS-Agent", "default-agent")
    entity = request.headers.get("X-AMFS-Entity", "openrouter/session")
    auth = request.headers.get("Authorization", "")
    api_key = auth[7:].strip() if auth.lower().startswith("bearer ") else os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return JSONResponse({"error": "No OpenRouter API key (Authorization header or OPENROUTER_API_KEY)."}, status_code=401)

    messages = list(body.get("messages", []))
    requested_model = body.get("model", "")
    stream = bool(body.get("stream", False))
    user_msg = _last_user_message(messages)

    mem = _memory(agent_id)

    # 1) RETRIEVE — pull scoped memory, inject as a system message.
    all_entries = mem.search(entity_path=entity, limit=100)
    injected = _rank_memories(all_entries, user_msg, MAX_INJECT) if all_entries else []
    injected_meta = []
    if injected:
        lines = []
        for e in injected:
            # explicit read → registers a causal link for the decision trace
            mem.read(entity, e.key)
            val = e.value if isinstance(e.value, str) else json.dumps(e.value)
            lines.append(f"- {val}")
            injected_meta.append({"key": e.key, "version": e.version, "value": val})
        mem_block = (
            "Relevant remembered facts about this agent/user (from persistent memory):\n"
            + "\n".join(lines)
            + "\n\nUse these facts if they are relevant to the user's request."
        )
        messages = [{"role": "system", "content": mem_block}] + messages

    payload = {**body, "messages": messages, "usage": {"include": True}}

    # 2) FORWARD to OpenRouter.
    resp = _forward_to_openrouter(api_key, payload, stream)

    if stream:
        # Relay chunks; accumulate assistant text + routed model for write-back.
        def _gen():
            acc_text, routed = [], {"model": requested_model}
            for raw in resp.iter_lines():
                if not raw:
                    continue
                yield raw + b"\n\n"
                line = raw.decode("utf-8", "ignore")
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        chunk = json.loads(line[6:])
                        routed["model"] = chunk.get("model", routed["model"])
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                        if delta:
                            acc_text.append(delta)
                    except Exception:
                        pass
            _record(mem, entity, agent_id, user_msg, "".join(acc_text),
                    requested_model, routed["model"], None, started, injected_meta)
        return StreamingResponse(_gen(), media_type="text/event-stream")

    if resp.status_code != 200:
        return JSONResponse({"error": "OpenRouter error", "detail": resp.text[:500]}, status_code=resp.status_code)

    data = resp.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    routed_model = data.get("model", requested_model)
    usage = data.get("usage", {}) or {}

    outcome_ref = _record(
        mem, entity, agent_id, user_msg, answer,
        requested_model, routed_model, usage, started, injected_meta,
    )

    # Expose the audit hooks as response headers (OpenAI clients ignore extras).
    headers = {
        "X-AMFS-Outcome-Ref": outcome_ref,
        "X-AMFS-Routed-Model": str(routed_model),
        "X-AMFS-Memory-Injected": str(len(injected_meta)),
        "X-AMFS-Cost-USD": str(usage.get("cost", "")),
    }
    return JSONResponse(data, headers=headers)


def _record(mem, entity, agent_id, user_msg, answer, requested_model,
            routed_model, usage, started, injected_meta) -> str:
    """Write-back + record the causal decision trace (the differentiator)."""
    ts_ms = int(time.time() * 1000)
    outcome_ref = f"req-{ts_ms}"
    usage = usage or {}

    # 3) WRITE-BACK (naive extraction — quality is deliberately out of scope here).
    written = None
    if _looks_like_statement(user_msg):
        key = f"obs-{ts_ms}"
        mem.write(entity, key, user_msg, memory_type=MemoryType.EXPERIENCE, confidence=0.8)
        written = {"key": key, "value": user_msg}

    # 4) ATTRIBUTION — record which model actually served this call.
    mem.record_context(
        "routed-model",
        f"model={routed_model} cost_usd={usage.get('cost')} "
        f"tokens={usage.get('prompt_tokens')}+{usage.get('completion_tokens')}",
        source="openrouter",
    )

    # 5) AUDIT TRAIL — explain() reconstructs this call's causal chain live
    #    (what was read + external contexts). We persist it so the benchmark can
    #    reconstruct "answer -> which memory -> which routed model" after the fact.
    explain = mem.explain(outcome_ref)
    latency_ms = round((time.time() - started) * 1000, 1)
    audit_record = {
        "outcome_ref": outcome_ref,
        "agent_id": agent_id,
        "entity": entity,
        "requested_model": requested_model,
        "routed_model": routed_model,
        "cost_usd": usage.get("cost"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_ms": latency_ms,
        "memory_injected": injected_meta,
        "memory_written": written,
        "user_msg": user_msg,
        "answer_excerpt": (answer or "")[:300],
        "causal_entries": explain.get("causal_entries", []),
        "external_contexts": explain.get("external_contexts", []),
    }
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_record, default=str) + "\n")

    # 6) Snapshot AMFS's native decision trace + outcome back-propagation.
    mem.commit_outcome(
        outcome_ref,
        OutcomeType.SUCCESS,
        decision_summary=f"Answered via {routed_model} using {len(injected_meta)} remembered fact(s)",
    )
    return outcome_ref


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "data_dir": str(DATA_DIR), "audit_log": str(AUDIT_LOG)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8088")))
