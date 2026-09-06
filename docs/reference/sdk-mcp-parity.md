---
title: SDK ↔ MCP Parity
layout: default
parent: Reference
nav_order: 3
description: "Which AMFS capabilities are available in the Python SDK, the TypeScript SDK, and over MCP — plus what's intentionally deferred."
---

# SDK ↔ MCP Parity
{: .no_toc }

AMFS exposes memory three ways: the **MCP server** (for agents), the **Python SDK**, and the **TypeScript SDK**. This page maps the capability surface across all three so you know where each operation lives, and records the deliberate boundaries.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Core memory

| Capability | MCP tool | Python SDK | TypeScript SDK |
|:--|:--|:--|:--|
| Read exact key | `amfs_read` / `amfs_recall` | `read` / `recall` | `read` / `recall`, `readAsync` |
| Write | `amfs_write` | `write` | `write`, `writeAsync` |
| List | `amfs_list` / `amfs_my_entries` | `list` / `my_entries` | `list`, `listAsync` / `myEntries` |
| Search (filters) | `amfs_search` | `search` | `search`, `searchAsync` |
| Semantic retrieve | `amfs_retrieve` | `retrieve` | `retrieveAsync` |
| Briefing (Cortex) | `amfs_briefing` | `briefing` | `briefingAsync` |
| Cross-agent read | `amfs_read_from` | `read_from` | `readFrom` |
| History | `amfs_history` | `history` | `history`, `historyAsync` |
| Stats | `amfs_stats` | `stats` | `stats`, `statsAsync` |
| Graph neighbors | `amfs_graph_neighbors` | `graph_neighbors` (adapter) | `graphNeighborsAsync` |
| Bulk export | `amfs_export` | — (use `list`) | — |
| Server-side aggregate | `amfs_aggregate` | — (HTTP `POST /api/v1/aggregate`) | — |

## Tracing & outcomes

| Capability | MCP tool | Python SDK | TypeScript SDK |
|:--|:--|:--|:--|
| Commit outcome | `amfs_commit_outcome` | `commit_outcome` | `commitOutcome`, `commitOutcomeAsync` |
| Record context | `amfs_record_context` | `record_context` | `recordContext` |
| Record action | `amfs_record_action` | `record_action` | `recordAction` (tracker) |
| Explain session | `amfs_explain` | `explain` | `explain` |
| Timeline | `amfs_timeline` | `timeline` | `timelineAsync` |
| List traces | `amfs_list_traces` | `list_traces` (cursor / since / until) | `listTracesAsync`, `listTracesPageAsync` (`{traces, nextCursor, hasMore}`) |
| Get trace | `amfs_get_trace` | `get_trace` | `getTraceAsync` |
| Session attributes on the trace | `amfs_set_trace_attributes` (Pro) | `set_session_attributes`, `commit_outcome(attributes=...)` | `setSessionAttributes`, `commitOutcome(..., {attributes})`, `commitOutcomeAsync(..., {attributes})` |
| Record an LLM call by hand | `amfs_record_llm_call` (Pro) | `record_llm_call` | `recordLlmCall` |
| Record LLM calls automatically | — | `amfs_pro.tracing.instrument_openai` / `instrument_anthropic` / `instrument_litellm` (Pro) | `instrumentOpenAI`, `instrumentAnthropic` |

### Capturing runs for the Behavior section

The dashboard's Behavior section shows, per run, a duration, a span tree, token counts, a cost and the attributes the run can be grouped by. None of that exists unless the SDK sends it: a trace committed by an uninstrumented agent reaches the server flat — blank duration, no spans, no attributes, and a cost it cannot know. What each figure needs:

| Figure | Captured | How |
|:--|:--|:--|
| Session duration (`session_started_at` / `session_ended_at` / `session_duration_ms`) | **Automatically** | Every `commit_outcome` / `commitOutcomeAsync` — the session starts at the first memory operation. |
| Memory operations as spans (reads, writes, `record_action`, `record_context`) | **Automatically inside `trace_session`** (Python, Pro) | `with trace_session(mem): ...` — the recorder mirrors the memory's own operations as `memory_read` / `memory_write` / `tool` / `context` spans under a `session` root, so even a run that records nothing else gets a real tree and duration. The Pro MCP server does the same for MCP agents. |
| LLM calls — model, tokens, latency | **One line** | Python (Pro): `client = instrument_openai(OpenAI())` / `instrument_anthropic(Anthropic())` / `instrument_litellm()`. TypeScript: `const openai = instrumentOpenAI(new OpenAI(), memory)` / `instrumentAnthropic(client, memory)`. Sync, async and streaming calls are all recorded; an instrumentation failure never breaks the call. Any other provider: `record_llm_call(model, input_tokens, output_tokens, ...)` / `memory.recordLlmCall({...})`. |
| Cost | **Only when LLM calls are recorded** | The Python instrumentation estimates `cost_usd` from a small price table (`AMFS_LLM_PRICE_TABLE` overrides it; LiteLLM's registry is used when `litellm` is already imported); a model it cannot price records `None`. The TypeScript SDK records tokens and leaves `cost_usd` null unless you pass `costUsd`. A run with no recorded LLM calls has an **unknown** cost, shown as a dash — never `$0.00`. |
| Your own steps (`plan`, `pagerduty.list_incidents`, ...) | **By the developer** (Python, Pro) | `with span("plan", kind="agent"):` / `@traced("fetch", kind="tool")`; exceptions mark the span `error` and propagate. Payloads are redacted and capped at 32 000 characters (`AMFS_SPAN_PAYLOAD_MAX_CHARS`). |
| Attributes (`customer`, `task_type`, `environment`, ...) | **By the developer** | `trace_session(mem, attributes={...})`, `mem.set_session_attributes({...})`, `mem.commit_outcome(..., attributes={...})`; TS: `memory.setSessionAttributes({...})` or `commitOutcomeAsync(..., { attributes })`. Scalars only, at most 20 keys of up to 64 characters, string values up to 256 characters; keys are lowercased. `source`, `model`, `client` and `outcome_type` are reserved and stamped by the server. |

```python
from amfs import AgentMemory
from amfs_core.models import OutcomeType
from amfs_pro.tracing import trace_session, span, instrument_openai

client = instrument_openai(OpenAI())
with trace_session(mem, attributes={"customer": "acme", "task_type": "deploy"}):
    with span("plan", kind="agent"):
        plan = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    with span("pagerduty.list_incidents", kind="tool", input={"service": "checkout"}) as s:
        s.set_output(incidents)
    mem.commit_outcome("deploy-42", OutcomeType.SUCCESS, task_input="roll back checkout")
```

```ts
const openai = instrumentOpenAI(new OpenAI(), memory);
memory.setSessionAttributes({ customer: "acme", task_type: "deploy" });
const r = await openai.chat.completions.create({ model: "gpt-4o-mini", messages });
await memory.commitOutcomeAsync("deploy-42", OutcomeType.SUCCESS);
```

Everything travels in `DecisionTrace.session_metadata` — `attributes` (a flat bag), `llm_calls` (`{call_id, model, provider, input_tokens, output_tokens, cost_usd, latency_ms, started_at}`) and, from `amfs_pro.tracing`, `spans` — and the server lifts those keys onto the sealed trace. OSS keeps only the generic parts: the two `AgentMemory` methods, the `attributes` argument, and the TypeScript instrumentation; cost estimation, span trees and the Python instrumentation are `amfs-sdk-pro`. Inspect the result with `amfs-pro traces show <trace_id>` or `amfs-pro traces search --agent X --attr customer=acme`.

### Spans & trace navigation (Pro)

These ship in the **Pro MCP server** (`amfs-mcp-server-pro`) only. The recording tools write into the current session through the span recorder and are sealed by `amfs_commit_outcome`, so they work against any adapter — local or HTTP. The navigation tools read sealed traces: with `AMFS_HTTP_URL` set they proxy to `/api/v1/pro/traces/...` (indexed, semantic search); without it they fall back to the local adapter's `list_traces` plus the trace the process committed last (`trace_id="last"`), filter in Python, and answer with `"mode": "local"`.

| Capability | MCP tool (Pro) | Python SDK | TypeScript SDK |
|:--|:--|:--|:--|
| Trace one run (root `session` span, memory ops mirrored as child spans) | — (the Pro MCP server does this per session) | `amfs_pro.tracing.trace_session(mem, attributes=...)` | — |
| Record a completed span | `amfs_record_span` | `SessionRecorder.record_span` (`amfs_pro.tracing`), `SpanRecorder.record_span` (`amfs_traces`) | — |
| Open / close a span | `amfs_start_span` / `amfs_end_span` | `with span("plan", kind="agent")`, `@traced(name, kind)` (`amfs_pro.tracing`); `SpanRecorder.start_span` / `end_span` | — |
| Trace attribute bag | `amfs_set_trace_attributes` | `trace_session(..., attributes=)`, `session.set_attributes`; OSS `set_session_attributes` | `setSessionAttributes` (OSS) |
| Record an LLM call (also emits an `llm` span) | `amfs_record_llm_call` | `amfs_pro.tracing.record_llm_call`, `instrument_openai` / `instrument_anthropic` / `instrument_litellm`; OSS `record_llm_call` (no span) | `recordLlmCall`, `instrumentOpenAI` / `instrumentAnthropic` (OSS, no span) |
| Search traces by content, attributes, span name/kind, errors | `amfs_trace_search` | `ProClient.traces.search(attributes=..., span_kind=...)`; CLI `amfs-pro traces search --agent X --attr customer=acme` | — |
| Span tree with text outline | `amfs_trace_spans` | `ProClient.traces.spans`; CLI `amfs-pro traces show <trace_id>` | — |
| One span's input/output payloads | `amfs_span_payload` | — (HTTP `GET /api/v1/pro/traces/{id}/spans/{span_id}`) | — |
| Compare two traces span by span | `amfs_trace_compare` | — (HTTP `POST /api/v1/pro/traces/compare`) | — |

{: .note }
The TypeScript SDK's `*Async` methods route through the `HttpAdapter` to a remote AMFS server. The synchronous methods operate on the in-process adapter. Construct `AgentMemory` with an `HttpAdapter` to use the async surface — the intended path for a Node orchestrator or a disposable sandbox pointing at hosted SenseLab.

{: .note }
**Paging traces in TypeScript.** `listTracesAsync` accepts `{ cursor, since, until, limit, offset, entityPath, agentId, outcomeType }` and still returns a flat array. `listTracesPageAsync` takes the same options and returns `{ traces, nextCursor, hasMore }`, matching the OSS server's `GET /api/v1/traces` page shape — follow `nextCursor` while `hasMore` is true. `DecisionTrace` now carries `taskInput`, `responseText`, `toolCalls[]` (with the caller's `arguments` keys left as-is) and `sessionMetadata`, mapped from the server's snake_case fields; the mappers are exported as `toDecisionTrace` / `toDecisionTracePage`.

### Agent evaluation (Pro, hosted only)

Judges, verdicts, behaviors, incidents, the fix loop and the investigator are **Pro, hosted-only** surfaces served by `/api/v1/eval` on SenseLab Cloud. They ship in three developer-facing forms, none of which live in the OSS SDKs:

- **Pro MCP tools** (`amfs-mcp-server-pro`, `tags={"extended"}`): proxy to the hosted API when `AMFS_HTTP_URL` is set and refuse with `{"error": "...requires the hosted API (AMFS_HTTP_URL)...", "mode": "local"}` otherwise. They are excluded from the builder profile.
- **`amfs_pro` Python client** (private dist `amfs-sdk-pro`): `ProClient(base_url=None, api_key=None)` reads `AMFS_HTTP_URL` / `AMFS_API_KEY`, sends `X-AMFS-API-Key`, and exposes `client.eval` and `client.traces`; `AsyncProClient` mirrors it on `httpx.AsyncClient`. List calls return `Page[...]` (`items`, `next_cursor`, `has_more`) and have `iter*` helpers that follow cursors.
- **`amfs-pro` CLI** (installed with `amfs-sdk-pro`): every command takes `--json` for CI; `eval backfill` and `cases run` exit non-zero with `--fail-on-fail`.

| Capability | MCP tool (Pro) | `amfs_pro` client | `amfs-pro` CLI |
|:--|:--|:--|:--|
| Grade one trace now | `amfs_judge_trace` | `eval.judge` | `eval judge <trace_id> <judge_id>` |
| Define a judge (optionally dry-run first) | `amfs_judge_define` | `eval.judges.create` / `.update` / `.dry_run` / `.history` / `.diff` | `eval judges create --id --name --prompt [--dry-run]`, `eval judges dry-run <agent> <judge>` |
| List judges | `amfs_judges` | `eval.judges.list` / `.get` / `.delete` / `.detection_rate` | `eval judges list <agent>` |
| Install the starter pack | `amfs_seed_starter_judges` | `eval.judges.seed_starter_pack` | `eval judges seed <agent>` |
| Browse verdicts | `amfs_verdicts` | `eval.verdicts.list` / `.iter` / `.get` / `.for_trace` | — (use `--json` on `eval judge` / `backfill`) |
| Agent summary | `amfs_eval_summary` | `eval.summary` | `eval summary <agent> [--window]` |
| Backfill past traces (spends budget) | `amfs_backfill` | `eval.backfill` / `eval.backfill_status` | `eval backfill <agent> <judge> [--since --until --limit --sampling-rate --wait --fail-on-fail]` |
| Judge budget | — | `eval.budget.get` / `.set` | — |
| Behaviors | `amfs_behaviors`, `amfs_behavior_define`, `amfs_behavior_from_trace` | `eval.behaviors.list` / `.create` / `.from_trace` / `.get` / `.update` / `.delete` / `.traces` | `behaviors list <agent>`, `behaviors from-trace <trace_id>` |
| Blast radius of a behavior | `amfs_blast_radius` | `eval.behaviors.blast_radius` | — |
| Incidents | `amfs_incident` (list / get / `action=acknowledge\|resolve\|mute`) | `eval.incidents.list` / `.get` / `.acknowledge` / `.resolve` / `.mute` | `incidents list [--agent --status]`, `incidents ack <id>`, `incidents mute <id> [--until]` |
| Segment by dimension | `amfs_segment`, `amfs_dimensions` | `eval.segment`, `eval.dimensions` | — |
| Attribute a failure to memory entries | `amfs_attribute_failure` | `eval.attribution` | — |
| Fix loop | `amfs_propose_fix` (propose, then returns the handoff), `amfs_fixes`, `amfs_approve_fix_memory` | `eval.fixes.propose` / `.handoff` / `.list` / `.get` / `.approve_memory` / `.dismiss` / `.prevention` / `.prevention_for` | `fixes list <agent>`, `fixes propose <agent> --verdict\|--behavior`, `fixes handoff <fix_id>` (prints `instructions` markdown for piping into a coding agent) |
| Regression cases | — | `eval.cases.sets` / `.create_set` / `.get_set` / `.cases` / `.run` (inline or queued) / `.get_run` / `.runs` / `.compare` | `cases run <set_id> --label [--since --compare-to --fail-on-fail]` |
| Investigate a question across runs | `amfs_investigate` (non-streaming) | `eval.investigate`, `eval.investigate_stream` (SSE events) | `investigate <agent> "<question>" [--trace --max-steps --no-stream]` (streams steps) |
| Alerts | `amfs_alerts` | `eval.alerts.list` / `.ack` / `.mute` / `.rules` | — |
| Sealed traces | `amfs_trace_search`, `amfs_trace_spans`, `amfs_span_payload`, `amfs_trace_compare` | `traces.search` / `.iter_search` / `.get` / `.spans` / `.span` / `.compare` / `.otlp_export` | `traces search [--query --agent --attr key=value --span-name --span-kind --since --until --has-error --outcome --limit --cursor]`, `traces show <trace_id>` (span tree with durations, tokens, cost, attributes; unknown cost and duration print as `—`) |

Where the Python SDK column above says "—", the same routes are reachable by any HTTP client with the `X-AMFS-API-Key` header; the `amfs_pro` client is the supported wrapper.

## Identity

Agent identity in the SDKs is set via the `AgentMemory` constructor (`agent_id`). **Sticky identity** (`amfs_set_identity` / `amfs_whoami` / `amfs_reset_identity`) is an MCP-server concept: it persists the last identity to `~/.amfs/.identity-<client>` so a fresh MCP process auto-restores it. The SDKs don't own that on-disk state, so they intentionally don't expose set/whoami/reset — pass `agent_id` explicitly instead.

---

## Deferred and Pro-only surface

These MCP tools are **not** mirrored 1:1 in the SDKs, by design. They are either Pro/Cortex features proxied over HTTP with no OSS adapter method, or they depend on backend endpoints that OSS does not ship.

| MCP tool | Status | Rationale |
|:--|:--|:--|
| `amfs_consolidate`, `amfs_consolidation_status/proposals/candidates` | Pro (HTTP-proxied) | Backed by the Cortex consolidation API, not the OSS adapter ABC. The TypeScript `HttpAdapter` exposes `consolidation*Async` passthroughs; the Python SDK reaches them via the HTTP API directly. See the caveat below. |
| `amfs_export_training_data` | Pro (HTTP-proxied) | Calls `GET /api/v1/pro/export`. Exposed on the TS `HttpAdapter` as `exportTrainingDataAsync`; Python calls the endpoint directly. |
| `amfs_record_span`, `amfs_start_span`, `amfs_end_span`, `amfs_set_trace_attributes`, `amfs_record_llm_call` | Pro (works locally and over HTTP) | Spans live on the Pro `ImmutableDecisionTrace`; the OSS `DecisionTrace` carries them only as `session_metadata` extras that the seal path lifts out. The Python side is `amfs_pro.tracing` (`trace_session`, `span`, `traced`, `record_llm_call`, `instrument_*`) in the private `amfs-sdk-pro` dist; OSS `AgentMemory` has only the generic `set_session_attributes` / `record_llm_call` / `commit_outcome(attributes=...)`. The TS SDK has `setSessionAttributes`, `recordLlmCall` and `instrumentOpenAI` / `instrumentAnthropic` but no span recorder. |
| `amfs_trace_search`, `amfs_trace_spans`, `amfs_span_payload`, `amfs_trace_compare` | Pro (HTTP-proxied, local fallback) | Backed by the Pro traces API (`/api/v1/pro/traces/search`, `/{id}/spans`, `/{id}/spans/{span_id}`, `/compare`). The local fallback has no semantic search or paging and can only see traces the adapter persists plus the last one committed in-process. No SDK methods yet; call the endpoints through the `HttpAdapter`. |
| `amfs_judge_trace`, `amfs_judge_define`, `amfs_judges`, `amfs_seed_starter_judges`, `amfs_verdicts`, `amfs_eval_summary`, `amfs_backfill`, `amfs_behaviors`, `amfs_behavior_define`, `amfs_behavior_from_trace`, `amfs_blast_radius`, `amfs_incident`, `amfs_segment`, `amfs_dimensions`, `amfs_attribute_failure`, `amfs_propose_fix`, `amfs_fixes`, `amfs_approve_fix_memory`, `amfs_investigate`, `amfs_alerts` | Pro (HTTP-proxied, **no local mode**) | Agent evaluation runs on the hosted `/api/v1/eval` service (LLM judges, budgets, behavior mining). Without `AMFS_HTTP_URL` the tools answer `{"mode": "local"}` and do nothing. The OSS SDKs have no methods for these; use the private `amfs_pro` client or the `amfs-pro` CLI (see *Agent evaluation* above). |
| `amfs_room_*`, `amfs_negotiate_*` | Pro (rooms backend) | Multi-agent rooms/negotiation require the Pro rooms service. The TS SDK has `room*`/`negotiate*` methods; `amfs_negotiate_cancel` has **no OSS backend endpoint**, so it's not added as an SDK method (it would be a broken stub). |
| `amfs_verify`, `amfs_commit_batch`, `amfs_merge_base` | SDK-covered | Present in the Python SDK (`verify`, `transaction`, `common_ancestor`) and TS SDK (`verify`, `transaction`, `commonAncestor`). |
| `amfs_export` | MCP-only | Spools full untruncated values to a **local file** on the MCP client's machine — a file-system convenience that only makes sense in-process. SDK callers already have `list`/`read` returning full values with no truncation, so there is nothing to mirror. |
| `amfs_aggregate` | MCP-backed by OSS endpoint | Reduces server-side via `POST /api/v1/aggregate` (visibility-filtered before the reduce). SDK callers can hit the same endpoint through the `HttpAdapter`; a dedicated `aggregate` method is not yet exposed. |

{: .warning }
**Consolidation contract caveat.** Two consolidation tools disagree with the API they proxy: `amfs_consolidate(dry_run=True)` is not honoured as a preview by the handler (it performs a real run), and `amfs_consolidation_candidates` requires an `entity_path` the tool doesn't always send. Until the server contract is fixed, prefer `amfs_consolidation_candidates(entity_path=...)` for previews and avoid relying on `dry_run`. SDK wrappers forward these params but inherit the same server behavior.

---

## See also

- [Python SDK guide](/amfs/guides/python/)
- [TypeScript SDK guide](/amfs/guides/typescript/)
- [MCP Server guide](/amfs/guides/mcp/)
