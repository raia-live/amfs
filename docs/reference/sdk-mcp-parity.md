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
| List traces | `amfs_list_traces` | `list_traces` | `listTracesAsync` |
| Get trace | `amfs_get_trace` | `get_trace` | `getTraceAsync` |

### Spans & trace navigation (Pro)

These ship in the **Pro MCP server** (`amfs-mcp-server-pro`) only. The recording tools write into the current session through the span recorder and are sealed by `amfs_commit_outcome`, so they work against any adapter — local or HTTP. The navigation tools read sealed traces: with `AMFS_HTTP_URL` set they proxy to `/api/v1/pro/traces/...` (indexed, semantic search); without it they fall back to the local adapter's `list_traces` plus the trace the process committed last (`trace_id="last"`), filter in Python, and answer with `"mode": "local"`.

| Capability | MCP tool (Pro) | Python SDK | TypeScript SDK |
|:--|:--|:--|:--|
| Record a completed span | `amfs_record_span` | `SpanRecorder.record_span` (`amfs_traces`) | — |
| Open / close a span | `amfs_start_span` / `amfs_end_span` | `SpanRecorder.start_span` / `end_span`, `with recorder.span(...)` | — |
| Trace attribute bag | `amfs_set_trace_attributes` | `SpanRecorder.set_attributes` | — |
| Record an LLM call (also emits an `llm` span) | `amfs_record_llm_call` | `SpanRecorder.record_llm_call` | — |
| Search traces by content, attributes, span name/kind, errors | `amfs_trace_search` | — (HTTP `POST /api/v1/pro/traces/search`) | — |
| Span tree with text outline | `amfs_trace_spans` | — (HTTP `GET /api/v1/pro/traces/{id}/spans`) | — |
| One span's input/output payloads | `amfs_span_payload` | — (HTTP `GET /api/v1/pro/traces/{id}/spans/{span_id}`) | — |
| Compare two traces span by span | `amfs_trace_compare` | — (HTTP `POST /api/v1/pro/traces/compare`) | — |

{: .note }
The TypeScript SDK's `*Async` methods route through the `HttpAdapter` to a remote AMFS server. The synchronous methods operate on the in-process adapter. Construct `AgentMemory` with an `HttpAdapter` to use the async surface — the intended path for a Node orchestrator or a disposable sandbox pointing at hosted SenseLab.

## Identity

Agent identity in the SDKs is set via the `AgentMemory` constructor (`agent_id`). **Sticky identity** (`amfs_set_identity` / `amfs_whoami` / `amfs_reset_identity`) is an MCP-server concept: it persists the last identity to `~/.amfs/.identity-<client>` so a fresh MCP process auto-restores it. The SDKs don't own that on-disk state, so they intentionally don't expose set/whoami/reset — pass `agent_id` explicitly instead.

---

## Deferred and Pro-only surface

These MCP tools are **not** mirrored 1:1 in the SDKs, by design. They are either Pro/Cortex features proxied over HTTP with no OSS adapter method, or they depend on backend endpoints that OSS does not ship.

| MCP tool | Status | Rationale |
|:--|:--|:--|
| `amfs_consolidate`, `amfs_consolidation_status/proposals/candidates` | Pro (HTTP-proxied) | Backed by the Cortex consolidation API, not the OSS adapter ABC. The TypeScript `HttpAdapter` exposes `consolidation*Async` passthroughs; the Python SDK reaches them via the HTTP API directly. See the caveat below. |
| `amfs_export_training_data` | Pro (HTTP-proxied) | Calls `GET /api/v1/pro/export`. Exposed on the TS `HttpAdapter` as `exportTrainingDataAsync`; Python calls the endpoint directly. |
| `amfs_record_span`, `amfs_start_span`, `amfs_end_span`, `amfs_set_trace_attributes`, `amfs_record_llm_call` | Pro (works locally and over HTTP) | Spans live on the Pro `ImmutableDecisionTrace`; the OSS `DecisionTrace` carries them only as `session_metadata` extras that the seal path lifts out. The Python side is the `amfs_traces.spans.SpanRecorder` in the private repo, not an `AgentMemory` method; the TS SDK has no span recorder yet. |
| `amfs_trace_search`, `amfs_trace_spans`, `amfs_span_payload`, `amfs_trace_compare` | Pro (HTTP-proxied, local fallback) | Backed by the Pro traces API (`/api/v1/pro/traces/search`, `/{id}/spans`, `/{id}/spans/{span_id}`, `/compare`). The local fallback has no semantic search or paging and can only see traces the adapter persists plus the last one committed in-process. No SDK methods yet; call the endpoints through the `HttpAdapter`. |
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
