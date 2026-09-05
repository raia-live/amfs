---
title: OSS vs Pro
layout: default
nav_order: 4
description: "What's in AMFS open-source vs. AMFS Pro — and when you need which."
permalink: /editions/
---

# OSS vs Pro
{: .no_toc }

AMFS is GitHub for agent memory, split into two layers. The open-source core gives you the full memory engine with a single-branch repo model — versioning, confidence, outcomes, causal traces, knowledge graph, and a git-like timeline. Pro unlocks the full Git model: branching, merge, pull requests, access control, tags, rollback, cherry-pick, fork — plus multi-tenant SaaS, immutable traces, pattern detection, intelligence, and a dashboard.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## At a Glance

```
┌──────────────────────────────────────────────────────────┐
│                    AMFS Pro (Proprietary)                  │
│                                                          │
│  ┌─ Memory Branching (Git for Agent Memory) ───────────┐ │
│  │  Branches · Merge · PRs · Diff · Access Control      │ │
│  │  Tags/Snapshots · Rollback · Cherry-pick · Fork      │ │
│  │  Branch-Level Permissions (user/team/API key)        │ │
│  │  Sacred Timeline (3D visualization)                   │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Multi-Tenant SaaS ─────────────────────────────────┐ │
│  │  Accounts · RBAC · Scoped API Keys · Audit · Quotas │ │
│  │  Row-Level Security · OAuth/OIDC · Rate Limiting     │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Immutable Decision Trace Store ────────────────────┐ │
│  │  Durable Causal Chains · Cryptographic Integrity     │ │
│  │  LLM Call Spans · Token/Cost Tracking · Replay       │ │
│  │  Enriched Traces · Error Events · State Diff         │ │
│  │  OpenTelemetry Export · Precedent Search              │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Cross-System Ingestion ───────────────────────────┐   │
│  │  Webhooks · HMAC Verification · Deduplication        │ │
│  │  PagerDuty · Slack · GitHub · Jira Connectors        │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Automated Pattern Detection ──────────────────────┐   │
│  │  Recurring Failures · Hot Entities · Stale Clusters  │ │
│  │  Confidence Drift · Configurable Alert Rules         │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Intelligence Layer ─────────────────────────────────┐ │
│  │  Extraction · Critic · Distiller · Safety · Retrieval │ │
│  │  Learned Ranking · Confidence Calibration · ML Export │ │
│  │  LLM Importance Scoring · TierWorker (background)    │ │
│  │  Auto Entity/Relationship Extraction from Traces     │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Dashboard ──────────────────────────────────────────┐ │
│  │  Memory Explorer · Interactive Causal Graph           │ │
│  │  Trace Detail with Enriched Context · Agent Graph     │ │
│  │  Sacred Timeline (3D) · Branch Manager · PR Viewer   │ │
│  │  API Key Console · Audit Viewer · Usage Analytics    │ │
│  │  Cursor plugin (hosted MCP to Pro API)              │ │
│  │  Pattern Detection · Memory Tiers · Snapshots        │ │
│  │  Team Management                                    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                    AMFS OSS (Apache 2.0)                  │
│                                                          │
│  CoW Engine · Memory Types · Provenance Tiers             │
│  Git-like Timeline (event logging on main branch)        │
│  Branch-Aware Read/Write/List/Search (main default)      │
│  Temporal Queries · Session-Level Causal Explainability   │
│  Enriched Decision Traces · Query/Error Event Tracking    │
│  Composite Recall Scoring · Multi-Scope Search            │
│  Knowledge Graph (auto-materialized edges)               │
│  Hybrid Search (full-text + semantic + composite)         │
│  Tiered Memory (Hot/Warm/Archive) · Priority Scoring      │
│  Frequency-Modulated Decay · Cortex Drift Gate            │
│  ImportanceEvaluator ABC · Progressive Retrieval          │
│  Per-Agent Memory Graph · Agent Activity Timeline         │
│  Connector Framework · CLI · Built-in Connectors          │
│  Webhook Receiver · Adapters (FS, Postgres, S3)           │
│  HTTP/REST API · MCP Server · Python SDK · TS SDK · CLI   │
│  Docker + Helm Charts                                     │
└──────────────────────────────────────────────────────────┘
```

---

## Feature Comparison

| Capability | OSS | Pro |
|:-----------|:---:|:---:|
| **Core Memory Primitives** | | |
| Copy-on-Write versioning | Yes | Yes |
| Confidence scoring with outcome back-propagation | Yes | Yes |
| Memory types (fact, belief, experience) | Yes | Yes |
| Type-specific confidence decay | Yes | Yes |
| Provenance tiers (production-validated → manual) | Yes | Yes |
| Temporal queries (`history`) | Yes | Yes |
| Session-level causal explainability (`explain`) | Yes | Yes |
| Enriched decision traces (query events, error events, state diff) | Yes | Yes |
| Session timing and per-operation latency tracking | Yes | Yes |
| Per-agent memory graph and activity timeline | Yes | Yes |
| Composite recall scoring | Yes | Yes |
| Multi-scope search | Yes | Yes |
| Knowledge graph (auto-materialized from writes/outcomes) | Yes | Yes |
| Hybrid search (full-text + semantic + composite scoring) | Yes | Yes |
| Frequency-modulated decay (4-signal model) | Yes | Yes |
| Tiered memory (Hot / Warm / Archive) | Yes | Yes |
| Progressive retrieval (`depth`) | Yes | Yes |
| Importance evaluator hook (`ImportanceEvaluator` ABC) | Yes | Yes |
| Cortex drift gate (skip-redundant recompilations) | Yes | Yes |
| LLM importance scoring (3-dimension) | — | Yes |
| Background tier recomputation (`TierWorker`) | — | Yes |
| **Git-like Timeline** | | |
| Event logging (writes, outcomes, reads, briefs) on main | Yes | Yes |
| Branch-aware read/write/list/search (`branch` parameter) | Yes | Yes |
| Timeline API endpoint per agent | Yes | Yes |
| MCP `amfs_timeline` tool | Yes | Yes |
| **Content Integrity** | | |
| SHA-256 content hashing on every write | Yes | Yes |
| Integrity chain linking across versions | Yes | Yes |
| `verify()` SDK + `amfs verify` CLI + `amfs_verify` MCP tool | Yes | Yes |
| Cryptographic signing + tamper-evident audit | — | Yes |
| **Atomic Multi-Key Commits** | | |
| Transaction buffer (`transaction()` context manager) | Yes | Yes |
| Commit model with ID, message, tree hash, parent linking | Yes | Yes |
| `commit_log()`, `get_commit()`, `amfs_commit_batch`, `amfs log` | Yes | Yes |
| Commit revert + dashboard commit grouping | — | Yes |
| **DAG Versioning** | | |
| Parent commit linking and common ancestor computation | Yes | Yes |
| Branch HEAD/base commit tracking | Yes | Yes |
| `amfs_merge_base` MCP tool, `POST /api/v1/merge-base` | Yes | Yes |
| Cherry-pick + Sacred Timeline DAG upgrade | — | Yes |
| **Agent-Memory Binding** | | |
| Agent profiles, capabilities, and memory contracts | Yes | Yes |
| Agent discovery by capability or entity path | Yes | Yes |
| Auto-computed capabilities + violation alerts | — | Yes |
| **Rich Diff & Patch** | | |
| Structural JSON diff with field-level changes (RFC 6901 paths) | Yes | Yes |
| Memory patches (create + apply) | Yes | Yes |
| `amfs_diff` MCP tool + HTTP diff/patch endpoints | Yes | Yes |
| Rich structural diffs in dashboard Branches/PRs | — | Yes |
| **Memory Branching (Pro)** | | |
| Create / close / list branches | — | Yes |
| Diff branch vs. main | — | Yes |
| Merge branches (fast-forward, ours, theirs) | — | Yes |
| Branch access control (grant/revoke per user/team/API key) | — | Yes |
| Access enforcement on non-main read/write | — | Yes |
| Pull Requests with review workflow | — | Yes |
| Tags / named snapshots | — | Yes |
| Rollback to tag or timestamp | — | Yes |
| Cherry-pick entries across branches | — | Yes |
| Fork agent memory to a new agent | — | Yes |
| Sacred Timeline (3D interactive visualization) | — | Yes |
| **Connectors** | | |
| Connector framework (`ConnectorABC`) | Yes | Yes |
| Connector CLI (`amfs connector install/list/remove`) | Yes | Yes |
| Built-in connectors (PagerDuty, GitHub, Slack, Jira) | Yes | Yes |
| Webhook receiver (`/api/v1/webhooks/{name}`) | Yes | Yes |
| **Adapters & Infrastructure** | | |
| Filesystem adapter | Yes | Yes |
| Postgres adapter (triggers, LISTEN/NOTIFY) | Yes | Yes |
| S3 adapter | Yes | Yes |
| HTTP/REST API server | Yes | Yes |
| Docker + Docker Compose + Helm charts | Yes | Yes |
| MCP server (12 tools) | Yes | Yes |
| **SDKs & Clients** | | |
| Python SDK (full parity) | Yes | Yes |
| TypeScript SDK (full parity) | Yes | Yes |
| CLI tools | Yes | Yes |
| Framework integrations (CrewAI, LangGraph, etc.) | Yes | Yes |
| Bundled lightweight embedder | Yes | Yes |
| **Multi-Tenant SaaS (Pro)** | | |
| Account-level tenant isolation (RLS) | — | Yes |
| RBAC (Admin, Developer, User) | — | Yes |
| Scoped API keys with entity-path permissions | — | Yes |
| Permissioned inference (scope-filtered search/explain) | — | Yes |
| OAuth 2.0 / OIDC for dashboard users | — | Yes |
| Append-only audit logging | — | Yes |
| Usage quotas + sliding-window rate limiting | — | Yes |
| **Immutable Decision Trace Store (Pro)** | | |
| Durable causal chains across sessions | — | Yes |
| Cryptographic signing (HMAC-SHA256) and Merkle chain linking | — | Yes |
| LLM call span tracking (model, tokens, cost, latency) | — | Yes |
| Write events, tool calls, and agent interaction recording | — | Yes |
| OpenTelemetry export (GenAI semantic conventions) | — | Yes |
| Historical `explain(outcome_ref)` | — | Yes |
| `search_traces` / precedent search API | — | Yes |
| Cross-agent, cross-session trace queries | — | Yes |
| **Cross-System Ingestion** | | |
| Generic webhook endpoint with HMAC verification | Yes | Yes |
| Payload deduplication (idempotency) | Yes | Yes |
| Pluggable transform pipeline with pattern matching | Yes | Yes |
| PagerDuty incident connector | Yes | Yes |
| Extensible connector framework (`ConnectorABC`) | Yes | Yes |
| **Automated Pattern Detection (Pro)** | | |
| Recurring failure detection across causal chains | — | Yes |
| Hot entity detection (disproportionate activity) | — | Yes |
| Stale cluster detection (unvalidated entries) | — | Yes |
| Confidence drift detection (outlier entries) | — | Yes |
| Configurable alert rules with cooldown suppression | — | Yes |
| Alert callbacks (Slack, PagerDuty, email routing) | — | Yes |
| **Intelligence Layer (Pro)** | | |
| LLM-driven memory extraction | — | Yes |
| Auto entity/relationship extraction from traces | — | Yes |
| Automated memory critic | — | Yes |
| Memory distillation & bootstrap sets | — | Yes |
| Memory safety validation | — | Yes |
| Multi-strategy retrieval with learned ranking (RRF + ML) | — | Yes |
| Learned retrieval ranking from outcome data | — | Yes |
| Adaptive confidence calibration | — | Yes |
| Training data export (SFT, DPO, reward model) | — | Yes |
| **Dashboard (Pro)** | | |
| Memory explorer with interactive causal graph | — | Yes |
| Decision trace explorer with enriched context and causal graph | — | Yes |
| Per-agent memory graph and activity timeline | — | Yes |
| Pattern detection dashboard with severity indicators | — | Yes |
| Team & user management with roles | — | Yes |
| API key management console with scopes | — | Yes |
| Audit log viewer with search/filter | — | Yes |
| Usage analytics & quota monitoring | — | Yes |
| Snapshot capture, compare, and export | — | Yes |
| Memory Tiers dashboard (distribution, entry browser) | — | Yes |
| Guided migration from Mem0 and Zep (preview, resume, undo) | — | Yes |
| Extended MCP server (Pro tools) | — | Yes |

---

## Hosted SaaS tiers (SenseLab Cloud)

**Stripe, subscriptions, and the tenant control plane** are implemented in the private SenseLab **`amfs-internal`** repository (not in the open-source tree). Self-serve tiers are summarized below; see also [Hosted billing & metering]({{ site.baseurl }}/guides/saas-billing-metering/).

| Tier | Indicative price | Included ops / month | Notes |
|:-----|:-----------------|:----------------------|:------|
| **Free** | $0 | 1K | Solo; no team invites; hard cap |
| **Starter** | $29 | 25K | Entry tier; auto top-up near limit (paid) |
| **Pro** | $149 | 50K | Production agent teams |
| **Teams** | $449 | 300K | Org scale; lower overage unit / 10K ops |
| **Enterprise** | Custom | Custom | SSO, committed volume, SLA |

**Ops (hosted):** read-like calls typically **1 op**, writes **2 ops**, `commit_outcome` **0 ops**. The HTTP + MCP mapping lives in the metering guide linked above.

**Onboarding:** Users must **choose a plan** (including Free) before the dashboard unlocks; paid plans complete **Stripe Checkout** and activation via **webhooks** (not browser success URL alone).

---

## OSS Layer — What's Included

The open-source layer ([github.com/raia-live/amfs](https://github.com/raia-live/amfs)) provides the full memory primitive: read, write, version, search, and learn from outcomes. It includes a **git-like timeline engine**, branch-aware operations, a connector framework, composite recall scoring, multi-scope search, a **knowledge graph** auto-materialized from writes and outcomes, **hybrid search** (full-text + semantic + composite scoring), **tiered memory** (Hot/Warm/Archive with progressive retrieval), **frequency-modulated decay** (4-signal model), and a **Cortex drift gate** that avoids redundant digest recompilations.

### Packages

| Package | Description |
|:--------|:------------|
| `amfs-core` | CoW engine, models, read tracking, causal tagging, default embedder, `tiering` (PriorityScorer, TierAssigner), `importance` (ImportanceEvaluator ABC) |
| `amfs` (SDK) | `AgentMemory` class — `read`, `write`, `list`, `search`, `history`, `timeline`, `explain`, `commit_outcome`, `record_context` |
| `amfs-adapter-filesystem` | JSON-file-based adapter for local development |
| `amfs-adapter-postgres` | PostgreSQL adapter with PL/pgSQL triggers for outcome propagation and `LISTEN/NOTIFY` for watch |
| `amfs-adapter-s3` | Amazon S3 / S3-compatible adapter for cloud-native storage |
| `amfs-http-server` | REST API server (FastAPI/Uvicorn) for remote access, branch-aware endpoints |
| `amfs-mcp-server` | MCP server exposing 12 tools: `amfs_read`, `amfs_write`, `amfs_search`, `amfs_retrieve`, `amfs_list`, `amfs_stats`, `amfs_commit_outcome`, `amfs_record_context`, `amfs_history`, `amfs_explain`, `amfs_graph_neighbors`, `amfs_timeline` |
| `amfs-cli` | Terminal tools for inspecting, diffing, snapshotting, and restoring memory |
| `@senselab-ai/amfs` | TypeScript SDK (full parity with Python: ReadTracker, search, stats, history, explain, recordContext) |

### Key Primitives

**Memory Types** — Every entry is classified as `fact`, `belief`, or `experience`, each with its own decay rate:

```python
from amfs import AgentMemory, MemoryType

mem = AgentMemory(agent_id="my-agent")

mem.write("svc", "config", {"pool": 10}, memory_type=MemoryType.FACT)
mem.write("svc", "hypothesis", "Likely N+1", memory_type=MemoryType.BELIEF)
mem.write("svc", "action-log", "Added index", memory_type=MemoryType.EXPERIENCE)
```

**Provenance Tiers** — Entries are automatically ranked by quality based on how they were created:

```python
from amfs import ProvenanceTier

entry = mem.read("svc", "pattern")
if entry.provenance_tier == ProvenanceTier.PRODUCTION_VALIDATED:
    print("Validated by production outcomes")
```

**Temporal Queries** — Retrieve the full version history of any entry:

```python
versions = mem.history("svc", "retry-pattern", since=last_week)
```

**Causal Explainability** — See which entries were read and how they connect to outcomes:

```python
chain = mem.explain()
```

---

## Pro Layer — What's Added

The Pro layer wraps the OSS layer — it never replaces it. All Pro features read from and write to the same memory store using the same adapters and SDK.

### Memory Branching — Git for Agent Memory

The defining Pro feature. While OSS provides the git-like timeline engine (event logging on `main`), Pro adds the full branching model — branches, merge, pull requests, access control, tags, rollback, cherry-pick, and fork. Install with `pip install amfs-branching`.

**Branches** — Create branches from an agent's main memory. Write to a branch without affecting main. Diff changes, then merge or discard.

**Access Control** — Grant read or read/write access to specific users, teams, or API keys per branch. Access is enforced at the HTTP layer — requests targeting a non-main branch are automatically checked against the branch's access grants.

**Pull Requests** — Create PRs for team review before merging branch changes into main. Includes review workflow with approve/reject/comment.

**Tags & Rollback** — Create named snapshots and roll back memory to any point in time or tagged state.

**Cherry-pick & Fork** — Selectively move entries between branches, or clone an entire agent's memory to a new agent.

**Sacred Timeline** — Interactive 3D visualization (React Three Fiber) showing how an agent's memory evolved over time, with branch forks, event nodes, and contextual detail panels.

```python
from amfs import AgentMemory
from amfs_branching.sdk import extend_with_branching

mem = AgentMemory(agent_id="deploy-agent")
extend_with_branching(mem)

mem.create_branch("experiment/retry-v3")
mem.switch_branch("experiment/retry-v3")
mem.write("checkout-service", "retry-pattern", {"max_retries": 5})

mem.grant_branch_access("experiment/retry-v3", "api_key", "amfs_sk_bob", "read")

diffs = mem.diff_branch("experiment/retry-v3")
result = mem.merge_branch("experiment/retry-v3")
```

### Multi-Tenant SaaS Foundation

The foundation for running AMFS as a hosted service. Every API request is authenticated, authorized, scoped, and audited.

**Account Isolation** — Hard isolation between tenants using Postgres Row-Level Security. Company A cannot see Company B's data, even if there's a bug in application code.

**RBAC** — Three roles with graduated permissions:

| Role | Can do |
|:-----|:-------|
| **Admin** | Full account management, user invites, key management, all memory ops |
| **Developer** | Create/revoke API keys, read/write memory, view audit logs |
| **User** | Read memory within scoped paths |

**Scoped API Keys** — Each agent/tool gets its own key with entity-path permissions:

```
amfs_sk_live_...  →  checkout-service/**  [READ_WRITE]
                     shared/patterns/*     [READ]
```

Agents can only access memory within their permitted scope — this is **permissioned inference** enforced at both the application and database level.

**HTTP API-Based Tenant Isolation** — All external agent access (MCP clients, SDKs, REST calls) is routed through the AMFS HTTP API using `AMFS_HTTP_URL` + `AMFS_API_KEY`. The API resolves the tenant, enforces entity-path scopes, sets the Postgres RLS context, and logs the operation — agents never touch the database directly. MCP clients connect transparently via the `HttpAdapter`, which converts memory operations into authenticated HTTP requests.

**Audit Logging** — Every state-changing operation is recorded in an append-only audit log with actor, action, resource, and IP address.

**Rate Limiting** — Per-key sliding-window rate limiting (RPM) with admin bypass. Response headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) let clients adapt.

**Usage Quotas** — Tiered limits on entries, API keys, users, and decision traces. Hard-capped at the database level, with external billing integration (Stripe, etc.) for metering. Three tiers: Starter, Team, Enterprise (unlimited).

### Immutable Decision Trace Store

The OSS `explain()` only works within the active session and captures enriched trace data (query events, error events, session timing, state diffs). Pro builds on this with persistent, cryptographically signed, immutable traces — the full causal chain is queryable forever.

**OSS trace enrichment** — every `commit_outcome` automatically captures:

- **Session timing** — `session_started_at`, `session_ended_at`, `session_duration_ms`
- **Query events** — every `search()` and `list()` call with parameters, result counts, and per-operation latency
- **Error events** — any errors during reads, writes, or tool calls
- **State diff** — entries created, updated, and confidence changes during the session
- **Full entry snapshots** — `value`, `memory_type`, and `written_by` captured at read time

**Pro immutable traces** add:

- **Cryptographic integrity** — HMAC-SHA256 content hashing with Merkle chain linking across sessions
- **LLM call spans** — model, provider, prompt/completion token counts, cost, latency, temperature, finish reason
- **Write events, tool calls, agent interactions** — full record of every action
- **Token and cost aggregates** — `total_llm_calls`, `total_tokens`, `total_cost_usd` per trace

```python
from amfs_traces import TraceRecorder, InMemoryTraceStore

recorder = TraceRecorder(memory, store, account_id=acct.id)

# Reads and external contexts are tracked automatically
recorder.memory.read("svc", "retry-pattern")
recorder.memory.record_context("pagerduty", "3 SEV-1", source="PagerDuty API")

# Record LLM calls for token/cost tracking (Pro: amfs_traces / amfs_record_llm_call;
# the OSS MCP server does not expose this tool)
recorder.record_llm_call(
    model="gpt-4o", provider="openai",
    prompt_tokens=1200, completion_tokens=450,
    latency_ms=2340, cost_usd=0.0285,
)

# Outcome commits automatically persist the trace
updated, trace = recorder.commit_outcome("DEP-500", OutcomeType.SUCCESS)

# Months later, explain still works
result = recorder.explain("DEP-500")

# Search across all decisions
traces = recorder.search_traces(entity_path="checkout-service", outcome_type="critical_failure")
```

### Cross-System Ingestion

Automatically ingest events from external systems into AMFS memory, turning your infrastructure events into queryable agent context.

**Webhook Ingester** — Generic endpoint for receiving JSON payloads from any system. Includes HMAC signature verification, payload size limits, idempotency deduplication, and a pluggable transform pipeline with glob-style pattern matching.

**Connector Framework** — Extensible `ConnectorABC` base class for building connectors to any external system. Each connector transforms raw events into AMFS `write()` or `record_context()` operations.

**Built-in connectors:**

| Connector | What it ingests |
|:----------|:---------------|
| **PagerDuty** | Incident webhooks (triggered/acknowledged/resolved), extracts severity, service, assignees |
| **Slack** | *(coming soon)* Channel messages, thread context, bot interactions |
| **GitHub** | *(coming soon)* PR events, deployment status, issue updates |
| **Jira** | *(coming soon)* Issue transitions, sprint events, release notes |

```python
from amfs_connectors import WebhookIngester, WebhookConfig
from amfs_connectors.providers.pagerduty import transform_pagerduty_incident

ingester = WebhookIngester(
    WebhookConfig(name="pd-webhook", entity_path="incidents", secret="whsec_..."),
    memory=mem,
)
ingester.register_transform("incident.*", transform_pagerduty_incident)

# In your FastAPI endpoint:
results = ingester.ingest(payload_bytes, headers, source="pagerduty", event_type="incident.triggered")
```

### Automated Pattern Detection

Continuously analyze your memory store to surface recurring patterns, anomalies, and risks — before they become incidents.

**Pattern Detector** — Scans memory entries and decision trace data for four pattern types:

| Pattern | What it finds | Severity |
|:--------|:-------------|:---------|
| **Recurring failures** | Entries that repeatedly appear in incident causal chains | Warning / Critical |
| **Hot entities** | Entities with disproportionate write/outcome activity vs. average | Info |
| **Stale clusters** | Groups of old entries with no outcome links that may need pruning | Warning |
| **Confidence drift** | Entries whose confidence diverges significantly from their entity average | Info |

**Alert Manager** — Configurable rules that fire when matching patterns are detected:

```python
from amfs_patterns import PatternDetector, AlertManager, AlertRule

detector = PatternDetector(incident_threshold=2, stale_days=30)
report = detector.analyze(entries, outcome_data=outcomes)

manager = AlertManager()
manager.add_rule(AlertRule(
    name="Critical recurring failures",
    pattern_type="recurring_failure",
    min_severity="critical",
    cooldown_minutes=60,
))
manager.on_alert(lambda eval: send_slack_notification(eval))

evaluations = manager.evaluate(report)
```

Features: severity filtering, entity-path scoping, cooldown-based suppression (prevent alert fatigue), and callback registration for routing to Slack, PagerDuty, email, or custom systems.

### Intelligence Layer

**Extraction** — Turns raw text (conversations, logs, documents) into structured memory operations using LLMs. The extractor classifies each piece of information as an **ADD**, **UPDATE**, **DELETE**, or **NOOP** operation.

**Auto Entity & Relationship Extraction** — Automatically extracts entities (services, people, tools, infrastructure) and relationships (depends_on, uses, manages, monitors) from decision traces and memory entries using LLMs. Extracted data is stored in dedicated tables with confidence scores, temporal validity, and trace provenance. Enable with `AMFS_AUTO_EXTRACT=true`.

**Memory Critic** — Automated quality analyzer that scans the memory store and detects five issue classes: Toxic (repeated negative correlations), Stale, Contradictory, Uncalibrated, and Orphaned entries.

**Memory Distiller** — Compacts large stores into smaller, higher-quality sets via pruning, consolidation, and bootstrap set generation for agent onboarding.

**Memory Safety Validator** — Pre-write guardrails: contradiction detection, temporal consistency, confidence thresholds, and causal chain integrity.

**Multi-Strategy Retrieval** — Combines semantic, BM25, temporal, and confidence signals via Reciprocal Rank Fusion (RRF).

**ML Layer** — Learned retrieval ranking (gradient-boosted model on outcome history), adaptive confidence calibration (learns optimal multipliers per entity), and training data export (SFT, DPO, reward model datasets from decision traces).

### Dashboard

A web dashboard (Next.js 15 + React 19) for exploring memory, visualizing decisions, and managing your AMFS deployment.

| Page | Description |
|:-----|:------------|
| **Overview** | Account-wide stats, recent activity, health indicators, live SSE status |
| **Entities** | Browse entities, view entries with confidence badges, version history, provenance details |
| **Traces** | Enriched trace cards with causal entries (full values), external contexts, query/error events, session timing, outcome badges, and search/filter |
| **Trace Detail** | Full trace view with interactive causal graph (D3), timeline, entry snapshots, and state diff |
| **Agents** | Per-agent overview with memory graph, activity, and **Git Repository** section (see below) |
| **Incidents** | Incident timeline with causal chain drill-down |
| **Patterns** | Detected pattern dashboard (recurring failures, hot entities, stale clusters, confidence drift) with severity indicators and resolution tracking |
| **Teams** | Team CRUD with member management (admin, developer, viewer roles) |
| **Snapshots** | Memory snapshot capture, comparison, and JSON export |
| **API Keys** | Key management console with scopes, rate limits, expiry, and usage |
| **Audit Log** | Searchable, filterable log of all state-changing operations |
| **Usage & Quotas** | Quota progress bars, request metrics, top agents/entities breakdown |
| **Pro Tools** | Retrieval playground, critic panel, distiller view, calibration dashboard, training data export |

**Agent Git Repository** — Each agent's detail page includes a "Git Repository" section with:

| Tab | Description |
|:----|:------------|
| **Sacred Timeline** | Interactive 3D visualization of the agent's memory history (React Three Fiber). Shows event nodes, branch forks, date markers, and contextual detail panels when clicking events. |
| **Event Log** | Chronological 2D list of all events (writes, outcomes, reads, briefs, webhooks) with icons, details, and date grouping. |
| **Branches** | Branch management — create, close, merge, view diffs, and manage access grants per branch. |
| **Pull Requests** | PR workflow — create, review, merge, and close pull requests for branch changes. |
| **Tags & Rollback** | Create named snapshots, roll back to tags or specific timestamps. |

### OpenTelemetry Export

Export AMFS decision traces as OpenTelemetry spans for integration with existing observability stacks (Jaeger, Grafana Tempo, Datadog, Honeycomb). Traces are mapped to hierarchical spans following GenAI semantic conventions — each LLM call becomes a `gen_ai` span, and memory operations become `amfs` spans.

```python
from amfs_otel import TraceExporter, configure_otel

configure_otel(endpoint="http://localhost:4317", service_name="my-agent")
exporter = TraceExporter()
exporter.export_trace(trace)  # sends spans to your OTel collector
```

### Pro MCP Server

On **Sense Lab**, Cursor uses **`uvx amfs-mcp-server`** with **`AMFS_HTTP_URL`** (dashboard **Server URL**, e.g. `https://amfs-login.sense-lab.ai`) and **`AMFS_API_KEY`**—same JSON as the **Agents** MCP Connection card. Install the **[Cursor plugin](https://github.com/raia-live/cursor-plugin)** or copy that snippet; see [MCP setup — AMFS Pro and Cursor](https://raia-live.github.io/amfs/guides/mcp/#amfs-pro-saas-and-cursor).

Hosted AMFS also serves **Streamable HTTP** directly at **`https://mcp.sense-lab.ai/mcp`** for clients that prefer a remote URL to a local process, authenticated with the same API key as a bearer token. It exposes a focused set of memory tools rather than the full stdio surface. Running your own MCP HTTP listener remains a separate, self-hosted deployment mode.

Extends the OSS MCP server with additional tools:

| Tool | Description |
|:-----|:------------|
| `amfs_critique` | Run the Memory Critic and get a quality report |
| `amfs_distill` | Trigger distillation (prune, consolidate, or generate bootstrap set) |
| `amfs_validate` | Validate a candidate memory before writing |
| `amfs_retrain` | Train the learned ranking model from outcome data |
| `amfs_calibrate` | Learn optimal confidence multipliers from outcome history |
| `amfs_export_training_data` | Export decision traces as SFT/DPO/reward model datasets |
| `amfs_record_llm_call` | Record an LLM call with model, tokens, cost, and latency. **Pro only** — not registered by the OSS MCP server |
| `amfs_graph_path` | Find shortest trust-weighted path between two entities in the knowledge graph |
| `amfs_graph_query` | Flexible graph edge search by relation, entity type, or confidence range |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent / IDE                          │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌─────────────────┐          ┌───────────────────────────┐
   │  MCP Server OSS │          │  MCP Server Pro            │
   │  (12 tools)     │          │  (12 + 8 tools)            │
   └────────┬────────┘          │                           │
            │                   │  Branching Layer:         │
            │                   │  Branches · Merge · PRs   │
            │                   │  Access Control · Tags    │
            │                   │  Rollback · Fork          │
            │                   │                           │
            │                   │  Multi-Tenant Layer:      │
            │                   │  Auth · RBAC · RLS        │
            │                   │  Scopes · Audit · Quotas  │
            │                   │  Rate Limiting             │
            │                   │                           │
            │                   │  Decision Traces:         │
            │                   │  Recorder · Store · Search │
            │                   │                           │
            │                   │  Ingestion Layer:         │
            │                   │  Webhooks · Connectors    │
            │                   │  PagerDuty · Slack · etc. │
            │                   │                           │
            │                   │  Pattern Detection:       │
            │                   │  Detector · AlertManager  │
            │                   │                           │
            │                   │  Intelligence Layer:       │
            │                   │  Critic · Distiller        │
            │                   │  Safety · Retrieval        │
            │                   │                           │
            │                   │  ML Layer:                │
            │                   │  Ranking · Calibration    │
            │                   │  Training Data Export     │
            │                   └─────────────┬─────────────┘
            │                                 │
            └──────────┬──────────────────────┘
                       ▼
            ┌─────────────────┐
            │   AgentMemory   │  ← Python / TypeScript SDK
            │   (CoW Engine)  │
            │   + Timeline    │  ← Git-like event log
            └────────┬────────┘
                     │
           ┌─────────┼─────────┐
           ▼         ▼         ▼
      Filesystem  Postgres    S3
       Adapter    Adapter   Adapter
```

The Pro API layer wraps `AgentMemory` and `CoWEngine` with authentication, tenant isolation, scope enforcement, rate limiting, and audit logging — all backed by Postgres RLS for defense-in-depth.

---

## When to Use Which

| Scenario | Recommendation |
|:---------|:---------------|
| Single developer, local memory | OSS |
| Small team sharing via Postgres | OSS |
| CI/CD outcome tracking | OSS |
| Remote HTTP API access | OSS |
| S3-based cloud storage | OSS |
| Git-like timeline (event logging on main branch) | OSS |
| Branch-aware read/write (main branch only) | OSS |
| Create branches and share memory selectively | **Pro** (branching) |
| Branch access control per user/team/API key | **Pro** (branching) |
| Merge branches, resolve conflicts | **Pro** (branching) |
| Pull request workflow for memory changes | **Pro** (branching) |
| Rollback memory to a point in time or tag | **Pro** (branching) |
| Fork agent memory to new agents | **Pro** (branching) |
| Sacred Timeline 3D visualization | **Pro** (dashboard) |
| Multi-tenant SaaS with account isolation | **Pro** |
| Scoped API keys per agent/tool | **Pro** |
| Compliance audit logging | **Pro** |
| Enriched decision traces with error/query events and session timing | OSS |
| Per-agent memory graph and activity timeline | OSS |
| Immutable, cryptographically signed traces | **Pro** |
| LLM call span tracking with token/cost analytics | **Pro** |
| OpenTelemetry export for existing observability stacks | **Pro** |
| Auto entity/relationship extraction from traces | **Pro** |
| Precedent search ("how did we handle similar situations?") | **Pro** |
| Auto-ingest PagerDuty/Slack/GitHub events | **Pro** (connectors) |
| Detect recurring failure patterns automatically | **Pro** (pattern detection) |
| Alert on stale memory or confidence drift | **Pro** (alert manager) |
| Need memory quality auditing at scale | **Pro** |
| Want LLM-driven extraction from conversations/logs | **Pro** |
| Onboarding new agents with curated knowledge | **Pro** (bootstrap sets) |
| Compliance or safety requirements for memory writes | **Pro** (safety validator) |
| Advanced retrieval across large stores | **Pro** (multi-strategy + learned ranking) |
| Retrieval that improves as outcomes accumulate | **Pro** (ML layer) |
| Need optimized confidence multipliers per entity | **Pro** (adaptive calibration) |
| Want to fine-tune agents on your decision history | **Pro** (training data export) |
| Team dashboard for non-technical users | **Pro** (dashboard) |

---

## Getting Started

### OSS

```bash
pip install amfs
```

See the [Quick Start guide](/amfs/getting-started/quickstart/) to begin.

### Pro

Contact us at [raia.live](https://raia.live) for Pro access and setup instructions.
