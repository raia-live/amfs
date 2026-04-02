---
title: OSS vs Pro
layout: default
nav_order: 4
description: "What's in AMFS open-source vs. AMFS Pro — and when you need which."
permalink: /editions/
---

# OSS vs Pro
{: .no_toc }

AMFS is split into two layers: a fully open-source core and a proprietary Pro layer. The OSS layer gives you everything you need to build production-ready agent memory. Pro adds multi-tenant SaaS infrastructure, persistent decision traces, cross-system ingestion, automated pattern detection, LLM-powered intelligence, and an enterprise dashboard.
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
│  ┌─ Multi-Tenant SaaS ─────────────────────────────────┐ │
│  │  Accounts · RBAC · Scoped API Keys · Audit · Quotas │ │
│  │  Row-Level Security · OAuth/OIDC · Rate Limiting     │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Persistent Decision Traces ─────────────────────────┐ │
│  │  Durable Causal Chains · Historical explain()        │ │
│  │  Precedent Search · Cross-Session Trace Queries      │ │
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
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ Dashboard ──────────────────────────────────────────┐ │
│  │  Memory Explorer · Context Graph · Trace Explorer     │ │
│  │  API Key Console · Audit Viewer · Usage Analytics    │ │
│  │  Pattern Alerts · Team Management                     │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                    AMFS OSS (Apache 2.0)                  │
│                                                          │
│  CoW Engine · Memory Types · Provenance Tiers             │
│  Temporal Queries · Session-Level Causal Explainability   │
│  Adapters (Filesystem, Postgres, S3) · HTTP/REST API      │
│  MCP Server · Python SDK · TypeScript SDK · CLI           │
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
| **Adapters & Infrastructure** | | |
| Filesystem adapter | Yes | Yes |
| Postgres adapter (triggers, LISTEN/NOTIFY) | Yes | Yes |
| S3 adapter | Yes | Yes |
| HTTP/REST API server | Yes | Yes |
| Docker + Docker Compose + Helm charts | Yes | Yes |
| MCP server (9 tools) | Yes | Yes |
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
| **Persistent Decision Traces (Pro)** | | |
| Durable causal chains across sessions | — | Yes |
| Historical `explain(outcome_ref)` | — | Yes |
| `search_traces` / precedent search API | — | Yes |
| Cross-agent, cross-session trace queries | — | Yes |
| **Cross-System Ingestion (Pro)** | | |
| Generic webhook endpoint with HMAC verification | — | Yes |
| Payload deduplication (idempotency) | — | Yes |
| Pluggable transform pipeline with pattern matching | — | Yes |
| PagerDuty incident connector | — | Yes |
| Extensible connector framework (`ConnectorABC`) | — | Yes |
| **Automated Pattern Detection (Pro)** | | |
| Recurring failure detection across causal chains | — | Yes |
| Hot entity detection (disproportionate activity) | — | Yes |
| Stale cluster detection (unvalidated entries) | — | Yes |
| Confidence drift detection (outlier entries) | — | Yes |
| Configurable alert rules with cooldown suppression | — | Yes |
| Alert callbacks (Slack, PagerDuty, email routing) | — | Yes |
| **Intelligence Layer (Pro)** | | |
| LLM-driven memory extraction | — | Yes |
| Automated memory critic | — | Yes |
| Memory distillation & bootstrap sets | — | Yes |
| Memory safety validation | — | Yes |
| Multi-strategy retrieval (semantic + BM25 + temporal) | — | Yes |
| Learned retrieval ranking from outcome data | — | Yes |
| Adaptive confidence calibration | — | Yes |
| Training data export (SFT, DPO, reward model) | — | Yes |
| **Dashboard (Pro)** | | |
| Memory explorer with context graph visualization | — | Yes |
| Decision trace explorer (expandable causal chains) | — | Yes |
| Pattern alert monitoring | — | Yes |
| Team & user management | — | Yes |
| API key management console | — | Yes |
| Audit log viewer | — | Yes |
| Usage analytics & quota monitoring | — | Yes |
| Extended MCP server (Pro tools) | — | Yes |

---

## OSS Layer — What's Included

The open-source layer ([github.com/raia-live/amfs](https://github.com/raia-live/amfs)) provides the full memory primitive: read, write, version, search, and learn from outcomes.

### Packages

| Package | Description |
|:--------|:------------|
| `amfs-core` | CoW engine, models (`MemoryEntry`, `MemoryType`, `ProvenanceTier`), read tracking, causal tagging, default embedder |
| `amfs` (SDK) | `AgentMemory` class — `read`, `write`, `list`, `search`, `history`, `explain`, `commit_outcome`, `record_context` |
| `amfs-adapter-filesystem` | JSON-file-based adapter for local development |
| `amfs-adapter-postgres` | PostgreSQL adapter with PL/pgSQL triggers for outcome propagation and `LISTEN/NOTIFY` for watch |
| `amfs-adapter-s3` | Amazon S3 / S3-compatible adapter for cloud-native storage |
| `amfs-http-server` | REST API server (FastAPI/Uvicorn) for remote access |
| `amfs-mcp-server` | MCP server exposing 9 tools: `amfs_read`, `amfs_write`, `amfs_search`, `amfs_list`, `amfs_stats`, `amfs_commit_outcome`, `amfs_record_context`, `amfs_history`, `amfs_explain` |
| `amfs-cli` | Terminal tools for inspecting, diffing, snapshotting, and restoring memory |
| `@amfs/sdk` | TypeScript SDK (full parity with Python: ReadTracker, search, stats, history, explain, recordContext) |

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

Agents can only access memory within their permitted scope — this is **permissioned inference** enforced at the database level.

**Audit Logging** — Every state-changing operation is recorded in an append-only audit log with actor, action, resource, and IP address.

**Rate Limiting** — Per-key sliding-window rate limiting (RPM) with admin bypass. Response headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) let clients adapt.

**Usage Quotas** — Tiered limits on entries, API keys, users, and decision traces. Hard-capped at the database level, with external billing integration (Stripe, etc.) for metering. Three tiers: Starter, Team, Enterprise (unlimited).

### Persistent Decision Traces

The OSS `explain()` only works within the active session. Pro persists the full causal chain — which AMFS entries were read, which external tools were consulted, what decision was made, and what outcome occurred — so it's queryable forever.

```python
from amfs_traces import TraceRecorder, InMemoryTraceStore

recorder = TraceRecorder(memory, store, account_id=acct.id)

# Reads and external contexts are tracked automatically
recorder.memory.read("svc", "retry-pattern")
recorder.memory.record_context("pagerduty", "3 SEV-1", source="PagerDuty API")

# Outcome commits automatically persist the trace
updated, trace = recorder.commit_outcome("DEP-500", OutcomeType.CLEAN_DEPLOY)

# Months later, explain still works
result = recorder.explain("DEP-500")

# Search across all decisions
traces = recorder.search_traces(entity_path="checkout-service", outcome_type="p1_incident")
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

**Memory Critic** — Automated quality analyzer that scans the memory store and detects five issue classes: Toxic (repeated negative correlations), Stale, Contradictory, Uncalibrated, and Orphaned entries.

**Memory Distiller** — Compacts large stores into smaller, higher-quality sets via pruning, consolidation, and bootstrap set generation for agent onboarding.

**Memory Safety Validator** — Pre-write guardrails: contradiction detection, temporal consistency, confidence thresholds, and causal chain integrity.

**Multi-Strategy Retrieval** — Combines semantic, BM25, temporal, and confidence signals via Reciprocal Rank Fusion (RRF).

**ML Layer** — Learned retrieval ranking (gradient-boosted model on outcome history), adaptive confidence calibration (learns optimal multipliers per entity), and training data export (SFT, DPO, reward model datasets from decision traces).

### Dashboard

A web dashboard (Next.js 15 + React 19) for exploring memory, visualizing decisions, and managing your AMFS deployment.

| Page | Description |
|:-----|:------------|
| **Overview** | Account-wide stats, recent activity, health indicators |
| **Memory Explorer** | Browse entities, view entries with confidence badges, version history |
| **Context Graph** | Interactive D3 force-directed graph of entity relationships and causal chains |
| **Decision Traces** | Expandable trace cards with causal entries, external contexts, outcome badges, and search |
| **Incidents** | Incident timeline with causal chain drill-down |
| **Patterns** | Detected pattern dashboard with severity indicators |
| **Teams** | User and role management |
| **API Keys** | Key management console with scopes, rate limits, expiry, and usage |
| **Audit Log** | Searchable, filterable log of all state-changing operations |
| **Usage & Quotas** | Quota progress bars, request metrics, top agents/entities breakdown |
| **Pro Tools** | Retrieval playground, critic panel, distiller view, calibration dashboard, training data export |

### Pro MCP Server

Extends the OSS MCP server with 7 additional tools:

| Tool | Description |
|:-----|:------------|
| `amfs_critique` | Run the Memory Critic and get a quality report |
| `amfs_distill` | Trigger distillation (prune, consolidate, or generate bootstrap set) |
| `amfs_validate` | Validate a candidate memory before writing |
| `amfs_retrieve` | Multi-strategy retrieval with RRF-merged results (+ learned ranking) |
| `amfs_retrain` | Train the learned ranking model from outcome data |
| `amfs_calibrate` | Learn optimal confidence multipliers from outcome history |
| `amfs_export_training_data` | Export decision traces as SFT/DPO/reward model datasets |

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
   │  (9 tools)      │          │  (9 + 7 tools)             │
   └────────┬────────┘          │                           │
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
| Multi-tenant SaaS with account isolation | **Pro** |
| Scoped API keys per agent/tool | **Pro** |
| Compliance audit logging | **Pro** |
| Persistent decision traces that survive sessions | **Pro** |
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
