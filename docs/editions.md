---
title: OSS vs Pro
layout: default
nav_order: 4
description: "What's in AMFS open-source vs. AMFS Pro — and when you need which."
permalink: /editions/
---

# OSS vs Pro
{: .no_toc }

AMFS is split into two layers: a fully open-source core and a proprietary intelligence layer. The OSS layer gives you everything you need to build production-ready agent memory. The Pro layer adds LLM-powered automation on top.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## At a Glance

```
┌──────────────────────────────────────────────────────┐
│                   AMFS Pro (Proprietary)              │
│                                                      │
│  Extraction · Critic · Distiller · Safety · Retrieval │
│  Learned Ranking · Confidence Calibration · ML Export │
│                                                      │
├──────────────────────────────────────────────────────┤
│                   AMFS OSS (Apache 2.0)               │
│                                                      │
│  CoW Engine · Memory Types · Provenance Tiers         │
│  Temporal Queries · Causal Explainability             │
│  Adapters (Filesystem, Postgres) · MCP Server         │
│  Python SDK · TypeScript SDK · CLI                    │
└──────────────────────────────────────────────────────┘
```

---

## Feature Comparison

| Capability | OSS | Pro |
|:-----------|:---:|:---:|
| Copy-on-Write versioning | Yes | Yes |
| Confidence scoring with outcome back-propagation | Yes | Yes |
| Memory types (fact, belief, experience) | Yes | Yes |
| Type-specific confidence decay | Yes | Yes |
| Provenance tiers (production-validated → manual) | Yes | Yes |
| Temporal queries (`history`) | Yes | Yes |
| Causal explainability (`explain`) | Yes | Yes |
| Filesystem adapter | Yes | Yes |
| Postgres adapter (triggers, LISTEN/NOTIFY) | Yes | Yes |
| MCP server (9 tools) | Yes | Yes |
| Python SDK | Yes | Yes |
| TypeScript SDK | Yes | Yes |
| CLI tools | Yes | Yes |
| Framework integrations (CrewAI, LangGraph, etc.) | Yes | Yes |
| Bundled lightweight embedder | Yes | Yes |
| LLM-driven memory extraction | — | Yes |
| Automated memory critic | — | Yes |
| Memory distillation & bootstrap sets | — | Yes |
| Memory safety validation | — | Yes |
| Multi-strategy retrieval (semantic + BM25 + temporal) | — | Yes |
| Learned retrieval ranking from outcome data | — | Yes |
| Adaptive confidence calibration | — | Yes |
| Training data export (SFT, DPO, reward model) | — | Yes |
| Extended MCP server (Pro tools) | — | Yes |

---

## OSS Layer — What's Included

The open-source layer ([github.com/raia-live/amfs](https://github.com/raia-live/amfs)) provides the full memory primitive: read, write, version, search, and learn from outcomes.

### Packages

| Package | Description |
|:--------|:------------|
| `amfs-core` | CoW engine, models (`MemoryEntry`, `MemoryType`, `ProvenanceTier`), read tracking, causal tagging, default embedder |
| `amfs` (SDK) | `AgentMemory` class — `read`, `write`, `list`, `search`, `history`, `explain`, `commit_outcome` |
| `amfs-adapter-filesystem` | JSON-file-based adapter for local development |
| `amfs-adapter-postgres` | PostgreSQL adapter with PL/pgSQL triggers for outcome propagation and `LISTEN/NOTIFY` for watch |
| `amfs-mcp-server` | MCP server exposing 9 tools: `amfs_read`, `amfs_write`, `amfs_search`, `amfs_list`, `amfs_stats`, `amfs_commit_outcome`, `amfs_record_context`, `amfs_history`, `amfs_explain` |
| `amfs-cli` | Terminal tools for inspecting, diffing, snapshotting, and restoring memory |
| `@amfs/sdk` | TypeScript SDK |

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

The Pro layer builds an intelligence layer on top of the OSS primitives. It adds LLM-powered automation for extraction, quality control, compaction, safety, and retrieval.

### Extraction

Turns raw text (conversations, logs, documents) into structured memory operations using LLMs. Instead of blind writes, the extractor classifies each piece of information as an **ADD**, **UPDATE**, **DELETE**, or **NOOP** operation.

```
Raw input → LLM Extractor → [ADD "svc/pattern-retry", UPDATE "svc/config", NOOP, ...]
```

Supports multiple LLM backends (OpenAI, Anthropic) with a common `ExtractorABC` interface.

### Memory Critic

An automated quality analyzer that scans the memory store and detects problematic entries. Runs offline or on a schedule to keep the store healthy.

Detects five issue classes:

| Issue | Description |
|:------|:------------|
| **Toxic** | Entries with repeated negative outcome correlations |
| **Stale** | Entries that haven't been read or validated in a long time |
| **Contradictory** | Entries that conflict with other entries in the same entity |
| **Uncalibrated** | Entries whose confidence doesn't match their outcome history |
| **Orphaned** | Entries with no reads, no outcomes, and no cross-references |

Produces a `CriticReport` with prioritized recommendations.

### Memory Distiller

Compacts large memory stores into smaller, higher-quality sets. Three operations:

- **Pruning** — Removes low-value entries (orphaned, expired, low-confidence)
- **Consolidation** — Merges related entries into unified summaries
- **Bootstrap sets** — Generates compact `DistilledSet` packages that can onboard new agents quickly (teacher-to-student knowledge transfer)

### Memory Safety Validator

Pre-write guardrails that validate candidate memories before they enter the store:

- **Contradiction detection** — Flags entries that conflict with existing knowledge
- **Temporal consistency** — Ensures timestamps and version sequences are coherent
- **Confidence thresholds** — Rejects entries with implausible confidence values
- **Causal chain integrity** — Verifies that referenced causal keys exist

### Multi-Strategy Retrieval

Advanced search that goes beyond single-embedding lookup by combining multiple retrieval strategies:

| Strategy | Signal |
|:---------|:-------|
| Semantic | Vector similarity via embeddings |
| BM25 | Keyword relevance scoring |
| Temporal | Recency weighting |
| Confidence | Trust-score ranking |

Results are merged using **Reciprocal Rank Fusion (RRF)** to produce a single ranked list.

### ML Layer

The ML layer learns from your outcome data to make AMFS smarter over time. Three capabilities:

**Learned Retrieval Ranking** — Trains a gradient-boosted model on your outcome history to predict which memories are most useful. Entries read before clean deploys are positive signals; entries read before incidents are negative. Once trained, the model automatically enhances `amfs_retrieve` results. Falls back to heuristic ranking when insufficient data (< 20 outcome-linked entries).

Features extracted from each `MemoryEntry`:

| Feature | Signal |
|:--------|:-------|
| Confidence | Current trust score |
| Outcome count | How many outcomes validated this entry |
| Version | How many times it's been updated |
| Age | Time since creation (days and log-hours) |
| Memory type | Fact, belief, or experience (one-hot) |
| Provenance tier | Production-validated through manual (one-hot) |
| Pattern refs | Whether cross-references exist and how many |

**Adaptive Confidence Calibration** — Learns optimal outcome multipliers from historical data instead of the fixed defaults (P1 = 1.15, clean_deploy = 0.97). Analyzes how entries linked to each outcome type perform in subsequent outcomes, then computes calibrated multipliers per entity. Also estimates optimal decay half-life from the age distribution of actively-used entries.

**Training Data Export** — Exports your decision traces as fine-tuning datasets. AMFS doesn't train your LLMs — it generates the structured training data from its outcome-linked causal chains:

| Format | Description |
|:-------|:------------|
| **SFT** | Successful decision traces as (context, decision) examples |
| **DPO** | Paired (chosen, rejected) from positive vs negative outcomes |
| **Reward Model** | Entries labeled by outcome quality score |

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

The Pro layer wraps the OSS layer — it never replaces it. All Pro features read from and write to the same memory store using the same adapters and SDK.

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
            │   AgentMemory   │  ← Python SDK
            │   (CoW Engine)  │
            └────────┬────────┘
                     │
           ┌─────────┼─────────┐
           ▼         ▼         ▼
      Filesystem  Postgres    Custom
       Adapter    Adapter    Adapter
```

The Pro MCP server imports and re-exports all 9 OSS tools, then adds the 7 Pro tools on top. You only run one server — either OSS or Pro.

---

## When to Use Which

| Scenario | Recommendation |
|:---------|:---------------|
| Single developer, local memory | OSS |
| Small team sharing via Postgres | OSS |
| CI/CD outcome tracking | OSS |
| Need memory quality auditing at scale | Pro |
| Want LLM-driven extraction from conversations/logs | Pro |
| Onboarding new agents with curated knowledge | Pro (bootstrap sets) |
| Compliance or safety requirements for memory writes | Pro (safety validator) |
| Advanced retrieval across large stores | Pro (multi-strategy + learned ranking) |
| Want retrieval that improves as outcomes accumulate | Pro (ML layer) |
| Need optimized confidence multipliers per entity | Pro (adaptive calibration) |
| Want to fine-tune agents on your decision history | Pro (training data export) |

---

## Getting Started

### OSS

```bash
pip install amfs
```

See the [Quick Start guide](/amfs/getting-started/quickstart/) to begin.

### Pro

Contact us at [raia.live](https://raia.live) for Pro access and setup instructions.
