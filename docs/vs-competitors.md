---
title: AMFS vs Competitors
layout: default
nav_order: 6
description: "How AMFS compares to Mem0, Zep, Hindsight, Letta/MemGPT, Cognee, and other AI memory systems."
permalink: /vs-competitors/
---

# AMFS vs Competitors
{: .no_toc }

Every other memory system is a smarter store. AMFS is the only one that treats agent memory like code — with version control, branching, pull requests, and rollback.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Feature Matrix

| Feature | AMFS | Mem0 | Zep / Graphiti | Hindsight | Letta | Cognee | LangMem |
|:--------|:----:|:----:|:--------------:|:---------:|:-----:|:------:|:-------:|
| **Git-like collaboration** | | | | | | | |
| Branching (isolated experiments) | Pro | No | No | No | No | No | No |
| Pull requests (review before merge) | Pro | No | No | No | No | No | No |
| Diff, merge, rollback | Pro | No | No | No | No | No | No |
| Tags / named snapshots | Pro | No | No | No | No | No | No |
| Branch-level access control | Pro | No | No | No | No | No | No |
| Fork (clone brain to new agent) | Pro | No | No | No | No | No | No |
| Git-like timeline (event log) | Yes | No | No | No | No | No | No |
| **Memory fundamentals** | | | | | | | |
| Persistent memory | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Versioning (full history) | CoW | No | Temporal | No | No | No | No |
| Provenance (who wrote, when) | Yes | No | Partial | No | No | No | No |
| Confidence scoring | Yes | No | No | Opinion only | No | No | No |
| Outcome back-propagation | Yes | No | No | No | No | No | No |
| Memory types (fact/belief/experience) | Yes | No | No | 4 networks | 3 stores | No | 3 types |
| Causal explainability | Yes | No | No | No | No | No | No |
| Knowledge graph | Auto | Optional | Yes | Manual | No | Yes | No |
| Semantic search | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Multi-agent native | Yes | Partial | No | No | No | No | No |
| Conflict detection | Yes | No | No | No | No | No | No |
| Tiered memory (hot/warm/archive) | Yes | No | No | No | 3-tier | No | No |
| Frequency-modulated decay | Yes | No | No | No | No | No | No |
| Progressive retrieval (depth) | Yes | No | No | No | No | No | No |
| Importance scoring (multi-dim) | Pro | No | No | No | No | No | No |
| Cortex drift gate | Yes | No | No | No | No | No | No |
| MCP server | Yes | Yes | No | Yes | No | No | No |
| Self-hosted / OSS | Apache 2.0 | OSS+Cloud | OSS+Cloud | OSS | OSS | OSS+Cloud | OSS |
| Import from another memory system | Mem0, Zep | No | No | No | No | No | No |
| Multi-tenant with RLS | Pro | Cloud | Cloud | No | No | Cloud | Cloud |
| Enterprise dashboard | Pro | Cloud | Cloud | No | No | Cloud | Cloud |
| Learned ranking from outcomes | Pro | No | No | No | No | No | No |

---

## AMFS vs Mem0

[Mem0](https://mem0.ai) is the most widely adopted memory library (41K+ stars). It extracts facts from conversations and stores them with ADD/UPDATE/DELETE/NOOP operations against a vector store.

**Where Mem0 excels:** Simple API, automatic fact extraction from chat, optional graph memory, wide framework integrations (CrewAI, LangGraph, Flowise).

**Where AMFS differs:**
- **Git model for collaboration** -- Branching, PRs, diff, merge, rollback, access control. Mem0 has no collaboration model — it's a single-writer store.
- **Outcome feedback loop** -- AMFS confidence evolves from production events. Mem0 stores facts without trust signals.
- **CoW versioning** -- Every write is immutable and replayable. Mem0 overwrites.
- **Multi-agent provenance** -- AMFS tracks authorship, detects conflicts, and auto-links causality. Mem0 is single-user oriented.
- **Four-signal decay** -- Time + type + outcomes + access frequency. Mem0 has no decay model.
- **Tiered memory** -- Hot/Warm/Archive with progressive retrieval. Mem0 searches everything.

**Already on Mem0:** Pro imports a project directly — memories, entities and optionally each memory's version history. See [Migrate from Mem0 or Zep](/amfs/guides/migrate/).

---

## AMFS vs Zep / Graphiti

[Zep](https://getzep.com) builds temporal knowledge graphs via [Graphiti](https://github.com/getzep/graphiti). Facts carry time ranges; queries can ask "what was true at time T?"

**Where Zep excels:** Temporal knowledge graphs with time-bounded edges, entity resolution across conversations, strong on multi-hop temporal queries.

**Where AMFS differs:**
- **Git model for collaboration** -- Branching, PRs, diff, merge, rollback. Zep has no collaboration or version control model.
- **Outcome back-propagation** -- AMFS learns from production events. Zep tracks temporal validity but doesn't learn from what happens after retrieval.
- **CoW vs temporal graph** -- Different models. AMFS versions individual entries; Graphiti maintains a graph with time-bounded edges. AMFS is simpler; Graphiti captures richer temporal relationships.
- **Multi-agent** -- AMFS has conflict detection and per-agent provenance. Zep targets single-assistant use.
- **Operational context** -- AMFS ingests infrastructure events via webhooks. Zep only processes conversations.

---

## AMFS vs Hindsight

[Hindsight](https://github.com/hindsight-ai/hindsight) maintains four separate memory networks (World, Experience, Opinion, Entity/Observation). It reports 91.4% on LongMemEval, the strongest published accuracy in the space.

**Where Hindsight excels:** Benchmark accuracy, clean separation of evidence vs inference (Opinion Network has confidence scores), multi-session temporal reasoning (21% -> 80% on LongMemEval multi-session questions).

**Where AMFS differs:**
- **Production feedback loop** -- Hindsight's Opinion Network has confidence scores, but they don't evolve from real-world outcomes. AMFS's confidence changes when deploys succeed or incidents occur.
- **Versioning** -- Hindsight overwrites network state. AMFS preserves full CoW history.
- **Tiered memory** -- AMFS's Hot/Warm/Archive with priority scoring is data-driven, vs Hindsight's fixed 4-network separation. AMFS's tiers rebalance automatically based on access patterns and importance.
- **Importance scoring** -- Pro evaluates entries across behavioral alignment, reasoning utility, and contextual persistence -- three LLM-scored dimensions that feed into tier assignment.
- **Knowledge graph** -- AMFS auto-materializes a graph from normal operations. Hindsight's networks are structurally defined.
- **Enterprise features** -- Multi-tenant isolation, RBAC, scoped API keys, audit logging, webhooks, dashboard. Hindsight is a research tool without enterprise infrastructure.

---

## AMFS vs Letta / MemGPT

[Letta](https://letta.com) (formerly MemGPT) treats the LLM as an OS managing its own memory: main context (RAM), recall store (recent history), and archival store (long-term).

**Where Letta excels:** Transparent memory management (the LLM decides what to page in/out), inspectable memory blocks, elegant OS metaphor.

**Where AMFS differs:**
- **Data-driven tiering** -- AMFS's Hot/Warm/Archive tiers are assigned by priority scoring (confidence, recency, recall frequency, importance), not by LLM paging decisions. This avoids the latency and cost of LLM-managed memory.
- **Outcome feedback** -- AMFS confidence evolves from production events. Letta doesn't connect memory to outcomes.
- **CoW versioning** -- Full history. Letta's archival store doesn't version.
- **Multi-agent native** -- AMFS supports per-agent provenance, conflict detection, and cross-agent knowledge transfer. Letta is single-agent.

---

## AMFS vs Cognee

[Cognee](https://cognee.ai) builds knowledge graphs from documents using LLM-powered extraction. Backed by OpenAI and FAIR founders.

**Where Cognee excels:** Document-to-graph construction, multi-hop reasoning (HotpotQA), ontology-based validation.

**Where AMFS differs:**
- **Agent-oriented vs document-oriented** -- AMFS is for agents that read, write, and act. Cognee processes documents into queryable graphs.
- **Outcome feedback** -- AMFS connects knowledge to production reality. Cognee's graph doesn't learn from post-retrieval events.
- **Versioning and provenance** -- AMFS preserves full history. Cognee updates its graph in place.

---

## AMFS vs LangMem

[LangMem](https://langchain-ai.github.io/long-term-memory/) is LangChain's long-term memory library for the LangGraph ecosystem.

**Where LangMem excels:** Native LangGraph integration, managed service via LangSmith, namespace scoping.

**Where AMFS differs:**
- **Framework-agnostic** -- AMFS works with CrewAI, LangGraph, AutoGen, or standalone. LangMem is tied to LangChain.
- **Outcome feedback** -- AMFS's core differentiator. No LangMem equivalent.
- **MCP-native** -- Built-in MCP server for IDE integration (Cursor, Claude Code).
- **Self-hosted** -- Pluggable backends (filesystem, Postgres, S3). LangMem is primarily managed.

---

## AMFS vs Memvid

[Memvid](https://github.com/memvid/memvid) packages memory into a single `.mv2` file -- data, embeddings, index, metadata. No database, no server.

**Where Memvid excels:** Zero infrastructure, 0.025ms P50 retrieval, portable single-file memory, ideal for offline/edge agents.

**Where AMFS differs:**
- **Different category** -- Memvid is a read-mostly, append-only search tool. AMFS is a multi-agent memory platform with versioning, outcomes, and enterprise features.
- **Multi-agent** -- Memvid has no concurrent writes, no conflict detection, no provenance.
- **Tiered memory** -- AMFS's Hot/Warm/Archive hierarchy doesn't exist in Memvid's flat file.
- **Production feedback** -- Memvid stores and retrieves. AMFS learns from outcomes.

Memvid is the right choice for single-user, offline, or edge scenarios where infrastructure is a non-starter. AMFS is for production multi-agent systems that need versioning, feedback, and collaboration.

---

## Coming from Mem0 or Zep

A comparison is only worth reading if you can act on it, so Pro imports directly from both. Paste an API key, look at the counts and what they'd cost against your plan, and start it — or run five users first and see what lands. Nothing is removed from the system you're leaving, and everything an import wrote can be removed again in one action if you decide against it.

What neither source has is a confidence score, so AMFS derives one: what a source holds verbatim comes in at 1.0, what it currently believes at 0.8, its own model's summaries at 0.6, and facts it has stopped believing at 0.3. Original timestamps are preserved. From there the [outcome loop](/amfs/concepts/confidence/) takes over and those starting numbers stop being guesses.

[Migrate from Mem0 or Zep](/amfs/guides/migrate/){: .btn .btn-outline }

---

## What Makes AMFS Unique

1. **GitHub for agent memory** -- No other system treats agent knowledge like code. AMFS gives every agent a brain (repo), with branching, pull requests, diff, merge, rollback, access control, and fork. The mental model is already in every developer's head. No competitor has any of this.

2. **Memory that learns from production** -- Confidence scoring evolves from incidents, deployments, and regressions. No other system connects memory to real-world outcomes.

3. **Self-organizing memory** -- Tiered memory (Hot/Warm/Archive), frequency-modulated decay, multi-dimensional importance scoring, and progressive retrieval. Not just a retrieval layer — a memory system that reorganizes itself based on what matters.

4. **Copy-on-Write versioning** -- Every write is immutable. Replay history, compare versions, audit decisions. Most competitors overwrite.

5. **Complete decision traces** -- `explain()` + `record_context()` capture the full causal chain. Pro persists traces permanently with cryptographic integrity.

6. **Multi-agent native** -- Provenance tracking, conflict detection, auto-causal linking, and per-agent identity. Not bolted on.

7. **Cross-system context** -- Webhooks from PagerDuty, Slack, GitHub, Jira flow into the same memory store. No competitor unifies agent memory with operational events.

8. **Framework and infrastructure agnostic** -- Any framework, any IDE via MCP, any storage backend via adapters.

9. **Enterprise-grade** -- Postgres RLS, RBAC, scoped API keys, audit logging, rate limiting, usage quotas. Purpose-built for multi-tenant SaaS.
