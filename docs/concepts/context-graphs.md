---
title: Context Graphs
layout: default
parent: Core Concepts
nav_order: 5
description: "How AMFS captures decision traces that explain not just what happened, but why."
---

# Context Graphs
{: .no_toc }

AMFS doesn't just store what agents know — it captures **why** they acted. Every read, write, tool consultation, and outcome forms a decision trace that becomes searchable precedent.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The Problem

AI agents are stateless. Each session starts from scratch with no memory of past decisions, patterns, or mistakes. When something goes wrong, there's no way to answer:

- **What information did the agent have when it decided?**
- **What external sources did it consult?**
- **Has a similar decision been made before, and what happened?**

Traditional systems of record (CRMs, ERPs, ticketing systems) store the **final state** — the deal was closed, the ticket was resolved, the deploy succeeded. But the reasoning that connected data to action was never treated as data in the first place.

---

## Decision Traces

A **decision trace** is a structured record of how an agent turned context into action:

```
Agent reads AMFS entries:
  checkout-service/retry-pattern (v3, confidence: 0.92)
  checkout-service/risk-race-condition (v1, confidence: 0.78)

Agent consults external sources:
  PagerDuty API → 3 SEV-1 incidents in last 24h
  git log → 15 commits since last deploy, 3 touching retry logic

Agent writes decision:
  checkout-service/decision-rollback-retry → "Rolling back retry changes due to
  incident correlation and high-risk signal from prior agent"

Outcome recorded:
  DEP-500 → clean_deploy → confidence on causal entries adjusted
```

This trace is queryable. The next agent — or a human auditor — can replay exactly what happened and why.

---

## How AMFS Captures Decision Traces

AMFS's existing primitives map directly to each component of a decision trace:

### 1. What the agent knew

Every `read()` call is automatically logged by the `ReadTracker`. When the agent later commits an outcome, AMFS knows exactly which entries informed the decision.

```python
entry = mem.read("checkout-service", "retry-pattern")
entry = mem.read("checkout-service", "risk-race-condition")
```

### 2. What external inputs were gathered

`record_context()` captures tool calls, API responses, and other external inputs without writing to storage. These appear in the causal chain alongside AMFS reads.

```python
mem.record_context(
    "pagerduty-incidents",
    "3 SEV-1 incidents in last 24h for checkout-service",
    source="PagerDuty API",
)

mem.record_context(
    "git-log",
    "15 commits since last deploy, 3 touching retry logic",
    source="git",
)
```

### 3. What the agent decided

The agent's decision is written as a versioned, provenanced entry:

```python
mem.write(
    "checkout-service",
    "decision-rollback-retry",
    "Rolling back retry changes due to incident correlation",
    confidence=0.9,
    memory_type=MemoryType.EXPERIENCE,
    pattern_refs=["checkout-service/retry-pattern"],
)
```

### 4. What happened next

Outcomes close the feedback loop. Confidence on causal entries adjusts automatically:

```python
mem.commit_outcome("DEP-500", OutcomeType.CLEAN_DEPLOY)
```

### 5. Why did we do that?

`explain()` returns the complete causal chain:

```python
chain = mem.explain("DEP-500")
```

Returns:

```json
{
  "outcome_ref": "DEP-500",
  "agent_id": "deploy-agent",
  "session_id": "sess-a1b2c3d4",
  "causal_chain_length": 2,
  "causal_entries": [
    {"entity_path": "checkout-service", "key": "retry-pattern", "confidence": 0.92, "...": "..."},
    {"entity_path": "checkout-service", "key": "risk-race-condition", "confidence": 0.78, "...": "..."}
  ],
  "external_contexts": [
    {"label": "pagerduty-incidents", "summary": "3 SEV-1 incidents in last 24h", "source": "PagerDuty API", "recorded_at": "..."},
    {"label": "git-log", "summary": "15 commits since last deploy, 3 touching retry logic", "source": "git", "recorded_at": "..."}
  ]
}
```

---

## From Traces to Graphs

Individual decision traces accumulate into a **context graph**: entities connected by decision events with "why" links.

```
checkout-service/retry-pattern
    ├── read by deploy-agent (sess-a1b2)
    │   ├── also consulted: PagerDuty API, git log
    │   ├── wrote: decision-rollback-retry
    │   └── outcome: DEP-500 (clean_deploy) → confidence adjusted
    │
    ├── read by review-agent (sess-e5f6)
    │   ├── also consulted: Sentry errors
    │   ├── wrote: risk-timeout-regression
    │   └── outcome: INC-1042 (p1_incident) → confidence boosted
    │
    └── read by onboard-agent (sess-g7h8)
        └── wrote: task-summary-retry-review
```

Over time, this graph captures institutional knowledge that no single agent or person holds:

- **Precedent** — "The last time we saw this pattern, we rolled back and it worked."
- **Exception logic** — "We always check PagerDuty before deploying changes to retry logic."
- **Cross-system synthesis** — "The decision combined AMFS memory, PagerDuty incidents, and git history."

---

## The Feedback Loop

Decision traces compound through AMFS's outcome system:

```
Agent makes decision → writes entry + records trace
                           │
                           ▼
                   Outcome occurs (deploy, incident)
                           │
                           ▼
                   Confidence adjusts on causal entries
                           │
                           ▼
                   Next agent sees high-confidence precedent
                           │
                           ▼
                   Agent makes better decision → new trace
```

Entries validated by positive outcomes decay slower. Entries correlated with incidents get boosted. The system learns which decisions lead to good outcomes without any explicit training.

---

## Immutability and Replay

Because AMFS uses Copy-on-Write versioning, decision traces are **immutable**. You can reconstruct the exact state of the world at any past decision point:

```python
from datetime import datetime, timezone

decision_time = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)

versions = mem.history(
    "checkout-service",
    "retry-pattern",
    until=decision_time,
)

state_at_decision = versions[-1] if versions else None
```

This is the difference between a system of record (stores the current state) and a context graph (stores the decision lineage). AMFS preserves both.

---

## MCP Integration

Via the MCP server, AI coding agents can build decision traces automatically:

```
1. amfs_search("checkout-service")           → find existing context
2. amfs_read("checkout-service", "pattern")  → read relevant entries (auto-tracked)
3. amfs_record_context("git-log", "...", "git")  → capture tool inputs
4. amfs_write("checkout-service", "decision-...", "...")  → record the decision
5. amfs_commit_outcome("DEP-500", "clean_deploy")  → close the loop
6. amfs_explain("DEP-500")                   → inspect the full trace
```

No manual instrumentation required. The causal chain builds itself from normal agent workflow.
