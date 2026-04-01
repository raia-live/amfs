---
title: AMFS vs Vector Databases
layout: default
nav_order: 5
description: "When to use AMFS, when to use a vector database, and how they work together."
permalink: /vs-vector-databases/
---

# AMFS vs Vector Databases
{: .no_toc }

Vector databases and AMFS solve different problems. Understanding the distinction helps you pick the right tool — or use both.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## The Short Version

**Vector databases** store embeddings and retrieve them by similarity. They answer: *"what is most relevant to this query?"*

**AMFS** stores versioned, provenanced knowledge and evolves it based on outcomes. It answers: *"what does this agent know, who wrote it, how confident are we, and what happened when we acted on it?"*

A vector database is a **retrieval engine**. AMFS is a **memory system**.

---

## Side-by-Side Comparison

| Dimension | Vector Database | AMFS |
|:----------|:----------------|:-----|
| **Primary operation** | Similarity search over embeddings | Read/write versioned knowledge with provenance |
| **Data model** | Vectors + metadata | Structured entries with entity/key scoping, confidence, memory type, provenance |
| **Versioning** | Overwrite or append | Copy-on-Write — every write creates a new version, full history preserved |
| **Who wrote it?** | Not tracked | Provenance: agent ID, session ID, timestamp, pattern refs |
| **Trust signal** | None | Confidence score that evolves based on real-world outcomes |
| **Feedback loop** | None | Outcome back-propagation — incidents boost confidence, clean deploys decay it |
| **Query style** | "Find similar to X" | "Read key Y", "Search by entity/agent/confidence", "What happened over time?" |
| **Temporal queries** | Snapshot at query time | Full version history with time-range filtering |
| **Explainability** | None | Causal chain: which entries + external contexts informed a decision |
| **Multi-agent** | Shared index | Shared memory with per-agent provenance, conflict detection, auto-causal linking |
| **Typical size** | Millions–billions of vectors | Thousands–millions of knowledge entries |
| **Update pattern** | Re-embed and upsert | CoW write with automatic version increment |

---

## What Vector Databases Do Well

Vector databases excel at **large-scale semantic retrieval**:

- **RAG (Retrieval-Augmented Generation)** — Finding relevant document chunks to inject into an LLM prompt. When you have 10M documents and need the top-10 most relevant passages, a vector database is the right tool.
- **Similarity search** — "Find products similar to this one", "Find code snippets that match this pattern."
- **Multimodal retrieval** — Searching across text, images, and audio using shared embedding spaces.
- **Real-time recommendation** — High-throughput, low-latency nearest-neighbor queries.

Popular options include Pinecone, Weaviate, Qdrant, Milvus, and Chroma.

---

## What Vector Databases Don't Do

Vector databases are **stateless retrieval indexes**. They don't track:

- **Who wrote the data** — No provenance. You don't know which agent or process created an entry.
- **How trust evolves** — No confidence scoring. A vector's relevance score is similarity to a query, not a measure of how trustworthy the information is.
- **What happened when you used it** — No outcome tracking. If an agent retrieves a vector and acts on it, and that action causes an incident, the vector database has no way to learn from that.
- **How data changed over time** — Vectors are overwritten or appended. You can't ask "what did this entry say last week?"
- **Why a decision was made** — No causal chain linking retrieved data to actions and outcomes.

---

## What AMFS Does Differently

AMFS is designed for **agent memory** — the layer between retrieval and action:

### Knowledge has identity

Every entry has an `entity_path` and `key` that give it a stable address. Agents read and write to specific keys, not anonymous vectors.

```python
mem.write("checkout-service", "retry-pattern", {"max_retries": 3})
entry = mem.read("checkout-service", "retry-pattern")
```

### Knowledge has provenance

Every entry records who wrote it, when, and in which session:

```python
print(entry.provenance.agent_id)    # "review-agent"
print(entry.provenance.written_at)  # 2026-03-15T14:30:00Z
```

### Knowledge evolves with outcomes

When a deploy succeeds or an incident occurs, confidence scores on related entries adjust automatically:

```python
mem.commit_outcome("INC-1042", OutcomeType.P1_INCIDENT)
# All entries this agent read get their confidence boosted
```

### Knowledge has full history

Every write creates a new version. You can replay the state at any point in time:

```python
versions = mem.history("checkout-service", "retry-pattern", since=last_week)
```

### Decisions are explainable

The causal chain shows exactly what informed a decision — both AMFS entries and external tool context:

```python
chain = mem.explain("DEP-500")
# → causal_entries: entries the agent read
# → external_contexts: tool/API inputs the agent consulted
```

---

## When to Use Which

| Scenario | Use |
|:---------|:----|
| RAG over large document corpus | Vector database |
| Semantic search across millions of records | Vector database |
| Agent memory that persists across sessions | AMFS |
| Multi-agent shared knowledge with provenance | AMFS |
| Confidence scoring based on real-world outcomes | AMFS |
| Decision audit trail and explainability | AMFS |
| Temporal queries ("what did we know last week?") | AMFS |
| Real-time similarity recommendations | Vector database |
| Knowledge that needs to learn from production | AMFS |

---

## Using Both Together

AMFS and vector databases are complementary. A common architecture:

```
                    Agent
                   /     \
                  /       \
         ┌──────▼──┐  ┌──▼──────────┐
         │  AMFS   │  │ Vector DB   │
         │ Memory  │  │ (RAG index) │
         └─────────┘  └─────────────┘
         What we know  What's relevant
         + trust       to this query
         + history
         + outcomes
```

1. **Vector DB for retrieval** — Agent queries the vector database to find relevant documents or code snippets for the current task.
2. **AMFS for memory** — Agent reads AMFS for known patterns, risks, and past decisions about the entity it's working on.
3. **AMFS for recording** — After completing its task, the agent writes findings, decisions, and risks to AMFS with provenance and confidence.
4. **AMFS for learning** — Outcomes (deploys, incidents) back-propagate through AMFS, adjusting confidence scores so future agents see which patterns are trustworthy.

AMFS even supports semantic search via pluggable embedders — so for small-to-medium memory stores, you can search by meaning without a separate vector database. For large-scale RAG over millions of documents, a dedicated vector database is the right choice.

---

## Summary

| | Vector Database | AMFS |
|:--|:----------------|:-----|
| **Think of it as** | A search index for embeddings | A version-controlled knowledge base for agents |
| **Best for** | Finding relevant data | Remembering, learning, and explaining decisions |
| **Data lifecycle** | Write once, query many | Write, version, track outcomes, decay, explain |
| **Multi-agent** | Shared index | Shared memory with provenance, conflicts, and causal chains |
