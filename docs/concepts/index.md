---
title: Core Concepts
layout: default
nav_order: 3
has_children: true
description: "Understand how AMFS stores, versions, and evolves knowledge."
permalink: /concepts/
---

# Core Concepts

Understanding the building blocks of AMFS.
{: .fs-6 .fw-300 }

AMFS is built around a few core ideas that work together: **memory entries** store knowledge with a **memory type** (fact, belief, or experience), **copy-on-write** preserves history, **provenance** tracks authorship and **provenance tiers** rank quality, **confidence** scores reflect trust with type-specific decay, **outcomes** close the feedback loop, and a **git-like timeline** records every operation as an event on the agent's history.
{: .fs-6 .fw-300 }

---

## At a Glance

```
Agent writes →  MemoryEntry created
                  ├── entity_path: "checkout-service"
                  ├── key: "retry-pattern"
                  ├── value: { ... }
                  ├── memory_type: fact   ← fact / belief / experience
                  ├── version: 3          ← CoW versioning
                  ├── confidence: 0.85    ← evolves with outcomes
                  ├── branch: "main"      ← git-like branch
                  └── provenance:
                      ├── agent_id: "review-agent"
                      ├── session_id: "abc-123"
                      └── written_at: 2025-06-15T10:30:00Z

              →  Event logged on agent timeline
                  ├── event_type: "write"
                  ├── branch: "main"
                  ├── summary: "Wrote checkout-service/retry-pattern v3"
                  └── details: { entity_path, key, version, confidence }
```

Read the pages in this section to understand each concept in depth.
