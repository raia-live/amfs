---
title: Confidence & Outcomes
layout: default
parent: Core Concepts
nav_order: 3
description: "How confidence scores evolve based on real-world outcomes."
---

# Confidence & Outcomes

Every memory entry carries a **confidence score** that evolves over time based on real-world outcomes. This is AMFS's feedback loop — connecting agent observations to production reality.

---

## Confidence Score

Confidence starts at `1.0` by default and represents how much trust to place in an entry:

```python
entry = mem.write("svc", "pattern", "use connection pooling", confidence=0.85)
print(entry.confidence)  # 0.85
```

{: .note }
Confidence is not capped at 1.0. An entry involved in multiple incidents can have confidence > 1.0, representing a strong risk signal.

---

## Outcome Types

When something significant happens in the real world, you record it as an **outcome**:

| Outcome | Multiplier | Effect |
|:--------|:-----------|:-------|
| `P1_INCIDENT` | × 1.15 | Strong confidence increase — pattern is a proven risk |
| `P2_INCIDENT` | × 1.10 | Moderate confidence increase |
| `REGRESSION` | × 1.08 | Mild confidence increase — pattern caused a regression |
| `CLEAN_DEPLOY` | × 0.97 | Confidence decay — pattern is proving safe over time |

---

## How It Works

### Recording an Outcome

```python
from amfs import OutcomeType

# An incident happened related to the retry pattern
updated = mem.commit_outcome(
    outcome_ref="INC-1042",         # reference ID (ticket, deploy ID, etc.)
    outcome_type=OutcomeType.P1_INCIDENT,
    causal_entry_keys=["checkout-service/retry-pattern"],
)

for entry in updated:
    print(f"{entry.key}: confidence {entry.confidence}")
```

### Confidence Formula

```
new_confidence = old_confidence × outcome_multiplier
```

### Confidence Over Time

Imagine an entry written with `confidence=0.85`:

```
Write          → 0.85
P1 incident    → 0.85 × 1.15 = 0.978
Clean deploy   → 0.978 × 0.97 = 0.948
Clean deploy   → 0.948 × 0.97 = 0.920
Clean deploy   → 0.920 × 0.97 = 0.892
P2 incident    → 0.892 × 1.10 = 0.981
Clean deploy   → 0.981 × 0.97 = 0.951
```

Over many clean deploys, confidence trends toward zero — the risk signal fades. A single incident spikes it back up.

---

## Auto-Causal Linking

If you don't specify `causal_entry_keys`, AMFS automatically links the outcome to **every entry the agent read during the current session**:

```python
# Agent reads several entries during its work
mem.read("svc", "retry-pattern")
mem.read("svc", "timeout-config")
mem.read("svc", "pool-settings")

# Record outcome without specifying keys —
# it applies to all three entries above
mem.commit_outcome("DEP-300", OutcomeType.CLEAN_DEPLOY)
```

This is powered by the **ReadTracker**, which logs every `read()` call during a session.

---

## Type-Specific Decay

When `decay_half_life_days` is configured, confidence decays over time. The decay rate varies by memory type:

| Memory Type | Decay Multiplier | Effect |
|:------------|:----------------|:-------|
| `fact` | 1.0× | Standard half-life |
| `belief` | 0.5× | Half-life is halved (decays 2× faster) |
| `experience` | 1.5× | Half-life is 50% longer (decays slower) |

Additionally, entries with `outcome_count > 0` decay at **half the rate** of unvalidated entries. This means a production-validated fact with a 30-day half-life effectively has a 60-day half-life.

```python
mem = AgentMemory(
    agent_id="my-agent",
    decay_half_life_days=30.0,  # facts decay with 30-day half-life
)

# A belief with the same setting decays with a 15-day half-life
# An experience decays with a 45-day half-life
```

---

## Filtering by Confidence

Use `min_confidence` to filter out low-confidence entries:

```python
# Only return entries with confidence >= 0.5
entry = mem.read("svc", "pattern", min_confidence=0.5)

# Search with confidence filter
results = mem.search(min_confidence=0.7)
```

---

## The Feedback Loop

```
Agent observes pattern → writes entry (confidence: 0.85)
                              │
                              ▼
                   Incident occurs → commit_outcome("P1_INCIDENT")
                              │
                              ▼
                   Confidence increases → 0.978
                              │
                              ▼
                   Next agent sees high-confidence risk signal
                              │
                              ▼
                   Agent avoids the pattern → clean deploy
                              │
                              ▼
                   Confidence decays → 0.948
```

This creates a self-correcting system: risky patterns get flagged, safe patterns fade, and agents inherit the accumulated wisdom of past sessions.
