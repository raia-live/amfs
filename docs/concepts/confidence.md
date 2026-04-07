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
| `CRITICAL_FAILURE` | × 1.15 | Strong confidence increase — knowledge linked to severe failure |
| `FAILURE` | × 1.10 | Moderate confidence increase |
| `MINOR_FAILURE` | × 1.08 | Mild confidence increase — knowledge linked to minor setback |
| `SUCCESS` | × 0.97 | Confidence decay — knowledge proving reliable over time |

---

## How It Works

### Recording an Outcome

```python
from amfs import OutcomeType

# An incident happened related to the retry pattern
updated = mem.commit_outcome(
    outcome_ref="INC-1042",         # reference ID (ticket, deploy ID, etc.)
    outcome_type=OutcomeType.CRITICAL_FAILURE,
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
Write              → 0.85
Critical failure   → 0.85 × 1.15 = 0.978
Success            → 0.978 × 0.97 = 0.948
Success            → 0.948 × 0.97 = 0.920
Success            → 0.920 × 0.97 = 0.892
Failure            → 0.892 × 1.10 = 0.981
Success            → 0.981 × 0.97 = 0.951
```

Over many successes, confidence trends toward zero — the risk signal fades. A single failure spikes it back up.

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
mem.commit_outcome("DEP-300", OutcomeType.SUCCESS)
```

This is powered by the **ReadTracker**, which logs every `read()` call during a session.

---

## Four-Signal Decay Model

When `decay_half_life_days` is configured, AMFS uses four signals to determine how fast an entry's effective confidence decays:

| Signal | Effect | Mechanism |
|:-------|:-------|:----------|
| **Time** | Base exponential decay | `effective_half_life` in days |
| **Memory type** | Facts 1×, beliefs 0.5× (faster), experiences 1.5× (slower) | Type multiplier on half-life |
| **Outcome validation** | Validated entries decay 2× slower | `outcome_count > 0` doubles half-life |
| **Access frequency** | Frequently read entries resist decay | `log1p(recall_count)` extends half-life |

The effective half-life formula:

```
effective_half_life = base × type_multiplier × (1 + log1p(recall_count)) × (2 if outcomes else 1)
```

For example, a fact with `decay_half_life_days=30`, 10 reads, and 1 outcome:

```
30 × 1.0 × (1 + log1p(10)) × 2 = 30 × 1.0 × 3.40 × 2 = 204 days
```

This means actively used, production-validated knowledge persists far longer than cold, unvalidated beliefs.

```python
mem = AgentMemory(agent_id="my-agent", decay_half_life_days=30.0)
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
                   Incident occurs → commit_outcome("CRITICAL_FAILURE")
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
