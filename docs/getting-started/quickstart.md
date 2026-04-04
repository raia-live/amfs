---
title: Quick Start
layout: default
parent: Getting Started
nav_order: 2
description: "Give your agent a persistent brain in 5 minutes."
---

# Quick Start

AMFS gives each agent a **persistent brain**. When an agent writes, it's forming a memory. When it recalls, it's accessing its own experience. When it reads shared knowledge, it benefits from what other agents have learned. Every interaction is tracked so you can always answer: *what did I know, when did I know it, and who told me?*

---

## 1. Create Your Brain

Every agent gets its own brain via `AgentMemory`:

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="review-agent")
```

The `agent_id` is the agent's identity. Everything it writes is tagged with this ID, and it can later recall only its own memories.

---

## 2. Form a Memory

When your agent learns something, write it to memory:

```python
entry = mem.write(
    "checkout-service",       # entity_path — what this knowledge is about
    "retry-pattern",          # key — name for this piece of knowledge
    {                         # value — any JSON-serializable data
        "pattern": "exponential-backoff",
        "max_retries": 3,
        "base_delay": "200ms",
    },
    confidence=0.85,
)

print(entry.version)                 # 1
print(entry.provenance.agent_id)     # "review-agent"
```

Every write creates an immutable copy-on-write version. Writing the same key again creates version 2, preserving the full history.

---

## 3. Recall Your Memory

Ask your brain: *"What do I know about this?"*

```python
entry = mem.recall("checkout-service", "retry-pattern")

print(entry.value)       # {"pattern": "exponential-backoff", ...}
print(entry.confidence)  # 0.85
```

`recall()` returns only entries **written by this agent**. If another agent wrote a different version, `recall()` ignores it — it's this brain's direct experience.

---

## 4. Read Shared Knowledge

Ask the shared pool: *"What does anyone know about this?"*

```python
entry = mem.read("checkout-service", "retry-pattern")
```

`read()` returns the latest version by **any agent**. This is how agents benefit from collective knowledge. Both `read()` and `recall()` return `None` if no entry exists.

---

## 5. Learn from Another Agent

Explicitly pull knowledge from a specific agent's brain:

```python
entry = mem.read_from("deploy-agent", "checkout-service", "deploy-config")

if entry:
    print(f"Learned from deploy-agent: {entry.value}")
```

`read_from()` makes cross-agent knowledge transfer explicit and trackable. The read is logged in the causal chain so you can always trace where knowledge came from.

---

## 6. See What's in Your Brain

List everything this agent has written:

```python
my_memories = mem.my_entries()
for e in my_memories:
    print(f"{e.entity_path}/{e.key} (v{e.version}, confidence={e.confidence})")

# Filter to a specific entity
checkout_memories = mem.my_entries("checkout-service")
```

---

## 7. Learn from Experience

When something significant happens, record the outcome. AMFS automatically adjusts confidence scores on related entries:

```python
from amfs import OutcomeType

# An incident related to the retry pattern — confidence increases
updated = mem.commit_outcome(
    outcome_ref="INC-1042",
    outcome_type=OutcomeType.P1_INCIDENT,
)
```

{: .tip }
If you don't pass `causal_entry_keys`, AMFS uses **auto-causal linking** — it applies the outcome to every entry the agent read during the current session.

---

## 8. Know Who You've Learned From

Track inter-agent memory relationships:

```python
# Which agents have I read from?
reads = mem.cross_agent_reads()
# {'deploy-agent': [{'entity_path': 'checkout-service', 'key': 'deploy-config', 'read_count': 2}]}

# Just the agent IDs
agents = mem.agents_i_read_from()
# ['deploy-agent']
```

---

## 9. Watch for Changes

Get notified in real-time when knowledge changes:

```python
def on_change(entry):
    print(f"Updated: {entry.entity_path}/{entry.key} v{entry.version}")

handle = mem.watch("checkout-service", on_change)

# ... later, stop watching
handle.cancel()
```

---

## 10. Context Manager

Use `AgentMemory` as a context manager for clean shutdown:

```python
with AgentMemory(agent_id="review-agent") as mem:
    mem.write("svc", "key", "value")
# Background threads cleaned up automatically
```

---

## The Mental Model

| What you want to do          | Method                  | Who wrote it?       |
| ---------------------------- | ----------------------- | ------------------- |
| Form a memory                | `write()`               | You (this agent)    |
| Recall your own knowledge    | `recall()`              | Only you            |
| Read shared knowledge        | `read()`                | Anyone (latest)     |
| Learn from a specific agent  | `read_from(agent_id)`   | That specific agent |
| See all your memories        | `my_entries()`          | Only you            |
| Know who taught you          | `cross_agent_reads()`   | Other agents        |

---

## Next Steps

- [Configuration](/amfs/getting-started/configuration/) — YAML config, adapters, and environment variables
- [Core Concepts](/amfs/concepts/) — understand CoW, confidence, and outcome propagation
- [Python SDK Guide](/amfs/guides/python/) — full SDK reference with advanced features
