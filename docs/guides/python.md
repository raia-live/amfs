---
title: Python SDK
layout: default
parent: Guides
nav_order: 1
description: "Complete guide to the AMFS Python SDK."
---

# Python SDK
{: .no_toc }

The Python SDK provides the `AgentMemory` class — the primary interface for reading, writing, and managing agent memory.

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

```bash
pip install amfs
```

---

## Creating an Instance

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="my-agent")
```

### With Custom Configuration

```python
from pathlib import Path

mem = AgentMemory(
    agent_id="my-agent",
    config_path=Path("./custom-amfs.yaml"),
    ttl_sweep_interval=60.0,
    decay_half_life_days=30.0,
)
```

### With a Pre-configured Adapter

```python
from amfs_filesystem import FilesystemAdapter

adapter = FilesystemAdapter(root=Path(".amfs"), namespace="staging")
mem = AgentMemory(agent_id="my-agent", adapter=adapter)
```

### With an Importance Evaluator

```python
from amfs_core.importance import ImportanceEvaluator

class MyEvaluator(ImportanceEvaluator):
    def evaluate(self, entity_path, key, value):
        score = 0.8 if "critical" in str(value).lower() else 0.3
        return score, {"criticality": score}

mem = AgentMemory(agent_id="my-agent", importance_evaluator=MyEvaluator())
```

Every `write()` call will automatically score the entry and set `importance_score` and `importance_dimensions`. If the evaluator raises, the write proceeds without scoring.

---

## Core Operations

### Write

```python
entry = mem.write(
    "checkout-service",          # entity_path
    "retry-pattern",             # key
    {"max_retries": 3},          # value (any JSON-serializable data)
    confidence=0.85,             # optional, default 1.0
    pattern_refs=["retry"],      # optional cross-references
    memory_type=MemoryType.FACT, # optional: fact (default), belief, or experience
)
```

Write with a TTL (time-to-live):

```python
from datetime import datetime, timedelta, timezone

mem.write(
    "svc", "temp-flag", {"active": True},
    ttl_at=datetime.now(timezone.utc) + timedelta(hours=24),
)
```

### Read

```python
entry = mem.read("checkout-service", "retry-pattern")

if entry:
    print(entry.value)
    print(entry.version)
    print(entry.confidence)
```

With minimum confidence filter:

```python
entry = mem.read("svc", "pattern", min_confidence=0.5)
```

### List

```python
# All entries
entries = mem.list()

# Entries for a specific entity
entries = mem.list("checkout-service")

# Include superseded versions
entries = mem.list("checkout-service", include_superseded=True)
```

### Search

```python
results = mem.search(entity_path="checkout-service", min_confidence=0.5)
```

Progressive retrieval with `depth` — search only high-priority tiers for fast, high-signal results:

```python
hot_only = mem.search(query="retry strategy", depth=1)   # Hot tier
hot_warm = mem.search(query="retry strategy", depth=2)   # Hot + Warm
all_tiers = mem.search(query="retry strategy")            # All (default)
```

Composite recall scoring (blends semantic similarity, recency, and confidence):

```python
from amfs_core.models import RecallConfig

scored = mem.search(
    query="how do we handle retries?",
    recall_config=RecallConfig(semantic_weight=0.5, recency_weight=0.3, confidence_weight=0.2),
)
for item in scored:
    print(f"{item.entry.key} — score={item.score:.3f}")
```

{: .note }
Semantic scoring requires an `embedder`. Without one, the semantic component is 0.0.

### Stats

```python
stats = mem.stats()
print(f"Total entries: {stats.total_entries}")
print(f"Total outcomes: {stats.total_outcomes}")
```

---

## Outcomes

### Recording Outcomes

```python
from amfs import OutcomeType

# With explicit causal keys
updated = mem.commit_outcome(
    outcome_ref="INC-1042",
    outcome_type=OutcomeType.CRITICAL_FAILURE,
    causal_entry_keys=["checkout-service/retry-pattern"],
)

# With auto-causal linking (uses everything read in this session)
updated = mem.commit_outcome(
    outcome_ref="DEP-300",
    outcome_type=OutcomeType.SUCCESS,
)
```

### Outcome Types

```python
OutcomeType.CRITICAL_FAILURE  # × 1.15
OutcomeType.FAILURE           # × 1.10
OutcomeType.MINOR_FAILURE     # × 1.08
OutcomeType.SUCCESS           # × 0.97
```

---

## Memory Types

Classify entries to control decay behavior:

```python
from amfs import MemoryType

# Facts (default) — objective knowledge, standard decay
mem.write("svc", "config", {"pool_size": 10}, memory_type=MemoryType.FACT)

# Beliefs — subjective inferences, decay 2× faster
mem.write("svc", "hypothesis", "Likely an N+1 query issue", memory_type=MemoryType.BELIEF)

# Experiences — action logs, decay 1.5× slower
mem.write("svc", "action-log", "Added index on user_id", memory_type=MemoryType.EXPERIENCE)
```

---

## History (Temporal Queries)

Retrieve the full version history of an entry with optional time filtering:

```python
from datetime import datetime, timedelta, timezone

# All versions
versions = mem.history("checkout-service", "retry-pattern")
for v in versions:
    print(f"v{v.version} — confidence: {v.confidence} — {v.provenance.written_at}")

# Versions from the last 7 days
since = datetime.now(timezone.utc) - timedelta(days=7)
recent = mem.history("checkout-service", "retry-pattern", since=since)
```

---

## Explainability

Inspect the causal chain — which entries were read during the current session and how they connect to outcomes:

```python
chain = mem.explain()
print(chain["session_id"])
print(chain["causal_keys"])   # list of entity_path/key pairs that were read
print(chain["entries"])       # full entry details for each causal key
```

Filter by outcome reference:

```python
chain = mem.explain(outcome_ref="INC-1042")
```

---

## Decision Traces

When you call `commit_outcome()`, AMFS snapshots the full decision trace — every entry that was read, every context that was recorded, and every query that was made. The resulting trace is persisted and can be retrieved later.

### Getting the trace from an outcome

```python
mem.record_context("ci-check", "All tests green", source="GitHub Actions")
entry = mem.read("checkout-service", "retry-pattern")

updated = mem.commit_outcome("DEP-500", OutcomeType.SUCCESS)

# The trace is attached to the outcome
trace = mem._last_trace
print(f"Trace ID: {trace.id}")
print(f"Causal entries: {len(trace.causal_entries)}")
print(f"External contexts: {len(trace.external_contexts)}")
```

### Browsing past traces

```python
# List recent traces
traces = mem._adapter.list_traces(limit=10)
for t in traces:
    print(f"{t['id']} — {t['agent_id']} — {t['outcome_ref']} ({t['outcome_type']})")

# Get a specific trace
trace = mem._adapter.get_trace("ddbcefff-901a-4fa6-...")
print(trace.decision_summary)
print(f"Session duration: {trace.session_duration_ms}ms")
for entry in trace.causal_entries:
    print(f"  Read: {entry.entity_path}/{entry.key} (v{entry.version})")
```

### Filtering traces

```python
traces = mem._adapter.list_traces(
    entity_path="checkout-service",
    agent_id="deploy-agent",
    outcome_type="success",
    limit=5,
)
```

---

## Tool Context

When agents call external tools or APIs, there are two ways to capture that context in AMFS depending on your needs.

### Record in the causal chain (lightweight)

Use `record_context()` to add external inputs to the causal chain without writing to storage. This makes `explain()` return a complete decision trace:

```python
entry = mem.read("checkout-service", "retry-pattern")

mem.record_context(
    "pagerduty-incidents",
    "3 SEV-1 incidents in the last 24h for checkout-service",
    source="PagerDuty API",
)
mem.record_context(
    "git-log",
    "15 commits since last deploy, 3 touching retry logic",
    source="git",
)

mem.commit_outcome("DEP-500", OutcomeType.SUCCESS)

chain = mem.explain()
print(chain["causal_entries"])     # AMFS entries that were read
print(chain["external_contexts"])  # tool/API inputs that informed the decision
```

### Persist for other agents (durable)

Use `MemoryType.EXPERIENCE` with a TTL to store tool results so downstream agents can retrieve them:

```python
from datetime import datetime, timedelta, timezone

mem.write(
    "checkout-service",
    "tool-result-pagerduty",
    {"incidents": 3, "sev1": True, "last_24h": True},
    memory_type=MemoryType.EXPERIENCE,
    ttl_at=datetime.now(timezone.utc) + timedelta(hours=1),
)
```

The next agent reads it with `mem.read("checkout-service", "tool-result-pagerduty")` instead of re-calling the API.

---

## Watch

Get real-time notifications when entries change:

```python
def on_change(entry):
    print(f"{entry.key} updated to v{entry.version}")

handle = mem.watch("checkout-service", on_change)

# Stop watching
handle.cancel()
```

---

## Briefing (Memory Cortex)

When the Memory Cortex is running, agents can retrieve pre-compiled knowledge digests ranked by relevance. This is how agents consume the "brain brief" — compiled summaries of entities, other agents, and external sources — without having to search through raw memory entries.

### Basic Usage

```python
mem = AgentMemory(agent_id="deploy-agent", adapter=adapter)

# Get a ranked briefing of compiled knowledge
briefs = mem.briefing(entity_path="myapp/checkout-service", limit=5)

for digest in briefs:
    print(digest.digest_type)   # "entity", "agent_brief", "source", or "connection_map"
    print(digest.scope)         # the entity path, agent ID, or source ID
    print(digest.summary)       # structured summary (varies by digest type)
    print(digest.entry_count)   # number of source entries compiled
    print(digest.compiled_at)   # when the digest was last compiled
```

### Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `entity_path` | `str \| None` | Focus on digests relevant to this entity |
| `agent_id` | `str \| None` | Focus on digests relevant to this agent |
| `limit` | `int` | Maximum number of digests to return (default: 10) |

### Digest Types

| Type | Scope | What It Contains |
|:-----|:------|:-----------------|
| `entity` | Entity path (e.g. `myapp/checkout-service`) | Summary of all knowledge about an entity — key count, average confidence, top keys, narrative |
| `agent_brief` | Agent ID (e.g. `deploy-agent`) | Summary of an agent's knowledge and activity — entries written, entities touched, outcomes |
| `source` | Source ID (e.g. `github`) | Summary of external data from a connector — events ingested, entities touched |
| `connection_map` | Cross-entity scope | Cross-entity relationships (Pro) |

### Workflow Integration

Call `briefing()` at the start of a task to get context before making decisions:

```python
with AgentMemory(agent_id="deploy-agent", adapter=adapter) as mem:
    # Step 1: Get a briefing on what you need to know
    briefs = mem.briefing(entity_path="myapp/checkout-service", limit=5)
    for digest in briefs:
        print(f"[{digest.digest_type}] {digest.scope}: {digest.summary.get('narrative', '')}")

    # Step 2: Read specific entries based on the briefing
    entry = mem.read("myapp/checkout-service", "retry-pattern")

    # Step 3: Do your work, record context
    mem.record_context("ci-pipeline", "All tests passing, deploy ready", source="GitHub Actions")

    # Step 4: Commit the outcome
    mem.commit_outcome("DEP-500", OutcomeType.SUCCESS)
```

If the Cortex is not running, `briefing()` returns an empty list — your agent code can safely call it without checking.

### Environment scoping

`briefing()` and `retrieve()` send the run's environment — `model`, `agent_version`, `runtime`, `platform` — read from the session (`set_session_metadata`, or `set_session_attributes` with those keys). A procedure whose `preconditions` name a different environment (`{"runtime": "python3.12"}` against a python3.9 run) is set apart as `procedures_not_applicable` in the briefing and dropped from retrieve hits (named in the trailing `_meta.not_applicable`), with the failing precondition spelled out. The lead digest and `_meta` also carry `guidance_strength` — `strong` (validated knowledge or a winning action here), `thin`, or `none` (only untested notes). Pass `environment={}` to send nothing.

---

## Guidance: the three seams (`Run`)

You control your agent's code; SenseLab does not. `amfs.Run` is the smallest way to wire it in — three calls at three points of a task — and it closes the loop that makes memory improve: guidance in, actions out, outcome in.

```python
from amfs import AgentMemory, Run

mem = AgentMemory(agent_id="ci-bot", adapter=adapter)
run = Run(mem)

# 1. Before the agent acts: guidance scoped to this run.
guidance = run.begin(
    "fix the failing CI on PR 42",
    entity_path="acme/ci",
    model="gpt-4o", agent_version="ci-bot@1.4.0", runtime="python3.12",
    candidate_actions=["shell:pip_install", "edit:requirements", "shell:pytest"],
)
if guidance.should_inject():          # strength is "strong" or "thin"
    system_prompt = guidance.text + "\n\n" + system_prompt

# 2. After each consequential tool call: record it; on failure, ask for a hint.
hint = run.on_tool_result(
    "shell", {"cmd": "pip install -r requirements.txt"},
    result=stderr, success=False, action_key="shell:pip_install",
)
if hint is not None and hint.should_inject():
    messages.append({"role": "system", "content": hint.text})

# When the approach you were following did not work and you switch — name
# the entry you followed (the keys are in guidance.text) or fall back to the
# top hit:
run.attempt_failed("pinning urllib3 did not resolve the conflict",
                   causal_entry_keys=cited or [guidance.top_key])

# 3. When the run ends: the outcome, who decided it, and what it acted on.
run.complete(True, verified_by="ci", evidence={"run_id": "98765"},
             response_text=final_answer,
             causal_entry_keys=cited or [guidance.top_key])

# 4. What this run learned, as a claim the record can follow.
run.learn("pip install fails on a urllib3/requests conflict", "edit:requirements", True,
          "pinning urllib3<2 resolves it; re-running the install does not")
```

**Name what the agent acted on.** `begin` serves several entries and the agent follows one. `causal_entry_keys` on `attempt_failed` and `complete` sends the outcome to that one; left unnamed, *every* entry read since the last boundary is charged, and a correct lesson that merely shared the context with a wrong one loses confidence alongside it — enough of that and the briefing flags a regime shift that never happened. Ask your model to return the keys it used (add a `used_memory_keys` field to its structured output; the keys appear as `entity/key` in the rendered text), intersect them with `guidance.entry_keys`, and fall back to `guidance.top_key`. When the agent took an action a lesson recommended, `guidance.lessons_claiming(action)` gives the keys of the lessons that claimed it — the causal keys without asking the model. Bare keys are qualified with the run's `entity_path`; `[]` credits nothing.

**Follow the plan, not just the first action.** `guidance.plan` is the order to try actions in for the whole budget: the recommendation's action first, then what has a winning record on tasks like this, then the untried candidates — never an action that failed here. `suggested_action` is one attempt's worth of advice; an agent with three attempts that follows it and fails is otherwise back on its own instinct, which on tasks like this has already failed. The rendered text carries the same order (`Try in this order: …`) and names what not to spend an attempt on.

**Lessons for the exact situation outrank the pooled record.** Priors are counted over the outcomes nearest the task, and that neighbourhood can hold two classes of task with opposite answers — a dependency audit on a pinned package and one on an unpinned package read alike, and `bump_dependency` wins on one while a house rule forbids it on the other. A lesson written with `run.learn(situation, action, worked)` is about one class. When a shown lesson's situation is the one the run declared (`begin(..., situation=...)`), or its words are all in the task text, the plan is re-ordered by it: an action a lesson says did not work here goes to the back, one it says worked goes to the front (after the recommendation's own action). `guidance.applicable_lessons` lists the claims that were read as about this task. The server does the same re-ordering when it computes the plan, so an MCP agent reads the same order.

**Write lessons as claims.** `run.learn(situation, action, worked, text)` writes a structured lesson: *situation* is the kind of task, *action* the `tool:action` key, *worked* the verdict, *text* your agent's words. The claim is `(situation, action, worked)`; the words may change every time. A restated lesson inherits its outcome record instead of opening an untested version, so eight confirmations stay eight, a discredited lesson does not come back clean, and the regime-shift reading — which needs the second failure in a row to land on the same record as the first — can fire. The key is one per situation (`learned-<slug>-<hash>`), so the lesson that said an action worked and the later one that says it did not are versions of one entry. Renders as `When: <situation>. <action> worked. <text>`.

**Memory being down is not your agent being down.** If the retrieve behind `begin()` fails (a timeout, a 5xx), `begin` returns empty guidance with `error` set instead of raising; the run continues without guidance and `complete()` still seals the outcome. `on_tool_result` returns `None` in the same case.

`Guidance` carries:

| Field | Meaning |
|-------|---------|
| `text` | Rendered blocks — procedures first, then memory context, then `Not for this run`, then priors, the recommendation, what not to retry, and the plan — in the same shape a tuned model is trained on (`amfs_core.render`). |
| `strength` | `strong` / `thin` / `none`. `should_inject()` is true for the first two. |
| `procedures` / `not_applicable` | Procedures that apply to this run, and those whose environment preconditions it contradicts (with `applicability_detail`). Never inject one from `not_applicable`. |
| `recommendation` | `act` / `explore` / `escalate` / `abstain` with `suggested_action`, `why` and `plan`; `priors` holds the per-action record. |
| `plan` / `next_action` | The order to try actions in (see above); `next_action` is its first entry. Computed from the priors when the server sent none. |
| `lessons` / `lessons_claiming(action)` | The structured lessons this run was shown, and the keys of those that claimed *action* worked — the causal keys for an outcome of taking it. |
| `applicable_lessons` | The lesson claims read as about *this* task (situation declared on `begin`, or its words in the task text); they re-order `plan`. |
| `error` | Why the guidance is empty when memory could not be reached; `None` when the read succeeded. |
| `guidance_id` | Names what was served (branch, entries and versions, render version). Stamped on the session as `guidance_id` / `guidance_count`, so the sealed trace says which guidance the agent saw. |

`model`, `agent_version` and `runtime` on `begin()` are the run's environment. They scope procedures to this run, and they are stamped on the sealed trace (`model` as its own field as well as an attribute) — which is what the repair loop's feedback contract reads: an automatic policy ships and rolls back fixes only when enough of your runs record which model and which version of your agent ran them, or a model swap on your side would be blamed on the fix. Pass them on every `begin`.

`verified_by` on `complete()` (or `commit_outcome()`) says who decided the outcome when the agent did not — `"ci"`, `"human"`, `"verifier"`, `"customer"`. It travels as the `verified_by` session attribute with `evidence_<key>` pointers; the repair loop weighs a verified outcome differently from an agent's own declaration, and holds automatic promotion until enough outcomes carry it.

`Run(mem, assign_branch=fn)` takes an optional hook `(agent_id, unit) -> branch | None` for a canary assignment; when it names a branch, the run reads memory from it and writes stay on `main`.

---

## Snapshots

Export and import the full state of your memory:

```python
from amfs_core.snapshot import SnapshotExporter, SnapshotImporter

# Export
exporter = SnapshotExporter(mem.adapter)
exporter.export("backup.json")

# Import into a different adapter
from amfs_filesystem import FilesystemAdapter
target = FilesystemAdapter(root=Path("/new/.amfs"), namespace="restored")
importer = SnapshotImporter(target)
importer.restore("backup.json")
```

---

## Knowledge Graph

The knowledge graph builds automatically as agents write, commit outcomes, and learn from each other. You can also traverse it directly:

```python
edges = mem.graph_neighbors(
    "checkout-service/retry-pattern",
    direction="both",
    depth=2,
    min_confidence=0.5,
)
for edge in edges:
    print(f"{edge.source_entity} --{edge.relation}--> {edge.target_entity}")
```

| Parameter | Description |
|:----------|:------------|
| `entity` | Starting entity to explore |
| `relation` | Filter by relation type (e.g. `"references"`, `"informed"`) |
| `direction` | `"outgoing"`, `"incoming"`, or `"both"` |
| `depth` | Traversal depth (1 = direct neighbors, >1 for multi-hop) |

{: .note }
Multi-hop traversal (`depth > 1`) requires the Postgres adapter. The Filesystem and S3 adapters return an empty list for graph methods.

---

## Semantic Search

If you configure an embedder, you can search by meaning:

```python
results = mem.semantic_search("how do we handle retries?", top_k=5)
for entry, score in results:
    print(f"{entry.key} (similarity: {score:.3f})")
```

---

## Context Manager

Use `AgentMemory` as a context manager for automatic cleanup:

```python
with AgentMemory(agent_id="my-agent") as mem:
    mem.write("svc", "key", "value")
    entry = mem.read("svc", "key")
# Watchers, TTL sweepers, and background threads are cleaned up
```

---

## Connecting to AMFS SaaS

When using AMFS as a hosted service (SaaS), connect through the HTTP API with your API key instead of a direct database connection.

### Environment Variables

```bash
export AMFS_HTTP_URL="https://amfs-login.sense-lab.ai"
export AMFS_API_KEY="amfs_sk_your_key_here"
```

With these set, the SDK auto-detects the HTTP adapter — no code changes needed:

```python
from amfs import AgentMemory

mem = AgentMemory(agent_id="my-agent")
mem.write("checkout-service", "retry-pattern", {"max_retries": 3})
```

### Explicit HttpAdapter

You can also configure the adapter directly:

```python
from amfs import AgentMemory
from amfs_adapter_http import HttpAdapter

adapter = HttpAdapter(
    base_url="https://amfs-login.sense-lab.ai",
    api_key="amfs_sk_your_key_here",
)
mem = AgentMemory(agent_id="my-agent", adapter=adapter)
```

{: .note }
Install the HTTP adapter with `pip install amfs-adapter-http`.

{: .warning }
Never use `AMFS_POSTGRES_DSN` for external agents in multi-tenant mode. Always use `AMFS_HTTP_URL` + `AMFS_API_KEY` to ensure tenant isolation, scope enforcement, and audit logging.

See the [SaaS Connection Guide](/amfs/guides/saas/) and [Environment Variables](/amfs/reference/environment-variables/) for details.

---

## Conflict Handling

Handle concurrent writes to the same key:

```python
from amfs_core.models import ConflictPolicy

# Raise an error on conflict
mem = AgentMemory(
    agent_id="my-agent",
    conflict_policy=ConflictPolicy.RAISE,
)

# Custom conflict resolution
def merge(existing, incoming, value):
    return {**existing.value, **value}

mem = AgentMemory(
    agent_id="my-agent",
    on_conflict=merge,
)
```
