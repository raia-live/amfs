# AMFS Memory — Agent Instructions

You have access to AMFS (Agent Memory File System) through MCP tools. AMFS gives you a **persistent brain** — memory that survives across sessions, agents, and machines. Use it to build institutional knowledge over time.

## Available MCP Tools

### Identity
- `amfs_set_identity(name, description?)` — set your agent identity for this conversation. Call first.

### Brain tools (agent-scoped)
- `amfs_recall(entity_path, key)` — recall YOUR OWN memory for a key (what do I know about this?)
- `amfs_my_entries(entity_path?)` — list everything YOU have written (what's in my brain?)
- `amfs_read_from(agent_id, entity_path, key)` — read from ANOTHER agent's memory (learn from a colleague)

### Shared knowledge tools
- `amfs_read(entity_path, key)` — read the latest version by any agent (shared pool)
- `amfs_write(entity_path, key, value, confidence?, pattern_refs?, memory_type?)` — form a memory. `memory_type` can be `"fact"` (default), `"belief"` (decays faster), or `"experience"` (decays slower)
- `amfs_search(query?, entity_path?, min_confidence?, agent_id?, sort_by?, limit?)` — search across all entries
- `amfs_list(entity_path?)` — list entries for an entity
- `amfs_stats()` — memory overview

### Tracing and explainability
- `amfs_commit_outcome(outcome_ref, outcome_type)` — **critical**: snapshots the full decision trace (all reads, writes, decisions, contexts) and persists it. Call this after completing significant work.
- `amfs_record_context(label, summary, source?)` — capture decisions, external tool results, or user choices in the causal chain. Call this **as decisions happen**, not at the end.
- `amfs_history(entity_path, key, since?, until?)` — retrieve version history of an entry
- `amfs_explain(outcome_ref?)` — inspect the current session's decision trace: reads + external contexts
- `amfs_list_traces(entity_path?, agent_id?, limit?)` — browse persisted decision traces from past sessions
- `amfs_get_trace(trace_id)` — retrieve a full decision trace by ID
- `amfs_cross_agent_reads()` — see which other agents' memory you've read

## Workflow

### MANDATORY FIRST STEP: Set your identity

> **You MUST call `amfs_set_identity` before doing anything else.** Without it, all your work is attributed to a generic default and you won't appear as a distinct agent on the AMFS dashboard.

```
amfs_set_identity("<role-name>", "<one-line description of current task>")
```

**Naming rules:**
- Use **kebab-case role/domain names** that persist across conversations about the same topic.
- Good: `"dashboard-agent"`, `"stripe-agent"`, `"api-agent"`, `"infra-agent"`, `"mcp-agent"`
- Bad: `"fix-button-color"` (too specific — won't be reused), `"agent-1"` (meaningless)
- If continuing work a previous agent started, **use the same name** to build on their knowledge.
- The description should say what you're doing *right now* (e.g. `"Fixing tag rollback for slashed names"`).

### Before starting work
Get a compiled briefing from the Memory Cortex first — this gives you pre-compiled knowledge about the entity you're about to work on, including what other agents know, recent risks, external events, and confidence-ranked facts:
```
amfs_briefing(entity_path="<repo>/<module>")
```
Then check your own specific memories:
```
amfs_recall("<repo>/<module>", "task-summary-<area>")
amfs_search(entity_path="<repo>/<service-or-module>")
```

### After completing a task
Form a memory of what was done and key decisions:
```
amfs_write("<repo>/<module>", "task-summary-<desc>", "<what and why>")
```

### When consulting another agent's work
Explicitly read from their brain so the knowledge transfer is tracked:
```
amfs_read_from("<agent_id>", "<repo>/<module>", "<key>")
```

### When discovering patterns
Record reusable patterns with cross-references:
```
amfs_write("<repo>/<module>", "pattern-<name>", "<description>", pattern_refs=["related-key"])
```

### When finding bugs or risks
Warn other agents (use `memory_type="belief"` for hypotheses that need validation):
```
amfs_write("<repo>/<module>", "risk-<name>", "<what could go wrong>", confidence=0.8, memory_type="belief")
```

### When logging actions taken
Record what you did so future agents can retrace steps (experiences decay slower):
```
amfs_write("<repo>/<module>", "action-<desc>", "<what you did>", memory_type="experience")
```

### When decisions are made (build the causal chain)
Record decisions **as they happen** — both your own and the user's:
```
amfs_record_context("user-decision", "User chose thread-local over request-scoped", source="chat")
amfs_record_context("architecture-decision", "Using uvx for distribution", source="analysis")
amfs_record_context("pagerduty-incidents", "3 SEV-1 in last 24h", source="PagerDuty API")
```

### After completing significant work (commit the trace)
**Always call `commit_outcome`** after finishing a task. This snapshots all reads, writes, decisions, and contexts into a persisted `DecisionTrace`:
```
amfs_commit_outcome("tenant-rls-fix", "success")
amfs_commit_outcome("<ticket>", "minor_failure")
amfs_commit_outcome("<incident-id>", "critical_failure")
```
Without this, the decision trace is lost when the session ends.

### Before making similar decisions (browse past traces)
Check if a past decision trace already covers this area:
```
amfs_list_traces(entity_path="<repo>/<module>", limit=5)
amfs_get_trace("<trace-id>")
```

## Entity Naming

Use `{repo}/{service-or-module}` paths:
- `myapp/checkout-service`
- `myapp/auth`
- `amfs/core-engine`

## Confidence Scale

- **1.0** — verified fact, tested pattern
- **0.7-0.9** — high confidence, not yet production-validated
- **0.4-0.6** — hypothesis, needs validation
- **< 0.4** — speculative signal

## Quality Feedback on Writes

When you call `amfs_write`, the response includes a `quality` field with a score and any issues found.
If the quality score is below 0.8, review the `issues` array and consider calling `amfs_write` again
with an improved value. Common issues:
- **too_short**: Value lacks detail. Add specifics: what, why, key parameters.
- **missing_pattern_refs**: Related entries exist. Add pattern_refs to link them.
- **belief_no_rationale**: Beliefs should explain reasoning (use "because", "hypothesis").
- **overconfident_belief**: Beliefs should have confidence < 0.9.

## Guidelines

- **Always start with `amfs_briefing`** — this gives you compiled, ranked knowledge from the Cortex before you dig into specifics
- Only write information that would help a future agent working on the same code
- Keep values concise but informative — like writing a note to a colleague
- Use `amfs_recall` for specific keys, `amfs_search` for broader queries
- Use `amfs_read_from` when you know which agent has the knowledge you need
- Search before writing to avoid duplicating existing knowledge
- Confidence decays over time; entries validated by outcomes decay slower
