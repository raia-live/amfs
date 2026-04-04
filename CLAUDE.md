# AMFS Memory — Agent Instructions

You have access to AMFS (Agent Memory File System) through MCP tools. AMFS gives you a **persistent brain** — memory that survives across sessions, agents, and machines. Use it to build institutional knowledge over time.

## Available MCP Tools

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
- `amfs_commit_outcome(outcome_ref, outcome_type)` — record outcomes, auto-links to read log
- `amfs_record_context(label, summary, source?)` — capture external tool/API context in the causal chain
- `amfs_history(entity_path, key, since?, until?)` — retrieve version history of an entry
- `amfs_explain(outcome_ref?)` — inspect the full decision trace: reads + external contexts
- `amfs_cross_agent_reads()` — see which other agents' memory you've read

## Workflow

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

### When consulting external tools or APIs
Record external context so the decision trace is complete:
```
amfs_record_context("pagerduty-incidents", "3 SEV-1 in last 24h", source="PagerDuty API")
```

### When something significant happens
Record outcomes to update confidence of related entries:
```
amfs_commit_outcome("<ref>", "clean_deploy")   # successful deploy
amfs_commit_outcome("<ref>", "regression")      # bug found
amfs_commit_outcome("<ref>", "p1_incident")     # major incident
amfs_commit_outcome("<ref>", "p2_incident")     # minor incident
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

## Guidelines

- **Always start with `amfs_briefing`** — this gives you compiled, ranked knowledge from the Cortex before you dig into specifics
- Only write information that would help a future agent working on the same code
- Keep values concise but informative — like writing a note to a colleague
- Use `amfs_recall` for specific keys, `amfs_search` for broader queries
- Use `amfs_read_from` when you know which agent has the knowledge you need
- Search before writing to avoid duplicating existing knowledge
- Confidence decays over time; entries validated by outcomes decay slower
