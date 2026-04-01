# AMFS Memory — Agent Instructions

You have access to AMFS (Agent Memory File System) through MCP tools. AMFS is shared memory that persists across sessions, agents, and machines. Use it to build institutional knowledge over time.

## Available MCP Tools

- `amfs_read(entity_path, key)` — read a specific memory entry
- `amfs_write(entity_path, key, value, confidence?, pattern_refs?, memory_type?)` — write knowledge with automatic provenance. `memory_type` can be `"fact"` (default), `"belief"` (decays faster), or `"experience"` (decays slower)
- `amfs_search(query?, entity_path?, min_confidence?, agent_id?, sort_by?, limit?)` — search across all entries
- `amfs_list(entity_path?)` — list entries for an entity
- `amfs_stats()` — memory overview
- `amfs_commit_outcome(outcome_ref, outcome_type)` — record outcomes, auto-links to read log
- `amfs_record_context(label, summary, source?)` — capture external tool/API context in the causal chain (appears in `amfs_explain` output)
- `amfs_history(entity_path, key, since?, until?)` — retrieve version history of an entry, optionally filtered by time range
- `amfs_explain(outcome_ref?)` — inspect the full decision trace: AMFS reads + external contexts

## Workflow

### Before starting work
Search AMFS for existing context about the code you're about to modify:
```
amfs_search(entity_path="<repo>/<service-or-module>")
```

### After completing a task
Write a structured summary of what was done and key decisions:
```
amfs_write("<repo>/<module>", "task-summary-<desc>", "<what and why>")
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

### When reviewing history
Check how a memory evolved over time:
```
amfs_history("<entity_path>", "<key>")
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

- Only write information that would help a future agent working on the same code
- Keep values concise but informative — like writing a note to a colleague
- Search before writing to avoid duplicating existing knowledge
- Confidence decays over time; entries validated by outcomes decay slower
