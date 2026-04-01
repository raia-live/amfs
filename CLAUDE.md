# AMFS Memory — Agent Instructions

You have access to AMFS (Agent Memory File System) through MCP tools. AMFS is shared memory that persists across sessions, agents, and machines. Use it to build institutional knowledge over time.

## Available MCP Tools

- `amfs_read(entity_path, key)` — read a specific memory entry
- `amfs_write(entity_path, key, value, confidence?, pattern_refs?)` — write knowledge with automatic provenance
- `amfs_search(query?, entity_path?, min_confidence?, agent_id?, sort_by?, limit?)` — search across all entries
- `amfs_list(entity_path?)` — list entries for an entity
- `amfs_stats()` — memory overview
- `amfs_commit_outcome(outcome_ref, outcome_type)` — record outcomes, auto-links to read log

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
Warn other agents:
```
amfs_write("<repo>/<module>", "risk-<name>", "<what could go wrong>", confidence=0.8)
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
