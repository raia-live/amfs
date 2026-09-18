# AMFS Memory — Agent Instructions

You have access to AMFS (Agent Memory File System) through MCP tools. AMFS gives you a **persistent brain** — memory that survives across sessions, agents, and machines. Use it to build institutional knowledge over time.

## Available MCP Tools

### Identity
- `amfs_set_identity(name, description?, model?, client_name?, tools_available?)` — set your agent identity. Persisted to disk (sticky) — survives process restarts. Always pass `model=` with your LLM model name. Call first.
- `amfs_whoami()` — check which identity is active and where it came from (in-process, sticky file, or auto-detected)
- `amfs_reset_identity()` — clear the sticky identity and revert to auto-detection

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
- `amfs_commit_outcome(outcome_ref, outcome_type, task_input?)` — **critical**: snapshots the full decision trace (all reads, writes, decisions, contexts, actions) and persists it. Call this after completing significant work, and pass `task_input` whenever you have it.
- `amfs_record_context(label, summary, source?)` — capture decisions, external tool results, or user choices in the causal chain. Call this **as decisions happen**, not at the end.
- `amfs_record_action(tool_name, arguments?, result?, success?)` — record a consequential action you took. Call this **right after taking it**.
- `amfs_record_attempt(outcome_type?, summary?, causal_entry_keys?)` — mark the approach you just tried as **failed** before trying another. The entries it relied on receive the failure at commit time instead of being reinforced by your recovery. No round-trip; travels with the outcome.
- `amfs_history(entity_path, key, since?, until?)` — retrieve version history of an entry
- `amfs_explain(outcome_ref?)` — inspect the current session's decision trace: reads + external contexts
- `amfs_list_traces(entity_path?, agent_id?, limit?)` — browse persisted decision traces from past sessions
- `amfs_get_trace(trace_id)` — retrieve a full decision trace by ID
- `amfs_cross_agent_reads()` — see which other agents' memory you've read

## Workflow

### MANDATORY FIRST STEP: Set your identity

> **You MUST call `amfs_set_identity` before doing anything else.** Without it, all your work is attributed to a generic default and you won't appear as a distinct agent on the AMFS dashboard.

```
amfs_set_identity("<role-name>", "<one-line description of current task>", model="<your-model-name>")
```

Always pass `model=` with your LLM model name (e.g. `"claude-4-opus"`, `"gpt-4o"`, `"claude-3.5-sonnet"`). This is recorded in decision traces so we know which model made each decision.

**Sticky identity**: Once set, the identity is saved to `~/.amfs/.identity` and automatically restored in future sessions — even across process restarts (e.g. Claude Desktop). You only need to call `amfs_set_identity` again to change roles or update the description.

Use `amfs_whoami()` to check the active identity. Use `amfs_reset_identity()` to clear it.

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
The lead digest carries the state of knowledge from the outcome record: `validated` (entries every outcome confirmed — act on these), `discredited` (entries a failure gated, with `replaced_by` where a later success is known — avoid these), and `regime_shift` (long-validated entries that recently started failing — something changed; verify before reusing anything in the scope). Every `hot_context` row carries an `evidence_status` of `untested`, `validated`, `contested` or `discredited`; an untested 0.9 and a validated 0.9 are different things to act on. Pass `compact=True` for just the lead digest with these sections, at a fraction of the tokens.

The briefing also carries `tried_here` — the actions taken on this entity and how each fared (`resolve:resend_email 6/6`, `resolve:update_payment_method 0/8`) — and, when a regime shift is flagged, an `explore` line naming the untried action assigned to you. Pass `since=<iso timestamp>` on a repeat call to receive only what changed.

Then check your own specific memories:
```
amfs_recall("<repo>/<module>", "task-summary-<area>")
amfs_search(entity_path="<repo>/<service-or-module>")
```

### Before choosing an action (ask for priors)
When the task ends in a choice among a fixed set of actions — a tool with an enum
`action` parameter, a runbook with numbered fixes — retrieve with the candidates
and follow the recommendation:
```
amfs_retrieve(query="<the task>", entity_path="<repo>/<module>", include_priors=True,
              candidate_actions=["resolve:resend_email", "resolve:update_payment_method", "resolve:refund"],
              compact=True)
```
The response ends with `action_priors` (per action: won/lost here on similar
tasks, who tried it, when) and a `recommendation`:
- `act` — the top hit is validated or an action has a good record here; take `suggested_action`.
- `explore` — everything tried here has failed or a regime shift is flagged; try `suggested_action` first. It is assigned per agent, so peers on the same problem explore different actions instead of the same one. The full `untried` list is included if you have reason to deviate.
- `escalate` — every candidate has been tried here and failed; escalate on the first attempt rather than after three.
`compact=True` returns the first two hits in full and the rest as one-liners; the priors and recommendation are always complete.

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

### When you take an action (record what you did)
Record consequential actions **right after taking them** — deploys, rollbacks, file
edits, refunds, PRs, setting changes. AMFS sees only its own tools, so a call to
your deploy or refund tool is invisible unless you record it:
```
amfs_record_action("deploy_rollback", {"service": "checkout", "to_version": "v41"})
amfs_record_action("refund_payment", {"charge_id": "ch_123"}, result="refunded")
amfs_record_action("deploy", {"service": "api"}, result="timed out", success=False)
```
Not for reads or searches — `record_context` covers what you learned, this covers
what you did. Use the real tool name, and use it consistently. When the action is one
of a fixed set, pass `action_key="<tool>:<action>"` (derived from an `action` argument
when you do not); it is what the outcome is credited to, and what the next agent's
`action_priors` are counted over. Never put free text in an action key.

### When an approach fails (mark the attempt, then try another)
The moment a remembered fix, runbook step or pattern did **not** work and you are
about to try something else, close the attempt:
```
amfs_record_attempt(summary="restarted worker per runbook; queue still stuck")
amfs_record_attempt(outcome_type="failure", causal_entry_keys=["acme/support/fix-restart"])
```
Everything you read and did since the previous attempt is attributed to it. At
`commit_outcome` those entries receive the attempt's failure and only what you read
afterwards is credited with the success — so a stale memory that sent you down the
wrong path loses confidence instead of gaining it, and SenseLab writes a
`lesson-contrast-*` entry naming what failed and what worked. Commit **once**, at the
end, whatever the final outcome; the attempts travel inside the same trace.

### After completing significant work (commit the trace)
**Always call `commit_outcome`** after finishing a task. This snapshots all reads, writes, decisions, contexts, and recorded actions into a persisted `DecisionTrace`:
```
amfs_commit_outcome("tenant-rls-fix", "success",
                    task_input="tenant queries leaking across accounts",
                    response_text="Added the tenant_id predicate to the RLS policy; the leak is closed.")
amfs_commit_outcome("<ticket>", "minor_failure")
amfs_commit_outcome("<incident-id>", "critical_failure")
```
Pass `task_input` whenever you have it — the request that started the work, in the
words it arrived in. Without it the trace records what you decided but not what you
were asked. Pass `response_text` on every commit — your final message to the user, in
full. AMFS sees only its own tools, so what you answered is invisible unless you hand
it over here, and a judge grading the answer reads nothing otherwise. Secrets are
scanned and redacted before storage.

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
