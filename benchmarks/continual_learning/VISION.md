# From memory with feedback to reinforcement learning for agents

Grid v2 established the first-order result: when outcomes flow back into memory, an agent
recovers from a changed world (0.71 first-attempt success in the last block vs 0.47 for
pgvector, 0.20 for Mem0) and a fleet that shares one store recovers faster than any of its
members could alone (0.54 vs 0.33). It also showed the limits of the current loop, and those
limits are the roadmap. This document lists the improvements, in the order they pay off,
each with the grid v2 measurement that motivates it and where it lives in the codebase.

The framing that makes SenseLab different from every memory product: **a memory store is a
replay buffer; the outcome loop is the reward signal; retrieval is the policy's state; the
briefing is the value function.** I1-I9 in `IMPROVEMENTS.md` built the reward channel. What
follows turns the store into a learner.

## Where grid v2 says the loop still leaks

| observation (grid v2) | number | what it means |
|---|---|---|
| Pre-change learning ceiling below no-outcome SenseLab | 0.68 vs 0.77 at ep 15-19 | one failed attempt contests a correct rule; the agent then avoids it; the rule never gets re-validated. Evidence is a point label, not a posterior. |
| Stale-pick still 0.15 in the first post-change block | vs 0.04 by the last block | unlearning takes ~20 exposures because each entry must be disproved on its own. Nothing propagates the failure to the entries derived from it. |
| Support post-change near-impossible for every arm | no-memory success 0.03 | the new fix is one of 8 actions; every agent rediscovers it by trial. The store has 60 traces containing exactly which actions were tried and which worked, and no arm aggregates them. |
| Tokens per successful task higher than pgvector | 16.5k vs 13.1k | evidence payload and briefing are additive to the context, not a replacement for it. |
| Escalations still 27 per 100 post-change | wasted 7.1k tokens/task | the agent guesses twice before escalating even when evidence is flat. Abstention is a prompt instruction, not a policy. |
| Fleet-6 propagation works but each peer's first exposure is still 0.42 | vs 0.54 overall | a peer's discredit reaches the store, but the briefing only says "discredited"; it does not say what to do instead unless a `lesson-contrast` exists. |

## The improvements

### V1. Action priors from traces (the Q-table)

**What.** For a task signature (scenario/entity + the tool the agent is about to call), return
the empirical outcome of every action tried in similar traces: `resend_email 9 won / 1 lost,
update_payment_method 0 / 7, escalate_tier2 1 / 6`. This is the missing aggregation in the
support tail: the traces already contain it. Retrieval today returns *what someone wrote*;
this returns *what happened*.

**Mechanism.** `amfs_record_action` already stores `(tool, arguments, success)` per trace.
Add a rollup keyed by `(entity_path, tool_name, normalised arguments)` with Beta posteriors,
refreshed by the outcome trigger. Expose as `action_priors` in the `retrieve` payload and a
`Tried here before` section in the briefing. Recency-weight so a regime change shows as a
posterior that moved, not a lifetime average.

**Expected effect.** Support post-change from 0.25 first-attempt toward the pgvector-free
ceiling of ~0.6; the "new quirk" classes become one-shot after any agent in the fleet
succeeds once.

**Where.** OSS: `amfs_core/models.py` (ActionRecord already exists), new
`amfs_core/priors.py`, Postgres migration for `amfs_action_priors` + trigger hook in
`amfs_propagate_outcome`; SDK `retrieve(..., include_priors=True)`. Pro: `retrieval/engine.py`
strategy feature, gateway payload, dashboard "actions" panel.

### V2. Posterior evidence and Thompson-sampled exploration

**What.** Replace the four-state label (untested / validated / contested / discredited) with a
Beta(α=wins+1, β=losses+1) posterior surfaced as `p_success` and `n`, and let the briefing
recommend *exploration* when two candidate rules have overlapping posteriors. A rule that
went 5/6 and then failed once is 0.75 ± wide, not "contested".

**Why.** The pre-change fragility (0.68 vs 0.77) is the cost of a categorical label: one loss
flips the label and the agent obeys the label. With a posterior the agent sees 0.83 → 0.75
and keeps using it. The same posterior drives the regime-change case correctly: 5/5 then
0/3 gives a recency-weighted posterior that collapses fast.

**Where.** OSS `amfs_core/evidence.py` (`evidence_signal` → posterior; keep the labels as
derived views for the dashboard), `MemoryHit.render` in the benchmark, briefing compiler in
`cortex`. Pro: `LLMStrategy` evidence prompt, ranker features.

### V3. Causal propagation of failure through the entry graph

**What.** When a validated entry is discredited, propagate a *provisional* demotion to every
entry that cites it (`pattern_refs`), that was distilled from it, or that the same agent
wrote in the same trace, and to entries whose claim shares the entry's action for the same
subject. Mark them `suspect` until re-validated.

**Why.** Unlearning took ~20 exposures because each of the 5-7 restatements of a stale rule
(agent reflections, lesson notes, the seed playbook) had to fail on its own. In the mechanism
table: 480 discredited versions to remove ~40 distinct stale claims.

**Where.** OSS: trigger step in migration `009`, `pattern_refs` graph already stored;
`amfs_graph_neighbors` for the query. Pro: `cortex-pro` digest already computes clusters —
reuse for "same subject, same action" grouping.

### V4. Regime-shift as an event, not a section

**What.** When a long-validated rule fails k=2 times in a row, write a system entry
`regime-shift-<entity>-<date>` ("`fix X` for `class Y` stopped working on <date>; last 2
attempts failed; alternatives not yet validated"), push it to the top of every agent's
briefing for that entity, and open an *exploration budget*: the next agents to see the class
are told to try a different action and report. The fleet then discovers the replacement in
parallel instead of serially.

**Why.** Fleet propagation of the *failure* is already what makes fleet-6 win (0.54 vs 0.33).
Propagating the *search* is the next step: six agents exploring six actions find the new fix
in one round.

**Where.** OSS: briefing compiler (`cortex`), `regime_shift` section already exists — add the
event entry and the `explore` hint to the payload. Pro: room discussion post for the entity
(rooms are the natural channel), dashboard alert (`amfs_alerts`).

### V5. Conditions, not just actions: distilled rules with scope

**What.** Offline consolidation that reads fail→succeed pairs across traces and distills rules
with explicit *conditions*: "if `issue=card-declined` and `after 2026-09` → `resend_email`;
before → `escalate_tier2`". The condition is what lets the rule survive the next change:
the store keeps both regimes with their validity windows instead of overwriting.

**Why.** Today `lesson-contrast-*` says what failed and what worked for one task. Generalising
to a conditional rule is what turns 20 traces into one procedure a newcomer can follow.
This is also the labeled-data product: (state, action, outcome, condition) tuples.

**Where.** Pro `distiller` package (exists; add the contrastive+conditional prompt),
`managed-models` renderer version bump, `amfs_consolidate` proposals. OSS: nothing beyond the
`SYNTHETIC_KEY_PREFIXES` allow-list.

### V6. Calibrated abstention as a learned policy

**What.** Decide `escalate now` vs `attempt` from the posterior, not from a prompt sentence:
if the best action prior is < p_min with n ≥ n_min, or evidence for the top hit is
`suspect`/`discredited` and no alternative is ≥ p_min, the briefing says *escalate first*.
Measure with wasted tokens per task and escalations-after-3-attempts.

**Why.** 7.1k wasted tokens per changed-class task and 27 escalations per 100 that came after
two burned attempts. The information to escalate at attempt 1 was in the store.

**Where.** OSS: payload field `recommendation: {"act"|"explore"|"escalate", why}` computed
in the SDK from V1/V2; benchmark prompt reads it. Pro: policy thresholds per tenant, tuned
from traces by `eval`.

### V7. Token economy: evidence replaces context, not adds to it

**What.** Return the posterior and a one-line claim per hit; drop the body of hits below the
top 2 unless the agent asks (`expand`). Briefing as a *diff* since the agent's last session
on the entity. Target: SenseLab tokens per successful task ≤ pgvector's.

**Why.** 16.5k vs 13.1k tokens per success is the weakest table in the paper and the first
thing a platform builder will attack.

**Where.** OSS SDK `retrieve(compact=True)`, `briefing(since=...)`; cortex digest. Pro
gateway default for MCP.

### V8. Reward shaping beyond binary

**What.** Use severity, attempts-to-success, tokens and wall time as graded reward, and let
`amfs_judge_trace` verdicts (already produced by the eval package) feed the same evidence
model as explicit outcomes. A success after three attempts should not reinforce as much as a
first-attempt success; a judge verdict of "correct but slow" is a weak positive.

**Where.** OSS `evidence.py` accepts a `weight` per outcome (`apply_outcome(..., weight)`);
SDK `commit_outcome(reward=...)`. Pro `eval` writes weighted outcomes from verdicts.

### V9. Managed models close the loop on the policy itself

**What.** With V1-V8 the store emits (state = briefing + hits + priors, action, reward)
tuples with correct within-trace credit. Train per-tenant: (a) the retrieval reranker on
"was this hit cited by a success" (exists), (b) an action-prior model that generalises across
entities with no traces yet, (c) DPO on fail→succeed pairs for the agent model where the
customer runs their own. This is the part no memory vendor can copy: they have no reward.

**Where.** Pro `managed-models` (contract already updated), `ml`, `eval`. Renderer v3 to
include priors and recommendation.

### V10. Fleet learning as a first-class object

**What.** Make transfer measurable and visible: every hit carries `author_agent`, every
briefing says how many peers validated a rule, and the dashboard shows propagation latency
(episodes between one agent's discredit and the last peer's first non-stale pick). The
pioneer / newcomer protocols added to the benchmark measure exactly this.

**Where.** OSS `MemoryHit`/payload `provenance.agent_id` (already in the entry; expose in
retrieve), `amfs_cross_agent_reads`. Pro dashboard fleet view, rooms briefing.

## Order of work and what each buys

| step | effort | expected movement (grid v2 metric) |
|---|---|---|
| V2 posterior evidence | S | pre-change ceiling 0.68 → ~0.75; no loss post-change |
| V7 token economy | S | tokens/success 16.5k → ≤ 13k |
| V1 action priors | M | support post-change 0.25 → 0.5+; block 20-29 stale-pick 0.15 → < 0.08 |
| V6 abstention policy | S (after V1/V2) | wasted tokens 7.1k → < 4k; escalations after 3 attempts → near 0 |
| V3 causal propagation | M | exposures-to-recover 9.6 → ~4 |
| V4 regime-shift event + parallel exploration | M | fleet block 20-29 → above single-agent block 40-49 |
| V8 reward shaping | S | fewer false validations of slow successes |
| V5 conditional distillation | L | newcomer first-exposure ≈ veteran late-block |
| V10 fleet provenance | S | measurement, not performance |
| V9 managed models | L | generalisation to entities with no traces |

## Why this is a moat

Mem0, Zep and a pgvector build each answer "what did we write down?". Grid v2 shows the
question that matters after the world changes is "what happened when we acted on it?", and
only a system that observes actions and outcomes can answer it. V1-V9 compound: priors need
outcomes, abstention needs priors, distillation needs credit assignment, managed models need
all of it. A vendor without the reward channel cannot bolt any of it on, and a customer who
builds it in a weekend has built the DIY+outcomes arm — which lost to SenseLab by +0.10
post-change and +0.18 late, on a benchmark we designed to be fair to it.
