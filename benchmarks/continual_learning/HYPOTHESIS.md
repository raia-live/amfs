# Preregistration: agents on a continual-learning layer vs pure memory vs pgvector

Written 2026-09-16, before any harness code or data. Numbers quoted in the paper come
from the metrics defined here; anything else is labelled exploratory.

## Question

When the same LLM agent repeats a class of task many times, does the memory layer behind
it make the agent *more reliable over time*, and at what cost in tokens and wall-clock?

We compare one continual-learning layer (SenseLab, production API) against two pure
memory products (Mem0, Zep), a plain pgvector RAG table, a "do-it-yourself" pgvector
build with reflection notes and periodic LLM consolidation, and no memory at all.

## Mechanism taxonomy (what we believe distinguishes the systems)

| Layer | When knowledge is reconciled | Signal used | Attribution to the read set | Confidence that moves |
|---|---|---|---|---|
| pgvector | never | none | no | no |
| pgvector-diy | every 5 episodes, LLM rewrite | agent-written notes | no | no |
| Mem0 | at write time (ADD/UPDATE/DELETE) | new statements | no | no |
| Zep | at write time (temporal edge invalidation) | new statements | no | no |
| SenseLab | at outcome time (`commit_outcome`) | what actually happened | yes, exact read set | yes |

Our claim is not that SenseLab recalls better. Our earlier head-to-head found recall at
parity with a plain vector baseline. The claim is that **reconciling on outcomes, over
the exact set of memories the agent read, is what makes behaviour converge**, and that
without it behaviour stays unpredictable no matter how good retrieval is.

## Arms

`none`, `pgvector`, `pgvector-diy`, `mem0`, `zep`, `senselab`, `senselab-nofeedback`.

All arms: identical agent prompt, identical tool schema (`memory_search`, `memory_write`,
domain tools, `finish`), identical models, seeds and task order, fresh scope per
(arm, scenario, seed), the same post-episode reflection note. Hosted arms are polled for
ingestion readiness before the next episode; the wait is reported separately, never
hidden in latency.

`senselab` additionally follows its documented contract: `briefing` at episode start,
`retrieve(min_confidence=0.5)`, `record_action` for domain tool calls, and
`commit_outcome(task_input, response_text)` at episode end with the environment's verdict.
`senselab-nofeedback` is identical but never calls `commit_outcome`.

## Scenarios

Two tiers. **Core scenarios** model what teams actually deploy agents for in 2026 —
customer service is the most common production use case (LangChain State of Agent
Engineering 2026: 26.5%; Aldric mid-2026 survey: 41% of enterprises in production),
followed by software engineering (38-53%), data & analytics / text-to-SQL (29-34%),
sales & marketing personalization (22-41%), and bounded operations workflows. **Probe
scenarios** each isolate one mechanism (stale fact, tool drift, poisoned handoff, fleet
sharing, abstention). Every core scenario follows the same discipline: no episode ever
repeats; the domain has hidden quirks where the *generic* answer is wrong; failure
feedback is what the real system would say (a reopened ticket, a reviewer rejection, a
number that does not reconcile), never the answer key. Offline checks confirm an agent
that always applies the generic playbook succeeds on 0-45% of episodes, so there is room
to learn and nothing to look up.

### Core (real-world) — 6 seeds × {1 agent, 6-agent fleet} × 40 episodes, 300-entry store

- **support** — tier-1 customer-support resolution. 8 issue types with paraphrased
  customer messages; in 6 of 8 the Acme-correct resolution differs from the generic
  playbook (a "double charge" is a pending authorization: explain, don't refund; an iOS
  login loop is a stale keychain: clear app data, don't reset the password). The action
  vocabulary is generic and the account lookup is non-diagnostic, so the answer cannot
  be read off tool names or tool output — the first pilot showed both leaks and they were
  closed. Unwarranted refunds are critical failures with a dollar flag. Metrics: first-contact resolution,
  reopen rate, unwarranted refunds, escalations bounced by tier 2.
- **ci-fix** — coding agent keeping CI green. 8 failure classes under hidden repo
  conventions (integration tests are rerun, not fixed; migrations are regenerated, never
  hand-edited; a UI snapshot changing on a backend-only PR is a regression). Wrong
  shortcuts turn CI green and then get rejected in review. Metrics: first-attempt green +
  approved, reviewer rejections.
- **analytics** — text-to-SQL style reporting. Finance asks a new question each episode;
  the warehouse has quirks (amounts in cents, status spelled `complete`, test accounts and
  soft-deleted rows to exclude). The engine runs the agent's query literally; finance says
  only whether the number reconciles. Metrics: first-attempt reconciliation, relative
  error, queries per answer.
- **concierge** — personalization for one user. New request each episode (dinner, lunch,
  gift, hotel, flight) with four options and exactly one that satisfies the user's hidden
  constraints. The user's terse rejection reveals one constraint at a time. A seeded
  onboarding profile is partly wrong ("loves sushi", "budget flexible"), and the user
  turns vegetarian at episode 13. Metrics: first-proposal acceptance, stale-profile
  follows, post-drift recovery.
- **retention** — predicting user behaviour. For each churn-risk account pick the one
  action that retains without wasted spend, under a hidden uplift model per segment
  (enterprise needs ownership, not discounts; high-ticket accounts need a call; stable
  accounts would stay anyway, so any offer is waste). The seeded playbook says "discount
  everyone". Metrics: retained-without-waste rate, MRR churned, wasted spend.
- **order-ops** — ecommerce order exceptions (cancel, re-address, return, price match,
  expedite) under hidden fulfilment and carrier rules (packed orders are warehouse-locked;
  only UPS supports intercepts and redirects; electronics have a 14-day window). Wrong
  denials get overturned by a supervisor and count as critical. Metrics: first-attempt OMS
  acceptance, wrong denials.
- **diagnose** — see below (item 8): incident response under a hidden causal model.

### Probes (mechanism isolation) — 5 seeds × 20 episodes, clean store

1. **runbook** — two near-tie procedures seeded in memory, one causes an outage.
2. **drift-fact** — an operational fact changes at episode k.
3. **drift-tool** — a domain tool's parameter shape changes at episode k.
4. **triage** — support tickets under a policy whose exceptions are learnable only from outcomes.
5. **handoff** — agent A writes 8 findings (1 wrong); agent B, a new identity, acts on them.
6. **unknowns** — questions with no answer in memory mixed into runbook episodes.
7. **fleet** — 4 identities run runbook round-robin on one shared store.
8. **diagnose** (generalization, flagship) — every episode is a *new* incident that has
   never occurred before: a fresh service name, fresh metric values, one or two distractor
   symptoms.    Incidents are generated from a hidden causal model of 6 rules
   (symptom pattern -> root cause -> the one remediation that works, plus two plausible
   remediations that do not). In 4 of the 6 rules the remediation that works at Acme is
   *not* the textbook one (e.g. pool starvation is fixed by shedding load, not by adding
   connections), so pretrained priors fail on first contact and the pilot confirmed a
   no-memory frontier model solves the textbook version first time. Failure feedback is
   only "symptoms persist" — the agent cannot read the answer off the environment; it has
   to discover it, generalize it, and then trust it. The agent has diagnostic tools
   (`get_metrics`, `get_logs`, `get_recent_changes`) and one `apply_fix`. Nothing in
   memory ever contains the answer to the current incident; what memory can hold is
   *heuristics* the agent wrote after earlier incidents, some right and some wrong. Episodes 16-20 additionally change the
   surface vocabulary (metric names, service naming scheme) so lexical retrieval of past
   incidents cannot help. Metrics: first-attempt fix rate on unseen incidents, diagnostic
   tool calls before the correct fix, judge-graded quality of the incident explanation
   (0-3, blind), and the share of *wrong* heuristics in the retrieved context.
9. **sizing** (generalization, numeric) — capacity planning for a new workload each
   episode: given a traffic profile, choose replica count and connection-pool size. The
   environment uses a hidden formula with two regimes (read-heavy vs write-heavy) and
   returns only pass/fail with a direction hint. The agent must learn a rule of thumb, not
   a lookup. Metrics: absolute error vs the hidden optimum per episode, first-attempt pass
   rate, attempts to pass.

Probes 1-7 test whether an agent stops repeating a *specific* mistake. The core scenarios
and probes 8-9 test whether an agent gets *better at a class of problem* — the difference
between a memory that is retrieved and knowledge that is reinforced. Both are needed: a
layer that only passes the probes is a good cache; a layer that passes the core set is a
learning system.

Every scenario has a deterministic environment that adjudicates success. An LLM judge is
used only for free-text answers and is blind to the arm.

## Design constants

- Episodes per (arm, scenario, model, seed): **N = 40** for core scenarios; **N = 20** for probes.
- Seeds: **6** for core scenarios, **5** for probes, **3** for sensitivity sweeps.
- Agents: every core scenario runs in **two configurations** — one identity handling all 40
  episodes (`fleet = 1`), and a **fleet of 6 identities** handling episodes round-robin on one
  shared store (`fleet = 6`), so each agent sees only ~7 episodes directly and must learn the
  rest from what its peers left behind. Probes are single-agent (the `fleet` probe keeps 4).
- Store size: core scenarios start from a store pre-loaded with **300 realistic distractor
  entries** (plausible but irrelevant operational facts), so retrieval must discriminate
  against a production-sized store rather than a toy one. Probes start from a clean store.
- Per model this is 868 cells and 29,120 agent episodes; two models, 58,240 episodes.
- **Zep exception (decided before the main run, 2026-09-16).** Zep's graph ingestion ran at
  roughly 5-8 episodes/min for our account during pilots, so pre-loading 300 entries into
  every Zep cell (124 cells × 300 = 37k graph episodes per model) would take days of
  ingestion. Zep therefore runs the full grid with a **clean store** (`d = 0`) — an easier
  condition than the other arms face, so it can only flatter Zep — plus a `d = 300`
  sensitivity subset (`support`, single agent, 2 seeds) seeded with a one-hour ingestion
  deadline. Per-episode Zep waits are bounded at 60 s; writes still unprocessed at read
  time are counted per episode as `zep_lag_at_read` and reported, not hidden.
- **SenseLab pacing.** The production key is limited to 120 requests/min. Calls are paced
  client-side below that limit; pacing sleeps and 429 back-off are subtracted from the arm's
  memory latency and reported separately as `rate_limit_wait_ms`, because waiting on our
  own quota is a property of this benchmark, not of the product.
- **Mem0 feedback.** The Mem0 arm also calls Mem0's documented `feedback` API (POSITIVE /
  NEGATIVE) on every memory it retrieved, at outcome time, so Mem0 receives every outcome
  signal its product exposes. Mem0 does not document that feedback re-ranks retrieval; if it
  does, that only helps Mem0.
  (Amended 2026-09-16 before the main run. First plan: N = 20, 5 seeds, single agent, clean
  store, USD 600 ceiling. The owner removed the ceiling and asked for scale in agents and data
  so self-improvement can be measured credibly; the design above is the result.)
- Retry budget per episode: **R = 3** attempts; after R failures the episode is `escalated`.
- Models: OpenAI `gpt-5.5` and one Anthropic frontier model (ID fixed at preflight and
  recorded in `results/run_manifest.json`). Judge: `gpt-5.4`, temperature 0, single vote.
- Burn-in for flip-rate and variance statistics: episodes 1-5 excluded.
- Convergence: first index at which 5 consecutive episodes succeed on the first attempt.
- Reliability target for cost-to-reliability: 95% first-attempt success over a trailing window of 10.

## Hypotheses

- **H1 Convergence.** `senselab` converges in more seeds and in fewer episodes than every
  other arm, in every scenario except `unknowns` (where the comparison is hallucination rate).
- **H2 Unpredictability.** After burn-in, pure-memory arms (`pgvector`, `pgvector-diy`,
  `mem0`, `zep`) show higher cross-seed variance of success per episode and higher flip
  rate than `senselab`, and their variance does not shrink between episodes 6-10 and 16-20.
- **H3 Degradation.** On the 40-episode horizon with reflection notes enabled, success in
  `pgvector` and `mem0` is non-increasing or declines from episodes 21-40 vs 6-20;
  `senselab` is non-decreasing.
- **H4 Cost of not learning.** Tokens-to-success and time-to-success are lower for
  `senselab` from episode 6 onward, and cumulative tokens to reach the 95% reliability
  target are finite for `senselab` in more seeds than for any pure-memory arm.
- **H5 Ablation.** `senselab-nofeedback` is statistically indistinguishable from
  `pgvector` on H1-H4 metrics, so the delta is the outcome loop, not retrieval.
- **H6 Transfer.** In `handoff` and `fleet`, a new identity's episode-1 first-attempt
  success is higher on `senselab` than on every other arm. In the `fleet = 6`
  configuration of every core scenario, the learning curve (first-attempt success vs
  store-wide episode index) for `senselab` is not flatter than its `fleet = 1` curve, i.e.
  lessons transfer across identities; for pure-memory arms the fleet curve is flatter and
  inter-agent disagreement on same-class tasks stays higher.
- **H7 Generalization.** In every core scenario and `sizing`, where no episode repeats,
  `senselab` shows (a) a steeper rise in first-attempt success on unseen instances,
  (b) fewer attempts and diagnostic calls per episode by episodes 16-20, (c) higher
  judge-graded explanation quality where a judge applies, and (d) a lower share of wrong
  heuristics in the retrieved context, than every pure-memory arm. The vocabulary shift
  at `diagnose` episode 16 and the preference drift at `concierge` episode 13 do not
  reverse (a) for `senselab` but do for arms that improved through lexical recall.
- **H8 Seeded misinformation.** Where the seeded knowledge is partly wrong (`support`
  playbook, `retention` playbook, `concierge` profile, `ci-fix` snapshot tip, `analytics`
  status tip), `senselab` stops following the wrong entry within 2 exposures after it is
  first cited in a failure; pure-memory arms keep retrieving it at unchanged rank.

## Primary metrics (per arm x scenario x model x seed x episode)

`success_first_attempt`, `success_any`, `attempts`, `escalated`, `wrong_guidance`,
`stale`, `hallucinated`, `abstained_correctly`, `prompt_tokens`, `completion_tokens`,
`tokens_total` (whole attempt chain), `retrieved_bytes`, `contradictions_in_context`,
`wall_ms`, `memory_ops`, `ingest_wait_ms`, `cost_usd`, `trace_verifiable`.

## Aggregates the paper quotes

1. Learning curve with 95% bands; mixed-effects logistic regression of
   `success_first_attempt` on `episode x arm` with a random intercept per seed.
2. Convergence episode; fraction of seeds converged; reliability ceiling.
3. Cross-seed variance per episode; flip rate; contradiction density; fleet disagreement.
4. Tokens-to-success, time-to-success, wasted tokens, escalations per 100 episodes,
   cumulative tokens to 95% reliability.
5. Transfer: episode-1 success for a new identity.
6. Auditability: fraction of decisions with a verifiable read set + action + outcome.
7. Failure taxonomy counts: stale-follow, poisoned-follow, contradiction-paralysis,
   hallucination-on-miss, oscillation, tool-shape-miss.

## Sensitivity sweeps (runbook only; `senselab`, `pgvector`, `mem0`; 3 seeds)

Distractor entries in the store: 0 / 200 / 2000. Outcome label noise: 0% / 10%.
Feedback delay: 0 / 3 episodes.

## Recursive-learning protocols (added 2026-09-20; not in the default grid)

Two protocols share one new arm. Both need a dev Pro deployment (`AMFS_PRO_URL`) with the
repair agent enabled; the arm refuses any host that does not look like dev.

**Arms**

- `senselab-repair` — `senselab` plus the shipped repair loop, driven from the harness between
  episodes: every sealed trace is graded by one judge on the environment's verdict; a failing
  verdict proposes a fix; its Tier 1 replay runs inline; a passed fix ships to memory
  (`auto_after_replay`). The next episodes read the corrective entry or procedure.
- `senselab-compose` — `senselab-repair` with the composition prompt on the repair agent: read
  the procedures already on the entity and compose them when each covers part of the failure.
- `raw-traces` — the "model already knew" control: no memory system; the transcripts of every
  prior episode in the cell (task, tool calls, results, verdict) in context, newest last, under
  a token cap. Whatever it gets right, the base model got right from raw experience.

**Protocol 1 — repair vs demote (the narrow intervention claim).** `ci-fix` × {`senselab`,
`senselab-repair`, `raw-traces`}, fixed model, core seeds. `senselab` can only demote a stale
lesson through outcomes; `senselab-repair` can also write the corrective entry. Read:
success, `unnecessary_edits` (terminal edit calls that did not resolve the task — a change
the agent made to the world and then had to undo), attempts to first success, latency, cost
including Pro calls. Claim: repair beats demotion-only on success *and* unnecessary edits at
comparable cost. Falsified if the repaired arm is not better on both, or if `raw-traces`
matches it.

**Protocol 2 — composition (the experiment that can kill the idea).** `fleet-disjoint` ×
{`raw-traces`, `pgvector-diy`, `senselab`, `senselab-repair`, `senselab-compose`}, 6 seeds.
Three agents share one store under three fictional payment policies the base model cannot
know; in the discovery phase each agent sees only its own policy's cases and is told the
policy in full when it fails; in the composition phase the cases trigger two or all three
policies and failures name only an error code. One trap is deliberate: the naive
concatenation of two discoveries (`hold, rotate_key, retry`) fails with `KEY-FROZEN`. Read:
success on `tags.phase == "composition"` by rule set, `flags.trap_naive_concat`, attempts to
first success per rule set, whether the lesson that carried a composition was written by a
peer, cost. Go on the composition engine only if a SenseLab arm beats `raw-traces` on the
composition tasks at comparable cost; `pgvector-diy` is the dream-style (LLM consolidation,
no outcomes) baseline.

## What would falsify us

- If `senselab-nofeedback` matches `senselab`, the loop is not doing the work.
- If Mem0 or Zep converge as fast as `senselab` on `runbook`, write-time reconciliation
  is sufficient and the outcome loop is not the differentiator we claim.
- If `pgvector-diy` reaches the same reliability at lower total tokens, the weekend build
  is the right answer and we say so.
- If no arm improves on `diagnose`/`sizing` beyond `none`, the models are not forming
  usable heuristics from outcomes at all and the generalization claim is withdrawn.

## Stopping rules and budget

Preflight gate: on the production API, a near-tie pair must reorder after 12 outcomes
and `outcome_count` must move. If it does not, the study does not run.
Budget: no ceiling (amended 2026-09-16, before the main run; the original USD 600 ceiling
and its fallback rules are superseded). Spend is reported in full in the results.

## Known limits, stated up front

- SenseLab is our product. The harness, seeds and raw per-episode JSONL are published.
- Reinforcement in SenseLab ranking is a tie-breaker; the `min_confidence` gate is what
  removes discredited knowledge. Scenarios are built so competing memories start near-tied.
- Hosted arms include network latency from Tokyo; pgvector is local. Latency is reported
  with the measured network baseline alongside.
- Scenarios are synthetic and deterministic by construction, which is what makes
  outcomes adjudicable without a human.
