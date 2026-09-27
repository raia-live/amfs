# Preregistration: an evidence-gated repair loop with canary release, against outcome learning alone, a vector store and a frontier model alone

Written 2026-09-26, before any repair-loop arm code or data. Numbers quoted in the paper
come from the metrics defined here; anything else is labelled exploratory. The commit that
adds this file is tagged `prereg-rsi-v1`; the hash is quoted in the paper. Amendments are
dated entries at the end and never edit the text above them.

Companions, not repeated here: `HYPOTHESIS.md` (the fast loop: outcome learning vs pure
memory, grids v4 and v5, paper v7) and `student/HYPOTHESIS_DISTILL.md` (the slow loop:
Managed Models distilled from sealed traces). This study measures the third loop — the
**repair loop** — and the behaviour-judging layer that feeds it, on the production SaaS,
with the two earlier loops as controls.

## Question

When a deployed agent's world changes and its memory goes stale, does a loop that
*proposes* a correction from the evidence, *tests* it offline against the runs it claims to
repair, *canaries* it on a share of live sessions, *ships* it to the shared store and
*verifies* it afterwards recover faster and further than (a) outcome learning alone, which
can only demote what failed, (b) a vector store with reflection notes, and (c) the frontier
model with no memory — and does it do so without shipping wrong fixes, without damaging
what still worked, at a cost we can state, and in a way an operator can govern?

## What we already know, and what this does not re-test

- Outcome learning beats pure memory after a regime change (paper v7, `HYPOTHESIS.md` H1–H8;
  headline +0.18 first-attempt success, loop on vs off, episodes 40–59). Not re-run at scale;
  `senselab` and `pgvector` are re-run here as controls under the new protocol only.
- Distilled students match or beat their teachers on the same tasks at a fraction of the
  cost (`HYPOTHESIS_DISTILL.md`). Only the narrow question in E7 is new.
- The repair loop, the canary and behaviour judging are shipped in the product and have
  **no benchmark result**. That is the gap.

## Mechanism taxonomy

| Arm | What can change the store | Signal | Gate before a change reaches live traffic | Revert path |
|---|---|---|---|---|
| `none` | nothing (no store) | in-episode feedback | none | none |
| `context-window` | nothing; prior notes and outcomes are in the prompt | the transcript | none | none |
| `pgvector` (+`rerank`) | agent-written reflection notes | agent's own notes | none | none |
| `pgvector-diy+outcomes` | notes, periodic LLM consolidation, win/loss counter | outcomes, marginal per entry | none | none |
| `senselab` | outcome evidence on the exact read set; agent writes | `commit_outcome`, `record_attempt` | `min_confidence` gate, avoid list | confidence decay |
| `senselab-rsi` | all of the above **plus** a repair agent that writes a corrective entry or procedure on a memory branch | attribution over verdicts and outcomes | Tier 1 stand-in replay, then a live canary with a significance test, then a policy | merge is reversible: `rollback` restores the pre-fix versions |
| `senselab-oracle-fix` | a scripted maintainer writes the correct entry `lag` episodes after each change | the answer key | none (it is the upper bound) | none |

The claim is not that the repair agent is clever. The claim is that **a correction is only
allowed to change behaviour after it has been tested on the runs it claims to repair and
has beaten the control arm on live traffic**, and that this discipline recovers faster
than demotion alone while shipping fewer wrong fixes than an ungated repair would.

## The engine under test (constants fixed before the run)

Everything below is production code on the SaaS (`amfs-login.sense-lab.ai`, http-server
version recorded in `manifest.json` at run time; 0.3.49 on the day of writing). No constant
is tuned for this study; each is quoted from the deployed source and reported in the paper.

- **Attribution** (`amfs_eval/attribution.py`): a memory entry is a candidate cause when its
  lift on failing runs is ≥ 2.0 with support ≥ 3 runs.
- **Lever** (`repair/agent.py`): `knowledge` (a fact is wrong), `procedure` (a conditional
  rule with conditions, steps and a verify), `code` (the agent's software; handed off, never
  auto-shipped), `unknown`. The lever classifier's choice is a measured quantity here
  (below), because the benchmark knows the true cause of every change.
- **Manifest** (`models_fix.RepairManifest`): every proposal predicts the runs it repairs
  and the runs it must not break; `expected_prevention = 0.5` by default.
- **Auto-propose**: on by default (`AMFS_EVAL_AUTO_PROPOSE` unset; only `off` disables). Two
  producers: an incident opening on a behaviour, and a regime shift (an entry validated ≥ 3
  times that now fails).
- **Incidents** (`incidents.py`): open when a behaviour's failing verdicts today reach
  `max(3, 2.0 × 14-day baseline mean)`; a fresh agent has an empty baseline, so the first
  incident opens at the third failing verdict on one behaviour cluster.
- **Behaviour discovery** (`discovery.py`): cosine ≥ 0.82 to an existing centroid; a
  candidate cluster is promoted at 3 members.
- **Judges**: production two-tier grading, `gpt-5-mini` prefilter → `gpt-5` on suspected
  failures (`GET /api/v1/eval/readiness` is recorded in the manifest). Rubrics are frozen
  before any run and published. Evidence must quote the trace.
- **Tier 1 replay** (`repair/replay.py`): a stand-in model re-runs each predicted case with
  the fix's branch as its memory, tool calls answered from the recorded run; `fidelity` =
  share of steps answered from the record; a test with fidelity below the agent's
  `replay_fidelity_floor` (0.5 default) does not count as a pass.
- **Tier 2 replay** (`repair/webhook.py`): HMAC-SHA256 signed cases delivered to the
  customer's agent; graded from the traces the agent seals on the fix's branch. Exercised in
  E2b with the OSS `ReplayReceiver` as the customer.
- **Canary** (`mcp_gateway/canary.py`, `repair/canary.py`): `bucket = int(sha256(f"{fix_id}:{session_key}")[:4], 16) % 100`,
  in canary iff `bucket < pct`; reads come from branch `repair/<fix_id[:8]>`, writes always
  go to `main`; the trace carries `attributes.memory_branch`. `decide()`: both arms must
  reach `min_outcomes`; the canary's success rate must not be below the control's (success
  guard); then a two-proportion z-test on the recurrence of the judged failure, `p < 0.05`
  one-sided → `passed`; a significant result the wrong way → `failed`; timeout →
  `inconclusive`. One canary per `(account, agent_id)` at a time.
- **Policies** (`RepairPolicy`): `review` (a person ships), `auto_after_replay`,
  `auto_after_canary`. The **feedback contract** (`repair/contract.py`) holds a passed fix
  at `policy_held` unless ≥ 0.8 of recent traces carry `model` and `agent_version`
  (`auto_after_replay`) and ≥ 0.8 of recent outcomes carry `verified_by` (`auto_after_canary`),
  judged over ≥ 10 traces in 14 days.
- **Verify** (`prevention.py`): prevention rate measured over a 7-day window after shipping;
  `verified` if ≥ `expected_prevention` and significant, else `regressed` → auto-rollback
  under automatic policies.
- **Repair-prompt A/B** (`repair/prompts.py`): a candidate repair prompt is served to a
  share of proposals; `repairer_stats` counts proven fixes per prompt. This is the
  "improver improving itself" test (H11).

## Environment and how the harness drives the loop

- **Production SaaS**, dedicated Enterprise account `clbench-rsi` (created 2026-09-26; no
  other agents write to it). Eval judge budget cap set to USD 3,000/month; Managed Models
  platform fee waived (operator path) for E7. Every SaaS call is paced by `loadguard.py`;
  rate-limit waits are reported as `rate_limit_wait_ms`, never folded into memory latency.
- **Harness-side canary routing.** The production router runs inside the MCP gateway; the
  harness talks to the HTTP API through the OSS `AgentMemory` + `HttpAdapter`, so it applies
  the same assignment itself: it imports `canary_bucket` (parity test in CI), sets
  `memory._branch` to the live canary's branch for the assigned share of sessions — exactly
  what the gateway's `set_branch()` does — forces writes to `main`, and stamps
  `memory_branch` on the sealed trace. One session per episode; assignment per session.
- **Loop advancement** through the account's own API, on a **preregistered look schedule**:
  after every episode the arm lists open fixes for its agent and, for each running canary,
  calls `POST /fixes/{id}/canary/evaluate` once every `k = 5` graded sessions of that agent
  (the hourly production sweep is the same code path; the schedule is faster and is what
  the A/A probe measures). Tier 1 tests are queued by the product; the harness polls
  `GET /fixes/{id}/tests`. Nothing that decides — propose, gate, `decide()`, ship,
  rollback — runs in the harness.
- **Provenance stamped on every session**: `model`, `agent_version` (harness git sha),
  `runtime`, and `verified_by = "benchmark-oracle"` when the outcome is oracle-graded; in the
  self-report condition (E3) `verified_by` is omitted on purpose.
- **Fleet = replicas.** In RSI cells a fleet is `n` concurrent sessions of **one**
  `agent_id`, because fixes, incidents, judges and the canary slot are scoped per agent.
  The v7 fleet of distinct identities is kept as the `handoff` probe.
- **Time.** Calendar-bound steps (incident resolution after 3 quiet days; prevention
  verification after 7 days) are reported post hoc when they close during the multi-week
  grid; the harness's own recurrence measure is primary.

## Arms

All arms: identical agent prompt, tool schema (`memory_search`, `memory_write`, domain tools,
`finish`), models, seeds, task order, retry budget `R = 3`, post-episode reflection note,
fresh scope per (arm, scenario, seed, model), 300 distractor entries, `opaque_ids = True`
(ticket/order/incident identifiers are random, so nothing can be read off an id).

- `none` — Claude Opus 5 or GPT-5.5 alone.
- `context-window` — `none` plus every prior reflection note and outcome of the cell in the
  prompt, most recent first, truncated at 24k tokens.
- `pgvector` — bge-small-en-v1.5 cosine top-5 over notes; `pgvector+rerank` adds the same
  cross-encoder SenseLab serves (`Xenova/ms-marco-MiniLM-L-6-v2`).
- `pgvector-diy+outcomes` — notes, LLM consolidation every 5 episodes, per-entry win/loss
  counter shown to the agent.
- `senselab` — the documented contract (`briefing`, `retrieve`, `record_action`,
  `record_attempt`, `commit_outcome`), repair loop **off** (policy `review`, proposals
  never approved; their count is reported as review load).
- `senselab-rsi` — `senselab` plus repair policy `auto_after_canary`.
  - `senselab-rsi-nocanary` — `auto_after_replay`.
  - `senselab-rsi-noreplay` — replay test skipped: canary starts on `proposed` (allowed by
    the product for a person; here scripted), then ships.
  - `senselab-rsi-review` — proposals held at `review`; measures what a human would read.
  - `senselab-rsi-minimal` — the agent calls only `retrieve` and `commit_outcome`; no
    `record_attempt`, no `record_action`. The protocol-compliance question.
- `senselab-oracle-fix-{5,10,20}` — `senselab` plus a scripted maintainer that writes the
  correct entry for each changed class `lag` episodes after the change, at confidence 1.0.

Mem0 and Zep are not re-run (appendix: gridv4 data).

## Scenarios and change sets

Ten families from `config.Study.core_scenarios`: `support`, `ci-fix`, `analytics`,
`concierge`, `retention`, `order-ops`, `diagnose`, `policy-qa`, `accounts`, `dispatch`.
Six carried a regime change in v7; the change-set machinery (`Scenario.change_variant`) is
extended so that **every** family has two change sets:

- **Set A** (public): the v7 changes where they exist; new A sets for the families that had
  none, written before any RSI-arm run. Used for the pilot, for developing the analysis
  code, and for E1–E3 as reported.
- **Set B** (private): different classes moved, defined for all ten families before the
  pilot, kept in `scenarios/change_sets_b.py`; its SHA-256 is registered by a dated entry
  in the Amendments section before the first pilot cell runs, and the file is published
  with the paper. Run **once**, last, after the analysis code is tagged
  `analysis-freeze-rsi`. E1-B is the headline held-out number.

Held-out **families**: `diagnose`, `policy-qa`, `accounts`, `dispatch` are never used to
develop or tune the repair prompt, the rubrics or the arm; they are run in E1 only.

## Experiments

Sizes use `config.Study`: 60 episodes, change at 20, `change_focus = 0.5`, seeds
`(11, 23, 37, 41, 59, 73)`, models GPT-5.5 and Claude Opus 5 at `low` effort.

- **E1 Head-to-head under regime change.** 10 families × 2 models × 6 seeds × fleet {1, 6}
  × 60 episodes; arms `none`, `context-window`, `pgvector`, `pgvector+rerank`,
  `pgvector-diy+outcomes`, `senselab`, `senselab-rsi`, `senselab-oracle-fix-10`. Change
  set A, then B. Primary endpoint: first-attempt success on changed-class tasks in
  episodes 40–59, paired on (family, seed, model, fleet). Secondary: episodes 20–39,
  exposures-to-recover, stale-pick rate, escalations, critical failures, unchanged-class
  success, cost.
- **E2 Long-horizon fleet with canary (flagship).** 6 families (`support`, `ci-fix`,
  `retention`, `order-ops`, `dispatch`, `policy-qa`) × 2 models × 3 seeds × fleet 10 ×
  200 episodes, changes at 50 and 130 (set A). Arms: `none`, `context-window`, `pgvector`,
  `senselab`, `senselab-rsi`, `-nocanary`, `-noreplay`, `-review`, `oracle-fix-{5,10,20}`.
  Canary parameters `pct = 50`, `min_outcomes = 8`, `timeout = 60` episodes (mapped from
  the production defaults 10 / 20 / 14 days: the same floor in sessions per arm within the
  cell's horizon); a sensitivity row at `pct = 10`, `min_outcomes = 20` on `support`.
  Endpoints: fix lifecycle counts by terminal state, time-to-repair (episodes from change
  to first shipped fix that lowers recurrence), false-ship rate, regressions caught by the
  canary after a Tier 1 pass, gate confusion matrix (Tier 1 verdict vs later live
  outcome), prevention predicted vs realised, propagation (episodes from first discredit
  to the last replica's non-stale pick), oscillation (ship/rollback cycles per key), store
  growth and memory latency over 200 episodes.
  - **E2b** — `support` and `ci-fix`, GPT-5.5, 3 seeds, `senselab-rsi` with the Tier 2
    webhook configured and the harness serving the OSS `ReplayReceiver`.
- **E3 Behaviour judging without an oracle.** 6 families × 2 models × 3 seeds × 60
  episodes, `senselab-rsi` under three verdict sources: `oracle` (the environment's
  verdict, `verified_by` stamped), `self-report` (the agent's own claimed outcome, no
  `verified_by`), `judges` (no outcome sent by the harness; verdicts from the frozen rubric
  pack become outcomes through `verdict_outcomes.py`). Cross-family judge as the default
  (OpenAI judges grading a Claude agent and vice versa where the platform allows; otherwise
  the platform pair is disclosed). Endpoints: judge precision/recall/F1 vs oracle by
  severity, inter-judge agreement, "failures called success", cluster-to-taxonomy match
  (discovered behaviours vs the scenario's failure classes), incident-open latency,
  auto-propose precision, recovery under each source, and the share of fixes held by the
  feedback contract under `self-report`.
- **E4 Public track: τ²-bench.** `telecom` and `retail` (fallback `banking_knowledge`),
  Claude Opus 5 primary, GPT-5.5 second; each domain's task set streamed over `k = 3`
  interleaved passes with arms `none`, `pgvector`, `senselab`, `senselab-rsi`; a
  **policy amendment** (one rule in the domain policy inverted, preregistered per domain
  in `tau2/amendments.json`) applied at the start of pass 2. Endpoints: pass^1 and pass^k
  before and after the amendment, learning curve across passes, simulator-error rate,
  baseline pass^1 vs the published leaderboard as a contamination check. Not a leaderboard
  submission.
- **E5 Level-1 discipline.** Fixed repair budget per cell (≤ 12 proposals, product token
  budgets as shipped: 50k per repair, 60k per stand-in run); public/private split;
  held-out families; oracle-fix baselines at three lags; the four-condition table (fair
  human baseline, sustained multi-step trend across two changes, generalisation beyond the
  optimised measurement, fixed budget). Repair-prompt A/B: a candidate prompt promoted by
  `repairer_stats` on `support` set A; its proven-fix rate re-measured on `ci-fix` and
  `dispatch`.
- **E6 Cost, latency, auditability.** Every arm-side LLM call charged to the arm: agent,
  reflection, repair agent, stand-in replay, judges (from the account's eval budget
  ledger), canary overhead. USD per solved task; cost per solved task with escalations
  priced at USD 5, 20, 50; Pareto frontier (success vs spend) per arm; memory latency ms;
  share of decisions with a verifiable chain (read set + actions + outcome + fix events);
  review load per policy tier (proposals a person would read); rollbacks executed and
  entries restored; blast radius of each shipped fix.
- **E7 The slow loop on the repair record.** Exactly two Managed Models (`senselab-s1`):
  one trained on `senselab-rsi` traces from E1 `support` (including `origin = repair`
  trajectories and repair-bridge pinned cases), one on `senselab` traces from the same
  cells; evaluated in decision mode on held-out seeds `(101, 113, 127)`. Endpoints: failure
  reproduction on pinned cases, post-change first-attempt success, cost per solved task.
- **Probes** (`support`, GPT-5.5, 3 seeds unless stated).
  - (a) **A/A null canary**: a fix whose branch is byte-identical to `main`; 50 canaries
    under the E2 look schedule → empirical false-pass rate vs nominal 0.05, and the same
    with one look at the floor only.
  - (b) **Adversarial**: a plausible wrong fix injected as a proposal, and a poisoned entry
    written by a peer → does Tier 1 + canary reject it; how many sessions read it first.
  - (c) **Drift and revert**: gradual change (classes flip one per 5 episodes) and a revert
    at episode 40 → does the loop unlearn its own repair, and in how many episodes.
  - (d) **Noise and delay**: 10 % outcome label noise; feedback delayed 3 episodes → wrong
    fixes shipped, prevention realised.
  - (e) **Reward-hacking check**: every shipped fix is tested for lowering recurrence while
    raising escalations or unchanged-class failures; flagged fixes counted.
  - (f) **Distractors**: 0 / 300 / 2,000 entries on `senselab-rsi`.
  - (g) **Lever attribution**: for every proposal, the lever chosen vs the true cause of
    the change (knowledge vs procedure, known to the benchmark) → confusion matrix.
  - (h) **Manifest calibration**: predicted repaired/regressed runs vs realised, per fix →
    Brier score of the repair agent's predictions.

## Metric glossary

- **changed class**: a task class whose correct action differs after the change.
- **first-attempt success**: the first `finish` is adjudicated correct.
- **stale pick**: the agent's first action is the pre-change correct action on a
  changed-class task after the change.
- **exposures-to-recover**: changed-class exposures after the change until 3 consecutive
  first-attempt successes on that class.
- **recurrence**: share of sessions in which the judged failure (or oracle failure class)
  recurs; the canary's test statistic.
- **time-to-repair**: episodes from the change to the first shipped fix whose class
  recurrence in the following 10 exposures is below half its pre-fix rate.
- **false ship**: a shipped fix whose realised prevention is ≤ 0, or which is later
  `regressed` or rolled back.
- **prevention rate**: 1 − (recurrence after ship / recurrence before ship), as
  `prevention.py` measures it.
- **propagation latency**: episodes from the first discredit of a stale entry to the last
  replica's first non-stale pick on that class.
- **review load**: proposals a person would have had to read under `review`.
- **verifiable chain**: a decision whose trace has the read set, recorded actions, the
  outcome, and — if a fix touched what it read — the fix events.

## Hypotheses and what falsifies them

- **H1 Repair beats demotion.** Post-change changed-class first-attempt success (40–59) is
  higher for `senselab-rsi` than `senselab` by ≥ 0.10 (paired bootstrap 95 % CI excludes
  0), and `senselab` > `pgvector` > `none` as in v7. Falsified if the CI includes 0 on set
  B, or if the ordering fails on the held-out families.
- **H2 No collateral damage.** Unchanged-class success for `senselab-rsi` is non-inferior
  to `senselab`, margin 0.02 (one-sided 97.5 % CI). Falsified otherwise.
- **H3 The replay gate earns its place.** `-noreplay` ships more fixes that later regress
  or are rolled back than `senselab-rsi` (rate ratio > 1, CI excludes 1).
- **H4 The canary earns its place.** `-nocanary` lets more regressions reach live sessions
  (sessions exposed to a later-regressed fix) than `senselab-rsi`; A/A false-pass rate
  under the look schedule ≤ 0.05, or the inflation is reported with the one-look rate.
- **H5 The procedure lever.** On `retention` and `order-ops`, `senselab-rsi` recovers
  post-change where `senselab` did not in v7 (`senselab-induce` flat): changed-class
  success 40–59 ≥ 0.6, and the fixes that did it carry lever `procedure`.
- **H6 Time-to-repair vs a human.** Median time-to-repair of `senselab-rsi` ≤ that of
  `oracle-fix-10`, and its 40–59 changed-class success ≥ 0.8 × `oracle-fix-5`'s, at the
  fixed budget.
- **H7 Without an oracle.** Under `judges`, `senselab-rsi` recovers ≥ 0.7 of its `oracle`
  recovery (difference vs `senselab`); under `self-report` the recovery is not
  distinguishable from 0 and ≥ 0.8 of passed fixes are held by the feedback contract;
  judge precision on critical classes ≥ 0.8.
- **H8 Fleet inheritance.** In E2, first-exposure success on a changed class for replicas
  that never saw the failure is higher for `senselab-rsi` than `senselab`; propagation
  latency ≤ one canary cycle (`min_outcomes / (pct/100)` sessions) after ship.
- **H9 Public benchmark.** On τ²-bench after the amendment, pass^1 ordering
  `senselab-rsi` ≥ `senselab` > `pgvector` ≥ `none` in both domains; before the amendment
  the arms are within 0.03 of each other (memory should not matter yet).
- **H10 Cost.** Cost per solved task including escalations at USD 20 is lower for
  `senselab-rsi` than `pgvector`; raw token cost per episode is reported either way (v7
  found `senselab` more expensive per episode).
- **H11 Ignition test.** The promoted repair prompt has a higher proven-fix rate than the
  default on families it was not promoted on. Reported pass or fail; no Level-2 claim
  either way.
- **H12 Better training set.** The RSI-trained student reproduces fewer pinned failures
  than the outcome-only student (paired on case) and is not worse on post-change
  first-attempt success.
- **H13 Lever and manifest are informative.** Lever agreement with the true cause ≥ 0.7
  (probe g); manifest Brier score < 0.25 and better than the constant 0.5 predictor
  (probe h).

## Statistics

Cell-level bootstrap (2,000 resamples) for every rate; paired deltas on (family, seed,
model, fleet); mixed-effects logistic regression of first-attempt success on
`episode × arm` with a random intercept per cell for the learning curves; non-inferiority
test for H2; rate ratios with CIs for H3/H4; Holm correction across the secondary
endpoints within each hypothesis; all preregistered hypotheses reported, including nulls;
exploratory analyses labelled. Power: from gridv4's cross-cell variance of the 40–59
changed-class rate (sd ≈ 0.16 per cell), 6 seeds × 2 models × 2 fleets per family give
80 % power for a 0.10 paired difference at α = 0.05; E2's 3 seeds are powered for the
lifecycle counts, not per-family rates, and E2 rates are reported pooled.

Infra errors: the purge rule of `HYPOTHESIS.md` (amendment 2026-09-19) applies unchanged —
a cell with any memory-API client failure is dropped and re-run in a fresh scope; LLM
provider incidents stay and are excluded from rates.

Blinding: the analysis code for E1–E3 is developed on change set A and tagged
`analysis-freeze-rsi` before any set-B cell runs; set B is run once. Threshold constants of
the engine are not changed after the pilot amendment.

## Cost accounting

Every LLM call made because an arm exists is charged to that arm: agent turns, reflection,
consolidation (`diy`), repair agent, stand-in replay, judges (read from the account's eval
budget ledger and attributed by agent), repair-prompt candidates. SaaS request costs are
reported as requests per episode. Nothing is charged to `none` that it does not make.

## What would make us claim less

- If `senselab-rsi` ≈ `senselab` on set B, the loop is not repairing; the paper reports
  the loop as a governance layer only.
- If `-noreplay` or `-nocanary` match `senselab-rsi` on false ships and regressions, the
  gates are not doing measurable work at this scale, and we say so.
- If the A/A false-pass rate exceeds 0.10 under the look schedule, the canary's
  significance test needs alpha spending; reported as a product defect, with the one-look
  rate beside it.
- If judges recover < 0.5 of oracle recovery, the engine depends on an oracle it will not
  have in production; C6 is withdrawn.
- If any Weco condition fails in E5's table, no Level-1-style claim is made.

## Amendments

Dated entries only, appended below this line. Nothing above is edited.

### 2026-09-26 (a) — routing is the product's, not the harness's

Written before any pilot cell. Rebasing the harness onto the OSS main that the SaaS runs
(http-server ≥ 0.3.45, PR #432 "canary routing hooks for any client") showed that the hosted
API itself routes any client that sends the SDK's identity headers: the bound `HttpAdapter`
carries `X-AMFS-Agent-Id` and `X-AMFS-Session` on every request; `amfs_tenant.http_deps.
route_canary` applies `amfs_tenant.canary_routing.decide` per request; reads of a canary-arm
session come from the fix's branch; the seal stamps `canary_fix_id`, `canary_arm` and
`memory_branch` server-side and strips a client's own claims. The "harness-side canary
routing" paragraph in *Environment* is therefore withdrawn: the harness routes nothing and
reproduces no hash. It records the `X-AMFS-Canary` / `X-AMFS-Memory-Branch` response headers
per session (`arm_notes.canary`, `arm_notes.memory_branch`) and the trace attributes are the
tally's source of truth. Verified live on the benchmark account (session headers bound; trace
attributes `model`, `agent_version`, `runtime`, `verified_by`, `replica`, `entity_path`
present; `GET /agents/{id}/canary/assignment` answers `main` with no canary running).

### 2026-09-26 (b) — arm names, E3 conditions, fleet replicas, judge cadence

- `context-window` is the existing `raw-traces` arm (prior transcripts in context under a
  token cap, `CL_RAW_TRACES_TOKENS`, set to 24000 for this study); no second implementation.
- **E3 conditions**, made precise now that the SDK contract is known (`commit_outcome` takes
  `verified_by`; there is no outcome-less seal):
  - `oracle` (`senselab-rsi`): the environment's severity, attempt boundaries, `verified_by =
    "benchmark-oracle"`; the continuous judge on.
  - `self-report` (`senselab-rsi-selfreport`): identical traffic with `verified_by` omitted.
    Isolates the provenance contract: same information, unverified.
  - `judges` (`senselab-rsi-judges`): every outcome committed as `success` unless the agent
    escalated, no attempt boundaries, no `verified_by`; the continuous judge is the only
    failure signal and its disagreements reach the evidence through `verdict_outcomes`
    (weight ≤ 0.6). Isolates detection.
- **Fleet replicas** apply to every SenseLab-family arm in a fleet cell (`--fleet-replicas`,
  cell id suffix `-rep`), not only to the repair arms, so `senselab` and `senselab-rsi` are
  compared under the same identity structure. The v7 one-identity-per-member rows are not
  reused for RSI comparisons.
- **Judge cadence.** On the production worker a sealed trace is graded 60–90 s after the
  seal (measured: two traces, `fail` and `pass`, USD 0.021 and 0.017). An incident, and so a
  proposal, therefore trails the failing episode by one to two episodes at the harness's
  pace. Time-to-repair is reported in episodes *and* in wall-clock seconds, and the pilot
  fixes a pacing floor per agent if the loop's latency is dominated by grading rather than
  by evidence.
- **Look schedule** as implemented (`arms/senselab_rsi_arm.py`): every `k = 5` sealed
  sessions, `POST /fixes/{id}/canary/evaluate` for the running canary; a canary older than
  60 sessions is stopped by the harness (`CL_RSI_CANARY_TIMEOUT`) and counted as timed out,
  since the product's own timeout is in days; a freed slot is given to the oldest
  `test_passed` fix with a branch (`POST /fixes/{id}/canary`, what `_start_next_canary`
  does hourly); shipped fixes get `POST /fixes/{id}/recheck`. `-noreplay` is realised as
  policy `review` with the harness starting the canary as soon as a fix has a branch,
  whatever its Tier 1 verdict, and shipping (`approve-memory`) on a passed canary; the test
  still runs and is recorded, so what it would have blocked is measured, not assumed.
- **Set B** for `analytics`, `concierge`, `retention`, `order-ops`, `diagnose`, `policy-qa`,
  `accounts`, `dispatch` does not exist yet (only `support` and `ci-fix` carry one). They are
  written before the pilot's analysis freeze; the SHA-256 of `scenarios/` at that commit is
  registered here in a further entry. Until then E1-B is defined for `support` and `ci-fix`
  only.
- Oracle-fix entries (`Scenario.oracle_fix_entries`) exist for `support`, `ci-fix`,
  `dispatch`, `diagnose`, `policy-qa`; `retention` and `order-ops` receive theirs with
  their set B.

### 2026-09-26 (c) — change sets for all ten families, pilot 1 findings, repair agent

Written after pilot cell 1 (`rsi-pilot-1`: `support` × `senselab-rsi` × GPT-5.5 × seed 11,
60 episodes on the SaaS) and before pilot 2. Pilot 1 is a **development** run: it is not
counted in any reported table.

- **Change sets.** Every family now carries set A and set B, in each family's own module
  (`scenarios/*.py`, keyed by `change_variant`) rather than a separate `change_sets_b.py`;
  the registered artefact is therefore the whole directory. Committed at `a534a917`
  (`paper/rsi-engine`); `shasum -a 256 scenarios/*.py | shasum -a 256` =
  `7d887e0c6d6ec7949887c2f72bbeeeaea658112c7660b5deccc9ec28d489f946`.
  - New **A** sets (families that had no change in v7): `analytics` — revenue is *booked*
    (complete or pending), AOV excludes orders under $50, refund rate is refunds over booked
    orders; `concierge` — a dinner and lunch dietary rule and a flight-carrier rule.
  - **B** sets: `support`, `ci-fix` (v7's); `diagnose` — `capacity → scale_out`, `cache →
    restart_pods`; `dispatch` — `dependency-upgrade → code-agent`, `dashboard-change →
    code-agent`, `incident-postmortem → support-agent`; `policy-qa` — SLA credit 25 %, trial
    45 days, annual-cancellation notice 90 days, SSO on all plans; `accounts` — four
    accounts change on `format`, `cadence`, `owner`, `followup`, disjoint from the A set and
    the wrong-list; `retention` — ≥ 4 tickets in 30 days → `assign_csm`, declining team
    without integrations → `offer_discount`; `order-ops` — packed orders may still change
    address / upgrade shipping, final-sale returns authorised within 14 days, price-match
    accepted on same-SKU; `analytics` — order count includes cancelled, customers are
    distinct over every order; `concierge` — alcohol allowed, no electronics gifts, hotel
    gym requirement dropped, hotel budget $250.
  - Rule for B: it moves classes A leaves alone, to (class, fix) pairs that were never the
    correct answer under A, so an entry learned under A is wrong under B and vice versa.
    Validated (`/tmp/scen_validate_b.py`, not part of the release): ≥ 17 changed-class
    post-change episodes per 60-episode cell in every family and variant; the smallest
    numeric shift in `analytics` is 15.7 % (A, refund rate) so the ±1 % stale-pick band
    never straddles both truths; legacy `change_at = 0` runs are unchanged.
  - Oracle-fix entries now exist for all ten families and both variants (2–4 per family);
    `retention`/`order-ops`/`accounts`/`analytics`/`concierge` gained theirs here.
  - **Opaque ids** (`--opaque-ids`) now cover `diagnose` (`INC-`), `dispatch` (`DSP-`),
    `policy-qa` (`POL-`) and `analytics` (`FIN-`) as well as the families that had them.
- **Two harness defects found by pilot 1, fixed before pilot 2.**
  - *Agent-id isolation.* The runner suffixes each cell's `agent_base` (`support-agent-388584`)
    so cells never share a platform agent; the arm's `saas_agent_id` stripped any trailing
    `-<digits>` and so registered the judge, settings and fixes under `support-agent`, shared
    across cells. Now only `<agent_base>-<replica>` is folded, relative to the scenario's
    suffixed base. Cross-cell contamination in pilot 1 was limited to that one cell (nothing
    else ran on the account); pilot 1 is discarded regardless.
  - *The judge could not see memory.* The rubric asked only for the tool result or summary
    line the verdict rests on, so every failing verdict cited a `tool` span; the platform's
    lever classifier (`amfs_eval/fixes.py classify_lever`) counts a cited error/tool span as
    code evidence, and both pilot-1 proposals came out `lever = code`, `source = template`,
    with no memory branch — nothing to replay or canary. The rubric now requires a
    `followed-memory` criterion on every verdict whose `evidence_quote` is the verbatim text
    of the memory entry that recommended the failing action (or the agent's own reasoning
    when none did); a verbatim entry quote resolves to an `entry` evidence ref, which the
    classifier counts as knowledge. This is a **harness** change to the judge rubric only;
    the classifier is untouched and its share of `code` / `knowledge` / `unknown` levers per
    arm is reported as a measured quantity (a reviewer should read it as "how often the
    engine looked in the right place", not as a tuned constant).
- **Repair agent.** The prereg's `senselab-rsi` definition includes a repair agent that
  drafts the corrective entry; on the production API this is an opt-in
  (`AMFS_EVAL_REPAIR_AGENT`), process-wide, and was **off** during pilot 1 (proposals were
  templates). It was enabled on the production `amfs-api` service at 2026-09-26
  (revision `amfs-api-01929-qph`) for the duration of the study. The paper states this and
  the `senselab-rsi-minimal` ablation is the arm that measures the engine without it.
- **Pacing.** Pilot 1: 60 episodes, 52 successes, agent-side USD 0.02–0.07 per episode;
  incidents and proposals trailed the failing episode by one to two episodes, as (b)
  predicted. The pacing floor is set after pilot 2, once a fix has been observed to reach
  `test_passed → canary → shipped`.
- **Session-level `canary` / `memory_branch` notes** are absent from a row when no canary is
  live (the SaaS sends `X-AMFS-Canary` / `X-AMFS-Memory-Branch` only under a running
  canary); absence is the control state, not a recording failure.

### 2026-09-26 (d) — the two-change protocol (E2), and a second harness defect

Written during pilot 2, before any E1/E2 cell.

- **What "changes at 50 and 130" means.** At `change_at` the cell's primary change set comes
  into force; at `change_at2` the *other* set comes into force **on top of it** (the classes
  the first change moved stay moved; a disjoint set of classes moves as well). Implemented as
  `--change-at2` (`Scenario.change_at2`, cell id and scope suffix `-cc<ep>`): `variants(ep)`
  is `""` → `"A"` → `"AB"`, `active_changes(ep)` the union of the sets in force, and every
  task carries `change_active` (has this task's class moved by this episode), `change_stage`
  (1 or 2) and `post_change2`. Sets A and B move disjoint classes by construction (amendment
  (c)); the one interaction — `order-ops` price match on an equivalent SKU 8–14 days out,
  right only under both sets — is tagged `change_stage = None` and is excluded from the
  changed-class endpoints. A repair for stage 1 therefore stays correct after stage 2, so E2
  measures whether the loop repairs a second time *without undoing the first* (oscillation,
  regression) — the "sustained multi-step trend across two changes" E5 asks for. The
  oracle-fix arm writes one batch per change (`lag` episodes after each).
- **Consequence for the run order.** The second change uses set B, so **E2 with both changes
  is run after `analysis-freeze-rsi`, together with E1-B**. The E2 analysis code is developed
  on a reduced single-change E2 run (`support`, `ci-fix`, GPT-5.5, 1 seed, fleet 10, 200
  episodes, change at 50 only), which is reported as a development run and not in any table.
- **Byte-identical single-change cells.** The protocol machinery replaced each family's
  `changes` property with `change_set(variant)`; prompts, seed entries and truths for every
  family × seed {11, 23} × `change_at` {0, 20} × variant {A, B} × 60 episodes are unchanged
  (checked with a snapshot before and after, `/tmp/scen_snapshot.py`, not part of the
  release). The `scenarios/` hash is now
  `531ee460a10c66c5bb76d6362eee7a27653cf965b4e145e365cf29449630ed4e` at `5b5915e0`; the
  change-set *definitions* are those hashed in (c).
- **Second harness defect (pilot 2, first attempt, discarded).** The runner's per-cell suffix
  was `int(time.time()) % 1e6`: two cells started by parallel workers in the same second
  shared it, and with it the hosted agent identity (`support-agent-390362` for both seeds) —
  a shared judge, repair settings, fix list and canary tally for `senselab-rsi`, and a shared
  agent brief for every SenseLab arm. Memory scopes carry the seed and were never shared.
  Fixed at `741fa668` (nine random digits, unique within the process). v7 rows are affected
  only where the first `--workers` cells of a run started in the same second; v7's endpoints
  are scope-based and do not depend on per-agent platform state, but the paper's
  reproducibility package notes it.

### 2026-09-26 (e) — pilots 2 and 3: the loop did not close; three engine defects and a capacity limit

Written after pilot cells 2 and 3 (`rsi-pilot-2`: `support` × `senselab-rsi` × GPT-5.5 ×
seeds 11, 23, rubric of (b); `rsi-pilot-3`: the same with the `followed-memory` rubric of
(c)), before any E1/E2 `senselab-rsi` cell. Both pilots are **development** runs and are not
counted in any reported table; they are kept as the "engine before" record for the
before/after comparison below.

**What happened.** 240 sessions, 28 fail verdicts, 6 behaviour incidents (`Escalation
ping-pong`, `Repeated unresolved ticket`, `Issue recurs after closure`, `Unnecessary
escalation loops`), 8 proposals. Six proposals were `lever = code`, `source = template`, no
branch. Two (`50738ba1`, `c98b5d12`) were `lever = procedure`, drafted by the repair agent on
a `repair/<id8>` branch, and both ended `test_failed` ("3 of 3 replayed run(s) still fail").
No fix reached a canary. The `followed-memory` criterion of (c) did what it was asked —
the judge quoted the entry verbatim with `span_id` null — and it still did not produce an
`entry` evidence ref.

**Three engine defects, each a reading inconsistent with the module's own documented
semantics.** Fixed in `amfs-internal` (branch `eval/worker-slot-filling`, one PR), deployed
to production before the first counted RSI cell. The pilot verdicts, fixes and tests remain
in the account as the record of the engine before the fix.

1. *Evidence resolution.* `amfs_eval/runner.py _resolve_evidence` matched a quote against
   sources in `collect_sources` order — spans, tool calls, then the entries read. The platform
   derives a `memory_read` span for every entry read whose output is the entry's value, so a
   verbatim quote of an entry always resolved to that span (`kind = span`) and never to the
   entry. Entry sources are now checked first (stable sort); a span quote that no entry
   contains resolves as before.
2. *Lever classifier.* `amfs_eval/fixes.py classify_lever` counted every cited span of kind
   `tool` as code evidence whatever its status, while a cited `tool_call` that *succeeded*
   was already read as "the sequence, not the tool". And the benchmark's terminal action is
   recorded with `record_action(success=False)` when the environment's verdict is negative —
   the SDK's documented won/lost semantics, which the action priors count — from which the
   platform derives a tool span with `status = error`; both the call and the derived span
   were read as broken tooling (`knowledge 0.3` from the entry overlap vs `code 0.6`). Now: a
   derived tool span is read as the call it stands for; an `ok` tool span is sequence
   evidence; a failed call carrying an `action_key` (a credited decision) is "the choice, not
   the tool" (`procedure + 0.2`, no code); a failed call without an `action_key` stays code.
3. *Tier 1 stand-in replay.* `amfs_eval/repair/replay.py ToolReplay.answer` fell back to
   matching on tool name alone, so when the repaired stand-in chose a *different* `resolve`
   it was handed the recorded output of the failed one and graded as still failing. The
   fallback now applies only to recorded steps that succeeded; a stand-in whose next step has
   no recorded answer and that never reached a final gets a case verdict `unreplayable`
   (ungraded), and the test is `inconclusive` — the verdict `verdict()` already defined for
   "could not be replayed". An inconclusive fix under `auto_after_canary` goes to the agent's
   own replay (Tier 2) where a webhook is set, and otherwise to a canary capped at
   `INCONCLUSIVE_CANARY_PCT_CAP = 10` percent.

**Consequence for the design.** Tier 1 cannot vouch for a fix that changes the terminal
action of a one-shot decision task — the record holds the outcome of the action the agent
took, not of the one the fix recommends. On this benchmark the loop therefore closes either
through the 10 % canary (needing about 80 post-change sessions for `min_outcomes = 8`, i.e.
fleet ≥ 6 in E1 and always in E2) or through Tier 2 (the harness serving the OSS
`ReplayReceiver`, E2b). The share of tests ending `passed` / `failed` / `inconclusive` per
family is a reported quantity; E2b is promoted from a side condition to the mechanism the
flagship analysis reports alongside the inconclusive-canary path.

**Capacity, not the algorithm.** The production eval worker graded in rounds of five, each
round lasting as long as its slowest job; a fail verdict escalates to the strong model
(`gpt-5`, avg 63 s, p90 111 s) so every round outlived the 45 s tick budget and throughput
was five verdicts a minute regardless of `AMFS_EVAL_WORKER_CONCURRENCY`. At pilot pace the
queue reached ~100 judge jobs and proposals trailed incidents by more than ten minutes.
Changes, all recorded as actions in AMFS: `AMFS_EVAL_WORKER_CONCURRENCY` 3 → 8 (revision
`amfs-api-01930-mf4`), `AMFS_EVAL_WORKER_BUDGET_SECONDS` 45 → 180, and in the same PR the
worker keeps its slots full (claims a job whenever one frees) and, opt-in
(`AMFS_EVAL_KICK_ON_SEAL=1`), starts a grading loop on whichever instance a seal lands on.
`deploy.yml` now sets concurrency 6 / budget 180 / kick-on-seal on for the study. None of
this touches what is graded or how.

**Pacing floor (fixes (b)/(c)).** The RSI arms take `CL_RSI_VERDICT_WAIT` (seconds): after
each seal the cell waits until the platform has graded that session, or the wait expires,
before the next episode. The wait converts grading lag into wall-clock — reported per
session as `verdict_wait_ms` / `verdict_lag_s` — so time-to-repair *in episodes* is a
property of the loop, not of the queue. Set to **240 s** for all counted RSI cells (pilot
judge latency: pass 6 s, fail 63 s avg, 231 s max). Cells without judges do not wait.

**Before/after.** Pilot 4 re-runs pilot 3's two cells against the fixed engine (fleet 6 so
the 10 % canary can reach its floor within 60 episodes). The prereg predicts: ≥ 1 proposal
per cell with `lever ∈ {knowledge, procedure}`, `source = repair_agent`, a branch, a test
ending `inconclusive` or `passed`, and a canary started. If no canary starts in pilot 4 the
E1 RSI arms are run with Tier 2 configured from the start rather than as E2b only.

### 2026-09-26 (f) — Tier 2 is the E1 mechanism; the fleet-6 arithmetic in (e) was wrong

Written before pilot 4, after the engine PR (`amfs-internal#779`, merged; prod revision
`amfs-api-01939-89r` carries worker concurrency 6, budget 180 s, kick-on-seal) and before
any counted RSI cell.

**Correction.** (e) said fleet 6 would let the 10 % inconclusive canary reach its floor
within 60 episodes. It would not: the harness's fleet is *round-robin over the same 60
episodes* (`Scenario.fleet_size`), so a fleet-6 cell seals 60 sessions, not 360. With 40
post-change sessions a canary capped at 10 % sees ~4 outcomes against `min_outcomes = 8`
whatever the fleet; the inconclusive-canary path cannot close inside an E1 cell. It can in
E2 (200 episodes, ~18 canary outcomes) and is kept there.

**Tier 2 in the harness (commit `50c9e550`).** The E1 and E2 `senselab-rsi*` cells run
with `CL_RSI_TIER2=1`: the runner process serves the OSS SDK's replay contract
(`amfs.replay`: signed webhook per case, 202 + background run, at-least-once dedup) behind
an ngrok tunnel, and every RSI cell registers that endpoint in its agent's repair settings
(`replay_wait_hours = 1`). When a fix's Tier 1 test is `inconclusive` (or nothing could be
replayed) the platform asks the agent to replay the failing cases; the cell re-runs the
episode each case came from — same task, same environment, same model — with memory
checked out on the fix's branch, and commits the outcome under `case_id` / `fix_id` /
`memory_branch` so the platform's judge grades it. The harness grades awaited replays at
each look (`POST /fixes/{id}/replay/grade`, what the hourly replay sweep does), so a pass
reaches the policy inside the cell: `test_passed` → canary at the arm's 50 % /
`min_outcomes = 8` → ship. This is the product's own Tier 2 exercised as a customer would
— the harness is the customer's runtime — and is disclosed as such; the ablations keep
their meaning (`-nocanary` ships on a passed Tier 1 *or* Tier 2 test; `-noreplay` skips
both). The prereg's "share of tests ending passed / failed / inconclusive" gains a
`tier2 passed / failed / expired` column, and time-to-repair counts the replay's own
sessions (ledgered apart, never as `sessions_sealed`).

**Smoke (not counted).** A scratch cell on the account replayed one episode on an existing
`repair/…` branch: the trace carried the grader's keys and the platform's ping through the
tunnel answered 200 with a valid signature. Reads on a branch owned by another agent are
refused (403) — in the real flow the replaying agent is the branch's owner.

**Pilot 4 (revised).** `support` × `senselab-rsi` × GPT-5.5 × seeds 11, 23, fleet 1,
`CL_RSI_TIER2=1`, `CL_RSI_VERDICT_WAIT=240`, `CL_SENSELAB_TIMEOUT=60` (the SDK's 30 s read
timeout dropped one smoke commit under the baseline run's load). Predicts, per cell: ≥ 1
proposal with `lever ∈ {knowledge, procedure}`, `source = repair_agent`, a branch; a Tier 1
test ending `inconclusive`; a Tier 2 replay requested and graded inside the cell; and a
canary started. Development run, not counted.

### 2026-09-26 (g) — headroom map, development seeds, and a declared target

**Why.** Bruno asked for the opportunities that would take SenseLab to ≥ 92 % to be mapped
so the product can be improved *while the study runs*. Improving the system under test
mid-study is legitimate only if the tuning and the measurement are kept apart; this
amendment writes down how.

**The map.** `HEADROOM_RSI.md` (committed with this amendment) analyses 9,480 SenseLab
episodes from the v7 grids and pilot 4. SenseLab is at 91.5 % (GPT-5.5) / 90.6 % (Claude)
over the six v7 families; `analytics`, `concierge`, `order-ops` are saturated; the gap is
`support` (82–84 %) and `ci-fix` (87–89 %), and within them it is entirely **post-change
recovery of the changed classes** (unchanged classes: 99–100 %). The mechanisms, each with
a measurement: the recommendation stays `act` on the turned action for two losses (`act`
on 400 / 720 support and 603 / 700 ci-fix post-change first attempts); 39 % / 29 % of
attempts re-try an action already failed since the change; 26 % of support episodes got
no recommendation; ci-fix follows the recommendation 41 % of the time; a stale entry is in
context for 40 % / 72 % of failures. Eight ranked changes with their predictions are
listed there.

**Development vs confirmatory.** Seeds **11 and 23** are development seeds: the pilots ran
on them and every change from the map is developed and checked on them, on `support`,
`ci-fix` and `retention` only. Seeds **37, 41, 59, 73** are confirmatory and are never run
with a SenseLab arm before `analysis-freeze-rsi`. E1–E3 SenseLab arms start after the
freeze. The primary analysis of every hypothesis uses the confirmatory seeds; 11 and 23 are
reported alongside, labelled development-exposed. Set B and the four held-out families
keep their roles from the prereg.

**Changes are logged, not silent.** Each engine or harness-agent change made from the map
is recorded here as its own dated amendment with (i) the mechanism, (ii) the pilot/dev
evidence that motivated it, (iii) the prediction it makes on the development seeds. No
change touches a baseline arm, a scenario, a change set, a rubric or a metric definition.

**A declared target, not a tuned one.** Added as a secondary, one-sided hypothesis:

- **H8 Late-window level.** On set A, confirmatory seeds, fleet 1, `senselab-rsi`
  success over episodes 20–59 across the ten E1 families is ≥ 0.92 (95 % Wilson lower
  bound ≥ 0.90). The engineering estimate in the map is ≈ 0.96 on the six v7 families if
  items 1–5 land. Falsified if the point estimate is < 0.92; the paper then reports the
  measured level and which of the map's predictions failed.

**Pilot 4 status at this amendment.** 65 episodes. The loop has produced its first
proposals under the fixed engine: two incidents ("3 runs today vs 0/day") → two fixes,
8–10 episodes after the change; one lever `knowledge` on branch `repair/fbb7c4cb` in Tier 1
testing, one lever `code`, no branch. Both `source = template`: the repair agent's 90 s
budget expired on its first gpt-5 call (two 45 s time-outs) — the (f) prediction
`source = repair_agent` is not met; the fix (`AMFS_EVAL_REPAIR_TIMEOUT_S`, default 300 s,
per-call cap 120 s, and the reason a template was kept written into the fix's rationale)
is a PR awaiting approval and is amendment-logged when it ships. Four `POST /traces` seals
on production stalled 37–224 s and were released at the same instant (05:01:17 UTC) — two
harness commits timed out client-side but the traces were sealed (no duplicates, one trace
per episode); the cause is being isolated with a lock sampler on the database.

### 2026-09-26 (h) — the harness adopts the product's situation contract; four engine fixes; repair-agent budget

**Mechanism.** Items 1–3 and 6 of the headroom map had one cause: the harness never declared
a `situation` on retrieve or commit and never asked for `abstain`, so the engine's per-class
record and everything computed from it (`turned_unsolved`, `_exact_first_win`,
`sweep_order`, the abstain modes) was inactive; the pooled record said `act` on the
class-agnostic top action and was silent on 17–26 % of searched attempts. The change is on
the caller's side and is the documented use of the product (`Run.begin(situation=...)`):

- harness `memory_search` gains a required `situation` label for arms whose task ends in an
  enum action; the label is folded, sticky per episode (refinable once the server reports it
  pools two classes), declared as the SDK's trace attribute, and passed with `entity_path`
  on `commit_outcome`; retrieves carry `abstain=True` and only the candidates not yet tried
  on this task; the footer contract (act on the footer unless the task rules it out; on a
  retry, the next untried in the footer's order; never repeat a failed action; reuse
  labels; refine a pooled label) is in every SenseLab arm's system prompt. Applied
  identically to `senselab`, `senselab-rsi` and every SenseLab ablation; ablations
  `senselab-nosituation` and `senselab-nocontract` isolate the two parts.
- engine (OSS PR #449, approval pending, deployed with the next amfs-internal deploy):
  labels compare as word sets without noise words; the sweep ranks a never-tried action at
  the uniform prior; a mixed record abstains out loud under `abstain=True`; the
  pooled-classes abstain names the declared label.
- repair loop (amfs-internal PR #780, approval pending): repair-agent wall-clock budget
  (300 s, 120 s per step, 10 steps) with a recorded stop reason; the template's rationale
  says why the agent's draft was not used; a derived error span among a retrying agent's
  same-tool calls reads as its lost credited call (the pilot-4 lever `code`).

**Evidence.** Grid v5c / pilot 4: `act` on 400/720 and 603/700 searched attempts, 39 % /
29 % re-tries of an already-failed action, 26 % of episodes with no recommendation,
changed-class post-change success ≈ 55 % (support) / 65 % (ci-fix). Live smoke on
`clbench-rsi` with the contract: `situation_record`, `situation_exact` rows, a one-loss
`explore` sweep ordered by `elsewhere`. Pilot 5a (first 34 ci-fix episodes, harness contract
on the deployed engine): every `abstain` on a label's first episode, every missing
recommendation on a retry over a mixed record (7/46), both audit classes pooled under one
label and detected as such by the engine.

**Predictions (development seeds 11/23, fleet 1, set A, `senselab` and `senselab-rsi`).**
(i) Recommendation present on ≥ 98 % of searched attempts once #449 is deployed (`rec_none`
< 2 %); (ii) re-try of an action that already failed on the same task < 5 % of attempts;
(iii) a changed class's first post-change loss on an unsolved task turns its record to
`explore` on the next episode of that class (recovery ≤ 3 episodes of the class for ≥ 80 %
of changed classes); (iv) changed-class post-change success ≥ 80 % in support and ≥ 85 % in
ci-fix; (v) cell success over episodes 20–59: support ≥ 0.86 with the deployed engine and
≥ 0.90 after #449, ci-fix ≥ 0.90; (vi) `senselab-rsi`: first proposal ≤ 3 episodes after
the change, `source = repair_agent` ≥ 80 %, lever ∈ {knowledge, procedure} after #780.
**Ceiling correction.** The map's `support ≈ 95 %` at fleet 1 was too high: with three
attempts, the sweep over ≈ 7 untried actions loses ≈ 2.3 episodes per changed class
uniformly and ≈ 1.2 in `elsewhere` order (two of three new support fixes are winners
elsewhere in set A), which bounds a support cell at ≈ 90–92 % from the fast loop alone;
95 % needs the repair loop's fleet-wide rewrite or a fleet. H8's threshold (0.92 over the
ten families, late window) is unchanged.

**Run status.** Pilot 4 complete (support `senselab-rsi`, s11/s23: 60 episodes each,
s23 49/60). Pilots 5a (`senselab`) and 5b (`senselab-rsi`), support + ci-fix, s11/s23, are
running with the contract on the deployed engine; a first 5a attempt (34 episodes) was lost
to a closed log pipe and is kept as `rsi-pilot-5a-partial`. Both are re-run after #449 and
#780 deploy; the three points are this amendment's development evidence.

**Pilot 5a result (2026-09-26, contract v2 — judgement-first sweep — on the deployed
engine; `analysis.rsi_dev`).** Support `senselab`, 120 episodes: success 92.5 % (grid v5c
82.1 %), late window 91.2 % (80.0 %), changed-class post-change 82.5 % (61.2 %), first-try
83.3 % (67.9 %), re-tries 21.6 % of episodes (36.3 %), repeat of a failed action 0.0 %,
recommendation present 100 %, `act` followed 100 %, recovery ≤ 3 class-episodes 6/6 (2/6).
Ci-fix `senselab`, 120 episodes: 95.8 % (92.1 %), late 95.0 % (88.8 %), changed-class
post-change 90.0 % (77.5 %), recovery 3/4 (the fourth exactly 3). Against the predictions:
(i) met (97.9–100 %); (ii) met (0 % repeats; 21.6 % of episodes still need a second
attempt, which is the sweep, not repetition); (iii) met; (iv) met (82.5 % / 90.0 %);
(v) met on the deployed engine (0.912 / 0.950); (vi) awaits 5b. The 13 residual failures
are 3 pre-change first encounters and 10 post-change sweeps, none with an error. In the
sweeps the new fix was reached in the 2nd or 3rd class-episode: the new fix that is the
textbook answer (`android-notifications → reinstall_app`) had never won anywhere on the
queue, so `elsewhere` order left it out of the five-action plan (`untried = 7`) behind
`explain_and_close`, `resend_email`, `advise_workaround`; and the agent's own picks under
the judgement-first rule (`clear_app_data`, `escalate_tier2`) were no better than the
order — at s23 ep 27 it overrode a correct `resend_email` suggestion on two attempts. The
ceiling paragraph above stands: a support cell at fleet 1 is bounded near 92–93 % by the
sweep's one unavoidable loss per changed class plus the search cost of the textbook case.

**Pilot 5c (launched 2026-09-26 07:5x UTC).** `senselab`, same cells as 5a, with #449's
`amfs_core` on the harness's `PYTHONPATH` (manifest `client_core.commit = 704fb753`): the
footer rendered client-side now brackets each untried candidate with its record elsewhere
(`resend_email (12/12 elsewhere)`, `reinstall_app (untested anywhere)`), while the
recommendation is still the deployed server's. Isolates the rendering half of #449 before
the deploy. Prediction: support late window ≥ 0.92 and changed-class post-change ≥ 85 %,
with `android-notifications` recovered in ≤ 2 class-episodes on both seeds.

**Seal stalls (root cause, 2026-09-26).** The `commit_outcome` stalls seen since pilot 2
(seals held 37–224 s, released together) were a lock convoy on `amfs_immutable_traces`:
the eval worker ran each job inside the tenant pool wrapper's per-checkout transaction,
so a repair job idle in an LLM call held `ACCESS SHARE` on the table for minutes; a
pro-api instance starting under load ran `schema_pro.sql` (whose `ALTER TABLE` takes
`ACCESS EXCLUSIVE`) and queued behind it; every seal's `SELECT content_hash` queued behind
the DDL. Captured live with a `pg_blocking_pids` sampler at 07:06–07:07 UTC. Fixes:
amfs-internal #780 (jobs and LLM-calling routes on a non-transactional checkout) and #781
(the traces schema runs once per database behind a fingerprint marker, under
`lock_timeout`). No 5a episode records an error, so the numbers above are not affected;
the stalls cost wall-clock, and in 5b they delayed the repair loop's proposals; the same
idle transaction was terminated twice by production's 5-minute
`idle_in_transaction_session_timeout` during 5b (07:08, 07:09 UTC), losing those repair
runs mid-flight.

**Pilot 5b result (2026-09-26 08:13 UTC).** `senselab-rsi`, 240 episodes. Support 92.5 %,
late 90.0 %, changed-class post-change 80.0 %, recovery 6/6; ci-fix 95.8 % / 93.8 % /
87.5 %, recovery 3/4. Indistinguishable from 5a's `senselab` — the repair loop no longer
costs the arm anything (pilot 4 had it at 85.0 % on support). Prediction (vi) **not met**:
every proposal came from the `template` source (0 % `repair_agent`; ≥ 80 % predicted),
first-proposal lag 13 / 5 / 3 episodes and none at all on ci-fix s11 (≤ 3 predicted), lever
`knowledge` throughout, nothing shipped (statuses `proposed`, `canary`, `test_failed`,
`test_inconclusive`). The repair agent's runs were the ones the seal-stall transaction
terminated and the ones #780's wall-clock budget is for, so (vi) is re-tested after the
deploy rather than counted against the arm here.

**Pilot 5c result (2026-09-26 08:30 UTC).** Support 91.7 %, late 88.8 %, changed-class
post-change 77.5 %; ci-fix 95.0 % / 92.5 % / 85.0 %. Prediction **not met** (late 0.888 <
0.92; changed-post 77.5 % < 85 %; android recovered in 2 class-episodes on s11 and 0 on
s23). Against 5a the brackets moved nothing the noise cannot explain (differences of 1–3
episodes per cell), and the failure analysis says why they could not: pooled over 5a/5b/5c,
the 23 changed-class post-change failures on support are 16 of 18 **first exposures** of a
class after the change (the record still says `act` on the old fix, nothing could know
otherwise, and the two in-episode retries recovered 2 of 18), 7 of 18 **second exposures**,
and 1 of 78 third-or-later (a fragmented label). On the second exposures the plan was
ordered by the record elsewhere and truncated to five, and the new fix was outside it or
fifth in 6 of 7; the bracket `reinstall_app (0/2 elsewhere)` — its wrong guesses on other
classes — read as a verdict. Reproducing the live footer for the worst ci-fix cell (s11,
`flaky integration failure`, seven actions tried) found a second defect: the footer's
`tried[:6]` and `failed[:5]` cut a posterior-sorted record, hiding `fix_code 0/4` and
`rerun_job 0/3` — the two actions the agent took a fifth and fourth time. Both are engine
fixes in amfs#450: losses elsewhere no longer order the sweep, the sweep names every untried
candidate, the footer never hides the most-failed action, brackets on winners elsewhere
only. The ceiling paragraph is now quantitative: with fleet 1 the first exposure of each
changed class costs one episode whatever the engine (6 of 120 on a support run = 5 %), so
a support cell is bounded near 93–94 % on the change-set-A schedule; the second-exposure
loss (2–4 per run) is what #450 is for.

**Pilots 5d/5e (planned, after the deploys of amfs#449, amfs-internal#780/#781).** 5d:
`senselab` on the deployed engine with #450's core client-side (manifest `client_core`),
same cells; 5e: `senselab-rsi`, same cells, for (vi). Predictions, dev seeds: 5d support
changed-class post-change ≥ 85 % with second-exposure failures ≤ 2 per run, late window
≥ 0.92, ci-fix `flaky-integration` recovered on s11 (no class at 4/4/4); 5e RSI metrics
first proposal ≤ 3 episodes after the change, `repair_agent` share ≥ 80 %, at least one fix
reaching `canary` per support cell; no `IDLETX` ≥ 60 s or `LOCKWAIT` in the sampler during
the runs.

**Pilots 5d/5e result (2026-09-26 10:00 UTC; deployed amfs#449 + amfs-internal#780/#781,
#450's core client-side, `client_core.commit = beace570`).** 5d (`senselab`): support
90.8 % / late 87.5 % / changed-class post-change 75.0 %; ci-fix 94.2 / 91.2 / 82.5. 5e
(`senselab-rsi`): support **94.2 / 92.5 / 85.0**; ci-fix 95.8 / 93.8 / 87.5. The 5d
prediction is **not met** on 5d (late 0.875, changed-post 75 %, `flaky-integration` s11
4/4/4 again) and met on 5e's support cells (late 0.925, changed-post 85 %, second-exposure
failures 2 on s23 and 1 on s11); the two runs are the same configuration up to the RSI
arm, whose loop shipped nothing (below), so the 3.4-point gap between them is the
run-to-run noise at n = 120, not an effect. Baselines on the same seeds, set and windows
(`e1-A-baselines`, GPT-5.5, fleet 1): support `none` 66.7 / 51.2 / 5.0, `pgvector`
73.3 / 66.2 / 35.0, `pgvector+rerank` 76.7 / 70.0 / 45.0, `pgvector-diy+outcomes`
70.0 / 61.2 / 27.5, `raw-traces` 75.8 / 63.8 / 27.5; ci-fix `none` 87.5 / 81.2 / 62.5,
the pgvector variants 86.7–89.2 / 81.2–83.8 / 62.5–67.5, `raw-traces` 87.5 / 81.2 / 62.5.
The 5e RSI predictions **failed on the mechanism**: first proposal came 3 episodes after
the change (met) and one fix reached `canary` on s23 (met), but the `repair_agent` share
was 0 % — all four fixes' rationales end in `template kept (repair agent gave up:
token_budget after 5 step(s), 221.9s)`; the 50k-token run budget bought five steps of a
conversation that re-sends a 24k-character evidence pack every step. The three Tier 2
replays the canary triggered on s23 each answered correctly on the fix's branch and each
failed its reflection step with `403 No access to branch 'repair/…'` (no API key created a
platform branch, so no grant exists). The regime-shift detector fired correctly on
`large-date-range-export-failure-workaround` (validated 3×, then failed) and the
escalation ping-pong incident opened. Both mechanism defects are fixed in
amfs-internal#782 (`AMFS_EVAL_REPAIR_TOKEN_BUDGET`, default 150k; the agent is told before
its last affordable step and asked to propose; repair branches readable by the account's
keys, search included) and, harness-side, a replay's notes are dropped rather than
written to the fix's branch. The sampler saw no `LOCKWAIT` during the runs but 126
`IDLETX` samples ≥ 60 s (max 384 s): the Cortex worker's tenant-pinned scan connection,
not the request path, and not a hazard while no DDL runs — logged as a separate risk. A
third diagnostic came from reading the lessons behind ci-fix s11's 4/4/4: after one failed
sweep the agent wrote `ci-flaky-integration-timeout-avoid-unrelated-explore` ("do not
follow generic explore-order unrelated fixes") and then obeyed that untested note over the
record on every later exposure, taking `fix_code` (0/4 here) first each time; this is the
class of failure the loop's corrective entries are for, and the reason the RSI arm's value
cannot show before #782 is live. Pilots 5f/5g: the same pair on the deployed #450 and
#782; predictions unchanged from 5d/5e, plus `repair_agent` share ≥ 80 % and replay
reflection errors = 0.

### 2026-09-26 (i) — the team grows: what a joining agent inherits

**Why.** E1/E2 measure a fixed team living through a change. Teams are not fixed: the
question a buyer of "RSI for agent fleets" asks is what happens to *quality* when agents
are added — does a newcomer start from the team's repaired state, or from the team's
mistakes? The transfer protocols of the continual-learning study (`Scenario.fleet_mode`
`pioneer` / `newcomer`, `analysis/transfer.py`) already showed that shared memory makes a
joiner *fast* (grid v3, support, newcomer at ep 40 onto a store with a change at 20:
first-exposure first-attempt `none` 0.32, `pgvector` 0.50, `senselab` 0.77; on changed
classes `pgvector` 0.00 with **9/9 poisoned first exposures** — an untested peer note in
context, nothing validated — against `senselab` 0.56 and 0/9). What they did not ask is
what the repair loop adds on top of the outcome loop for the joiner. This amendment adds
that probe, on the RSI arm, with the veterans' loop given time to act before the join.

**Design (probe (i), development seeds first).** `support`, `ci-fix` × seeds 11, 23 ×
protocols `pioneer` (agent 1 alone for 60 episodes, agents 2..6 join) and `newcomer`
(agents 1..5 for 60 episodes, agent 6 joins) × arms `pgvector+rerank`, `senselab`,
`senselab-rsi`, GPT-5.5, 100 episodes, change at 20, join at 60. Veterans are **replicas
of one hosted identity** (`--fleet-replicas`; always on for `senselab-rsi*`): one fix list,
one canary, one judge, so the team's repair loop is the loop of a team and not of six
strangers. Each joiner is a **new identity** (`Scenario.is_joiner`; `SenseLabArm.
saas_agent_id` keeps a joiner's member id) opened on the platform the moment it arrives:
nothing reaches it by construction, only through the shared memory — the entries, their
evidence status, the loop's corrective entries, fix memories. Join at 60 rather than 40
so that a fix proposed near the change has the ≥ 8-outcome canary floor behind it before
anyone new reads the store. The confirmatory version (seeds 37/41/59/73, both models)
runs after `analysis-freeze-rsi` with E1.

**Harness.** `SenseLabRsiArm` now holds every identity a cell opens (`_agents`), readies
each on first session, polls verdicts on the identity that sealed the trace, unions fixes
across identities, and reports the veterans' identity at the top level with each joiner
under `peers`. `analysis/team_grows.py` reads the cells and writes `TEAM_GROWS.md`.

**Endpoints (joiners; first exposure = the joiner's first episode on a class after
joining).** (A) first-attempt success on changed and on unchanged classes; stale-pick on
changed classes; escalation. (B) inheritance quality — per first exposure, what the
searches served: validated hits, untested peer notes, discredited entries in the avoid
list, platform-authored entries (keys no agent in the cell wrote); *poisoned* first
exposures (failed first attempt, untested peer note in context, no validated hit) and
*guided* ones (succeeded with a validated hit); the veterans' own last ten pre-join
episodes as the reference. (C) episodes and tokens to parity with the veterans (transfer.py's
units); cold-start curve. (D) disagreement with the veterans' late modal answer per class.
(E) tokens per joiner episode and per joiner-solved task. (F) RSI only: fixes proposed and
shipped before the join, levers, the joiners' own verdict and fix counts.

**Predictions, dev seeds.** (P1) Speed is the outcome loop's: `senselab` and `senselab-rsi`
joiners both reach parity within 5 episodes in ≥ 80 % of units; `pgvector+rerank` in
≤ 60 %. (P2) Quality is where the loop shows: `senselab-rsi` joiners' changed-class
first-exposure first-attempt ≥ `senselab`'s by ≥ 10 points **in cells where the veterans
shipped at least one fix before the join**, and no worse elsewhere; poisoned first
exposures ≤ 5 % for both SenseLab arms against ≥ 40 % for `pgvector+rerank`. (P3) In the
RSI arm ≥ 1 platform-authored hit appears in ≥ 50 % of joiner changed-class first
exposures (the loop's corrective entries reached a reader who never saw the failure); in
`senselab` that share is what the outcome loop's lesson-contrast entries alone give, and
the difference is reported. (P4) Disagreement with the veterans' late answer is lowest for
`senselab-rsi` on changed classes when the veterans' late success is ≥ 0.8, and *highest*
on the cells where the veterans' late success is < 0.5 (a joiner reading the record rather
than the team's habit). **What falsifies the claim:** if `senselab-rsi` joiners are no
better than `senselab` joiners on changed-class first exposures in cells where a fix
shipped before the join, the repair loop adds nothing to onboarding beyond the outcome
loop, and the paper says so; if the veterans' loop shipped nothing by episode 60 in most
cells, the probe is reported as inconclusive on P2/P3 and the join is moved later in the
confirmatory version, not the prediction. Cell counts are small (24 cells, 2 seeds); this
is a development readout to set the confirmatory design, not a confirmatory statistic.

**Readout, 2026-09-26 — `pgvector+rerank` and `senselab` halves (16 cells, 1,600
episodes; `senselab-rsi` held until amfs-internal#783 is live).** Report:
`results/rsi-teamgrows-dev-report/TEAM_GROWS.md`. Joiner first exposures after the join
(pioneer n=90 / newcomer n=32), first-attempt success: `senselab` 1.00 / 1.00 (changed
classes 1.00 on 36 and 10); `pgvector+rerank` 0.68 / 0.75 (changed 0.31 / 0.40). Every
SenseLab joiner episode after the join succeeded on the first attempt (pioneer 136/136,
newcomer 160/160); `pgvector+rerank` 0.62 / 0.62. Inheritance: on 100 % of SenseLab
joiner first exposures the context held a validated hit *and* a discredited one shown as
such (2.6 validated / 3.0 avoid hits per episode; 0 poisoned, 122/122 guided); on 100 %
of `pgvector+rerank` first exposures it held only untested peer notes (7.3 per episode,
0.96 stale), poisoned 0.32 / 0.25 overall and 0.69 / 0.60 on changed classes.
Onboarding: `senselab` parity 20/20 and 4/4 joiners within a median 5 episodes, cold-start
1.00 on each of the first five episodes; `pgvector+rerank` 16/20 (20 % never within the
window) and 4/4 at a median 11.5, cold-start 0.25–1.00. Disagreement with the veterans'
late answer: 0.02 / 0.00 vs 0.06 / 0.05. Cost per solved joiner task: 4.4k / 4.1k tokens
vs 7.6k / 6.8k. **P1 met** for `senselab` (100 % within 5) and for the baseline (80 %
pioneer, but newcomer 100 % at a median 11.5 — the ≤ 60 % clause fails on the newcomer
half; the speed gap shows in the median, not the eventual-parity share). **P2's poisoned
clause met** (0 % vs 69 % / 60 % on changed classes; ≤ 5 % vs ≥ 40 %); the `senselab-rsi`
vs `senselab` clause is untestable on this half and, with `senselab` at 1.00, has no room
on dev seeds — the confirmatory version must make the changed-class first exposure harder
(join earlier relative to the change, or a second change after the join) for P2/P3 to
discriminate the two SenseLab arms. **P3** pending the RSI half. **P4** consistent with
the record (lowest disagreement where veterans' late success ≥ 0.8; no cell had veterans
< 0.5). The mechanism the record shows is not the joiner being smarter: it is reading a
scope where the stale rule is labelled discredited and the replacement validated, both
written by veterans it never met.

### 2026-09-26 (j) — the loop's own history reaches memory and the next draft

**Why.** The repair loop's record of what it tried — a fix's corrective entries, its Tier 1
verdict and notes, the replay or canary that failed it, the sweep that called it
regressed — lived only in the fix ledger (`amfs_eval_fixes`, `amfs_eval_fix_tests`).
Nothing read that ledger on the paths this study measures: a briefing compiles memory
entries, and the repair agent's evidence pack carried the failing runs, the implicated
entries and a template, not the previous draft on the same entry. So the two places a
failed repair could teach were both blind to it. The second draft on an entry was a fresh
guess (`_fix_covering` lets a `test_failed` fix be followed by a new proposal, and on 5h
the first drafts on `playbook-tier1` and `android-alerts` came back `test_failed` /
`inconclusive`), and a joiner in probe (i) inherited "what failed" from the outcome loop's
evidence labels but not "what the loop already tried and rejected". The agents in this
study follow a rule the repairer did not: mark the attempt before trying another.

**Mechanism (amfs-internal#784; `repair/record.py`).** (1) On `test_failed`,
`replay_failed`, `canary_failed`, `regressed` and a rollback by hand, the loop writes a
`lesson-fix-<key>` belief (0.7) on the entity of every entry the fix changed: fix id, the
value it tried, the verdict and its notes, a one-line lesson not to reuse the change as it
stands. One key per repaired entry, versioned per attempt. On ship it writes a
`decision-fix-<key>` fact (0.8): what changed, from which version, why, on what evidence
(canary / customer replay / offline replay / a person). Both are ordinary entries, so the
briefing and search that already serve the agents surface them with no change to the
retrieval path; the `lesson-contrast-repair-*` pointer that feeds `discredited[].
replaced_by` is unchanged. (2) `FixService._prior_fixes` reads the ledger's `test_failed`
/ `regressed` / `dismissed` fixes on the same behavior or entry and `build_evidence`
carries them as `RepairEvidence.prior_fixes` (newest first, ≤ 5: what each changed, its
rationale, how it was judged); the drafting message names them and instructs the agent
not to propose the same change again and to state what it does differently. Both propose
paths (judge, regime shift) pass them. Never fatal on either side.

**What changes in the measured arms.** Only `senselab-rsi*` (the loop is off elsewhere).
The outcome-loop arm `senselab` is unaffected, so the E1 contrast `senselab-rsi` vs
`senselab` now includes this mechanism; the `senselab-rsi-review` ablation (proposals
held at `review`, nothing ships on its own) still exercises (1) for failed drafts and (2),
which is the point — a loop that cannot ship can still teach.

**Endpoints (added to the fix lifecycle readout, `analysis/rsi.py`).** (G1) *Repeat
drafts*: among second-or-later drafts on an entry whose previous draft failed, the share
whose corrective value is a near-duplicate of a prior failed value (normalised Jaccard on
tokens ≥ 0.9). (G2) *Second-draft pass rate*: Tier 1 pass rate of second drafts on an
entry, against first drafts. (G3) *Time to first shipped fix per changed class*, in
episodes, for classes whose first draft failed. (G4) Joiner-side, probe (i): share of
joiner changed-class first exposures whose context held a `lesson-fix-*` or
`decision-fix-*` entry (a platform-authored hit of this kind), and first-attempt success
conditioned on it.

**Predictions, dev seeds (5i and the `senselab-rsi` halves of probe (i) run before #784 is
live are the pre-registered baseline; the first run after it is live is the treated
readout).** (P1) G1 falls from whatever the pre-#784 run shows to ≤ 10 %. (P2) G2 rises by
≥ 15 points for second drafts (a prior failed draft in the pack is informative, not
noise). (P3) G3 falls on classes whose first draft failed; no change on classes whose
first draft passed (the mechanism only fires on failure). (P4) In probe (i), `decision-
fix-*` / `lesson-fix-*` hits appear in ≥ 50 % of `senselab-rsi` joiner changed-class
first exposures where the veterans' loop drafted before the join, and P3 of amendment (i)
is evaluated with these counted as platform-authored hits. **What falsifies the claim:**
if G1 was already ≤ 10 % before #784 the repairer was not repeating itself and the paper
says the mechanism is prophylactic, not measured; if G2 does not move, the failed draft's
record is not usable evidence for the model and the paper says so; if joiners see the
entries (P4) but their changed-class first-attempt success does not move relative to
`senselab` joiners, the entries are read and not acted on, which the paper reports as a
limit of memory-only transfer.

**Scope note.** This is an engine change during the development phase, before
`analysis-freeze-rsi`. It is preregistered here before the treated run, the untreated 5h
and probe-(i) RSI cells are kept as the pre-#784 baseline, and no confirmatory seed has
been run on either side. Dev-seed readouts on both sides are reported; the confirmatory
grid runs with #784 live.

**Cut time (recorded 2026-09-26).** #784 merged at 14:49Z; `amfs-api-02003-hcb` took
100 % of traffic between 14:59:44Z (revision created) and 15:00:46Z (first observation at
100 %). Pre-#784 baseline: pilot 5h (complete, 240 episodes) and the 29 episodes of 5i
that ran before the merge (`results/rsi-pilot-5i-aborted-pre784`, stopped at 14:50Z so
the treated run would not straddle the deploy). The `senselab-rsi` halves of probe (i)
were left running: their `support` cells (join at 60; joiner first exposures at episodes
60–70, all sealed before 14:50Z) are pre-#784, their `ci-fix` cells post-#784 — a
scenario × treatment confound that the (i)/(j) readout reports rather than pools. Treated
run of the 5h protocol: `rsi-pilot-5j`, launched 15:05Z on the new revision.

**Treated-run readout (recorded 2026-09-26, dev seeds).** 5j completed 16:12Z: support
57/60 (s11), 55/60 (s23); ci-fix 57/60, 56/60 — pooled 93.3 % / 94.2 % against 5h's
92.5 % / 95.0 %, flat within noise, as it has to be: the mechanism fires only on a failed
or shipped fix, and on the new revision **no fix failed, regressed or shipped**. Post-#784
ledger (10 fixes, 5j and the probe-(i) `ci-fix` cells together): `canary` 5,
`test_passed` 2, `test_inconclusive` 2, `proposed` 1; Tier 1 tests 3 passed (fidelity
1.0), 5 inconclusive (0.59, "could not be replayed"), 2 skipped. Hence 0 `lesson-fix-*`,
0 `decision-fix-*`, 0 events carrying `recorded` — the hooks were never reached, not
bypassed. **G1–G3 are undefined on both sides**: the pre-#784 baseline has two
`test_failed` fixes (`playbook-tier1`, `procedure-avoid-improper-escalation-loop`) and no
later draft by the same agent on either entry, so the repeat-draft denominator is 0/0
before and after; G2 has no second drafts; G3 no shipped fix on a class whose first draft
failed. G4: the probe-(i) `ci-fix` joiner contexts held no `lesson-fix-*` /
`decision-fix-*` entry (none existed to be retrieved), so P4 is not evaluable there
either. The structural reason is stronger than sample size: `prior_fixes` is agent-scoped
and every cell is a fresh agent, so a second draft on a failed entry needs ≥ 2 drafts on
*one* entry inside *one* 60-episode cell, and the observed rate is ~1.25 drafts per cell
(5 across 5j's four cells, 0–2 each). On dev seeds the paper therefore reports (j) as
**prophylactic, exercised by its tests, not measured** — the falsifier's own wording — and
defers the measured readout to E2 (200 episodes, fleet-10, two changes), where the same
agent drafts repeatedly, and to amendment (k) if adopted, which raises the number of fixes
that reach a ship or a failed canary at all. Nothing about the mechanism is changed by
this readout.

### 2026-09-26 (k) — the canary gate inside a cell: two defects, corrected before the run

**Why.** Across every `senselab-rsi` cell to date (5h, 5i, 5j, both probe-(i) RSI halves)
the loop drafted fixes and shipped none: 11 fixes sat in `canary` at the end of the day.
The pre-registered look schedule (E2 mapping: `canary_pct` 50, `canary_min_outcomes` 8,
`POST /fixes/{id}/canary/evaluate` every 5 sessions, harness timeout 60 sessions) *was*
running — the canaries carry 5–16 evaluations each — so the earlier reading in
`HEADROOM_RSI.md` ("evaluated only by the hourly sweep") was wrong and is corrected there.
Reading every canary's arms against the traffic behind it found two defects and one
limit, none of which the look schedule reaches:

1. **An outcomes-only canary could never be graded** (engine, amfs-internal#785). A
   regime-shift fix has neither `judge_id` nor `behavior_id`; `FixStore.canary_arm_counts`
   graded such a fix's arms with `SELECT NULL AS verdict`, so `graded` was 0 whatever the
   traffic. Fix `7a7a164b` (5j `ci-fix` s11, a rewrite of `repo-readme-ci`, 50 %) had 27
   canary / 26 control stamped runs after start, all 53 with a judge verdict, and arms 0/0
   after 16 evaluations. #785 grades an outcomes-only fix on the signal that proposed it:
   a run is graded when it reported an outcome *and* read the entry the fix is about (the
   rewritten key, or what the risk entry's `pattern_refs` name — `canary.implicated_entries`),
   and has recurred when that outcome was not a success; a run that never read the entry
   stays in the arm for the guard but is not graded, so a canary on a rarely-read entry
   waits rather than deciding on runs the fix never touched. Judge- and behavior-scoped
   grading is unchanged.
2. **Inconclusive fixes waited for the platform's hourly sweep** (harness). Under
   `auto_after_canary` the harness filled a freed slot only with `test_passed` fixes; a
   `test_inconclusive` fix (Tier 1 could not replay the case — the common verdict on this
   benchmark, fidelity 0.5–0.75) was queued by the platform for the small canary and
   started by its hourly `canary_sweep` — at 15:20, 15:52, 16:21 for fixes proposed at
   13:36–15:24, every one after its cell had ended (4 of the 11). The look schedule now
   fills a freed slot from the same queue the sweep reads (`FixStore.canary_queue`: passed,
   inconclusive, or `proposed` with nothing to replay; no replay awaited; oldest by
   `tested_at`), at the product's `INCONCLUSIVE_CANARY_PCT_CAP` of 10 % for a fix no
   offline test vouched for — exactly `_canary_overrides`. Nothing about *what* the gate
   requires changes; only *when* the loop looks, which is what the look schedule already
   was for.
3. **The limit that stays: cell length.** Three 50 % canaries started within minutes of
   their test and saw no traffic because the cell was over; the best-filled judge-scoped
   canary (`58f433d4`, 17/13) cleared the floor of 8 and is a legitimate "no significant
   difference yet" (1 recurrence in the canary arm, 0 in control); `a4daf346` (8/6) needed
   two more control runs when its cell ended. A fleet-1 60-episode cell cannot fill a 10 %
   canary (8 graded canary runs need ~80 sessions after start) and fills a 50 % one only
   if the fix lands by about episode 40. Not amended: fleet-1 cells report the canary as it
   stood at cell end (`canary_undecided_at_end`), and the measured ship-rate readout is
   E1's fleet-6 cells and E2 (fleet-10, 200 episodes), where the arms fill in minutes.

**What changes in the measured arms.** `senselab-rsi` and its `-review` / `-selfreport` /
`-judges` variants (all `auto_after_canary` or `review`); `-nocanary` (ships on Tier 1) and
`-noreplay` (starts every fix's canary already) are untouched by (2) and reached by (1)
only where a regime fix's canary is read. Baseline arms, scenarios, change sets, rubric
and metrics unchanged.

**Endpoints (fix lifecycle readout).** (K1) Share of canaries that *decide* (pass, fail or
inconclusive-by-rule) before cell end, by fix path (judge / regime) and Tier 1 verdict
(passed / inconclusive). (K2) Median sessions from proposal to canary start, by Tier 1
verdict. (K3) Graded runs per arm at cell end for regime-path canaries (0/0 before #785 by
construction). (K4) Fixes shipped per cell, and the changed-class success after the first
ship (the RSI-specific outcome the paper is about).

**Predictions, dev seeds (5j is the baseline; the first 5h-protocol run with #785 live and
this harness is the treated run, `rsi-pilot-5k`).** (P1) K3: every regime-path canary with
≥ 1 stamped run in each arm has graded > 0 in both (0 of 2 before). (P2) K2 for
inconclusive fixes falls from "after cell end" to ≤ 10 sessions. (P3) K1 rises above 0 for
50 % canaries whose fix landed by episode 40; it stays near 0 for 10 % canaries in fleet-1
cells (the limit in 3, reported as such). (P4) K4 > 0 on at least one dev cell — the first
ship of the study — and no `regressed` within the cell. **What falsifies the claim:** if
K3 stays 0/0 with stamped runs present, #785's grading does not read this traffic and the
canary is not the right gate for outcomes-only fixes on this benchmark; if K1 stays 0 with
arms filled, the fixes are not moving the outcome and the paper says the loop drafts but
does not repair at this scale; if a shipped fix regresses within its cell, the gate is
letting through changes the outcome record then rejects, which the paper reports as a
false-ship rate rather than hides.

**Scope note.** Engine change (#785) opened as a PR for approval, benchmark change
(harness look schedule) committed here, both before the treated run; 5j and the earlier
RSI cells are the pre-(k) baseline; no confirmatory seed has been run on either side.
Bruno approved (1) and option (a) on 2026-09-26; the corrected reading of what (a) is —
the look schedule already existed; the change is what the look schedule *starts* — is
recorded here rather than silently substituted.

**Cut time (recorded 2026-09-26).** amfs-internal #785 merged to `main` at 22:15:19Z as
`ea189b5e` after a Bugbot round (`5ca0a4cb`: attempt-charged regime failures — a failed
`amfs_record_attempt` whose `causal_entry_keys` name the implicated entry — now count as
a read and as a recurrence, so a run that read the shifted entry, failed on it and
recovered is graded and recurred rather than passed). Deployed by run 36275674268:
`amfs-api-02011-4wj` at 100 % from 22:23:47Z, `pro-api-00775-dkt` from 22:21:43Z; deploy,
Room lifecycle smoke and Notify SRE dispatch all green. Treated run `rsi-pilot-5k`
launched 22:27:29Z from harness commit `98ecd4c3` with the 5h knobs (`CL_RSI_TIER2=1`,
`CL_RSI_VERDICT_WAIT=240`, `CL_SENSELAB_TIMEOUT=90`, fleet 1, workers 2, dev seeds 11/23,
support + ci-fix, 240 episodes). Every RSI cell that started before 22:23:47Z (5h, 5j,
the team-grows halves) is pre-(k) baseline for K1–K4; the platform's hourly sweep also
runs the new grading, so canaries from those cells that are still running on the ledger
will start grading from this revision — the K3 "before" number is the arm state read
before the cut, not a later re-read. Stale ngrok tunnels for the team-grows hubs
(8788/8789, no listeners) were removed before launch so the reserved domain routes every
Tier 2 webhook to the single live hub on 8787.

**Readout (recorded 2026-09-27, `rsi-pilot-5k` complete, 240/240).** Success pooled over
the two dev seeds: ci-fix 95.8 % (5j 94.2), support 93.3 % (5j 93.3); changed-class
post-change success 90.0 / 82.5 % (5j 82.5 / 82.5) — flat, as expected with nothing
shipped. Fix lifecycle: three fixes, all `repair_agent`-drafted, all knowledge lever, all
outcomes-only (regime path; no judge or behaviour on the fix), all in `canary` at cell end;
0 shipped, 0 `test_inconclusive`. Routing worked in every canary (the `X-AMFS-Canary`
header on the session, the stamp on the sealed trace): ci-fix s11 drew 32 canary / 17
control sessions from session 10, ci-fix s23 19 / 19 from session 20, support s23 (10 %
canary, Tier 1 inconclusive) 1 / 17 from session 40. Against the predictions:

* **P1 (K3 > 0 in both arms with stamped runs) — falsified, 3 of 3.** Every arm had
  outcomes on every run (32 / 17, 19 / 19, 1 / 17) and `graded = 0` on both sides for the
  whole window. The predicate is not at fault; the population is. #785 grades an
  outcomes-only fix on *runs that read the entry the fix is about*. For both ci-fix fixes
  that entry is `repo-readme-ci` — the seeded README — and no run in either arm read it:
  the agent read it in episodes 1–4 only, before it had notes of its own, and from
  episode 5 its retrievals return its own situation-keyed notes. The support fix's
  implicated entry is one of the agent's notes (`…tier1-after-settings-fail`) that no run
  in its 20-session window read. A fix to an entry the traffic has stopped reading cannot
  be graded on reads of it — and it cannot repair anything either.
* **How the README came to be "a validated entry that stopped working".** Episodes 1–4
  read it and succeeded (4 successes: validated). Episodes 5–6 were first exposures to a
  new class (`pip-audit`); each failed attempt had read only the README, so under #783
  the attempt failures were charged to it (`successes 4, failures 2` on the job payload,
  proposed on the episode-7 seal). The README was not wrong — the class was new. On a
  benchmark where every class is a first exposure once, any broadly read entry becomes a
  regime candidate after its first three successes. **A first-exposure failure is not a
  regime shift**, and the detector cannot tell them apart from the entry's aggregate
  record alone.
* **P2 (K2 ≤ 10 sessions for inconclusive fixes) — met.** Support s23: proposed at 38,
  Tier 1 inconclusive, canary at 40 (2 sessions, the 10 % cap). Passed fixes: 2–3 and 6
  sessions.
* **P3 (K1 > 0 for 50 % canaries landed by episode 40) — not met, for the P1 reason.** Both
  50 % canaries filled both arms well inside the window (17 and 19 outcomes on the smaller
  side against a floor of 8) and could not decide because `graded` never left 0. The slot was held for 50 and 40 sessions by
  canaries that could not be read, and would have been held until the 1-day timeout.
* **P4 (K4 > 0, a first ship, no regression) — not met.** 0 shipped.

**What this means for the loop, and two engine corrections proposed (PRs for approval).**
(1) *Canary futility on ungraded arms.* `decide()` already frees the slot when both arms
are graded and show no effect (rule 6); it has no rule for arms full of outcomes and
empty of grades. Proposed rule 1b: with `FUTILITY_MULTIPLE × min_outcomes` outcomes in
both arms and no graded run in either, the canary is **inconclusive ("ungraded")**, the
note naming what grading needed (the fix's judge, its behaviour, or reads of the entries
it is about). On this cell that frees the slot at ~24 sessions instead of a day. (2) *A
first-exposure failure is not a regime shift.* Proposed guard in `regime.on_trace_sealed`:
when the sealed trace carries a situation label, an entry is a regime candidate only if
some earlier successful run of the same agent **in the same situation** read it — "this
used to work *here*". The README had no success in the `pip-audit` situation, so no
proposal; a set A/C change (the same situation, the same note, now failing) still
proposes. Traces without a situation label are unchanged. Both are engine changes and go
to Bruno as PRs; **`rsi-c1-rsi` is held until they are deployed**, because a
false-positive README canary would otherwise occupy the cell's one canary slot before
the set C change at episode 30 produces the fix the run is meant to measure (exactly the
5k sequence), and under `-nocanary` a README rewrite that passed Tier 1 would ship as a
false-positive K4. The paper reports the false-positive proposal rate as a cost of the
loop either way (pre-fix cells: 2 of 3 proposals).

### 2026-09-26 (l) — SenseLab is the engine; the repair loop is measured where exhaustion cannot recover

**Why.** Six pilots and a mechanism reading (HEADROOM_RSI.md, "What the tiers can and
cannot do on set A") say the same thing: on set A the `senselab-rsi` arm is at parity with
`senselab` (5f 93.3 / 92.5 / 85.0 vs 5h 92.5 / 90.0 / 80.0 and 5j 93.3 / 91.2 / 82.5 on
support; 95.0 / 92.5 / 85.0 vs 95.0 and 94.2 on ci-fix), zero fixes have shipped in any
cell, and both sit at the computed ceiling for a 60-episode fleet-1 cell (93–94 %: first
exposures are a 5 % floor, second exposures the only headroom). Four properties of set A
make this a result about the *benchmark*, not about the repair loop: (1) every task is a
choice among ten enumerated actions and a change moves one of them, so the correct answer is
recoverable by exhaustion — action priors and `explore` mode are the right algorithm and
they converge in ≤ 2 exposures; (2) a wrong action returns "Still happening" — a failed
trace carries no information about the new answer, so a repair agent drafting from it knows
what the agent knows; (3) the environment's verdict is verified, binary and immediate, so
the judge can only agree with a label the outcome loop already has; (4) the regime
predicate wants ≥ 3 validations (`REGIME_MIN_SUCCESSES`) and the canary 8 outcomes per arm
after a 4–16-episode detection lag, while a class recurs every ~8 episodes — a perfect fix
ships after the second exposure, when the outcome loop has already recovered. E1's primary
endpoint (first-attempt success on changed classes, episodes 40–59) therefore cannot
separate the tiers on set A, and the study as preregistered would report the repair loop
as a null without having tested what it is for.

**What the paper says.** SenseLab is one self-improving memory engine and is presented as
one system, in three tiers that the ablations isolate:

- *Tier 1 — the outcome loop.* Entity-scoped versioned memory with semantic retrieval; the
  Cortex briefing with `validated` / `discredited` / `regime_shift` sections and an
  `evidence_status` on every hit; per-situation action priors and the `act` / `explore` /
  `escalate` recommendation with per-agent exploration assignment; attempt boundaries with
  credit assignment and `lesson-contrast-*` entries; confidence decay by memory type;
  cross-agent inheritance (a joiner reads the fleet's record); decision traces with
  `task_input` / `response_text`. The memory improves itself from outcomes, no human in the
  loop.
- *Tier 2 — the repair loop.* Continuous trace judges with a customer rubric; verdicts
  clustered into behaviours and incidents; auto-proposed fixes drafted by a repair agent
  with `search_traces` / `compare_traces` / `find_similar_traces` / `read_procedures` over
  the account's record, with a first-class `procedure` memory type (goal, steps,
  on_failure, verify); corrective entries on `repair/<id>` branches; Tier 1 memory test on
  a case set, Tier 2 customer replay, Tier 3 canary routing with stamped arms; `decide()`,
  ship, recheck, rollback; review policy. The engine rewrites and validates its own
  knowledge and procedures before anyone acts on them.
- *Tier 3 — the recursion.* `lesson-fix-*` / `decision-fix-*` entries from the loop's own
  history into briefings and the next draft's evidence (`prior_fixes`, amendment (j)); the
  repairer's prompt store with candidate A/B and `repairer_stats`; the Managed Models
  distilled from the repair record (E7). The loop improves the loop.

Arm names in the paper map to harness arms as follows; the harness names do not change.

| paper | harness | what it is |
|---|---|---|
| **SenseLab** | `senselab-rsi` | the full engine: Tiers 1–3, `auto_after_canary` |
| SenseLab − repair loop | `senselab` | Tier 1 only |
| SenseLab − canary | `senselab-rsi-nocanary` | ships on a passed replay |
| SenseLab − replay | `senselab-rsi-noreplay` | canary is the only test |
| SenseLab (review) | `senselab-rsi-review` | nothing ships; review load |
| SenseLab − protocol | `senselab-rsi-minimal` | retrieve + commit only |
| SenseLab (self-report) / (judges only) | `senselab-rsi-selfreport` / `-judges` | E3 verdict sources |
| SenseLab − repair loop − labels | `senselab-unlabeled` (new) | Tier 1 with no failure signal |
| maintainer oracle, lag *k* | `senselab-oracle-fix-k` | a human told about the change |

The headline tables (E1, E2, E4) compare **SenseLab** — the full engine — against
frontier-alone and the retrieval baselines. **Attribution rule:** a gain is credited to a
tier only where the ablation that isolates that tier shows a paired difference. On set A the
gains over every baseline are credited to Tier 1, and the paper says so in the sentence that
reports them; the paper does not contain the claim that the repair loop produced the set-A
result. The repair loop's contribution is measured where exhaustion cannot recover — the
regimes below — and reported with the same rule, including a null if that is what is found.

**Feature → evidence map.** Every feature the paper describes has one place where it is
measured; a feature without a row is described as unmeasured.

| feature | tier | isolated by | endpoint | where |
|---|---|---|---|---|
| evidence gating, briefing sections | 1 | `senselab` vs `pgvector-diy+outcomes`, `raw-traces` | changed-class recovery, late-phase success | E1 |
| action priors, recommendation, situations | 1 | `-nopriors`, `-nosituation`, `-nocontract` | exposures-to-recover, first-attempt success | E1 secondary |
| attempt boundaries, lesson-contrast | 1 | `-episode`, `-attempts`, `-minimal` | recovery within a task, stale-pick rate | E1 secondary |
| cross-agent inheritance | 1 | team-grows probe (i) — done: 1.00 vs 0.31–0.40 first-exposure, poisoned 0 vs 0.60–0.69 | joiner first exposure, poisoned follow | probe (i) |
| judges | 2 | `senselab-unlabeled` vs `senselab-rsi-judges` | recovery without labels; judge P/R vs oracle | E3 (primary, below) |
| behaviours, incidents | 2 | `senselab-rsi` verdict record | cluster-to-taxonomy match, incident latency | E3 |
| repair-agent synthesis, procedures | 2 | set C, `senselab-rsi-nocanary` and `senselab-rsi` vs `senselab` | first-attempt success on the moved half after the first ship; procedures shipped | E1-C (below) |
| Tier 1 / Tier 2 replay | 2 | `-noreplay`; probe (b) | gate confusion matrix, false-ship, adversarial rejection | E2, probe (b) |
| canary | 2 | `-nocanary`; probe (a) | regressions caught after a Tier 1 pass, A/A false-pass | E2, probe (a) |
| ship, recheck, rollback, blast radius | 2 | probes (c), (e) | unlearn-own-repair episodes, reward-hacking flags, entries restored | probes, E6 |
| lesson-fix / decision-fix, prior_fixes | 3 | (j) G1–G4; T1 below | second-draft pass rate after a failed first draft | E2 |
| repairer prompt A/B | 3 | E5 | proven-fix rate by prompt version, re-measured on other families | E5 |
| distilled students | 3 | E7 | post-change first-attempt success, pinned-case reproduction | E7 |
| cost, latency, auditability | all | E6 | USD per solved task, ms, verifiable-chain share | E6 |

**What changes in the study.**

1. **Set C — inference-required changes.** A third change set in which the new correct
   action depends on an attribute of the instance that the tools show, so the class's old
   fix keeps working on a *kept* half and fails on a *moved* half (50/50 per instance,
   drawn deterministically from seed, episode and salt; `truth["cond"]` and the episode tag
   `change_cond` mark the moved half). A single-action sweep cannot represent the rule; the
   record shows the stale entry *contested*, not discredited; a repair agent that compares
   failing and succeeding runs can write the rule as a `procedure`. Defined for the
   development pair here; written for the remaining eight families by the same construction
   and registered by hash before E1, the four held-out families blind.
   - *support* (attribute in the ticket header or `lookup_account`, which now also shows
     the app version and the card's issuer country): `android-notifications` — app major
     ≥ 9 → `reinstall_app` (v9 moved to a new notification channel; a reinstall migrates
     it), older → `advise_settings_change` unchanged; `export-timeout` — plan `team` →
     `advise_settings_change` (asynchronous export with notifications shipped to the team
     plan), `free` / `pro` → `advise_workaround` unchanged; `card-declined` — domestic
     issuer → `resend_email` (the new provider's one-time re-verification mail), foreign
     issuer → `escalate_tier2` unchanged (the allow-list).
   - *ci-fix* (attribute in `get_pr_files`): `flaky-integration` — the PR touches
     `tests/integration/cassettes/` → `edit_generated_file` (regenerate the recorded
     cassette), otherwise `rerun_job` unchanged; `audit-pinned` — the PR touches
     `legacy_sdk/` → `add_audit_exception` unchanged, otherwise `bump_dependency` (the
     modules off the legacy SDK are unpinned).
   - C moves A's classes, disjoint from B, so `OTHER_VARIANT["C"] = "B"` and the
     two-change protocol is C then B. Oracle entries state the rule as a maintainer would.
   - **C protocol**, fleet 1: 100 episodes, change at 30, `change_focus = 0.5`. Thirty
     pre-change episodes give each of eight classes ≥ 3 validations (the regime predicate);
     70 post-change episodes at focus 0.5 give ~12 draws per changed class, ~6 on the moved
     half; a proposal plus Tier 2 plus 8 outcomes per arm at 50 % routing is ~25 sessions,
     so a canary-gated ship lands around episode 65 and `-nocanary` within minutes of its
     replay. A wrong *unconditional* fix regresses the kept half, so the design carries its
     own false-ship detector.
2. **E3 becomes a primary comparison.** New arm `senselab-unlabeled`: Tier 1 with the
   verdict the agent would report (`success` unless it escalated), no `verified_by`, no
   attempt boundaries — the mirror of `senselab-rsi-judges`. The weak-label comparison is
   `senselab-unlabeled` vs `senselab-rsi-judges` on sets A and C, with the labelled pair as
   reference; probe (d) (feedback delayed three episodes, so the sealed trace is neutral and
   the judge must infer failure from behaviour) is the hard case and is reported next to it.
3. **`senselab-rsi-nocanary` is co-primary at fleet 1** (sets A and C): by the arithmetic
   in 1 it is the only policy that can ship before the second exposure at this cell length.
   The canary-gated arm stays primary at fleet ≥ 6 and in E2, where the arms fill in
   minutes, and is reported at fleet 1 with K1–K4.
4. **Endpoints for the RSI claim.** First-attempt success on changed classes (E1 primary,
   unchanged) *and*: exploration cost — failed attempts weighted by severity (`refund` is a
   critical failure the outcome loop pays for on a live customer) and tokens per solved
   task; time-to-repair; false-ship and regression rate; joiner zero-discovery; detection
   latency under weak labels. Tier 3, observational, with data only where fixes exist
   (E2, set C): (T1) among fixes whose first draft failed Tier 2, the share whose second
   draft passed, and whether the first draft's `lesson-fix` was in its evidence; (T2)
   repairer prompt versions and proven-fix rate per version (E5); (T3) `lesson-fix` /
   `decision-fix` entries surfaced in later briefings and drafts.
5. **Oracle bound on the development seeds.** `senselab-oracle-fix-5` on set A, support and
   ci-fix, seeds 11 / 23, run now: a maintainer who writes the correct entry five episodes
   after the change is the upper bound on any repair of set A. It tests the ceiling claim
   directly (P4).
6. **Control separation.** `AMFS_EVAL_AUTO_PROPOSE` is a process-wide switch, so control
   agents receive proposals (on branches; nothing shipped; zero reads — documented in
   HEADROOM). A per-agent `auto_propose` in the repair settings is an amfs-internal PR
   candidate needing approval; until it lands the exposure is reported, not assumed away.
7. **Harness.** `--change-variant C`; `change_cond` tag; `lookup_account` shows app version
   and issuer country in every variant (irrelevant to A and B, whose dev cells ran without
   the two fields; noted); `procedure` values rendered legibly to every arm's agent (goal,
   steps, on_failure, verify) instead of a dict repr.

**What does not change.** Sets A and B; the judge rubric; canary knobs; the seeds and the
dev / confirmatory split; the freeze rule; E2, E4, E7 as specified; the (k) endpoints and
`rsi-pilot-5k`, which runs to completion on the pre-(l) harness and is read as planned.

**Predictions (development seeds; C protocol on support + ci-fix unless stated).**
(P1) *Tier 1 on C:* `senselab` first-attempt success on the moved half, episodes 30–99,
between 40 and 75 % — the record is contested and the agent may write its own conditional
note — with episode success ≥ 85 % from in-task retries; `none` and the pgvector arms
≤ 55 % on the moved half. (P2) *Tier 2 on C:* `senselab-rsi-nocanary` ships ≥ 1 corrective
or procedure by episode 60 in ≥ 3 of 4 cells; after a cell's first ship, first-attempt
success on the moved half ≥ 85 % and ≥ 15 points above `senselab` paired on the same
episodes; the kept half and unchanged classes within 5 points of `senselab` (no
regression). (P3) *Canary-gated on C:* ships by episode 80 in ≥ 2 of 4 cells; the same
direction in the post-ship tail; K1–K4 reported. (P4) *Oracle bound on A:*
`senselab-oracle-fix-5` within 3 points of `senselab` (support 93.3, ci-fix 95.0) — the
ceiling is real. (P5) *Weak labels on A:* `senselab-unlabeled` late-phase support success at
or below `pgvector+rerank` (70.0) — the loop validates the stale pick — while
`senselab-rsi-judges` stays within 5 points of labelled `senselab-rsi`. **What falsifies
the claim:** if P2 fails *with fixes shipped*, the repair loop's synthesis adds nothing over
a capable agent with priors even where exhaustion cannot recover, and the paper reports the
repair loop as a validation and safety layer, not an intelligence layer; if P4 fails
(oracle ≥ 5 points above `senselab`), the ceiling argument is withdrawn and set-A parity is
reported as a repair-loop shortfall; if P5's judges arm falls with the unlabeled one, the
judges do not turn this benchmark's traces into usable verdicts.

**Scope note.** Development phase; no confirmatory seed has run on any arm. Harness and
scenario changes are committed with this amendment; the engine change (6) is a PR candidate
and is not assumed. Bruno approved the reframing and the redesign on 2026-09-27 JST
(2026-09-26 UTC) and asked for the recommendation to be re-checked against three
requirements — SenseLab in the strongest defensible position, the reader understanding
that SenseLab *is* the engine and what its features do, and the tests exercising Tiers 2
and 3 — which is what the framing, the attribution rule, the feature map and items 1–5
respond to. Two things were considered and not taken: a platform-served "SOP slot" the
agent always reads (an affordance the baselines lack; procedures are surfaced by the same
retrieval every arm gets, only rendered legibly), and loosening the frozen judge rubric so
a recovered attempt counts as a failure (the regime path already charges attempt failures
since #783/#785, so set C reaches the loop without touching the rubric).

**Cut time, item 6 (recorded 2026-09-27).** amfs-internal #786 (migration 107,
`RepairSettings.auto_propose`; the gate sits in `enqueue_fix_propose` after the process
switch and the plan gate, so the incident and regime producers both honour it; manual
proposals unaffected) merged to `main` at 23:34:18Z as `1d0f5102`; deploy run 36279860901
green (deploy, Room lifecycle smoke, Notify SRE dispatch). Verified live at ~23:52Z: `GET
/agents/{id}/repair-settings` carries `auto_propose`. Harness commit `bf589c41` holds it on
every Tier-1-only arm from the identity's first session and records the result per
identity as `arm_notes.auto_propose`; repair-loop arms set it `true` explicitly. Cells
already running from older harness code (`rsi-c1-tier1` control cells, `rsi-oracle-5A`)
had the hold applied out of band at ~23:53Z and as their remaining identities appear
(`results/<run>/auto_propose_hold.jsonl`, one row per identity with the platform's answer).
Every control cell that ran before this cut is separated by the fix ledger (0 shipped,
corrective entries on branches), not by construction; every one after it is separated by
construction. Confirmatory runs start after this cut.

**Engine corrections from the 5k readout (PRs for approval, recorded 2026-09-27).**
amfs-internal #787 — canary rule 1b, `ungraded`: arms with `UNGRADED_MULTIPLE ×
min_outcomes` outcomes between them (the floor each), none graded, older than 15 min, end
inconclusive with a note naming what grading needed. amfs-internal #788 — the situation
guard on the regime path: where the sealed run declared a situation, an entry is a
candidate only if an earlier success of the same agent in that situation read it; set-aside
entries are reported as `first_exposure`. Neither changes what is measured; both change
whether the loop's one canary slot can be blocked by a fix to an entry nobody reads.
`rsi-c1-rsi` (the P2/P3 run: `senselab-rsi-nocanary` + `senselab-rsi`, set C) is held until
both are deployed; their cut times will be recorded here. *2026-09-27 09:5x JST:* at Bruno's
request both were combined into amfs-internal #789 for a single deployment; #787 and #788
closed as superseded (branches kept). #789 is the PR whose cut time gates `rsi-c1-rsi`.
**Cut time (recorded 2026-09-27).** #789 merged 00:56:43Z (`0713035b`); "Deploy to Google
Cloud" run 36284021830 green at 01:14:44Z. `rsi-c1-rsi` launched 01:17Z (8 cells, 800
episodes: `senselab-rsi-nocanary` + `senselab-rsi`, support + ci-fix, seeds 11/23, set C,
`--episodes 100 --change-at 30 --change-focus 0.5`, Tier 2 hub on 8787) — every session of
this run is on the post-#789 engine. `rsi-c1-tier1` finished 1200/1200 at 01:19Z; its two
late ci-fix control agents (`ci-agent-311372896`, `ci-agent-773578895`) received the
`auto_propose=false` hold at 01:19:26Z, after their cells had ended (the watcher had died
with its shell). Control-arm proposals before the holds: 3 and 2 on those two agents, 5 / 1
/ 5 on the three held support agents, 0 on the other three — all `proposed`, all on
`repair/*` branches, none tested, shipped or read by a control session (control reads
main). No proposal on any held agent was enqueued after its hold: the three fixes created
23:59:00–00:01:20Z on `support-agent-432812137` (held 23:58:24Z) came from a `fix_propose`
job queued 23:58:04Z, checked in `amfs_eval_jobs`. The separation is now by mechanism for
every control agent of the (l) runs; the pre-hold proposals are a cost line (repair-agent
calls on control identities), not a contamination path.

### 2026-09-27 (m) — the Managed Model is evaluated on tickets and logs it never trained on

**Why.** E7 trains two Managed Models (`senselab-s1`) on E1 `support` traces — one from
`senselab-rsi` cells, one from `senselab` cells — and evaluates them in decision mode on
held-out seeds 101/113/127. Held-out seeds change the schedule, the instances (order ids,
plans, platforms, issuers) and, under a change set, the answers; they do **not** change the
wording. `realworld.py` draws each support ticket from three fixed paraphrases per class
and each CI failure from one fixed log line per failure, so every eval prompt's text has
appeared in training. A reviewer will say the student was fine-tuned on the questions and
their answers, and on that evidence they would be right. The distillation study's set B
result (a student without memory 0.26 on changed support classes, with memory 0.50,
frontier with memory 0.90) already separates *memorised answers* from *reads at eval
time*, and its known-limits note concedes that eval scenarios share generators with
training. (m) makes the separation a design, not a concession, and adds the tests Bruno
asked for: the same evaluation, and then evaluations on similar but different tickets,
logs and classes, to see how the student *with SenseLab connected* responds when what it
could have memorised does not apply.

**What changes.**

1. **A contamination ladder for E7**, every rung on seeds 101/113/127, every rung run for
   both students and three conditions: **S0** student alone (no memory), **S1** student +
   SenseLab (the same protocol the teacher used; the memory is the eval cell's own,
   built by S1's traces under the Tier 1 loop — nothing from training cells is read),
   **F1** frontier + SenseLab (the bar).
   - **L0 — held-out seeds, training wording** (the E7 as preregistered). Reported; not the
     headline.
   - **L1 — held-out wording.** Eval-only message sets: three new paraphrases per support
     class (`msgs_eval`) and an eval-only log variant per CI failure (`log_eval`: a
     different file, line, tool phrasing or error text that still identifies the failure
     class as the fixed log does). Training cells never draw from these; eval cells draw
     only from these. Same classes, same answers.
   - **L2 — held-out answers.** Eval under a change set the student never saw (train on set
     A cells, evaluate on set B; set C where the family has one), with training wording.
     The distillation study's H8, now in the ladder.
   - **L3 — held-out classes.** Eval-only classes with their own conventions: three support
     classes and two CI failures that appear in no training cell (`eval_only=True` in the
     issue tables), mixed into the eval schedule at the scenario's usual class weights. The
     student cannot have memorised their answers; whatever S1 gains over S0 here it gained
     by reading memory at eval time.
   - **L4 — held-out family** (exploratory, reported, not claimed): the support-trained
     students evaluated on ci-fix. The distillation study declines this claim and (m) does
     not reverse that; the row is there so the reader sees the number rather than wonders.
2. **A memorisation diagnostic**, reported beside every student number: S0 on L0 minus S0
   on L1. If a student alone does well on training wording and falls on new wording, the
   difference is what it memorised, stated in points rather than argued.
3. **Endpoints per rung** (as E7): post-change first-attempt success on changed classes,
   first-attempt success on eval-only classes (L3), pinned-case reproduction (L0 only —
   pinned cases are training artefacts by definition), cost per solved task.
4. **What is claimed.** The E7 claim moves from L0 to **L1 and L3**: a Managed Model
   distilled from the repair record, with SenseLab connected, holds its advantage on
   tickets and logs it never trained on and on classes it never saw, and that advantage is
   attributable to reads (S1 − S0 on L3, where S0 has nothing to remember). The
   `senselab-rsi`-trained student against the `senselab`-trained student is compared on the
   same rungs; the attribution rule of (l) applies to what the difference is credited to.
5. **Registration.** `msgs_eval`, `log_eval` and the eval-only classes are written before
   any E1 cell runs and registered by the scenarios hash with set C; the E7 training
   manifest lists the training cells and asserts none drew from an eval-only set or class.
   "Exactly two Managed Models" stands.

**Predictions.** (P1) L0 → L1: S0 falls by more than S1 does (memorisation shows in the
student alone, and memory covers it). (P2) On L3, S1 > S0 for both students with the
gap credited to reads; F1 > S1 (the small model uses memory less well than the frontier,
as the distillation study found). (P3) The `senselab-rsi`-trained student ≥ the
`senselab`-trained student on L1–L3, the difference small and reported as such unless the
E1 repair record differs materially between the arms (the (l) attribution rule decides what
it is credited to). **What falsifies:** S1 ≈ S0 on L1 or L3 means the student is not
reading memory in decision mode and the E7 value was memorisation — reported as such; a
large L0 − L1 drop for S1 as well means the wording, not the memory, carried the E7
number.

**Scope note.** Development phase. No student has been trained; E1 has not run. (m) adds
eval-only content and an evaluation design; it changes no arm, no training recipe and no
scenario the teachers see. Bruno raised the concern on 2026-09-27 JST and asked for the
same tests plus tests on similar-but-different tickets, logs and issues; L0, L1 and L3 are
those.

### 2026-09-27 (n) — the canary is built around exposure; a fix is graded in the situation it failed in

**Why.** (k) and (l) found the same thing from two sides: a memory canary is not a deploy
canary. Routing a session to the fix's branch does not mean the session *read* the changed
entry, and a canary graded on reads of an entry nothing reads fills both arms and decides
nothing — 5k held the agent's one slot for 50 sessions on a README rewrite no run read,
graded 0/0. #789 made that canary end early (`ungraded`); it did not stop the next one
from taking the slot, and it did not give a rewrite that has stopped being read any way to
be graded at all. Bruno asked (2026-09-27 JST) for the four items below in one PR, the
fourth decided by us toward "a novel and solid RSI approach", and for the canary to be
visible on the agent's Behavior page.

**Engine changes (amfs-internal #790, open for approval; migration 108).**

1. *Exposure pre-check.* Before a canary starts, `recent_exposure` counts how many of the
   agent's last 50 served sessions the fix's grading would have read (its judge's verdict;
   any verdict; an outcome in its situation; an outcome on a run that read an implicated
   entry; any outcome). When the window holds at least `canary_min_outcomes` sessions and
   none could be graded, the fix is **held** with the reason on it (`canary_hold`) and does
   not take the slot; the hold is re-evaluated each sweep and cleared when a canary starts.
2. *Session window.* A canary now ends `inconclusive` (`window`) once
   `window_sessions` runs have been graded across both arms without a verdict —
   `max(2 × floor, setting or 10 × floor)` — with `canary_timeout_days` kept only as the
   wall-clock ceiling. The trial is measured in the agent's own traffic, not in days.
3. *Priority and preemption.* The queue is ordered by grading rank (judge, behaviour,
   situation, reads, outcomes), exposed before unexposed, `test_passed` before
   `test_inconclusive`, exposure rate, age. A running canary with nothing graded in either
   arm after 15 minutes yields the slot to a waiting fix the traffic can grade when that
   fix has a better grading rank or the running one's exposure is 0; it stops `preempted`
   and is the one ended canary the policy queues again.
4. *Situation-conditioned grading — the decision.* A fix proposed by the regime path
   carries the folded `situation` its failing runs declared (the harness has declared
   `attributes.situation` since (h)). Its canary is graded on **every run in that
   situation**, whatever the run read: the population is the one the failure was observed
   in, not the runs whose retrieval happened to surface the entry. A rewrite the agent has
   stopped reading is then graded as *no effect* (same recurrence in both arms) instead of
   0/0, and a rewrite that is read and works shows as lower recurrence on the branch. No
   new LLM judge: the signal is the outcome the agent already reports, so this costs
   nothing per run and cannot be gamed by the wording of the fix. The alternative — a
   fix-scoped judge asking "did this run exhibit the failure the fix addresses?" — is
   deferred as a follow-up; it would add a per-run judge call and its own calibration
   question. `GET /agents/{id}/canary` now carries `active_exposure` and the ordered
   `queue` with each fix's grading mode, exposure and hold; the dashboard renders it as a
   panel on the Fixes tab and, compact, on the Overview.

**What this changes for the runs.** Nothing already running: `rsi-c1-rsi` (launched
01:17Z) is on the post-#789, pre-#790 engine end to end, and is analysed as such. Once
#790 is merged and deployed the cut time is recorded here; any `senselab-rsi` cell started
after it is on the (n) engine, and E1 is. The predictions of (l) stand; (n) changes how
fast the loop can reach a verdict on a regime fix (P2/P3 latency, time-to-repair), not what
counts as one. **Prediction added:** on set C, a `senselab-rsi` cell whose regime fix is
proposed for a situation the agent keeps meeting reaches a canary verdict (pass, fail or
`window`) within the cell under (n), where under (k)/(l) it ended `ungraded` or ran out
the clock; *what falsifies* — a regime fix in a recurring situation still ending
`ungraded`, or being held for exposure while runs in its situation are being sealed.

**Cut time (recorded 2026-09-27).** #790 merged 02:22:31Z (`c4949443`); "Deploy to Google
Cloud" run 36288330859: the `deploy` job finished 02:33:07Z (Cloud Run `amfs-api-02035-cdm`
ready 02:31:27Z, `pro-api-00778-mvr`), the run green at 02:34:34Z. **The cut is 02:34:34Z**,
by the convention used for #789. *Correction to the paragraph above:* `rsi-c1-rsi` is not
pre-#790 end to end. Its cells run in sequence, and at the cut the support-`s11` cells had
finished (100/100 each, all sessions pre-#790), the support-`s23` cells were mid-run —
`senselab-rsi-nocanary` s23 at ep 23 (episodes 0–22 pre, 23–99 post), `senselab-rsi` s23 at
ep 3 (0–2 pre, 3–99 post) — and the four ci-fix cells had not started, so they are on the
(n) engine end to end. **Analysis rule, fixed now:** engine version is a per-episode
covariate (`ts < 02:34:34Z` → `pre-n`, else `post-n`); the (l) P2/P3 verdict counts are
reported per cell with the split stated, and any latency claim (time to a canary verdict,
time-to-repair) is made only within one engine version — never across the cut. The s23
support cells are the one pair where a regime fix proposed pre-cut could be canaried
post-cut; if either has such a fix, it is reported as a straddling case, not counted for
or against the (n) prediction. The ci-fix cells are the clean (n) cells of this run.

### 2026-09-27 (o) — one system in every result; seams 4 and 5 as engine changes; the E7 stand-in

**Why.** Bruno (2026-09-27 JST): SenseLab and its recursive self-improvement are one system —
RSI is *what SenseLab delivers* — so the paper never shows "SenseLab vs SenseLab-RSI" as two
results; the RSI features show up as SenseLab doing better than every other approach on the
scenarios only a self-improving system can handle. (l) already fixed this mapping; the P2/P3
readout of `rsi-c1-rsi` in HEADROOM_RSI.md, and messages about it, presented `senselab-rsi`
against `senselab` as competing systems. That was a presentation error, and this amendment
makes the rule binding on every readout from now on. Nothing about the measurement changes.

**Presentation rule (binding).**

1. **One system.** *SenseLab* in every results table, figure and sentence is the full engine
   — harness arm `senselab-rsi`. It is compared with frontier alone (`none`), `pgvector`,
   `pgvector+rerank`, `pgvector-diy+outcomes`, `context-window`, and the maintainer oracle as
   the upper bound. Mem0 and Zep are not run — the Arms section has said so since v1 (appendix:
   gridv4 data), and the Mem0 plan has no retrieval budget; they are not added back.
2. **Ablations in one place.** The rows "SenseLab − repair loop" (`senselab`), "− canary"
   (`senselab-rsi-nocanary`), "− replay", "(review)", "− protocol", "(judges only)" appear in
   exactly one table — the attribution table that credits gains to tiers by the (l) attribution
   rule — and nowhere else. They are never a column beside SenseLab in a headline table and
   never described as a system or a competitor.
3. **No harness names in results.** `senselab`, `senselab-rsi`, … appear only in this document,
   the reproducibility appendix (command lines, cell ids) and code. Results text says
   "SenseLab", "SenseLab without the repair loop", "the retrieval baselines".
4. **Readouts follow the rule.** HEADROOM readouts from this amendment on are written this
   way. The `rsi-c1-rsi` P2/P3 readout is left as the record and re-stated under the rule below
   it, with one correction found while re-stating it (the table labelled "moved half, n = 35"
   was all changed-class post-change episodes; the moved half is n = 13–23 per cell; and "kept
   half 94–100 %" was the *unchanged* classes). The numbers themselves are unchanged.

**Seam 4 — engine change, deployed (amfs-internal #793).** *What:* a fix's case set now
includes the runs its manifest predicts it fixes (`FixService.evidence_trace_ids`), the
customer replay re-attaches the suite when the set is empty, and a case without a judge is
graded on the outcome the replayed run reports (previously it passed on nothing). *Why:*
regime fixes — the fixes set C is built to produce — had empty case sets (Tier 1 tests the
manifest, the case set came from verdicts and behaviours only), so Tier 2 could not be asked
(`replay_pending: the fix's case set has no cases`) and the `-nocanary` policy, which ships on
a passed replay, had nothing to ship on: in `rsi-c1-rsi` all four ci-fix regime fixes ended
the cell `test_passed`, unshipped. **Cut:** merged 06:50:12Z (`9f8c7b55`); "Deploy to Google
Cloud" run 36301272477 green at 07:00:22Z. **The cut is 07:00:22Z.** *Prediction (P6, dev
seeds, set C):* every regime fix that passes Tier 1 receives a Tier 2 verdict (pass or fail)
inside its cell; `replay_pending: … no cases` does not appear on any regime fix; under
`-nocanary` a regime fix whose replay passes ships in the same cell. (l) P2 is unchanged and
is now testable. *What falsifies:* a regime fix that passed Tier 1 ending its cell without a
Tier 2 verdict; or a replay passing a fix whose replayed runs reported failure.

**Seam 5 — engine change, PR open (amfs-internal #794).** *What:* server-side situation
identity. When a trace seals with `attributes.situation`, the label is folded and resolved to
the agent's canonical situation — the first-seen spelling among the agent's canonicals whose
stemmed tokens it shares (Jaccard ≥ 0.5, or ≥ ⅔ of the shorter label's tokens with ≥ 2 shared)
— and the pair is stored in `amfs_eval_situation_aliases`; the regime guard, the canary's
situation grading and the exposure check compare canonicals (alias joined at read time, traces
immutable, fall-back to the spelling). Deterministic and stated; no model call. *Why:* the (n)
falsifier fired in `rsi-c1-rsi` — canary `9ca1e6b5` graded 2 of 15 runs in a recurring
situation because the agent spelled it ten ways. *Prediction (P7, dev seeds, set C):* for a
situated canary, of the post-start runs whose episode the harness tags with the fix's class
(`change_class`, the benchmark's ground truth), ≥ 80 % are graded (recall), and ≥ 90 % of the
graded runs are in that class (purity). This is the acceptance check of the engine change,
computed from tags the runner already writes — not a benchmark endpoint. The (n) prediction is
re-stated on resolved situations: a regime fix in a recurring situation reaches a canary
verdict within the cell. *What falsifies:* recall < 50 % (the rule does not capture the
drift) or purity < 70 % (the rule merges classes). Cut time recorded here when #794 deploys.

**Run plan.** `rsi-c2` after the #794 cut: `senselab-rsi` and `senselab-rsi-nocanary`, set C,
support + ci-fix, seeds 11 / 23, 100 episodes, change at 30, focus 0.5, Tier 2 on 8787 — the
`rsi-c1-rsi` command unchanged. Paired with the `rsi-c1-tier1` cells (`none`,
`pgvector+rerank`, `senselab`) and `rsi-c1-baselines` (`pgvector`, `pgvector-diy+outcomes`,
`context-window`; running) on the same protocol and seeds. **Per-cell K1–K4 with the engine
version as a covariate**: `pre-n` (< 02:34:34Z), `post-n`, `post-seam4` (≥ 07:00:22Z),
`post-seam5` (≥ the #794 cut); latency claims within one version only, as (n) fixed.

**E7 — a disclosed stand-in, not a change to (m).** (m) trains the two Managed Models on E1
traces; E1 has not run. Bruno asked for the Managed Models to be built now, so two development
students are training on the production SaaS from the dev-seed set-C support traces:
`sup-senselab-c1` (SenseLab — the `rsi-c1-rsi` `senselab-rsi` cells, 200 sealed traces, family
`senselab-s1`, serving `context_k = 3`, priors, server retrieval, no fallback) went live as v1
at 07:00Z under the relaxed gate — the default gate would have failed it on
`exact_agreement 0.000 < 0.45`, which is stated wherever it is reported — and `sup-norepair-c1`
(the ablation — `rsi-c1-tier1` `senselab` cells) is training. They are evaluated at L0 on
seeds 101 / 113 / 127 (S0 student alone, S1 student + SenseLab, F1 frontier + SenseLab) and
reported as development students. The confirmatory E7 students remain the ones (m) specifies,
trained on E1 traces after the freeze.

**What does not change.** No arm, scenario, change set, rubric or metric is added or altered;
the seeds, the dev / confirmatory split, the freeze rule, and E1–E7 as specified stand.
