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
