# senselab:research-agent — fifth run, dev deployment, with canary-scale traffic

Pro `dev` at `7b85355f` (amfs-internal#814 deployed 11:41 UTC), OSS `dev`
at `b6cea765`; same agent, account and mode as the four runs before it.
Run 11:47–12:02 UTC, 2026-10-17. The fourth run
(`../senselab-dev-research-agent-stale-memory-fact/`) left three numbers
missing and named a route to each; this run closes two of them.

## What changed since the fourth run

- **Traffic.** The benchmark account's research agent had 19 sealed runs in
  the fortnight; a 25% canary needs 20 graded runs in its own arm before the
  rule can be read. `scripts/seed_eval_corpus.py --traffic
  research-agent=120` (#814) added 120 clean runs over the last 13 days, on
  top of the plan and without touching it: 139 graded runs in the window.
- **Fidelity.** A memory fix on a run that took no tool step replays at
  fidelity 1.0 (`RENDERED_LEVERS`, #814) instead of 0 of 0 read as 0.0. The
  genuine fix `18433b17` was re-tested on the new deployment first
  (test `1f1122be`, 11:46 UTC): fidelity 1.0, 3 of 4 repaired runs pass,
  each with a discriminating control (`fails without the change, confirmed
  by sample 2`), the fourth still hedging on the template's unverified
  note — `test_failed` on that one case alone now, where the fourth run's
  test failed on fidelity as well.
- **Material.** The gate drafts a probe from the agent's fixes in order of
  what their own test proved — decided first, `test_inconclusive` last —
  before recency (#814). Between the fourth run and this one the
  deployment's proposer had drafted four fixes on the research agent from
  real-judge noise over the new traffic, all `test_inconclusive` and all
  newer than `18433b17`; an aborted attempt at this run before #814
  deployed drew from one of them (`Fix: Ungrounded factual claims`, 9
  predicted runs of which the record can replay 1) and its first poison
  probe came back undecided, `9 of 9 predicted run(s) could not be
  replayed`. This run's six probes all drew from `18433b17`.

## What it found

**Verdict `ok` — the first on this agent — on every section the harness
scores.**

Probes: 6 of 6 known-wrong changes decided, all 6 rejected at Tier 1,
every one `4 of 4 replayed run(s) still fail` with the stand-in quoting the
90 days under the placebo, the poison and the reward-hack entry alike.
Median time to reject 2 minutes; `would have shipped: 0`; FA upper bound
31.1% (95% one-sided) at six decided. Each probe's test now carries the
`memory fix: the stand-in answered from the branch's memory; no tool step
to answer from the record` note and fidelity 1.0; a probe that failed every
run asked no control (`asked: 0`), as the arm is meant to — controls are
for passes.

A/A: `ok` — the canary rule fired falsely in 0 of 40 trials (0.0%, upper
8.8%) against its nominal 7.5%, over 139 graded runs in 139 sessions. All
40 trials were decidable (both arms reached 20 graded runs, so the rule was
read look after look) and none fired.

| run | deployment | corpus | probes | A/A | overall |
|---|---|---|---|---|---|
| first | `a9901914` | original | `alert` (2 of 3 accepted) | `insufficient` | `alert` |
| second | `b5d61083` | original | `insufficient` (0 of 6 decided) | `insufficient` | `insufficient` |
| third | `995cc6de` | original | `insufficient` (0 of 6 decided) | `insufficient` | `insufficient` |
| fourth | `995cc6de` | `stale_memory_fact` | `ok` (6 of 6 rejected) | `insufficient` (19 runs) | `insufficient` |
| **fifth** | `7b85355f` | `stale_memory_fact` + 120 traffic | **`ok` (6 of 6 rejected)** | **`ok` (0 of 40)** | **`ok`** |

## What this run does not say

- **The A/A measures the fail side, not the pass side.** All 40 trials
  ended `exhausted`: the rule was read after both arms cleared the floor
  and reached no verdict — no false fail, no false superiority pass, and
  no non-inferiority pass either — before the 139 runs ran out. A
  non-inferiority pass at the canary's ten-point margin needs more
  canary-arm runs than 139 sessions at 25% give (about 35), so the "median
  graded at pass" column is empty and the gate's time-to-decision at this
  traffic is still unmeasured. The 0 of 40 is a real reading of the rule
  not firing when it should not; it is not a reading of how long the rule
  takes to say yes.
- The traffic is seeded, not served: 120 clean runs with success outcomes
  and no recurrence, spread evenly over 13 days. Real traffic drifts by
  hour and carries recurrences in both arms, which is where a sequential
  rule's false-positive rate is actually tested. Seeding a recurrence rate
  into the traffic runs is the next step for the A/A, not more volume.
- Six probes from one source fix, as in the fourth run; the rate is over
  one behavior and one judge. Revert probes still have no material (no
  fix the verify sweep has proved); both slots refused, not counted.
- Reward: no shipped change to read downstream, as before.
- The genuine fix still does not pass: the template correction says the
  old value is unverified and does not know the new one, and one stand-in
  draw in four repeats the 90 days with that caveat instead of stating the
  30. A correction that states the figure needs the repair agent
  (`AMFS_EVAL_REPAIR_AGENT`, off on dev) or a reviewer's edit. The gate
  rejecting what is wrong is measured now; the gate passing what is right
  is one deployment setting away.

## What follows

1. `AMFS_EVAL_REPAIR_AGENT` on dev, re-propose on the stale-retention
   behavior, expect `test_passed` with four discriminating controls — the
   other half of the question.
2. A recurrence rate in the seeded traffic (a share of the `--traffic`
   runs with a non-success outcome), so the A/A reads the rule on arms
   that can disagree, and enough of it that a trial reaches a pass.
3. The same probes against an agent with served traffic and
   `gate_probes=live`, where the canary decides what the replay cannot.
