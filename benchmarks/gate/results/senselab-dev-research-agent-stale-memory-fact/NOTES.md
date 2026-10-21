# senselab:research-agent — fourth run, dev deployment, on a corpus with a run the record can fail

Same deployment as the third run (Pro `dev` at `995cc6de`, amfs-internal#813;
OSS `dev` at `b6cea765`), same agent, same mode (`dry_run`, probes close at
Tier 1). Run 10:05–10:22 UTC, 2026-10-17. What changed is the corpus: the
benchmark account was re-seeded (`--reset`) with a planted failure whose
cause is in what the agent *read*, which the three runs before this one
said was the number worth having (raia-live/amfs#473).

## The planted failure

`scripts/seed_eval_corpus.py` (amfs-internal#814) adds `stale_memory_fact`:
four `research-agent` runs read the memory entry
`research/policies/data-retention` ("Raw trace data is retained for 90 days;
aggregated metrics for 13 months"), take no tool step, and answer the
retention review with the 90 days. An authored judge, `Stale retention
figure`, carries the policy of record (30 days) in its prompt and fails any
other figure, quoting the sentence. The failure reproduces under replay
because nothing about it is in a tool output: the stand-in reads the same
entry and gives the same figure, and a fix that corrects the entry is the
only thing that changes the answer.

On the re-seeded account the deployment discovered the behavior
(`Stale retention figure: tells team raw`, 4 runs) and drafted the fix
through `POST /fixes/propose` — `18433b17`, lever `knowledge`, a correction
of the entry. The repair agent is off on dev (`AMFS_EVAL_REPAIR_AGENT` not
set), so the correction is the template's: the stale value carried forward
with "Treat this value as unverified until reviewed", confidence × 0.6, next
to a risk entry drafted from the judge's evidence — which names the 30 days.

## What it found

**Probes `ok`: 6 of 6 known-wrong changes decided, all 6 rejected at
Tier 1.** Every constructed probe — two placebos, two poisons, two reward
hacks, all drafted from `18433b17` — ends `4 of 4 replayed run(s) still
fail`: under a note that recommends nothing, under an entry endorsing the
90 days, under an entry that tells the agent to report success whatever
happened, the stand-in keeps quoting the stale figure and the judge keeps
failing it. Median time to reject 1–3 minutes; `would have shipped: 0`
(policy `review`). The FA upper bound at six decided probes is 31.1%
(95% one-sided) — a rate, finally, rather than a refusal to give one.

| run | deployment | corpus | probes decided | accepted | probes verdict |
|---|---|---|---:|---:|---|
| first (`../senselab-dev-research-agent/`) | `a9901914` | original | 3 of 6 | 2 | `alert` |
| second (`../senselab-dev-research-agent-control-arm/`) | `b5d61083` | original | 0 of 6 | 0 | `insufficient` |
| third (raia-live/amfs#473) | `995cc6de` | original | 0 of 6 | 0 | `insufficient` |
| **this run** | `995cc6de` | `stale_memory_fact` | **6 of 6** | **0** | **`ok`** |

**The genuine fix, on the same deployment:** `18433b17`'s own Tier 1 test
(`f40de03f`, run automatically on propose, 10:08 UTC) reads 3 of 4 repaired
runs passing, each with a **discriminating control** — the run fails without
the change, `confirmed by sample 2` — and one still failing. That is the
path #813 added and only the unit tests had exercised until now: three
controls failed their first sample, three confirming samples were taken,
three agreed. Control cost $0.0052 for six replays; the test itself
$0.0131.

The one failing case (`e6ec3552`): the stand-in hedged — "the previously
cited figure of 90 days ... is flagged as unverified/stale and may be
incorrect" — without stating the 30 days, and the judge fails any figure
but 30. That is what a template correction can do: it says the old value is
unverified, it does not know the new one. A correction that states the
30 days needs the repair agent (or a reviewer's edit), which is a
deployment setting, not a gate property.

The test's **fidelity is 0.0**, so its status is `test_failed` on two
counts, and would have been `inconclusive` on fidelity alone had all four
passed. The stand-in took no tool step, and the deployed replay counts
fidelity over tool steps answered from the record — 0 of 0, read as 0.0,
under the 0.5 floor. Only harness levers were exempt. amfs-internal#814
exempts every lever the stand-in is rendered with (`RENDERED_LEVERS`:
knowledge, procedure, unknown, plus the harness levers; `code` keeps 0), so
a memory fix on a tool-less run replays at fidelity 1.0 — the unit test
asserts the passed verdict and the `discriminating` flag on every case.
Not yet deployed when this run was made.

## What this run does not say

- **Overall verdict is still `insufficient`**, now entirely from the A/A
  section: `0 of 40 trials reached 20 graded runs in both arms over 19
  graded runs in 19 sessions; a canary at 25% cannot decide at this
  traffic`. The research agent has 19 sealed runs in the fortnight; a 25%
  canary needs 20 graded runs in its own arm, so about a hundred a
  fortnight before the rule can be read at all. The harness ranks an
  unmeasured false-positive rate above a clean probe card, which is the
  right order — but the finding is about the account's traffic, not about
  what the gate decided. Reward: no shipped change to read.
- Revert probes: both slots refused, as before — no fix the verify sweep
  has proved. Not counted.
- Six probes from one source fix. The rate is over one behavior and one
  judge; the first run's other source (`Fix: Ungrounded external claims`)
  was not drawn because `18433b17` is the newest fix with a prediction.
- The source fix's own test is not a probe and is not on the card; it is
  reported here because it is the other half of the question — the gate
  rejects what is wrong, does it pass what is right — and because fidelity
  made its verdict wrong on the deployment this run used.

## What follows

Three numbers are missing and each has a known route:

1. The genuine fix at fidelity 1.0 — re-test `18433b17` once #814 is on
   dev; expect `3 of 4` with the same controls, status decided by the
   hedged case alone.
2. A correction that states the 30 days — `AMFS_EVAL_REPAIR_AGENT` on dev
   (an LLM cost the deployment's owner decides), then re-propose and
   expect `test_passed` with four discriminating controls.
3. An A/A that decides — the benchmark agent needs canary-scale traffic
   (≈100 graded runs in the window); the seed's clean runs are the place
   to add it, and nothing about the probes changes.
