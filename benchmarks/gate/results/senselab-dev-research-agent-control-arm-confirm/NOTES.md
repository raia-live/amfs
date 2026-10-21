# senselab:research-agent — third run, dev deployment, after the control arm's sampling change

Same account, agent and mode as the two runs before it
(`../senselab-dev-research-agent/`, `../senselab-dev-research-agent-control-arm/`).
Pro `dev` at `995cc6de` (amfs-internal#813: a failing control is sampled
once more before it counts, a pass needs more runs failing without the
change than passing either way, and the control cache is keyed by the
sampling policy). Run 17:51–17:57 JST (08:51–08:57 UTC), 2026-10-17.

## Why a third run

#813 changes what a control is — one sample became "one, and a second if
the first fails" — so controls cached under the old rule had to stop
answering for the new one. This run checks that on the deployment, and
adds three more draws of the stand-in on the runs the first run was wrong
about.

## What it found

**Verdict `insufficient`, as before: 0 of 6 probes decided, all six
`undecided` at Tier 1.** All six were drafted from `Fix: Retry storm`
again.

- The first probe's test shows `cached: 0` and `cost_usd: 0.0091`: the
  three controls recorded under the one-sample rule missed, as the new
  fingerprint intends, and were sampled afresh. Every probe after it reads
  `cached: 3, cost_usd: 0`.
- All three fresh controls passed. Six draws of the stand-in (`gpt-5`) on
  these runs without any change — three in the second run, three here —
  and six passes. No flub observed; the confirming sample guards against a
  rate this sample is too small to measure, not one it has seen.
- No control failed, so no confirming sample was taken. The path this
  change adds (`failed sample 1 and passed sample 2`, or `confirmed by
  sample 2` on the case note) is exercised by the unit tests and not by
  this corpus, whose runs the record cannot fail.
- One treated replay (probe 5, run `ba11a43b`) ended unreplayable: the
  stand-in called `search_kb`, which the record holds no output for. The
  treated arm is as stochastic as the control arm; the case was not a
  pass and its control was not asked (`asked: 2` on that probe).

A/A and reward: `insufficient` and empty, unchanged.

## What follows

Unchanged from the second run: a decided number needs a run the record
can fail. A planted failure whose cause is in what the agent *read*
(a wrong memory entry the judge can see in the answer) would give the
gate a run that fails without the fix and passes with it — a placebo
rejected, a genuine fix passed with its controls confirmed — and would
exercise on a deployment the path only the unit tests exercise now.
