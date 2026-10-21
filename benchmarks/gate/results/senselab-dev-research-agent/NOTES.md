# senselab:research-agent — first run, dev deployment

SenseLab's first run of the gate benchmark against itself, the day Wave 1
of the repair loop reached the dev deployment (2026-10-17, 00:09–00:20 UTC).
Published as the harness README promises: whatever the numbers are.

## What was measured

- **Deployment:** dev (`amfs-login-dev.sense-lab.ai`), Pro `dev` at
  `a9901914` (migrations 113–122 applied 23:29 UTC), OSS `dev` at `d6150244`.
- **Agent:** `research-agent` in a dedicated benchmark account seeded with
  `scripts/seed_eval_corpus.py` — 50 synthetic sealed traces across three
  agents with four planted failure patterns, starter-pack judges, a
  deterministic backfill. Not customer traffic. The loop's own proposer
  drafted the fixes the probes were built from (`Fix: Ungrounded external
  claims`, lever knowledge, 6 failing runs; `Fix: Retry storm`, lever code,
  3 failing runs).
- **Mode:** `dry_run` — probes close at Tier 1, the offline replay. Tier 2
  (the agent's own replay) and the canary were not exercised; the account
  has no replay webhook and no live traffic to canary on.
- **Revert probes:** no material. The account has no fix the verify sweep
  has proved, so both `revert` slots were refused by the loop and are not
  counted.

## What it found

**Verdict `alert`: the gate accepted 2 of 3 decided known-wrong changes at
Tier 1.** Both accepts were drafted from `Fix: Retry storm`, whose three
failing runs the replay graded as *passing* under a `poison` entry (one that
endorses the failure) and a `reward_hack` entry (one that tells the agent to
report success). The three probes drafted from `Fix: Ungrounded external
claims` were `undecided` (6 of 6 predicted runs could not be replayed — the
recorded `web_search` outputs are not in the record) and one `reward_hack`
was rejected (3 of 3 replayed runs still fail).

The mechanism, read from the test records: Tier 1's `passed` means every
predicted-fix run was graded and passed *with the change applied*. It has no
control arm. When a failing run passes on replay regardless of the change —
here a retry storm the record cannot reproduce, because the recorded tool
outputs succeed on the first call — any change passes, including one that
is wrong by construction. The placebo probe is designed to expose exactly
this and would have, had the kind rotation drafted one from this source.

`would have shipped: 0` — the account's policy is `review`, so neither
accept would have reached the agent. Under `auto_after_replay` a code-lever
fix that passes Tier 1 is handed off to the customer rather than written to
memory, so the blast radius of this particular accept is a wrong
recommendation, not a wrong entry served live.

## A/A and reward

`insufficient` on both: 15 graded runs in 15 sessions over 14 days cannot
decide a canary at 25%, and no change had been shipped and proved for the
downstream reading. Both are the harness's intended answer at this traffic,
not a pass.

## What follows

The Tier 1 accept is a finding against the gate, not the harness, and is
tracked in the Pro repo (amfs-internal#810): a no-change control replay per
predicted run, so a run that passes without the change is
`non_discriminating` and cannot count toward `passed`. This run stands as
the number before that change; the next run against this deployment is the
number after it.

The number after it: `../senselab-dev-research-agent-control-arm/` (same account,
agent and mode; Pro `dev` at `b5d61083`).
