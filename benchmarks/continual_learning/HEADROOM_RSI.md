# Headroom map: where SenseLab loses episodes, and what closes the gap

Written 2026-09-26 from the v7 grids (`gridv4-core-sl`, `gridv5-core-sl`, `gridv5c-core-sl`,
fleet 1, GPT-5.5 and Claude Opus 5; 9,480 SenseLab episodes) and pilot 4 of the RSI paper
(`rsi-pilot-4`, current engine, 65 episodes at the time of writing). Every number below is
reproducible from `results/` with the snippets in `analysis/live.py` and the queries kept in
the paper's analysis code. This is an engineering map, not a result: it names the
mechanisms that cost episodes today and the change that should recover each, with the
prediction each change makes. It exists so the product can be improved **during the
development phase** and frozen before the confirmatory runs (amendment (g) of
`HYPOTHESIS_RSI.md`).

## Where SenseLab stands (v7 engine, set A, fleet 1)

| family | GPT-5.5 | Claude Opus 5 | best baseline | headroom to 94 % |
|---|---|---|---|---|
| analytics | 100.0 | 99.2 | 100.0 (pgvector) | none |
| concierge | 99.9 | 95.8 | 100.0 | none / 1.8 pts (Claude) |
| order-ops | 99.3 | – | 99.2 | none |
| retention | 90.0 | 91.3 | 94.7 / 95.8 (none) | 3–4 pts, all in episodes 0–19 |
| ci-fix | 89.1 | 87.4 | 86.4 / 84.7 | 5–7 pts |
| support | 81.9 | 84.3 | 73.9 / 79.7 | 10–12 pts |
| **all six** | **91.5** | **90.6** | | |

Two families carry the whole gap. On the classes a change does not touch, SenseLab is at
98.9 % (support) and 100 % (ci-fix) after episode 20. **Every point of headroom is
post-change recovery of the changed classes.**

Changed-class success by episode window (fleet 1; the change is at episode 20):

| family / arm | 0–20 | 20–25 | 25–30 | 30–40 | 40–50 | 50–60 |
|---|---|---|---|---|---|---|
| support · senselab | 95.5 | 25.6 | 46.7 | 38.0 | 73.6 | 80.6 |
| support · pgvector-diy+outcomes | 98.9 | 19.2 | 46.7 | 26.6 | 46.6 | 33.9 |
| support · none | 100 | 0.0 | 3.3 | 1.6 | 1.7 | 4.8 |
| ci-fix · senselab | 96.1 | 71.2 | 64.0 | 65.1 | 65.8 | 63.8 |
| ci-fix · pgvector-diy+outcomes | 95.2 | 72.2 | 61.8 | 57.6 | 51.9 | 51.9 |

SenseLab recovers in `support` — in 20–30 episodes, to an 80 % plateau — and in `ci-fix`
it does not recover at all (65 % plateau; 6 of 12 cells under 70 % in episodes 40–60).
Per changed class, that is 5–6 failed episodes in `support` and effectively permanent
failure for some classes in `ci-fix`.

## The mechanics of a slow recovery (one class, one cell)

`support` × GPT-5.5 × seed 11, class `card-declined`; the change at 20 makes `resend_email`
the right resolution:

```
ep20 F  escalate_tier2, advise_workaround, update_payment_method   rec = act / escalate_tier2
ep28 F  advise_settings_change, escalate_tier2, refund              rec = (none)
ep30 F  reset_password, explain_and_close, escalate_tier2           rec = explore / reset_password
ep31 F  clear_app_data, reinstall_app, explain_and_close            rec = (none)
ep32 F  clear_app_data, reinstall_app, reset_password               rec = explore / clear_app_data
ep42 OK resend_email                                                rec = explore / resend_email
```

Fifteen attempts over five episodes; `escalate_tier2` and `explain_and_close` tried three
times each, `clear_app_data`, `reinstall_app`, `reset_password` twice each. Twelve
candidate actions should cost at most four episodes if every attempt were a new one.

Across all post-change changed-class episodes (support + ci-fix, v7 grids):

- **39 % / 29 %** of attempts (support / ci-fix) re-tried an action that had already failed
  for the same situation since the change.
- **26 %** of support episodes received **no recommendation at all** (`priors` block
  absent); those succeed at 45 % against 61 % with one.
- The first-attempt recommendation after the change was still **`act`** in 400 / 720
  support episodes and 603 / 700 ci-fix episodes — acting on the stale rule.
- In ci-fix the agent **followed** the recommendation on only 41 % of first attempts
  (support: 78 %); its off-recommendation choice was `rerun_job` 117 times — the generic
  action.
- `stale_in_context` (a served, non-avoid entry naming the pre-change answer) is 40 % among
  support failures vs 11 % among passes; 72 % vs 15 % in ci-fix.

Pilot 4 (current engine, `REGIME_TURN_STREAK = 2` in `amfs_core.actions`, 14 post-change
changed-class episodes so far) shows the same shape at small n: first-attempt mode `act`
10 / 14, none 3 / 14, explore 1 / 14; 7 / 33 attempts re-tried a failed action; seed 23's
`android-notifications` failed at ep 21 and ep 29 both under `act`, `export-timeout` failed
ep 23–25 (none / act / none) before ep 26 recovered.

## Opportunities, ranked by expected episodes recovered

Predictions are for `support` and `ci-fix` changed-class post-change success (currently
≈ 55 % and ≈ 65 % over episodes 20–60). "Where" says which component changes; every engine
change is a PR for Bruno's approval and is frozen at `analysis-freeze-rsi`.

1. **Turn to `explore` after the first post-change loss when the scope has already
   turned.** `change_focus = 0.5` moves half the classes at once, so the first class that
   turns is evidence for every other validated action in the scope. Today each class pays
   `REGIME_TURN_STREAK = 2` losses (six attempts) before the recommendation stops saying
   `act`, and `RECENT_FAIL_STREAK = 3` before the lifetime record stops counting. A
   scope-level change-point (≥ 1 validated action turned in the last *k* sessions ⇒ one
   loss suffices for the others; and within an episode, a failed `act` attempt ⇒ the
   next attempt is `explore`) cuts the cost to one failed episode per class.
   *Where:* `amfs_core.actions` (recommendation), served through `retrieve`.
   *Predicts:* post-change changed-class success ≥ 85 % in both families; recovery within
   4 episodes of the change for every class.

2. **Never re-suggest, never re-try.** `explore` must exclude every action that has lost
   for this situation since the regime turned (not only the newest three takes, which is
   all `last_3` keeps), across attempts and across episodes; the harness agent, on its
   2nd/3rd attempt, must treat the `untried` list as the candidate set. The 39 % / 29 %
   wasted attempts are the direct cost.
   *Where:* `amfs_core.actions` (`untried` from the full post-turn record, not `last_3`);
   `agent/loop.py` (attempt 2–3 candidate set = `untried`).
   *Predicts:* re-try rate < 5 %; worst-case recovery = ⌈|candidates| / R⌉ episodes.

3. **A recommendation every time.** Find why 26 % of episodes get none (server error or
   time-out under load, priors skipped when the situation has too few outcomes, a
   compact-mode omission) and make the block unconditional — `explore` over all
   candidates when there is no record is still better than nothing.
   *Where:* SaaS `retrieve` priors path; `pro_client`/SDK fallback.
   *Predicts:* none-rate < 2 %; the 45 % → 61 % gap between the two groups closes.

4. **Stale rule out of the context.** A validated entry that has turned should be served
   as `superseded` (with `replaced_by` once a successor exists) — not as a normal hit the
   model may follow — and the successor written after the first post-change success
   should outrank it. This is where the **repair loop** does what the fast loop cannot:
   the fix rewrites the entry itself, on a branch, tested, canaried and shipped to every
   member of the fleet, so no session sees the old rule again.
   *Where:* `amfs_core.evidence` (serving status), repair loop (lever `knowledge`,
   corrective = the successor rule carrying the new action from the passing sibling).
   *Predicts:* `stale_in_context` among failures falls from 40 % / 72 % to < 10 %;
   fleet-6 and fleet-10 cells recover in the same number of *scope* episodes as fleet 1
   (one member's discovery is the fleet's).

5. **Repair proposals within a few episodes of the change, drafted by the agent.** In
   pilot 4 the first proposals came 8–10 episodes after the change (incident threshold
   "3 runs today vs 0/day") and both were template-authored because the repair agent's
   90 s budget expired on its first gpt-5 call (`AMFS_EVAL_REPAIR_TIMEOUT_S`, PR in
   progress); one was classified lever `code` and stayed `proposed`. The regime hook
   (`fix_propose` on a validated entry's failure) should fire on the first turned entry,
   the agent must get a realistic budget, and lever `code` on a memory-only benchmark is a
   misclassification to fix in `classify_lever`.
   *Where:* `amfs_eval.regime`, `amfs_eval.repair.agent`, `amfs_eval.fixes.classify_lever`.
   *Predicts:* per cell, first proposal ≤ 3 episodes after the change, `source =
   repair_agent` ≥ 80 %, lever ∈ {knowledge, procedure} for every proposal; time-to-repair
   (change → ship) ≤ 15 episodes.

6. **Follow the recommendation.** In ci-fix the model overrode the recommendation 59 % of
   the time, mostly for `rerun_job`. The harness prompt renders the block; make its
   contract explicit (first attempt = `suggested_action` unless the task text contradicts
   it; report `followed_recommendation` per attempt) and measure. This is how the product
   tells customers to use it, so it is a fair change, and it is applied identically to
   every SenseLab arm.
   *Where:* `agent/loop.py`, `agent/clients.py` prompt. *Predicts:* follow rate ≥ 85 %.

7. **No lost episodes under load.** A `save_trace` that times out client-side is a
   sealed trace the harness cannot see, a verdict wait that starts late, and — when the
   commit itself is lost — an outcome the priors never learn from. Pilot 4 saw 2 timeouts
   in 55 episodes at 2 cells; E1 runs 10. The seal stall on production (four `POST /traces`
   held 37–224 s, released together) is under investigation; the fix is an engine PR.
   *Where:* `amfs-internal` seal path; harness commit retry (idempotent by `outcome_ref`).
   *Predicts:* zero `commit_error` / `record_attempt_error` notes in E1.

8. **Retention's cold start.** 49 % first-attempt success before the change, 95 % after
   30 episodes: the segment → action rule is learnable but learnt slowly. A `procedure-`
   entry induced from the first successes (the lever the RSI arm already has) should
   pull the early window up. Lower priority: retention is already 90–96 %.
   *Predicts:* episodes 0–19 first-attempt success ≥ 70 %.

Saturated families (`analytics`, `concierge`, `order-ops` at 99–100 %) have no headroom and
are left alone; regressions there are what H2 guards.

## What this adds up to

If items 1–5 land, changed-class post-change success in `support` and `ci-fix` goes from
≈ 55 % / 65 % to ≈ 90 %: `support` ≈ 95 %, `ci-fix` ≈ 95 %, `retention` ≈ 93 % (item 8),
others unchanged — a six-family mean of ≈ 96 % against 91.5 % today, and ≥ 92 % on the ten
E1 families if the four held-out families behave like the six. That is the engineering
target; the paper reports what the confirmatory runs measure, whatever that is.

## Discipline

- Development runs: pilots and `-dev` cells on seeds **11 and 23** only, families
  `support`, `ci-fix`, `retention` (never the four held-out families).
- Every engine or agent change made from this map is logged in the Amendments section
  with its date, the mechanism it changes, and the prediction it makes; the confirmatory
  E1/E2/E3 SenseLab runs start only after `analysis-freeze-rsi`, and set B is run once.
- Confirmatory analysis reports seeds 37, 41, 59, 73 as the primary set; 11 and 23 are
  reported separately as development-exposed.
- Nothing here changes a baseline arm, a scenario, a change set, a rubric or a metric.
