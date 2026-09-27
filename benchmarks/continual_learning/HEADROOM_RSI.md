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

*Revised 2026-09-26 (see "What the map turned out to be" below).* With fleet 1 and three
attempts, the ceiling on a changed class is set by how many episodes the sweep over the
untried needs: a uniform no-repeat sweep over the ≈ 7 untried support actions loses ≈ 2.3
episodes per changed class (≈ 20 changed-class episodes per cell ⇒ support cell ≈ 86–87 %);
a sweep ordered by the entity's record elsewhere, where two of set A's three new support
fixes are already winners, loses ≈ 1.2 (⇒ 90–92 %). `support` ≈ 95 % is therefore not
reachable at fleet 1 from the fast loop alone; it needs the repair loop's fleet-wide entry
rewrite (item 4) or a fleet, and the six-family mean the engineering target should be read
against is ≈ 93–94 %.

## What the map turned out to be (2026-09-26)

Items 1–3 and 6 have one root cause, found while implementing them: **the harness never used
the product's situation contract.** `Run.begin(situation=...)` and `retrieve(situation=,
abstain=True)` are the documented call; the harness's `memory_search` had no `situation`
parameter and never passed `abstain`. Without a declared situation the engine has no
per-class record, so every mechanism the map asked for — a turn on one loss on an unsolved
task (`turned_unsolved`), a first exact win acted on (`_exact_first_win`), the sweep over
the untried in `elsewhere` order (`sweep_order`), the abstain that says the record cannot
say — was already in `amfs_core.actions` (since 3c523b86, 2026-09-22, on production as
amfs-http 0.3.49) and inactive. The pooled record instead said `act` on the class-agnostic
top action, returned nothing on 17–26 % of searched attempts, and explored by agent hash.
Verified live on `clbench-rsi`: declaring the situation and committing with `situation=`
and `entity_path=` yields `situation_record`, `situation_exact` rows and a one-loss
`explore` sweep ordered by `elsewhere`.

What was done, and where (development seeds 11/23 only; nothing touches a baseline arm, a
scenario, a change set, a rubric or a metric):

- **Harness (this branch, 1263a82b · 35c1a9cc):** `memory_search` gains a required
  `situation` parameter for arms whose task ends in an enum action; the label is folded,
  sticky per episode, declared to the SDK (`Run.begin`'s attribute) and passed on
  `commit_outcome` with `entity_path`; retrieves carry `abstain=True` and the candidates
  not yet tried on this task; the priors contract (act on the footer unless the task rules
  it out; on a retry take the next untried in the footer's order; never repeat a failed
  action; reuse labels) is in the system prompt of every SenseLab arm. Two ablations:
  `senselab-nosituation` (no label) and `senselab-nocontract` (no contract text). A label
  the server reports as pooling two classes is retired from the reuse list and may be
  refined within the episode. `searches_log` records `situation`, `recommendation`
  (`mode`, `plan`, `sweep`, `situation_outcomes`, `pooled`, `mixed`) and the action
  `taken`, so follow rate is measured per attempt.
- **Engine, OSS PR [#449](https://github.com/raia-live/amfs/pull/449) (awaiting approval):**
  situation labels compare as word sets without noise words (`card declined` ≡
  `Card-Declined` ≡ `declined card ticket`; `ios` ≠ `android`); the sweep ranks an action
  with no record elsewhere at the uniform prior, above one that only lost elsewhere; a
  mixed record (two actions, wins and losses, no winner) abstains out loud with
  `abstain=True` instead of returning nothing (this was the 15 % `rec_none` on retries in
  pilot 5a); the pooled-classes abstain names the declared label and asks for a finer one.
- **Repair loop, amfs-internal PR [#780](https://github.com/raia-live/amfs-internal/pull/780)
  (awaiting approval):** the repair agent gets a wall-clock budget
  (`AMFS_EVAL_REPAIR_TIMEOUT_S` 300, step 120, `MAX_STEPS` 10) and records why it stopped;
  a template fix's rationale says why the agent's draft was not used; a derived error span
  among a retrying agent's three `resolve` calls reads as its lost credited call, not as a
  tool error (the lever `code` misclassification in pilot 4).

What pilot 5a (harness contract, engine as deployed) showed in its first 34 episodes,
ci-fix, before the run was lost to a closed log pipe: labels were consistent within a cell
(`snapshot diff on ui pr` / `snapshot diff on backend pr` kept apart), recommendation modes
`act` 16 / `abstain` 14 / `explore` 9 / none 7 over 46 searched attempts, every `abstain`
on a label's first episode, every none on a retry over a mixed record — and both audit
classes, whose fixes are opposite, under one label `audit failure` with the record
alternating (the engine's `pooled` abstain fired, correctly). Pilots 5a (`senselab`) and 5b
(`senselab-rsi`) are re-running on support and ci-fix, seeds 11/23, with the label-retirement
harness; they will be re-run once #449 and #780 are deployed, and the three runs (gridv5c /
pilot 4 → 5a/5b → post-deploy) are the development evidence for amendment (h).

### Pilot 5a, complete (contract v2 on the deployed engine)

| cell (senselab, gpt-5.5, s11+s23) | grid v5c | pilot 5a |
|---|---|---|
| support success / late (20+) | 82.1 % / 80.0 % | **92.5 % / 91.2 %** |
| support changed-class post-change | 61.2 % | **82.5 %** |
| support recovery ≤ 3 class-episodes | 2/6 | **6/6** |
| support retries / repeats of a failed action | 36.3 % / 0.8 % | 21.6 % / **0.0 %** |
| ci-fix success / late | 92.1 % / 88.8 % | **95.8 % / 95.0 %** |
| ci-fix changed-class post-change | 77.5 % | **90.0 %** |
| recommendation present | 72–88 % | **98–100 %** |

Items 1–3 and 6 of the map are closed by the contract alone (no engine change deployed yet).
The 13 residual failures (3 pre-change first encounters, 10 post-change sweeps, no errors)
locate the remaining headroom precisely: the post-change **search order**. Two of set A's
three new support fixes are winners elsewhere and the sweep finds them in ≤ 2 attempts;
the third (`android → reinstall_app`, the textbook answer) has never won anywhere on the
queue and sits at the tail of the `elsewhere` order, costing 2–3 class-episodes per seed.
The agent's own judgement, which contract v2 asks for first, did not pick it either — the
pre-change memory notes say reinstall is wrong for android — and at s23 ep 27 the agent
overrode a correct `resend_email` suggestion twice. Two remedies are in flight: #449 item 6
brackets every untried candidate with its record elsewhere (`reinstall_app (untested
anywhere)`), giving the judgement rule something to weigh; pilot 5c runs 5a's cells with
that rendering client-side (`client_core.commit = 704fb753`) before the deploy. What would
close the rest is a fleet or the repair loop: one agent's sweep cannot avoid the first
post-change loss per class, so a support cell at fleet 1 is bounded near 92–93 %; the
six-family late-window target of 93–94 % rests on ci-fix-like families averaging above it.

### The seal stall, found

The `commit_outcome` stalls since pilot 2 were a lock convoy on `amfs_immutable_traces`,
captured live with a 5-second `pg_blocking_pids` sampler at 07:06–07:07 UTC: an eval repair
job idle in an LLM call inside the tenant pool wrapper's per-checkout transaction (holding
`ACCESS SHARE`) ← a pro-api instance's start-up run of `schema_pro.sql` (`ACCESS EXCLUSIVE`)
← every seal's `SELECT content_hash`. Fixes in amfs-internal #780 (jobs and LLM-calling
routes on a non-transactional checkout) and #781 (traces schema once per database behind a
fingerprint marker, under `lock_timeout`). The same idle transaction is what production's
5-minute `idle_in_transaction_session_timeout` terminated twice during pilot 5b (07:08 and
07:09 UTC, `FATAL: terminating connection due to idle-in-transaction timeout`), losing those
repair runs mid-flight; #780's non-transactional checkout removes that failure mode too.

### Pilots 5b and 5c, complete — and what the failures are made of

| cell (gpt-5.5, s11+s23) | 5a `senselab` | 5b `senselab-rsi` | 5c `senselab` + brackets |
|---|---|---|---|
| support success / late | 92.5 / 91.2 | 92.5 / 90.0 | 91.7 / 88.8 |
| support changed-class post-change | 82.5 | 80.0 | 77.5 |
| ci-fix success / late | 95.8 / 95.0 | 95.8 / 93.8 | 95.0 / 92.5 |
| ci-fix changed-class post-change | 90.0 | 87.5 | 85.0 |

5b says the repair loop is free (pilot 4 had it costing 7 points on support) but produced
nothing yet: template-source proposals only, first lag 3–13 episodes, nothing shipped — the
runs the seal stall terminated; re-tested as 5e after #780 deploys. 5c says brackets on
every candidate do not help, and the per-episode analysis says why. Pooled over 5a/5b/5c,
the 23 changed-class post-change failures on support split by **exposure of the class
since the change**:

| exposure | episodes | failed | what the record said |
|---|---|---|---|
| 1st | 18 | 16 | `act` on the old fix — nothing could know; in-episode retries recovered 2 |
| 2nd | 18 | 7 | `explore`/sweep; the new fix outside the five-slot plan or fifth in 6 of 7 |
| 3rd | 18 | 0 | |
| 4th+ | 60 | 1 | a fragmented label (`ios data export error` vs `csv export timeout`) |

So the **first exposure is a floor**: 3 classes × 2 seeds = 6 of 120 episodes (5 %) per
support run, whatever the engine does at fleet 1. The second exposure is the headroom, and
reproducing the live footer against production for the worst ci-fix cell (s11, `flaky
integration failure`, seven actions tried, 4/4/4) showed the engine losing it twice over:

1. **The sweep buried the textbook fix.** `sweep_order` ranked `(-won, -p, i)`, so an
   action with losses on *other* situations sat below every never-tried one. Losses
   elsewhere are the times an action was the wrong guess for another class — the textbook
   answers are guessed most — and both quirk-to-textbook changes (`reinstall_app` 0/2
   elsewhere, `edit_generated_file` 0/4 elsewhere) were outside the plan on all nine
   post-change exposures.
2. **The footer hid the most-failed actions.** `tried[:6]` and `failed[:5]` cut a
   posterior-sorted record from the top, so the Tried line showed five 0/1s and not
   `fix_code 0/4`, and "Do not spend an attempt on" showed neither `fix_code` nor
   `rerun_job 0/3` — the two actions the agent took a fifth and fourth time.

Both fixed in amfs#450 (core 0.3.43): `(-won, i)`; every untried candidate named in the
sweep's order; the footer shows all tried up to ten, most-lost first in the avoid list;
brackets only on winners elsewhere. Label fragmentation (the agent's own situation strings)
costs ~1–2 blank records per cell (~0.5 %) and is left as a known residual: a server-side
match by task-text neighbourhood is the candidate remedy, not attempted before the freeze.

Revised ceiling, support at fleet 1 on set A: 100 % − 5 % (first exposures) − ~1.5 %
(pre-change first encounters) ≈ **93–94 %** if second exposures are solved; 5a/5b sit at
92.5 % with 2–4 second-exposure losses per run. Ci-fix has one first exposure per changed
class (2 classes) and sits at 95–96 %.

### Pilots 5d and 5e, complete — the pair, the baselines, and why the loop is still silent

Same cells, on the deployed #449 + #780/#781 with #450's core client-side; the
`e1-A-baselines` run (GPT-5.5, set A, fleet 1) supplies the other arms on the same seeds.

| support, s11+s23 (success / late / changed-class post-change) | |
|---|---|
| `none` (frontier alone) | 66.7 / 51.2 / 5.0 |
| `pgvector` · `pgvector+rerank` · `pgvector-diy+outcomes` · `raw-traces` | 73.3 / 66.2 / 35.0 · 76.7 / 70.0 / 45.0 · 70.0 / 61.2 / 27.5 · 75.8 / 63.8 / 27.5 |
| `senselab` 5a · 5d | 92.5 / 91.2 / 82.5 · 90.8 / 87.5 / 75.0 |
| `senselab-rsi` 5b · 5e | 92.5 / 90.0 / 80.0 · **94.2 / 92.5 / 85.0** |

| ci-fix, s11+s23 | |
|---|---|
| `none` · pgvector variants · `raw-traces` | 87.5 / 81.2 / 62.5 · 86.7–89.2 / 81.2–83.8 / 62.5–67.5 · 87.5 / 81.2 / 62.5 |
| `senselab` 5a · 5d | 95.8 / 95.0 / 90.0 · 94.2 / 91.2 / 82.5 |
| `senselab-rsi` 5b · 5e | 95.8 / 93.8 / 87.5 · 95.8 / 93.8 / 87.5 |

Three readings:

1. **The outcome loop's margin is the large, stable effect**: +16–18 points over the best
   baseline on support (+35–40 on the changed classes after the flip), +6–9 on ci-fix
   (+20–25 on the changed classes). `none` never learns a flipped fix (5 %).
2. **Identical configurations differ by up to 5 points per cell at n = 60** (5d s11 support
   90.0 vs 5e s11 95.0; 5d s23 85.0 vs 5e s23 90.0). Nothing between 5a, 5b, 5c, 5d and 5e
   is resolvable at this size; the dev pilots find defects, the six-seed E1 grid measures.
3. **The RSI arm has not yet been tested as an RSI arm.** In 5e the loop detected what it
   should — a regime-shift fix on the export-timeout flip three episodes after it, an
   escalation ping-pong incident, a canary on s23 with three Tier 2 replays — and shipped
   nothing: the repair agent gave up on its 50k-token budget after five steps on every fix
   (template kept), and each replay's reflection step was refused on the repair branch
   (no grant can exist on a platform-created branch). amfs-internal#782 fixes both.

Where the shipped fixes have to act is now specific. Reading the lessons behind ci-fix
s11's `flaky-integration` 4/4/4 (5a, 5c, 5d alike): after its first failed sweep the agent
wrote `ci-flaky-integration-timeout-avoid-unrelated-explore` — "do not follow generic
explore-order unrelated fixes (dependency bump, migrations…)" — an untested note scoring
0.95 by similarity, and on every later exposure it obeyed that note over the record,
taking `fix_code` (0/4 on this situation) first and `rerun_job` (0/3) second, so each
exposure advanced the sweep by one action at most. The engine's evidence gating did its
part (the two older "treat it as flakiness and rerun" notes are discredited 0/3 and shown
as such); what the agent trusts is its own fresh, specific, wrong hypothesis. A corrective
entry from the loop — validated by a Tier 2 replay before it ships — is the mechanism
that outranks such a note; nothing in the retrieval ranking should, since the note *is*
the most similar text. Pilots 5f/5g measure that once #782 is live; the harness now keeps
the rendered footer with every search in the episode record, so the next reading of
"followed / ignored" is against the text the agent saw rather than a reconstruction.

### Pilots 5f and 5g, complete — the loop was never reached, and why

On the #782 engine (`client_core` ebf930e3), pooled over seeds 11/23, 120 episodes per
family × arm:

| run | family | arm | success | late (20+) | changed-post | RSI fixes |
|---|---|---|---|---|---|---|
| 5f | support | senselab | 93.3 | 92.5 | 85.0 | — |
| 5f | ci-fix | senselab | 95.0 | 92.5 | 85.0 | — |
| 5g | support | senselab-rsi | 90.8 | 88.8 | 77.5 | 0 |
| 5g | ci-fix | senselab-rsi | 95.0 | 92.5 | 85.0 | 0 |

5f met the 5d prediction (support late 92.5 vs 87.5 on 5d; changed-post 85.0 vs 75.0).
5g is not a reading of the mechanism: **no 5g agent received a single `fix_propose`
job**, so `senselab-rsi` ran as `senselab` plus verdict latency. The production record
says why, and it is not what 5e's silence was.

Of the 54 entries that took a failure across the four 5g cells, only six had been
validated three or more times before failing — 60-episode cells with the change at
episode 20 give each of ~9 changed-class entries about two pre-change validations, below
`REGIME_MIN_SUCCESSES`. Of those six, the two that ended discredited
(`repo-readme-ci` 4/3, `android-alerts-missing-settings-change` 3/2) both have
`last_outcome = success`: the runs that discredited them recovered after a failed attempt
(`record_attempt`), so the trace sealed as a success and `regime.on_trace_sealed`, which
read only `outcome_type`, never asked. The detector fired correctly on 5e s23 (first-lag
3–4 episodes) because those entries happened to fail terminally.

Reading the loop end to end against the day's eight proposals (all cells) found the loop
complete and three seams open — logged as amfs-internal#783, wiring rather than new
machinery:

1. `propose_from_regime_shift` built a template and called only `repair.stage`, never
   `repair.draft` — every outcomes-path proposal was a 0.3 s job with `model: null` and no
   `correction`. Now the judge path's draft-or-template block is one helper both paths
   use; the regime job carries the failed trace, the agent gets it as the failing run and
   the shifted entry as the implicated memory, and a drafted correction makes that run the
   Tier 1 case.
2. `on_trace_sealed` now also takes the failed attempts' `causal_entry_keys` (which are
   not in `causal_entries` once an attempt boundary exists) — the 5g case above.
3. A fix with nothing to replay skipped the Tier 1 test *and* the policy (`_apply_policy`
   only ran off a test result; `canary_queue` only listed tested statuses), so every
   regime-shift fix stayed `proposed` under `auto_after_canary` — 5e s23's two regime
   fixes included. `stage()` now runs the inconclusive ladder for such a fix (capped
   small canary under `auto_after_canary`, held with a reason otherwise) and the sweep
   queues it.

Prediction for the re-run of 5g once #783 is live: `fix_propose` jobs on every cell
whose discredited entries include an attempt-charged one; ≥ 80 % of regime-shift fixes
`source = repair_agent`; fixes reach `canary` on the `auto_after_canary` agents rather
than resting at `proposed`. The threshold finding stands on its own: at 60 episodes the
outcomes-only detector has little to detect, which is the argument for E2's 200-episode,
fleet-10 protocol rather than for lowering `REGIME_MIN_SUCCESSES`.

One more thing the record showed: `AMFS_EVAL_AUTO_PROPOSE` is account-wide, so plain
`senselab` agents (5d, 5f, the team-grows cells) also received proposals. Corrective
entries live only on `repair/<id>` branches until shipped, and nothing shipped, so the
control arms read nothing from the loop — but the confirmatory runs should place the
non-RSI SenseLab arms on an account (or agents) with auto-propose off, so the separation
is by construction and not by the loop's silence.

### Pilot 5h, in progress — the loop is reached, and two seams below it

#783 deployed at 13:05Z (`amfs-api-01987-rq2`, `pro-api-00772-844`). 5h is 5g's protocol
on that engine; the two `senselab-rsi` team-grows halves were launched beside it on
replay ports 8788/8789. Within twenty minutes the prediction above held on the parts it
could: the first regime-shift `fix_propose` jobs fired on 5h and the newcomer cell (59 s
and 173 s — the agent drafting, not the 0.3 s template of 5g), **4 of 4 fixes that hour
were `source = repair_agent`, model gpt-5**, two from the regime path and two from the
judge path (`Escalation pingpong`, `Repeated unresolved handoff`), and each went straight
to Tier 1. The first drafted regime correction rewrote `playbook-tier1` v8 → v9 for the
iOS login loop; its replay still failed (`Ticket reopened, CSAT 2`) and the fix ended
`test_failed` — the gate catching a wrong guess, which is the loop working. The second
(`android-alerts-stopped-settings-fix`) came back `test_inconclusive` at fidelity 0.00:
a regime trace's tool calls are the harness's, so the internal stand-in replay cannot
vouch, and the fix moved to `canary_pending`. There it stopped, for a reason below the
eval package:

1. **Repair branches for rewritten keys never opened.** `open_repair_branch` →
   `memory.write(..., branch="repair/<id>")` raised
   `VersionConflictError: expected 9, found 0` on every regime fix. `CoWEngine.write`
   numbers a write from `read()`, which on a fresh branch resolves to the parent's live
   row (the overlay); `PostgresAdapter.write` looked only at the branch's own rows and
   insisted on 1. Fixes that add a new key (the judge path's lessons) were untouched,
   which is why those two got branches and Tier 2 while the regime fixes fell back to
   "writes to main on Ship" with no branch to canary or replay. amfs#451 makes the
   adapter take the parent's live row as the baseline (branch continues main's
   numbering; both adapters; integration tests through the engine and the async path).
2. **Tier 2 webhooks were split three ways.** The free ngrok agent gave the three hubs'
   tunnels one reserved domain and pooled them round-robin (confirmed on the tunnel
   metrics), and each runner has its own webhook secret, so two thirds of
   `replay_requested` deliveries hit a hub that refused them (`bad_signature`, no retry).
   The hub now hands such a request once to its siblings on `CL_REPLAY_SIBLINGS`
   (`a64666eb`). The two team-grows RSI halves were stopped at 85/71 episodes (results
   kept as `*-aborted`) and their tunnels removed; they re-run with 5i once #451 is live.

Readout of 5h itself (rates, `rsi_dev`) follows when it completes; what it can already
answer is the 5g question — the loop is reached, the agent drafts, and Tier 1 grades.

### Pilot 5h complete, the RSI team-grows halves complete, #784 live — and the third seam

5h finished 57/60, 54/60 (support) and 57/60, 57/60 (ci-fix) on the #451 engine for its
second half; the two `senselab-rsi` team-grows halves (relaunched 14:05Z, 400 episodes
each) finished support 98/96 (pioneer) and 96/95 (newcomer), ci-fix 98/100 and 97/97.
`analysis/team_grows.py` over all 24 cells (`results/rsi-teamgrows-dev-report/`):

- **Joiners on the RSI arm are indistinguishable from joiners on `senselab`.** First
  exposures after the join, first-attempt success 1.00 on both arms and both protocols
  (changed classes 1.00 on 36 / 10), 0 poisoned, parity 20/20 and 4/4 at a median 5
  episodes, disagreement with the veterans 0.00 vs 0.02 / 0.00; `pgvector+rerank` 0.68 /
  0.75 (changed 0.31 / 0.40, poisoned 0.69 / 0.60 on changed classes). Tokens per joiner
  episode 4.3k / 4.2k vs 4.4k / 4.1k vs 5.6k / 5.2k. Amendment (i) P1 met; P2's poisoned
  clause met; **P2/P3's `senselab-rsi` vs `senselab` clause is inconclusive as
  pre-specified**: the veterans' loop shipped nothing before any join (proposed 0–2 per
  cell, 4 of 8 cells with ≥ 1 draft, all `repair_agent`, 0 shipped), so platform-authored
  hits in joiner contexts were 0.00 on both SenseLab arms. The joiners' advantage over the
  baseline is the outcome loop's — validated and discredited labels on entries written by
  veterans they never met — and the paper says so.
- **#451 holds in production.** Every fix created after 14:03Z has a branch (12/12);
  before it, regime fixes had none.
- **#784 is live (15:00Z) and has had nothing to record yet.** Since then 8 fixes were
  proposed; their statuses are `canary` (4), `test_passed` (2), `test_inconclusive` (1),
  `proposed` (1) — none has failed a test, replay or canary, regressed or shipped, so no
  `lesson-fix-*` / `decision-fix-*` entry exists and no event carries `recorded` yet.
  Consistent with the mechanism; the first one appears with the first failed or shipped
  fix on the new revision.

**Seam 3 — the ship gate runs on production time, the cells on benchmark time.** All day,
zero fixes shipped. Of 21 fixes since 13:05Z, 9 are in `canary` and none has decided.
Their arm counts explain it: 6 of 9 have 0 graded outcomes in *both* arms after 2–6
evaluations, one has 1 vs 9, one 17 vs 13 (1 recurrence). Two cadences are at work.
Canaries capped at 10 % (the inconclusive-test ladder) start on the hourly `canary_sweep`
(three sweeps ran today: 13:20, 14:20, 15:20) — a fix proposed at 13:36 started its
canary at 15:20, after its 60-episode cell had ended, so the arms never filled. Canaries
at 50 % start within minutes of the test but are *evaluated* only by the same hourly
sweep, and `canary.decide` waits for `canary_min_outcomes` in each arm before it will
call anything; a 60-episode cell with one identity produces ~30 sessions after the
canary starts, half of them routed to the branch, judged asynchronously, and the cell is
over before the second sweep. The sweep is idempotent per UTC hour by design, so calling
it more often is a no-op. The loop is not silent for want of drafts any more — it is
silent because its ship gate was built for a customer with hours of traffic, and a pilot
cell has one. Three options, none taken without an amendment: (a) the harness calls
`POST /fixes/{id}/canary/evaluate` (a customer route, account API key) on running
canaries at each episode boundary, so a canary decides as soon as the arms suffice; (b)
cells long enough for two sweeps after the change (≥ 150 episodes, or fleet-6 replicas
so the arms fill in a tenth of the time — E2 already has both); (c) `auto_after_replay`
for the primary arm, with the canary-gated arm as the safety ablation. (a) changes
nothing about what ships or on what evidence, only *when the loop looks*; it is the one
that keeps the canary as the primary gate and is the recommendation for (k).

### Pilot 5j complete — (j) is live and untriggered, seam 3 confirmed, the lock sampler retired

5j (the 5h protocol on `amfs-api-02003-hcb`, launched 15:05Z) finished 16:12Z with 0
errors: support 57/60, 55/60; ci-fix 57/60, 56/60. Pooled against 5h — 93.3 % vs
92.5 % (support), 94.2 % vs 95.0 % (ci-fix); late-window 91.2 % / 91.2 % vs 90.0 % /
92.5 %; changed-post 82.5 % / 82.5 % vs 80.0 % / 85.0 % — flat, and flat is the correct
result: (j) only writes on a failed or shipped fix and the new revision has seen neither.
The (j) readout is in `HYPOTHESIS_RSI.md` under the amendment; the headroom finding is the
same one from a different angle:

- **The loop drafted 5 fixes across 5j's four cells (0, 2, 1, 2) and shipped none** —
  three sit in `canary`, two are `test_inconclusive`. Same signature as 5h (3 fixes, 0
  shipped) and the probe-(i) RSI cells. Across everything on the new revision: 10 fixes,
  `canary` 5, `test_passed` 2, `test_inconclusive` 2, `proposed` 1. Zero failed, zero
  regressed, zero shipped, so zero `lesson-fix-*` / `decision-fix-*` entries and zero
  events with `recorded`. The mechanism sits behind seam 3 like everything else that
  needs a verdict.
- **(j) also has a reach problem of its own at this cell length.** `prior_fixes` is
  agent-scoped and each cell is a fresh agent; a second draft on a failed entry needs two
  drafts on one entry inside one 60-episode cell, at an observed ~1.25 drafts per cell.
  Pre-#784 the two `test_failed` fixes of the day were never followed by a second draft
  on their entry, so G1 is 0/0 on both sides. E2 (200 episodes, fleet-10, two changes)
  is where the same agent drafts often enough for a second draft to occur; (k)(a) is
  what lets a first draft reach `shipped` or `canary_failed` inside a cell at all. Both
  are already on the plan; neither changes the mechanism.
- **Replay fidelity is still the Tier 1 ceiling.** Post-#784 tests: 3 passed at fidelity
  1.0 (all three rewrites of `repo-readme-ci`, one per `ci-fix` cell), 5 inconclusive at
  a mean 0.59 (three judge-path lessons at 0.70–0.75, two rewrites at 0.75 and 0.00; "1
  of 3 / 3 of 3 predicted run(s) could not be replayed"), 2 skipped ("nothing could be
  replayed"). Inconclusive fixes take the 10 % canary ladder, the slow lane of seam 3.
- **The lock sampler is retired.** `LOCKWAIT` samples by UTC hour: 07Z 165, 11Z 6, 13Z
  115, 14Z 152, 15Z 1, 16Z 0 — every burst sits inside a rollout window (07Z, the
  13:57–14:01Z #451 rollout, one sample at 14:59 for #784), steady state is clean, and 5h
  and 5j both finished with 0 errors, so the seal stall of 5f/5g is closed by #780/#781.
  `IDLETX` (idle-in-transaction sessions holding the Cortex's `SELECT … FROM
  amfs_memory_entries`) ran at a steady ~340 samples per hour all day, independent of
  load; it is not a benchmark blocker and stays a separate PR candidate against
  amfs-internal. Sampler stopped 16:14Z; `/tmp/lockwatch.log` kept for that PR.

### Seam 3, re-read against the traffic — a correction, an engine bug, and amendment (k)

The seam 3 paragraph above is wrong on one point and this note corrects it rather than
edits it. The canaries were **not** "evaluated only by the hourly sweep": the arm's
pre-registered look schedule calls `POST /fixes/{id}/canary/evaluate` every 5 sessions,
which is where the 5–16 evaluations per canary came from, and the benchmark agents run at
`canary_min_outcomes` 8 (E2 mapping), not the product default 20. Reading every canary's
arms against the sealed traces behind it (`amfs_immutable_traces`, `attributes.canary_arm`)
gives the real picture for the 11 canaries of the day:

- **7 saw no traffic at all** after they started. Four are 10 %-capped canaries of
  `test_inconclusive` fixes, started by the *platform's* hourly sweep at 15:20 / 15:52 /
  16:21 for fixes proposed 13:36–15:24 — the harness filled a freed slot only with
  `test_passed` fixes, so these waited for the sweep, which fired after the cell. Three are
  50 % canaries that started within minutes of their test and found the cell already over.
- **1 is an engine bug.** `7a7a164b` (5j `ci-fix` s11, `repo-readme-ci` rewrite): 53 runs
  after start, 27 stamped canary / 26 control, all 53 carrying a judge verdict — arms 0/0
  after 16 evaluations. A regime-shift fix has neither `judge_id` nor `behavior_id`, and
  `canary_arm_counts` graded such a fix with `SELECT NULL AS verdict`. **No outcomes-only
  canary could ever decide, on any account, at any traffic.** `7963790e` is the same case.
  amfs-internal#785 grades these fixes on the signal that proposed them — a run that
  reported an outcome and read the entry the fix is about is graded, and has recurred when
  the outcome was not a success (`canary.implicated_entries`; the regime path now pins the
  shifted key in an agent-drafted risk entry's `pattern_refs` so the link survives the
  draft). Unit + Postgres tests; PR opened for Bruno's approval.
- **3 filled and behaved.** `58f433d4` 17/13 over a floor of 8: "no significant
  difference yet, 6 % vs 0 %" — one recurrence in the canary arm, none in control, a
  legitimate running canary. `a4daf346` 8/6, two control runs short when the cell ended.
  `26b1444d` 1/9 — a 50 % canary whose sticky session hash put 1 of 10 sessions in the
  canary arm; one case, noted, not chased.

**Amendment (k)** (`HYPOTHESIS_RSI.md`): (1) #785; (2) the look schedule fills a freed slot
from the same queue the platform's sweep reads — passed, inconclusive, or `proposed` with
nothing to replay, no replay awaited, oldest by `tested_at` — at the product's 10 % cap
for a fix no offline test vouched for (`_canary_ready` / `_replay_awaited` in
`arms/senselab_rsi_arm.py`; `canary_start` events now carry `pct` and `from_status`; the
report carries `canary_undecided_at_end`). (3) Cell length is the limit that stays: a
fleet-1 60-episode cell cannot fill a 10 % canary and fills a 50 % one only if the fix
lands by about episode 40 — reported, not amended; E1 fleet-6 and E2 carry the measured
ship-rate readout. Treated run `rsi-pilot-5k` (5h protocol) once #785 is live; 5j is the
baseline.

### What the tiers can and cannot do on set A — and amendment (l)

Asked whether `senselab-rsi` shows value over `senselab`, the answer from six pilots is
parity (HYPOTHESIS_RSI.md (l) has the numbers), and the reason is the benchmark, not the
loop. Reading the two arms against the scenario code:

- `senselab` is Tier 1 — the outcome loop. It cannot write a corrected entry; it can
  discredit a stale one, steer the sweep with action priors, and record a recovery as a
  `lesson-contrast`. `senselab-rsi` adds Tier 2 — judge, behaviours, repair agent, replay,
  canary, ship — whose output is a *corrected* entry or procedure, validated before agents
  read it. That is the whole difference.
- Set A gives Tier 2 nothing to add. The task is a choice among ten enumerated actions and
  the change moves one; the sweep recovers it in ≤ 2 exposures (`realworld.py`
  `SUPPORT_CHANGES`, `CI_CHANGES`). A wrong action returns "Customer replied 24h later:
  'Still happening.'" — nothing in a failed trace names the new answer, so a drafted
  correction knows what the agent knows. Outcomes are verified and immediate, so the judge
  can only agree. The regime predicate wants ≥ 3 validations and the canary 8 outcomes per
  arm; a class recurs every ~8 episodes; a perfect fix lands after the second exposure.
  The ceiling (93–94 %) is computed above and both arms sit on it.
- What Tier 2 is *for* is absent from set A: a rule that cannot be found by exhaustion, a
  failure nobody labelled, exploration too expensive to do on a live customer. The repair
  agent's tools say so — `search_traces`, `compare_traces`, `find_similar_traces`,
  `read_procedures`, a structured `procedure` type — none of which a single-action flip
  needs.

Two things were checked before recommending the reframing. First, that presenting the full
engine as "SenseLab" in the headline is not a claim the ablation contradicts: it is not, as
long as the set-A gain is credited to Tier 1 in the sentence that reports it (the
attribution rule in (l)), and the maintainer-oracle bound (`rsi-oracle-5A`, running) tests
that no repair could have done better on A. Second, that the new regimes are a fair fight
rather than a rigged one: under set C the `senselab` agent sees the same attribute in the
same tool output and is told by the `situation` parameter to split labels when "the
deciding detail differs", so it *can* learn the rule by labelling — P1's 40–75 % range is
that possibility, and P2 is the claim that the repair loop gets there faster and more
reliably by comparing runs. If P2 fails with fixes shipped, the paper says so.

Two candidate tweaks were rejected. A platform-served "SOP slot" the agent always reads
would give SenseLab an affordance the baselines lack; procedures reach the agent through
the same retrieval every arm gets (rendered legibly by `arms.base.entry_text`, which is
the only harness change on the read path). Loosening the frozen rubric so a recovered
attempt counts as a failure is unnecessary: since #783/#785 the regime path charges attempt
failures, so set C reaches the loop through outcomes alone, and the judge path fires on the
tasks whose retries run out.

Engine-side, one thing stays open: `AMFS_EVAL_AUTO_PROPOSE` is read from the process
environment (`amfs_eval/fixes.py` `auto_propose_enabled`), so every agent on the API gets
proposals — control arms included. Nothing shipped and corrective entries live on branches,
so the control arms can have read nothing from the loop (the fix ledger shows 0 shipped
across every cell to date), but the separation is by circumstance. A per-agent
`auto_propose` in the repair settings row (`fix_store.py` already keeps `repair_policy`,
`canary_pct`, `canary_min_outcomes` per agent) is the PR candidate; it needs approval.
*Resolved 2026-09-27:* amfs-internal #786 (migration 107, `RepairSettings.auto_propose`,
gate in `enqueue_fix_propose` after the process switch and the plan gate) merged at
23:34:18Z. Harness commit `bf589c41`: every Tier-1-only arm PUTs `auto_propose=false` on
each hosted identity's first session and records the result as `arm_notes.auto_propose`
(`held` / `unsupported` before the deploy / `skipped` under `CL_CONTROL_AUTO_PROPOSE=skip`
/ `error:<status>`); the repair-loop arms set it `true` explicitly in `agent_ready`.

### 5k complete — the canary reads nothing, and why: a first-exposure failure is not a regime shift

`rsi-pilot-5k` (the #785 treated run) finished 240/240 with success flat against 5j and
three canaries that never decided. The (k) readout in HYPOTHESIS_RSI.md has the numbers;
the mechanism is worth stating here because it is the third seam of the same kind — the
loop's gates were designed for a customer's traffic and this benchmark's traffic has a
shape they had not met.

Routing was fine (32/17, 19/19 and 1/17 stamped sessions per arm). `graded` was 0 in every
arm because the fixes rewrote entries the traffic no longer reads. The two ci-fix fixes
rewrote `repo-readme-ci`, the seeded README, which the agent read in episodes 1–4 and
never again once it had notes of its own. The README became a regime candidate because
episodes 5–6 were first exposures to a new class whose failed attempts had read nothing
else, so #783 charged the failures to it: four successes, then two failures, "a validated
entry that stopped working". It had not stopped working; the class was new. A canary on
that fix then held the cell's one slot for 50 sessions with nothing to read, and would have
held it for a day.

Two engine corrections follow, opened as PRs for approval: a canary futility rule for arms
with outcomes and no grades (`decide()` rule 1b, "ungraded", freeing the slot at
`FUTILITY_MULTIPLE × min_outcomes` outcomes), and a regime guard — an entry is a candidate
only if an earlier successful run of the same agent in the same situation read it, so a
first exposure cannot indict what it happened to read. `rsi-c1-rsi` waits for both to
deploy: on set C the real regime fix lands after episode 30, and a README canary started at
episode 10 would hold the slot through the whole measurement window; under `-nocanary` the
README rewrite (Tier 1 passed on s11) would ship as a false-positive K4.

### P4 readout — the maintainer oracle on set A (`rsi-oracle-5A`, 240/240)

`senselab-oracle-fix-5` (the `senselab` protocol plus the correct rule text written to
memory five episodes after the change, readable from episode 25): support 95.8 %, ci-fix
99.2 %; changed-class post-change success 90.0 / 97.5 %. Against the (l) reference for
`senselab` (93.3 / 95.0) that is +2.5 on support — inside the 3-point prediction — and +4.2
on ci-fix, between the prediction and the 5-point falsifier. Against `senselab-rsi` on the
same protocol (5k: 93.3 / 95.8; changed classes 82.5 / 90.0) the oracle is +7.5 on the
changed classes in both families. Reading: the set A ceiling is real for the unchanged
classes and the first exposures (the oracle cannot help there and does not), and the
headroom exhaustion leaves on the changed classes is about 7.5 points per family at 60
episodes — the number a shipped fix would have to recover to match a maintainer who knew
the rule. The ceiling argument in (l) stands; the paper states the bound as measured rather
than as "within 3 points". Dev seeds only; the confirmatory oracle cells run with E1.

### P2/P3 readout — `rsi-c1-rsi` complete (800/800): the canary grades for the first time, and the two seams it finds

`rsi-c1-rsi` (set C, seeds 11/23, support + ci-fix, `senselab-rsi` and
`senselab-rsi-nocanary`, 100 episodes, change at 30, focus 0.5) ran 01:17Z–05:06Z on
2026-09-27. Engine: the support-s11 cells pre-#790, the support-s23 cells straddling the
02:34:34Z cut, the four ci-fix cells post-#790 (HYPOTHESIS (n) cut record). Seventeen fixes
were proposed across the eight agents; three canaries started; one memory shipped.

**Moved half, episodes 30–99, first-attempt success (n = 35 per cell), with the paired
`senselab` cell from `rsi-c1-tier1`:**

| family | seed | `senselab` | `senselab-rsi` | `senselab-rsi-nocanary` |
|---|---|---|---|---|
| support | 11 | 34 | 31 | 20 |
| support | 23 | 60 | 51 | 34 |
| ci-fix | 11 | 77 | 66 | 74 |
| ci-fix | 23 | 66 | 49 | 66 |

The kept half stayed 94–100 % in every cell (no regression). Neither RSI arm beats
`senselab` on the moved half on any of the eight cells; the nocanary support cells are
14–26 points below it.

**P2 — not met.** `senselab-rsi-nocanary` shipped in 1 of 4 cells, and at episode 94
(support s23, `3e521221`, a judge fix on the situation "ios team card decline"; two moved
episodes remained, both succeeded, as did `senselab`'s). The support-s11 cell's one fix
and the s23 cell's other two ended `test_failed` at Tier 2 — the agent's own replay still
failed 1–3 cases. The two ci-fix nocanary cells proposed four fixes, all regime-shift
proposals ("… stopped working", carrying a situation), all Tier 1 `test_passed` — **and none
shipped**: a regime proposal has no failing runs behind it, so its case set is empty, the
customer replay cannot be asked (`replay_pending: the fix's case set has no cases`), and the
nocanary policy, which ships on a passed replay, has nothing to ship on. The clause in (l)
does not fire — P2 failed *without* fixes shipped, so it says nothing yet about what a
shipped fix is worth; it says the nocanary arm cannot ship the one kind of fix set C is
built to produce. **Seam 4:** the regime path needs a Tier 2 of its own (the fix's
*situation* as the case set — the failing runs in it since the shift — or a direct route to
the canary when the replay has no cases, which is what the sweep gave the canary arm).

**P3 — not met, and the mechanism is different from 5k.** Three canaries started, none by
the (l) route — every one via the sweep's slot fill or a passed replay — and **none decided
within its cell**; all three were `running` when traffic stopped and will end at the 24-hour
ceiling. But two of them graded, which no canary before #790 did:

- `025df85b` (support s23, judge `cl-task-failure`, started episode 45): 54 sessions routed
  26/28, **54/54 graded**, recurrence 1/26 canary vs 2/28 control, p = 0.60. The canary
  works end to end; the effect it measured is nil on 26 runs. Post-start moved half: 50 %
  vs `senselab` 69 % paired (n = 26).
- `11dcc6b1` (ci-fix s23, judge `cl-task-failure`, started episode 50): 49/49 graded,
  0 recurrences in either arm (p = 1.0). The behaviour the judge flags did not recur on
  main either after episode 50 — nothing for the branch to prevent.
- `9ca1e6b5` (ci-fix s11, **situation-graded**, "pip audit pinned dependency", started
  episode 45): the first (n) canary. 2 graded canary, 0 control, in 57 sessions over 24
  minutes. The situation *was* recurring — the agent declared a pip-audit situation on 15
  of those sessions — under ten different labels ("pip audit pinned dependency", "ci
  pip audit pinned related", "pip audit failure", "pip audit legacy sdk pin", "pip audit
  reports export pin", …). In the cell's first 45 episodes it had written "pip audit pinned
  dependency" nine times against three variants; after the canary started its wording
  drifted with the tasks. Situation grading matches the folded
  string exactly, so 13 of 15 runs in the situation were invisible to it. **This is the (n)
  falsifier as written** ("a regime fix in a recurring situation … held for exposure while
  runs in its situation are being sealed"), and the cause is not the grading rule but the
  label: the situation is agent-declared free text, and the whole (h) contract — records,
  priors, `pooled`, and now the canary — keys on it. **Seam 5:** situation identity has to
  be the server's, not the agent's — cluster the labels an agent uses (the Cortex already
  detects two kinds of task pooled under one label; the inverse, one kind of task under
  many labels, is the same computation), and let a situated fix be graded on the cluster its
  failing runs fell in. Until then a situated canary is exactly as reliable as the agent's
  spelling.

Two smaller findings. `graded_by` stored the "and none reached these runs" sentence on a
canary that was being graded (fixed, amfs-internal #792). And the queue behaved as #790
says: on `ci-agent-782226690` two more tested fixes waited behind `9ca1e6b5` for the rest
of the cell with no event — the slot was held by a canary that had 2 graded runs, one more
than the preemption rule's zero. Whether a canary at 2/0 after 50 sessions in a situation
the traffic keeps declaring should keep the slot is a question for (n)'s successor, not a
bug.

**What the paper takes from this run (dev seeds).** Tier 3 is now a working instrument:
routing, exposure, judge grading and the tally all did what #790 says on two canaries, and
the numbers it returned are honest nulls, not artefacts. What the loop still cannot do on
this benchmark is produce a *shippable regime fix inside the measurement window*: the
judge fixes it can ship or canary target failures the judge sees (task failure), not the
moved rule, and the regime fixes that do target the moved rule cannot pass Tier 2 (no
cases) or be graded at Tier 3 (label drift). The moved-half deficit of the RSI arms against
`senselab` is the cost of that: extra tested-but-unshipped branches and lesson entries in
the record, no recovered rule. Amendment (o), to be preregistered before the next dev run:
seams 4 and 5 as engine changes (PR candidates), the (n) prediction re-stated on
server-clustered situations, and a per-cell K1–K4 table with the engine version as a
covariate.

### The same run under the (o) rule — SenseLab against the baselines on set C (dev seeds)

The readout above presents `senselab-rsi` against `senselab` as two systems. They are one
system; the paper compares **SenseLab** (the full engine) with the baselines, and the
repair-loop ablation belongs in the attribution table only (HYPOTHESIS (o)). Re-stated here
with the paired cells of `rsi-c1-tier1` and `rsi-c1-baselines` on the same protocol and
seeds. **Correction:** the table above labelled "moved half, n = 35" is all changed-class
post-change episodes (moved and kept halves together); the moved half is 13–23 episodes per
cell, and the "kept half 94–100 %" line describes the *unchanged* classes. The three
populations are separated below. First-attempt success, episodes 30–99, per cell; dev
seeds, small n, no significance claimed.

**Moved half — the instances whose correct action changed (the RSI scenario):**

| cell | frontier alone | context-window | pgvector | pgvector+rerank | pgvector-diy+outcomes | **SenseLab** |
|---|---|---|---|---|---|---|
| support s11 | 0 (0/23) | 0 | 4 | 4 | 0 | **30** (7/23) |
| support s23 | 0 (0/20) | 0 | 0 | 0 | 0 | **25** (5/20) |
| ci-fix s11 | 47 (9/19) | 84 | 47 | 11 | 53 | **68** (13/19) |
| ci-fix s23 | 31 (4/13) | 69 | 23 | 8 | 31 | **15** (2/13) |

**Kept half — same classes, old action still correct:**

| cell | frontier alone | context-window | pgvector | pgvector+rerank | pgvector-diy+outcomes | **SenseLab** |
|---|---|---|---|---|---|---|
| support s11 | 83 | 92 | 83 | 75 | 75 | 33 |
| support s23 | 73 | 100 | 73 | 80 | 80 | 87 |
| ci-fix s11 | 12 | 100 | 50 | 100 | 44 | 62 |
| ci-fix s23 | 9 | 100 | 0 | 86 | 36 | 68 |

**Unchanged classes:**

| cell | frontier alone | context-window | pgvector | pgvector+rerank | pgvector-diy+outcomes | **SenseLab** |
|---|---|---|---|---|---|---|
| support s11 | 51 | 100 | 86 | 86 | 66 | 94 |
| support s23 | 54 | 100 | 86 | 94 | 71 | 97 |
| ci-fix s11 | 89 | 100 | 100 | 100 | 97 | 97 |
| ci-fix s23 | 71 | 100 | 100 | 97 | 86 | 97 |

`rsi-c1-baselines` finished 07:30Z (800/800 on the eight `pgvector` / `pgvector-diy+outcomes`
cells, 0 errors). Its four `context-window` cells did not run — the runner was given the paper's
name and the harness arm is `raw-traces` (amendment (b), `CL_RAW_TRACES_TOKENS = 24000`) — and
ran as `rsi-c1-baselines-cw`, 07:32Z–07:45Z, 400/400, 0 errors.

**Reading.** On support the retrieval baselines sit at 0–4 % on the moved half: they retrieve
the stale fix and follow it, every time, and frontier alone is at 0 because the new rule is
not inferable from the ticket without memory of the change. SenseLab is at 25–30 %: its
record marks the entry *contested*, the briefing says so, and the agent explores instead of
repeating. On ci-fix, frontier alone beats every retrieval baseline on the moved half (47 /
31 vs 11 / 8): the PR's files show the attribute and an agent with no memory reads them,
while an agent served a confident stale memory follows it — the cost of memory without
evidence; plain `pgvector` (no reranker) lands where frontier alone does (47 / 23), and
`pgvector-diy+outcomes` a little above it (53 / 31). SenseLab is 21 points above frontier
alone on s11 (68) and below every arm on s23 (15).
On the kept half frontier alone cannot know the old fix on ci-fix (9–12 %) and the memory
systems can to the degree they still serve it (SenseLab 62–68, pgvector+rerank 86–100,
pgvector 0–50); on support SenseLab pays for its
exploration on the moved half with a lower kept half in s11 (33). On the unchanged classes
SenseLab is 94–97 % in every cell against 51–89 % for frontier alone and 66–94 % for the
retrieval baselines.

**The context-window baseline is the finding of this fill-in.** With every prior transcript
of the cell in the prompt (24 000-token cap), the frontier model on ci-fix reaches **84 / 69 %
on the moved half** — the highest of any arm, SenseLab included — and 100 % on the kept half
and the unchanged classes. It is reading the rule off the history: the transcripts hold the
PR files, the action taken and the environment's verdict for every earlier task, so the
conditional is inferable in-context once a few moved and kept instances have been seen. On
support the same arm is at **0 %** on the moved half in both cells, like every other baseline:
the ticket transcripts do not show the attribute the rule turns on (app version, plan, issuer
country are in `lookup_account`, which the earlier transcripts do not always call), so history
alone carries the old answer. The cost is the other axis the paper reports (E6): post-change,
context-window is $0.075 (ci-fix) and $0.105 (support) per episode against SenseLab's $0.024 /
$0.026 and frontier alone's $0.008 — three to four times SenseLab, and it grows with the cell.
The paper states both: on ci-fix set C a long context recovers the rule better than SenseLab on
the dev seeds at 3× the cost, and on support only SenseLab's record moves the agent off the
stale answer at all. The E1 confirmatory cells decide whether this holds on the six other
families.

**Attribution (the one table the ablations appear in), moved-half first-attempt success:**

| cell | **SenseLab** | SenseLab − repair loop | SenseLab − canary |
|---|---|---|---|
| support s11 | 30 | 26 | 4 |
| support s23 | 25 | 55 | 5 |
| ci-fix s11 | 68 | 79 | 63 |
| ci-fix s23 | 15 | 46 | 31 |

By the (l) attribution rule the set-C gains over the baselines on these dev cells are
credited to **Tier 1, the outcome loop**: the ablation without the repair loop is equal or
higher in every cell, so the repair loop's contribution here is zero or negative. The
mechanism is the one the readout above diagnosed — no regime fix reached Tier 2 (empty case
sets, seam 4) and the one situated canary graded 2 of 15 runs (label drift, seam 5) — so the
loop added tested-but-unshipped branches and lesson entries to the record and recovered no
rule inside the window. Both seams are engine changes (#793 deployed 07:00:22Z; #794 open)
with predictions P6 / P7 in (o); `rsi-c2` is the run that tests whether the repair loop turns
this attribution positive. The paper's sentence for the dev seeds is therefore: on support SenseLab is
the only arm above 4 % on the moved half (25–30 %) and is 94–97 % on the unchanged classes;
on ci-fix the context-window baseline leads the moved half (84 / 69) at 3× SenseLab's cost,
SenseLab is second on s11 (68) and last on s23 (15); the gain SenseLab has is the outcome
loop's, and the repair loop's contribution on set C is the open question `rsi-c2` answers —
not a "SenseLab vs SenseLab-RSI" comparison.

### E7 at L0 — the two development Managed Models on held-out seeds (`e7-l0-students`, `e7-l0-frontier`)

The stand-in (o) discloses, evaluated at the ladder's first rung. **L0 is reported, not
claimed** ((m) §1: same classes, same wording as training; the claim lives at L1 and L3,
which need the eval-only content (m) specifies and which is not yet written). Two development
students, both `senselab-s1`, both trained on the dev-seed set-C support traces of one arm
(200 sealed traces, 2 cells each), both live as v1 under the **relaxed gate** — the default
gate fails both on `exact_agreement 0.000 < 0.45`, stated here as (o) requires: the
**SenseLab-trained student** (`sup-senselab-c1`, `senselab-s1-n3an6qag44m0`, teacher = the
full engine, 181/200 training successes) and the **ablation-trained student**
(`sup-norepair-c1`, `senselab-s1-ohhw29s7hz2q`, teacher = SenseLab without the repair loop,
175/200). Serving: `context_k = 3`, priors on, server retrieval, no fallback. Eval: support set
C, seeds 101 / 113 / 127, 100 episodes per cell, change at episode 30; **S0** student alone,
**S1** student with SenseLab's outcome loop as (m) specifies (the eval cell's own memory,
built from S1's sealed traces; the student writes no notes — what it reads is the seeded
record plus the engine's evidence statuses, `action_priors` and `recommendation`; nothing from
a training cell is readable), **F1** GPT-5.5 with the same outcome loop. 1 500 / 1 500
episodes, 0 cell errors; three F1 episodes in s127 lost their reflection write to a
`ReadTimeout` (fail-open, the episodes stand). Post-change first-attempt success, episodes
30–99, pooled over the three seeds:

| condition | moved half (n = 54) | kept half (n = 51) | unchanged (n = 105) | pre-change (n = 90) | $ / episode | $ / solved |
|---|---|---|---|---|---|---|
| SenseLab-trained student alone (S0) | 9 | 88 | 69 | 80 | 0.0007 | 0.0008 |
| **SenseLab-trained student + SenseLab (S1)** | **33** | 69 | 98 | 71 | 0.0015 | 0.0017 |
| ablation-trained student alone (S0) | 0 | 100 | 90 | 94 | 0.0007 | 0.0008 |
| ablation-trained student + SenseLab (S1) | 30 | 65 | 97 | 84 | 0.0017 | 0.0020 |
| GPT-5.5 + SenseLab (F1) | 22 | 69 | 99 | 80 | 0.0348 | 0.0418 |

Moved half per seed (s101 / s113 / s127; n = 16 / 21 / 17): SenseLab-trained S1 **12 / 38 /
47**, ablation-trained S1 19 / 38 / 29, F1 6 / 29 / 29, both S0 ≤ 12 in every seed.

**Reading.** (1) *The gain is the read, not the weights.* S1 − S0 on the moved half is
+24 points for the SenseLab-trained student and +30 for the ablation-trained one, in every
seed; alone, neither student takes the new action first (9 % and 0 %). The record does the
work the way it does for the frontier: ~4 discredited entries served per episode with
`avoid` set, and a `recommendation` the student follows. (2) *The two students are close,
and the direction is (m) P3's.* Paired per seed the SenseLab-trained student is −7 / 0 /
+18 points against the ablation-trained one — mean +4, sign-changing across three seeds;
"small and reported as such" is the pre-specified sentence, and it is the one this rung
supports. The attribution rule of (l) applies: the teachers' records differ (repair
trajectories and lesson entries in one, not the other) and nothing at L0 separates that
from noise. (3) *F1 < S1 at L0 is the contamination the ladder exists for.* The students
beat the frontier by 8–11 points on the moved half and match it on the unchanged classes at
**1/23 of the cost** ($0.0015 vs $0.0348 per episode); (m) P2 predicts F1 > S1 *on L3*, where
the student has nothing to have memorised, and that is where the direction is tested — an L0
student win is expected and is not evidence of anything but training on these classes.
(4) *The memorisation diagnostic is already visible in S0.* The SenseLab-trained student alone
solves 96 % of moved-half episodes (mean 2.1 attempts) with 9 % first-attempt: it has learnt
its teacher's recovery path — first the stale action, then the one the environment accepted —
and replays it without a memory read; the ablation-trained student alone is 0 % first-attempt,
44 % episode success, 100 % on the kept half and 90 % on the unchanged classes — it learnt the
*old* rule better and the recovery worse. That is H12's premise in the raw (the repair record
contains the repair) and it is exactly what L1's S0 drop is designed to measure; nothing is
claimed from it here. (5) *S1 pays on the kept half* like every SenseLab cell in this document
(65–69 vs S0's 88–100): the record marks the class contested and the student explores where
the old action was still right. (6) *Not computed at this rung:* pinned-case reproduction
(E7's third endpoint, L0 only) needs the training cells' pinned cases replayed against the
students; it is in the analysis backlog, not omitted by design. The confirmatory E7 students
remain the ones (m) specifies — trained on E1 traces after the freeze — and the ladder's L1–L3
rungs need (m)'s eval-only message sets, log variants and classes written and registered
first.

## Discipline

- Development runs: pilots and `-dev` cells on seeds **11 and 23** only, families
  `support`, `ci-fix`, `retention` (never the four held-out families).
- Every engine or agent change made from this map is logged in the Amendments section
  with its date, the mechanism it changes, and the prediction it makes; the confirmatory
  E1/E2/E3 SenseLab runs start only after `analysis-freeze-rsi`, and set B is run once.
- Confirmatory analysis reports seeds 37, 41, 59, 73 as the primary set; 11 and 23 are
  reported separately as development-exposed.
- Nothing here changes a baseline arm, a scenario, a change set, a rubric or a metric.
