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

## Discipline

- Development runs: pilots and `-dev` cells on seeds **11 and 23** only, families
  `support`, `ci-fix`, `retention` (never the four held-out families).
- Every engine or agent change made from this map is logged in the Amendments section
  with its date, the mechanism it changes, and the prediction it makes; the confirmatory
  E1/E2/E3 SenseLab runs start only after `analysis-freeze-rsi`, and set B is run once.
- Confirmatory analysis reports seeds 37, 41, 59, 73 as the primary set; 11 and 23 are
  reported separately as development-exposed.
- Nothing here changes a baseline arm, a scenario, a change set, a rubric or a metric.
