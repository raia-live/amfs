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

### `rsi-c2` complete (800/800) — the repair loop on the `post-seam5` engine: Tier 2 works, Tier 3 grades, and what it ships is mostly the wrong rule

`rsi-c2` (set C, seeds 11/23, support + ci-fix, SenseLab and the no-canary ablation, 100
episodes, change at 30, focus 0.5, Tier 2 replay hub on 8787) ran 08:30Z–12:45Z on
2026-09-27, all eight cells on the `post-seam5` engine (#793 and #794 live; cut 08:27:24Z),
0 episode errors. 33 fixes were proposed across the eight agents (16 on SenseLab, 17 on the
ablation), four canaries started in-cell, nine memories shipped — all nine on the ablation.
Paired with `rsi-c1-rsi` (same cells, `pre-n`/`post-n` engine) and `rsi-c1-tier1`
(`senselab` = SenseLab − repair loop). First-attempt success, episodes 30–99, per cell; dev
seeds, small n, no significance claimed.

**Moved half (n = 23 / 20 / 19 / 13):**

| cell | SenseLab − repair loop | **SenseLab** c1 → c2 | SenseLab − canary c1 → c2 |
|---|---|---|---|
| support s11 | 26 | 30 → **39** (9/23) | 4 → 22 |
| support s23 | 55 | 25 → **40** (8/20) | 5 → 30 |
| ci-fix s11 | 79 | 68 → **63** (12/19) | 63 → 53 |
| ci-fix s23 | 46 | 15 → **46** (6/13) | 31 → 38 |

**Kept half (n = 12 / 15 / 16 / 22):** SenseLab 33 → 33, 87 → 47, 62 → 62, 68 → 82; the
ablation 50 → 42, 73 → 73, 88 → 75, 86 → 73; `senselab` 50 / 67 / 75 / 77. **Unchanged
classes:** SenseLab 100 / 100 / 94 / 100; the ablation 100 / 100 / 100 / 97; `senselab` 97 /
100 / 94 / 94. **Pre-change (episodes 0–29):** 73–93 % in every c2 cell. Cost per episode
post-change is flat c1 → c2 ($0.051 → 0.053 support, $0.045 → 0.046 ci-fix, both arms
alike); the Tier 2 replays cost $0.12–1.17 per pair of cells on top, on the hub's side.

**Reading of the outcome table.** The moved half moved up on three of four SenseLab cells
(+9, +15, +31) and down on one (−5); the no-canary ablation rose on three (+18, +25, +7) and
fell on one (−10).
Against `senselab` the picture is unchanged from c1: SenseLab is above it on support s11
(39 vs 26) and ci-fix s23 (46 vs 46, level), below on support s23 (40 vs 55) and ci-fix s11
(63 vs 79). On these n the c1 → c2 differences are within a few episodes; what changed
between the runs is the engine (seams 4 and 5), and the fix ledger below is where that
shows, not this table.

**P6 (seam 4 — a regime fix gets a Tier 2 verdict).** Zero `replay_pending: … no cases`
events in 33 fixes (in c1 it was every regime fix). **On the ablation P6 holds:** ten
regime fixes passed Tier 1, ten received a Tier 2 verdict in-cell (8 passed, 2 failed), and
every one of the eight passed replays shipped in the same cell (the ninth ship, `5d317c0a`,
came through an inconclusive Tier 1 and a passed replay); no replay passed a fix whose
replayed run reported failure. **On SenseLab P6 is falsified, by seam 7:** `0dde8588`
(support s11, `test_passed` at episode 85) had its one case delivered, replayed and sealed as
a success at 09:33:47Z and ended the cell with the case `missing`, because the running canary
`4c01d9d5` re-stamped the replay's branch (HYPOTHESIS (p)). Eight replay traces were
re-stamped in total — four under `4c01d9d5` (fixes `b66f87c1`, `8fca099f`, `0dde8588`,
`76148c9c`), four under `2b87f293` (`273dfacc`, `b96968fa` ×2, `41b0ba5d`) — and every
re-stamped one was sealed on the *canary* bucket; the 40 replays whose sessions the router
bucketed control (5) or that ran with no canary live (35) were all graded. The six
SenseLab support fixes left `test_inconclusive` with replays `requested` at cell end are
this mechanism, not missing deliveries (the hub ledger shows every delivery sealed; the
one `commit_error` in the run is a client `ReadTimeout` on the ablation's `53c3b04f`).
Amendment (q), PR amfs-internal #795, is the fix; P8 is its prediction.

**P7 (seam 5 — situation identity), per situated canary, post-start served runs; ground
truth = the harness's class tag on the run's episode; graded = the run's canonical situation
equals the fix's:**

| canary | cell | fix's situation (canonical) | recall | purity | what was missed / merged |
|---|---|---|---|---|---|
| `4c01d9d5` | support s11 | csv export timeout | 3/6 = **0.50** | 3/3 = 1.00 | `data export error` ×2, `large export failing` |
| `2b87f293` | support s23 | card declined payment | 9/9 = **1.00** | 9/9 = 1.00 | — |
| `71b44f84` | ci-fix s11 | pip audit pinned change → `audit pinned related change` | 5/11 = **0.45** | 5/7 = **0.71** | missed `pip audit requests pinned requirement` ×4, `… direct pin`, `… pinned dependency`; merged two `audit-free` runs labelled `pip audit (pinned) related (change)` |
| `5811ca2e` | ci-fix s23 | pip audit pinned related requests → `pip audit related change` | 7/8 = **0.88** | 7/7 = 1.00 | `audit pinned ci failure` |
| pooled | | | 24/34 = **0.71** | 24/26 = **0.92** | |

Against `rsi-c1-rsi`'s one situated canary (2/15 = 0.13) the rule moved recall to 0.71
pooled and 0.45–1.00 per canary; purity is 0.92 pooled. **P7 as preregistered (≥ 0.80 /
≥ 0.90) is met on two canaries and pooled on purity only; recall pooled 0.71 is short of
0.80 and above the 0.50 falsifier; `71b44f84` alone fires the recall falsifier (0.45 <
0.50) and is under the purity bar (0.71, at the 0.70 falsifier).** Two mechanisms, both
visible in the misses: (i) synonyms with no shared stem (`data export error` for `csv export
timeout`) — the offline check in HYPOTHESIS (q) shows an embedding second stage recovers
these and the token rule cannot; (ii) the canonical is the *first-seen* spelling, so
`pip audit pinned change` resolved to an earlier `audit pinned related change` and later
spellings are compared against that, not against the fix's own label — `pip audit requests
pinned requirement` shares two stems with it, under both thresholds. The merge on
`71b44f84` is the agent's own: it labelled two `audit-free` PRs `pip audit related change`,
which is the same label it uses for the pinned class, and no rule on labels separates them.
The two ci-fix canaries also show what the rule is for: with 55 and 40 post-start runs, 11
and 8 in class, the canaries graded 6 and 6 runs in 55 and 40 sessions — the situation
recurs about once in five sessions, and at `min_outcomes 8` per arm a decision needs ~80
sessions of traffic. That is the (n) latency question, unchanged by seam 5.

**K1–K4 (`post-seam5` for every cell).** *K1:* 0 of 4 in-cell canaries decided (all four
`running` at cell end; `2b87f293` was stopped by the benchmark's 60-session timeout at 5/5
graded, one recurrence, and returned to `test_passed`; the other three wait for 6–7 more
graded runs at 60 % of a cell's traffic). *K2:* proposal → canary start 4, 1, 4, 5
sessions (median 4) — every canary via a passed Tier 2 replay or the slot fill, none held
behind a replay. *K3:* graded canary/control at cell end 2/1, 5/5, 2/4, 4/2 — every regime
canary graded in both arms (5k's 0/0 is gone). *K4:* SenseLab shipped 0 in 4 cells (its
Tier 2-passed fixes are in canary or queued behind one — `a0a40f5d`, `e4ddb72a` sat
`canary_pending` on ci-fix s23 behind `5811ca2e`); the ablation shipped 1 / 1 / 3 / 4, first
post-change ship at episodes 45 / 35 / 55 / 40. One of the ablation's ships is
**pre-change**: `1196bfc0` (ci-fix s11) was proposed at episode 7 on the situation `unknown
ci failure` and shipped at episode 10 — a regime proposal on a first-exposure failure
(5k's false-positive mechanism), shipping a README-shaped note; it is counted as a false
ship. Post-first-ship, moved half, the ablation against `senselab` on the same episodes:
27 vs 27, 33 vs 61, 67 vs 83, 45 vs 55 — **0 / −28 / −17 / −9 points**; kept half −10 /
+7 / 0 / −11; unchanged classes within 4 points.

**(l) P2 — fails with fixes shipped.** The ablation shipped by episode 60 in 4 of 4 cells
(the first clause holds), and after the first ship the moved half is 27–67 %, never ≥ 85 %
and never above `senselab` paired — in three cells below it. This is the pre-specified
clause: *"if P2 fails with fixes shipped, the repair loop's synthesis adds nothing over a
capable agent with priors … the paper reports the repair loop as a validation and safety
layer, not an intelligence layer."* On this engine cut that is the paper's sentence, and
the shipped content says why. Every regime fix that passed Tier 2 passed on **one case**
(seam 6); read against set C's actual conditionals (`realworld.py`
`SUPPORT_CHANGES_C` / `CI_CHANGES_C`), the fourteen post-change regime fixes that passed
Tier 2 are: **the true conditional as the first action — 2** (`c39ec61a`, `c116e01c`,
ci-fix s23 ablation, one situation, shipped 4 s apart: "cassette in the diff → regenerate;
only `webhooks.py` → rerun once"); **the right attribute but sequenced behind the old action
— 3** (`abee5262`, `a0a40f5d`, `e4ddb72a`: "rerun first; if it persists *and* the PR touches
a cassette, regenerate" — right on episode success, wrong on first attempt on the moved
half); **not the conditional — 9**: restatements of the pre-change answer (`5d317c0a`
split-the-range for export timeout, `0a0c2b9a` settings steps for notifications, `4c01d9d5`
workaround-then-escalate), unconditional flips (`71b44f84`, `5811ca2e` "bump, do not add the
exception" — right on the moved half, wrong for every `legacy_sdk/` PR), sequential hedges
(`6e3dcf9b`, `6236e8e2` "try one, if rejected the other"; `2b87f293` "update card, refresh
session, then escalate" — neither `resend_email` nor the issuer), and one conditional on the
wrong attribute (`2d4d0cac`: "pin *and* related code in the diff → exception" — both halves
of set C's PRs have both). 9/14 do not encode the rule and 12/14 do not act on it first.
The one arm-level success — ci-fix s23's two conditional ships at episode 55 — is followed
by a moved half of 45 % (5/11) against `senselab`'s 55 %: the record also holds `2d4d0cac`,
shipped at 40 with the wrong attribute for the other class, and the agent reads both.

**Two further findings.** (1) **Seam 8, not yet a PR candidate until Bruno says so:**
`4c01d9d5` — the support s11 canary — was drafted from trace `2813e38b`, which is not a
served run: it is the Tier 2 replay of judge fix `0cb64b9f`'s case, sealed on
`repair/0cb64b9f` with a failing outcome. The regime-shift pass reads a failing seal on an
entry the run read, and a replay's failure on a fix branch counts to it as the entry
failing on main. The loop proposed a regime fix from its own test of another fix; the
`_TRAFFIC_SQL` rule #795 applies to the canary tally is the rule the regime pass needs on
its trigger. (2) The support s23 judge fix `273dfacc` entered **canary after the cell**
(12:20Z, pct 10, via the sweep's slot fill once its lost replay expired), on an agent that
seals nothing; it will time out at 24 h with 0/0. Left untouched, as every prod canary is.

**What the paper takes from `rsi-c2` (dev seeds).** Seams 4 and 5 did what their
predictions said within the stated bars on the ablation and on two of four canaries: a
regime fix now gets a Tier 2 verdict, ships under the replay policy, and a situated canary
grades 0.45–1.00 of its class instead of 0.13. What the run then exposes is the next layer
down: (a) a one-case Tier 2 passes the wrong rule 9 times in 14 (seam 6, in #795 with P9);
(b) a running canary swallows later replays (seam 7, in #795 with P8); (c) the token rule
misses synonyms and inherits its first spelling (seam 5b, offline result in (q), decision
pending); (d) the regime trigger reads the loop's own replays as traffic (seam 8, above).
The attribution table's sentence does not change on this cut: the repair loop's
contribution to the moved half is zero or negative on 3 of 4 cells, and (l)'s P2 clause
fires *with fixes shipped*. The paper reports that as the finding for this engine version,
with the content analysis above as the mechanism, and `rsi-c3` on the #795 cut as the test
of whether a proof set that covers both halves changes it.

### `rsi-c3` complete (800/800) — the repair loop on the `post-seam5b` engine: the proof set is built, and the outcome it grades cannot see the conditional

`rsi-c3` is `rsi-c2`'s command with the run name changed (set C, seeds 11/23, support +
ci-fix, SenseLab and the no-canary ablation, 100 episodes, change at 30, focus 0.5, Tier 2
hub on 8787 through the same tunnel). It ran 13:40Z–18:10Z on 2026-09-27, all eight cells on
the `post-seam5b` engine (#795 and #796 live: seams 4, 5, 5b, 6, 7; cut 13:39:13Z). 800/800
episodes; 4 fail-open reflection `ReadTimeout`s (one per ablation cell, the episode completed)
and 2 client-side seal `ReadTimeout`s whose traces the server did persist. 40 fixes were
proposed (24 SenseLab, 16 ablation; 32 regime, 8 judge), 149 Tier 2 replays were delivered
and all but one sealed (48 in `rsi-c2` — the proof set is what grew), six canaries started in-cell on
SenseLab (four regime, two judge), 11 memories shipped — all 11 on the ablation, as in c2.
Paired with `rsi-c2` and `rsi-c1-tier1` (`senselab` = SenseLab − repair loop). First-attempt
success, episodes 30–99; dev seeds, small n, no significance claimed.

**Moved half (n = 23 / 20 / 19 / 13):**

| cell | SenseLab − repair loop | **SenseLab** c2 → c3 | SenseLab − canary c2 → c3 |
|---|---|---|---|
| support s11 | 26 | 39 → **30** (7/23) | 22 → 9 |
| support s23 | 55 | 40 → **45** (9/20) | 30 → 15 |
| ci-fix s11 | 79 | 63 → **58** (11/19) | 53 → 68 |
| ci-fix s23 | 46 | 46 → **38** (5/13) | 38 → 54 |

**Kept half (n = 12 / 15 / 16 / 22):** SenseLab 33 → 33, 47 → 53, 62 → 62, 82 → 77; the
ablation 42 → 50, 73 → 60, 75 → 88, 73 → 86; `senselab` 50 / 67 / 75 / 77. **Unchanged
classes:** SenseLab 100 / 100 / 97 / 97; the ablation 97 / 100 / 100 / 94. **Pre-change
(episodes 0–29):** 67–90 % in every c3 cell. Cost per episode is flat c2 → c3 ($25.04 →
$24.75 for the 800); the Tier 2 replays cost $6.08 on the hub's side against $2.49 — the
price of a 3–6-case proof set, at $0.20–2.05 per cell.

**Reading of the outcome table.** The moved half is level or down on three of four SenseLab
cells (−9, +5, −5, −8) and, on the ablation, down sharply on support (−13, −15) and up on
ci-fix (+15, +16). Against `senselab`, SenseLab is above it on support s11 (30 vs 26) and
below on the other three (45 vs 55, 58 vs 79, 38 vs 46); the ablation is below it on three and
above on ci-fix s23 (54 vs 46). **(l)'s P2 clause fires again, with fixes shipped**, on both
arms. The mechanism is in the ledger below.

**P8 (seam 7 — replays during a canary): met.** 39 replays were sealed while `a9a53641`'s
canary was running on support s23 and 33 while `d27a641f`'s ran on ci-fix s23; the router
re-stamped 26 of them onto the canary's branch (19 and 7; the OSS root cause is untouched,
as stated), and every one was graded on the replayed fix's branch — the eight support s23
fixes with a re-stamped replay all received Tier 2 verdicts (`63098274`, `e5de8605`,
`35676701`, `faeaf921`, `87bf3b8a` failed; `a6cac7e6`, `46da73a6`, `f694a0cd` passed), as did
the five on ci-fix s23, where in c2 they would have ended `missing`. The
one `missing` case in the run (`76b09639`, case `ae6fd09c`) has no sealed trace anywhere: the
replay episode itself raised `ReadTimeout` and the hub ledger shows `trace_id: None`.
`a9a53641`'s tally ended 25 canary / 30 control served runs with 19 canary-stamped replays
excluded; `d27a641f`'s 30 / 25 equals the served-run count exactly. No replay trace is in any
canary tally. Both falsifiers clear.

**P9 (seam 6 — the proof set).** 31 regime fixes reached Tier 2 with a case set; sizes 1 ×1,
2 ×2, 3 ×5, 4 ×17, 5 ×4, 6 ×2. **(a) met:** the one one-case set (`8e4f98ca`, support
ablation s11, `mobile full data export failure`) is on a canonical situation with no earlier
served run on that agent; every other set has ≥ 2. **(c) met:** two regime fixes failed Tier 2
on a kept case — `4667af4b` (support s11, 1 kept + 3 failed cases failed) and `87bf3b8a`
(support s23, 6/6 failed, including all three kept cases) — and four more failed only on
their drafted or failed-run cases (`7e6b87a7`, `d7e91556`, `e5de8605`, `faeaf921`). All six
Tier 2 failures are on support; **ci-fix had 19 regime fixes reach Tier 2 and 19 pass.**
**(b) falsified, and worse than the baseline.** The corrections of the 24 post-change regime
fixes that passed Tier 2 (`a4e16179`, proposed at episode 8, is pre-change and excluded, like
c2's `1196bfc0`), classified against `SUPPORT_CHANGES_C` / `CI_CHANGES_C` before their Tier 2
records were read:

| reading | support (6) | ci-fix (18) | all |
|---|---|---|---|
| conditional on the right attribute, as the first action | 0 | 1 (`c1c5b3ad`: cassette in the diff → regenerate, else rerun) | **1/24** |
| right attribute named, but not decisive or not first (`cb8c7221` code fix before the rerun; `0ab5cc39`, `126ff64c` name `legacy SDK/client` behind an OR on the PR title, which both halves carry) | 0 | 3 | 3/24 |
| conditional on the wrong attribute (`816ffd48`, `fc4a65d9`, `ce242c82`: "manifest-only vs has code" — every set C PR has code) | 0 | 3 | 3/24 |
| unconditional flip to the moved half's answer (bump) | 0 | 11 | 11/24 |
| restatement of the pre-change answer or a sequential hedge without the condition (smaller ranges → web export → escalate; reinstall → settings) | 6 | 0 | 6/24 |

**20/24 (83 %) do not encode the conditional and 23/24 do not act on it first**, against
c2's 9/14 and 12/14. Five of the ci-fix s23 flips state the anti-rule outright — `28a3bcb8`
"do not add an audit exception simply because the PR title mentions … legacy-SDK",
`66a04308`, `6f993614`, `b20c20aa`, `3d349afe` likewise — and every one passed a 3–4-case
proof set that included kept-half runs where the exception is the right answer.

**Why the proof set passed them: the outcome it grades.** The cases are graded on the
replayed run's `outcome_type` against `expected_outcome: success`, and this customer's
outcome is *episode* success within three attempts. On the 280 post-change runs of the
changed classes (both arms, prod traces joined to the ledger): the kept half reports
`success` on 118/130 (87 first-attempt, 23 second, 8 third) and the moved half on 108/150
(57 / 39 / 12). A flipped rule on a kept-half PR bumps, is rejected, adds the exception and
seals `success`; the case passes. The proof set can only fail a fix that makes the agent
lose the episode outright — and on ci-fix, where there is no escalation and every cell is at
99–100 % episode success, that never happens. On support, where escalating is a failure, it
happened six times. **The corollary for the proof set's composition:** 25 of the 31 sets are
{the drafted run + kept-half passes}; only six carry any other failed run, because a regime
fix is proposed at the situation's *first* outright failure, before others exist, and the
moved-half runs that took two attempts are `success` to the engine. The conditional set C
encodes is visible in first-attempt success — which the engine never receives. This is a
finding about what the customer's outcome signal carries, not about case selection; the
remedy is a decision for Bruno (options in the message, none taken): the harness reporting a
second-attempt success as `minor_failure` changes a preregistered outcome definition for
every arm; the engine grading a proof case on the attempt count the SDK already stamps
changes the customer contract. Neither is in this run.

**P7′ (seam 5b — the embedding second stage), per situated regime canary, post-start served
runs, ground truth = the class tag; graded = canonical situation equals the fix's:**

| canary | cell | fix's situation → canonical | recall | purity | what was missed / merged |
|---|---|---|---|---|---|
| `a9a53641` | support s23 | data export timeout → `large export failure` | 6/10 = **0.60** | 6/6 = 1.00 | `ios pro all data export timeout`, `ios all data export error`, `web data export error`, `android data export failure` — all four went to a second export canonical, `data export android`, by the **token rule** (containment), which runs first |
| `a4e16179` | ci-fix s11 | pip audit failure (pre-change) | 9/21 = **0.43** | 9/13 = **0.69** | missed `pip audit pinned dependency` ×10 → own canonical `audit pinned dependency` by the token rule; merged four `audit-free` runs the agent labelled `(pip) audit (ci) failure` |
| `cb8c7221` | ci-fix s11 | flaky integration ci failure → **`type error ci`** | 1/2 = 0.50 | 1/2 = **0.50** | joined `type error ci` **by embedding** (cosine 0.793 ≥ 0.75); only 2 in-class runs post-start |
| `d27a641f` | ci-fix s23 | pip audit failure → **`audit check failure`** | 2/12 = **0.17** | 2/6 = **0.33** | `audit check failure` absorbed `format check failure` ×8 (cosine 0.793) and `type check failure` **by embedding**; the moved half's `pip audit pinned dependency` ×4 and five other spellings sit in `audit pinned related change` |
| pooled | | | 18/45 = **0.40** | 18/27 = **0.67** | |

**P7′ is falsified on all three falsifiers** (pooled recall < 0.80, two canaries under 0.50,
pooled purity < 0.90), and both numbers are below c2's token-rule-only 0.71 / 0.92. The two
mechanisms are visible in the alias table: (i) the embedding stage over-merges short generic
CI labels — with the in-image model, `flaky integration ci failure` ~ `type error ci` is
0.793 and `format check failure` ~ `audit check failure` 0.793, both above the 0.75 floor
chosen on the support export in (q), where no such pairs exist; on ci-agent `253808283`,
`unknown ci failure` absorbed `type error ci`, `integration flaky ci failure` and `format ci
failure` the same way. (ii) The token rule runs first and pre-empts a better embedding match:
`android data export failure` contains `data export android` (a canonical created earlier)
and joins it at containment 1.0, although it is 0.830 from `large export failure` and the
first-seen-canonical effect from c2 is thereby kept. The in-sample caveat in (r) cuts the
other way from the one anticipated: the floor was chosen on support labels and fails on
ci-fix labels in the same seeds. **No change is made.** The stage has an off switch
(`AMFS_EVAL_SITUATION_EMBED=0` on the service) and a floor variable; raising the floor to 0.80
would cut the two 0.793 false joins and keep `pip audit failure` ~ `audit check failure`
(0.808) and `pip audit pinned dependency` ~ `pip audit failure` (0.847), on a margin of
0.01–0.02 — a number to distrust on this evidence. Both are Bruno's call.

**Seam 8, now the largest untreated seam.** In c2 one fix was drafted from the loop's own
replay; in c3, with 149 replays instead of 48, **13 of 40 fixes** (10 regime, 3 judge) have a
`manifest.predicted_fixes` trace carrying `replay_delivery_id`: on ci-fix s23 SenseLab the
chain `3d349afe` → (its replay) → `66a04308` → (its replay) → `6f993614`, and `b20c20aa` →
`ce242c82`, `fc4a65d9` → `126ff64c`; on the ablation `2b326866` → `816ffd48` → `dbe883f4` +
`1931f1c3` and `c6906527` → `fae636c0`, **four of which shipped**; on support s23 the judge
fix `f694a0cd` (now in canary) was drafted from two `critical_failure` replays of `87bf3b8a`
sealed on `repair/a9a53641`. Eight of the 24 passed regime fixes above were drafted from a
replay. Seam 6 amplified it: a bigger proof set is more replays, and the regime and judge
triggers read every one as served traffic. (A side effect visible in the export: the Tier 1
run of a fix drafted from a replay replays the replay's task and inherits its
`replay_delivery_id` — nine such `origin` traces, harmless to the tallies, which exclude
either flag.) The rule #795 gave the canary tally
(`_TRAFFIC_SQL`) is the rule both triggers need; (q)'s P10 stands as the prediction if Bruno
approves it. Not built.

**K1–K4, briefly.** K2 (proposal → canary start) on SenseLab: 5, 4, 7, 3 sessions for
`a9a53641`, `cb8c7221`, `a4e16179`, `d27a641f`. K3 at cell end: `a9a53641` graded 4 canary / 2
control before its 60-session timeout, `a4e16179` 8 / 1, `d27a641f` 5 / 1 (running), `cb8c7221`
0 / 2 (running); none reached the 8 / 8 decision bar in-cell — at a 50 % split and 1–2
in-situation runs per 10 sessions the canary needs more than a 100-episode cell, as in c2.
K4, moved half after the ablation's first ship vs the other arm on the same episodes: support
s11 2/15 vs 6/15, s23 3/16 vs 9/16; ci-fix s11 10/12 vs 9/12, s23 3/5 vs 2/5 — the support
ships hurt, the ci-fix ships (flips, on a moved half whose answer is the flip) helped there
and cost the kept half nothing the outcome can see.

**What the paper takes from `rsi-c3` (dev seeds).** Seams 6 and 7 did what they were built
to do at the mechanism level: P8 met on both falsifiers, the case set went from 1 to 3–6, and
Tier 2 failed six regime fixes it would have passed on one case, two of them on the kept
half (P9 a, c). The headline prediction, P9(b), is falsified in the direction opposite to the
one predicted — 83 % non-conditional against 64 % — and the mechanism is now identified
exactly: **the outcome a customer reports is coarser than the behaviour the fix changes**;
episode success after retries is `success`, so the proof set grades a flipped rule as
harmless wherever the agent can recover, which on ci-fix is always. Seam 5b, deployed on an
in-sample offline result, made situation identity worse on the family it was not checked on
(P7′ falsified on every falsifier) and is left switched on pending Bruno's decision. Seam 8
grew from one fix to thirteen because seam 6 multiplied the replays it feeds on. The
attribution table's sentence does not change: the repair loop's contribution to the moved
half is zero or negative on three of four cells for the third run on the third engine cut,
and (l)'s P2 clause fires with fixes shipped. What has changed is the explanation: it is no
longer "the proof is one case" but "the proof grades the wrong signal", which is a statement
about the outcome contract between the customer and the loop, and a candidate for the
paper's central negative result.

## Discipline

- Development runs: pilots and `-dev` cells on seeds **11 and 23** only, families
  `support`, `ci-fix`, `retention` (never the four held-out families).
- Every engine or agent change made from this map is logged in the Amendments section
  with its date, the mechanism it changes, and the prediction it makes; the confirmatory
  E1/E2/E3 SenseLab runs start only after `analysis-freeze-rsi`, and set B is run once.
- Confirmatory analysis reports seeds 37, 41, 59, 73 as the primary set; 11 and 23 are
  reported separately as development-exposed.
- Nothing here changes a baseline arm, a scenario, a change set, a rubric or a metric.

## 2026-09-28 — exploratory: what the replay gate alone let through (amendment (t))

`analysis/rsi_posthoc_ships.py` over `rsi-c1-rsi`, `rsi-c2`, `rsi-c3` (report:
`results/rsi-report/POSTHOC_SHIPS.md`). The no-canary ablation shipped 21 regime fixes over nine
cells, every one past Tier 1 and Tier 2; the paired `senselab-rsi` cell shipped nothing. Window:
first ship in the cell to its end; recurrence = not a first-attempt success (the canary's
reading). Kept half: ablation 13/46 (28 %) before → 29/99 (29 %) after; SenseLab over the same
episodes 11/46 (24 %) → 43/99 (43 %); DiD −18 pp (ships associated with *less* kept-half
recurrence). Moved half: ablation 50/59 (85 %) → 68/111 (61 %); SenseLab 46/59 (78 %) → 52/111
(47 %); DiD +8 pp (ships associated with less moved-half recovery). Reading: the shipped fixes
were the old rule restated (P9(b): 20/24 non-conditional) — they protect the kept half and slow
the moved half; and the kept-half loss in the headline table is level 0 demoting an entry for
everyone when it fails for half. Not evidence for or against the canary as a false-ship gate: on
net these ships did not damage the situation, so a tally over the situation would have found little
to stop. H3/H4 are to be read in E2 (canaries decide) and against the adversarial probe (net
damage by construction). Not preregistered; reported in the paper as §4.3, labelled exploratory.

## 2026-09-29 — `rsi-c5` readout (cut `post-clean-success`, #797 + #798; amendment (t))

800/800 episodes, 7 infra errors; support + ci-fix × seeds 11/23 × {SenseLab, SenseLab − canary};
launched 11:35Z, finished 16:24Z. Report: `results/rsi-report/REPORT.md` (`--senselab-run rsi-c5`).

**Headline (SenseLab, 4 cells).** Moved half 34 % (25/74) [17, 56]; kept 63 % (41/65) [29, 77];
unchanged 96 % (132/137); escalated 8 %; $0.033/ep. Baselines unchanged (frontier 17, pgvector
17, +rerank 5, DIY 19, context window 33 at $0.070). Paired moved-half deltas: SenseLab wins 4/4
cells against frontier (+16), pgvector (+17), +rerank (+30), DIY (+15); 2/4 against the context
window (mean −3: +17/+16 support, −21/−23 ci-fix). By family: support moved 17 % (7/42), kept
44 % (12/27); ci-fix moved 56 % (18/32), kept 76 % (29/38). Ablation − canary: moved 37 %, kept
75 %; by family ci-fix 62/84, support 17/63. Level 0 alone (rsi-c1-tier1): 51/69. Compared with
`rsi-c3` (43/60) SenseLab's moved half is 9 pp lower on the same seeds; the interval covers both.

**Lifecycle.** SenseLab: 16 proposed (13 regime, 3 judge); Tier 1 9 pass / 7 inconclusive / 0
fail; **Tier 2 3 pass / 13 fail**; 3 canaries started, 3 undecided at ep 100 (canary-arm graded
0–1, control 0–6, floor 8); 0 shipped. Ablation: 17 proposed (13 regime); Tier 2 2 pass / 12 fail
/ 3 open; **2 shipped** (`repo-readme-ci`, regime, ci-fix s11; one judge fix, support s11).
`rsi-c3` for comparison: SenseLab 24 proposed, Tier 2 15 pass / 8 fail; ablation shipped 11/16.

**P10 (seam 8) — met.** Of the 47 runs cited by the 33 fix manifests created after the #798 cut,
0 carry `replay_delivery_id` or `origin=repair` (all 47 joined to traces); 134 of the 928
sessions sealed in the window were replays. Alias rows since the cut: 364; the seal hook returns
`not traffic` for a replay before situation identity runs, so none can come from a replay.

**P11 — (a) falsified, (b) met, (c) met.** Regime fixes reaching Tier 2: 23 (SenseLab 13,
ablation 10 + 3 still open). Failed 19: **5 on a kept case** (SenseLab support 1, ci-fix 2;
ablation support 1, ci-fix 1), 10 on a moved case only, 4 on a pre-change/unchanged case only.
Passed 4 about the changed class (+1 `procedure-verify-before-refund-double-billed`, a regime
shift on an unchanged class): `pip-audit-pinned-related-add-exception-requests` (SenseLab ci-fix
s11, canary), `pip-audit-pinned-dependency-add-exception-after-source-bump-fails` (SenseLab ci-fix
s23, canary), `repo-readme-ci` (ablation ci-fix s11, shipped), and the support canary fix. Hand-
coded against `CI_CHANGES_C["audit-pinned"]` (touches `legacy_sdk/` → keep pin + exception; else
bump): all three ci-fix passers say "bump, exception only if policy permits" — unconditional Y;
`repo-readme-ci` also restates "rerun only for clear flakes" unconditionally. (a) 4/4 = 100 % ≥
70 % → falsified. (b) ci-fix kept-case Tier 2 failures: 3 ≥ 1 → met. (c) ships 11 → 2 → met.
Reading: the proof now rejects both ways a draft misses a conditional — the old rule restated
fails its moved case; the wholesale flip fails its kept case — and the survivors are wholesale
flips whose kept-half replays passed cleanly because the replaying agent did not follow the entry
into the wrong action. The conditional is lost in the repair agent's draft, not in the proof. Not
done here: a change to the drafting prompt (engine change; proposal to Bruno, not made).

**P8/P9 re-read.** P8: every regime fix's case set has 3–7 cases mixing failed and passed runs
of the situation (kept, moved, pre) ✓. P9(b) on this cut: 4/4 non-conditional among passers (see
P11(a)). P7′ not re-read in this pass.

**Paper.** Refilled from `rsi-c5` (Tables 1–5, 7; caption cut label `post-clean-success`);
abstract/§4.1/§4.2/§5 ledger/§6 updated to these numbers and readouts.

## 2026-09-29 — `rsi-e2b-c5` readout (E2b dev run on `post-clean-success`; amendment (t)) — exploratory

2,400/2,400 episodes, 2 infra errors; support + ci-fix × seeds 11/23 × {SenseLab, − canary,
− replay}; fleet 10, 200 episodes, change at 50 (set C). Launched 21:40Z, finished 07:2xZ.

**First-attempt success, pooled over seeds (episodes 50–199; `@130+` = episodes 130–199).**
ci-fix: SenseLab moved 70 % (46/66), kept 90 % (76/84), unchanged 99 %, @130+ moved 97 % (28/29)
kept 96 %; − canary 77 / 83 / 100, @130+ 93 / 94; − replay 77 / 85 / 100, @130+ 100 / 96. support:
SenseLab moved 29 % (24/82), kept 62 % (42/68), unchanged 100 %, @130+ moved 41 % (15/37) kept
64 %; − canary 32 / 51, @130+ 46 / 53; − replay 26 / 63, @130+ 35 / 64. On ci-fix every arm
recovers both halves by episode 130 with 10 agents sharing the store; on support none does. The
three arms are indistinguishable on both families at this sample.

**Ships.** − canary 4 (support s11: 1 judge fix; ci-fix s11: 1 regime; ci-fix s23: 2 regime);
SenseLab 0; − replay 0. Tier 2 (SenseLab cells): support 4 passed / 17 failed / 1 inconclusive;
ci-fix 2 passed / 4 failed. − replay (no Tier 2): Tier 1 passed 11, inconclusive 10.

**Canaries: 20 started, 0 decided.** 13 stopped by the benchmark timeout (60 sessions), 3 ended
`inconclusive: ungraded`, 4 still running at episode 200. Tallies at stop (canary / control):
traces 22 / 33 graded 1 / 0; 29 / 31 graded 4 / 1; 25 / 30 graded 25 / 30 (recurred 3 / 5 — both
arms past `min_outcomes`, recurrence 12 % vs 17 %, no significant difference, no decision within
the timeout); 30 / 25 graded 8 / 5; 16 / 14 graded 0 / 0 ("none declared it"); 17 / 12 and 19 / 11
graded 0 / 0; 29 / 25 graded 2 / 1; 28 / 26 graded 7 / 11; 27 / 28 graded 4 / 6; 7 / 46 graded
2 / 2; 3 / 52 graded 3 / 52; 6 / 48 graded 1 / 3; 8 / 46 graded 1 / 13.

**Finding 11 — the canary grades by canonical situation, and the agent's declarations fragment
into several canonicals.** `graded_by` reads "graded on outcomes of runs in the situation
'<canonical>'". Every run declares a situation (200/200 in the ci-fix s11 − replay cell), in the
agent's own words and often carrying the very condition the change turns on: 'pip audit requests
pin with legacy sdk' (15), 'pip audit pinned related change' (11), 'pip audit related change' (9),
'pip audit requirements plus nonclient code' (6), 'pip audit pinned dependency' (5), … Situation
identity (seams 5/5b) folded 56 declarations of that agent into 10 canonicals, and the pip-audit
declarations to the canonical **'pip audit failure'** — while the four pip-audit fixes carry the
canonicals 'pip audit pinned change', 'pip audit pinned dependency' and 'pip audit pinned related
change', assigned when each was proposed. Exposure is counted (traces = runs that read the entry:
16–30 per arm) but grading requires the run's canonical to equal the fix's, so 0–8 of 25–30
exposed runs are graded and the note says "none declared it; there is nothing here for the canary
to read". This — not the horizon — is why no canary in `rsi-c3`, `rsi-c5` or this run reached a
decision except the one whose declarations happened to be stable ('payment declined checkout',
25/30 graded), and that one found no significant difference. Two engine questions for Bruno, not
acted on: (i) should a run that read the fix's entry and reported an outcome count as graded
whatever its declared canonical (exposure ∪ situation), and (ii) should situation identity merge
canonicals within an agent whose declarations it already folds to different targets. A third
observation for the design: the decision rule waits for a *significant* difference; a fix that is
merely not worse never ships inside the timeout.

**Hand-coding of Tier-2 passers / ships (P9(b) reading):** deferred to the v0.2 pass with two
coders; 15 fixes (12 `test_passed`, 3 regime ships).
