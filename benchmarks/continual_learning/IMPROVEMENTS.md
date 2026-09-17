# Improvements to the continual-learning loop, and why each one exists

Written after the first regime-change run (`results/change*`, 2026-09-16) and before grid
v2. Each item names the data that motivated it, the mechanism that changed, the gate that
verifies it, and where it lives. Numbers under "motivating data" are from that run;
numbers under "verified" are from the local end-to-end run of 2026-09-17
(`preflight.py cl`, filesystem and Postgres tests) and will be re-checked on the dev API
before grid v2.

Baseline the run established, post-change, changed-class tasks, n=97 per arm: stale-pick
rate 13-16% for every memory arm with no downward trend over 8 exposures; `senselab` and
`senselab-nofeedback` indistinguishable on every metric; first-attempt success 0.54-0.62
vs 0.66 for `none`; tokens per episode 4.9k-7.3k vs 0.95k for `none`; SenseLab wall 25s vs
pgvector 10s.

## I1. Credit dilution onto generic entries

**Motivating data.** A `success` multiplied every cited key by 1.03 with no split. Per
episode `senselab` cited the three generic seed entries (`playbook-tier1`,
`repo-readme-ci`, `ops-policy-generic`) about 3x more often than `senselab-nofeedback`;
the specific lessons agents wrote were cited less. Feedback was reinforcing the playbook,
not the lesson, and the playbook's confidence saturated at 1.0.

**Mechanism.** Each outcome step distributes one unit of evidence over the keys it cites
(`w / n_keys`). A success cited to three keys gives each one third of what a solo
success gives; failures are not diluted the same way because a failed attempt usually
cites the one entry it acted on.

**Verified.** `preflight evidence`: `credit_split_gain per_split_key = [0.0534]*3`,
`solo_key = 0.1182`. Unit: `tests/unit/test_evidence_model.py`; SQL parity:
`tests/integration/test_postgres_outcome_evidence.py`.

**Where.** `packages/core/src/amfs_core/evidence.py` (`apply_record_to_entry`);
`packages/adapters/postgres/.../migrations/008_outcome_evidence.sql`
(`amfs_apply_outcome_step`); Pro copy `amfs-internal/packages/tenant/.../096_outcome_evidence.sql`.

## I2. Evidence-based confidence instead of multipliers

**Motivating data.** x0.90 per failure compounding: a 0.9 entry needed six failures to
cross the 0.5 gate, and the ranking term moved 0.018 per failure. With ~15 changed-class
tasks per cell after the change the stale entry never left the agent's context.

**Mechanism.** Recency-weighted Beta posterior: `E_s, E_f *= 0.8` per outcome, add
severity weight (success 1, minor_failure 1, failure 2, critical 3) x `causal_confidence`
x surprise `(1 + |outcome - confidence_before|)`, then
`confidence = (2 * prior + E_s) / (2 + E_s + E_f)`. `discredited_at` set below 0.5,
cleared on recovery. `AMFS_OUTCOME_MODEL=multiplicative` keeps the old behaviour.

**Verified.** `preflight evidence`: fresh 0.7 entry after one failure 0.259
(discredited); entry with 5 successes 0.898 (validated) then 0.506, 0.369, 0.293 across
three failures. An unchanged claim restated by its author keeps its record
(`inherit_evidence`; `tests/unit/test_rewrite_keeps_evidence.py`) — found when a
reflecting agent's identical end-of-task note was resetting a validated lesson to
untested every episode.

**Where.** `amfs_core/evidence.py`; migration 008; `amfs_core/engine.py` (`write` ->
`inherit_evidence`); Postgres adapters (`_inherit_from_row`).

## I3. Per-attempt credit inside one trace

**Motivating data.** Fail on the stale fix, succeed on retry, commit `success`: the stale
key was never penalised and, worse, was credited with the success. Roughly 30% of
episodes were fail-then-succeed. The `senselab-attempts` variant that split commits per
attempt never ran, and would have broken the one-trace-per-task contract the training
pipelines rely on.

**Mechanism.** `record_attempt` is a client-side boundary in the read tracker (no HTTP
call): it snapshots the reads and actions since the last boundary with the attempt's
verdict. `commit_outcome` sends `attempts=[...]` and `final_action_index` in the one
request; the server applies each failed attempt's outcome to that attempt's keys, then
the terminal outcome to the final attempt's keys. One `amfs_outcomes` row, one sealed
trace.

**Verified.** Local E2E (`/tmp/cl_e2e.py`, support `card-declined`, change at episode 4):
the post-change episode ran 3 attempts (`escalate_tier2` stale, `update_payment_method`,
`resend_email`); the stale playbook fell to 0.286 (discredited) on that one episode and
the agent's own correction was validated 5/0 by episode 9 with first-attempt success on
every later episode. `tests/unit/test_attempt_boundaries.py`;
`tests/integration/test_postgres_outcome_evidence.py::test_attempt_then_success_credits_each_side`.

**Where.** `amfs_core/engine.py` (`ReadTracker.record_attempt`), `amfs/memory.py`,
`amfs_adapter_http/adapter.py`, `amfs_http/server.py`, migration 008 (`attempts JSONB`);
MCP tool `amfs_record_attempt`; benchmark arm `arms/senselab_arm.py` (`attempt_failed`).

## I3b. Credit the claim that was read, not the key's current claim

**Motivating data.** Found closing the loop end to end with I3 in place: the agent tried
the stale lesson (v7), failed, found the fix, wrote the corrected lesson under the same
key (v8) in its reflection, then committed. The attempt's failure landed on v8 — the
correction was born discredited and the agent avoided its own fix on the next episode.

**Mechanism.** `OutcomeRecord.causal_entry_versions` and
`AttemptRecord.causal_entry_versions` carry `entry_key -> version read`. When the live
version differs and the value changed in between, that step is skipped for that key.
Restatements and propagation-opened versions still take the credit (`same_claim`).

**Verified.** `preflight claim`: rewritten key stays `untested`, `failure_count 0`,
confidence 0.7; restated key `validated 1/0`. `tests/unit/test_outcome_credits_the_claim_read.py`;
Postgres tests `test_a_rewritten_claim_is_not_charged_for_the_old_one`,
`test_an_attempt_charges_the_claim_it_tried_and_the_success_goes_to_the_fix`.

**Where.** `amfs_core/models.py`, `amfs_core/evidence.py` (`claim_still_held`),
migration 008 (`causal_entry_versions`, version check in `amfs_apply_outcome_step`),
filesystem/S3 adapters via `read_at_version`.

## I4. Evidence the agent can see

**Motivating data.** Retrieve exposed a bare `outcome_count`; `(confidence 0.87)` looked
the same for a rule validated five times and one never tested. The agent had no way to
use the signal even where it existed.

**Mechanism.** Every entry carries `evidence_status` (`untested | validated | contested |
discredited`), `success_count`, `failure_count`, `last_outcome`. The benchmark renders
`[key] (confidence 0.93; validated: 5 won / 0 failed)` for every arm that has the data
(`arms/base.py: MemoryHit.render`), and the shared system prompt tells all arms how to
read contested/discredited evidence. `pgvector-diy+outcomes` gets the same rendering
from a hand-rolled tally table so the comparison is with "build the counter yourself",
not with nothing.

**Verified.** `preflight payload`: `good -> (validated, 1, 0)`, `bad -> (discredited, 0, 1)`.

**Where.** `amfs_core/models.py` (`MemoryEntry.evidence_status`), HTTP payload fields,
`_ENTRY_PUBLIC_FIELDS` in the Pro MCP gateway, `benchmarks/.../arms/base.py`.

## I5. Evidence-aware retrieval: exclusion, avoid list, adaptive k

**Motivating data.** Post-change, the discredited entry kept surfacing at rank 1-3 because
retrieval scored on similarity + 0.2 x confidence; and when it did drop out the agent was
simply not told, so it re-derived the stale answer from the playbook.

**Mechanism.** `retrieve` drops discredited entries by default; with
`RecallConfig(include_avoid=True)` it returns them in a separate avoid list ("Do NOT rely
on these entries; they were discredited by recent outcomes"); `evidence_signal` enters
the score; `adaptive_k` shrinks k when the top hit is validated with no recent failure.
Synthetic `lesson-contrast-*` entries never surface as knowledge. Pro's
`MultiStrategyRetriever` gained an evidence strategy in RRF with competition ranking so
ties in the other strategies cannot outvote it.

**Verified.** `preflight payload`: plain retrieve returns only `good`; with `include_avoid`
`bad` comes back flagged `_avoid`. `tests/unit/test_retrieve_evidence.py`;
`amfs-internal/tests/unit/test_retrieval_evidence_strategy.py`.

**Where.** `amfs_http/server.py` (`retrieve_entries`), `amfs/memory.py` (SDK fallback,
`is_avoid`), `amfs-internal/packages/retrieval/.../engine.py`.

## I6. Briefing: validated, discredited, regime shift

**Motivating data.** Briefing digests knew nothing about outcomes. After a change the
agent opened every task with a narrative that still recommended the stale rule.

**Mechanism.** The lead digest gains `validated` (top by `success_count`), `discredited`
(with `replaced_by` where a contrast lesson exists) and `regime_shift` (>= 2 entries
with a validated history failing recently in the entity). `compact=True` returns only
these sections plus hot context, for token economy.

**Verified.** `preflight briefing`: after `rule-a`, `rule-b` (validated 3/0 each) fail
three times, `discredited = [rule-a, rule-b]`, `validated = [rule-c]`,
`regime_shift.suspected = true` with the message "2 previously validated entries in this
scope started failing recently...". Local E2E briefing at episode 10 led with "Discredited
by outcomes: playbook-tier1 (0.29)" and "Validated by outcomes: card-declined-fix (5 won)".

**Where.** `packages/cortex/src/amfs_cortex/briefing.py` (`_inject_evidence_sections`,
`_regime_shift`); Pro `cortex-pro/llm_strategy.py` carries the evidence into the LLM
narrative.

## I7. Contrastive lessons without depending on agent reflection

**Motivating data.** Every arm gets the same agent-written reflection note, and its
quality varied by episode; several post-change notes restated the stale fix because the
agent never noticed it had failed first.

**Mechanism.** On a terminal success with >= 1 failed attempt the server writes
`lesson-contrast-<ref>` under the committing agent's identity: what failed, what worked,
which keys not to rely on. Template-based, no LLM. It is folded into the briefing's
`replaced_by` for the discredited entry and excluded from retrieval as knowledge, from
training prompts, from SFT targets and from grounding sources
(`SYNTHETIC_KEY_PREFIXES`).

**Verified.** `tests/unit/test_attempt_boundaries.py` (lesson written, prefix registered);
`amfs-internal/tests/unit/test_managed_models_contract.py::TestSyntheticExclusion`.

**Where.** `amfs/memory.py` (`_commit_outcome`), `amfs_core/evidence.py`
(`is_synthetic_key`), briefing `replaced_by`, Pro dataset/gateway/exporter/grounding.

## I8. Operation and token economy

**Motivating data.** A SenseLab episode cost ~7 HTTP round-trips (briefing, retrieve, 2-4
`record_action`, write, commit) and 7.0k tokens vs pgvector's 4.9k; wall 25s vs 10s, part
of it the benchmark's own rate-limit pacing.

**Mechanism.** `record_attempt` is local; `tool_calls` are batched onto the commit;
`briefing(compact=True)`; `adaptive_k`; synthetic entries out of the context. The arm
reports `ops` per episode and subtracts pacing (`rate_limit_wait_ms`) from memory latency
so the product's latency and the benchmark's quota are reported separately.

**Verified.** Local E2E: 4 round-trips per single-attempt episode (briefing, retrieve,
write, commit); `record_action` and `record_attempt` are local. Memory latency ~160 ms
per episode against the local server. Grid v2 reports tokens and wall per arm; target senselab <= pgvector tokens per
episode.

**Where.** `arms/senselab_arm.py` (`_PacedTimer`, `_ThrottledHttpAdapter`), SDK
`commit_outcome(tool_calls=...)`, `briefing(compact=...)`.

## I9. OSS / Pro trigger parity

**Motivating data.** The OSS boot-time trigger used an explicit column list that dropped
`embedding`, `branch`, `tier`, `priority_score`, `account_id`; Pro migration 094 used
row-copy. `ci-tenant-migrate.sh` applies Pro at deploy and the OSS adapter re-applies its
body at boot when the fingerprint changes, so the two alternated. Self-hosted users had
the broken body; the hosted platform's arithmetic depended on which had run last.

**Mechanism.** One body: OSS 008 is the source; Pro 096 carries it verbatim below its
header; `schema.sql` mirrors the functions. Row-copy (`to_jsonb(cur)`) with the version
fields overridden; `account_id IS NOT DISTINCT FROM p_account`.

**Verified.** `preflight embedding`: the entry keeps rank 1 and a non-zero semantic score
after an outcome opened version 2. `amfs-internal/tests/unit/test_outcome_propagation_sql.py::test_the_tenant_body_is_byte_identical_to_the_installed_oss_migration`
and the row-copy / override-set tests.

**Where.** migration 008, `schema.sql`, Pro 096, the guard test.

## What is still open before grid v2

- All gates above pass on the local OSS server; they must pass on `amfs-api-dev` once the
  branch is deployed and a dev key is available (`preflight.py cl` with `.env` pointed at
  dev).
- The Pro `LearnedRanker` has the new evidence features but no retrained model; the RRF
  evidence strategy is what grid v2 exercises.
- Calibrator and `CalibrationDashboard` are still multiplier-shaped (follow-up).
- Judge-closed loop (B5) is not in the benchmark's path.
- A superseded version cannot be charged: when I3b skips a step because the key was
  rewritten, the failure is preserved in the outcome row and trace but no live entry
  records it. The stale claim is gone, which is the right outcome for retrieval; the
  labeled data keeps the attempt.
