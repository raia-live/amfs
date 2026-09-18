-- Migration 010: query-conditioned ("local") evidence
--
-- An entry's evidence columns (008) sum every outcome it was ever credited
-- with, whatever the task. Grid v3's diagnose scenario showed the limit of
-- that: a rule validated on one class of task and discredited on another
-- looks "contested" to every query, so the class it still works for loses
-- it and the class it stopped working for keeps seeing it until the pooled
-- record tips. The outcome rows carry what is needed to do better — the
-- entry keys each outcome credited (`causal_entry_keys`, and per failed
-- attempt inside `attempts`) and the task embedding (`task_embedding`,
-- provisioned by the adapter at the entries' dimension). `evidence_near`
-- reads them: for the entries a retrieve is about to rank, the outcomes
-- whose task was like this query, weighted by similarity.
--
-- The read is "outcomes crediting any of these keys, nearest to this
-- vector". The keys are matched with the array overlap operator, which
-- needs a GIN index to avoid a sequential scan over the outcomes table on
-- every retrieve. `attempts` is JSONB; the keys inside it are matched by
-- unnesting in the query, bounded by the outcome rows the array index has
-- already narrowed to — so no index on the JSONB is needed.

CREATE INDEX IF NOT EXISTS idx_outcomes_causal_entry_keys
    ON amfs_outcomes USING gin (causal_entry_keys);
