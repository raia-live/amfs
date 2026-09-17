-- Migration 008: evidence-based confidence, per-attempt credit, discredit flag
--
-- 006 fixed the direction of the outcome multipliers and 007 fixed what the
-- new version copies. This migration changes what an outcome *means* to an
-- entry, because with the multipliers the answer was: almost nothing.
--
--   * A stale entry at 0.9 needed six straight failures (0.9 * 0.90^6 ≈ 0.48)
--     to fall under a 0.5 retrieval gate. An agent that hits a stale fix
--     retries around it and commits `success` for the task, so the failure
--     was not even counted — the stale entry was *reinforced*.
--   * Every entry cited in a successful session took the full *1.03, so an
--     entry read in every session climbed to 1.0 and stayed there regardless
--     of whether it ever contributed anything.
--
-- The replacement is a recency-weighted Beta posterior, computed identically
-- in amfs_core/evidence.py (the filesystem and S3 adapters) and here:
--
--     confidence = (2 * prior + E_s) / (2 + E_s + E_f)
--
-- `prior` is the confidence the author wrote (stored once in
-- prior_confidence). E_s / E_f are evidence masses; both decay by 0.8 on every
-- update so the last few outcomes dominate, and each outcome adds
--
--     w = severity(type) * causal_confidence / n_causal_keys * (1 + |target - confidence|)
--
-- to one of them. severity is 1.0 for success, 1.5 / 2.0 / 3.0 for minor /
-- plain / critical failure. Dividing by the number of keys the step cites is
-- the credit split. The last factor is the surprise: a failure on an entry
-- trusted at 0.95 weighs nearly twice one on an entry at 0.5.
--
-- A failure that leaves the posterior under 0.5 stamps discredited_at; read
-- paths exclude those rows by default and briefings list them as anti-
-- patterns. Evidence that lifts the posterior back over 0.5 clears the stamp.
--
-- Per-attempt credit: amfs_outcomes gains `attempts JSONB`, an ordered list
-- of {attempt, outcome_type, causal_entry_keys, action_indices, summary} for
-- the failed attempts that preceded the terminal outcome. The trigger applies
-- each attempt's outcome to that attempt's keys first, then the terminal
-- outcome to causal_entry_keys. One outcome row, one trace, one terminal
-- label — but the entries that led the agent astray now receive the failure.
-- `final_action_index` names which tool call produced the terminal outcome so
-- training pipelines stop guessing "the first one".
--
-- The claim that was read is the claim that is credited: amfs_outcomes gains
-- `causal_entry_versions JSONB` ({entry_key: version read}), and each attempt
-- carries the same map. When the live version of a cited key differs from the
-- one the agent read, the step compares the two values; if the claim changed
-- in between (the agent's own reflection rewrote the key with a new lesson
-- before committing, or a colleague did), the outcome is not applied to the
-- new claim. Without this, an agent that tried a stale lesson, succeeded by
-- another route, wrote the better lesson under the same key and then
-- committed handed the stale lesson's failure to its replacement.
--
-- AMFS_OUTCOME_MODEL=multiplicative (the adapter sets the `amfs.outcome_model`
-- session setting) restores the 006 arithmetic; the evidence columns are still
-- maintained so the status vocabulary keeps working.
--
-- Must stay identical to schema.sql and PostgresAdapter._apply_migrations. All
-- three are CREATE OR REPLACE and whichever runs last wins.

ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS account_id UUID;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS success_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS failure_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS evidence_success NUMERIC(10,4) NOT NULL DEFAULT 0;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS evidence_failure NUMERIC(10,4) NOT NULL DEFAULT 0;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS prior_confidence NUMERIC(6,4);
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS last_outcome TEXT;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS last_outcome_at TIMESTAMPTZ;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS discredited_at TIMESTAMPTZ;

ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS account_id UUID;
ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS attempts JSONB NOT NULL DEFAULT '[]';
ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS final_action_index INTEGER;
ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS causal_entry_versions JSONB NOT NULL DEFAULT '{}';

-- Discredited rows are excluded from most reads; the partial index keeps the
-- "show me what stopped working" queries cheap on large tables.
CREATE INDEX IF NOT EXISTS idx_entries_discredited
    ON amfs_memory_entries (namespace, entity_path, discredited_at)
    WHERE discredited_at IS NOT NULL AND superseded_at IS NULL;

CREATE OR REPLACE FUNCTION amfs_outcome_is_success(t TEXT) RETURNS BOOLEAN AS $$
    SELECT t IN ('success', 'clean_deploy');
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION amfs_outcome_is_known(t TEXT) RETURNS BOOLEAN AS $$
    SELECT t IN ('success', 'clean_deploy', 'minor_failure', 'regression',
                 'failure', 'p2_incident', 'critical_failure', 'p1_incident');
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION amfs_outcome_severity(t TEXT) RETURNS NUMERIC AS $$
    SELECT CASE t
        WHEN 'success' THEN 1.0
        WHEN 'clean_deploy' THEN 1.0
        WHEN 'minor_failure' THEN 1.5
        WHEN 'regression' THEN 1.5
        WHEN 'failure' THEN 2.0
        WHEN 'p2_incident' THEN 2.0
        WHEN 'critical_failure' THEN 3.0
        WHEN 'p1_incident' THEN 3.0
        ELSE 1.0
    END;
$$ LANGUAGE sql IMMUTABLE;

-- The 006 multipliers, kept for AMFS_OUTCOME_MODEL=multiplicative.
CREATE OR REPLACE FUNCTION amfs_outcome_multiplier(t TEXT) RETURNS NUMERIC AS $$
    SELECT CASE t
        WHEN 'critical_failure' THEN 0.85
        WHEN 'failure' THEN 0.90
        WHEN 'minor_failure' THEN 0.92
        WHEN 'success' THEN 1.03
        WHEN 'p1_incident' THEN 0.85
        WHEN 'p2_incident' THEN 0.90
        WHEN 'regression' THEN 0.92
        WHEN 'clean_deploy' THEN 1.03
        ELSE 1.0
    END;
$$ LANGUAGE sql IMMUTABLE;

-- One step of an outcome: one outcome type applied to one set of causal keys,
-- credit-split over those keys. The trigger calls this once per failed
-- attempt and once for the terminal outcome.
-- The six-argument form predates causal_entry_versions; the trigger below
-- calls the seven-argument one, so the old overload is retired here.
DROP FUNCTION IF EXISTS amfs_apply_outcome_step(TEXT, UUID, TEXT[], TEXT, NUMERIC, TEXT);

CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(
    p_namespace TEXT,
    p_account UUID,
    p_keys TEXT[],
    p_outcome_type TEXT,
    p_causal_confidence NUMERIC,
    p_model TEXT,
    p_versions JSONB
) RETURNS INTEGER AS $$
DECLARE
    keys TEXT[];
    entry_key TEXT;
    ep TEXT;
    k TEXT;
    cur RECORD;
    read_version INTEGER;
    read_value JSONB;
    n_keys INTEGER;
    is_ok BOOLEAN;
    target NUMERIC;
    w NUMERIC;
    prior NUMERIC;
    e_s NUMERIC;
    e_f NUMERIC;
    new_conf NUMERIC;
    disc TIMESTAMPTZ;
    touched INTEGER := 0;
BEGIN
    -- Distinct, well-formed specs only; a key cited twice is one citation.
    SELECT COALESCE(array_agg(DISTINCT x), '{}') INTO keys
    FROM unnest(COALESCE(p_keys, '{}')) AS x
    WHERE position('/' in x) > 0;
    n_keys := cardinality(keys);
    IF n_keys = 0 THEN
        RETURN 0;
    END IF;

    -- An outcome type neither side of the model knows is not evidence of
    -- anything (the multipliers treated it as x1.0; this treats it as no step).
    IF NOT amfs_outcome_is_known(p_outcome_type) THEN
        RETURN 0;
    END IF;
    is_ok := amfs_outcome_is_success(p_outcome_type);
    target := CASE WHEN is_ok THEN 1.0 ELSE 0.0 END;

    FOREACH entry_key IN ARRAY keys
    LOOP
        -- Last-slash split, matching Python's rsplit("/", 1).
        k := substring(entry_key from '([^/]+)$');
        ep := left(entry_key, length(entry_key) - length(k) - 1);

        SELECT * INTO cur FROM amfs_memory_entries
        WHERE namespace = p_namespace
          AND entity_path = ep
          AND key = k
          AND superseded_at IS NULL
          AND account_id IS NOT DISTINCT FROM p_account
        ORDER BY version DESC LIMIT 1;

        IF NOT FOUND THEN
            CONTINUE;
        END IF;

        -- Credit the claim that was read. If the key has been rewritten since
        -- and now says something else, this outcome is about the old claim.
        read_version := NULLIF(COALESCE(p_versions, '{}'::jsonb)->>entry_key, '')::INTEGER;
        IF read_version IS NOT NULL AND read_version <> cur.version THEN
            SELECT value INTO read_value FROM amfs_memory_entries
            WHERE namespace = p_namespace
              AND entity_path = ep
              AND key = k
              AND version = read_version
              AND account_id IS NOT DISTINCT FROM p_account
            LIMIT 1;
            IF FOUND AND read_value IS DISTINCT FROM cur.value THEN
                CONTINUE;
            END IF;
        END IF;

        prior := LEAST(1.0, GREATEST(0.0, COALESCE(cur.prior_confidence, cur.confidence)));

        IF p_model = 'multiplicative' THEN
            new_conf := LEAST(1.0, GREATEST(0.0,
                cur.confidence * amfs_outcome_multiplier(p_outcome_type) * p_causal_confidence));
            e_s := COALESCE(cur.evidence_success, 0) + CASE WHEN is_ok THEN 1 ELSE 0 END;
            e_f := COALESCE(cur.evidence_failure, 0) + CASE WHEN is_ok THEN 0 ELSE 1 END;
        ELSE
            w := amfs_outcome_severity(p_outcome_type)
                 * GREATEST(0.0, p_causal_confidence)
                 / n_keys
                 * (1.0 + abs(target - LEAST(1.0, GREATEST(0.0, cur.confidence))));
            e_s := COALESCE(cur.evidence_success, 0) * 0.8 + CASE WHEN is_ok THEN w ELSE 0 END;
            e_f := COALESCE(cur.evidence_failure, 0) * 0.8 + CASE WHEN is_ok THEN 0 ELSE w END;
            new_conf := LEAST(1.0, GREATEST(0.0, (2.0 * prior + e_s) / (2.0 + e_s + e_f)));
        END IF;

        IF new_conf < 0.5 AND NOT is_ok THEN
            disc := COALESCE(cur.discredited_at, NOW());
        ELSIF new_conf >= 0.5 THEN
            disc := NULL;
        ELSE
            disc := cur.discredited_at;
        END IF;

        UPDATE amfs_memory_entries
        SET superseded_at = NOW()
        WHERE id = cur.id;

        -- Copy the row (see 007); override only what this version changes.
        INSERT INTO amfs_memory_entries
        SELECT * FROM jsonb_populate_record(
            NULL::amfs_memory_entries,
            to_jsonb(cur) || jsonb_build_object(
                'id', gen_random_uuid(),
                'version', cur.version + 1,
                'confidence', new_conf,
                'outcome_count', cur.outcome_count + 1,
                'success_count', COALESCE(cur.success_count, 0) + CASE WHEN is_ok THEN 1 ELSE 0 END,
                'failure_count', COALESCE(cur.failure_count, 0) + CASE WHEN is_ok THEN 0 ELSE 1 END,
                'evidence_success', e_s,
                'evidence_failure', e_f,
                'prior_confidence', prior,
                'last_outcome', p_outcome_type,
                'last_outcome_at', NOW(),
                'discredited_at', disc,
                'superseded_at', NULL
            )
        );
        touched := touched + 1;
    END LOOP;
    RETURN touched;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION amfs_propagate_outcome() RETURNS TRIGGER AS $$
DECLARE
    model TEXT;
    att JSONB;
    att_keys TEXT[];
BEGIN
    model := COALESCE(NULLIF(current_setting('amfs.outcome_model', true), ''), 'evidence');

    -- Failed attempts first, oldest to newest: each hands its own failure to
    -- the entries that attempt relied on.
    FOR att IN
        SELECT value FROM jsonb_array_elements(COALESCE(NEW.attempts, '[]'::jsonb))
        ORDER BY COALESCE((value->>'attempt')::INTEGER, 0)
    LOOP
        SELECT COALESCE(array_agg(x), '{}') INTO att_keys
        FROM jsonb_array_elements_text(COALESCE(att->'causal_entry_keys', '[]'::jsonb)) AS x;
        PERFORM amfs_apply_outcome_step(
            NEW.namespace, NEW.account_id, att_keys,
            COALESCE(att->>'outcome_type', 'minor_failure'),
            NEW.causal_confidence, model,
            COALESCE(att->'causal_entry_versions', '{}'::jsonb)
        );
    END LOOP;

    -- Then the terminal outcome to the entries the resolution relied on.
    PERFORM amfs_apply_outcome_step(
        NEW.namespace, NEW.account_id, NEW.causal_entry_keys,
        NEW.outcome_type, NEW.causal_confidence, model,
        COALESCE(NEW.causal_entry_versions, '{}'::jsonb)
    );

    PERFORM pg_notify('amfs_outcome', json_build_object(
        'namespace', NEW.namespace,
        'outcome_ref', NEW.outcome_ref,
        'outcome_type', NEW.outcome_type,
        'agent_id', NEW.agent_id,
        'causal_confidence', NEW.causal_confidence,
        'attempts', jsonb_array_length(COALESCE(NEW.attempts, '[]'::jsonb))
    )::TEXT);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
