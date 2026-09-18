-- Migration 009: action-level learning, first-strike tolerance, validators
--
-- Grid v2 of the continual-learning benchmark showed two things the evidence
-- model of 008 could not do:
--
--   * Memory entries record what someone *wrote*; nothing recorded what
--     happened when an agent *did* something. A support fleet spent 24
--     attempts per ticket class on the same two failing actions and never
--     tried the one that worked, because "tried and failed here" was not a
--     thing memory could say. `actions_taken` on the outcome row is that
--     record: the action that ended each failed attempt (a loss) and the
--     terminal action (the outcome). `entity_paths` scopes it to the entities
--     the outcome was about; `situation` is an optional caller label. The
--     server adds `task_embedding` (a pgvector column at the same dimension as
--     the entries' embedding — provisioned by the adapter, not here, because
--     the dimension is deployment-specific) so a retrieve can ask "on the
--     tasks most like this one, what worked?".
--   * A single failure on a rule validated four times discredited it
--     (0.89 -> 0.49) and pulled it out of retrieval. That is how the surprise
--     term should treat a fresh claim, not a proven one. The first failure of
--     an entry with a clean record of at least three successes is now weighed
--     without surprise: the rule is left contested (~0.62), and a second
--     failure still discredits it (~0.40), so a regime change is unlearned in
--     the same two strikes as before.
--
-- `validators` on entries lists the distinct agents whose successes credited
-- the claim; the trigger maintains it. The step function gains the committing
-- agent as an eighth argument; the seven-argument form is retired.
--
-- Arithmetic mirrors amfs_core/evidence.py and must stay identical to
-- schema.sql, which carries the same function bodies.

ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS actions_taken JSONB NOT NULL DEFAULT '[]';
ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS entity_paths TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS situation TEXT;
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS validators JSONB NOT NULL DEFAULT '[]';

-- Priors are scoped by entity overlap, then narrowed by vector similarity.
CREATE INDEX IF NOT EXISTS idx_outcomes_entity_paths
    ON amfs_outcomes USING gin (entity_paths);
CREATE INDEX IF NOT EXISTS idx_outcomes_ns_committed
    ON amfs_outcomes (namespace, committed_at DESC);

-- The seven-argument step predates validators; the trigger below calls the
-- eight-argument one, so the old overload is retired here.
DROP FUNCTION IF EXISTS amfs_apply_outcome_step(TEXT, UUID, TEXT[], TEXT, NUMERIC, TEXT, JSONB);

CREATE OR REPLACE FUNCTION amfs_apply_outcome_step(
    p_namespace TEXT,
    p_account UUID,
    p_keys TEXT[],
    p_outcome_type TEXT,
    p_causal_confidence NUMERIC,
    p_model TEXT,
    p_versions JSONB,
    p_agent TEXT
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
    surprise NUMERIC;
    vals JSONB;
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
            -- First strike: the first failure of an entry with a clean record
            -- of at least three successes is weighed without the surprise
            -- term (amfs_core.evidence.first_strike). One slip on a validated
            -- rule leaves it contested; the second still discredits it.
            surprise := 1.0 + abs(target - LEAST(1.0, GREATEST(0.0, cur.confidence)));
            IF NOT is_ok AND COALESCE(cur.failure_count, 0) = 0
               AND COALESCE(cur.success_count, 0) >= 3 THEN
                surprise := 1.0;
            END IF;
            w := amfs_outcome_severity(p_outcome_type)
                 * GREATEST(0.0, p_causal_confidence)
                 / n_keys
                 * surprise;
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

        -- Validators: distinct agents whose successes credited this claim,
        -- most recent last, capped at ten (amfs_core.evidence.validators_after).
        vals := COALESCE(cur.validators, '[]'::jsonb);
        IF is_ok AND COALESCE(p_agent, '') <> '' THEN
            vals := vals - p_agent;
            vals := vals || to_jsonb(ARRAY[p_agent]);
            IF jsonb_array_length(vals) > 10 THEN
                SELECT COALESCE(jsonb_agg(v), '[]'::jsonb) INTO vals
                FROM (
                    SELECT v FROM jsonb_array_elements(vals) WITH ORDINALITY AS t(v, ord)
                    ORDER BY ord OFFSET jsonb_array_length(vals) - 10
                ) s;
            END IF;
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
                'validators', vals,
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
            COALESCE(att->'causal_entry_versions', '{}'::jsonb),
            NEW.agent_id
        );
    END LOOP;

    -- Then the terminal outcome to the entries the resolution relied on.
    PERFORM amfs_apply_outcome_step(
        NEW.namespace, NEW.account_id, NEW.causal_entry_keys,
        NEW.outcome_type, NEW.causal_confidence, model,
        COALESCE(NEW.causal_entry_versions, '{}'::jsonb),
        NEW.agent_id
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
