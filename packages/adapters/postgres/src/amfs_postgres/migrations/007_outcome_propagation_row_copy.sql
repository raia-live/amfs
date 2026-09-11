-- Migration 007: propagate outcomes by copying the row, not by listing columns
--
-- Migration 006 fixed the direction of the outcome→confidence multipliers and
-- added the clamp. It did not change the shape of the INSERT, which names the
-- columns a new version should carry. That list is the remaining bug.
--
-- A column list inside a trigger is a copy of the schema frozen at the moment
-- someone wrote it. Every column added afterwards silently takes its DEFAULT on
-- every outcome, and nothing fails — the entry simply comes back subtly
-- different from the one that was superseded. Observed consequences, each from
-- a column the list did not name:
--
--   * recall_count reset to 0, erasing an entry's entire reuse history.
--   * shared reset to TRUE, republishing a private entry.
--   * tier reset to its default, moving the entry between the hot and warm
--     partial indexes and changing which queries can see it cheaply.
--   * embedding reset to NULL, dropping the live version out of vector search
--     until something backfilled it.
--
-- Deployments that extend this table fare worse still, because their added
-- columns are invisible to a list written here. A tenancy column that defaults
-- to NULL, combined with a row-level security policy that admits NULL, turns
-- outcome propagation into a way to widen access to an entry.
--
-- So this migration removes the list. to_jsonb(cur) carries whatever the row
-- actually has, and only the five fields a new version genuinely changes are
-- overridden: a fresh id (the old one is the superseded row's primary key), the
-- incremented version and outcome_count, the new confidence, and superseded_at
-- back to NULL (copying it would make the new version born invisible to every
-- `superseded_at IS NULL` read).
--
-- The account filter on the SELECT is new here too. Without it, an outcome
-- recorded under one account can reinforce an identically-pathed entry
-- belonging to another.
--
-- It is written IS NOT DISTINCT FROM rather than the more obvious
-- `account_id = NEW.account_id OR NEW.account_id IS NULL`. That version looks
-- like it is being generous to single-account installs, where every account_id
-- is NULL on both sides, but the generosity is unbounded in the other
-- direction: an outcome row that happens to carry no account matches entries
-- in EVERY account, which is precisely the cross-account reinforcement the
-- clause exists to prevent. Nothing in this adapter sets account_id on the
-- outcome — deployments rely on a column default for that — so an outcome
-- written by a path that did not establish one would have reinforced the whole
-- table.
--
-- IS NOT DISTINCT FROM treats NULL as a value: NULL matches NULL, so the
-- single-account case still works, and an outcome with no account reaches only
-- entries with no account. Where a deployment expects propagation and it stops
-- happening, the account on the outcome row is the thing to look at.
--
-- Idempotent: CREATE OR REPLACE FUNCTION, and the trigger binding is unchanged.
-- Existing stored confidences are NOT rewritten, following 006.

-- The function references account_id on both tables. schema.sql declares it and
-- the adapter's own migrations add it, but an older database predating either
-- would only find out when the trigger next fired.
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS account_id UUID;
ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS account_id UUID;

CREATE OR REPLACE FUNCTION amfs_propagate_outcome() RETURNS TRIGGER AS $$
DECLARE
    multiplier NUMERIC;
    entry_key TEXT;
    ep TEXT;
    k TEXT;
    cur RECORD;
BEGIN
    -- SUCCESS reinforces confidence (>1.0), failures erode it (<1.0).
    -- Unchanged from 006; repeated here because CREATE OR REPLACE rewrites the
    -- whole body, so every definition of this function has to carry all of it.
    CASE NEW.outcome_type
        WHEN 'critical_failure' THEN multiplier := 0.85;
        WHEN 'failure' THEN multiplier := 0.90;
        WHEN 'minor_failure' THEN multiplier := 0.92;
        WHEN 'success' THEN multiplier := 1.03;
        WHEN 'p1_incident' THEN multiplier := 0.85;
        WHEN 'p2_incident' THEN multiplier := 0.90;
        WHEN 'regression' THEN multiplier := 0.92;
        WHEN 'clean_deploy' THEN multiplier := 1.03;
        ELSE multiplier := 1.0;
    END CASE;

    FOREACH entry_key IN ARRAY NEW.causal_entry_keys
    LOOP
        -- Last-slash split, matching Python's rsplit("/", 1):
        -- "myapp/checkout/risk" -> ep="myapp/checkout", k="risk".
        IF position('/' in entry_key) = 0 THEN
            CONTINUE;
        END IF;
        k := substring(entry_key from '([^/]+)$');
        ep := left(entry_key, length(entry_key) - length(k) - 1);

        SELECT * INTO cur FROM amfs_memory_entries
        WHERE namespace = NEW.namespace
          AND entity_path = ep
          AND key = k
          AND superseded_at IS NULL
          AND account_id IS NOT DISTINCT FROM NEW.account_id
        ORDER BY version DESC LIMIT 1;

        IF FOUND THEN
            UPDATE amfs_memory_entries
            SET superseded_at = NOW()
            WHERE id = cur.id;

            -- Copy the row; override only what a new version changes. See the
            -- header for why this is not a column list.
            INSERT INTO amfs_memory_entries
            SELECT * FROM jsonb_populate_record(
                NULL::amfs_memory_entries,
                to_jsonb(cur) || jsonb_build_object(
                    'id', gen_random_uuid(),
                    'version', cur.version + 1,
                    'confidence', LEAST(1.0, GREATEST(0.0,
                        cur.confidence * multiplier * NEW.causal_confidence)),
                    'outcome_count', cur.outcome_count + 1,
                    'superseded_at', NULL
                )
            );
        END IF;
    END LOOP;

    PERFORM pg_notify('amfs_outcome', json_build_object(
        'namespace', NEW.namespace,
        'outcome_ref', NEW.outcome_ref,
        'outcome_type', NEW.outcome_type,
        'agent_id', NEW.agent_id,
        'causal_confidence', NEW.causal_confidence
    )::TEXT);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
