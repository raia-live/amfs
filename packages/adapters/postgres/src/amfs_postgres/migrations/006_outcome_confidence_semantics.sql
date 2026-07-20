-- Migration 006: correct outcome→confidence direction + clamp to [0,1]
--
-- The original amfs_propagate_outcome() trigger applied INVERTED multipliers:
-- a SUCCESS multiplied confidence DOWN (x0.97) while every failure multiplied
-- it UP (x1.08 .. x1.15), and the result was never clamped — so repeated
-- failures pushed confidence past 1.0 (rendering as ">100%") and repeated
-- successes decayed a well-validated fact below retrieval gates.
--
-- This migration replaces the trigger function so that:
--   * SUCCESS / clean_deploy reinforce confidence (x1.03)
--   * failures erode it (minor 0.92, failure 0.90, critical/p1 0.85, p2 0.90)
--   * the new confidence is clamped to [0.0, 1.0]
--
-- Idempotent: CREATE OR REPLACE FUNCTION. The trigger binding is unchanged, so
-- only the function body is updated. Existing stored confidences are NOT
-- rewritten (future outcomes use the corrected math); run an optional one-time
-- backfill separately if you want historical values recomputed.

CREATE OR REPLACE FUNCTION amfs_propagate_outcome() RETURNS TRIGGER AS $$
DECLARE
    multiplier NUMERIC;
    entry_key TEXT;
    k TEXT;
    ep TEXT;
    cur RECORD;
BEGIN
    -- SUCCESS reinforces confidence (>1.0), failures erode it (<1.0).
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
        ORDER BY version DESC LIMIT 1;

        IF FOUND THEN
            UPDATE amfs_memory_entries
            SET superseded_at = NOW()
            WHERE id = cur.id;

            INSERT INTO amfs_memory_entries (
                namespace, entity_path, key, version, value,
                agent_id, session_id, written_at, pattern_refs,
                confidence, outcome_count, recall_count,
                ttl_at, memory_type, shared, artifact_refs
            ) VALUES (
                cur.namespace, cur.entity_path, cur.key, cur.version + 1, cur.value,
                cur.agent_id, cur.session_id, cur.written_at, cur.pattern_refs,
                LEAST(1.0, GREATEST(0.0, cur.confidence * multiplier * NEW.causal_confidence)),
                cur.outcome_count + 1, cur.recall_count,
                cur.ttl_at, cur.memory_type,
                cur.shared, cur.artifact_refs
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
