-- AMFS Postgres schema
-- Run this DDL to set up the AMFS tables and triggers.

CREATE TABLE IF NOT EXISTS amfs_memory_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL,
    entity_path TEXT NOT NULL,
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    value JSONB,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    written_at TIMESTAMPTZ NOT NULL,
    pattern_refs TEXT[] DEFAULT '{}',
    confidence NUMERIC(6,4) DEFAULT 1.0,
    outcome_count INTEGER DEFAULT 0,
    ttl_at TIMESTAMPTZ,
    memory_type TEXT DEFAULT 'fact',
    superseded_at TIMESTAMPTZ,
    CONSTRAINT uq_entry_version UNIQUE (namespace, entity_path, key, version)
);

CREATE INDEX IF NOT EXISTS idx_entries_current
    ON amfs_memory_entries (namespace, entity_path, key)
    WHERE superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_entries_entity
    ON amfs_memory_entries (namespace, entity_path);

CREATE TABLE IF NOT EXISTS amfs_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL,
    outcome_ref TEXT NOT NULL,
    outcome_type TEXT NOT NULL,
    causal_confidence NUMERIC(5,4) DEFAULT 1.0,
    committed_at TIMESTAMPTZ NOT NULL,
    causal_entry_keys TEXT[] DEFAULT '{}',
    agent_id TEXT NOT NULL
);

-- Back-propagation trigger: when an outcome is inserted,
-- for each causal_entry_key: supersede current entry and insert
-- a new version with confidence *= multiplier * causal_confidence.

CREATE OR REPLACE FUNCTION amfs_propagate_outcome() RETURNS TRIGGER AS $$
DECLARE
    multiplier NUMERIC;
    entry_key TEXT;
    parts TEXT[];
    ep TEXT;
    k TEXT;
    cur RECORD;
BEGIN
    -- Determine multiplier from outcome type
    CASE NEW.outcome_type
        WHEN 'p1_incident' THEN multiplier := 1.15;
        WHEN 'p2_incident' THEN multiplier := 1.10;
        WHEN 'regression' THEN multiplier := 1.08;
        WHEN 'clean_deploy' THEN multiplier := 0.97;
        ELSE multiplier := 1.0;
    END CASE;

    FOREACH entry_key IN ARRAY NEW.causal_entry_keys
    LOOP
        -- Parse "entity_path/key" using last-slash split (matches Python rsplit("/", 1))
        -- e.g. "myapp/checkout/risk" -> ep="myapp/checkout", k="risk"
        IF position('/' in entry_key) = 0 THEN
            CONTINUE;
        END IF;
        k := substring(entry_key from '([^/]+)$');
        ep := left(entry_key, length(entry_key) - length(k) - 1);

        -- Find current (non-superseded) entry
        SELECT * INTO cur FROM amfs_memory_entries
        WHERE namespace = NEW.namespace
          AND entity_path = ep
          AND key = k
          AND superseded_at IS NULL
        ORDER BY version DESC LIMIT 1;

        IF FOUND THEN
            -- Supersede the current entry
            UPDATE amfs_memory_entries
            SET superseded_at = NOW()
            WHERE id = cur.id;

            -- Insert new version with updated confidence
            INSERT INTO amfs_memory_entries (
                namespace, entity_path, key, version, value,
                agent_id, session_id, written_at, pattern_refs,
                confidence, outcome_count, ttl_at, memory_type
            ) VALUES (
                cur.namespace, cur.entity_path, cur.key, cur.version + 1, cur.value,
                cur.agent_id, cur.session_id, cur.written_at, cur.pattern_refs,
                cur.confidence * multiplier * NEW.causal_confidence,
                cur.outcome_count + 1, cur.ttl_at, cur.memory_type
            );
        END IF;
    END LOOP;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_propagate_outcome ON amfs_outcomes;
CREATE TRIGGER trg_propagate_outcome
    AFTER INSERT ON amfs_outcomes
    FOR EACH ROW EXECUTE FUNCTION amfs_propagate_outcome();

-- LISTEN/NOTIFY trigger: notify on new entry writes for watch()

CREATE OR REPLACE FUNCTION amfs_notify_write() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.superseded_at IS NULL THEN
        PERFORM pg_notify('amfs_write', json_build_object(
            'namespace', NEW.namespace,
            'entity_path', NEW.entity_path,
            'key', NEW.key,
            'version', NEW.version
        )::TEXT);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_notify_write ON amfs_memory_entries;
CREATE TRIGGER trg_notify_write
    AFTER INSERT ON amfs_memory_entries
    FOR EACH ROW EXECUTE FUNCTION amfs_notify_write();
