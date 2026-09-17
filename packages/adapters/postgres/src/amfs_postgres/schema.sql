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
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    evidence_success NUMERIC(10,4) NOT NULL DEFAULT 0,
    evidence_failure NUMERIC(10,4) NOT NULL DEFAULT 0,
    prior_confidence NUMERIC(6,4),
    last_outcome TEXT,
    last_outcome_at TIMESTAMPTZ,
    discredited_at TIMESTAMPTZ,
    recall_count INTEGER DEFAULT 0,
    priority_score NUMERIC(10,6),
    tier SMALLINT DEFAULT 3,
    importance_score NUMERIC(6,4),
    importance_dimensions JSONB,
    ttl_at TIMESTAMPTZ,
    memory_type TEXT DEFAULT 'fact',
    shared BOOLEAN NOT NULL DEFAULT TRUE,
    artifact_refs JSONB DEFAULT '[]',
    superseded_at TIMESTAMPTZ,
    account_id UUID,
    CONSTRAINT uq_entry_version UNIQUE (namespace, entity_path, key, version, account_id)
);

CREATE INDEX IF NOT EXISTS idx_entries_current
    ON amfs_memory_entries (namespace, entity_path, key)
    WHERE superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_entries_entity
    ON amfs_memory_entries (namespace, entity_path);

-- Tier partial indexes (idx_entries_hot, idx_entries_warm) are created
-- in _apply_migrations() so the tier column exists first on legacy DBs.

CREATE TABLE IF NOT EXISTS amfs_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL,
    outcome_ref TEXT NOT NULL,
    outcome_type TEXT NOT NULL,
    causal_confidence NUMERIC(5,4) DEFAULT 1.0,
    committed_at TIMESTAMPTZ NOT NULL,
    causal_entry_keys TEXT[] DEFAULT '{}',
    agent_id TEXT NOT NULL,
    account_id UUID,
    attempts JSONB NOT NULL DEFAULT '[]',
    causal_entry_versions JSONB NOT NULL DEFAULT '{}',
    final_action_index INTEGER
);

-- Range-partitioned by month on created_at. Traces are append-only and read by
-- recency, so a page of recent traces touches one or two partitions, retention
-- drops a month instead of deleting rows, and payload stripping runs per
-- partition. The primary key carries created_at because a partitioned table's
-- key must include the partition column; id is still unique per row and
-- get_trace(id) still resolves by id alone.
--
-- Databases created before this was partitioned are rebuilt in place by
-- _apply_migrations (see trace_partitions.migrate_to_partitioned). Monthly
-- partitions are created by ensure_partitions at adapter start; the DEFAULT
-- partition below catches anything created_at outside them.
CREATE TABLE IF NOT EXISTS amfs_decision_traces (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome_ref TEXT,
    outcome_type TEXT,
    decision_summary TEXT,
    task_input TEXT,
    response_text TEXT,
    tool_calls JSONB DEFAULT '[]',
    causal_entries JSONB DEFAULT '[]',
    external_contexts JSONB DEFAULT '[]',
    query_events JSONB DEFAULT '[]',
    error_events JSONB DEFAULT '[]',
    state_diff JSONB,
    session_started_at TIMESTAMPTZ,
    session_ended_at TIMESTAMPTZ,
    session_duration_ms NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Guarded so this file stays runnable against a database whose traces table
-- is still flat and about to be rebuilt by _apply_migrations: PARTITION OF a
-- non-partitioned table is an error, not a no-op.
DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'amfs_decision_traces' AND c.relkind = 'p'
          AND n.nspname = ANY (current_schemas(false))
    ) THEN
        CREATE TABLE IF NOT EXISTS amfs_decision_traces_default
            PARTITION OF amfs_decision_traces DEFAULT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_traces_namespace
    ON amfs_decision_traces (namespace, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_traces_agent
    ON amfs_decision_traces (namespace, agent_id);

-- Keyset pages per agent: ORDER BY created_at DESC, id DESC under an agent
-- filter is answered by this index alone.
CREATE INDEX IF NOT EXISTS idx_traces_agent_created
    ON amfs_decision_traces (namespace, agent_id, created_at DESC, id DESC);

-- Serves the entity_path filter, causal_entries @> '[{"entity_path": ...}]'.
-- jsonb_path_ops supports only containment, which is the only operator used,
-- and is smaller and faster for it than the default opclass.
CREATE INDEX IF NOT EXISTS idx_traces_causal_entries_gin
    ON amfs_decision_traces USING gin (causal_entries jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_traces_outcome
    ON amfs_decision_traces (namespace, outcome_type)
    WHERE outcome_type IS NOT NULL;

-- Resolves an entry back to the trace that committed it. An entry carries the
-- session that wrote it (provenance.session_id) and so does a trace, which is
-- the only link between the two — causal_entries records reads, not writes.
-- Without this, knowledge lineage would filter traces by an unindexed JSONB
-- scan over every trace on the account.
CREATE INDEX IF NOT EXISTS idx_traces_session
    ON amfs_decision_traces (namespace, agent_id, session_id)
    WHERE session_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS amfs_api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    prefix TEXT NOT NULL,
    key_type TEXT NOT NULL DEFAULT 'agent',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    scopes JSONB DEFAULT '[]',
    rate_limit_rpm INTEGER DEFAULT 120,
    last_used TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    created_by UUID
);

CREATE INDEX IF NOT EXISTS idx_api_keys_namespace
    ON amfs_api_keys (namespace, active);

CREATE TABLE IF NOT EXISTS amfs_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    actor_type TEXT NOT NULL DEFAULT 'system',
    actor_name TEXT NOT NULL DEFAULT 'system',
    action TEXT NOT NULL,
    resource TEXT,
    ip_address TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_namespace
    ON amfs_audit_log (namespace, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_action
    ON amfs_audit_log (namespace, action);

-- ──────────────────────────────────────────────────────────────────────
-- Teams & Members (Pro)
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_team_slug UNIQUE (namespace, slug)
);

CREATE INDEX IF NOT EXISTS idx_teams_namespace
    ON amfs_teams (namespace);

CREATE TABLE IF NOT EXISTS amfs_team_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    team_id UUID NOT NULL REFERENCES amfs_teams(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'developer',
    invited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accepted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_team_member UNIQUE (team_id, email),
    CONSTRAINT chk_member_role CHECK (role IN ('admin', 'developer', 'viewer'))
);

CREATE INDEX IF NOT EXISTS idx_team_members_team
    ON amfs_team_members (team_id);

CREATE INDEX IF NOT EXISTS idx_team_members_email
    ON amfs_team_members (namespace, email);

-- removed_at, removed_by columns and related indexes are added by
-- _apply_migrations() so existing databases don't crash on startup.

-- ──────────────────────────────────────────────────────────────────────
-- Pattern Detection Results (Pro)
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_detected_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    pattern_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    entity_path TEXT NOT NULL,
    description TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    category TEXT NOT NULL DEFAULT 'collaboration',
    CONSTRAINT chk_pattern_type CHECK (
        pattern_type IN (
            'knowledge_conflict', 'stale_knowledge', 'orphaned_branch',
            'redundant_writes', 'single_point_of_knowledge', 'passive_consumer',
            'unreviewed_changes', 'recurring_failure',
            'hot_entity', 'stale_cluster', 'confidence_drift'
        )
    ),
    CONSTRAINT chk_severity CHECK (severity IN ('info', 'warning', 'critical'))
);

CREATE INDEX IF NOT EXISTS idx_patterns_namespace
    ON amfs_detected_patterns (namespace, detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_patterns_type
    ON amfs_detected_patterns (namespace, pattern_type);

CREATE INDEX IF NOT EXISTS idx_patterns_unresolved
    ON amfs_detected_patterns (namespace)
    WHERE resolved = FALSE;

-- ──────────────────────────────────────────────────────────────────────
-- Agent Registration (Pro) — auto-created on first write
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    agent_id TEXT NOT NULL,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    entry_count INTEGER DEFAULT 0,
    profile JSONB,
    capabilities JSONB DEFAULT '[]'::jsonb,
    contracts JSONB DEFAULT '[]'::jsonb,
    CONSTRAINT uq_agent UNIQUE (namespace, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_agents_namespace
    ON amfs_agents (namespace);

-- ──────────────────────────────────────────────────────────────────────
-- Unified Event / Timeline Log (Pro) — the git commit log
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    agent_id TEXT NOT NULL,
    branch TEXT NOT NULL DEFAULT 'main',
    event_type TEXT NOT NULL,
    summary TEXT,
    details JSONB DEFAULT '{}',
    actor_agent_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_event_type CHECK (event_type IN (
        'write', 'read', 'outcome', 'webhook', 'brief_compiled', 'cross_agent_read',
        'branch_created', 'branch_merged', 'branch_closed',
        'access_granted', 'access_revoked',
        'rollback', 'tag_created', 'cherry_pick', 'fork',
        'snapshot_taken', 'snapshot_recovered'
    ))
);

CREATE INDEX IF NOT EXISTS idx_events_agent_timeline
    ON amfs_events (namespace, agent_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_branch
    ON amfs_events (namespace, agent_id, branch, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_events_type
    ON amfs_events (namespace, event_type);

-- ──────────────────────────────────────────────────────────────────────
-- Branches (Pro) — memory branch metadata
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_branches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    parent_branch TEXT NOT NULL DEFAULT 'main',
    branched_at TIMESTAMPTZ NOT NULL,
    created_by TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    merged_at TIMESTAMPTZ,
    merged_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    account_id UUID,
    CONSTRAINT uq_branch_name UNIQUE (namespace, name, account_id),
    CONSTRAINT chk_branch_status CHECK (status IN ('active', 'merged', 'closed'))
);

CREATE INDEX IF NOT EXISTS idx_branches_namespace
    ON amfs_branches (namespace, status);

-- ──────────────────────────────────────────────────────────────────────
-- Branch Access Control (Pro) — who can read/write a branch
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_branch_access (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    branch_name TEXT NOT NULL,
    grantee_type TEXT NOT NULL,
    grantee_id TEXT NOT NULL,
    permission TEXT NOT NULL DEFAULT 'read',
    granted_by TEXT NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    account_id UUID,
    CONSTRAINT uq_branch_access UNIQUE (namespace, branch_name, grantee_type, grantee_id, account_id),
    CONSTRAINT chk_grantee_type CHECK (grantee_type IN ('user', 'team', 'api_key')),
    CONSTRAINT chk_permission CHECK (permission IN ('read', 'read_write'))
);

-- ──────────────────────────────────────────────────────────────────────
-- Tags / Snapshots (Pro) — named point-in-time markers
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    branch TEXT NOT NULL DEFAULT 'main',
    tagged_at TIMESTAMPTZ NOT NULL,
    description TEXT,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    account_id UUID,
    CONSTRAINT uq_tag_name UNIQUE (namespace, name, account_id)
);

-- ──────────────────────────────────────────────────────────────────────
-- Pull Requests (Pro) — review workflow for branch merges
-- ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS amfs_pull_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    title TEXT NOT NULL,
    description TEXT,
    source_branch TEXT NOT NULL,
    target_branch TEXT NOT NULL DEFAULT 'main',
    status TEXT NOT NULL DEFAULT 'open',
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    merged_at TIMESTAMPTZ,
    merged_by TEXT,
    closed_at TIMESTAMPTZ,
    closed_by TEXT,
    merge_strategy TEXT,
    account_id UUID,
    CONSTRAINT chk_pr_status CHECK (status IN ('open', 'approved', 'merged', 'closed'))
);

CREATE INDEX IF NOT EXISTS idx_prs_namespace
    ON amfs_pull_requests (namespace, status);

CREATE TABLE IF NOT EXISTS amfs_pr_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    pr_id UUID NOT NULL REFERENCES amfs_pull_requests(id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    status TEXT NOT NULL,
    comment TEXT,
    entry_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    account_id UUID,
    CONSTRAINT chk_review_status CHECK (status IN ('approved', 'changes_requested', 'commented'))
);

CREATE INDEX IF NOT EXISTS idx_pr_reviews_pr
    ON amfs_pr_reviews (pr_id);

-- Back-propagation trigger: when an outcome is inserted, apply each failed
-- attempt's outcome to that attempt's causal entries, then the terminal outcome
-- to causal_entry_keys, through the evidence model (recency-weighted Beta
-- posterior with credit split and surprise scaling). Each touched entry gets a
-- new version carrying the updated evidence columns.
--
-- These definitions must stay identical to migrations/008_outcome_evidence.sql,
-- which PostgresAdapter._apply_migrations applies verbatim. Every copy is a
-- CREATE OR REPLACE: whichever runs last silently becomes the function, so a
-- stale copy here is not dead code, it is a regression waiting for the next
-- deploy. The arithmetic is also implemented in amfs_core/evidence.py for the
-- filesystem and S3 adapters; tests/unit/test_evidence.py pins the numbers.

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

-- Created only when absent, rather than dropped and recreated every time.
--
-- This file is not run once. The adapter applies it on server startup, and CI
-- applies it on every deploy, so an unconditional DROP TRIGGER took ACCESS
-- EXCLUSIVE on these tables at each of those moments. That lock conflicts with
-- reads, and Postgres queues later lock requests behind a pending exclusive
-- one, so every query arriving while it waited queued behind it. Two
-- production outages came out of that, the second with fifteen sessions
-- blocked behind a single statement on amfs_memory_entries.
--
-- Dropping first was never doing anything anyway: the trigger is recreated
-- identically, so on all but the first run the pair is an expensive no-op.
--
-- What a trigger *does* still updates normally, because the logic is in the
-- function and CREATE OR REPLACE FUNCTION above needs no lock on the table.
-- Only the binding itself -- which table, which events -- is frozen by this,
-- and changing that is rare enough to belong in migrations/ where existing
-- databases pick it up, which is how every other change to them travels.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_propagate_outcome'
          AND tgrelid = 'amfs_outcomes'::regclass
          AND NOT tgisinternal
    ) THEN
        CREATE TRIGGER trg_propagate_outcome
            AFTER INSERT ON amfs_outcomes
            FOR EACH ROW EXECUTE FUNCTION amfs_propagate_outcome();
    END IF;
EXCEPTION WHEN duplicate_object THEN
    -- Another process got there between the check and the CREATE, which the
    -- drop-then-create pair tolerated by construction and a bare check does
    -- not. Containers start in parallel and each applies this file, so the
    -- window is reached in practice on a database being bootstrapped: both see
    -- no trigger, the second blocks on the first's lock, and inherits an
    -- already-existing trigger the moment it is granted.
    --
    -- Swallowing it is the whole point. _apply_schema retries only errors
    -- mentioning a lock or a deadlock, so this one would propagate and leave
    -- that container unable to boot -- and the outcome it is complaining about
    -- is the one wanted anyway: the trigger exists.
    NULL;
END $$;

-- LISTEN/NOTIFY trigger: notify on new entry writes for watch()

CREATE OR REPLACE FUNCTION amfs_notify_write() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.superseded_at IS NULL THEN
        PERFORM pg_notify('amfs_write', json_build_object(
            'namespace', NEW.namespace,
            'entity_path', NEW.entity_path,
            'key', NEW.key,
            'version', NEW.version,
            'agent_id', NEW.agent_id,
            'branch', NEW.branch,
            'account_id', NEW.account_id
        )::TEXT);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Conditional for the reason given at trg_propagate_outcome. This is the one
-- that did the damage: amfs_memory_entries is the busiest table here, so it is
-- where an exclusive lock is least likely to be granted quickly and most
-- expensive to wait for.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_notify_write'
          AND tgrelid = 'amfs_memory_entries'::regclass
          AND NOT tgisinternal
    ) THEN
        CREATE TRIGGER trg_notify_write
            AFTER INSERT ON amfs_memory_entries
            FOR EACH ROW EXECUTE FUNCTION amfs_notify_write();
    END IF;
EXCEPTION WHEN duplicate_object THEN
    -- Concurrent bootstrap; see trg_propagate_outcome.
    NULL;
END $$;

-- Compiled digests table (Memory Cortex)

CREATE TABLE IF NOT EXISTS amfs_digests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    digest_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    summary JSONB NOT NULL,
    entry_count INTEGER NOT NULL DEFAULT 0,
    source_agents TEXT[] DEFAULT '{}',
    anticipation_score NUMERIC(6,4) DEFAULT 0.0,
    compiled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    branch TEXT NOT NULL DEFAULT 'main',
    account_id UUID,
    CONSTRAINT uq_digest UNIQUE (namespace, branch, digest_type, scope, account_id)
);

CREATE INDEX IF NOT EXISTS idx_digests_type ON amfs_digests(digest_type);
CREATE INDEX IF NOT EXISTS idx_digests_scope ON amfs_digests(scope);

CREATE OR REPLACE FUNCTION amfs_notify_digest() RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('amfs_digest', json_build_object(
        'namespace', NEW.namespace,
        'digest_type', NEW.digest_type,
        'scope', NEW.scope
    )::TEXT);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Conditional for the reason given at trg_propagate_outcome.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_notify_digest'
          AND tgrelid = 'amfs_digests'::regclass
          AND NOT tgisinternal
    ) THEN
        CREATE TRIGGER trg_notify_digest
            AFTER INSERT OR UPDATE ON amfs_digests
            FOR EACH ROW EXECUTE FUNCTION amfs_notify_digest();
    END IF;
EXCEPTION WHEN duplicate_object THEN
    -- Concurrent bootstrap; see trg_propagate_outcome.
    NULL;
END $$;

-- ──────────────────────────────────────────────────────────────────────
-- Commits — atomic groups of writes
-- ──────────────────────────────────────────────────────────────────────
--
-- Added late. Commits have existed in the model since the beginning, and
-- TransactionBuffer has always assembled one and handed it to save_commit —
-- which every adapter but the filesystem one inherited as a no-op. So a
-- transaction returned a real commit id that resolved to nothing, commit_log
-- was empty for every account, and common_ancestor found no ancestor for any
-- pair. All three read as an honest answer about an empty history, which is
-- why nothing complained for so long.
--
-- ``id`` is TEXT rather than UUID: the id is minted by the SDK as a content
-- hash of the commit, not by the database, and it is what a caller has already
-- been handed by the time this row is written.
--
-- ``entries`` and ``parent_ids`` are JSONB rather than child tables. A commit
-- is immutable once written and is always read whole, so there is nothing to
-- join for and nothing to update in place. Parent ids are a list because a
-- merge commit has two.

CREATE TABLE IF NOT EXISTS amfs_commits (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    branch TEXT NOT NULL DEFAULT 'main',
    message TEXT NOT NULL DEFAULT '',
    author_agent_id TEXT NOT NULL,
    session_id TEXT,
    entries JSONB NOT NULL DEFAULT '[]',
    tree_hash TEXT,
    parent_ids JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The commit log query: newest first, within one namespace and branch.
CREATE INDEX IF NOT EXISTS idx_commits_log
    ON amfs_commits (namespace, branch, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_commits_author
    ON amfs_commits (namespace, author_agent_id);

-- ──────────────────────────────────────────────────────────────────────
-- Reuse events — one row each time a memory is credited as reused
-- ──────────────────────────────────────────────────────────────────────
--
-- recall_count on amfs_memory_entries already counts reuse, and it is the right
-- thing for "which memories earn their keep". What it cannot answer is anything
-- with a WHEN or a WHO in it: when reuse happened, which agent did the reusing,
-- and whether the agent that reused a memory is the one that wrote it. Those are
-- the questions behind every user-facing claim memory can make — a weekly digest
-- of what your agents learned, a panel showing reuse without an agent having to
-- narrate it, and above all "your Claude session just used what your Cursor agent
-- worked out on Tuesday", which is the one thing a local file or a single tool's
-- memory structurally cannot do. A counter has no Tuesday in it.
--
-- Until now the only per-event record of reuse was an in-memory session ledger
-- whose own docstring said "never persisted", so it died with the session and
-- nothing could be shown to anyone who was not watching the chat at the time.
--
-- Not amfs_events, which was the obvious candidate and is the wrong home twice
-- over: it has no RLS (the adapter says so where it reads it back), and it backs a
-- user-facing timeline that one row per credited read would bury.
--
-- account_id is deliberately absent here, as it is from every other table in this
-- file: the hosted product adds the column, its default and its RLS policy in a
-- tenant migration, and OSS INSERTs omit it. A self-hoster gets the table from
-- this file with no tenancy at all, which is correct for one account.
--
-- No partitioning and no rollup table yet. Volume is bounded by billing — a
-- credited read is a metered read — and the eval package already establishes the
-- partition-plus-daily-rollup shape to copy if that stops being true.
CREATE TABLE IF NOT EXISTS amfs_reuse_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    branch TEXT NOT NULL DEFAULT 'main',
    entity_path TEXT NOT NULL,
    key TEXT NOT NULL,
    -- Which version was reused. The row is a fact about a specific version, and
    -- recall_count cannot say which one it counted.
    entry_version INTEGER,
    -- The author, and the agent that reused it. Equal on an ordinary re-read;
    -- different is the cross-surface moment worth telling someone about.
    written_by TEXT,
    reused_by TEXT,
    -- Estimated, and stored as such. Same clamp as the block the caller was shown,
    -- so a figure in a digest cannot disagree with the figure in the chat.
    est_tokens_saved INTEGER NOT NULL DEFAULT 0,
    -- Which read credited it: read, search or retrieve.
    surface TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The digest and the account-level panel: everything in a window, newest first.
CREATE INDEX IF NOT EXISTS idx_reuse_events_recent
    ON amfs_reuse_events (namespace, created_at DESC);

-- The per-agent panel: what this agent has been drawing on.
CREATE INDEX IF NOT EXISTS idx_reuse_events_reader
    ON amfs_reuse_events (namespace, reused_by, created_at DESC);

-- The cross-surface claim. Partial, so the index stays small: one agent reusing
-- another's memory is the minority of reuse and the only part this serves.
CREATE INDEX IF NOT EXISTS idx_reuse_events_cross_surface
    ON amfs_reuse_events (namespace, created_at DESC)
    WHERE written_by IS DISTINCT FROM reused_by;

-- Reuse of one entry over time, for the entry and entity pages.
CREATE INDEX IF NOT EXISTS idx_reuse_events_entry
    ON amfs_reuse_events (namespace, entity_path, key, created_at DESC);
