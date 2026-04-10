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
    CONSTRAINT uq_entry_version UNIQUE (namespace, entity_path, key, version)
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
    agent_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS amfs_decision_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome_ref TEXT,
    outcome_type TEXT,
    decision_summary TEXT,
    causal_entries JSONB DEFAULT '[]',
    external_contexts JSONB DEFAULT '[]',
    query_events JSONB DEFAULT '[]',
    error_events JSONB DEFAULT '[]',
    state_diff JSONB,
    session_started_at TIMESTAMPTZ,
    session_ended_at TIMESTAMPTZ,
    session_duration_ms NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_traces_namespace
    ON amfs_decision_traces (namespace, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_traces_agent
    ON amfs_decision_traces (namespace, agent_id);

CREATE INDEX IF NOT EXISTS idx_traces_outcome
    ON amfs_decision_traces (namespace, outcome_type)
    WHERE outcome_type IS NOT NULL;

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
    expires_at TIMESTAMPTZ
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
    removed_at TIMESTAMPTZ,
    removed_by TEXT,
    CONSTRAINT chk_member_role CHECK (role IN ('admin', 'developer', 'viewer'))
);

-- Only one active (non-removed) membership per (team_id, email)
CREATE UNIQUE INDEX IF NOT EXISTS uq_team_member_active
    ON amfs_team_members (team_id, email)
    WHERE removed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_team_members_team
    ON amfs_team_members (team_id);

CREATE INDEX IF NOT EXISTS idx_team_members_email
    ON amfs_team_members (namespace, email);

CREATE INDEX IF NOT EXISTS idx_team_members_removed
    ON amfs_team_members (namespace, email)
    WHERE removed_at IS NOT NULL;

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
        'write', 'outcome', 'webhook', 'brief_compiled', 'cross_agent_read',
        'branch_created', 'branch_merged', 'branch_closed',
        'access_granted', 'access_revoked',
        'rollback', 'tag_created', 'cherry_pick', 'fork'
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
    CONSTRAINT uq_branch_name UNIQUE (namespace, name),
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
    CONSTRAINT uq_branch_access UNIQUE (namespace, branch_name, grantee_type, grantee_id),
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
    CONSTRAINT uq_tag_name UNIQUE (namespace, name)
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
    CONSTRAINT chk_review_status CHECK (status IN ('approved', 'changes_requested', 'commented'))
);

CREATE INDEX IF NOT EXISTS idx_pr_reviews_pr
    ON amfs_pr_reviews (pr_id);

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
        WHEN 'critical_failure' THEN multiplier := 1.15;
        WHEN 'failure' THEN multiplier := 1.10;
        WHEN 'minor_failure' THEN multiplier := 1.08;
        WHEN 'success' THEN multiplier := 0.97;
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
                confidence, outcome_count, recall_count, ttl_at, memory_type,
                shared, artifact_refs
            ) VALUES (
                cur.namespace, cur.entity_path, cur.key, cur.version + 1, cur.value,
                cur.agent_id, cur.session_id, cur.written_at, cur.pattern_refs,
                cur.confidence * multiplier * NEW.causal_confidence,
                cur.outcome_count + 1, cur.recall_count, cur.ttl_at, cur.memory_type,
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
            'version', NEW.version,
            'agent_id', NEW.agent_id,
            'branch', NEW.branch
        )::TEXT);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_notify_write ON amfs_memory_entries;
CREATE TRIGGER trg_notify_write
    AFTER INSERT ON amfs_memory_entries
    FOR EACH ROW EXECUTE FUNCTION amfs_notify_write();

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
    CONSTRAINT uq_digest UNIQUE (namespace, branch, digest_type, scope)
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

DROP TRIGGER IF EXISTS trg_notify_digest ON amfs_digests;
CREATE TRIGGER trg_notify_digest
    AFTER INSERT OR UPDATE ON amfs_digests
    FOR EACH ROW EXECUTE FUNCTION amfs_notify_digest();
