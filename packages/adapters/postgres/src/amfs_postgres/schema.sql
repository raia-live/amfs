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
    account_id UUID
);

CREATE TABLE IF NOT EXISTS amfs_decision_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_traces_namespace
    ON amfs_decision_traces (namespace, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_traces_agent
    ON amfs_decision_traces (namespace, agent_id);

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
