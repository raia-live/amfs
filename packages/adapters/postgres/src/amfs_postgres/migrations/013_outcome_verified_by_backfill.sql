-- Migration 013: verified_by on the outcomes that predate the column
--
-- 012 added `amfs_outcomes.verified_by`, which the adapter writes at commit
-- from the record's `session_metadata.attributes.verified_by`, and the
-- learning curve reads it for `verified_share` — how much of an agent's
-- record was decided outside the agent. Every outcome committed before 012
-- has NULL there, while the same attribute sits on its sealed trace: on the
-- first database upgraded, 1 967 of the window's 3 135 outcomes were graded
-- by a monitor and the curve read 0.3 %.
--
-- Copy it across once, from the trace with the same namespace, account and
-- outcome_ref (the newest, where a ref was sealed twice). Only rows still
-- NULL are written, so an outcome that was never verified stays NULL. The
-- join has no index to lean on and reads the traces once, which is fine for
-- a one-time pass and not for every deploy: `amfs_backfills` records that
-- the pass ran, and every later application returns at the first SELECT.
-- Like `amfs_schema_state`, that table holds no tenant data and needs no
-- policy; nothing tenant-scoped may be added to it.
--
-- Guarded for a database that has no traces table yet (a fresh bootstrap
-- applies this before anything is sealed; there is nothing to copy) and for
-- one whose adapter has not added `session_metadata` to the traces. Under
-- row-level security it does nothing and is not marked done — the hosted
-- product runs the same pass per tenant from its own migration.

CREATE TABLE IF NOT EXISTS amfs_backfills (
    name TEXT PRIMARY KEY,
    done_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rows_written BIGINT NOT NULL DEFAULT 0
);

DO $$
DECLARE
    written BIGINT := 0;
    -- The hosted product adds account_id to the traces in a tenant migration;
    -- a self-hosted database has it on neither side or on outcomes only, and
    -- a namespace is one installation there. Match on the account only where
    -- both tables carry it.
    by_account BOOLEAN;
BEGIN
    IF EXISTS (SELECT 1 FROM amfs_backfills WHERE name = '013_outcome_verified_by') THEN
        RETURN;
    END IF;
    IF to_regclass('amfs_decision_traces') IS NULL
       OR NOT EXISTS (
           SELECT 1 FROM information_schema.columns
           WHERE table_name = 'amfs_decision_traces' AND column_name = 'session_metadata')
       OR NOT EXISTS (
           SELECT 1 FROM information_schema.columns
           WHERE table_name = 'amfs_outcomes' AND column_name = 'verified_by')
    THEN
        -- Nothing to copy from, or to. Not marked done: the pass should run
        -- once the column exists, on the next application.
        RETURN;
    END IF;
    -- Under row-level security a session that names no tenant reads no rows,
    -- and a pass that read nothing must not be recorded as done. The hosted
    -- product has policies on both tables and runs this pass per tenant from
    -- its own migration, which knows the accounts; a self-hosted database has
    -- no policies and runs it here. A role that bypasses them may run it too.
    IF EXISTS (
           SELECT 1 FROM pg_class c
           WHERE c.oid IN ('amfs_outcomes'::regclass, 'amfs_decision_traces'::regclass)
             AND c.relrowsecurity)
       AND NOT EXISTS (
           SELECT 1 FROM pg_roles r
           WHERE r.rolname = current_user AND (r.rolsuper OR r.rolbypassrls))
    THEN
        RETURN;
    END IF;
    by_account := EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'amfs_decision_traces' AND column_name = 'account_id'
    ) AND EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'amfs_outcomes' AND column_name = 'account_id'
    );

    EXECUTE format($sql$
        UPDATE amfs_outcomes o
           SET verified_by = v.verified_by
          FROM (
              SELECT DISTINCT ON (t.namespace, %1$s t.outcome_ref)
                     t.namespace, %2$s t.outcome_ref,
                     t.session_metadata->'attributes'->>'verified_by' AS verified_by
                FROM amfs_decision_traces t
               WHERE t.outcome_ref IS NOT NULL
                 AND t.session_metadata->'attributes'->>'verified_by' IS NOT NULL
               ORDER BY t.namespace, %1$s t.outcome_ref, t.created_at DESC
          ) v
         WHERE o.verified_by IS NULL
           AND o.namespace = v.namespace
           %3$s
           AND o.outcome_ref = v.outcome_ref
    $sql$,
        CASE WHEN by_account THEN 't.account_id,' ELSE '' END,
        CASE WHEN by_account THEN 't.account_id,' ELSE '' END,
        CASE WHEN by_account THEN 'AND o.account_id IS NOT DISTINCT FROM v.account_id' ELSE '' END
    );
    GET DIAGNOSTICS written = ROW_COUNT;

    INSERT INTO amfs_backfills (name, rows_written)
    VALUES ('013_outcome_verified_by', written)
    ON CONFLICT (name) DO NOTHING;
END $$;
