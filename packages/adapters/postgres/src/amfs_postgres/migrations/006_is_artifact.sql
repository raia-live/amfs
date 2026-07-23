-- Migration 006: artifact classification flag on memory entries.
--
-- Distinguishes stored working files (source code, markup, config) from
-- knowledge facts so retrieval can demote them. Applied automatically by
-- PostgresAdapter._apply_migrations() on adapter init; this file is the manual
-- reference for operators applying schema by hand.
--
--     psql "$AMFS_POSTGRES_DSN" -f 006_is_artifact.sql
--
-- After applying, populate historical rows (classify + re-embed artifacts):
--     AMFS_POSTGRES_DSN=... python -c "from amfs_postgres import PostgresAdapter; \
--         print(PostgresAdapter('$AMFS_POSTGRES_DSN', embedder=<embedder>).backfill_is_artifact())"

ALTER TABLE amfs_memory_entries
    ADD COLUMN IF NOT EXISTS is_artifact BOOLEAN NOT NULL DEFAULT FALSE;

-- Partial index: only artifact rows are indexed, so it stays small while still
-- supporting fast "exclude/demote artifacts" filters.
CREATE INDEX IF NOT EXISTS idx_entries_is_artifact
    ON amfs_memory_entries (is_artifact) WHERE is_artifact;
