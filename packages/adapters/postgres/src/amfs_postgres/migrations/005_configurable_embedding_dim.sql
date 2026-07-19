-- Migration 005: configurable embedding dimension
--
-- Migration 002 hardcodes `embedding vector(384)`, which pins the store to a
-- 384-dim embedder (e.g. all-MiniLM / bge-small). To use a larger/stronger
-- embedder (e.g. bge-large at 1024 dims) the column must be provisioned at the
-- matching dimension. pgvector cannot change a vector column's dimension in
-- place, so switching dimensions requires dropping and recreating the column
-- (and re-embedding existing rows).
--
-- This file is parameterized via a psql variable. Apply with, e.g.:
--     psql "$DSN" -v embedding_dim=1024 -f 005_configurable_embedding_dim.sql
--
-- Prefer the programmatic path in application code:
--     PostgresAdapter(dsn, embedder=<embedder>, embedding_dim=1024)
--         .ensure_embedding_column()   -- creates column + HNSW index if absent
--     adapter.backfill_embeddings()    -- fills embeddings for existing rows
--
-- WARNING: the DROP below destroys existing vectors. Only run it when
-- intentionally changing dimension; you must re-embed afterwards (backfill).

\set embedding_dim :embedding_dim

CREATE EXTENSION IF NOT EXISTS vector;

-- Recreate the embedding column at the requested dimension.
-- (Commented DROP is intentional — uncomment only when changing dimension.)
-- DROP INDEX IF EXISTS idx_entries_embedding;
-- ALTER TABLE amfs_memory_entries DROP COLUMN IF EXISTS embedding;

ALTER TABLE amfs_memory_entries
    ADD COLUMN IF NOT EXISTS embedding vector(:embedding_dim);

CREATE INDEX IF NOT EXISTS idx_entries_embedding
    ON amfs_memory_entries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- After (re)creating the column, backfill embeddings from application code:
--   adapter.backfill_embeddings()
