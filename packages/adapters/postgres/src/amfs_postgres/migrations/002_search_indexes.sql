-- Migration: Add full-text search and vector indexes

-- Full-text search column and index
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS search_tsv TSVECTOR;

CREATE INDEX IF NOT EXISTS idx_entries_fts
    ON amfs_memory_entries USING GIN (search_tsv);

-- Trigger to auto-populate tsvector from key + value
CREATE OR REPLACE FUNCTION amfs_update_search_tsv() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_tsv := to_tsvector('english',
        coalesce(NEW.key, '') || ' ' ||
        coalesce(NEW.entity_path, '') || ' ' ||
        coalesce(NEW.value::text, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_search_tsv ON amfs_memory_entries;
CREATE TRIGGER trg_update_search_tsv
    BEFORE INSERT OR UPDATE ON amfs_memory_entries
    FOR EACH ROW EXECUTE FUNCTION amfs_update_search_tsv();

-- Update existing rows
UPDATE amfs_memory_entries SET search_tsv = to_tsvector('english',
    coalesce(key, '') || ' ' ||
    coalesce(entity_path, '') || ' ' ||
    coalesce(value::text, '')
) WHERE search_tsv IS NULL;

-- pgvector extension and embedding column (optional, requires pgvector extension)
-- Users must run: CREATE EXTENSION IF NOT EXISTS vector;
-- before applying this migration.
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS embedding vector(384);

CREATE INDEX IF NOT EXISTS idx_entries_embedding
    ON amfs_memory_entries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
