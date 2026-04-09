-- Migration: Knowledge graph table for materialized entity relationships
-- Depends on: 001_initial.sql

CREATE TABLE IF NOT EXISTS amfs_knowledge_graph (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    source_entity TEXT NOT NULL,
    source_type TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    target_type TEXT NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    evidence_count INT NOT NULL DEFAULT 1,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance JSONB,
    branch TEXT NOT NULL DEFAULT 'main',
    CONSTRAINT uq_graph_edge UNIQUE (namespace, branch, source_entity, relation, target_entity)
);

CREATE INDEX IF NOT EXISTS idx_kg_source ON amfs_knowledge_graph (namespace, source_entity);
CREATE INDEX IF NOT EXISTS idx_kg_target ON amfs_knowledge_graph (namespace, target_entity);
CREATE INDEX IF NOT EXISTS idx_kg_relation ON amfs_knowledge_graph (namespace, relation);
CREATE INDEX IF NOT EXISTS idx_kg_confidence ON amfs_knowledge_graph (namespace, confidence DESC);
