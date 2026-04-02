-- Migration: Add artifact_refs column for external blob references
ALTER TABLE amfs_memory_entries ADD COLUMN IF NOT EXISTS artifact_refs JSONB DEFAULT '[]';
