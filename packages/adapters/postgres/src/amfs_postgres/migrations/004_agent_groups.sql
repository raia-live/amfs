-- Agent groups (user-defined)
CREATE TABLE IF NOT EXISTS amfs_agent_groups (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  namespace TEXT NOT NULL DEFAULT 'default',
  account_id UUID,
  name TEXT NOT NULL,
  description TEXT DEFAULT '',
  color TEXT DEFAULT NULL,
  icon TEXT DEFAULT NULL,
  position FLOAT DEFAULT 0,
  auto_generated BOOLEAN DEFAULT FALSE,
  source_cluster_id TEXT DEFAULT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT uq_agent_group UNIQUE (namespace, account_id, name)
);

CREATE INDEX IF NOT EXISTS idx_agent_groups_account
  ON amfs_agent_groups(account_id) WHERE account_id IS NOT NULL;

-- Group membership (agent belongs to at most one group)
CREATE TABLE IF NOT EXISTS amfs_agent_group_members (
  group_id UUID NOT NULL REFERENCES amfs_agent_groups(id) ON DELETE CASCADE,
  agent_id TEXT NOT NULL,
  namespace TEXT NOT NULL DEFAULT 'default',
  added_at TIMESTAMPTZ DEFAULT NOW(),
  added_by TEXT DEFAULT 'user',
  PRIMARY KEY (group_id, agent_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_group_member_unique
  ON amfs_agent_group_members(namespace, agent_id);

-- Dismissed cluster suggestions (so they don't reappear)
CREATE TABLE IF NOT EXISTS amfs_agent_group_suggestions_dismissed (
  account_id UUID NOT NULL,
  cluster_id TEXT NOT NULL,
  dismissed_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (account_id, cluster_id)
);
