/**
 * AMFS TypeScript types — mirrors the Python Pydantic models.
 */

export enum OutcomeType {
  SUCCESS = "success",
  MINOR_FAILURE = "minor_failure",
  FAILURE = "failure",
  CRITICAL_FAILURE = "critical_failure",
  /** @deprecated Use SUCCESS */ CLEAN_DEPLOY = "clean_deploy",
  /** @deprecated Use MINOR_FAILURE */ REGRESSION = "regression",
  /** @deprecated Use FAILURE */ P2_INCIDENT = "p2_incident",
  /** @deprecated Use CRITICAL_FAILURE */ P1_INCIDENT = "p1_incident",
}

/**
 * Classification of a memory entry, mirroring `amfs_core.models.MemoryType`.
 *
 * `fact` is stable knowledge; `belief` a hypothesis that decays faster;
 * `experience` a record of an action taken, decaying slower; `procedure` is
 * how to do a task — a goal, ordered steps and what to do when one fails —
 * decaying slowest and listed by the briefing in its own `procedures` section.
 */
export enum MemoryType {
  FACT = "fact",
  BELIEF = "belief",
  EXPERIENCE = "experience",
  PROCEDURE = "procedure",
}

export const OUTCOME_MULTIPLIERS: Record<string, number> = {
  [OutcomeType.CRITICAL_FAILURE]: 0.85,
  [OutcomeType.FAILURE]: 0.90,
  [OutcomeType.MINOR_FAILURE]: 0.92,
  [OutcomeType.SUCCESS]: 1.03,
  [OutcomeType.P1_INCIDENT]: 0.85,
  [OutcomeType.P2_INCIDENT]: 0.90,
  [OutcomeType.REGRESSION]: 0.92,
  [OutcomeType.CLEAN_DEPLOY]: 1.03,
};

/** Clamp a confidence score to the valid [0.0, 1.0] range. */
export function clampConfidence(value: number): number {
  return Math.max(0.0, Math.min(1.0, value));
}

export interface Provenance {
  agentId: string;
  sessionId: string;
  writtenAt: string; // ISO 8601
  patternRefs: string[];
}

export interface ArtifactRef {
  uri: string;
  mediaType?: string | null;
  label?: string | null;
  sizeBytes?: number | null;
}

export interface MemoryEntry {
  amfsVersion: string;
  entityPath: string;
  key: string;
  version: number;
  value: unknown;
  provenance: Provenance;
  confidence: number;
  outcomeCount: number;
  ttlAt: string | null;
  artifactRefs?: ArtifactRef[];
  shared: boolean;
  contentHash: string | null;
  integrityChain: string | null;
  commitId: string | null;
  memoryType?: string;
  branch?: string;
  recallCount?: number;
  priorityScore?: number;
  tier?: string;
}

export interface CommitEntry {
  entityPath: string;
  key: string;
  version: number;
  contentHash: string | null;
}

export interface Commit {
  id: string;
  message: string;
  authorAgentId: string;
  sessionId: string | null;
  entries: CommitEntry[];
  treeHash: string | null;
  parentIds: string[];
  branch: string;
  createdAt: string;
  namespace: string;
}

export interface OutcomeRecord {
  outcomeRef: string;
  outcomeType: OutcomeType;
  causalConfidence: number;
  committedAt: string; // ISO 8601
  causalEntryKeys: string[];
  agentId: string;
}

export interface AgentProfile {
  description: string;
  defaultBranch: string;
  autoContextPaths: string[];
  tags: string[];
}

export interface AgentCapability {
  name: string;
  description: string;
  entityPaths: string[];
}

export interface MemoryContract {
  entityPath: string;
  keyPattern: string;
  minConfidence: number;
  maxConfidence: number;
  requiredFields: string[];
  ttlRequired: boolean;
  description: string;
}

export interface FieldChange {
  path: string;
  operation: "add" | "remove" | "replace";
  oldValue?: unknown;
  newValue?: unknown;
}

export interface MemoryPatch {
  entityPath: string;
  key: string;
  changes: FieldChange[];
  sourceVersion: number | null;
  targetVersion: number | null;
}

export interface ScopeInfo {
  path: string;
  entryCount: number;
  avgConfidence: number;
  keys: string[];
  oldest: string | null;
  newest: string | null;
}

export interface LayerConfig {
  adapter: string;
  options: Record<string, unknown>;
}

export interface AMFSConfig {
  namespace: string;
  layers: Record<string, LayerConfig>;
}

export interface RecallConfig {
  semanticWeight?: number;
  recencyWeight?: number;
  confidenceWeight?: number;
  recencyHalfLifeDays?: number;
}

export interface ScoredEntry {
  entry: MemoryEntry;
  score: number;
  breakdown: Record<string, number>;
}
