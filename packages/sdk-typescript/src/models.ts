/**
 * AMFS TypeScript types — mirrors the Python Pydantic models.
 */

export enum OutcomeType {
  P1_INCIDENT = "p1_incident",
  P2_INCIDENT = "p2_incident",
  REGRESSION = "regression",
  CLEAN_DEPLOY = "clean_deploy",
}

export const OUTCOME_MULTIPLIERS: Record<OutcomeType, number> = {
  [OutcomeType.P1_INCIDENT]: 1.15,
  [OutcomeType.P2_INCIDENT]: 1.1,
  [OutcomeType.REGRESSION]: 1.08,
  [OutcomeType.CLEAN_DEPLOY]: 0.97,
};

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
}

export interface OutcomeRecord {
  outcomeRef: string;
  outcomeType: OutcomeType;
  causalConfidence: number;
  committedAt: string; // ISO 8601
  causalEntryKeys: string[];
  agentId: string;
}

export interface LayerConfig {
  adapter: string;
  options: Record<string, unknown>;
}

export interface AMFSConfig {
  namespace: string;
  layers: Record<string, LayerConfig>;
}
