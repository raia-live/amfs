/**
 * @senselab-ai/amfs — Agent Memory File System TypeScript SDK
 */

export { AgentMemory, MemoryScope } from "./memory.js";
export type { AgentMemoryOptions, SearchOptions, MemoryStats } from "./memory.js";
export type { AmfsAdapter, WatchHandle } from "./adapter.js";
export { createWatchHandle } from "./adapter.js";
export { InMemoryAdapter } from "./adapters/filesystem.js";
export { HttpAdapter, toDecisionTrace, toDecisionTracePage } from "./adapters/http.js";
export type {
  HttpAdapterOptions,
  DecisionTrace,
  DecisionTraceSummary,
  DecisionTracePage,
  ListTracesOptions,
  ToolCall,
  SessionMetadata,
} from "./adapters/http.js";
export {
  validateSessionAttributes,
  SESSION_ATTRIBUTES_MAX_KEYS,
  SESSION_ATTRIBUTE_KEY_MAX_LEN,
  SESSION_ATTRIBUTE_VALUE_MAX_LEN,
} from "./session.js";
export type {
  AttributeValue,
  SessionAttributes,
  RecordLlmCallInput,
  LlmCallRecord,
} from "./session.js";
export { instrumentOpenAI, instrumentAnthropic } from "./instrument/index.js";
export type { LlmCallSink } from "./instrument/index.js";
export { CausalTagger, CoWEngine } from "./engine.js";
export { ReadTracker } from "./tracker.js";
export type { ExternalContext } from "./tracker.js";
export { OutcomeBackPropagator } from "./outcome.js";
export { defaultConfig } from "./config.js";
export {
  OutcomeType,
  OUTCOME_MULTIPLIERS,
} from "./models.js";
export type {
  MemoryEntry,
  OutcomeRecord,
  Provenance,
  ArtifactRef,
  AMFSConfig,
  LayerConfig,
  RecallConfig,
  ScopeInfo,
  ScoredEntry,
} from "./models.js";
