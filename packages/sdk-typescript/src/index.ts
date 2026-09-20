/**
 * @senselab-ai/amfs — Agent Memory File System TypeScript SDK
 */

export { AgentMemory, MemoryScope, BRANCH_ENV, MEMORY_BRANCH_ATTRIBUTE } from "./memory.js";
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
  SDK_STAMPED_ATTRIBUTES,
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
export {
  ReplayReceiver,
  ReplayError,
  SignatureError,
  PayloadError,
  parseReplayRequest,
  signReplayBody,
  verifyReplaySignature,
  replayAttributes,
  replayOutcomeRef,
  createFetchHandler,
  commitAttributes,
  serveReplay,
  CASE_ID_ATTRIBUTE,
  EVENT_PING,
  EVENT_REPLAY_REQUESTED,
  SIGNATURE_HEADER,
  EVENT_HEADER,
  DELIVERY_HEADER,
} from "./replay.js";
export type {
  ReplayRequest,
  ReplayResult,
  ReplayAnswer,
  ReplayMemory,
  ReplayReceiverOptions,
  ReplayResponse,
  HeaderBag,
} from "./replay.js";
export { defaultConfig } from "./config.js";
export {
  MemoryType,
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
