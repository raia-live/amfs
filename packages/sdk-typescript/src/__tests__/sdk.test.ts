import { describe, it, expect } from "vitest";
import {
  AgentMemory,
  InMemoryAdapter,
  CausalTagger,
  CoWEngine,
  ReadTracker,
  OutcomeBackPropagator,
  OutcomeType,
  OUTCOME_MULTIPLIERS,
} from "../index.js";
import type { MemoryEntry, Provenance } from "../index.js";

function makeProvenance(agentId = "test-agent"): Provenance {
  return {
    agentId,
    sessionId: "sess-1",
    writtenAt: new Date().toISOString(),
    patternRefs: [],
  };
}

function makeEntry(overrides?: Partial<MemoryEntry>): MemoryEntry {
  return {
    amfsVersion: "0.1.0",
    entityPath: "checkout-service",
    key: "retry-pattern",
    version: 1,
    value: { maxRetries: 3 },
    provenance: makeProvenance(),
    confidence: 1.0,
    outcomeCount: 0,
    ttlAt: null,
    shared: true,
    contentHash: null,
    integrityChain: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------
// InMemoryAdapter contract tests
// ---------------------------------------------------------------

describe("InMemoryAdapter", () => {
  it("write and read", () => {
    const adapter = new InMemoryAdapter();
    const written = adapter.write(makeEntry());
    expect(written.version).toBe(1);

    const result = adapter.read("checkout-service", "retry-pattern");
    expect(result).not.toBeNull();
    expect(result!.version).toBe(1);
  });

  it("read nonexistent returns null", () => {
    const adapter = new InMemoryAdapter();
    expect(adapter.read("nope", "nope")).toBeNull();
  });

  it("write increments version", () => {
    const adapter = new InMemoryAdapter();
    adapter.write(makeEntry());
    const w2 = adapter.write(makeEntry({ value: { updated: true } }));
    expect(w2.version).toBe(2);
  });

  it("min confidence filter", () => {
    const adapter = new InMemoryAdapter();
    adapter.write(makeEntry({ confidence: 0.5 }));
    expect(adapter.read("checkout-service", "retry-pattern", { minConfidence: 0.3 })).not.toBeNull();
    expect(adapter.read("checkout-service", "retry-pattern", { minConfidence: 0.8 })).toBeNull();
  });

  it("list current only", () => {
    const adapter = new InMemoryAdapter();
    adapter.write(makeEntry({ key: "key-a" }));
    adapter.write(makeEntry({ key: "key-b" }));
    adapter.write(makeEntry({ key: "key-a", value: { v: 2 } }));

    const entries = adapter.list("checkout-service");
    expect(entries).toHaveLength(2);
    const keyA = entries.find((e) => e.key === "key-a");
    expect(keyA!.version).toBe(2);
  });

  it("list include superseded", () => {
    const adapter = new InMemoryAdapter();
    adapter.write(makeEntry({ key: "key-x" }));
    adapter.write(makeEntry({ key: "key-x", value: { v: 2 } }));

    const entries = adapter.list("checkout-service", { includeSuperseded: true });
    const versions = entries.filter((e) => e.key === "key-x").map((e) => e.version).sort();
    expect(versions).toEqual([1, 2]);
  });

  it("commit outcome p1 incident", () => {
    const adapter = new InMemoryAdapter();
    adapter.write(makeEntry());

    const updated = adapter.commitOutcome({
      outcomeRef: "INC-001",
      outcomeType: OutcomeType.P1_INCIDENT,
      causalConfidence: 1.0,
      committedAt: new Date().toISOString(),
      causalEntryKeys: ["checkout-service/retry-pattern"],
      agentId: "release-agent",
    });
    expect(updated).toHaveLength(1);
    expect(updated[0].confidence).toBeCloseTo(1.15);
    expect(updated[0].outcomeCount).toBe(1);
  });

  it("commit outcome clean deploy", () => {
    const adapter = new InMemoryAdapter();
    adapter.write(makeEntry());

    const updated = adapter.commitOutcome({
      outcomeRef: "DEP-001",
      outcomeType: OutcomeType.CLEAN_DEPLOY,
      causalConfidence: 1.0,
      committedAt: new Date().toISOString(),
      causalEntryKeys: ["checkout-service/retry-pattern"],
      agentId: "release-agent",
    });
    expect(updated).toHaveLength(1);
    expect(updated[0].confidence).toBeCloseTo(0.97);
  });

  it("watch receives new writes", () => {
    const adapter = new InMemoryAdapter();
    const received: MemoryEntry[] = [];
    const handle = adapter.watch("checkout-service", (e) => received.push(e));

    adapter.write(makeEntry());
    expect(received).toHaveLength(1);
    expect(received[0].key).toBe("retry-pattern");

    handle.cancel();
    expect(handle.cancelled).toBe(true);
  });
});

// ---------------------------------------------------------------
// CausalTagger
// ---------------------------------------------------------------

describe("CausalTagger", () => {
  it("creates provenance", () => {
    const tagger = new CausalTagger("test-agent", "sess-1");
    const prov = tagger.tag();
    expect(prov.agentId).toBe("test-agent");
    expect(prov.sessionId).toBe("sess-1");
    expect(prov.patternRefs).toEqual([]);
  });

  it("auto session id", () => {
    const tagger = new CausalTagger("a");
    expect(tagger.sessionId).toMatch(/^sess-/);
  });
});

// ---------------------------------------------------------------
// CoWEngine
// ---------------------------------------------------------------

describe("CoWEngine", () => {
  it("write first version", () => {
    const adapter = new InMemoryAdapter();
    const engine = new CoWEngine(adapter, new CausalTagger("eng"));
    const written = engine.write("svc", "k", { data: 42 });
    expect(written.version).toBe(1);
    expect(written.provenance.agentId).toBe("eng");
  });

  it("write increments version", () => {
    const adapter = new InMemoryAdapter();
    const engine = new CoWEngine(adapter, new CausalTagger("eng"));
    engine.write("svc", "k", { v: 1 });
    const w2 = engine.write("svc", "k", { v: 2 });
    expect(w2.version).toBe(2);
  });

  it("read and list delegate", () => {
    const adapter = new InMemoryAdapter();
    const engine = new CoWEngine(adapter, new CausalTagger("eng"));
    engine.write("svc", "k1", 1);
    engine.write("svc", "k2", 2);
    expect(engine.read("svc", "k1")).not.toBeNull();
    expect(engine.list("svc")).toHaveLength(2);
  });
});

// ---------------------------------------------------------------
// OutcomeBackPropagator
// ---------------------------------------------------------------

describe("OutcomeBackPropagator", () => {
  it("compute new confidence", () => {
    const result = OutcomeBackPropagator.computeNewConfidence(1.0, OutcomeType.P1_INCIDENT, 0.9);
    expect(result).toBeCloseTo(1.0 * 1.15 * 0.9);
  });

  it("make record", () => {
    const record = OutcomeBackPropagator.makeRecord("INC", OutcomeType.REGRESSION, ["svc/k"], "a");
    expect(record.outcomeRef).toBe("INC");
    expect(record.causalConfidence).toBe(1.0);
  });

  it("all outcome types have multipliers", () => {
    for (const ot of Object.values(OutcomeType)) {
      expect(OUTCOME_MULTIPLIERS[ot]).toBeDefined();
    }
  });
});

// ---------------------------------------------------------------
// AgentMemory (main SDK)
// ---------------------------------------------------------------

describe("AgentMemory", () => {
  it("write and read", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "key", { data: 1 });
    const entry = mem.read("svc", "key");
    expect(entry).not.toBeNull();
    expect(entry!.value).toEqual({ data: 1 });
  });

  it("version increments", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "key", "v1");
    mem.write("svc", "key", "v2");
    const entry = mem.read("svc", "key");
    expect(entry!.version).toBe(2);
  });

  it("list entries", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "k1", "a");
    mem.write("svc", "k2", "b");
    expect(mem.list("svc")).toHaveLength(2);
  });

  it("commit outcome", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "key", "data");
    const updated = mem.commitOutcome("INC-001", OutcomeType.P1_INCIDENT, ["svc/key"]);
    expect(updated).toHaveLength(1);
    expect(updated[0].confidence).toBeCloseTo(1.15);
  });

  it("properties", () => {
    const mem = new AgentMemory("test-agent");
    expect(mem.agentId).toBe("test-agent");
    expect(mem.sessionId).toMatch(/^sess-/);
    expect(mem.namespace).toBe("default");
  });

  it("write with confidence and pattern refs", () => {
    const mem = new AgentMemory("test-agent");
    const entry = mem.write("svc", "key", "val", {
      confidence: 0.8,
      patternRefs: ["retry-logic"],
    });
    expect(entry.confidence).toBe(0.8);
    expect(entry.provenance.patternRefs).toEqual(["retry-logic"]);
  });

  it("min confidence filter", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "key", "val", { confidence: 0.3 });
    expect(mem.read("svc", "key", { minConfidence: 0.5 })).toBeNull();
    expect(mem.read("svc", "key", { minConfidence: 0.2 })).not.toBeNull();
  });

  it("read tracker records reads automatically", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "k1", "v1");
    mem.write("svc", "k2", "v2");
    mem.read("svc", "k1");
    mem.read("svc", "k2");
    expect(mem.readTracker.readCount).toBe(2);
    expect(mem.readTracker.causalKeys).toContain("svc/k1");
    expect(mem.readTracker.causalKeys).toContain("svc/k2");
  });

  it("auto-causal commitOutcome uses read tracker", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "k1", "v1");
    mem.read("svc", "k1");
    const updated = mem.commitOutcome("DEP-1", OutcomeType.CLEAN_DEPLOY);
    expect(updated).toHaveLength(1);
    expect(updated[0].confidence).toBeCloseTo(0.97);
  });

  it("clearReadLog resets tracker", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "k1", "v1");
    mem.read("svc", "k1");
    expect(mem.readTracker.readCount).toBe(1);
    mem.clearReadLog();
    expect(mem.readTracker.readCount).toBe(0);
  });

  it("history returns all versions", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "k", "v1");
    mem.write("svc", "k", "v2");
    mem.write("svc", "k", "v3");
    const versions = mem.history("svc", "k");
    expect(versions).toHaveLength(3);
    expect(versions[0].version).toBe(1);
    expect(versions[2].version).toBe(3);
  });

  it("search with confidence filter", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "high", "v", { confidence: 0.9 });
    mem.write("svc", "low", "v", { confidence: 0.2 });
    const results = mem.search({ minConfidence: 0.5 });
    expect(results).toHaveLength(1);
    expect(results[0].key).toBe("high");
  });

  it("search with limit", () => {
    const mem = new AgentMemory("test-agent");
    for (let i = 0; i < 10; i++) {
      mem.write("svc", `k${i}`, `v${i}`);
    }
    const results = mem.search({ limit: 3 });
    expect(results).toHaveLength(3);
  });

  it("stats returns correct counts", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc-a", "k1", "v1");
    mem.write("svc-b", "k2", "v2");
    const s = mem.stats();
    expect(s.totalEntries).toBe(2);
    expect(s.totalEntities).toBe(2);
    expect(s.totalAgents).toBe(1);
  });

  it("recordContext and explain", () => {
    const mem = new AgentMemory("test-agent");
    mem.write("svc", "pattern", "retry logic");
    mem.read("svc", "pattern");
    mem.recordContext("pagerduty", "3 SEV-1 incidents", { source: "PagerDuty API" });
    const result = mem.explain("DEP-500");
    expect(result.outcomeRef).toBe("DEP-500");
    expect(result.agentId).toBe("test-agent");
    expect(result.causalChainLength).toBe(1);
    expect((result.causalEntries as unknown[]).length).toBe(1);
    expect((result.externalContexts as unknown[]).length).toBe(1);
  });
});

// ---------------------------------------------------------------
// ReadTracker
// ---------------------------------------------------------------

describe("ReadTracker", () => {
  it("records reads and returns causal keys in order", () => {
    const tracker = new ReadTracker();
    tracker.record(makeEntry({ entityPath: "svc", key: "k1" }));
    tracker.record(makeEntry({ entityPath: "svc", key: "k2" }));
    expect(tracker.causalKeys).toEqual(["svc/k1", "svc/k2"]);
  });

  it("records external contexts", () => {
    const tracker = new ReadTracker();
    tracker.recordContext("pd", "3 incidents", { source: "PagerDuty" });
    expect(tracker.externalContexts).toHaveLength(1);
    expect(tracker.externalContexts[0].source).toBe("PagerDuty");
  });

  it("read version tracking", () => {
    const tracker = new ReadTracker();
    tracker.record(makeEntry({ entityPath: "svc", key: "k1", version: 5 }));
    expect(tracker.readVersion("svc/k1")).toBe(5);
    expect(tracker.readVersion("svc/missing")).toBeUndefined();
  });

  it("clear resets everything", () => {
    const tracker = new ReadTracker();
    tracker.record(makeEntry({ entityPath: "svc", key: "k1" }));
    tracker.recordContext("test", "test");
    tracker.clear();
    expect(tracker.readCount).toBe(0);
    expect(tracker.externalContexts).toHaveLength(0);
  });

  it("contains check", () => {
    const tracker = new ReadTracker();
    tracker.record(makeEntry({ entityPath: "svc", key: "k1" }));
    expect(tracker.contains("svc/k1")).toBe(true);
    expect(tracker.contains("svc/k2")).toBe(false);
  });
});
