/**
 * OutcomeBackPropagator — TypeScript port.
 */

import type { AmfsAdapter } from "./adapter.js";
import type { MemoryEntry, OutcomeRecord } from "./models.js";
import { OUTCOME_MULTIPLIERS, OutcomeType } from "./models.js";

export class OutcomeBackPropagator {
  private adapter: AmfsAdapter;

  constructor(adapter: AmfsAdapter) {
    this.adapter = adapter;
  }

  propagate(record: OutcomeRecord): MemoryEntry[] {
    return this.adapter.commitOutcome(record);
  }

  propagateBatch(records: OutcomeRecord[]): MemoryEntry[] {
    const allUpdated: MemoryEntry[] = [];
    for (const record of records) {
      allUpdated.push(...this.propagate(record));
    }
    return allUpdated;
  }

  static computeNewConfidence(
    currentConfidence: number,
    outcomeType: OutcomeType,
    causalConfidence = 1.0
  ): number {
    return currentConfidence * OUTCOME_MULTIPLIERS[outcomeType] * causalConfidence;
  }

  static makeRecord(
    outcomeRef: string,
    outcomeType: OutcomeType,
    causalEntryKeys: string[],
    agentId: string,
    options?: { causalConfidence?: number; committedAt?: string }
  ): OutcomeRecord {
    return {
      outcomeRef,
      outcomeType,
      causalConfidence: options?.causalConfidence ?? 1.0,
      committedAt: options?.committedAt ?? new Date().toISOString(),
      causalEntryKeys,
      agentId,
    };
  }
}
