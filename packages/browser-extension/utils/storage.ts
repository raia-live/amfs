import { browser } from "wxt/browser";
import { FREE_TIER_OPS_LIMIT } from "./config";
import type { LastSave, RoomsState, Settings, UsageInfo } from "./types";

const SETTINGS_KEY = "settings";
const USAGE_KEY = "usage";
const LAST_SAVE_KEY = "lastSave";
const ROOMS_KEY = "roomsState";

export const DEFAULT_SETTINGS: Settings = {
  apiKey: null,
  accountEmail: null,
  agentId: null,
  defaultDestination: null,
  disabledSites: [],
  customBlocklist: [],
  analyticsEnabled: true,
  firstSaveDone: false,
};

export async function getSettings(): Promise<Settings> {
  const stored = await browser.storage.local.get(SETTINGS_KEY);
  return { ...DEFAULT_SETTINGS, ...(stored[SETTINGS_KEY] ?? {}) };
}

export async function updateSettings(patch: Partial<Settings>): Promise<Settings> {
  const current = await getSettings();
  const next = { ...current, ...patch };
  await browser.storage.local.set({ [SETTINGS_KEY]: next });
  return next;
}

function currentMonth(): string {
  return new Date().toISOString().slice(0, 7);
}

export async function getUsage(): Promise<UsageInfo> {
  const stored = await browser.storage.local.get(USAGE_KEY);
  const usage = stored[USAGE_KEY] as UsageInfo | undefined;
  // Local counters reset each calendar month; header-sourced values are
  // authoritative and carry whatever period the server enforces.
  if (!usage || (usage.source === "local" && usage.month !== currentMonth())) {
    return { opsUsed: 0, opsLimit: FREE_TIER_OPS_LIMIT, source: "local", month: currentMonth() };
  }
  return usage;
}

/** Exact usage from X-AMFS-Ops-Remaining / X-AMFS-Ops-Limit headers. */
export async function setUsageFromHeaders(remaining: number, limit: number): Promise<UsageInfo> {
  const usage: UsageInfo = {
    opsUsed: Math.max(0, limit - remaining),
    opsLimit: limit,
    source: "headers",
    month: currentMonth(),
  };
  await browser.storage.local.set({ [USAGE_KEY]: usage });
  return usage;
}

/** Fallback estimate when the hosted gateway doesn't send usage headers. */
export async function bumpLocalUsage(ops: number): Promise<UsageInfo> {
  const current = await getUsage();
  if (current.source === "headers" && current.month === currentMonth()) return current;
  const usage: UsageInfo = {
    opsUsed: current.opsUsed + ops,
    opsLimit: current.opsLimit,
    source: "local",
    month: currentMonth(),
  };
  await browser.storage.local.set({ [USAGE_KEY]: usage });
  return usage;
}

export async function getLastSave(): Promise<LastSave | null> {
  const stored = await browser.storage.local.get(LAST_SAVE_KEY);
  return (stored[LAST_SAVE_KEY] as LastSave | undefined) ?? null;
}

export async function setLastSave(save: LastSave): Promise<void> {
  await browser.storage.local.set({ [LAST_SAVE_KEY]: save });
}

export async function getRoomsState(): Promise<RoomsState | null> {
  const stored = await browser.storage.local.get(ROOMS_KEY);
  return (stored[ROOMS_KEY] as RoomsState | undefined) ?? null;
}

export async function setRoomsState(state: RoomsState): Promise<void> {
  await browser.storage.local.set({ [ROOMS_KEY]: state });
}
