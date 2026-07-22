import { API_URL, CLIPPER_AGENT_ID, OPS_PER_SAVE, ROOMS_TIERS, UPGRADE_URL } from "./config";
import { bumpLocalUsage, setUsageFromHeaders } from "./storage";
import type { ClipContent, Room, RoomsState } from "./types";

export class QuotaExceededError extends Error {
  upgradeUrl: string;
  constructor(upgradeUrl: string, message: string) {
    super(message);
    this.name = "QuotaExceededError";
    this.upgradeUrl = upgradeUrl;
  }
}

export class AuthError extends Error {
  constructor(message = "Invalid or missing API key") {
    super(message);
    this.name = "AuthError";
  }
}

interface WriteResult {
  version: number;
  [k: string]: unknown;
}

/**
 * Every call runs from the background service worker with host_permissions
 * for the API origin, so CORS (and the server's missing expose_headers)
 * doesn't apply and X-AMFS-* response headers are readable when present.
 */
async function request(
  apiKey: string,
  method: string,
  path: string,
  body?: unknown,
  opsCost = 0,
): Promise<Response> {
  const resp = await fetch(`${API_URL}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-AMFS-API-Key": apiKey,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  // Usage: exact from headers when the gateway sends them (dashboard traffic
  // does today; API-key traffic is being verified — see spec), otherwise a
  // local per-month estimate.
  const remaining = resp.headers.get("X-AMFS-Ops-Remaining");
  const limit = resp.headers.get("X-AMFS-Ops-Limit");
  if (remaining !== null && limit !== null) {
    await setUsageFromHeaders(Number(remaining), Number(limit));
  } else if (resp.ok && opsCost > 0) {
    await bumpLocalUsage(opsCost);
  }

  if (resp.status === 402) {
    let upgradeUrl = UPGRADE_URL;
    let detail = "You've used all your free memories this month.";
    try {
      const data = await resp.json();
      upgradeUrl = data.upgrade_url ?? data.plan_comparison_url ?? upgradeUrl;
      detail = data.detail ?? data.message ?? detail;
    } catch {
      // keep defaults
    }
    throw new QuotaExceededError(upgradeUrl, detail);
  }
  if (resp.status === 401 || resp.status === 403) {
    throw new AuthError();
  }
  if (!resp.ok) {
    throw new Error(`API error ${resp.status}: ${await resp.text()}`);
  }
  return resp;
}

export async function writeClip(
  apiKey: string,
  entityPath: string,
  key: string,
  clip: ClipContent,
): Promise<WriteResult> {
  const resp = await request(
    apiKey,
    "POST",
    "/api/v1/entries",
    {
      entity_path: entityPath,
      key,
      value: clip,
      confidence: 1.0,
      memory_type: "experience",
      agent_id: CLIPPER_AGENT_ID,
      shared: true,
    },
    OPS_PER_SAVE,
  );
  return resp.json();
}

export interface RetrievedEntry {
  entity_path: string;
  key: string;
  value: unknown;
  [k: string]: unknown;
}

export async function retrieve(
  apiKey: string,
  query: string,
  limit = 3,
): Promise<RetrievedEntry[]> {
  const resp = await request(apiKey, "POST", "/api/v1/retrieve", { query, limit }, 1);
  const data = await resp.json();
  return data.entries ?? data.results ?? [];
}

export async function whoami(apiKey: string): Promise<Record<string, unknown>> {
  const resp = await request(apiKey, "GET", "/api/v1/auth/whoami");
  return resp.json();
}

/**
 * Rooms are hosted-only (amfs_rooms). Tier check fails open like the Pro MCP:
 * if the endpoint is missing or errors, we don't unlock rooms but we also
 * don't break saving.
 */
export async function fetchRoomsState(apiKey: string): Promise<RoomsState> {
  let tier: string | null = null;
  try {
    const resp = await request(apiKey, "GET", "/api/v1/rooms/account-tier");
    const data = await resp.json();
    tier = (data.tier ?? data.account_tier ?? null) as string | null;
  } catch {
    tier = null;
  }

  const roomsUnlocked = tier !== null && ROOMS_TIERS.includes(tier.toLowerCase());
  let rooms: Room[] = [];
  if (roomsUnlocked) {
    try {
      const resp = await request(apiKey, "GET", "/api/v1/rooms");
      const data = await resp.json();
      rooms = Array.isArray(data) ? data : (data.rooms ?? []);
    } catch {
      rooms = [];
    }
  }
  return { tier, roomsUnlocked, rooms };
}
