import { API_URL, OPS_PER_SAVE, ROOMS_TIERS, UPGRADE_URL } from "./config";
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

/**
 * The agent identity on the write is owned by another user in the account.
 * Recoverable: the caller retries under the next candidate identity.
 */
export class AgentIdentityConflictError extends Error {
  constructor(message = "Another member of this account already uses this saving identity.") {
    super(message);
    this.name = "AgentIdentityConflictError";
  }
}

export interface WriteResult {
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
    // Hosted gateway wraps the payload: { detail: { error, message,
    // upgrade_url, plan_comparison_url } } — detail can also be a string.
    let upgradeUrl = UPGRADE_URL;
    let message = "You've used all your free memories this month.";
    try {
      const data = await resp.json();
      const detail = typeof data.detail === "object" && data.detail !== null ? data.detail : data;
      upgradeUrl = detail.upgrade_url ?? detail.plan_comparison_url ?? upgradeUrl;
      message =
        (typeof data.detail === "string" ? data.detail : detail.message) ?? message;
    } catch {
      // keep defaults
    }
    throw new QuotaExceededError(upgradeUrl, message);
  }
  if (resp.status === 401 || resp.status === 403) {
    throw new AuthError();
  }
  if (!resp.ok) {
    const detail = await resp.text();
    if (resp.status === 409 && /agent identity/i.test(detail)) {
      throw new AgentIdentityConflictError();
    }
    throw new Error(`API error ${resp.status}: ${detail}`);
  }
  return resp;
}

export async function writeClip(
  apiKey: string,
  entityPath: string,
  key: string,
  clip: ClipContent,
  agentId: string,
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
      agent_id: agentId,
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
  // Hosted API returns a bare array; tolerate wrapped shapes too.
  return Array.isArray(data) ? data : (data.entries ?? data.results ?? []);
}

export async function whoami(apiKey: string): Promise<Record<string, unknown>> {
  const resp = await request(apiKey, "GET", "/api/v1/auth/whoami");
  return resp.json();
}

function firstNonEmpty(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return undefined;
}

/**
 * Rooms are hosted-only (amfs_rooms). GET /api/v1/rooms/account-tier returns
 * { plan_tier, plan_status, rooms_enabled }; if the endpoint is missing or
 * errors we don't unlock rooms but we also don't break saving.
 */
export async function fetchRoomsState(apiKey: string): Promise<RoomsState> {
  let tier: string | null = null;
  let roomsUnlocked = false;
  try {
    const resp = await request(apiKey, "GET", "/api/v1/rooms/account-tier");
    const data = await resp.json();
    tier = (data.plan_tier ?? null) as string | null;
    roomsUnlocked =
      typeof data.rooms_enabled === "boolean"
        ? data.rooms_enabled
        : tier !== null && ROOMS_TIERS.includes(tier.toLowerCase());
  } catch {
    roomsUnlocked = false;
  }

  let rooms: Room[] = [];
  if (roomsUnlocked) {
    try {
      const resp = await request(apiKey, "GET", "/api/v1/rooms");
      const data = await resp.json();
      const raw: Record<string, unknown>[] = Array.isArray(data) ? data : (data.rooms ?? []);
      // RoomSummaryResponse: { id, entity_path, display_name, ... }. display_name
      // can come back as "", which must not become a blank label in the picker.
      rooms = raw.map((r) => ({
        room_id: String(r.id ?? r.room_id ?? ""),
        name: firstNonEmpty(r.display_name, r.name, r.slug),
        entity_path: firstNonEmpty(r.entity_path),
      }));
    } catch {
      rooms = [];
    }
  }
  return { tier, roomsUnlocked, rooms };
}
