export interface ClipContent {
  url: string;
  title: string;
  text: string;
  byline?: string;
  excerpt?: string;
  selection?: string;
  note?: string;
  saved_at: string;
}

export interface Settings {
  apiKey: string | null;
  accountEmail: string | null;
  /** null = personal memory; otherwise a room id from the rooms list */
  defaultDestination: string | null;
  /** hostnames the user disabled saving on ("never save here") */
  disabledSites: string[];
  /** user-added blocklist entries, on top of the built-in defaults */
  customBlocklist: string[];
  analyticsEnabled: boolean;
  firstSaveDone: boolean;
}

export interface UsageInfo {
  /** ops, not memories; divide by OPS_PER_SAVE for user-facing counts */
  opsUsed: number;
  opsLimit: number;
  /** "headers" = exact from API responses; "local" = client-side estimate */
  source: "headers" | "local";
  /** "YYYY-MM" the local counter applies to */
  month: string;
}

export interface LastSave {
  url: string;
  title: string;
  key: string;
  entityPath: string;
  version: number;
  at: string;
  destination: "memory" | "room";
}

export interface Room {
  room_id: string;
  name?: string;
  entity_path?: string;
  [k: string]: unknown;
}

export interface RoomsState {
  tier: string | null;
  roomsUnlocked: boolean;
  rooms: Room[];
}

export type SaveTrigger = "popup" | "shortcut" | "context-menu";

export interface SaveRequest {
  type: "save";
  trigger: SaveTrigger;
  /** selection text when triggered from the selection context menu */
  selection?: string;
  note?: string;
  /** room id, or null for personal memory */
  roomId?: string | null;
}

export interface SaveOutcome {
  ok: boolean;
  quotaHit?: boolean;
  upgradeUrl?: string;
  blocked?: "blocklist" | "disabled-site" | "not-connected" | "unsupported-page";
  error?: string;
  lastSave?: LastSave;
  /** set on the very first successful save: proof the memory is live */
  firstSave?: { retrieved: boolean; topic: string };
}

export interface ExtractResult {
  title: string;
  text: string;
  byline?: string;
  excerpt?: string;
  selection?: string;
}
