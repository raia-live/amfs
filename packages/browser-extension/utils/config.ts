declare const __AMFS_API_URL__: string;
declare const __AMFS_DASHBOARD_URL__: string;

export const API_URL = __AMFS_API_URL__;
export const DASHBOARD_URL = __AMFS_DASHBOARD_URL__;

export const CONNECT_URL = `${DASHBOARD_URL}/connect/extension`;
export const UPGRADE_URL = `${DASHBOARD_URL}/settings/usage`;

/** Agent identity every clip is attributed to. */
export const CLIPPER_AGENT_ID = "web-clipper";

/** Catch-all entity for selections and notes without a clean domain. */
export const CLIPS_ENTITY = "web/clips";

/** Free tier: 1K ops/month, a write costs 2 ops => ~500 memories. */
export const FREE_TIER_OPS_LIMIT = 1000;
export const OPS_PER_SAVE = 2;

/** Soft-warning thresholds, matching the server's X-AMFS-Usage-Warning. */
export const WARN_THRESHOLD = 0.8;
export const CRITICAL_THRESHOLD = 0.95;

/** Max extracted text size sent to the API (~40KB). */
export const MAX_TEXT_BYTES = 40_000;

/** Tiers that unlock Rooms (mirrors dashboard PRO_TIERS / Pro MCP _ROOMS_TIERS). */
export const ROOMS_TIERS = ["pro", "teams", "enterprise"];
