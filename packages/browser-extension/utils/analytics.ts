import { getSettings } from "./storage";

/**
 * Mixpanel via HTTP ingestion API — the JS SDK can't run in an MV3 service
 * worker. Token is injected at build time (WXT_MIXPANEL_TOKEN); when absent
 * (dev, self-hosted) tracking is a no-op. Users can opt out in Options.
 */
const MIXPANEL_TOKEN = import.meta.env.WXT_MIXPANEL_TOKEN ?? "";

export type AnalyticsEvent =
  | "extension_save"
  | "extension_save_failed"
  | "extension_quota_hit"
  | "extension_upgrade_cta_clicked"
  | "extension_rooms_cta_clicked"
  | "extension_connected"
  | "extension_first_save"
  | "extension_identity_reassigned";

export async function track(
  event: AnalyticsEvent,
  props: Record<string, unknown> = {},
): Promise<void> {
  if (!MIXPANEL_TOKEN) return;
  const settings = await getSettings();
  if (!settings.analyticsEnabled) return;
  try {
    await fetch("https://api.mixpanel.com/track?verbose=0", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([
        {
          event,
          properties: {
            token: MIXPANEL_TOKEN,
            distinct_id: settings.accountEmail ?? "anonymous",
            source: "browser-extension",
            ...props,
          },
        },
      ]),
    });
  } catch {
    // analytics must never break the product
  }
}
