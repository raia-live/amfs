import { browser, type Browser } from "wxt/browser";
import { defineBackground } from "wxt/utils/define-background";
import { track } from "@/utils/analytics";
import { AuthError, QuotaExceededError, fetchRoomsState, retrieve, writeClip } from "@/utils/api";
import { isBlockedHost, isUnsupportedUrl } from "@/utils/blocklist";
import { CLIPS_ENTITY, MAX_TEXT_BYTES } from "@/utils/config";
import { clipKey, entityForUrl } from "@/utils/slug";
import {
  getRoomsState,
  getSettings,
  setLastSave,
  setRoomsState,
  updateSettings,
} from "@/utils/storage";
import type {
  ClipContent,
  ExtractResult,
  SaveOutcome,
  SaveRequest,
  SaveTrigger,
} from "@/utils/types";

const MENU_SAVE_PAGE = "senselab-save-page";
const MENU_SAVE_SELECTION = "senselab-save-selection";

export default defineBackground(() => {
  browser.runtime.onInstalled.addListener(() => {
    browser.contextMenus.create({
      id: MENU_SAVE_PAGE,
      title: "Save page to SenseLab memory",
      contexts: ["page"],
    });
    browser.contextMenus.create({
      id: MENU_SAVE_SELECTION,
      title: "Save selection to SenseLab memory",
      contexts: ["selection"],
    });
  });

  browser.contextMenus.onClicked.addListener((info, tab) => {
    if (!tab?.id) return;
    if (info.menuItemId === MENU_SAVE_PAGE) {
      void saveAndNotify(tab, "context-menu", {});
    } else if (info.menuItemId === MENU_SAVE_SELECTION) {
      void saveAndNotify(tab, "context-menu", {
        selection: info.selectionText ?? undefined,
      });
    }
  });

  browser.commands.onCommand.addListener((command) => {
    if (command !== "save-page") return;
    void (async () => {
      const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
      if (tab) await saveAndNotify(tab, "shortcut", {});
    })();
  });

  // Popup -> background
  browser.runtime.onMessage.addListener((message: SaveRequest | { type: string }, _sender, sendResponse) => {
    void (async () => {
      if (message.type === "save") {
        const req = message as SaveRequest;
        const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
        if (!tab) {
          sendResponse({ ok: false, error: "No active tab" } satisfies SaveOutcome);
          return;
        }
        sendResponse(await saveAndNotify(tab, req.trigger, req));
      } else if (message.type === "refresh-rooms") {
        sendResponse(await refreshRooms());
      }
    })();
    return true; // async sendResponse
  });

  // Dashboard connect flow (externally_connectable). See spec/connect-extension.md.
  browser.runtime.onMessageExternal.addListener((message, _sender, sendResponse) => {
    void (async () => {
      if (message?.type === "amfs-connect" && typeof message.apiKey === "string") {
        await updateSettings({
          apiKey: message.apiKey,
          accountEmail: typeof message.email === "string" ? message.email : null,
        });
        await track("extension_connected");
        void refreshRooms();
        sendResponse({ ok: true });
      } else {
        sendResponse({ ok: false, error: "Unknown message" });
      }
    })();
    return true;
  });
});

async function refreshRooms() {
  const settings = await getSettings();
  if (!settings.apiKey) return null;
  try {
    const state = await fetchRoomsState(settings.apiKey);
    await setRoomsState(state);
    return state;
  } catch {
    return getRoomsState();
  }
}

async function saveAndNotify(
  tab: Browser.tabs.Tab,
  trigger: SaveTrigger,
  options: Pick<SaveRequest, "selection" | "note" | "roomId">,
): Promise<SaveOutcome> {
  const outcome = await performSave(tab, trigger, options);
  if (outcome.ok) {
    void flashBadge("✓", "#22c55e");
  } else if (outcome.quotaHit) {
    void flashBadge("!", "#f59e0b");
  } else {
    void flashBadge("✕", "#ef4444");
  }
  return outcome;
}

async function performSave(
  tab: Browser.tabs.Tab,
  trigger: SaveTrigger,
  options: Pick<SaveRequest, "selection" | "note" | "roomId">,
): Promise<SaveOutcome> {
  const settings = await getSettings();
  if (!settings.apiKey) return { ok: false, blocked: "not-connected" };

  const url = tab.url;
  if (isUnsupportedUrl(url) || !tab.id) {
    return { ok: false, blocked: "unsupported-page" };
  }
  const hostname = new URL(url!).hostname;
  if (settings.disabledSites.includes(hostname)) {
    return { ok: false, blocked: "disabled-site" };
  }
  if (isBlockedHost(hostname, settings.customBlocklist)) {
    return { ok: false, blocked: "blocklist" };
  }

  // Extraction runs only here — explicit user action grants activeTab.
  let extracted: ExtractResult;
  try {
    const results = await browser.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["/extractor.js"],
    });
    extracted = results[0]?.result as ExtractResult;
    if (!extracted) throw new Error("empty extraction result");
  } catch (e) {
    return { ok: false, error: `Could not read this page: ${(e as Error).message}` };
  }

  const selection = options.selection ?? extracted.selection;
  const clip: ClipContent = {
    url: url!,
    title: extracted.title,
    text: truncateUtf8(selection && trigger === "context-menu" ? selection : extracted.text, MAX_TEXT_BYTES),
    byline: extracted.byline,
    excerpt: extracted.excerpt,
    selection: selection ? truncateUtf8(selection, MAX_TEXT_BYTES) : undefined,
    note: options.note || undefined,
    saved_at: new Date().toISOString(),
  };

  // Destination: personal memory (web/<domain>) or a room's entity.
  let entityPath = entityForUrl(url!);
  let destination: "memory" | "room" = "memory";
  const roomId = options.roomId ?? settings.defaultDestination;
  if (roomId) {
    const roomsState = await getRoomsState();
    const room = roomsState?.rooms.find((r) => r.room_id === roomId);
    if (room?.entity_path) {
      entityPath = room.entity_path;
      destination = "room";
    }
    // Room gone or tier downgraded: fall back to personal memory silently
    // rather than losing the save.
  }
  if (entityPath === "web/clips") entityPath = CLIPS_ENTITY;

  const key = clipKey(clip.title, clip.url);
  try {
    const result = await writeClip(settings.apiKey, entityPath, key, clip);
    const lastSave = {
      url: clip.url,
      title: clip.title,
      key,
      entityPath,
      version: result.version ?? 1,
      at: clip.saved_at,
      destination,
    };
    await setLastSave(lastSave);
    await track("extension_save", { trigger, destination, domain: hostname });

    // First-save aha: prove the memory is live with one retrieve (1 op, once).
    let firstSave: SaveOutcome["firstSave"];
    if (!settings.firstSaveDone) {
      let retrieved = false;
      try {
        const hits = await retrieve(settings.apiKey, clip.title, 1);
        retrieved = hits.length > 0;
      } catch {
        retrieved = false;
      }
      await updateSettings({ firstSaveDone: true });
      await track("extension_first_save", { retrieved });
      firstSave = { retrieved, topic: clip.title };
    }

    return { ok: true, lastSave, firstSave };
  } catch (e) {
    if (e instanceof QuotaExceededError) {
      await track("extension_quota_hit");
      return { ok: false, quotaHit: true, upgradeUrl: e.upgradeUrl, error: e.message };
    }
    if (e instanceof AuthError) {
      return { ok: false, blocked: "not-connected", error: e.message };
    }
    await track("extension_save_failed", { error: (e as Error).message?.slice(0, 120) });
    return { ok: false, error: (e as Error).message };
  }
}

function truncateUtf8(text: string, maxBytes: number): string {
  const encoder = new TextEncoder();
  if (encoder.encode(text).length <= maxBytes) return text;
  // Binary-search the largest prefix under the byte budget.
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (encoder.encode(text.slice(0, mid)).length <= maxBytes - 1) lo = mid;
    else hi = mid - 1;
  }
  return `${text.slice(0, lo)}…`;
}

async function flashBadge(text: string, color: string): Promise<void> {
  await browser.action.setBadgeBackgroundColor({ color });
  await browser.action.setBadgeText({ text });
  setTimeout(() => {
    void browser.action.setBadgeText({ text: "" });
  }, 2500);
}
