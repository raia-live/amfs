import { defineConfig } from "wxt";

/**
 * API + dashboard origins. The dashboard origin must stay in sync with the
 * amfs-internal /connect/extension page (see spec/connect-extension.md).
 * Override at build time: WXT_API_URL / WXT_DASHBOARD_URL.
 */
const API_URL = import.meta.env.WXT_API_URL ?? "https://amfs-login.sense-lab.ai";
const DASHBOARD_URL =
  import.meta.env.WXT_DASHBOARD_URL ?? "https://app.sense-lab.ai";

export default defineConfig({
  modules: ["@wxt-dev/module-react"],
  manifest: {
    name: "SenseLab — Save to Memory",
    description:
      "Save any page or selection into your SenseLab agent memory. Your agents remember what you read.",
    // activeTab + scripting: extraction runs ONLY on explicit user action
    // (popup click, shortcut, context menu). No <all_urls> content script,
    // which keeps Chrome Web Store review simple and the privacy story clean.
    permissions: ["activeTab", "scripting", "storage", "contextMenus"],
    host_permissions: [`${API_URL}/*`],
    externally_connectable: {
      matches: [`${DASHBOARD_URL}/*`],
    },
    commands: {
      "save-page": {
        suggested_key: { default: "Ctrl+Shift+S", mac: "Command+Shift+S" },
        description: "Save the current page to SenseLab memory",
      },
    },
    web_accessible_resources: [],
  },
  vite: () => ({
    define: {
      __AMFS_API_URL__: JSON.stringify(API_URL),
      __AMFS_DASHBOARD_URL__: JSON.stringify(DASHBOARD_URL),
    },
  }),
});
