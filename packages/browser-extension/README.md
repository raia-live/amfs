# SenseLab — Save to Memory (browser extension)

Chrome/Edge/Firefox extension that saves web pages and text selections into
SenseLab agent memory. Clips are written through the hosted REST API
(`POST /api/v1/entries`) under the `web-clipper` agent and are immediately
retrievable by every agent on the account.

## Quick start

```bash
npm install
npm run dev        # hot-reloading dev build (Chromium)
npm run build      # production build -> .output/chrome-mv3
npm run compile    # typecheck
```

Load unpacked: `chrome://extensions` → Developer mode → Load unpacked →
`.output/chrome-mv3`.

## Layout

- `entrypoints/background.ts` — service worker: context menus, keyboard
  shortcut, save orchestration, badge, connect-flow message handler
- `entrypoints/extractor.ts` — Readability extraction, injected into the
  active tab only on explicit user action (activeTab + scripting; no
  `<all_urls>` content script)
- `entrypoints/popup/` — React popup: save button, destination picker
  (rooms for Pro+, locked upsell otherwise), usage meter, first-save moment
- `entrypoints/options/` — settings: API key, blocklist, privacy
- `utils/` — API client, storage, blocklist, key/slug helpers, analytics
- `spec/connect-extension.md` — spec for the amfs-internal
  `/connect/extension` dashboard page (one-click connect flow)

## Build-time env

| Variable | Default | Purpose |
|----------|---------|---------|
| `WXT_API_URL` | `https://amfs-login.sense-lab.ai` | AMFS HTTP API base |
| `WXT_DASHBOARD_URL` | `https://app.sense-lab.ai` | dashboard origin (connect flow + `externally_connectable`) |
| `WXT_MIXPANEL_TOKEN` | unset (analytics disabled) | Mixpanel project token |

Docs: `docs/guides/browser-extension.md` in the repo root.
