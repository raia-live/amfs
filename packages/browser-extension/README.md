# SenseLab — Save to Memory (browser extension)

Chrome/Edge/Firefox extension that saves web pages and text selections into
SenseLab agent memory. Clips are written through the hosted REST API
(`POST /api/v1/entries`) under a `web-clipper` agent and are immediately
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

## Agent identity convention

Clips are authored under `web-clipper` (`CLIPPER_AGENT_ID` in
`utils/config.ts`) — but an agent identity is owned by exactly one user per
account, and the API answers 409 to a write carrying somebody else's identity.
On a shared account the first person to save claims `web-clipper`, so each
further user falls back to a derived variant (`web-clipper-<name>`, see
`utils/identity.ts`). The resolved id is stored in settings and shown in
Options.

The fallbacks are derived from the account email rather than random, so a
reinstall returns to the same identity: AMFS has no agent rename/merge, and a
split identity can only be reconciled by a costly multi-table migration.

When working on this extension with AMFS memory, set your identity to the
canonical id so build notes and real user clips stay on one agent page (and
reuse/recall metrics don't get split):

```
amfs_set_identity("web-clipper", "<what you're working on>", model="<your-model>")
```

Do **not** use ad-hoc dev ids like `chrome-extension-agent` for extension work —
AMFS has no agent rename/merge, so a split identity can only be reconciled by a
costly multi-table migration.

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
| `WXT_DASHBOARD_URL` | `https://amfs.sense-lab.ai` | dashboard origin (connect flow + `externally_connectable`) |
| `WXT_MIXPANEL_TOKEN` | unset (analytics disabled) | Mixpanel project token |

Docs: `docs/guides/browser-extension.md` in the repo root.
