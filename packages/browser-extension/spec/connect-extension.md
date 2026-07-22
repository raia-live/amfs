# Spec: `/connect/extension` dashboard page (amfs-internal)

Companion work item for the browser extension in `packages/browser-extension`.
The extension ships with a paste-key fallback, so this page is not a launch
blocker — but it is the intended primary onboarding path.

## Flow

1. Extension opens `https://app.sense-lab.ai/connect/extension` in a new tab
   (origin configurable at build time via `WXT_DASHBOARD_URL`; it must match
   the `externally_connectable.matches` entry in the extension manifest).
2. If the visitor has no session: run the normal signup/login flow, then
   return to `/connect/extension`. **New users must be able to complete
   signup, auto-select the Free plan, and land back here in one uninterrupted
   pass** — extension installs will often precede account creation, and any
   detour kills the funnel.
3. The page shows a confirmation card ("Connect the SenseLab extension to
   <account email>?") with a Connect button.
4. On confirm, the page mints an API key and delivers it to the extension via
   `chrome.runtime.sendMessage`:

```js
chrome.runtime.sendMessage(
  EXTENSION_ID, // allowlist, see below
  {
    type: "amfs-connect",
    apiKey: "amfs_…",
    email: "user@example.com",
  },
  (response) => {
    // response.ok === true when the extension stored the key
  },
);
```

5. On `response.ok`, show "Connected — you can close this tab" plus a short
   "save your first page" nudge.

## Message contract (implemented in the extension)

The extension listens on `chrome.runtime.onMessageExternal` and accepts only:

| Field    | Type   | Required | Notes                                  |
|----------|--------|----------|----------------------------------------|
| `type`   | string | yes      | must be `"amfs-connect"`               |
| `apiKey` | string | yes      | raw API key, stored in `storage.local` |
| `email`  | string | no       | shown in Options; Mixpanel distinct_id |

Response: `{ ok: true }` or `{ ok: false, error: string }`.

## Key minting

- Label: `Browser extension` (visible in Settings → API Keys, revocable there).
- One key per connect; re-connecting mints a new key (old one can be revoked
  by the user; consider auto-revoking previous extension keys).
- **Scope: full-account `READ_WRITE` — do NOT scope to `web/**` only.** The
  extension writes to `web/<domain>` entities for personal saves, but Pro
  users also save into room entities whose paths are arbitrary, and the
  extension calls `GET /api/v1/rooms`, `GET /api/v1/rooms/account-tier`, and
  `POST /api/v1/retrieve` (first-save proof + future related-memories). A
  `web/**`-scoped key breaks all of those. If finer scoping is wanted later,
  it needs "web/** plus room entities plus rooms/retrieve endpoints"
  semantics that the scope model doesn't express today.

## Extension ID allowlist

`EXTENSION_ID` must be pinned server-side (env/config), not derived from the
request. Expect two IDs: the published Chrome Web Store ID (stable) and a dev
ID for unpacked builds. Reject `sendMessage` targets outside the allowlist.

## Verification items for amfs-internal (blocking questions)

1. **Usage headers on API-key traffic**: `docs/guides/saas-billing-metering.md`
   documents `X-AMFS-Ops-Remaining` / `X-AMFS-Ops-Limit` / `X-AMFS-Usage-Warning`
   for dashboard traffic. The extension reads them from every API response and
   falls back to local per-month save counting when absent. If the hosted
   gateway can also attach them to API-key responses, the extension's usage
   meter becomes exact for free — please confirm/enable.
2. **`GET /api/v1/rooms/account-tier`**: confirm the response field name
   (extension accepts `tier` or `account_tier`) and that it is reachable with
   an extension-minted key.
3. **Room entity path**: confirm the room objects returned by
   `GET /api/v1/rooms` include `entity_path` (the extension writes room saves
   to that entity; rooms without it fall back to personal memory).
4. **402 body**: extension uses `upgrade_url` / `plan_comparison_url` /
   `detail` from the 402 response; confirm shape.

## Growth recommendations (pricing decisions, not extension blockers)

1. **Let Free users JOIN rooms they're invited to** (room creation stays
   Pro+). Every paid team then recruits free users through invites; those
   users build clip habits and hit the free quota wall, which is the upgrade
   moment. Today the extension shows a locked "Team room — Pro" chip to
   free/starter users.
2. **Public share link per clip**: a branded read-only "Saved with SenseLab"
   page per clip gives free users a shareable artifact (the Loom/Notion
   loop). Would need a dashboard route + opt-in share flag on the entry.

## Analytics

The extension fires Mixpanel events (HTTP ingestion API, token injected at
build via `WXT_MIXPANEL_TOKEN`): `extension_connected`, `extension_save`,
`extension_first_save`, `extension_save_failed`, `extension_quota_hit`,
`extension_upgrade_cta_clicked`, `extension_rooms_cta_clicked`. Dashboard
funnels should join on account email (distinct_id).
