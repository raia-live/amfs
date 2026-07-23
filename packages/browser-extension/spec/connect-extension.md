# Spec: `/connect/extension` dashboard page (amfs-internal)

> **Status: IMPLEMENTED** in amfs-internal branch `feat/connect-extension`
> (`dashboard/src/app/connect/extension/`). The funnel below (login/signup
> `callbackUrl`, onboarding `returnTo` + sessionStorage across Stripe) is
> also implemented. The allowlist is wired into the deploy pipeline: the
> dashboard Dockerfile takes a `NEXT_PUBLIC_EXTENSION_IDS` build arg, fed
> from the `CHROME_EXTENSION_IDS` GitHub repo variable in both deploy
> workflows (prod `deploy.yml` and dev/env `deploy-env.yml`). Once the
> Chrome Web Store assigns the stable extension ID, run:
> `gh variable set CHROME_EXTENSION_IDS -R raia-live/amfs-internal -b "<store-id>"`
> and redeploy. While unset, the allowlist is empty = accept any `?ext=` id
> (dev/self-host behavior).

Companion work item for the browser extension in `packages/browser-extension`.
The extension ships with a paste-key fallback, so this page is not a launch
blocker — but it is the intended primary onboarding path.

## Flow

1. Extension opens `https://amfs.sense-lab.ai/connect/extension` in a new tab
   (origin configurable at build time via `WXT_DASHBOARD_URL`; it must match
   the `externally_connectable.matches` entry in the extension manifest).
   `amfs.sense-lab.ai` is the production dashboard (`AMFS_DASHBOARD_PUBLIC_URL`
   in amfs-internal deploy workflow).
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

The extension opens `/connect/extension?ext=<chrome.runtime.id>`. The page
validates `ext` against `NEXT_PUBLIC_EXTENSION_IDS` (comma-separated: the
published Chrome Web Store ID plus optionally a dev ID for unpacked builds)
and uses it as the `sendMessage` target. When the allowlist env is empty
(local dev / self-hosted), any provided id is accepted. If `sendMessage`
fails or is unavailable (Firefox has no `externally_connectable`), the page
falls back to showing the key once for manual paste into the extension's
Options page.

## Verified against amfs-internal (was: blocking questions)

1. **Usage headers on API-key traffic — confirmed.** The tenant middleware
   (`packages/tenant/src/amfs_tenant/http_deps.py`) attaches
   `X-AMFS-Ops-Remaining` / `X-AMFS-Ops-Limit` / `X-AMFS-Usage-Warning` to
   billable API-key responses (op cost > 0). Free routes (cost 0) don't carry
   them; the extension keeps its local fallback for that case.
2. **`GET /api/v1/rooms/account-tier` — confirmed.** Returns
   `{ plan_tier, plan_status, rooms_enabled }` (`amfs_rooms/routes.py`). The
   extension uses `rooms_enabled` directly.
3. **Room entity path — confirmed.** `GET /api/v1/rooms` returns
   `RoomSummaryResponse` objects with `id`, `entity_path`, `display_name`.
4. **402 body — confirmed.** The gateway wraps the payload as
   `{ "detail": { "error", "message", "upgrade_url", "plan_comparison_url" } }`
   (detail can also be a plain string for "No active plan"). The extension
   parses the nested shape.

## Key minting endpoint (existing, reuse it)

The dashboard BFF already exposes `POST /api/settings/api-keys`
(`dashboard/src/app/api/settings/api-keys/route.ts`), which proxies to the
OSS admin endpoint `POST /api/v1/admin/api-keys` and returns the raw key
once. The connect page posts:

```json
{
  "name": "Browser extension",
  "key_type": "agent",
  "rate_limit_rpm": 120,
  "scopes": [{ "pattern": "*", "permission": "read_write" }]
}
```

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
