---
title: Browser Extension
layout: default
parent: Guides
nav_order: 11
description: "Save web pages and highlights into your agent memory with the SenseLab browser extension."
---

# Browser Extension (Web Clipper)
{: .no_toc }

The SenseLab extension saves any web page or text selection into your agent memory. Everything you clip is embedded on write and immediately retrievable by every agent you run — Cursor, Claude, or anything else connected to your account.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## What it does

- **Save a page** — popup button, `Cmd/Ctrl+Shift+S`, or right-click → *Save page to SenseLab memory*
- **Save a selection** — highlight text, right-click → *Save selection to SenseLab memory*
- **Add a note** — attach an optional note before saving from the popup
- **Save to a Room** (Pro, Teams, Enterprise) — share clips with your team by picking a room as the destination
- **Usage meter** — see how many memories you have left this month, right in the popup

Clips are written as `experience` memories under the `web-clipper` agent, organized per site (`web/<domain>`). Re-saving the same URL updates the existing memory instead of duplicating it.

## Install

The extension lives in [`packages/browser-extension`](https://github.com/raia-live/amfs/tree/main/packages/browser-extension). Until it's published to the Chrome Web Store, load it unpacked:

```bash
cd packages/browser-extension
npm install
npm run build
```

Then in Chrome/Edge: `chrome://extensions` → enable **Developer mode** → **Load unpacked** → select `packages/browser-extension/.output/chrome-mv3`.

For Firefox: `npm run build:firefox` and load `.output/firefox-mv2` via `about:debugging`.

## Connect your account

Two options:

1. **Connect via dashboard** (recommended): click *Connect to SenseLab* in the popup. The dashboard mints an API key labeled "Browser extension" and hands it to the extension automatically.
2. **Paste an API key**: create a key in the dashboard (Settings → API Keys), then open the extension's settings page and paste it.

Self-hosted deployments: build with `WXT_API_URL` pointing at your server:

```bash
WXT_API_URL=https://amfs.mycompany.com npm run build
```

## Privacy

- Content is read **only when you explicitly save** — there is no background collection while you browse.
- What gets sent per save: the page URL, title, readable article text (or your selection), and your optional note.
- Saving is disabled by default on banking, health, webmail, and password-manager sites. You can extend the blocklist and toggle "never save on this site" per domain in the extension settings.

## Free tier and quotas

The Free plan includes 1,000 ops/month; a save costs 2 ops, so roughly **500 memories per month**. The popup shows your usage, and when you hit the limit the extension links you to the upgrade page — your existing memories keep working either way. See [Hosted billing & metering]({{ site.baseurl }}/guides/saas-billing-metering/) for the ops model.

## How clips are stored

Each save is one memory entry:

| Field | Value |
|:------|:------|
| `entity_path` | `web/<domain>` (e.g. `web/news.ycombinator.com`), or the room's entity when saving to a room |
| `key` | `<title-slug>-<url-hash>` — stable per URL, so re-saves bump the version |
| `value` | `{ url, title, text, byline?, excerpt?, selection?, note?, saved_at }` |
| `memory_type` | `experience` |
| `agent_id` | `web-clipper` |

Ask any connected agent things like *"what did I save about vector databases last week?"* — retrieval works by meaning, not exact keywords.

## Development

```bash
cd packages/browser-extension
npm install
npm run dev        # hot-reloading dev build (Chromium)
npm run compile    # typecheck
npm run zip        # production zip for store submission
```

Build-time configuration:

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `WXT_API_URL` | `https://amfs-login.sense-lab.ai` | AMFS HTTP API base URL |
| `WXT_DASHBOARD_URL` | `https://amfs.sense-lab.ai` | Dashboard origin for the connect flow |
| `WXT_MIXPANEL_TOKEN` | *(unset — analytics off)* | Mixpanel project token |
