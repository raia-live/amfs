# Chrome Web Store submission

Build the store zip:

```bash
npm run zip          # -> .output/senselab-browser-extension-<version>-chrome.zip
WXT_MIXPANEL_TOKEN=<prod-token> npm run zip   # with analytics enabled
```

Firefox (AMO): `npm run zip:firefox`.

After the store assigns the stable extension ID, set
`NEXT_PUBLIC_EXTENSION_IDS=<store-id>[,<dev-id>]` on the dashboard deployment
so `/connect/extension` accepts it.

## Listing copy

**Name:** SenseLab — Save to Memory

**Summary (132 chars max):**
Save any page or highlight into your agent memory. Cursor, Claude, and every
agent you run will remember what you read.

**Description:**

Give your AI agents memory of the web.

SenseLab — Save to Memory clips web pages and highlights into your SenseLab
agent memory. Everything you save becomes instantly retrievable — by meaning,
not just keywords — from every agent connected to your account: Cursor,
Claude, or anything speaking MCP.

- Save a page with one click, Cmd/Ctrl+Shift+S, or right-click
- Save highlighted text with a note
- Ask your agent later: "what did I save about vector databases?"
- Pro teams: save clips straight into a shared team Room
- Built-in usage meter; free plan includes ~500 memories/month

Privacy first: the extension reads page content ONLY when you explicitly
save. Saving is disabled by default on banking, health, webmail, and
password-manager sites, and you can turn it off per-site.

Requires a free SenseLab account (senselab.ai).

**Category:** Productivity / Tools
**Language:** English

## Privacy disclosures (CWS "privacy practices" tab)

- **Single purpose:** save user-selected web content into the user's own
  SenseLab memory account.
- **Data collected:** website content (page text/title/URL, ONLY on explicit
  user action), authentication information (API key stored locally). Optional
  anonymous usage analytics (user can opt out in Options; disabled entirely
  when built without a Mixpanel token).
- **Limited use:** data is transmitted solely to the user's own SenseLab
  account (`amfs-login.sense-lab.ai`); not sold, not used for advertising,
  not shared with third parties.
- **Privacy policy URL:** https://amfs.sense-lab.ai/privacy

## Permission justifications

| Permission | Justification |
|------------|---------------|
| `activeTab` | Read the current tab's content only when the user clicks save / uses the shortcut / context menu |
| `scripting` | Inject the Readability extractor into the active tab on that explicit action |
| `storage` | Store the API key, settings, and usage counters locally |
| `contextMenus` | "Save page/selection to SenseLab memory" right-click items |
| Host `amfs-login.sense-lab.ai` | Send saved clips to the SenseLab API |

No remote code execution; all code is bundled in the package.

## Assets needed before submission

- 128x128 store icon (current `public/icon/128.png` is a placeholder — get a
  final brand icon from design)
- 1280x800 screenshots: popup save, first-save "ask your agent" moment,
  room picker, options page
- Optional 440x280 small promo tile
