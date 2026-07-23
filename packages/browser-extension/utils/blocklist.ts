/**
 * Domains where saving is disabled by default: banking, health, webmail,
 * password managers. Users can extend (not shrink) this list in Options;
 * per-site "never save here" toggles are stored separately in settings.
 */
export const DEFAULT_BLOCKLIST: string[] = [
  // Banking / finance
  "*.chase.com",
  "*.bankofamerica.com",
  "*.wellsfargo.com",
  "*.citi.com",
  "*.capitalone.com",
  "*.usbank.com",
  "*.schwab.com",
  "*.fidelity.com",
  "*.vanguard.com",
  "*.paypal.com",
  "*.venmo.com",
  "*.wise.com",
  "*.stripe.com",
  // Health
  "*.mychart.com",
  "*.myuclahealth.org",
  "*.kaiserpermanente.org",
  "*.cvs.com",
  "*.walgreens.com",
  // Webmail
  "mail.google.com",
  "outlook.live.com",
  "outlook.office.com",
  "mail.yahoo.com",
  "mail.proton.me",
  // Password managers / auth
  "*.1password.com",
  "*.lastpass.com",
  "*.bitwarden.com",
  "accounts.google.com",
  "login.microsoftonline.com",
];

function matchesPattern(hostname: string, pattern: string): boolean {
  const p = pattern.toLowerCase().trim();
  const h = hostname.toLowerCase();
  if (!p) return false;
  if (p.startsWith("*.")) {
    const base = p.slice(2);
    return h === base || h.endsWith(`.${base}`);
  }
  return h === p;
}

export function isBlockedHost(hostname: string, customBlocklist: string[]): boolean {
  return [...DEFAULT_BLOCKLIST, ...customBlocklist].some((p) =>
    matchesPattern(hostname, p),
  );
}

/** Pages we can't inject into at all. */
export function isUnsupportedUrl(url: string | undefined): boolean {
  if (!url) return true;
  return !/^https?:\/\//.test(url);
}
