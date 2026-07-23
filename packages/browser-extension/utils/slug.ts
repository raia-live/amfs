/** FNV-1a 32-bit hash, hex-encoded. Stable key component per URL. */
export function urlHash(input: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function slugify(text: string, maxLen = 48): string {
  const slug = text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, maxLen)
    .replace(/-+$/g, "");
  return slug || "untitled";
}

/**
 * Canonicalize a URL before hashing so re-saves of the same article are
 * idempotent: strips hash fragments and common tracking params.
 */
export function canonicalUrl(rawUrl: string): string {
  try {
    const u = new URL(rawUrl);
    u.hash = "";
    const tracking = /^(utm_|fbclid|gclid|ref_src|mc_cid|mc_eid)/;
    for (const key of [...u.searchParams.keys()]) {
      if (tracking.test(key)) u.searchParams.delete(key);
    }
    return u.toString();
  } catch {
    return rawUrl;
  }
}

/**
 * Entry key: `<title-slug>-<url-hash>`. No date component — re-saving the
 * same URL hits the same key and bumps the version via copy-on-write.
 */
export function clipKey(title: string, url: string): string {
  return `${slugify(title)}-${urlHash(canonicalUrl(url))}`;
}

/** Entity path per site: web/<registrable-ish domain>. */
export function entityForUrl(rawUrl: string): string {
  try {
    const host = new URL(rawUrl).hostname.replace(/^www\./, "");
    return `web/${host}`;
  } catch {
    return "web/clips";
  }
}
