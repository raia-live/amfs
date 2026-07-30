import { CLIPPER_AGENT_ID } from "./config";
import { slugify, urlHash } from "./slug";

export interface IdentitySeed {
  /** Account email from the connect handshake, when we have one. */
  email?: string | null;
  /** User id from /auth/whoami — the fallback when a key was pasted manually. */
  userId?: string | null;
}

/**
 * An agent identity belongs to exactly one user per account: the API rejects a
 * write with 409 when the identity it carries is owned by somebody else. A
 * single shared `web-clipper` therefore locks every member of a team account
 * except whoever saved first.
 *
 * These are tried in order and the one that works is persisted, so a solo
 * account keeps the plain `web-clipper` (and the clips already filed under it)
 * and only colliding users take a suffix. Each candidate is derived rather
 * than random: a reinstall lands on the same identity instead of scattering
 * someone's clips over several agent pages, which AMFS cannot merge back.
 */
export function agentIdCandidates(seed: IdentitySeed): string[] {
  const label = seedLabel(seed);
  const fingerprint = (seed.email ?? seed.userId ?? "").trim().toLowerCase();
  const candidates = [CLIPPER_AGENT_ID];
  if (label) candidates.push(`${CLIPPER_AGENT_ID}-${label}`);
  if (fingerprint) {
    // Two members can share a local part (bruno@a.com, bruno@b.com), so the
    // readable candidate alone isn't guaranteed to be free.
    const suffix = label ? `${label}-${urlHash(fingerprint)}` : urlHash(fingerprint);
    candidates.push(`${CLIPPER_AGENT_ID}-${suffix}`);
  }
  return [...new Set(candidates)];
}

/** First candidate not already rejected by the API, or null when exhausted. */
export function nextAgentId(seed: IdentitySeed, tried: Iterable<string>): string | null {
  const used = new Set(tried);
  return agentIdCandidates(seed).find((id) => !used.has(id)) ?? null;
}

function seedLabel({ email, userId }: IdentitySeed): string | null {
  const localPart = email?.trim().toLowerCase().split("@")[0];
  if (localPart) {
    const slug = slugify(localPart, 24);
    if (slug !== "untitled") return slug;
  }
  const id = userId?.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
  return id ? id.slice(0, 8) : null;
}
