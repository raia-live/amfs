#!/usr/bin/env node
/**
 * Authenticated end-to-end smoke test for the clipper save path.
 * Replicates exactly what the background worker sends, then verifies
 * read-back, version bump on re-save, retrieval, quota headers, and the
 * rooms tier endpoint.
 *
 * Usage: AMFS_API_KEY=amfs_… node scripts/e2e-save.mjs [base-url]
 */
const API_URL = process.argv[2] ?? process.env.AMFS_HTTP_URL ?? "https://amfs-login.sense-lab.ai";
const API_KEY = process.env.AMFS_API_KEY;
if (!API_KEY) {
  console.error("Set AMFS_API_KEY");
  process.exit(1);
}

const headers = { "Content-Type": "application/json", "X-AMFS-API-Key": API_KEY };
let failures = 0;

function check(name, cond, extra = "") {
  const mark = cond ? "PASS" : "FAIL";
  if (!cond) failures++;
  console.log(`${mark}  ${name}${extra ? ` — ${extra}` : ""}`);
}

// Same key scheme as utils/slug.ts (FNV-1a).
function urlHash(input) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

const testUrl = "https://example.com/e2e-clipper-test";
const entityPath = "web/example.com";
const key = `e2e-clipper-test-${urlHash(testUrl)}`;
const clip = {
  url: testUrl,
  title: "E2E clipper test page",
  text: "This is an end-to-end test clip written by scripts/e2e-save.mjs to verify the extension save path.",
  saved_at: new Date().toISOString(),
};

const writeBody = {
  entity_path: entityPath,
  key,
  value: clip,
  confidence: 1.0,
  memory_type: "experience",
  agent_id: "web-clipper",
  shared: true,
};

// 1. Write (the extension's save)
const w1 = await fetch(`${API_URL}/api/v1/entries`, {
  method: "POST",
  headers,
  body: JSON.stringify(writeBody),
});
const w1Body = await w1.json();
check("write clip", w1.ok, `status ${w1.status}, version ${w1Body.version}`);

// 2. Quota headers on billable API-key traffic
const remaining = w1.headers.get("x-amfs-ops-remaining");
const limit = w1.headers.get("x-amfs-ops-limit");
check(
  "usage headers present on write",
  remaining !== null && limit !== null,
  `remaining=${remaining}, limit=${limit}`,
);

// 3. Re-save same URL => same key, version bump (idempotency)
const w2 = await fetch(`${API_URL}/api/v1/entries`, {
  method: "POST",
  headers,
  body: JSON.stringify({ ...writeBody, value: { ...clip, saved_at: new Date().toISOString() } }),
});
const w2Body = await w2.json();
check(
  "re-save bumps version (CoW)",
  w2.ok && Number(w2Body.version) > Number(w1Body.version),
  `v${w1Body.version} -> v${w2Body.version}`,
);

// 4. Read back
const r = await fetch(`${API_URL}/api/v1/entries/${entityPath}/${key}`, { headers });
const rBody = await r.json();
check(
  "read back clip",
  r.ok && rBody.value?.url === testUrl && rBody.provenance?.agent_id === "web-clipper",
  `agent_id=${rBody.provenance?.agent_id}`,
);

// 5. Semantic retrieve (first-save aha path)
const ret = await fetch(`${API_URL}/api/v1/retrieve`, {
  method: "POST",
  headers,
  body: JSON.stringify({ query: "E2E clipper test page", limit: 5 }),
});
const retBody = await ret.json();
const hits = Array.isArray(retBody) ? retBody : (retBody.entries ?? retBody.results ?? []);
check(
  "retrieve finds the clip",
  ret.ok && hits.some((e) => e.key === key),
  `${hits.length} results`,
);

// 6. Rooms tier endpoint (destination picker)
const tier = await fetch(`${API_URL}/api/v1/rooms/account-tier`, { headers });
const tierBody = tier.ok ? await tier.json() : {};
check(
  "rooms account-tier",
  tier.ok && "plan_tier" in tierBody && "rooms_enabled" in tierBody,
  `plan_tier=${tierBody.plan_tier}, rooms_enabled=${tierBody.rooms_enabled}`,
);

// 7. Rooms list shape when unlocked
if (tierBody.rooms_enabled) {
  const rooms = await fetch(`${API_URL}/api/v1/rooms`, { headers });
  const roomsBody = await rooms.json();
  const list = Array.isArray(roomsBody) ? roomsBody : (roomsBody.rooms ?? []);
  check(
    "rooms list has id + entity_path",
    rooms.ok && (list.length === 0 || (list[0].id && list[0].entity_path)),
    `${list.length} rooms`,
  );
}

console.log(failures === 0 ? "\nAll checks passed." : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
