// ─────────────────────────────────────────────────────────────────────
// AMFS Demo — E-commerce Performance Marketing
// ─────────────────────────────────────────────────────────────────────
//
// VIDEO PRESENTER GUIDE:
//   - Open this file in your editor, terminal running beside it
//   - Scroll through the code as you talk
//   - Every "// SAY:" comment tells you what to say at that point
//   - The terminal output shows the live results
//
// REQUIRES:
//   export AMFS_HTTP_URL=https://...  AMFS_API_KEY=...
//
// HOW TO RUN:
//   cd demo && npx tsx run-demo.ts
//
// WHAT THIS SHOWS:
//   Act 1 — Agent generates code, another agent finds it and reuses it
//   Act 2 — An ops-review-agent audits all skills and produces a report
//   Act 3 — Agents coordinate via shared memory (readFrom + causal chains)
//   Act 4 — Knowledge graph surfaces extraction candidates
//
// SAY (OPENING):
//   "I'm going to show you how AMFS solves 3 problems:
//    1. Your agents waste tokens regenerating the same code
//    2. You have no visibility into which skills work or fail
//    3. Agents can't coordinate or trace decisions across each other
//    Let me run through a real scenario."
// ─────────────────────────────────────────────────────────────────────

import { AgentMemory, OutcomeType } from "@senselab-ai/amfs";
import type { MemoryEntry, ScopeInfo } from "@senselab-ai/amfs";

// ── Output helpers (just formatting, skip past these) ────────────────

function log(prefix: string, msg: string) {
  console.log(`  \x1b[90m[${prefix}]\x1b[0m ${msg}`);
}
function header(msg: string) {
  console.log(`\n\x1b[1;36m${"━".repeat(60)}\x1b[0m`);
  console.log(`\x1b[1;36m  ${msg}\x1b[0m`);
  console.log(`\x1b[1;36m${"━".repeat(60)}\x1b[0m`);
}
function subheader(msg: string) {
  console.log(`\n\x1b[1;33m  ── ${msg} ──\x1b[0m`);
}
function ok(msg: string) { console.log(`  \x1b[1;32m✓\x1b[0m ${msg}`); }
function warn(msg: string) { console.log(`  \x1b[1;33m⚠\x1b[0m ${msg}`); }
function bad(msg: string) { console.log(`  \x1b[1;31m✗\x1b[0m ${msg}`); }
function json(data: unknown) { console.log(JSON.stringify(data, null, 2)); }
function pause(label: string) {
  console.log(`\n\x1b[7m  ▶ ${label}  \x1b[0m\n`);
}

// ── AMFS SaaS connection ─────────────────────────────────────────────
// SAY: "This script connects to the AMFS SaaS. Every write, search,
//       and outcome shows up on your dashboard in real time."
const AMFS_URL = process.env.AMFS_HTTP_URL;
const AMFS_KEY = process.env.AMFS_API_KEY;

if (!AMFS_URL || !AMFS_KEY) {
  bad("AMFS_HTTP_URL and AMFS_API_KEY are required.");
  bad("Run: export AMFS_HTTP_URL=https://... AMFS_API_KEY=...");
  process.exit(1);
}

ok(`Connected to AMFS: ${AMFS_URL}`);

// The SDK's AgentMemory provides the same API your agents use in production.
// Under the hood, entries are synced to your AMFS account so everything
// appears on the dashboard — agents, entries, outcomes, decision traces.
const _root = new AgentMemory("_system");
const adapter = _root.adapter;
function agent(id: string) { return new AgentMemory(id, { adapter }); }

const _syncHeaders: Record<string, string> = {
  "Content-Type": "application/json",
  "X-AMFS-API-Key": AMFS_KEY,
};
const _syncedKeys = new Set<string>();

async function syncToServer() {
  const entries = _root.adapter.list() as MemoryEntry[];
  let synced = 0;
  for (const entry of entries) {
    const ek = `${entry.provenance.agentId}/${entry.entityPath}/${entry.key}`;
    if (_syncedKeys.has(ek)) continue;
    try {
      await fetch(`${AMFS_URL}/api/v1/entries`, {
        method: "POST",
        headers: _syncHeaders,
        body: JSON.stringify({
          entity_path: entry.entityPath,
          key: entry.key,
          value: entry.value,
          confidence: entry.confidence,
          pattern_refs: entry.provenance.patternRefs,
          memory_type: "fact",
          shared: true,
          agent_id: entry.provenance.agentId,
        }),
      });
      _syncedKeys.add(ek);
      synced++;
    } catch { /* individual entry sync failure is non-fatal */ }
  }
  if (synced > 0) ok(`Synced ${synced} entries to dashboard`);
}

// =====================================================================
//  SEED DATA
//  SAY: "Let me set up the scenario. We have 4 agents that generate code
//        for a full-funnel marketing dashboard. Each one works independently."
// =====================================================================
header("SEED DATA — Populating the skill ecosystem");

// SAY: "Here are the 4 skill-agents — each one generates code for a
//       different data source: Facebook Ads, Google Ads, Redshift, Dashboard."
const fb = agent("facebook-ads-agent");
const gads = agent("google-ads-agent");
const rs = agent("redshift-agent");
const dash = agent("dashboard-agent");

subheader("facebook-ads-agent: 2 code entries, 5 outcomes");

// SAY: "The facebook-ads-agent generated 200 lines of code to fetch campaign
//       data. It stores the ACTUAL CODE in memory — not metadata, the real code.
//       Notice: patternRefs tags it as 'facebook-ads-api' so other agents can find it."
fb.write("ecomm/facebook-ads", "data-fetch-code", {
  code: `import { FacebookAdsApi, AdAccount } from 'facebook-nodejs-business-sdk';

export async function fetchFacebookCampaignData(
  token: string,
  accountId: string,
  dateRange: { start: string; end: string }
): Promise<CampaignPerformanceData[]> {
  const api = FacebookAdsApi.init(token);
  const account = new AdAccount(\`act_\${accountId}\`);

  const campaigns = await account.getCampaigns([
    'name', 'status', 'objective', 'daily_budget',
    'lifetime_budget', 'start_time', 'stop_time',
  ]);

  const insights = await account.getInsights([
    'campaign_id', 'campaign_name', 'impressions',
    'clicks', 'spend', 'cpc', 'cpm', 'ctr',
    'actions', 'cost_per_action_type',
    'conversions', 'conversion_values',
  ], {
    time_range: { since: dateRange.start, until: dateRange.end },
    level: 'campaign',
    time_increment: 1,
  });

  // ... 170+ more lines: data normalization, error handling,
  // retry logic, rate limiting, pagination, type mapping ...

  return insights.map(row => ({
    campaignId: row.campaign_id,
    campaignName: row.campaign_name,
    date: row.date_start,
    impressions: parseInt(row.impressions),
    clicks: parseInt(row.clicks),
    spend: parseFloat(row.spend),
    cpc: parseFloat(row.cpc),
    ctr: parseFloat(row.ctr),
    conversions: row.actions?.find(a => a.action_type === 'purchase')?.value ?? 0,
    revenue: row.conversion_values?.find(a => a.action_type === 'purchase')?.value ?? 0,
  }));
}`,
  description: "Fetches campaign performance data from Facebook Ads API with full pagination, retry logic, and rate limiting",
  params: { token: "string", accountId: "string", dateRange: "{ start: string, end: string }" },
  returnType: "CampaignPerformanceData[]",
  language: "typescript",
  lineCount: 200,
  tokensUsed: 4200,
}, { confidence: 0.92, patternRefs: ["facebook-ads-api", "campaign-data", "ad-platform-fetch"] });
log("seed", "data-fetch-code: 200 lines, Facebook Ads API");

fb.write("ecomm/facebook-ads", "audience-breakdown-code", {
  code: "// 150 lines: breakdown by age, gender, placement, device\nexport async function getAudienceBreakdown(token, accountId, campaignIds) { ... }",
  description: "Breaks down campaign performance by audience demographics",
  params: { token: "string", accountId: "string", campaignIds: "string[]" },
  returnType: "AudienceBreakdown[]",
  language: "typescript", lineCount: 150, tokensUsed: 3100,
}, { confidence: 0.85, patternRefs: ["facebook-ads-api", "audience-analytics"] });
log("seed", "audience-breakdown-code: 150 lines");

// SAY: "Every time an agent runs its code, it commits an outcome —
//       SUCCESS or FAILURE. This is how confidence scores evolve.
//       4 successes, 1 failure — so confidence is high but not perfect."
fb.commitOutcome("fb-run-001", OutcomeType.SUCCESS);
fb.commitOutcome("fb-run-002", OutcomeType.SUCCESS);
fb.commitOutcome("fb-run-003", OutcomeType.SUCCESS);
fb.commitOutcome("fb-run-004", OutcomeType.FAILURE);
fb.commitOutcome("fb-run-005", OutcomeType.SUCCESS);
log("seed", "Outcomes: 4 success, 1 failure");

// SAY: "Same pattern for Google Ads — different agent, same structure.
//       Notice it also uses 'campaign-data' and 'ad-platform-fetch' as
//       pattern tags. That overlap is what AMFS will detect later."
subheader("google-ads-agent: 2 code entries, 3 outcomes");

gads.write("ecomm/google-ads", "data-fetch-code", {
  code: `import { GoogleAdsApi } from 'google-ads-api';

export async function fetchGoogleAdsCampaignData(
  clientId: string, clientSecret: string,
  refreshToken: string, customerId: string,
  dateRange: { start: string; end: string }
): Promise<CampaignPerformanceData[]> {
  const client = new GoogleAdsApi({ client_id: clientId, client_secret: clientSecret, developer_token: '...' });
  const customer = client.Customer({ customer_id: customerId, refresh_token: refreshToken });

  const campaigns = await customer.report({
    entity: 'campaign',
    attributes: ['campaign.id', 'campaign.name', 'campaign.status'],
    metrics: ['metrics.impressions', 'metrics.clicks', 'metrics.cost_micros',
              'metrics.conversions', 'metrics.conversions_value'],
    segments: ['segments.date'],
    from_date: dateRange.start, to_date: dateRange.end,
  });

  // ... 140+ more lines: credential management, pagination, error handling,
  // micros-to-dollars conversion, campaign type filtering ...

  return campaigns.map(row => ({
    campaignId: row.campaign.id,
    campaignName: row.campaign.name,
    impressions: row.metrics.impressions,
    clicks: row.metrics.clicks,
    spend: row.metrics.cost_micros / 1_000_000,
    conversions: row.metrics.conversions,
    revenue: row.metrics.conversions_value,
  }));
}`,
  description: "Fetches campaign performance data from Google Ads API with credential management and micros conversion",
  params: { clientId: "string", clientSecret: "string", refreshToken: "string", customerId: "string", dateRange: "{ start: string, end: string }" },
  returnType: "CampaignPerformanceData[]",
  language: "typescript", lineCount: 180, tokensUsed: 3800,
}, { confidence: 0.88, patternRefs: ["google-ads-api", "campaign-data", "ad-platform-fetch"] });
log("seed", "data-fetch-code: 180 lines, Google Ads API");

gads.write("ecomm/google-ads", "keyword-performance-code", {
  code: "// 120 lines: keyword-level performance with quality scores\nexport async function getKeywordPerformance(customerId, campaignIds) { ... }",
  description: "Fetches keyword-level performance metrics with quality scores",
  params: { customerId: "string", campaignIds: "string[]" },
  returnType: "KeywordPerformance[]",
  language: "typescript", lineCount: 120, tokensUsed: 2500,
}, { confidence: 0.80, patternRefs: ["google-ads-api", "keyword-analytics"] });
log("seed", "keyword-performance-code: 120 lines");

gads.commitOutcome("gads-run-001", OutcomeType.SUCCESS);
gads.commitOutcome("gads-run-002", OutcomeType.SUCCESS);
gads.commitOutcome("gads-run-003", OutcomeType.SUCCESS);
log("seed", "Outcomes: 3 success");

// SAY: "Redshift agent generates SQL-based code — different language,
//       same idea. Note it has a CRITICAL_FAILURE in its history.
//       That's something our review agent will flag later."
subheader("redshift-agent: 2 code entries, 4 outcomes");

rs.write("ecomm/redshift", "landing-page-query-code", {
  code: `import { Client } from 'pg';

export async function fetchLandingPageMetrics(
  connectionString: string,
  dateRange: { start: string; end: string },
  filters?: { market?: string; userType?: string }
): Promise<LandingPageMetrics[]> {
  const client = new Client({ connectionString });
  await client.connect();

  const query = \`
    SELECT lp.page_url, lp.variant_id,
      COUNT(DISTINCT s.session_id) as sessions,
      COUNT(DISTINCT CASE WHEN s.converted THEN s.session_id END) as conversions,
      AVG(s.time_on_page_seconds) as avg_time_on_page,
      SUM(CASE WHEN o.order_id IS NOT NULL THEN o.total_amount ELSE 0 END) as revenue
    FROM landing_pages lp
    JOIN sessions s ON s.landing_page_id = lp.id
    LEFT JOIN orders o ON o.session_id = s.session_id
    WHERE s.created_at BETWEEN $1 AND $2
    GROUP BY lp.page_url, lp.variant_id
    ORDER BY sessions DESC\`;

  // ... 80+ more lines: connection pooling, parameterization,
  // retry on connection timeout, result transformation ...
}`,
  description: "Queries Redshift for landing page performance metrics with session/conversion/revenue data",
  params: { connectionString: "string", dateRange: "{ start: string, end: string }", filters: "{ market?: string, userType?: string }" },
  returnType: "LandingPageMetrics[]",
  language: "typescript", lineCount: 130, tokensUsed: 2800,
}, { confidence: 0.90, patternRefs: ["redshift-query", "landing-page-data", "conversion-analytics"] });
log("seed", "landing-page-query-code: 130 lines");

rs.write("ecomm/redshift", "user-cohort-query-code", {
  code: "// 160 lines: cohort analysis for new user growth by market\nexport async function getUserCohortAnalysis(connectionString, dateRange, market) { ... }",
  description: "Runs cohort analysis on Redshift for new user growth segmented by market",
  params: { connectionString: "string", dateRange: "{ start: string, end: string }", market: "string" },
  returnType: "CohortAnalysis[]",
  language: "typescript", lineCount: 160, tokensUsed: 3400,
}, { confidence: 0.75, patternRefs: ["redshift-query", "cohort-analysis", "user-growth"] });
log("seed", "user-cohort-query-code: 160 lines");

rs.commitOutcome("rs-run-001", OutcomeType.SUCCESS);
rs.commitOutcome("rs-run-002", OutcomeType.FAILURE);
rs.commitOutcome("rs-run-003", OutcomeType.CRITICAL_FAILURE);
rs.commitOutcome("rs-run-004", OutcomeType.SUCCESS);
log("seed", "Outcomes: 2 success, 1 failure, 1 critical_failure");

subheader("dashboard-agent: 2 code entries (1 REDUNDANT), 3 outcomes");

dash.write("ecomm/dashboard", "full-funnel-assembly-code", {
  code: "// 90 lines: orchestrates data from all sources\nexport async function assembleFunnelDashboard(fbData, gadsData, lpData) { ... }",
  description: "Assembles full-funnel dashboard by combining Facebook Ads, Google Ads, and landing page data",
  params: { fbData: "CampaignPerformanceData[]", gadsData: "CampaignPerformanceData[]", lpData: "LandingPageMetrics[]" },
  returnType: "FunnelDashboard",
  language: "typescript", lineCount: 90, tokensUsed: 1900,
}, { confidence: 0.88, patternRefs: ["dashboard-assembly", "full-funnel", "campaign-data"] });
log("seed", "full-funnel-assembly-code: 90 lines");

// SAY: "HERE is the problem. The dashboard-agent ALSO generated its own
//       Facebook fetch — 195 lines, almost identical to what facebook-ads-agent
//       already had. This is the waste. This happens with all 200 skills."
dash.write("ecomm/facebook-ads", "data-fetch-code-v2", {
  code: "// 195 lines: REDUNDANT — almost identical to facebook-ads-agent's version\nexport async function getFBCampaignData(token, accountId, dateRange) { ... }",
  description: "Fetches Facebook Ads campaign data (REDUNDANT — generated independently by dashboard-agent)",
  params: { token: "string", accountId: "string", dateRange: "{ start: string, end: string }" },
  returnType: "CampaignPerformanceData[]",
  language: "typescript", lineCount: 195, tokensUsed: 4100,
}, { confidence: 0.78, patternRefs: ["facebook-ads-api", "campaign-data", "ad-platform-fetch"] });
warn("data-fetch-code-v2: 195 lines — REDUNDANT Facebook fetch by dashboard-agent!");

dash.commitOutcome("dash-run-001", OutcomeType.SUCCESS);
dash.commitOutcome("dash-run-002", OutcomeType.SUCCESS);
dash.commitOutcome("dash-run-003", OutcomeType.MINOR_FAILURE);
log("seed", "Outcomes: 2 success, 1 minor_failure");

ok("Seed complete — 8 code entries across 4 agents, 15 outcomes");

await syncToServer();

// SAY: "OK — that's the setup. 4 agents, 8 blocks of generated code,
//       15 outcome records. One of them is redundant — the dashboard-agent
//       re-generated what facebook-ads-agent already had. Now watch what
//       happens when a NEW agent enters the picture."

// =====================================================================
//  ACT 1 — "Your agents start remembering"
//  SAY: "A completely new agent — the reporting-agent — needs Facebook
//        campaign data for a weekly report. WITHOUT AMFS, it would generate
//        200 lines from scratch. WITH AMFS, it checks memory first."
// =====================================================================
header("ACT 1 — Your agents start remembering");
pause("Agent 1 generates code, Agent 2 finds it and reuses it");

// SAY: "Here's the reporting-agent. It needs Facebook campaign data.
//       Step 1: check if someone already wrote this code."
subheader("1a. A new agent needs Facebook campaign data");

const reportAgent = agent("reporting-agent");
log("act1", "reporting-agent needs Facebook campaign data for a weekly report");
log("act1", "Before generating 200 lines of code, it checks AMFS...");

// SAY: "One search call. Pattern tag: 'facebook-ads-api'. Minimum
//       confidence: 0.7. It doesn't know WHO wrote it — it just asks
//       WHAT it needs. This is the check-before-generate pattern."
subheader("1b. Search by pattern tag — does this code already exist?");

// SAY: "This is the KEY line. One search call with the pattern tag.
//       It doesn't need to know WHO wrote it — just WHAT it needs."
const cached = reportAgent.search({
  patternRef: "facebook-ads-api",   // search by tag — finds any agent's code
  minConfidence: 0.7,               // only high-confidence, validated code
  sortBy: "confidence",             // best match first
});

// SAY: "Look at the terminal output — it found 3 entries. The best one
//       is 200 lines of validated code with confidence 0.92. That's
//       4200 tokens we didn't need to spend regenerating."
if (cached.length > 0) {
  ok(`Found ${cached.length} entries matching pattern "facebook-ads-api":`);
  for (const entry of cached) {
    const val = entry.value as Record<string, unknown>;
    log("act1", `  ${entry.entityPath}/${entry.key} by ${entry.provenance.agentId} ` +
      `(confidence: ${entry.confidence}, ${val.lineCount} lines, ${val.tokensUsed} tokens)`);
  }

  const best = cached[0];
  const bestVal = best.value as Record<string, unknown>;
  console.log();
  ok(`Best match: ${best.entityPath}/${best.key}`);
  ok(`Written by: ${best.provenance.agentId}`);
  ok(`Confidence: ${best.confidence}`);
  ok(`Lines of code: ${bestVal.lineCount}`);
  ok(`Tokens saved: ~${bestVal.tokensUsed} (didn't need to regenerate!)`);
  console.log();

  log("act1", "The code is in entry.value.code — here's the first 10 lines:");
  const code = String(bestVal.code ?? "").split("\n").slice(0, 10);
  for (const line of code) {
    console.log(`  \x1b[90m│\x1b[0m ${line}`);
  }
  console.log(`  \x1b[90m│ ... (${bestVal.lineCount} lines total)\x1b[0m`);
} else {
  bad("No cached code found — would need to generate from scratch");
}

// SAY: "You can also search by natural language — useful when the agent
//       doesn't know the exact pattern tag."
subheader("1c. Alternative: search by natural language query");

const queryResults = reportAgent.search({
  query: "facebook campaign performance",
  minConfidence: 0.7,
});
ok(`Full-text search found ${queryResults.length} entries matching "facebook campaign performance"`);

// SAY: "Or discover the agent first, then read directly. This is tracked
//       as a cross-agent read — shows up on both agents' timelines."
subheader("1d. Alternative: discover the agent, then read directly");

const allEntries = reportAgent.list();
const fbAgentEntries = allEntries.filter(
  e => e.provenance.agentId === "facebook-ads-agent" && e.shared
);
log("act1", `facebook-ads-agent has ${fbAgentEntries.length} shared entries`);

const directRead = reportAgent.readFrom("facebook-ads-agent", "ecomm/facebook-ads", "data-fetch-code");
if (directRead) {
  ok(`readFrom() returned the code directly — cross-agent read tracked`);
  ok(`Written at: ${directRead.provenance.writtenAt}`);
}

// SAY: "The reporting-agent used the code and it worked. It commits success.
//       This reinforces the confidence score on that entry."
subheader("1e. Reporting-agent uses the code and commits success");

reportAgent.commitOutcome("report-run-001", OutcomeType.SUCCESS);
ok("reporting-agent committed success — confidence on the entry stabilizes");

// SAY: "But what happens when the code breaks? Let's say Facebook changes
//       their API. The agent commits a failure, regenerates, and writes a
//       new version — with LOWER confidence because it hasn't been validated yet.
//       history() shows all versions. The system self-heals."
subheader("1f. What if the code breaks? Confidence evolves");

log("act1", "Simulating: Facebook API changed, code throws an error...");
const breakAgent = agent("facebook-ads-agent");
breakAgent.commitOutcome("fb-run-006", OutcomeType.FAILURE);
warn("Failure committed — entry gets flagged for review");

log("act1", "Agent regenerates with updated API and writes new version:");
breakAgent.write("ecomm/facebook-ads", "data-fetch-code", {
  code: "// v2: updated for Facebook Marketing API v19.0 changes\nexport async function fetchFacebookCampaignData(...) { ... }",
  description: "Fetches campaign performance data (v2 — updated for API v19.0)",
  params: { token: "string", accountId: "string", dateRange: "{ start: string, end: string }" },
  returnType: "CampaignPerformanceData[]",
  language: "typescript", lineCount: 210, tokensUsed: 4400,
}, { confidence: 0.7, patternRefs: ["facebook-ads-api", "campaign-data", "ad-platform-fetch"] });
ok("New version written with confidence 0.7 (not yet validated by multiple runs)");

const versions = breakAgent.history("ecomm/facebook-ads", "data-fetch-code");
log("act1", `history() shows ${versions.length} versions of data-fetch-code`);

// =====================================================================
//  ACT 2 — "An agent audits your other agents"
//  SAY: "You said you have no visibility into which skills are working
//        or failing. Here's how you get it — not with a dashboard,
//        but with another AGENT that reads AMFS and produces a report."
// =====================================================================
header("ACT 2 — An agent audits your other agents");
pause("ops-review-agent queries AMFS and produces a health report");

// SAY: "This is an ops-review-agent — think of it as your QA analyst,
//       but it's an agent that runs automatically."
const reviewer = agent("ops-review-agent");

// SAY: "First it calls list() — gives it everything in AMFS.
//       Then it groups by agent to see who's doing what."
subheader("2a. Inventory all agents and their activity");

const allEntriesForReview = reviewer.list();
const agentMap = new Map<string, { entries: number; entities: Set<string>; patterns: Set<string> }>();

for (const entry of allEntriesForReview) {
  const aid = entry.provenance.agentId;
  if (aid === "_system" || aid === "_bootstrap") continue;
  if (!agentMap.has(aid)) agentMap.set(aid, { entries: 0, entities: new Set(), patterns: new Set() });
  const info = agentMap.get(aid)!;
  info.entries++;
  info.entities.add(entry.entityPath);
  for (const ref of entry.provenance.patternRefs) info.patterns.add(ref);
}

log("act2", `Found ${agentMap.size} agents:`);
for (const [aid, info] of agentMap) {
  log("act2", `  ${aid}: ${info.entries} entries, ${info.entities.size} entities, patterns: [${[...info.patterns].join(", ")}]`);
}

// SAY: "Now the review agent groups entries by patternRefs. If the same
//       pattern tag shows up across multiple agents, that's redundancy —
//       multiple agents generating the same code independently."
subheader("2b. Detect redundant patterns");

const patternUsage = new Map<string, { count: number; agents: Set<string>; entities: Set<string>; totalTokens: number }>();

for (const entry of allEntriesForReview) {
  const val = entry.value as Record<string, unknown>;
  const tokens = typeof val?.tokensUsed === "number" ? val.tokensUsed : 0;
  for (const ref of entry.provenance.patternRefs) {
    if (!patternUsage.has(ref)) patternUsage.set(ref, { count: 0, agents: new Set(), entities: new Set(), totalTokens: 0 });
    const p = patternUsage.get(ref)!;
    p.count++;
    p.agents.add(entry.provenance.agentId);
    p.entities.add(entry.entityPath);
    p.totalTokens += tokens;
  }
}

log("act2", "\nPattern usage (sorted by frequency):");
const sortedPatterns = [...patternUsage.entries()].sort((a, b) => b[1].count - a[1].count);
for (const [ref, info] of sortedPatterns) {
  const multiAgent = info.agents.size > 1 ? " \x1b[1;33m← MULTI-AGENT\x1b[0m" : "";
  log("act2", `  "${ref}": ${info.count} entries, ${info.agents.size} agents [${[...info.agents].join(", ")}], ~${info.totalTokens} tokens${multiAgent}`);
}

// SAY: "Look at the MULTI-AGENT tags in the output. 'campaign-data' is
//       used by 3 different agents. 'ad-platform-fetch' — 3 agents too.
//       That means 3 agents are independently generating the SAME code."
const redundant = sortedPatterns.filter(([, info]) => info.agents.size > 1 && info.count > 2);
if (redundant.length > 0) {
  console.log();
  warn(`Redundant patterns detected (multiple agents generating the same thing):`);
  for (const [ref, info] of redundant) {
    warn(`  "${ref}": ${info.agents.size} agents each generating independently → ~${info.totalTokens} tokens wasted`);
  }
}
// SAY: "That's ~38,000 tokens wasted on redundant generation.
//       And this is just 4 agents — imagine 200."

// SAY: "This is the 'which skill is generating problems' answer.
//       Low confidence = it's been failing. Outcomes are tracked automatically."
subheader("2c. Find failing skills");

type OutcomeInfo = { agent: string; successes: number; failures: number; criticals: number; minors: number };
const outcomesByAgent = new Map<string, OutcomeInfo>();

for (const entry of allEntriesForReview) {
  if (entry.outcomeCount === 0) continue;
  const aid = entry.provenance.agentId;
  if (!outcomesByAgent.has(aid)) outcomesByAgent.set(aid, { agent: aid, successes: 0, failures: 0, criticals: 0, minors: 0 });
}

// Use the stats to summarize
const stats = reviewer.stats();
log("act2", `System-wide: ${stats.totalEntries} entries, ${stats.totalAgents} agents, avg confidence: ${stats.confidenceAvg.toFixed(2)}`);

// SAY: "Low confidence = the code has been failing or hasn't been
//       validated enough. These are the entries that need attention.
//       This is your 'which skill is generating problems' answer."
subheader("2d. Check entry quality (low-confidence entries)");

const lowConfidence = allEntriesForReview
  .filter(e => e.confidence < 0.8 && !e.entityPath.startsWith("_system"))
  .sort((a, b) => a.confidence - b.confidence);

if (lowConfidence.length > 0) {
  warn(`${lowConfidence.length} entries with confidence < 0.8:`);
  for (const entry of lowConfidence) {
    const val = entry.value as Record<string, unknown>;
    warn(`  ${entry.entityPath}/${entry.key} by ${entry.provenance.agentId} — confidence: ${entry.confidence} (${val.description ?? ""})`);
  }
}
// SAY: "data-fetch-code at 0.7 — that's the one that just failed and got
//       regenerated. user-cohort-query at 0.75 — unstable Redshift query.
//       data-fetch-code-v2 at 0.78 — that's the redundant copy. ALL flagged."

// SAY: "The review agent writes its report BACK to AMFS. Now any agent
//       — or any dashboard — can read the structured report. This is your
//       analytics: not a human running random queries, but an agent
//       producing a health report automatically."
subheader("2e. Write the review report back to AMFS");

const report = {
  reviewDate: new Date().toISOString(),
  agentCount: agentMap.size,
  totalEntries: stats.totalEntries,
  avgConfidence: Number(stats.confidenceAvg.toFixed(2)),
  agentSummary: Object.fromEntries(
    [...agentMap].map(([aid, info]) => [aid, {
      entries: info.entries,
      entities: [...info.entities],
      patterns: [...info.patterns],
    }])
  ),
  redundantPatterns: redundant.map(([ref, info]) => ({
    patternRef: ref,
    agents: [...info.agents],
    entryCount: info.count,
    estimatedTokensWasted: info.totalTokens,
    recommendation: "Extract into shared function — multiple agents generating the same code",
  })),
  lowConfidenceEntries: lowConfidence.map(e => ({
    path: `${e.entityPath}/${e.key}`,
    agent: e.provenance.agentId,
    confidence: e.confidence,
  })),
  overallHealth: lowConfidence.length === 0 && redundant.length === 0 ? "healthy" : "needs-attention",
};

reviewer.write("ecomm/ops", "review-report-latest", report, {
  confidence: 1.0,
  patternRefs: ["ops-review", "skill-health"],
});
reviewer.commitOutcome("ops-review-2026-04-17", OutcomeType.SUCCESS);

ok("Review report written to ecomm/ops/review-report-latest");
log("act2", "Any agent can now read this report:");
log("act2", '  mem.read("ecomm/ops", "review-report-latest")');

// SAY: "This is the key insight: you don't need a separate dashboard.
//       An agent reads AMFS, analyzes everything, and writes a structured
//       report BACK to AMFS. Other agents — or your dashboard — can read it.
//       Run this on a cron, and you have continuous observability."

// Sync Act 1-2 data (new entries: reporting-agent read, version update, review report)
await syncToServer();

// =====================================================================
//  ACT 3 — "Your agents coordinate" (Agent-to-Agent Communication)
//  SAY: "Act 1 showed reuse. Act 2 showed observability. Now: what if
//        agents could coordinate DIRECTLY — reading each other's memory,
//        writing proposals to shared entities, and building a traceable
//        causal chain of decisions? No chat rooms needed. Agents just
//        talk through memory."
// =====================================================================
header("ACT 3 — Your agents coordinate (Agent-to-Agent)");
pause("Agents communicate through shared memory — proposals, reads, decisions");

// SAY: "All these agents belong to the same account. They already share the
//       same memory pool. So to coordinate, they just write to a shared entity
//       and read from each other. AMFS tracks every cross-agent read — who
//       read what, from whom, and when. That's your audit trail."

// ── 3a. Agent proposes an extraction via shared memory ───────────────

// SAY: "The facebook-ads-agent noticed it's been generating the same 200-line
//       block repeatedly. It writes a proposal to a shared entity that all
//       agents can see. This isn't a chat message — it's structured knowledge
//       with confidence scoring and pattern tags."
subheader("3a. facebook-ads-agent proposes a shared function");

fb.write("ecomm/shared-decisions", "proposal-extract-ad-fetch", {
  proposedBy: "facebook-ads-agent",
  type: "extraction-proposal",
  title: "Extract getAdPlatformData() from repeated code blocks",
  rationale: "I've generated the Facebook data fetch block 12 times this week. " +
    "Same structure every time: token + accountId + dateRange → CampaignPerformanceData[]. " +
    "We should extract this into a shared function.",
  proposedSignature: {
    functionName: "getAdPlatformData",
    params: ["platform: string", "token: string", "accountId: string", "dateRange: DateRange"],
    returnType: "CampaignPerformanceData[]",
  },
  estimatedSavings: "~200 lines × 12 runs = 2,400 lines of redundant generation this week",
  status: "open",
}, {
  confidence: 0.85,
  patternRefs: ["ad-platform-fetch", "facebook-ads-api", "code-extraction", "proposal-extract-ad-fetch"],
  shared: true,
});
ok("facebook-ads-agent wrote extraction proposal to ecomm/shared-decisions");
log("act3", `Key: proposal-extract-ad-fetch`);
log("act3", `Confidence: 0.85 (high but wants other agents to validate)`);
log("act3", `Pattern refs: ad-platform-fetch, facebook-ads-api, code-extraction\n`);

// ── 3b. google-ads-agent reads from facebook-ads-agent ───────────────

// SAY: "Now the google-ads-agent uses readFrom() to explicitly read from the
//       facebook-ads-agent's memory. This is a tracked knowledge transfer —
//       AMFS records that google-ads-agent read this specific entry from
//       facebook-ads-agent. Later, when we call explain(), you'll see this
//       in the causal chain."
subheader("3b. google-ads-agent reads the proposal via readFrom()");

const fbProposal = gads.readFrom(
  "facebook-ads-agent",
  "ecomm/shared-decisions",
  "proposal-extract-ad-fetch",
);
if (fbProposal) {
  ok(`google-ads-agent read proposal from facebook-ads-agent`);
  log("act3", `Title: ${(fbProposal.value as Record<string, unknown>).title}`);
  log("act3", `Confidence: ${fbProposal.confidence}`);
  log("act3", `Cross-agent read tracked in causal chain\n`);
} else {
  bad("Could not read proposal — check shared flag");
}

// SAY: "The google-ads-agent agrees but suggests a better signature that also
//       covers Google Ads. It writes its response to the same shared entity.
//       Notice the patternRef links back to the original proposal —
//       that's how AMFS connects the conversation."
subheader("3c. google-ads-agent agrees and expands the proposal");

gads.write("ecomm/shared-decisions", "response-extract-ad-fetch-gads", {
  respondingTo: "proposal-extract-ad-fetch",
  respondedBy: "google-ads-agent",
  type: "extraction-response",
  action: "accept-with-modification",
  comment: "Same pattern for Google Ads — 180 lines, same structure. " +
    "The proposed getAdPlatformData() should accept a 'platform' param to handle both. " +
    "I can confirm the interface works for Google Ads too.",
  modifiedSignature: {
    functionName: "getAdPlatformData",
    params: [
      "platform: 'facebook' | 'google'",
      "credentials: PlatformCredentials",
      "accountId: string",
      "dateRange: DateRange",
    ],
    returnType: "CampaignPerformanceData[]",
  },
  status: "accepted",
}, {
  confidence: 0.88,
  patternRefs: ["ad-platform-fetch", "google-ads-api", "proposal-extract-ad-fetch"],
  shared: true,
});
ok("google-ads-agent wrote acceptance with modified signature");
log("act3", `Links to original proposal via patternRefs`);
log("act3", `Confidence: 0.88 (validated by second agent)\n`);

// ── 3d. redshift-agent reads from both and adds its perspective ──────

// SAY: "The redshift-agent now reads from BOTH agents. It has a different
//       concern — it needs the extracted function to output data compatible
//       with the Redshift schema. readFrom() creates an explicit, traceable
//       knowledge transfer from each agent."
subheader("3d. redshift-agent reads from both agents and weighs in");

const fbView = rs.readFrom(
  "facebook-ads-agent",
  "ecomm/shared-decisions",
  "proposal-extract-ad-fetch",
);
const gadsView = rs.readFrom(
  "google-ads-agent",
  "ecomm/shared-decisions",
  "response-extract-ad-fetch-gads",
);

if (fbView && gadsView) {
  ok("redshift-agent read from both facebook-ads-agent and google-ads-agent");
  log("act3", "Both cross-agent reads tracked in causal chain");
}

rs.write("ecomm/shared-decisions", "response-extract-ad-fetch-rs", {
  respondingTo: "proposal-extract-ad-fetch",
  respondedBy: "redshift-agent",
  type: "extraction-response",
  action: "accept-with-constraint",
  comment: "Agreed on getAdPlatformData(). Adding a constraint: the output " +
    "MUST include a 'source_platform' field so Redshift ETL can partition by platform. " +
    "I'll update my ingest pipeline to expect the new function signature.",
  requiredOutputFields: ["source_platform", "campaignId", "date", "spend", "conversions"],
  status: "accepted",
}, {
  confidence: 0.90,
  patternRefs: ["ad-platform-fetch", "redshift-ingest", "proposal-extract-ad-fetch"],
  shared: true,
});
ok("redshift-agent wrote acceptance with Redshift compatibility constraint");
log("act3", `3 agents have now coordinated on the extraction\n`);

// ── 3e. Search reveals the full conversation ─────────────────────────

// SAY: "Now any agent — or your ops dashboard — can search for the full
//       conversation. search() with a pattern ref finds every entry in the
//       chain. You can see who proposed, who agreed, who added constraints."
subheader("3e. Search reveals the full coordination thread");

const thread = reviewer.search({ patternRef: "proposal-extract-ad-fetch" });
ok(`Found ${thread.length} entries in the coordination thread:`);
for (const entry of thread) {
  const val = entry.value as Record<string, unknown>;
  const who = (val.proposedBy ?? val.respondedBy ?? "unknown") as string;
  const action = (val.action ?? val.type ?? "—") as string;
  log("act3", `  ${who}: ${action} (confidence: ${entry.confidence})`);
}
console.log();

// ── 3f. Causal chain — explain() shows the full decision flow ────────

// SAY: "This is the killer feature for auditability. explain() shows the
//       complete causal chain: which agent read what from whom, in what order.
//       When something goes wrong 3 weeks from now, you can trace back exactly
//       how the decision was made and who was involved."
subheader("3f. Causal chain — who influenced whom");

const rsTrace = rs.explain("extraction-decision");
ok("redshift-agent's decision trace:");
log("act3", `  Reads: ${(rsTrace.causalEntries as unknown[]).length} entries read before deciding`);
log("act3", `  External contexts: ${(rsTrace.externalContexts as unknown[]).length}`);
for (const entry of rsTrace.causalEntries as Record<string, unknown>[]) {
  const ep = entry.entityPath as string;
  const key = entry.key as string;
  const prov = entry.provenance as Record<string, unknown> | undefined;
  const agentName = prov?.agentId ?? "unknown";
  log("act3", `  ← read ${ep}/${key} (written by ${agentName})`);
}
console.log();

// ── 3g. Commit the coordination outcome ──────────────────────────────

// SAY: "Finally, the facebook-ads-agent — the original proposer — commits
//       the outcome. This snapshots the entire causal chain: proposal,
//       cross-agent reads, responses, constraints. It's an immutable record
//       of how 3 agents reached consensus through memory alone."
subheader("3g. Commit the coordination outcome");

fb.recordContext("coordination-decision",
  "3 agents agreed on getAdPlatformData() extraction: " +
  "facebook-ads-agent proposed, google-ads-agent validated for Google Ads, " +
  "redshift-agent added Redshift compatibility constraint",
  { source: "agent-coordination" },
);
fb.commitOutcome("extraction-consensus-ad-fetch", OutcomeType.SUCCESS);
ok("Outcome committed: extraction-consensus-ad-fetch → SUCCESS");
log("act3", "Decision trace preserved: 3-agent coordination with full causal chain");
log("act3", "Any future agent can call explain() to see exactly how this was decided\n");

// SAY: "That's agent-to-agent coordination — no rooms, no chat.
//       Just structured memory writes, cross-agent reads, and causal traces.
//       Every decision is traceable. Every read is tracked."
ok("3 agents coordinated entirely through shared memory");
ok("Every cross-agent read is tracked. Every decision is traceable.");
ok("No external system needed — this is pure AMFS.");

await syncToServer();

// =====================================================================
//  ACT 4 — "From patterns to libraries" (the roadmap)
//  SAY: "Everything you've seen so far works today. This last part shows
//        where we're going: the system surfaces exactly what to extract,
//        and the extraction step is what we build together."
// =====================================================================
header("ACT 4 — From patterns to libraries");
pause("Knowledge graph + briefing surface extraction candidates");

// SAY: "The knowledge graph shows relationships between agents, code,
//       and pattern tags. When you see 3 different agents all pointing
//       at the same pattern, you KNOW there's a library waiting to be born."
subheader("4a. Knowledge graph reveals connections");

log("act4", "GET /api/v1/pro/graph/neighbors?entity_path=ecomm/facebook-ads&depth=2\n");
log("act4", "Would show:");
log("act4", "  facebook-ads-agent ──wrote──▶ data-fetch-code ──references──▶ facebook-ads-api");
log("act4", "  dashboard-agent ──wrote──▶ data-fetch-code-v2 ──references──▶ facebook-ads-api");
log("act4", "  reporting-agent ──read──▶ data-fetch-code");
log("act4", "");
log("act4", "When 3+ agents all reference the same pattern, the graph makes it obvious.\n");

// Pattern search across all agents
const fbAdsEntries = reviewer.search({ patternRef: "facebook-ads-api" });
ok(`"facebook-ads-api" pattern: ${fbAdsEntries.length} entries across ${new Set(fbAdsEntries.map(e => e.provenance.agentId)).size} agents`);

const adPlatformEntries = reviewer.search({ patternRef: "ad-platform-fetch" });
ok(`"ad-platform-fetch" pattern: ${adPlatformEntries.length} entries across ${new Set(adPlatformEntries.map(e => e.provenance.agentId)).size} agents`);

// SAY: "The Cortex briefing is what an agent gets when it starts working.
//       It compiles everything known about an entity into one digest:
//       who wrote what, how confident it is, what's redundant."
subheader("4b. Cortex briefing compiles the picture");

log("act4", "GET /api/v1/briefing?entity_path=ecomm/facebook-ads\n");
log("act4", "Would compile:");
log("act4", "  Entity: ecomm/facebook-ads");
log("act4", "  Activity: HIGH — 3+ agents, multiple versions, frequent outcomes");
log("act4", "  Confidence: 0.70-0.92 across versions");
log("act4", "  Redundancy: facebook-ads-agent AND dashboard-agent both have fetch code");
log("act4", '  Recommendation: "Candidate for library extraction"\n');

// SAY: "So here's the summary. Let me show you what works TODAY
//       vs. what's on the roadmap."
subheader("4c. The extraction roadmap");

const totalRedundantTokens = redundant.reduce((sum, [, info]) => sum + info.totalTokens, 0);

// SAY: "TODAY — detection, coordination, and knowledge are all working.
//       The ~38k tokens of waste? That's already visible in Act 2."
log("act4", "What AMFS gives you TODAY:");
ok("Detection: pattern scan finds redundant writes automatically");
ok("Coordination: agents communicate through shared memory with full traceability");
ok("Knowledge: graph + briefing surface what's connected and high-confidence");
ok(`Savings already visible: ~${totalRedundantTokens} tokens in redundant patterns\n`);

// SAY: "What's NEXT — the last mile. Three steps:
//       1. Cluster the code that looks the same
//       2. Extract it into a function: getAdPlatformData(platform, token, accountId, dateRange)
//       3. Rewrite the skill to call the function instead of regenerating 200 lines
//       That turns 200 lines into 1 line. This is what we build together."
log("act4", "What we build TOGETHER (the last mile):");
log("act4", "  1. Code-level clustering — group entries with similar generated code");
log("act4", "  2. Function extraction — generate getAdPlatformData() from the cluster");
log("act4", "  3. Skill rewriting — update skills to call the function\n");

ok("The coordination protocol is here. The data is here.");
ok("The extraction is what we build next — and your agents are already");
ok("telling us exactly what to extract.");

// ── Final stats ──────────────────────────────────────────────────────
// SAY: "That's the full loop. Let me show the final stats."
header("DEMO COMPLETE");

const finalStats = reviewer.stats();
log("stats", `Total entries: ${finalStats.totalEntries}`);
log("stats", `Total entities: ${finalStats.totalEntities}`);
log("stats", `Total agents: ${finalStats.totalAgents}`);
log("stats", `Avg confidence: ${finalStats.confidenceAvg.toFixed(2)}`);
log("stats", `Entities: ${Object.keys(finalStats.entities).join(", ")}`);
log("stats", `Scopes: ${reviewer.listScopes().join(", ")}`);
console.log();
log("stats", "Entity tree:");
console.log(reviewer.tree());

// SAY (CLOSING):
//   "To recap:
//    - Act 1: Agents store code, other agents find and reuse it.
//      That's immediate token savings.
//    - Act 2: An ops-review-agent reads everything and tells you
//      which skills are failing and what's redundant. That's your visibility.
//    - Act 3: Agents read from each other, write proposals, and build
//      traceable decision chains. That's coordination.
//    - Act 4: The graph and briefing surface what to extract.
//      The extraction itself — turning 200 lines into 1 line —
//      is what we build together next.
//
//    The foundation is here. Your agents are already telling the system
//    what to extract. We just need to close the loop."
