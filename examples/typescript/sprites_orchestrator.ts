/**
 * Fly.io Sprites × SenseLab — TypeScript orchestrator.
 *
 * A Node orchestrator spins up Sprites and needs each one to boot already
 * knowing the shared context. This example uses the AMFS TypeScript SDK's async
 * HTTP bridge to talk to hosted SenseLab: it hydrates a system prompt from
 * durable memory, (pretends to) run the agent, then commits the outcome so the
 * next Sprite is better briefed.
 *
 * Setup:
 *   npm install @senselab-ai/amfs
 *   export AMFS_HTTP_URL=https://amfs-login.sense-lab.ai
 *   export AMFS_API_KEY=amfs_sk_...
 *   export AMFS_ENTITY_PATH=sprites/acme/checkout
 *
 * Run (with tsx or ts-node):
 *   npx tsx examples/typescript/sprites_orchestrator.ts
 */

import { AgentMemory, HttpAdapter, OutcomeType } from "@senselab-ai/amfs";

/** Build the memory session a fresh Sprite would use. */
function provisionMemory(agentId: string): { memory: AgentMemory; entityPath: string } {
  const url = process.env.AMFS_HTTP_URL;
  const apiKey = process.env.AMFS_API_KEY;
  const entityPath = process.env.AMFS_ENTITY_PATH ?? "sprites/acme/checkout";

  if (!url || !apiKey) {
    throw new Error(
      "Set AMFS_HTTP_URL and AMFS_API_KEY to point at hosted SenseLab. " +
      "See the header of this file for setup."
    );
  }

  const adapter = new HttpAdapter({ url, apiKey, agentId });
  const memory = new AgentMemory(agentId, { adapter });
  return { memory, entityPath };
}

/** Turn durable memory for an entity into a system-prompt block. */
async function hydratePrompt(memory: AgentMemory, entityPath: string): Promise<string> {
  // The compiled Cortex briefing is the fastest way to load context.
  const briefing = (await memory.briefingAsync({ entityPath, limit: 5 })) as unknown;

  // Concrete high-signal memories make the carry-over tangible.
  const top = await memory.retrieveAsync(`important context for ${entityPath}`, {
    entityPath,
    limit: 6,
  });

  if (top.length === 0) {
    return (
      `## Memory\n\nNo prior memory for \`${entityPath}\` yet — fresh start. ` +
      `Save durable decisions and risks as you work.`
    );
  }

  const lines = [
    `## What earlier sessions learned about \`${entityPath}\``,
    "",
    "This was loaded from SenseLab, your persistent memory. Build on it.",
    "",
    "### Key memories",
    ...top.map(({ entry, score }) => {
      const e = entry as unknown as { key: string; value: unknown };
      const value = typeof e.value === "string" ? e.value : JSON.stringify(e.value);
      return `- **${e.key}** (score ${score.toFixed(2)}): ${value.slice(0, 400)}`;
    }),
  ];
  // briefing is available for callers that want the compiled narrative too.
  void briefing;
  return lines.join("\n");
}

async function runSprite(agentId: string, learn: boolean): Promise<void> {
  const { memory, entityPath } = provisionMemory(agentId);

  console.log(`\n=== Sprite (${learn ? "cold, will learn" : "fresh, hydrated"}) ===`);
  console.log(await hydratePrompt(memory, entityPath));

  if (learn) {
    // The agent forms durable knowledge during its run.
    await memory.writeAsync(
      entityPath,
      "decision-idempotency",
      "Charge calls carry an idempotency key derived from the cart id so retries never double-bill.",
      { confidence: 0.9 }
    );
    await memory.commitOutcomeAsync("checkout-hardening", OutcomeType.SUCCESS, {
      entityPath,
    });
    console.log("Learned + committed. This microVM can now be discarded.");
  }
}

async function main(): Promise<void> {
  // First Sprite learns; a second, brand-new Sprite hydrates what it learned.
  await runSprite("checkout-agent", true);
  await runSprite("checkout-agent", false);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
