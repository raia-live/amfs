---
title: Migrate from Mem0 or Zep
layout: default
parent: Guides
nav_order: 13
description: "Bring your memory across from Mem0 or Zep with a guided import: preview the counts first, watch it run, and remove it again if you change your mind."
---

# Migrate from Mem0 or Zep
{: .no_toc }

Paste an API key, look at what would come over, and start it. The import runs on our side while you close the tab, and everything it wrote can be removed in one action if you change your mind.
{: .fs-6 .fw-300 }

Available on Pro, Teams and Enterprise.
{: .fs-3 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Before you start

You need a read-capable API key for the account you're moving out of, and enough room on your plan for what's coming. The preview tells you both before anything is written.

Nothing is removed from Mem0 or Zep. An import reads; it never deletes, and you can keep running the old system alongside AMFS for as long as you want to compare them.

## Run the import

Open **Migrate** in the dashboard and pick the source. It's three steps.

**1. Connect.** Paste the API key. It's checked against the source immediately, so a wrong or expired key fails here rather than twenty minutes into a run. Once the key works you'll see which account it belongs to — worth a glance if you have more than one project.

**2. Review.** This is the whole point of the flow. You get counts of what's there, how many memories that becomes in AMFS, roughly how long it will take, and whether your plan has room for it. Adjust the options (below), refresh, and the numbers re-price for the import you actually chose. Counting reads your source at a deliberately slow pace, so give it a few seconds; if the source is rate limiting you, the counts may be unavailable and you can still go ahead.

**3. Import.** It runs on our infrastructure, not in your browser. Close the tab, come back tomorrow, reopen the page — the progress is where you left it. You can stop it at any point.

### Options worth knowing about

| Option | What it does |
|:--|:--|
| **Import a few users first** | Caps the import to N subjects. The honest way to try this: run five users, look at what landed, then run the rest. On Mem0 this makes an import smaller but not shorter — its list can't be filtered to a set of users, so the pages still have to be walked |
| **Trial run** | Reads and maps everything and writes nothing, then tells you what it would have written. It still reads from the source, so it costs whatever that reading costs you there |
| **Include facts that are no longer true** | Facts the source has stopped believing. Off by default. On, they come in at low confidence, so you keep the history of what changed without it outranking anything live |
| **Include raw conversation events** | The messages the facts were derived from. On by default for Zep; off for Mem0, where reading history is metered against your **monthly** quota there rather than a rate limit |

## What comes over

**From Zep:** users and threads, the facts on your knowledge graph with their entities and relationships, the graph edges between them, your custom entity types, and — unless you turn it off — the episodes behind the facts.

**From Mem0:** every memory in the project, the entities they belong to, and optionally each memory's version history.

Both sources are read into the same shape before anything is written, so a Zep fact and a Mem0 memory are treated identically wherever they differ only in where they came from.

### Confidence

Neither Mem0 nor Zep has a confidence score, so AMFS assigns one by what the record is:

| What it is | Confidence |
|:--|:--|
| Transferred verbatim — a profile, an entity, a message | 1.0 |
| A fact the source currently believes | 0.8 |
| A summary the source's model wrote | 0.6 |
| A fact the source has stopped believing | 0.3 |

The alternative would be 1.0 across the board, which puts another product's guess level with a pattern your agents watched succeed in production. Confidence exists to prevent exactly that, so an import is not allowed to flatten it.

These are a starting point, not a verdict. Once your agents start reading these entries and committing outcomes, [confidence moves with what actually happened](/amfs/concepts/confidence/) and the imported guesses stop being guesses.

### Timestamps and authorship

Entries keep the timestamps they had at the source, so a fact your assistant learned in March is dated March here, not the day you migrated. Everything an import writes is attributed to a synthetic author — `imported-from-zep`, `imported-from-mem0` — which is what makes it distinguishable from your agents' own work forever after.

Imported entries live under a path named for the source: `zep/subjects/alice`, `mem0/alice/facts`. Your agents can read them like anything else.

## Stopping, resuming, removing

**Stop** takes effect between batches. It is not a rollback — what was already written is memory you can query, and leaving it is usually what you want.

**Resume** picks up from the last checkpoint rather than starting over. It asks for your API key again, because we clear it the moment a job stops. Holding a live credential for another product indefinitely, against the chance you come back, is a worse trade than one paste.

**Remove imported data** deletes everything that import wrote, matched on the synthetic author and the source's path prefix. Two things survive it deliberately: entries under your own paths, and any imported entry one of your agents later corrected — that version carries the agent's id, so it is the agent's, not the import's.

## Limits

- **Zep counts are estimates.** Zep gives an exact user count but no way to count facts without walking them, so the preview samples and extrapolates. Anything estimated is labelled "about". Mem0 previews are exact when Mem0 lets us count — see below.
- **Very long Zep histories can't be walked to the end.** Zep's episode endpoint serves a fixed window with no cursor past it. The preview says so rather than quietly truncating.
- **Your source may not let us count.** Mem0's free plan allows about ten API requests a minute, and counting a project takes three, so a busy account can be told to wait. The preview says so and offers to try again — and you can start the import without counts. It runs in the background at a pace the source allows, and reports the real numbers as it goes.
- **One import per source at a time**, per account.
- **Imported memories count against your plan** like any other write, which is why the plan check runs in the preview instead of failing you four fifths of the way through.

## What AMFS won't do for you

Mem0 and Zep both extract memory from conversations with an LLM as they run. AMFS doesn't: an entry exists because an agent decided to write it. So an import brings across what those systems already extracted, and from then on your agents do the deciding.

If continuous extraction from a raw conversation stream is the thing you rely on, read [how the two models differ](/amfs/vs-competitors/) before you migrate — that is a real difference in approach, and it is better to find it now.

## Other sources

LangMem, Letta and raw pgvector are the next three we're asked about. If you need one of those, or a source not listed here, [tell us](https://github.com/raia-live/amfs/issues) — a new source is one module against the same import machinery.
