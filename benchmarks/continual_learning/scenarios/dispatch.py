"""dispatch: routing and delegation inside an agentic platform.

The most common shape of a multi-agent system deployed in 2026 is a planner that receives
a task and hands it to one of several specialist agents (or to a human queue). The
planner's whole job is the routing decision, and the routing decision is exactly the kind
of knowledge that lives nowhere but in outcomes: which specialist actually has the
credentials, the context and the taste to get a class of task past QA at this company.
Specialist names say what the team *intended* each agent to do; access rights, model
downgrades and ownership boundaries say what each agent can *actually* do, and the two
drift apart.

Same discipline as ``support``: no episode repeats (ten task classes, paraphrased request
templates, random fillers); in 6 of 10 classes the specialist that succeeds is not the one
the name suggests, for a hidden reason; the status tool is realistic but non-diagnostic
(queue depth and latency never reveal access rights or fit); failure feedback is what the
platform would say (a bounced task, a QA rejection, a CI failure), never the answer key.
The terminal tool has an ``action`` enum over the specialists, so the SenseLab arm gets
action priors (act / explore / escalate) over ``route:<specialist>``.

Regime change (``change_at``): the platform team reshuffles access and models. One class
changes back to the textbook routing (pure unlearning), two change to a new quirk (the
name-based pick is still wrong, so no arm wins by forgetting alone).
"""

from __future__ import annotations

import random
from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params
from .realworld import change_tags

# The action enum. Nothing in a specialist's name points at a task class it secretly
# cannot or can do; "human-queue" is the escalation analogue and always accepted.
SPECIALISTS = ["code-agent", "data-agent", "research-agent", "ops-agent", "writer-agent", "support-agent", "human-queue"]

FILLERS: dict[str, list[str]] = {
    "repo": ["acme/platform", "acme/billing-svc", "acme/web", "acme/data-pipelines", "acme/mobile", "acme/gateway"],
    "table": ["fct_orders", "dim_customers", "fct_payments", "stg_events", "fct_subscriptions", "dim_products"],
    "customer": ["Northwind Traders", "Globex", "Initech", "Umbrella Corp", "Hooli", "Vandelay Industries",
                 "Stark Industries", "Wonka Ltd", "Soylent Corp"],
    "package": ["requests", "numpy", "react", "fastapi", "pydantic", "openssl", "grpcio", "sqlalchemy"],
    "competitor": ["Contoso", "Fabrikam", "Wayne Analytics", "Tyrell Data", "Cyberdyne Cloud"],
    "service": ["checkout", "ledger", "search", "gateway", "mailer", "catalog", "wallet"],
    "region": ["EMEA", "APAC", "North America", "LATAM"],
    "dashboard": ["Revenue Overview", "Weekly Actives", "Churn Funnel", "Ops SLA", "Pipeline Health"],
    "segment": ["enterprise", "SMB", "self-serve", "mid-market"],
    "plan": ["Starter", "Pro", "Team", "Enterprise"],
    "metric": ["net revenue retention", "activation rate", "gross margin", "median time-to-value"],
    "requester": ["Priya", "Marcus", "Lena", "Tomas", "Aiko", "Diego", "Sam", "Nadia"],
    "team": ["Finance", "Growth", "Platform", "Customer Success", "Product", "Security"],
}

# task class -> paraphrased request templates, the specialist the name suggests (``obvious``,
# what an LLM picks cold), the specialist that actually gets the task past QA at Acme
# (``fix``), and why. In 6 of 10 classes fix != obvious.
DISPATCH_CLASSES: dict[str, dict[str, Any]] = {
    "sql-report": {
        "msgs": ["Need the {metric} numbers for {region} by {plan} plan for the {month} board deck, pulled from {table}.",
                 "Can someone run the monthly {segment} revenue report off {table}? {team} needs it by {date}.",
                 "Report request: cohort retention for {region} customers, source table {table}, cut by {plan}."],
        "obvious": "data-agent", "fix": "code-agent",
        "why": "data-agent has no warehouse credentials; code-agent runs the query through the reporting service account checked into acme/data-pipelines"},
    "customer-email": {
        "msgs": ["Draft a reply to {customer} about the outage on {date}; they have an open case and are on the {plan} plan.",
                 "{customer} asked for an update on their escalation. Please write the email; {requester} will send it.",
                 "Write a renewal-risk email to {customer} ({segment}) addressing what they raised in their last two tickets."],
        "obvious": "writer-agent", "fix": "support-agent",
        "why": "writer-agent has no CRM context and QA rejects drafts that ignore the customer's open case and plan; support-agent writes from the case history"},
    "dependency-upgrade": {
        "msgs": ["Bump {package} to the latest patch release in {repo} and get CI green.",
                 "Security advisory on {package}; upgrade it across {repo} before {date}.",
                 "Routine upgrade: {package} in {repo} is two minors behind, please update."],
        "obvious": "code-agent", "fix": "ops-agent",
        "why": "ops-agent owns the lockfiles and the CI build cache; a code-agent PR always lands with the lockfile out of sync"},
    "market-brief": {
        "msgs": ["Put together a one-page market brief on {segment} demand in {region} for {team}'s planning session on {date}.",
                 "{requester} needs a market brief: pricing trends among {competitor} and two other players in {region}.",
                 "Brief on the {region} {segment} market: size, growth, top three vendors. Due {date}."],
        "obvious": "research-agent", "fix": "research-agent", "why": "generic is right"},
    "incident-postmortem": {
        "msgs": ["Write the postmortem for the {service} incident on {date}; {team} wants it circulated by end of week.",
                 "Postmortem needed for INC on {service} ({region} impact). Timeline is in the incident channel.",
                 "Draft the blameless postmortem document for the {service} outage; {requester} will review."],
        "obvious": "ops-agent", "fix": "writer-agent",
        "why": "ops-agent writes terse timelines that QA rejects for lacking an impact statement and narrative; writer-agent with the ops timeline passes"},
    "dashboard-change": {
        "msgs": ["Add a {region} breakdown to the {dashboard} dashboard and switch the top tile to {metric}.",
                 "The {dashboard} dashboard filters by {plan} incorrectly; fix the filter and add a {segment} view.",
                 "Update {dashboard}: new tile for {metric}, sourced from {table}."],
        "obvious": "data-agent", "fix": "data-agent", "why": "generic is right"},
    "bug-triage": {
        "msgs": ["{customer} reports intermittent 500s on {service} since {date}; triage and attach a reproduction.",
                 "Bug from {customer} ({plan} plan): exports from {service} come back empty. Please triage.",
                 "Triage: {customer} says the {service} webhook fires twice. Repro steps and severity please."],
        "obvious": "code-agent", "fix": "support-agent",
        "why": "reproduction needs the customer's environment and case history, which only support-agent can see; code-agent cannot reproduce on main"},
    "pricing-page-copy": {
        "msgs": ["Rewrite the pricing page copy for the {plan} tier launch; keep the {segment} positioning.",
                 "New pricing page: headline plus three tier blurbs ({plan} is the anchor). {team} signs off.",
                 "Refresh the pricing page copy for {region}; the {plan} plan gets a new tagline."],
        "obvious": "writer-agent", "fix": "writer-agent", "why": "generic is right"},
    "data-backfill": {
        "msgs": ["Backfill {table} for {region} from {date} onward; the upstream job dropped a week.",
                 "{table} is missing {segment} rows since {date}. Please backfill and verify counts.",
                 "Run the backfill for {table} ({plan} plan customers, {region}) that the pipeline skipped."],
        "obvious": "data-agent", "fix": "ops-agent",
        "why": "backfills need prod write access and a dry-run gate; only ops-agent has both"},
    "competitor-teardown": {
        "msgs": ["Teardown of {competitor}: onboarding flow, pricing, integrations. {team} wants it by {date}.",
                 "{requester} needs a competitor teardown on {competitor} for the {segment} segment.",
                 "Competitive analysis: {competitor} vs us in {region}. Feature matrix and pricing table."],
        "obvious": "research-agent", "fix": "research-agent", "why": "generic is right"},
}

# Regime change (when ``change_at`` is set): the platform team reshuffles access and models
# and tells nobody downstream. 3 of 10 classes change. One changes to the *textbook*
# routing (the pure unlearning case: forgetting the quirk is enough); two change to a *new*
# quirk (the name-based pick is still wrong, so an arm with no memory cannot win by default
# and every memory arm has to relearn).
DISPATCH_CHANGES: dict[str, dict[str, str]] = {
    "sql-report": {"fix": "data-agent",
                   "why": "data-agent was granted warehouse credentials in the Q3 access review and now runs the report "
                          "end to end; the code-agent detour through the service account is no longer needed"},
    "market-brief": {"fix": "writer-agent",
                     "why": "research-agent was moved to a cheaper model to cut cost and its briefs now fail QA on synthesis; "
                            "writer-agent working from the research feed passes"},
    "bug-triage": {"fix": "ops-agent",
                   "why": "support-agent lost repository access in the security tightening; ops-agent, which holds both "
                          "repo access and the prod tenant logs, is now the only specialist that can reproduce a "
                          "customer-specific bug (code-agent still cannot reproduce on main)"},
}
# Change set "B" (see Scenario.change_variant): three classes "A" leaves alone, moved to
# specialists no "A" episode ever pairs them with. Same shape as "A": one change to the
# name-based routing (dependency upgrades go back to code-agent — the pure unlearning case)
# and two to a new quirk (dashboards became dashboards-as-code, so a dashboard change is a PR;
# postmortems ship through the status-page workflow the support specialist owns).
DISPATCH_CHANGES_B: dict[str, dict[str, str]] = {
    "dependency-upgrade": {"fix": "code-agent",
                           "why": "the lockfile and CI build cache are now maintained by the repo bot on every PR, so a "
                                  "code-agent upgrade lands in sync; the ops-agent detour is no longer needed"},
    "dashboard-change": {"fix": "code-agent",
                         "why": "dashboards moved to dashboards-as-code under acme/data-pipelines; a dashboard change is a "
                                "reviewed PR, and data-agent has no write access to the repo (the BI tool is read-only)"},
    "incident-postmortem": {"fix": "support-agent",
                            "why": "postmortems now ship through the customer-facing status-page workflow, whose template "
                                   "and customer-impact figures live in the case system; support-agent writes from "
                                   "there, while writer-agent's narrative drafts lack the impact data QA requires"},
}
DISPATCH_CHANGE_SETS = {"A": DISPATCH_CHANGES, "B": DISPATCH_CHANGES_B}

# What the platform says when the wrong specialist gets a task. Keyed (class, specialist);
# realistic, and never names the specialist that would have worked. Missing pairs fall back
# to a per-specialist generic bounce below.
FAILURES: dict[tuple[str, str], str] = {
    ("sql-report", "data-agent"): "data-agent returned: 'no credentials for warehouse acme_dw'. Task bounced back.",
    ("sql-report", "code-agent"): "code-agent returned: 'reporting service account was rotated out; query job failed on auth'. Task bounced back.",
    ("sql-report", "ops-agent"): "ops-agent ran the report query against the prod replica with no LIMIT; replica lag hit 12 minutes and paged the on-call. Query killed, no report.",
    ("customer-email", "writer-agent"): "QA rejected the draft: 'does not reference the customer's open case or plan.'",
    ("dependency-upgrade", "code-agent"): "PR opened but CI failed: lockfile out of sync with the build cache.",
    ("market-brief", "research-agent"): "research-agent returned a brief; QA rejected it: 'reads like a list of headlines; no synthesis, two figures unsourced.'",
    ("incident-postmortem", "ops-agent"): "ops-agent posted a 9-line timeline. QA rejected it: 'no impact statement, no root-cause narrative, no action items.'",
    ("bug-triage", "code-agent"): "code-agent returned: 'cannot reproduce on main; need the customer's environment and steps'. Task bounced back.",
    ("bug-triage", "support-agent"): "support-agent returned: 'repository access denied (403) pulling the reproduction branch'. Task bounced back.",
    ("data-backfill", "data-agent"): "data-agent returned: 'write access to prod tables denied'. Task bounced back.",
    ("data-backfill", "code-agent"): "code-agent ran the backfill script against prod without a dry run; 40k rows double-written. Rollback required.",
    # Pairs that are correct under change set "A" and wrong under "B"; never reached under "A".
    ("dependency-upgrade", "ops-agent"): "ops-agent returned: 'lockfile is bot-managed now; nothing for infra to do here'. Task bounced back after 5h in its queue.",
    ("dashboard-change", "data-agent"): "data-agent returned: 'dashboard definitions are version-controlled; direct edits in the BI tool are locked'. Task bounced back.",
    ("incident-postmortem", "writer-agent"): "QA rejected the draft: 'not in the status-page format; no customer-impact figures.'",
}
# Destructive outcomes: the specialist did something that has to be undone.
CRITICAL_PAIRS = {("data-backfill", "code-agent"), ("sql-report", "ops-agent")}
GENERIC_BOUNCE: dict[str, str] = {
    "code-agent": "code-agent returned: 'no code change identified for this request'. Task bounced back.",
    "data-agent": "data-agent returned a partial result; QA rejected it: 'numbers do not reconcile with the source of record.'",
    "research-agent": "research-agent returned a summary; QA rejected it: 'generic, nothing Acme-specific, no primary sources.'",
    "ops-agent": "ops-agent returned: 'not an infrastructure task'. Task bounced back after 6h in its queue.",
    "writer-agent": "QA rejected the draft: 'well written but unsupported; the underlying facts and numbers are missing.'",
    "support-agent": "support-agent returned: 'no customer case attached to this task; cannot proceed'. Task bounced back.",
}


class DispatchScenario(Scenario):
    name = "dispatch"
    agent_base = "dispatch-agent"
    role = ("You are the dispatcher of Acme's internal agent platform. For each incoming task choose the ONE "
            "specialist to delegate it to. You will see the specialist's result; a task that fails QA comes back to you.")

    tools = [
        ToolSpec("specialist_status", "Current queue depth, average turnaround and uptime for every specialist.",
                 {"type": "object", "properties": {
                     "specialist": {"type": "string", "description": "Optional: one specialist. Omit for all."}},
                  "required": []}),
        ToolSpec("route", "Delegate the task to one specialist with a one-line brief.",
                 finish_params({"action": {"type": "string", "enum": SPECIALISTS,
                                           "description": "the specialist (or human queue) to delegate to"},
                                "brief": {"type": "string", "description": "one-line brief for the specialist"}},
                               ["action", "brief"]), terminal=True),
    ]

    def build(self) -> None:
        rng = self.rng
        ids = list(DISPATCH_CLASSES)
        # step 3 is coprime with len(ids)=10, so the first 20 episodes cover every class twice
        self.sched = [ids[(ep * 3 + self.seed) % len(ids)] if ep < 2 * len(ids) else rng.choice(ids)
                      for ep in range(self.episodes)]
        self.instances = [self._instance(ep, cid, 0) for ep, cid in enumerate(self.sched)]

    def _instance(self, ep: int, cid: str, salt: int) -> dict[str, Any]:
        irng = random.Random(f"dsp-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
        cls = DISPATCH_CLASSES[cid]
        fix, why, old = cls["fix"], cls["why"], None
        active = self.active_changes(ep)
        if cid in active:
            old, fix, why = fix, active[cid]["fix"], active[cid]["why"]
        fillers = {k: irng.choice(v) for k, v in FILLERS.items()}
        fillers["date"] = f"2026-{irng.randint(7, 11):02d}-{irng.randint(1, 28):02d}"
        fillers["month"] = irng.choice(["July", "August", "September", "October", "November"])
        return {"task_class": cid, "msg": irng.choice(cls["msgs"]).format(**fillers), "fillers": fillers,
                "priority": irng.choice(["P2", "P2", "P3", "P1"]),
                "fix": fix, "generic": cls["obvious"], "why": why, "old_fix": old,
                "change_class": cid in self.changes}

    def change_set(self, variant: str) -> dict[str, dict[str, str]]:
        return DISPATCH_CHANGE_SETS[variant]

    def regenerate(self, ep: int, salt: int) -> None:
        cid = random.Random(f"dsp-focus-{self.seed}-{ep}-{salt}").choice(self.focus_classes(ep))
        self.sched[ep] = cid
        self.instances[ep] = self._instance(ep, cid, salt)

    def oracle_fix_entries(self, variant: str | None = None) -> list[tuple[str, str, float]]:
        return [(f"update-routing-{cid}", f"Routing update — {cid.replace('-', ' ')} tasks now go to {c['fix']} "
                                          f"(previously {DISPATCH_CLASSES[cid]['fix']}). Reason: {c['why']}.", 1.0)
                for cid, c in self.change_set(variant or self.change_variant).items()]

    def seed_entries(self) -> list[tuple[str, str, float]]:
        # The onboarding routing guide maps every class to the specialist its name suggests:
        # right in 4 of 10 classes, wrong by design in the other 6 (like support's tier-1 playbook).
        out = [("routing-guide", "Dispatch routing guide: SQL and reporting requests -> data-agent; customer emails -> "
                                 "writer-agent; dependency upgrades -> code-agent; market briefs -> research-agent; "
                                 "incident postmortems -> ops-agent; dashboard changes -> data-agent; bug triage -> "
                                 "code-agent; pricing and marketing copy -> writer-agent; data backfills -> data-agent; "
                                 "competitor teardowns -> research-agent; anything unclear -> human-queue.", 0.8),
               ("platform-specialists", "Acme agent platform specialists: code-agent (repositories, PRs, CI); data-agent "
                                        "(warehouse, SQL, dashboards); research-agent (web research, briefs); ops-agent "
                                        "(infrastructure, deploys, on-call tooling); writer-agent (long-form copy and "
                                        "documents); support-agent (CRM, customer cases); human-queue (manual fallback, "
                                        "SLA-tracked, to be avoided when a specialist can do the job).", 0.9)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        i = self.instances[ep]
        f = i["fillers"]
        # See Scenario.opaque_ids: a counted task number is the episode index.
        dsp = random.Random(f"dsp-id-{self.seed}-{ep}").randint(10000, 99999) if self.opaque_ids else 5200 + ep
        prompt = (f"Task DSP-{dsp} ({i['priority']}) from {f['requester']} in {f['team']}:\n\"{i['msg']}\"\n"
                  f"Route it to one specialist.")
        return Task(ep, prompt, dict(i), self.agent_for(ep),
                    {"task_class": i["task_class"], "class": i["task_class"], "quirk": i["fix"] != i["generic"],
                     **change_tags(self, ep, i["change_class"], cls=i["task_class"])})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        if name == "specialist_status":
            # Realistic but non-diagnostic: load and health say nothing about access rights or fit.
            srng = random.Random(f"status-{self.seed}-{task.episode}")
            want = str(args.get("specialist", "") or "")
            lines = []
            for s in SPECIALISTS:
                if want and s != want:
                    continue
                if s == "human-queue":
                    lines.append(f"human-queue: queue {srng.randint(9, 30)}, avg turnaround {srng.randint(14, 40)}h, "
                                 f"SLA breaches this week {srng.randint(0, 4)}")
                else:
                    lines.append(f"{s}: queue {srng.randint(0, 9)}, avg turnaround {srng.randint(12, 95)} min, "
                                 f"uptime {srng.choice(['99.5', '99.7', '99.8', '99.9', '100.0'])}%")
            return StepResult("\n".join(lines) or "unknown specialist")
        if name != "route":
            return StepResult(f"Unknown tool {name}")
        act = str(args.get("action", ""))
        stale = {"stale_pick": act == t.get("old_fix")} if t.get("old_fix") else {}
        flags = {"generic_pick": act == t["generic"], "human_queue": act == "human-queue", **stale}
        if act == t["fix"]:
            return StepResult(f"{act} delivered; QA passed; task closed.", terminal=True, success=True,
                              flags=flags, answer=act)
        if act == "human-queue":
            return StepResult("Queued for a human. SLA clock running; platform lead asked why this was not automated.",
                              terminal=True, success=False, severity="minor_failure", flags=flags, answer=act)
        if act not in SPECIALISTS:
            return StepResult(f"Platform rejected the route: no specialist named {act!r}.", terminal=True,
                              success=False, severity="failure", flags=flags, answer=act)
        key = (t["task_class"], act)
        fb = FAILURES.get(key, GENERIC_BOUNCE[act])
        sev = "critical_failure" if key in CRITICAL_PAIRS else "failure"
        return StepResult(fb, terminal=True, success=False, severity=sev,
                          flags={**flags, "destructive": key in CRITICAL_PAIRS}, answer=act)
