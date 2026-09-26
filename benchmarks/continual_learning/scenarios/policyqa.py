"""Policy Q&A: retrieval's home turf, with stale versions and drifted practice.

Grounding: the internal knowledge assistant ("ask the policy bot") is the most widely
deployed form of retrieval-augmented generation in companies today, and the one setting
where a plain vector store plays its strongest game: the answers ARE in the documents.
This scenario seeds those documents into every memory arm and asks factual questions about
Acme policies and plans, so a retriever that surfaces the right chunk can answer with no
learning at all. It then adds the two things every real corpus has and no RAG demo shows:

  * stale versions that were never removed. Five topics are "versioned": both the current
    doc chunk and the previous one sit in memory, titled like real documents (Billing
    Policy v2.1 vs v3.0). Retrieval has to pick the right version.
  * practice that drifted from the written policy. Three topics are "doc-lags-practice":
    the only chunk in memory carries the OLD value; the policy changed in practice and
    nobody updated the doc. These are learnable only from outcomes.

Four topics are "current-only" (one chunk, correct): RAG answers these outright, which is
what keeps the scenario fair. Which topic falls in which corpus class is drawn from the seed.

Regime change (``change_at``): four topics get a new value nobody writes down. The seeded
docs are NOT updated, so the chunk retrieval trusted yesterday now recommends the
pre-change value; ``truth.old_fix`` carries that value verbatim (it appears in the doc
text) so the loop's stale-context accounting works. Same discipline as the other
real-world scenarios: no episode repeats (fresh asker, account and paraphrase every time),
the prompt never contains the answer or the options, and failure feedback is what the
colleague or compliance actually says, never the value.

Scoring is deterministic. Every value has a canonical token (a word-boundary number such
as 14/30/60, or a keyword such as Team/Enterprise); for set-like values the token carries
an exclusion so that "CSV only" is CSV without JSON and "US and EU" is EU without APAC. An
answer succeeds iff it matches the truth token and matches neither the old-version token
nor the generic token: hedging between two values is wrong. Saying "not sure, check with
the policy owner" is a minor failure, cheaper than a confident wrong answer.
"""

from __future__ import annotations

import random
import re
from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params
from .realworld import change_tags


def _num(n: str) -> str:
    """Pattern for a number that is not glued to other digits or to a decimal/thousands
    separator: 14 matches "14 days" and "14." but not 4414, 140, 1.4 or 2,014."""
    return rf"(?<!\d)(?<!\d[.,]){n}(?!\d)(?![.,]\d)"


def _v(value: str, pat: str, exclude: str | None = None) -> dict[str, str | None]:
    """A canonical value: the string an agent would say, the token that identifies it, and
    an optional exclusion (a value whose distinguishing feature is the ABSENCE of a word)."""
    return {"value": value, "pat": pat, "not": exclude}


def _match(spec: dict[str, Any], text: str) -> bool:
    low = text.lower()
    if not re.search(spec["pat"], low):
        return False
    return not (spec.get("not") and re.search(spec["not"], low))


_DAYS14 = _num("14") + r"|\bfourteen\b|\btwo weeks\b|\b2 weeks\b"
_DAYS21 = _num("21") + r"|\btwenty-one\b|\bthree weeks\b|\b3 weeks\b"
_DAYS30 = _num("30") + r"|\bthirty\b|\bone month\b|\b1 month\b"
_DAYS45 = _num("45") + r"|\bforty-five\b"
_DAYS60 = _num("60") + r"|\bsixty\b|\btwo months\b|\b2 months\b"

# topic -> document identity, chunk template ({v} = the value), current/old/generic values,
# and paraphrased questions ({customer}/{company}/{plan}/{country} are filled per episode).
# current = truth before any regime change; old = the previous document version; generic =
# the industry default a model guesses with no Acme knowledge. All three are distinct. The
# generic values were checked against the agent model's actual no-document answers (gpt-5.5,
# 2026-09-19): every topic's first guess was remapped to ``generic`` so that a bare model does
# not land on the truth by prior (it did on 4 of 12 topics under the first draft).
POLICY_TOPICS: dict[str, dict[str, Any]] = {
    "refund-window": {
        "doc": "Billing Policy", "cur_ver": "v3.0 (2026-01)", "old_ver": "v2.1 (2025-03)",
        "text": "Section 4, Refunds: customers may request a full refund within {v} of any charge on a self-serve "
                "plan; later requests go to the account owner for discretionary approval.",
        "current": _v("45 days", _DAYS45), "old": _v("60 days", _DAYS60), "generic": _v("14 days", _DAYS14),
        "qs": ["{customer} at {company} was charged for a renewal they say they didn't want. How long after a charge "
               "can they still get a full refund?",
               "Before {company} upgrades they want to know our refund window on self-serve plans. What is it?",
               "A {plan} customer wants to return the subscription they just bought. Up to how long after the charge "
               "do we refund in full?",
               "What's the refund period I can quote to {customer}? They bought last week and want out."]},
    "log-retention": {
        "doc": "Data Retention Schedule", "cur_ver": "v4 (2025-11)", "old_ver": "v3 (2024-09)",
        "text": "Audit and access logs are retained for {v} and then purged; customers may request an export before "
                "the purge date.",
        "current": _v("120 days", _num("120") + r"|\bfour months\b|\b4 months\b"),
        "old": _v("30 days", _DAYS30),
        "generic": _v("90 days", _num("90") + r"|\bninety\b|\bthree months\b|\b3 months\b"),
        "qs": ["{company}'s security lead asks how long we keep audit and access logs before they're purged.",
               "Compliance questionnaire from {customer}: what is Acme's log retention period?",
               "{customer} needs to pull access logs for an investigation. How far back can they go?"]},
    "sla-credit": {
        "doc": "Service Level Agreement", "cur_ver": "2026 edition", "old_ver": "2024 edition",
        "text": "Section 6, Service credits: for any month in which monthly uptime falls below the target, the "
                "customer is entitled to a service credit of {v} of that month's fees, claimed within the next cycle.",
        "current": _v("15%", _num("15") + r"|\bfifteen percent\b"), "old": _v("5%", _num("5") + r"|\bfive percent\b"),
        "generic": _v("10%", _num("10") + r"|\bten percent\b"),
        "qs": ["{company} had an outage last month that took uptime under the SLA target. What service credit are "
               "they entitled to?",
               "{customer} is asking what percentage of the monthly fee we credit when we miss the uptime SLA.",
               "Renewal negotiation with {company}: what's the standard SLA credit in our agreement when we breach "
               "availability?"]},
    "team-seat-cap": {
        "doc": "Plans and Pricing Guide", "cur_ver": "2026-Q1", "old_ver": "2025-Q2",
        "text": "The Team plan includes up to {v} per workspace; organisations that need more move to Enterprise.",
        "current": _v("25 seats", _num("25") + r"|\btwenty-five\b"), "old": _v("10 seats", _num("10") + r"|\bten\b"),
        "generic": _v("50 seats", _num("50") + r"|\bfifty\b"),
        "qs": ["{company} is on {plan} and wants to move to the Team plan. What's the maximum number of seats Team "
               "allows?",
               "How many users can a Team plan workspace have before they need Enterprise? {customer} is asking.",
               "Seat cap question: {customer} wants to add their whole department to a Team plan. Where's the ceiling?"]},
    "export-formats": {
        "doc": "Data Export Guide", "cur_ver": "v2 (2025-10)", "old_ver": "v1 (2024-06)",
        "text": "Full data exports from the dashboard and the export API are delivered as {v}; exports run "
                "asynchronously and the download link is emailed when ready.",
        "current": _v("CSV, JSON and Parquet", r"\bparquet\b"),
        "old": _v("CSV only", r"\bcsv\b", r"\b(json|parquet|xlsx|excel|xml|avro|pdf)\b"),
        "generic": _v("CSV and JSON", r"\bjson\b", r"\b(parquet|avro)\b"),
        "qs": ["{customer} wants to pull all their data out for a warehouse load. Which file formats do our exports "
               "come in?",
               "{company}'s data engineer asks what formats the export API delivers.",
               "Format question from {customer}: when they run a full data export, what do they get back?"]},
    "sso-plan": {
        "doc": "Security Features Matrix", "cur_ver": "2026-02", "old_ver": "2025-01",
        "text": "SAML/OIDC single sign-on is available from the {v} upward; SCIM provisioning ships with it.",
        "current": _v("Team plan", r"\bteam\b"), "old": _v("Business plan", r"\bbusiness\b"),
        "generic": _v("Enterprise plan", r"\benterprise\b"),
        "qs": ["{company} ({plan}) wants to turn on SAML single sign-on. Which plan tier do they need to be on?",
               "{customer} asks at what plan SSO becomes available. What do I tell them?",
               "Sales question: the lowest tier that includes single sign-on with SCIM is which one? {company} is "
               "deciding."]},
    "api-rate-limit": {
        "doc": "API Reference: Limits", "cur_ver": "v2026.1", "old_ver": "v2025.2",
        "text": "Default rate limit is {v} per API key across all endpoints; requests above the limit return HTTP 429 "
                "with a Retry-After header.",
        "current": _v("750 requests per minute", _num("750") + r"|\bseven hundred (and )?fifty\b"),
        "old": _v("300 requests per minute", _num("300") + r"|\bthree hundred\b"),
        "generic": _v("1000 requests per minute", _num("1,?000") + r"|\b1k\b|\bone thousand\b"),
        "qs": ["{company}'s developers keep hitting HTTP 429 on the API. What's the default per-key rate limit I can "
               "quote them?",
               "{customer} is planning an integration and asks how many API requests per minute a single key gets by "
               "default.",
               "Rate limit question from {company}: what is the standard API quota before throttling kicks in?"]},
    "trial-length": {
        "doc": "Trials and Onboarding Playbook", "cur_ver": "2026-01", "old_ver": "2025-04",
        "text": "New self-serve workspaces get a free trial of {v} with all Pro features enabled; sales may extend "
                "a trial once.",
        "current": _v("21 days", _DAYS21), "old": _v("30 days", _DAYS30), "generic": _v("14 days", _DAYS14),
        "qs": ["{customer} just signed up for a self-serve workspace at {company}. How long does their free trial run?",
               "Prospect at {company} asks how many days the trial lasts before they have to pick a plan.",
               "Trial length question: {customer} wants to know when their trial workspace converts or locks."]},
    "support-hours": {
        "doc": "Support Operations Handbook", "cur_ver": "v5 (2025-12)", "old_ver": "v4 (2024-10)",
        "text": "Standard support coverage is {v} via chat and email; the Enterprise incident hotline is a separate "
                "entitlement.",
        "current": _v("24x5", r"24\s*[x\u00d7/]\s*5|(24 ?hours?|round[- ]the[- ]clock|around the clock)[^.;\n]{0,30}"
                              r"(monday|mon-fri|mon\u2013fri|weekdays|business days)"),
        "old": _v("24x7", r"24\s*[x\u00d7/]\s*7|\b(7|seven) days a week\b|\bevery day of the week\b|"
                          r"\baround[- ]the[- ]clock\b|\bround[- ]the[- ]clock\b",
                  r"weekdays|monday|mon-fri|mon\u2013fri|business days|24\s*[x\u00d7/]\s*5"),
        "generic": _v("business hours", r"\bbusiness hours\b|\boffice hours\b|\b9\s*(am)?\s*(-|to|\u2013)\s*5\s*(pm)?\b|"
                                        r"\bnine to five\b"),
        "qs": ["{company} is in {country} and asks when standard support is actually staffed. What coverage do we "
               "promise?",
               "{customer} wants to know if they can reach chat support on a Sunday night. What are our standard "
               "support hours?",
               "Onboarding call with {company}: what support coverage does the standard plan include, hours-wise?"]},
    "data-residency-regions": {
        "doc": "Data Residency Overview", "cur_ver": "v3 (2025-09)", "old_ver": "v1 (2024-05)",
        "text": "At signup a workspace can be pinned to {v}; the region choice is fixed for the life of the workspace.",
        "current": _v("US, EU and APAC", r"\bapac\b|\basia[- ]pacific\b|\basia\b"),
        "old": _v("US only", r"\b(usa|u\.s\.|united states|us[- ]only|us region|us data|us[- ]east|us[- ]west)\b|^us$",
                  r"\b(eu|europe|european|apac|asia)\b"),
        "generic": _v("US and EU", r"\b(eu|europe|european)\b", r"\b(apac|asia)\b"),
        "qs": ["{company} has a procurement requirement on where their data lives. Which hosting regions can a "
               "workspace be pinned to?",
               "{customer} asks which data-residency options Acme offers at signup.",
               "Sales call with {company} in {country}: what regions can they choose for data residency?"]},
    "annual-cancellation-notice": {
        "doc": "Terms of Service", "cur_ver": "2026 revision", "old_ver": "2025 revision",
        "text": "Section 9, Term: annual subscriptions renew automatically. Notice required to cancel before the "
                "renewal date: {v}.",
        "current": _v("60 days", _DAYS60),
        "old": _v("none (cancel any time before renewal)",
                  r"\bnone\b|\bno (advance |prior |minimum )?notice\b|\bany ?time\b|\bimmediately\b|" + _num("0") + r" days?"),
        "generic": _v("30 days", _DAYS30),
        "qs": ["{company} is on an annual contract renewing soon and wants out. How much notice do they have to give "
               "before the renewal date?",
               "{customer} asks what the cancellation notice period is on annual subscriptions.",
               "Contract question: how far ahead of an annual renewal does {company} have to tell Acme they're "
               "cancelling?"]},
    "admin-mfa": {
        "doc": "Security Policy", "cur_ver": "v2.3 (2026-01)", "old_ver": "v2.0 (2025-02)",
        "text": "Multi-factor authentication for workspace administrators is {v}; the setting lives under "
                "Security > Authentication.",
        "current": _v("recommended", r"\brecommended\b|\bencouraged\b|\bbest practice\b|\badvised\b"),
        "old": _v("optional", r"\boptional\b|\bopt[- ]in\b|\bat (the admin's|their) discretion\b"),
        # only negations are excluded: "recommended but not required" is the current policy,
        # "recommended and required" is a hedge and fails
        "generic": _v("required", r"\b(required|mandatory|enforced|compulsory|must)\b",
                      r"\bnot (required|mandatory|enforced)\b|\bisn't required\b|\bno longer required\b"),
        "qs": ["{customer} is setting up admins for {company}'s workspace and asks about the MFA rule for admin "
               "accounts. What does the policy say?",
               "Security review from {company}: do workspace administrators have to use multi-factor authentication "
               "under Acme's policy?",
               "{customer} asks whether admins can skip MFA. What's the actual policy on MFA for admin roles?"]},
}

# Regime change (when ``change_at`` is set): four policies change in practice and nobody
# re-issues the document. Each new value is distinct from current, old and generic, so no arm
# wins by default and every memory arm has to unlearn a chunk it can still retrieve.
POLICY_CHANGES: dict[str, dict[str, Any]] = {
    "refund-window": {"value": _v("21 days", _DAYS21),
                      "why": "after a finance review of refund abuse legal tightened the self-serve refund window "
                             "to 21 days; the Billing Policy document was not re-issued"},
    "log-retention": {"value": _v("180 days", _num("180") + r"|\bsix months\b|\b6 months\b"),
                      "why": "the SOC 2 auditor asked for six months of audit logs; retention moved to 180 days in "
                             "the platform config while the schedule document still says 120"},
    "api-rate-limit": {"value": _v("1500 requests per minute", _num("1,?500") + r"|\b1\.5k\b"),
                       "why": "the gateway migration doubled default per-key limits to 1500 requests per minute; "
                              "the API reference was not regenerated"},
    "team-seat-cap": {"value": _v("40 seats", _num("40") + r"|\bforty\b"),
                      "why": "the Q1 pricing refresh raised the Team plan seat cap to 40; the pricing guide was not "
                             "updated"},
}

CORPUS_VERSIONED, CORPUS_CURRENT, CORPUS_LAGS = "versioned", "current-only", "doc-lags-practice"
CORPUS_MIX = [CORPUS_VERSIONED] * 5 + [CORPUS_CURRENT] * 4 + [CORPUS_LAGS] * 3

ASKER_NAMES = ["Priya", "Marcus", "Elena", "Tomas", "Aiko", "Daniel", "Fatima", "Jonas"]
ASKER_ROLES = ["support rep", "account executive", "customer success manager", "ops analyst"]
FIRST_NAMES = ["Harper", "Wei", "Olu", "Sofia", "Ravi", "Ingrid", "Mateo", "Chioma", "Kenji", "Lena"]
LAST_NAMES = ["Lane", "Zhang", "Adeyemi", "Rossi", "Menon", "Berg", "Alvarez", "Okafor", "Sato", "Fischer"]
COMPANIES = ["Northwind Labs", "Contoso Health", "Bluebird Logistics", "Kestrel Analytics", "Marigold Retail",
             "Orbital Fintech", "Pinecrest Media", "Sable Robotics", "Tidewater Energy", "Vega Learning"]
# Customer plans never coincide with an answer option (sso-plan answers are plan names).
ACCOUNT_PLANS = ["Starter", "Pro"]
COUNTRIES = ["Germany", "Canada", "Brazil", "Japan", "Australia", "the UK", "India", "Mexico", "France",
             "the Netherlands"]
# Account-lookup user counts that equal no topic's numeric value.
SAFE_COUNTS = [3, 8, 12, 17, 33, 41, 64, 77]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

ABSTAIN = re.compile(r"not sure|don't know|do not know|check with|unsure|cannot confirm|can't confirm|"
                     r"unable to (confirm|find)|not (documented|covered)", re.I)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _chunk(tid: str, tp: dict[str, Any], ver_key: str, value: str) -> tuple[str, str, float]:
    ver = tp[ver_key]
    return (f"doc-{tid}-{_slug(ver)}", f"{tp['doc']} {ver}. {tp['text'].format(v=value)}", 0.8)


class PolicyQaScenario(Scenario):
    name = "policy-qa"
    agent_base = "policy-agent"
    role = ("You are Acme's internal policy and product-knowledge assistant. Colleagues in support, sales and ops "
            "ask you factual questions about Acme policies and plans; answer with the specific current value.")

    tools = [
        ToolSpec("lookup_account", "Look up the customer's account: company, plan, billing country, users, renewal.",
                 {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}),
        ToolSpec("answer", "Send the answer to the colleague.",
                 finish_params({"topic": {"type": "string", "enum": list(POLICY_TOPICS)},
                                "answer": {"type": "string",
                                           "description": "the specific value, e.g. '12 months' or 'Pro plan'"},
                                "source": {"type": "string",
                                           "description": "which policy/doc you relied on (or 'none')"}},
                               ["topic", "answer"]), terminal=True),
    ]

    def build(self) -> None:
        rng = self.rng
        ids = list(POLICY_TOPICS)
        order = list(ids)
        random.Random(f"policyqa-corpus-{self.seed}").shuffle(order)
        self.corpus = {tid: CORPUS_MIX[i] for i, tid in enumerate(order)}
        # step 5 is coprime with len(ids)=12, so the first 24 episodes cover every topic twice
        self.sched = [ids[(ep * 5 + self.seed) % len(ids)] if ep < 2 * len(ids) else rng.choice(ids)
                      for ep in range(self.episodes)]
        self.instances = [self._instance(ep, tid, 0) for ep, tid in enumerate(self.sched)]

    def _instance(self, ep: int, tid: str, salt: int) -> dict[str, Any]:
        irng = random.Random(f"pol-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
        tp = POLICY_TOPICS[tid]
        fix, old_fix, why = tp["current"]["value"], None, None
        in_change = tid in POLICY_CHANGES
        if self.changed(ep) and in_change:
            old_fix, fix, why = fix, POLICY_CHANGES[tid]["value"]["value"], POLICY_CHANGES[tid]["why"]
        first, last = irng.choice(FIRST_NAMES), irng.choice(LAST_NAMES)
        company, plan, country = irng.choice(COMPANIES), irng.choice(ACCOUNT_PLANS), irng.choice(COUNTRIES)
        customer = f"{first} {last}"
        question = irng.choice(tp["qs"]).format(customer=customer, company=company, plan=plan, country=country)
        return {"topic": tid, "corpus": self.corpus[tid], "question": question,
                "asker": f"{irng.choice(ASKER_NAMES)} ({irng.choice(ASKER_ROLES)})",
                "customer": customer, "company": company, "plan": plan, "country": country,
                "email": f"{first.lower()}.{last.lower()}@{_slug(company).replace('-', '')}.com",
                "fix": fix, "old": tp["old"]["value"], "generic": tp["generic"]["value"], "old_fix": old_fix,
                "why": why, "change_class": in_change}

    def regenerate(self, ep: int, salt: int) -> None:
        tid = random.Random(f"pol-focus-{self.seed}-{ep}-{salt}").choice(list(POLICY_CHANGES))
        self.sched[ep] = tid
        self.instances[ep] = self._instance(ep, tid, salt)

    def seed_entries(self) -> list[tuple[str, str, float]]:
        docs: list[tuple[str, str, float]] = []
        for tid, tp in POLICY_TOPICS.items():
            kind = self.corpus[tid]
            if kind in (CORPUS_VERSIONED, CORPUS_LAGS):
                docs.append(_chunk(tid, tp, "old_ver", tp["old"]["value"]))
            if kind in (CORPUS_VERSIONED, CORPUS_CURRENT):
                docs.append(_chunk(tid, tp, "cur_ver", tp["current"]["value"]))
        # store order carries no recency signal
        random.Random(f"policyqa-docs-{self.seed}").shuffle(docs)
        out = [("policy-hub-readme", "Policy Hub: Acme policy and product documents are published per owning team "
                                     "(billing, security, platform, legal, support). Answer from the latest version "
                                     "of a document; if a question is not covered, check with the owning team before "
                                     "replying.", 0.9)]
        return out + docs + self.distractor_entries()

    def task(self, ep: int) -> Task:
        i = self.instances[ep]
        prompt = (f"Question POL-{4400 + ep} from {i['asker']} in #ask-policy:\n\"{i['question']}\"\n"
                  f"Account context: {i['customer']} <{i['email']}>, {i['company']} ({i['plan']} plan, {i['country']}).")
        return Task(ep, prompt, dict(i), self.agent_for(ep),
                    {"topic": i["topic"], "class": i["topic"], "corpus": i["corpus"],
                     "quirk": i["fix"] != i["generic"], **change_tags(self, ep, i["change_class"])})

    @staticmethod
    def _source_version(tp: dict[str, Any], src: str) -> str:
        low = src.lower()
        if not low or low == "none":
            return "none"
        if tp["old_ver"].lower() in low or _slug(tp["old_ver"]) in low:
            return "old"
        if tp["cur_ver"].lower() in low or _slug(tp["cur_ver"]) in low:
            return "current"
        return "other"

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        if name == "lookup_account":
            # Realistic but non-diagnostic: the account view never carries a policy value.
            irng = random.Random(t["email"])
            return StepResult(f"Account {t['email']}: {t['company']}, {t['plan']} plan, billing country {t['country']}, "
                              f"{irng.choice(SAFE_COUNTS)} active users, workspace admin {t['customer']}, renews in "
                              f"{irng.choice(MONTHS)}, {irng.choice(['no', 'one', 'two'])} open tickets, health "
                              f"{irng.choice(['green', 'amber'])}.")
        if name != "answer":
            return StepResult(f"Unknown tool {name}")
        tp = POLICY_TOPICS[t["topic"]]
        ans = str(args.get("answer", "")).strip()
        src = str(args.get("source", "") or "")
        truth_spec = POLICY_CHANGES[t["topic"]]["value"] if t.get("old_fix") else tp["current"]
        hit_truth = _match(truth_spec, ans)
        hit_old_doc = _match(tp["old"], ans)
        hit_generic = _match(tp["generic"], ans)
        hit_old_fix = bool(t.get("old_fix")) and _match(tp["current"], ans)
        abstained = bool(ABSTAIN.search(ans))
        flags = {
            # after a regime change "stale" means the pre-change value the docs still carry;
            # before it, the superseded document version
            "stale_pick": hit_old_fix if t.get("old_fix") else hit_old_doc,
            "old_doc_pick": hit_old_doc, "generic_pick": hit_generic,
            "topic_wrong": str(args.get("topic", "")) != t["topic"], "abstained": abstained,
            "source_version": self._source_version(tp, src),
        }
        if hit_truth and not (hit_old_doc or hit_generic or hit_old_fix):
            return StepResult("Colleague: 'Thanks, confirmed with the customer.'", terminal=True, success=True,
                              flags=flags, answer=ans[:120])
        wrong_value = hit_old_doc or hit_generic or hit_old_fix
        if hit_old_fix or hit_old_doc:
            fb = ("The colleague came back: the customer's account manager says that figure is from the old policy; "
                  "ticket reopened.")
        elif hit_generic:
            fb = "Compliance flagged the reply: not Acme's policy."
        elif abstained:
            fb = "Colleague: 'OK, I'll ask the policy owner directly.' Question closed unanswered."
        else:
            fb = "Colleague replied: 'That's not what billing told the customer.'"
        sev = "minor_failure" if abstained and not wrong_value else "failure"
        return StepResult(fb, terminal=True, success=False, severity=sev, flags=flags, answer=ans[:120])
