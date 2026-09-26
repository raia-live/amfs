"""Real-world agent workloads, modelled on what teams deploy today.

Grounding (2026 surveys): customer service is the most common production agent use case
(LangChain State of Agent Engineering: 26.5%; Aldric mid-2026: 41% in production, 62% in
DigitalApplied's sample), followed by software engineering (38-53%), data & analytics /
text-to-SQL (29-34%), sales & marketing personalization (22-41%) and bounded operations
workflows. Every scenario here follows the same discipline as ``diagnose``: no episode
repeats; the domain has hidden quirks that differ from the generic answer; failure
feedback is what the real system would say, not the answer key.

  support     tier-1 customer-support resolution (product quirks beat generic playbooks)
  concierge   personalization for one user (hidden, drifting preferences; wrong seeded profile)
  retention   predicting user behaviour: choose the retention action per customer segment
  order-ops   ecommerce order exceptions under hidden fulfilment / carrier rules
  ci-fix      coding agent fixing failing CI under hidden repo conventions
  analytics   text-to-SQL style reporting over a warehouse with schema quirks
"""

from __future__ import annotations

import random
from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params

# ─────────────────────────────────────────────────────────────────── support ──

# Generic action vocabulary (nothing in the names points at a specific issue).
SUPPORT_ACTIONS = ["reset_password", "clear_app_data", "reinstall_app", "refund", "explain_and_close", "resend_email",
                   "escalate_tier2", "advise_settings_change", "update_payment_method", "advise_workaround"]
SUPPORT_ISSUES: dict[str, dict[str, Any]] = {
    # issue -> messages (paraphrases), generic fix, Acme fix, why. In 6 of 8 the Acme fix differs.
    "ios-login-loop": {"msgs": ["I keep getting bounced back to the login screen on my iPhone after entering my password.",
                                "iOS app: log in, spinner, back to login. Every time. Password is definitely right.",
                                "Can't get past login on the iPhone app, it just loops."],
                       "generic": "reset_password", "fix": "clear_app_data",
                       "why": "a stale keychain token; password resets and reinstalls do nothing"},
    "pending-auth": {"msgs": ["You charged me twice for my order! I see two $49 charges.",
                              "Two identical charges on my card for one order. Refund one now please.",
                              "Double billed. Fix it."],
                     "generic": "refund", "fix": "explain_and_close",
                     "why": "the second line is a pending authorization that drops off in 3-5 days"},
    "verification-email": {"msgs": ["Never got the verification email, checked spam.",
                                    "Signed up an hour ago, no confirmation email yet.",
                                    "The verify-your-email link never arrives."],
                           "generic": "resend_email", "fix": "resend_email", "why": "generic is right"},
    "export-timeout": {"msgs": ["CSV export of my transactions fails with a timeout every time.",
                                "Exporting all my data just spins and then errors.",
                                "Data export keeps failing on the big date range."],
                       "generic": "escalate_tier2", "fix": "advise_workaround",
                       "why": "exports over 90 days time out by design; splitting the range works"},
    "android-notifications": {"msgs": ["Push notifications stopped on my Android phone.",
                                       "No alerts on Android since last week.",
                                       "Android: notifications don't come through anymore."],
                              "generic": "reinstall_app", "fix": "advise_settings_change",
                              "why": "Android battery optimisation kills the background service"},
    "card-declined": {"msgs": ["My card keeps getting declined but it works everywhere else.",
                               "Payment declined at checkout, card is fine.",
                               "Can't pay, says declined."],
                      "generic": "update_payment_method", "fix": "escalate_tier2",
                      "why": "Acme's fraud rules block the card's issuer country; tier 2 must whitelist"},
    "2fa-lost-phone": {"msgs": ["Lost my phone and I'm locked out by 2FA.",
                                "New phone, no authenticator app, can't log in.",
                                "2FA codes go to a phone I don't have anymore."],
                       "generic": "reset_password", "fix": "escalate_tier2",
                       "why": "2FA recovery requires identity verification by tier 2; never bypass"},
    "webhook-late": {"msgs": ["Our webhooks arrive 10+ minutes late since yesterday.",
                              "Webhook delivery delayed massively.",
                              "Events show up in your dashboard but our endpoint gets them way later."],
                     "generic": "escalate_tier2", "fix": "explain_and_close",
                     "why": "their endpoint returned 5xx earlier, so deliveries are in exponential backoff"},
}
# Regime change (when ``change_at`` is set): Acme ships changes nobody tells the agent about.
# The old Acme fix stops working; the new one is different. 3 of 8 issue classes change. Two
# of the three change to a *new* quirk (the generic answer is still wrong, so an arm with no
# memory cannot win by default and every memory arm has to relearn); one changes to the
# textbook answer (the pure "unlearning" case, where forgetting is enough).
SUPPORT_CHANGES: dict[str, dict[str, str]] = {
    "card-declined": {"fix": "resend_email",
                      "why": "payments moved to a new provider that requires a one-time card re-verification; the "
                             "verification mail is what unblocks the card, not a new card and not tier 2"},
    "android-notifications": {"fix": "reinstall_app",
                              "why": "app v9 moved to a new notification channel; the settings advice no longer applies, a reinstall migrates it"},
    "export-timeout": {"fix": "advise_settings_change",
                       "why": "the export service now emails the file asynchronously; customers must enable export "
                              "notifications in settings, neither splitting the range nor escalating helps"},
}
# Change set "B" (see Scenario.change_variant): three classes that "A" leaves alone, moved to
# fixes no "A" episode ever pairs them with — so a model that learned "A" from traces has
# never seen these (class, fix) pairs and can only get them from memory. Same design as "A":
# two quirk-to-quirk changes and one change to the textbook answer.
SUPPORT_CHANGES_B: dict[str, dict[str, str]] = {
    "ios-login-loop": {"fix": "reinstall_app",
                       "why": "iOS app v9 keeps the session token in a new keychain access group that clearing app data "
                              "does not touch; only a reinstall drops the stale token"},
    "webhook-late": {"fix": "advise_settings_change",
                     "why": "webhook delivery moved to per-endpoint concurrency limits; the customer raises the "
                            "endpoint's concurrency in dashboard settings, tier 2 cannot change it and backoff is not the cause"},
    "2fa-lost-phone": {"fix": "reset_password",
                       "why": "self-service 2FA recovery now runs inside the password-reset flow (email plus ID check); "
                              "tier 1 sends the reset, escalations bounce back"},
}
SUPPORT_CHANGE_SETS = {"A": SUPPORT_CHANGES, "B": SUPPORT_CHANGES_B}


class SupportScenario(Scenario):
    name = "support"
    agent_base = "support-agent"
    role = "You are Acme's tier-1 customer-support agent. Resolve the ticket in one action with a short reply."

    tools = [
        ToolSpec("lookup_account", "Look up the customer's account and recent activity.",
                 {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}),
        ToolSpec("resolve", "Apply one resolution action and send the reply.",
                 finish_params({"action": {"type": "string", "enum": SUPPORT_ACTIONS},
                                "reply": {"type": "string"}}, ["action", "reply"]), terminal=True),
    ]

    @property
    def changes(self) -> dict[str, dict[str, str]]:
        return SUPPORT_CHANGE_SETS[self.change_variant]

    def oracle_fix_entries(self) -> list[tuple[str, str, float]]:
        return [(f"update-{iid}", f"Playbook update — {iid.replace('-', ' ')} tickets: the correct resolution is now "
                                  f"{c['fix']} (it used to be {SUPPORT_ISSUES[iid]['fix']}). Reason: {c['why']}.", 1.0)
                for iid, c in self.changes.items()]

    def build(self) -> None:
        rng = self.rng
        ids = list(SUPPORT_ISSUES)
        self.sched = [ids[(ep * 3 + self.seed) % len(ids)] if ep < 2 * len(ids) else rng.choice(ids)
                      for ep in range(self.episodes)]
        self.instances = [self._instance(ep, iid, 0) for ep, iid in enumerate(self.sched)]

    def _instance(self, ep: int, iid: str, salt: int) -> dict[str, Any]:
        irng = random.Random(f"sup-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
        iss = SUPPORT_ISSUES[iid]
        fix, why, old = iss["fix"], iss["why"], None
        if self.changed(ep) and iid in self.changes:
            old, fix, why = fix, self.changes[iid]["fix"], self.changes[iid]["why"]
        return {"issue": iid, "msg": irng.choice(iss["msgs"]),
                # See Scenario.opaque_ids: a counted ticket number is the episode index.
                "ticket": irng.randint(10000, 99999) if self.opaque_ids else 4400 + ep,
                "email": f"user{irng.randint(1000, 9999)}@example.com",
                "plan": irng.choice(["free", "pro", "team"]),
                "platform": "iOS" if "ios" in iid else "Android" if "android" in iid else irng.choice(["web", "iOS", "Android"]),
                "fix": fix, "generic": iss["generic"], "why": why, "old_fix": old,
                "change_class": iid in self.changes}

    def regenerate(self, ep: int, salt: int) -> None:
        iid = random.Random(f"sup-focus-{self.seed}-{ep}-{salt}").choice(list(self.changes))
        self.sched[ep] = iid
        self.instances[ep] = self._instance(ep, iid, salt)

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [("playbook-tier1", "Tier-1 playbook: login problems -> reset_password; duplicate charges -> refund; "
                                  "missing emails -> resend_email; app misbehaviour -> reinstall_app; declined cards -> "
                                  "update_payment_method; anything unclear -> escalate_tier2.", 0.8),
               ("policy-refunds-support", "Refunds under $100 can be issued by tier 1 without approval.", 0.9)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        i = self.instances[ep]
        prompt = (f"Ticket SUP-{i['ticket']} from {i['email']} ({i['plan']} plan, {i['platform']}):\n\"{i['msg']}\"")
        return Task(ep, prompt, dict(i), self.agent_for(ep),
                    {"issue": i["issue"], "quirk": i["fix"] != i["generic"],
                     **change_tags(self, ep, i["change_class"])})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        if name == "lookup_account":
            # Realistic but non-diagnostic: the account view does not explain the issue.
            irng = random.Random(t["email"])
            return StepResult(f"Account {t['email']}: plan {t['plan']}, platform {t['platform']}, member since "
                              f"2025-{irng.randint(1, 12):02d}, last login {irng.randint(1, 30)}h ago, "
                              f"{irng.randint(0, 3)} prior tickets, payment method on file: card ending {irng.randint(1000, 9999)}.")
        if name != "resolve":
            return StepResult(f"Unknown tool {name}")
        act = str(args.get("action", ""))
        stale = {"stale_pick": act == t.get("old_fix")} if t.get("old_fix") else {}
        if act == t["fix"]:
            return StepResult("Customer replied: 'That fixed it, thanks!' Ticket closed with CSAT 5.", terminal=True,
                              success=True, flags={"generic_pick": act == t["generic"], **stale}, answer=act)
        sev = "critical_failure" if act == "refund" else "minor_failure" if act == "escalate_tier2" else "failure"
        fb = {"refund": "Refund issued. Finance flagged it: the charge was not a duplicate; the refund was unwarranted.",
              "escalate_tier2": "Escalated. Tier 2 bounced it back: 'tier-1 resolvable, please handle'."}.get(
            act, "Customer replied 24h later: 'Still happening.' Ticket reopened, CSAT 2.")
        return StepResult(fb, terminal=True, success=False, severity=sev,
                          flags={"generic_pick": act == t["generic"], "unwarranted_refund": act == "refund", **stale}, answer=act)


def change_tags(sc: Scenario, ep: int, in_change_class: bool) -> dict[str, Any]:
    """Tags for the regime-change analysis: is this task in a class whose truth changes, and
    are we before or after the change."""
    if not sc.change_at:
        return {}
    return {"change_class": in_change_class, "post_change": sc.changed(ep)}


# ───────────────────────────────────────────────────────────────── concierge ──

CUISINES = ["Italian", "Japanese sushi", "steakhouse", "Thai", "Mexican", "seafood grill", "vegan", "Indian", "French"]
GIFTS = ["whisky set", "noise-cancelling headphones", "cookbook", "wine subscription", "board game", "plant", "running watch"]
HOTELS = ["boutique downtown (no gym)", "business hotel with gym", "resort with spa and gym", "hostel", "airport hotel (gym)"]


# Regime change for ``concierge`` (when ``change_at`` is set): Dana's life changes and she tells
# nobody. Set "A": she goes vegetarian (the steakhouse stops being an option — this is the v7
# preference drift, moved from episode 12 to ``change_at``) and she swaps airlines (SkyJet fixed
# its lounge problem; Northwind lost her bag twice, so Northwind is out and SkyJet is fine).
# Set "B" (see Scenario.change_variant) moves the two classes "A" leaves alone: gifts (her
# clients' compliance rules now bar electronics; a good bottle is fine again) and hotels (a knee
# injury means no gym for a while, and the travel budget was cut to $250). Nothing in "B"
# touches meals or flights. ``CONCIERGE_CHANGES`` maps class -> (old_fix phrase, why).
CONCIERGE_CHANGES: dict[str, dict[str, tuple[str, str]]] = {
    "A": {"dinner": ("steakhouse", "Dana went vegetarian; steakhouses are out"),
          "lunch": ("steakhouse", "Dana went vegetarian; steakhouses are out"),
          "flight": ("SkyJet", "Dana now avoids Northwind (lost her bag twice); SkyJet fixed its lounge problem and is fine again")},
    "B": {"gift": ("alcohol", "Dana's clients' compliance rules now bar electronics gifts; alcohol gifts are fine again"),
          "hotel": ("gym", "Dana's knee injury means no gym for a while; the travel budget was cut to $250 a night")},
}


class ConciergeScenario(Scenario):
    name = "concierge"
    agent_base = "concierge-agent"
    role = "You are a personal concierge agent for one executive, Dana. Pick the single best option for each request."
    drift_at = 12   # legacy (no ``change_at``): the vegetarian drift happens here

    tools = [
        ToolSpec("propose", "Propose exactly one option to Dana.",
                 finish_params({"option": {"type": "string", "description": "the option id, e.g. B"},
                                "rationale": {"type": "string"}}, ["option", "rationale"]), terminal=True),
    ]

    # hidden preferences
    PREFS = {"no_seafood": True, "dinner_budget_pp": 90, "lunch_budget_pp": 40, "dinner_not_before": 19,
             "no_alcohol_gifts": True, "hotel_needs_gym": True, "avoid_airline": "SkyJet"}
    KINDS = ["dinner", "lunch", "gift", "hotel", "flight"]

    @property
    def changes(self) -> dict[str, tuple[str, str]]:
        return CONCIERGE_CHANGES[self.change_variant] if self.change_at else {}

    def rules(self, ep: int) -> dict[str, Any]:
        """Dana's preferences as they stand at episode ``ep``."""
        r = {"veg": False, "avoid_airline": self.PREFS["avoid_airline"], "no_alcohol": True, "no_electronics": False,
             "needs_gym": True, "hotel_budget": 320}
        if not self.change_at:
            r["veg"] = ep >= self.drift_at
        elif self.changed(ep) and self.change_variant == "A":
            r.update(veg=True, avoid_airline="Northwind")
        elif self.changed(ep) and self.change_variant == "B":
            r.update(no_alcohol=False, no_electronics=True, needs_gym=False, hotel_budget=250)
        return r

    def build(self) -> None:
        self.reqs = [self._request(ep, self.KINDS[(ep + self.seed) % len(self.KINDS)], 0) for ep in range(self.episodes)]

    def _request(self, ep: int, kind: str, salt: int) -> dict[str, Any]:
        irng = random.Random(f"con-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
        rules = self.rules(ep)
        options, good = self._gen(kind, irng, rules)
        return {"kind": kind, "options": options, "good": good, "veg": rules["veg"], "rules": rules,
                "change_class": kind in self.changes}

    def regenerate(self, ep: int, salt: int) -> None:
        kind = random.Random(f"con-focus-{self.seed}-{ep}-{salt}").choice(list(self.changes))
        self.reqs[ep] = self._request(ep, kind, salt)

    def oracle_fix_entries(self) -> list[tuple[str, str, float]]:
        seen: set[str] = set()
        out = []
        for kind, (_, why) in self.changes.items():
            if why in seen:
                continue
            seen.add(why)
            out.append((f"profile-update-{kind}", f"Profile update for Dana — {kind} requests: {why}. Earlier notes on this are out of date.", 1.0))
        return out

    def _violations(self, kind: str, o: dict[str, Any], rules: dict[str, Any] | bool) -> list[str]:
        if isinstance(rules, bool):   # legacy signature: ``veg`` flag only
            rules = {**self.rules(0), "veg": rules}
        v = []
        if kind in ("dinner", "lunch"):
            if "seafood" in o["cuisine"] or "sushi" in o["cuisine"]:
                v.append("Dana does not eat seafood")
            if rules["veg"] and o["cuisine"] in ("steakhouse",):
                v.append("Dana went vegetarian recently")
            cap = self.PREFS["dinner_budget_pp"] if kind == "dinner" else self.PREFS["lunch_budget_pp"]
            if o["price_pp"] > cap:
                v.append(f"over budget (${o['price_pp']}/person)")
            if kind == "dinner" and o["time"] < self.PREFS["dinner_not_before"]:
                v.append("too early for dinner")
        elif kind == "gift":
            if rules["no_alcohol"] and o["alcohol"]:
                v.append("no alcohol gifts")
            if rules["no_electronics"] and o["item"] in ("noise-cancelling headphones", "running watch"):
                v.append("no electronics gifts, Dana's clients' compliance rules")
            if o["price"] > 150:
                v.append("over the gift budget")
        elif kind == "hotel":
            if rules["needs_gym"] and not o["gym"]:
                v.append("needs a gym")
            if o["price"] > rules["hotel_budget"]:
                v.append("over the hotel budget")
        elif kind == "flight":
            if o["airline"] == rules["avoid_airline"]:
                v.append(f"never {rules['avoid_airline']} again")
            if o["depart"] < 7:
                v.append("no departures before 7am")
        return v

    def _gen(self, kind: str, rng: random.Random, rules: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        while True:
            opts = []
            for i in range(4):
                if kind in ("dinner", "lunch"):
                    o = {"cuisine": rng.choice(CUISINES), "price_pp": rng.choice([25, 35, 55, 75, 95, 140]),
                         "time": rng.choice([12, 13, 18, 19, 20]) if kind == "dinner" else rng.choice([12, 13])}
                elif kind == "gift":
                    g = rng.choice(GIFTS)
                    o = {"item": g, "price": rng.choice([40, 90, 140, 220]), "alcohol": g in ("whisky set", "wine subscription")}
                elif kind == "hotel":
                    h = rng.choice(HOTELS)
                    o = {"hotel": h, "price": rng.choice([180, 260, 340, 420]), "gym": "gym" in h and "no gym" not in h}
                else:
                    o = {"airline": rng.choice(["SkyJet", "Northwind", "Aero", "Pacifica"]), "depart": rng.choice([6, 8, 11, 15]),
                         "price": rng.choice([320, 450, 600])}
                o["id"] = "ABCD"[i]
                opts.append(o)
            ok = [o["id"] for o in opts if not self._violations(kind, o, rules)]
            if len(ok) == 1:
                return opts, ok[0]

    def seed_entries(self) -> list[tuple[str, str, float]]:
        # onboarding profile written by an assistant: partly wrong
        out = [("profile-dana-food", "Dana's profile (onboarding): loves Japanese sushi and seafood; budget is flexible for dinners.", 0.8),
               ("profile-dana-travel", "Dana's profile (onboarding): prefers early morning flights to maximise the day; any airline.", 0.8),
               ("profile-dana-gifts", "Dana's profile (onboarding): a whisky set is always a safe gift for Dana's clients.", 0.7)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        r = self.reqs[ep]
        lines = []
        for o in r["options"]:
            desc = ", ".join(f"{k}={v}" for k, v in o.items() if k != "id")
            lines.append(f"  {o['id']}: {desc}")
        prompt = f"Dana asks: please book a {r['kind']} for me. Options:\n" + "\n".join(lines) + "\nPropose one."
        affected = r["change_class"] and self.changed(ep)
        truth = {**r, "old_fix": self.changes[r["kind"]][0] if affected else None,
                 "why": self.changes[r["kind"]][1] if affected else None}
        return Task(ep, prompt, truth, self.agent_for(ep),
                    {"kind": r["kind"], "class": r["kind"], "post_drift": r["veg"], **change_tags(self, ep, r["change_class"])})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        if name != "propose":
            return StepResult(f"Unknown tool {name}")
        t = task.truth
        pick = str(args.get("option", "")).strip().upper()[:1]
        o = next((x for x in t["options"] if x["id"] == pick), None)
        # stale: the option Dana would have accepted before the change (only defined after one)
        stale = ({"stale_pick": o is not None and not self._violations(t["kind"], o, self.rules(0))}
                 if t.get("old_fix") else {})
        if pick == t["good"]:
            return StepResult("Dana: 'Perfect, book it.'", terminal=True, success=True, flags=stale, answer=pick)
        if o is None:
            return StepResult("Dana: 'That's not one of the options.'", terminal=True, success=False, severity="failure", answer=pick)
        why = self._violations(t["kind"], o, t["rules"])
        return StepResult(f"Dana: 'No — {why[0]}.'", terminal=True, success=False, severity="minor_failure",
                          flags={"violation": why[0], **stale}, answer=pick)


# ───────────────────────────────────────────────────────────────── retention ──

RET_ACTIONS = ["offer_discount", "schedule_success_call", "assign_csm", "feature_unlock", "no_action"]


class RetentionScenario(Scenario):
    name = "retention"
    agent_base = "retention-agent"
    role = ("You are the customer-retention agent at Acme SaaS. For each at-risk account pick the ONE action "
            "that keeps the customer without wasting spend.")

    tools = [
        ToolSpec("act", "Take one retention action for the account.",
                 finish_params({"account": {"type": "string"}, "action": {"type": "string", "enum": RET_ACTIONS},
                                "rationale": {"type": "string"}}, ["account", "action", "rationale"]), terminal=True),
    ]

    @staticmethod
    def truth_for(c: dict[str, Any], changed: bool = False, variant: str = "A") -> tuple[str, str]:
        """(action that retains without waste, explanation). ``changed``: after the regime
        change. Set "A": the CSM team was folded into Customer Success (assign_csm no longer
        exists as an outcome; a success call is what retains enterprise) and a pricing change
        made discounts ineffective for pro accounts (feature_unlock retains them). Set "B"
        (see Scenario.change_variant) moves the two segments "A" leaves alone: high-ticket
        accounts now get a named CSM under the support-to-success handoff (a one-off call no
        longer retains them), and the team tier bundles integrations by default, so a team
        account without them sits on the legacy price and a discount to the new list price is
        what retains it. Neither (segment, action) pair is a truth under "A"."""
        a, b = changed and variant == "A", changed and variant == "B"
        if c["plan"] == "enterprise":
            if a:
                return "schedule_success_call", "after the CS reorg there are no dedicated CSMs; a success call is what retains enterprise"
            return "assign_csm", "enterprise accounts churn on lack of ownership; discounts do not move them"
        if c["tickets_30d"] >= 4:
            if b:
                return "assign_csm", "under the support-to-success handoff high-ticket accounts get a named CSM; a one-off success call no longer retains them"
            return "schedule_success_call", "high ticket volume signals a product blocker, not a price problem"
        if c["usage_trend"] == "declining" and c["plan"] == "pro":
            if a:
                return "feature_unlock", "since the pricing change pro discounts no longer retain; unlocking features does"
            return "offer_discount", "price-sensitive pro accounts with declining usage respond to a discount"
        if c["usage_trend"] == "declining" and c["plan"] == "team" and not c["uses_integrations"]:
            if b:
                return "offer_discount", "the team tier now bundles integrations, so an account without them is on the legacy price above the new list; a discount to list price retains it"
            return "feature_unlock", "team accounts without integrations churn from low value, not price"
        return "no_action", "this account would stay anyway; any offer is wasted spend"

    RETENTION_SEGMENTS = {"A": ["enterprise accounts", "pro accounts with declining usage"],
                          "B": ["accounts with 4+ support tickets in 30 days", "team accounts with declining usage and no integrations"]}

    def oracle_fix_entries(self) -> list[tuple[str, str, float]]:
        probes = {"A": [{"plan": "enterprise", "tickets_30d": 0, "usage_trend": "stable", "uses_integrations": True},
                        {"plan": "pro", "tickets_30d": 0, "usage_trend": "declining", "uses_integrations": True}],
                  "B": [{"plan": "team", "tickets_30d": 4, "usage_trend": "stable", "uses_integrations": True},
                        {"plan": "team", "tickets_30d": 0, "usage_trend": "declining", "uses_integrations": False}]}
        out = []
        for seg, probe in zip(self.RETENTION_SEGMENTS[self.change_variant], probes[self.change_variant]):
            new, why = self.truth_for(probe, True, self.change_variant)
            old, _ = self.truth_for(probe, False, self.change_variant)
            out.append((f"update-retention-{seg.split()[0]}-{new}", f"Retention playbook update — {seg}: the action that "
                                                                    f"retains is now {new} (it used to be {old}). Reason: {why}.", 1.0))
        return out

    def build(self) -> None:
        self.accounts = [self._account(ep, 0) for ep in range(self.episodes)]

    def _account(self, ep: int, salt: int) -> dict[str, Any]:
        irng = random.Random(f"ret-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
        return {"id": f"acct-{irng.randint(10000, 99999)}", "plan": irng.choice(["pro", "team", "enterprise", "pro", "team"]),
                "tenure_months": irng.choice([2, 5, 9, 14, 26]),
                "usage_trend": irng.choice(["declining", "declining", "declining", "stable", "growing"]),
                "tickets_30d": irng.choice([0, 0, 1, 2, 4, 6]), "uses_integrations": irng.random() < 0.5,
                "mrr": irng.choice([49, 99, 299, 1200, 4000]), "nps": irng.choice([3, 5, 7, 8, 9])}

    def regenerate(self, ep: int, salt: int) -> None:
        self.accounts[ep] = self._account(ep, salt)

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [("retention-playbook", "Retention playbook: at-risk (declining usage or low NPS) -> offer_discount 20%; "
                                      "enterprise -> offer_discount plus call.", 0.8),
               ("retention-kpi", "Retention KPI: retained accounts per campaign; discount budget is capped monthly.", 0.9)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        c = self.accounts[ep]
        act, why = self.truth_for(c, self.changed(ep), self.change_variant)
        old, _ = self.truth_for(c, False, self.change_variant)
        in_class = old != self.truth_for(c, True, self.change_variant)[0]
        prompt = (f"Churn-risk alert for {c['id']}: plan={c['plan']}, MRR=${c['mrr']}, tenure={c['tenure_months']}mo, "
                  f"usage_trend={c['usage_trend']}, support tickets (30d)={c['tickets_30d']}, "
                  f"integrations={'yes' if c['uses_integrations'] else 'no'}, NPS={c['nps']}. Choose one action.")
        return Task(ep, prompt, {"account": c, "expected": act, "why": why,
                                 "old_fix": old if (self.changed(ep) and in_class) else None}, self.agent_for(ep),
                    {"expected": act, "would_stay": act == "no_action", **change_tags(self, ep, in_class)})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        if name != "act":
            return StepResult(f"Unknown tool {name}")
        t = task.truth
        act = str(args.get("action", ""))
        c = t["account"]
        if t.get("old_fix") and act == t["old_fix"]:
            return StepResult(f"60-day follow-up: account CHURNED despite {act.replace('_', ' ')} (${c['mrr']} MRR lost).",
                              terminal=True, success=False, severity="failure",
                              flags={"churned_mrr": c["mrr"], "wasted_spend_usd": 150, "stale_pick": True}, answer=act)
        if act == t["expected"]:
            return StepResult("60-day follow-up: account retained; finance confirms spend was justified.", terminal=True,
                              success=True, answer=act)
        if t["expected"] == "no_action":
            return StepResult(f"60-day follow-up: account retained, but finance flagged the {act.replace('_', ' ')} as "
                              f"unnecessary: usage was {c['usage_trend']}, this account was never leaving.", terminal=True,
                              success=False, severity="minor_failure",
                              flags={"wasted_spend_usd": round(c["mrr"] * 0.2 * 12) if act == "offer_discount" else 150}, answer=act)
        if act == "no_action":
            return StepResult(f"60-day follow-up: account CHURNED (${c['mrr']} MRR lost). No intervention was made.",
                              terminal=True, success=False, severity="critical_failure",
                              flags={"churned_mrr": c["mrr"]}, answer=act)
        return StepResult(f"60-day follow-up: account CHURNED despite {act.replace('_', ' ')} (${c['mrr']} MRR lost).",
                          terminal=True, success=False, severity="failure",
                          flags={"churned_mrr": c["mrr"], "wasted_spend_usd": 150}, answer=act)


# ───────────────────────────────────────────────────────────────── order-ops ──

OPS_ACTIONS = ["cancel_order", "intercept_shipment", "update_address", "carrier_redirect", "authorize_return",
               "issue_price_adjustment", "upgrade_shipping", "deny_explain"]


class OrderOpsScenario(Scenario):
    name = "order-ops"
    agent_base = "order-ops-agent"
    role = "You are the order-operations agent for Acme Shop. Handle each customer request with one OMS action."

    tools = [
        ToolSpec("get_order", "Fetch order status, carrier, items and dates.",
                 {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}),
        ToolSpec("apply", "Apply one OMS action.",
                 finish_params({"order_id": {"type": "string"}, "action": {"type": "string", "enum": OPS_ACTIONS},
                                "customer_message": {"type": "string"}}, ["order_id", "action", "customer_message"]),
                 terminal=True),
    ]

    @staticmethod
    def truth_for(o: dict[str, Any], changed: bool = False, variant: str = "A") -> str:
        """``changed``: after the regime change. Set "A": Acme's new WMS allows cancelling packed
        orders, the UPS contract lost in-transit changes (FedEx gained them), electronics get the
        common 30-day return window and the price-match window grew from 7 to 14 days. Set "B"
        (see Scenario.change_variant) moves the rules "A" leaves alone: packed orders can be
        re-addressed (label reprint at the pack station) and expedited (upgrade until the carrier
        scan), final-sale items became returnable within 14 days of delivery, and price match
        covers equivalent SKUs (colour and size variants), not only the identical one. Every
        (situation, action) pair "B" makes right is a denial under "A" in both regimes."""
        a, b = changed and variant == "A", changed and variant == "B"
        req, st, car = o["request"], o["status"], o["carrier"]
        redirect_ok = (car == "FedEx") if a else (car == "UPS")
        if req == "cancel":
            if st == "placed" or (st == "packed" and a):
                return "cancel_order"
            if st == "shipped":
                return "intercept_shipment" if redirect_ok else "deny_explain"
            return "deny_explain"  # packed cannot be cancelled at Acme (WMS lock); delivered -> return flow
        if req == "change_address":
            if st == "placed" or (st == "packed" and b):
                return "update_address"
            if st == "shipped":
                return "carrier_redirect" if redirect_ok else "deny_explain"
            return "deny_explain"
        if req == "return":
            # "A": one 30-day window for every category (electronics no longer 14)
            window = 30 if a else (14 if o["category"] == "electronics" else 30)
            if o["final_sale"]:
                # "B": final-sale items are returnable within 14 days of delivery
                return "authorize_return" if b and st == "delivered" and o["days_since_delivery"] <= 14 else "deny_explain"
            return "authorize_return" if st == "delivered" and o["days_since_delivery"] <= window else "deny_explain"
        if req == "price_match":
            # "A": price-match window extended from 7 to 14 days; "B": equivalent SKUs qualify
            return "issue_price_adjustment" if o["days_since_order"] <= (14 if a else 7) and (o["same_sku"] or b) else "deny_explain"
        if req == "expedite":
            return "upgrade_shipping" if st == "placed" or (st == "packed" and b) else "deny_explain"
        return "deny_explain"

    # What a maintainer who has been told about the change writes (``oracle_fix_entries``).
    OPS_CHANGE_NOTES = {
        "A": [("cancel-packed", "cancel requests: packed orders can now be cancelled (cancel_order) — the new WMS releases "
                                "the pick before the carrier scan. Previously packed meant deny_explain."),
              ("carrier-flip", "shipped orders: in-transit changes (intercept_shipment, carrier_redirect) now work for FedEx "
                               "shipments only; the UPS contract lost them, so UPS shipments are deny_explain. It used to be the reverse."),
              ("return-electronics", "return requests: electronics now get the common 30-day window (authorize_return up to 30 "
                                     "days after delivery); it used to be 14 days for electronics."),
              ("price-match-14", "price-match requests: the window is now 14 days from the order (issue_price_adjustment), not 7.")],
        "B": [("readdress-packed", "change-of-address requests: packed orders can now be re-addressed (update_address) — the pack "
                                   "station reprints the label before the carrier scan. Previously packed meant deny_explain."),
              ("expedite-packed", "expedite requests: shipping can be upgraded until the carrier scan, so packed orders now get "
                                  "upgrade_shipping instead of deny_explain."),
              ("return-final-sale", "return requests: final-sale items are now returnable within 14 days of delivery "
                                    "(authorize_return); they used to be deny_explain always."),
              ("price-match-equivalent", "price-match requests: equivalent SKUs (colour and size variants) now qualify for "
                                         "issue_price_adjustment within the 7-day window; it used to require the identical SKU.")],
    }

    def oracle_fix_entries(self) -> list[tuple[str, str, float]]:
        return [(f"update-ops-{key}", f"Order policy update — {text}", 1.0) for key, text in self.OPS_CHANGE_NOTES[self.change_variant]]

    def build(self) -> None:
        self.orders = [self._order(ep, 0) for ep in range(self.episodes)]

    def regenerate(self, ep: int, salt: int) -> None:
        self.orders[ep] = self._order(ep, salt)

    def _order(self, ep: int, salt: int) -> dict[str, Any]:
        reqs = ["cancel", "change_address", "return", "price_match", "expedite"]
        if True:
            irng = random.Random(f"ops-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
            req = reqs[(ep * 2 + self.seed + salt) % len(reqs)]
            # status distribution depends on the request so that eligible and ineligible cases are balanced
            status_pool = {"return": ["delivered"] * 4 + ["shipped"],
                           "cancel": ["placed", "placed", "packed", "shipped", "shipped"],
                           "change_address": ["placed", "placed", "packed", "shipped", "shipped"],
                           "price_match": ["delivered", "shipped", "placed"],
                           "expedite": ["placed", "placed", "packed", "shipped"]}[req]
            o = {"id": f"ORD-{irng.randint(100000, 999999)}", "request": req, "status": irng.choice(status_pool),
                 "carrier": irng.choice(["UPS", "FedEx"]), "category": irng.choice(["apparel", "electronics", "home"]),
                 "final_sale": irng.random() < 0.15, "days_since_delivery": irng.choice([2, 5, 10, 16, 28, 40]),
                 "days_since_order": irng.choice([1, 3, 5, 6, 9, 20]), "same_sku": irng.random() < 0.75,
                 "amount": irng.choice([39, 89, 149, 399, 899])}
            return o

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [("ops-policy-generic", "Order policy: orders can be cancelled or re-addressed until they ship; returns "
                                      "accepted within 30 days of delivery; price match within 7 days.", 0.8),
               ("ops-carriers", "Carriers: UPS and FedEx, both support in-transit changes via their portals.", 0.7)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        o = self.orders[ep]
        text = {"cancel": "I want to cancel this order.", "change_address": "I moved; please ship to my new address.",
                "return": "I'd like to return this.", "price_match": "This just went on sale elsewhere; can you match the price?",
                "expedite": "Can you make this arrive faster? I'll pay."}[o["request"]]
        prompt = f"Customer message about order {o['id']}: \"{text}\" Use get_order, then apply one action."
        exp = self.truth_for(o, self.changed(ep), self.change_variant)
        old, new = self.truth_for(o, False, self.change_variant), self.truth_for(o, True, self.change_variant)
        return Task(ep, prompt, {"order": o, "expected": exp, "old_fix": old if (self.changed(ep) and old != new) else None},
                    self.agent_for(ep), {"request": o["request"], "expected": exp, **change_tags(self, ep, old != new)})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        o = t["order"]
        if name == "get_order":
            return StepResult(f"{o['id']}: status={o['status']}, carrier={o['carrier']}, category={o['category']}, "
                              f"final_sale={o['final_sale']}, amount=${o['amount']}, days_since_order={o['days_since_order']}, "
                              f"days_since_delivery={o['days_since_delivery'] if o['status'] == 'delivered' else 'n/a'}, "
                              f"price_match_same_sku={o['same_sku']}")
        if name != "apply":
            return StepResult(f"Unknown tool {name}")
        act = str(args.get("action", ""))
        if act == t["expected"]:
            return StepResult("OMS: action accepted. Customer notified.", terminal=True, success=True, answer=act)
        reasons = {"cancel_order": "OMS rejected: order is locked by the warehouse once packed.",
                   "intercept_shipment": "Carrier rejected: intercept not supported for this shipment.",
                   "update_address": "OMS rejected: address changes not allowed at this status.",
                   "carrier_redirect": "Carrier rejected: redirect not supported by this carrier.",
                   "authorize_return": "OMS rejected: outside return eligibility for this item.",
                   "issue_price_adjustment": "Finance rejected: price match conditions not met.",
                   "upgrade_shipping": "OMS rejected: shipping cannot be upgraded at this status.",
                   "deny_explain": "Customer escalated to a supervisor, who found the request WAS eligible and approved it."}
        sev = "critical_failure" if act == "deny_explain" else "failure"
        return StepResult(reasons.get(act, "OMS rejected."), terminal=True, success=False, severity=sev,
                          flags={"wrong_denial": act == "deny_explain", "stale_pick": act == t.get("old_fix")}, answer=act)


# ──────────────────────────────────────────────────────────────────── ci-fix ──

CI_ACTIONS = ["run_formatter", "fix_code", "rerun_job", "update_snapshots", "regen_migrations", "bump_dependency",
              "add_audit_exception", "edit_generated_file"]
CI_FAILURES: dict[str, dict[str, Any]] = {
    "format": {"log": "black --check: would reformat 3 files (src/api/handlers.py, ...)", "fix": "run_formatter",
               "generic": "fix_code", "files": ["src/api/handlers.py"], "convention": "we never hand-edit formatting; run make fmt"},
    "flaky-integration": {"log": "FAILED tests/integration/test_billing_integration.py::test_webhook_roundtrip - TimeoutError",
                          "fix": "rerun_job", "generic": "fix_code", "files": ["src/billing/webhooks.py"],
                          "convention": "tests under tests/integration are known-flaky; rerun before touching code"},
    "type-error": {"log": "mypy: src/orders/service.py:88: error: Argument 1 has incompatible type 'str'; expected 'int'",
                   "fix": "fix_code", "generic": "fix_code", "files": ["src/orders/service.py"], "convention": "generic is right"},
    "snapshot-ui": {"log": "jest: 2 snapshots failed in web/components/Checkout.test.tsx", "fix": "update_snapshots",
                    "generic": "fix_code", "files": ["web/components/Checkout.tsx"],
                    "convention": "snapshot diffs on PRs that intentionally touch web/components are expected; update them"},
    "snapshot-nonui": {"log": "jest: 1 snapshot failed in web/components/Checkout.test.tsx", "fix": "fix_code",
                       "generic": "update_snapshots", "files": ["src/pricing/rules.py"],
                       "convention": "a UI snapshot changing on a backend-only PR is a real regression; never update"},
    "migration-drift": {"log": "alembic check: target database is not up to date; model changes not reflected in migrations",
                        "fix": "regen_migrations", "generic": "edit_generated_file", "files": ["src/models/order.py"],
                        "convention": "migrations are generated with make migrate; hand edits are rejected in review"},
    "audit-pinned": {"log": "pip-audit: requests==2.31.0 has CVE-2024-XXXX; fix version 2.32.0", "fix": "add_audit_exception",
                     "generic": "bump_dependency", "files": ["requirements.txt"],
                     "convention": "requests is pinned for the legacy SDK; bumping breaks it, use the audit exception process"},
    "audit-free": {"log": "pip-audit: urllib3==1.26.5 has CVE-2023-XXXX; fix version 1.26.18", "fix": "bump_dependency",
                   "generic": "bump_dependency", "files": ["requirements.txt"], "convention": "generic is right"},
}
# Regime change: the integration suite now replays recorded fixtures (a TimeoutError means the
# recorded cassette under tests/integration is stale and must be regenerated — reruns just
# fail again, and touching the code is wrong too), and the legacy SDK was retired (the requests
# pin is gone; reviewers now reject audit exceptions and expect the bump). One quirk-to-quirk
# change and one quirk-to-textbook change, deliberately.
CI_CHANGES: dict[str, str] = {"flaky-integration": "edit_generated_file", "audit-pinned": "bump_dependency"}
# Change set "B" (see Scenario.change_variant): two classes "A" leaves alone. The checkout
# components went under visual-regression review (a snapshot diff is now fixed in code; jest -u
# is rejected — quirk to textbook), and urllib3 got pinned for the new vendor SDK (the bump
# breaks it; the audit exception process applies — textbook to quirk, a new quirk appearing
# where the generic answer used to be right). Neither (class, fix) pair occurs under "A".
CI_CHANGES_B: dict[str, str] = {"snapshot-ui": "fix_code", "audit-free": "add_audit_exception"}
CI_CHANGE_SETS = {"A": CI_CHANGES, "B": CI_CHANGES_B}
# The reasons behind each change, as a maintainer would write them (``oracle_fix_entries``).
CI_CHANGE_WHY: dict[tuple[str, str], str] = {
    ("A", "flaky-integration"): "The integration suite replays recorded fixtures; a TimeoutError means the cassette under "
                                "tests/integration is stale and must be regenerated. Reruns fail again; code changes are wrong.",
    ("A", "audit-pinned"): "The legacy SDK was retired, so the requests pin is gone; reviewers reject audit exceptions and expect the bump.",
    ("B", "snapshot-ui"): "Checkout components are under visual-regression review: a snapshot diff is fixed in code; jest -u is rejected.",
    ("B", "audit-free"): "urllib3 is pinned for the new vendor SDK; the bump breaks it, so the audit exception process applies.",
}


class CiFixScenario(Scenario):
    name = "ci-fix"
    edit_tools = ("fix",)
    agent_base = "ci-agent"
    role = "You are a coding agent keeping CI green on the acme/platform repository. Fix the failing check with one action."

    tools = [
        ToolSpec("get_failure_log", "Read the failing CI job log.", {"type": "object", "properties": {"pr": {"type": "integer"}}, "required": ["pr"]}),
        ToolSpec("get_pr_files", "List files changed by the PR.", {"type": "object", "properties": {"pr": {"type": "integer"}}, "required": ["pr"]}),
        ToolSpec("fix", "Apply one fix action and re-run CI.",
                 finish_params({"pr": {"type": "integer"}, "action": {"type": "string", "enum": CI_ACTIONS},
                                "summary": {"type": "string"}}, ["pr", "action", "summary"]), terminal=True),
    ]

    def build(self) -> None:
        ids = list(CI_FAILURES)
        rng = self.rng
        # step 5 is coprime with len(ids)=8, so the first 16 episodes cover every failure type twice
        self.sched = [ids[(ep * 5 + self.seed) % len(ids)] if ep < 2 * len(ids) else rng.choice(ids) for ep in range(self.episodes)]

    @property
    def changes(self) -> dict[str, str]:
        return CI_CHANGE_SETS[self.change_variant]

    def oracle_fix_entries(self) -> list[tuple[str, str, float]]:
        return [(f"update-{fid}", f"CI convention update — {fid.replace('-', ' ')} failures ({CI_FAILURES[fid]['log'][:90]}): "
                                  f"the right fix is now {fix} (it used to be {CI_FAILURES[fid]['fix']}). "
                                  f"{CI_CHANGE_WHY.get((self.change_variant, fid), '')}".strip(), 1.0)
                for fid, fix in self.changes.items()]

    def regenerate(self, ep: int, salt: int) -> None:
        self.sched[ep] = random.Random(f"ci-focus-{self.seed}-{ep}-{salt}").choice(list(self.changes))

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [("repo-readme-ci", "CI: black, mypy, jest, alembic check, pip-audit run on every PR. Fix failures in the PR branch.", 0.8),
               ("repo-tip-snapshots", "Tip from a contributor: when jest snapshots fail, just run jest -u and commit.", 0.6)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        fid = self.sched[ep]
        f = CI_FAILURES[fid]
        # See Scenario.opaque_ids: a counted PR number is the episode index.
        pr = (random.Random(f"ci-pr-{self.seed}-{ep}").randint(1000, 9999) if self.opaque_ids
              else 3000 + ep * 7 + self.seed)
        prompt = f"CI is red on PR #{pr} (\"{fid.replace('-', ' ')} related change\"). Investigate and fix it."
        truth = {"pr": pr, "fid": fid, **f, "old_fix": None}
        if self.changed(ep) and fid in self.changes:
            truth["old_fix"], truth["fix"] = f["fix"], self.changes[fid]
        return Task(ep, prompt, truth, self.agent_for(ep),
                    {"failure": fid, "quirk": truth["fix"] != f["generic"], **change_tags(self, ep, fid in self.changes)})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        if name == "get_failure_log":
            return StepResult(t["log"])
        if name == "get_pr_files":
            return StepResult("\n".join(t["files"]))
        if name != "fix":
            return StepResult(f"Unknown tool {name}")
        act = str(args.get("action", ""))
        stale = {"stale_pick": act == t["old_fix"]} if t.get("old_fix") else {}
        if act == t["fix"]:
            return StepResult("CI green. Reviewer approved.", terminal=True, success=True,
                              flags={"generic_pick": act == t["generic"], **stale}, answer=act)
        if act in ("edit_generated_file", "update_snapshots", "add_audit_exception") and act != t["fix"]:
            return StepResult(f"CI green, but the reviewer REJECTED the change: '{act.replace('_', ' ')} is not how we do "
                              f"this here'. Changes requested.", terminal=True, success=False, severity="failure",
                              flags={"reviewer_reject": True, "generic_pick": act == t["generic"], **stale}, answer=act)
        return StepResult("CI still red after the fix. Same check failing.", terminal=True, success=False,
                          severity="failure", flags={"generic_pick": act == t["generic"], **stale}, answer=act)


# ───────────────────────────────────────────────────────────────── analytics ──

REGIONS = ["NA", "EU", "APAC"]
MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08"]


# Regime change (when ``change_at`` is set): finance redefines a metric and tells nobody; the
# warehouse is unchanged, so the only signal is "does not reconcile". Set "A": finance moves
# to *booked* figures — revenue, AOV and the refund-rate denominator are computed over orders
# with status complete or pending (test and soft-deleted rows still excluded). Set "B" (see
# Scenario.change_variant) moves the two kinds "A" leaves alone: order and customer counts
# become gross order volume and reach, i.e. every status including cancelled. Each redefinition
# moves the true number by well over the 1% reconciliation tolerance (pending is ~1/5 of orders,
# cancelled another ~1/5), so the old answer never passes by accident. ``old_fix`` is None for
# this family — the stale answer is a number, not an action; the ``stale_pick`` flag is set when
# the reported value reconciles with the pre-change definition.
ANALYTICS_CHANGES: dict[str, dict[str, str]] = {
    "A": {"revenue": "revenue is reported as booked revenue: amounts of orders with status complete OR pending (test and soft-deleted rows still excluded), not completed orders only",
          "aov": "average order value excludes add-on orders under $50 (amount_cents < 5000): the average is over completed orders of $50 or more",
          "refund_rate": "the refund rate is refunds as a percentage of booked orders (status complete or pending), not of completed orders"},
    "B": {"orders": "order counts are gross order volume: every status including cancelled and pending (test and soft-deleted rows still excluded)",
          "customers": "distinct customers are counted over every order regardless of status (reach), not completed orders only"},
}


class AnalyticsScenario(Scenario):
    """Text-to-SQL style reporting. The warehouse has quirks: amounts are in cents, the
    completed status is spelled 'complete', test accounts and soft-deleted rows must be
    excluded, refunds live in a separate table. The agent specifies a query; the engine
    executes it literally; finance compares the result with the true number."""

    name = "analytics"
    agent_base = "analytics-agent"
    role = "You are the analytics agent answering finance's questions from the orders warehouse."

    @property
    def changes(self) -> dict[str, str]:
        return ANALYTICS_CHANGES[self.change_variant]

    def oracle_fix_entries(self) -> list[tuple[str, str, float]]:
        return [(f"update-metric-{kind}", f"Finance definition update — {kind.replace('_', ' ')}: {why}. "
                                          f"Reports computed the old way no longer reconcile with the books.", 1.0)
                for kind, why in self.changes.items()]

    tools = [
        ToolSpec("describe_table", "Describe a table's columns.", {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}),
        ToolSpec("run_query", "Aggregate a table. Filters are simple 'column op value' strings.",
                 {"type": "object", "properties": {
                     "table": {"type": "string", "enum": ["orders", "refunds"]},
                     "metric": {"type": "string", "enum": ["sum", "count", "avg", "count_distinct"]},
                     "column": {"type": "string"},
                     "filters": {"type": "array", "items": {"type": "string"}, "description": "e.g. \"status = complete\", \"month = 2026-07\", \"is_test = false\", \"deleted_at is null\""}},
                  "required": ["table", "metric", "column", "filters"]}),
        ToolSpec("report", "Report the final number to finance.",
                 finish_params({"value": {"type": "number"}, "unit": {"type": "string", "enum": ["usd", "count"]},
                                "method": {"type": "string"}}, ["value", "unit", "method"]), terminal=True),
    ]

    def build(self) -> None:
        rng = random.Random(f"warehouse-{self.seed}")
        self.orders = []
        for i in range(2400):
            self.orders.append({"order_id": i, "customer_id": rng.randint(1, 600), "month": rng.choice(MONTHS),
                                "region": rng.choice(REGIONS), "status": rng.choice(["complete", "complete", "complete", "cancelled", "pending"]),
                                "amount_cents": rng.choice([1999, 4900, 12900, 25000, 89900]),
                                "is_test": rng.random() < 0.08, "deleted_at": (None if rng.random() < 0.95 else "2026-08-01")})
        self.refunds = [{"order_id": o["order_id"], "month": o["month"], "region": o["region"],
                         "amount_cents": o["amount_cents"], "is_test": o["is_test"]}
                        for o in self.orders if o["status"] == "complete" and rng.random() < 0.06]
        self.questions = []
        kinds = ["revenue", "orders", "aov", "customers", "refund_rate"]
        for ep in range(self.episodes):
            irng = random.Random(f"q-{self.seed}-{ep}")
            self.questions.append({"kind": kinds[(ep + self.seed) % len(kinds)], "month": irng.choice(MONTHS),
                                   "region": irng.choice(REGIONS + [None])})

    def regenerate(self, ep: int, salt: int) -> None:
        irng = random.Random(f"q-focus-{self.seed}-{ep}-{salt}")
        self.questions[ep] = {"kind": irng.choice(list(self.changes)), "month": irng.choice(MONTHS),
                              "region": irng.choice(REGIONS + [None])}

    # -- truth -------------------------------------------------------------------
    def _clean(self, month: str, region: str | None, statuses: tuple[str, ...] = ("complete",)) -> list[dict[str, Any]]:
        return [o for o in self.orders if o["status"] in statuses and not o["is_test"] and o["deleted_at"] is None
                and o["month"] == month and (region is None or o["region"] == region)]

    def truth(self, q: dict[str, Any], changed: bool = False) -> tuple[float, str]:
        """The number finance's books show for question ``q``; ``changed``: under the regime
        change's redefinition (set A or B per ``change_variant``), see ANALYTICS_CHANGES."""
        redefined = changed and q["kind"] in self.changes
        a, b = redefined and self.change_variant == "A", redefined and self.change_variant == "B"
        rows = self._clean(q["month"], q["region"])
        ref = [r for r in self.refunds if not r["is_test"] and r["month"] == q["month"] and (q["region"] is None or r["region"] == q["region"])]
        # "A": booked = complete or pending; "B": gross = every status
        base = self._clean(q["month"], q["region"], ("complete", "pending")) if a else \
            self._clean(q["month"], q["region"], ("complete", "pending", "cancelled")) if b else rows
        revenue = sum(o["amount_cents"] for o in base) / 100
        if q["kind"] == "revenue":
            return revenue, "usd"
        if q["kind"] == "orders":
            return float(len(base)), "count"
        if q["kind"] == "aov":
            # "A": add-on orders under $50 leave the average (a booked-orders average would barely
            # move: pending orders are drawn from the same amount distribution)
            big = [o for o in rows if o["amount_cents"] >= 5000] if a else rows
            return (sum(o["amount_cents"] for o in big) / 100) / max(len(big), 1), "usd"
        if q["kind"] == "customers":
            return float(len({o["customer_id"] for o in base})), "count"
        return 100.0 * len(ref) / max(len(base), 1), "count"  # percent

    # -- literal query engine -----------------------------------------------------
    def _filter(self, rows: list[dict[str, Any]], filters: list[str]) -> list[dict[str, Any]] | str:
        for f in filters:
            f = f.strip()
            low = f.lower()
            if low.endswith("is null") or low.endswith("is not null"):
                col = f.split()[0]
                want_null = low.endswith("is null")
                rows = [r for r in rows if (r.get(col) is None) == want_null]
                continue
            parts = f.split(None, 2)
            if len(parts) != 3:
                return f"bad filter: {f!r}"
            col, op, val = parts
            val = val.strip("'\"")
            if col not in rows[0] if rows else False:
                return f"unknown column {col!r}"
            def cv(v):
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return float(v)
                return str(v)
            def pv(v, sample):
                if isinstance(sample, bool):
                    return v.lower() in ("true", "1", "t")
                if isinstance(sample, (int, float)):
                    try:
                        return float(v)
                    except ValueError:
                        return v
                return v
            out = []
            for r in rows:
                a = cv(r.get(col)); b = pv(val, r.get(col))
                if op == "=" and a == b or op in ("!=", "<>") and a != b or op == "<" and a < b or op == ">" and a > b \
                        or op == "<=" and a <= b or op == ">=" and a >= b:
                    out.append(r)
            rows = out
        return rows

    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = [("warehouse-overview", "Warehouse: orders(order_id, customer_id, month, region, status, amount_cents, is_test, deleted_at); "
                                      "refunds(order_id, month, region, amount_cents, is_test). Revenue = sum of order amounts for completed orders.", 0.8),
               ("warehouse-tip-status", "Tip: completed orders have status = 'completed'.", 0.6)]
        return out + self.distractor_entries()

    def task(self, ep: int) -> Task:
        q = self.questions[ep]
        where = f" in {q['region']}" if q["region"] else ""
        text = {"revenue": f"What was recognised revenue (USD) for {q['month']}{where}?",
                "orders": f"How many completed orders in {q['month']}{where}?",
                "aov": f"What was the average order value (USD) for {q['month']}{where}?",
                "customers": f"How many distinct customers placed a completed order in {q['month']}{where}?",
                "refund_rate": f"What was the refund rate (% of completed orders refunded) for {q['month']}{where}?"}[q["kind"]]
        in_class = q["kind"] in self.changes
        val, unit = self.truth(q, self.changed(ep))
        old_val = self.truth(q, False)[0] if (self.changed(ep) and in_class) else None
        # See Scenario.opaque_ids: a counted request number is the episode index.
        fin = random.Random(f"fin-{self.seed}-{ep}").randint(1000, 9999) if self.opaque_ids else 800 + ep
        prompt = f"Finance asks (FIN-{fin}): {text} Use run_query, then report."
        return Task(ep, prompt, {"q": q, "expected": val, "unit": unit, "old_expected": old_val, "old_fix": None,
                                 "why": self.changes.get(q["kind"]) if in_class else None},
                    self.agent_for(ep), {"kind": q["kind"], "class": q["kind"], **change_tags(self, ep, in_class)})

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        if name == "describe_table":
            tbl = str(args.get("table", ""))
            if tbl == "orders":
                return StepResult("orders: order_id int, customer_id int, month text (YYYY-MM), region text, status text, "
                                  "amount_cents int, is_test bool, deleted_at text|null")
            if tbl == "refunds":
                return StepResult("refunds: order_id int, month text, region text, amount_cents int, is_test bool")
            return StepResult("unknown table")
        if name == "run_query":
            rows = self.orders if args.get("table") == "orders" else self.refunds
            res = self._filter(rows, [str(f) for f in args.get("filters", [])])
            if isinstance(res, str):
                return StepResult(f"query error: {res}")
            col = str(args.get("column", ""))
            m = args.get("metric")
            if m == "count":
                return StepResult(f"{len(res)}")
            if m == "count_distinct":
                return StepResult(f"{len({r.get(col) for r in res})}")
            vals = [r.get(col) for r in res if isinstance(r.get(col), (int, float))]
            if not vals:
                return StepResult("0")
            return StepResult(f"{sum(vals)}" if m == "sum" else f"{sum(vals) / len(vals):.4f}")
        if name != "report":
            return StepResult(f"Unknown tool {name}")
        try:
            v = float(args.get("value"))
        except (TypeError, ValueError):
            return StepResult("Finance: that is not a number.", terminal=True, success=False, severity="failure", answer=str(args))
        exp = t["expected"]
        rel = abs(v - exp) / max(abs(exp), 1e-9)
        # stale: the number reconciles with the pre-change definition (only defined after a change)
        old = t.get("old_expected")
        stale = {"stale_pick": abs(v - old) / max(abs(old), 1e-9) <= 0.01} if old is not None else {}
        if rel <= 0.01:
            return StepResult("Finance: matches our books. Thanks.", terminal=True, success=True,
                              flags={"rel_error": round(rel, 4), **stale}, answer=f"{v:.2f}")
        hint = "orders of magnitude off" if rel > 5 else "a few percent off" if rel < 0.15 else "materially off"
        return StepResult(f"Finance: that does not reconcile with our books ({hint}). Please re-check your query.",
                          terminal=True, success=False, severity="failure", flags={"rel_error": round(rel, 4), **stale},
                          answer=f"{v:.2f}")
