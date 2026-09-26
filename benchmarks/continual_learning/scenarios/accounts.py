"""Multi-user personalisation: one account-management assistant serving many customers.

Personalising for one user (``concierge``) is the easy case: every preference in memory
belongs to the person asking. Real customer-success teams run one assistant across a whole
book of accounts, and the preferences overlap, collide and drift: two contacts share a first
name at different companies, two contacts at the same company want opposite things, and the
person who "only takes calls" changes role and moves to Slack. This is the home turf of
user-scoped memory products, so the test is deliberately fair to them: everything the agent
needs is exactly the kind of per-user fact such systems store.

  accounts    12 customer contacts with hidden preferences (a meeting window derived from
              their timezone, channel, report format and cadence, discount stance, renewal
              term, no-contact day, language, tone, escalation owner, follow-up timing).
              Each episode is one request kind for one contact with four lettered options
              and exactly one that fits. The contact's terse reply reveals one violated
              preference; a discount handed to a never-discount contact is critical.

Failure modes isolated: cross-user contamination (applying one contact's preference to a
similar-looking contact; every confusable episode carries a decoy option that would have
been right for the twin), stale preferences after a contact changes (four contacts change
one preference each at ``change_at``; the option that satisfied the old preference is the
``stale_pick``), and trusting a partly wrong seeded CRM profile (three profiles carry one
wrong fact, one of them the colleague's channel copied across; one note conflates the two
same-name contacts). Same discipline as ``realworld``: no episode repeats, the prompt never
shows a preference, the CRM lookup is non-diagnostic, feedback is what the contact would
say and never the answer key.
"""

from __future__ import annotations

import random
from typing import Any

from .base import Scenario, StepResult, Task, ToolSpec, finish_params
from .realworld import change_tags

KINDS = ["meeting", "report", "renewal", "outreach", "escalation"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DAY_NAME = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}
TZ_OFFSET: dict[str, float] = {"PT": -7, "ET": -4, "GMT": 0, "CET": 2, "IST": 5.5, "JST": 9}
WIN_STARTS = [8, 9, 10, 13, 14]            # local hour at which a contact's 3-hour meeting window opens
CHANNELS = ["email", "phone", "slack", "video"]
FORMATS = ["pdf", "csv", "dashboard-link"]
CADENCES = ["weekly", "monthly", "quarterly"]
TERMS = ["monthly", "annual", "multi-year"]
DISCOUNTS: list[Any] = ["never", 5, 10, 15, 20]   # "never" = wants features, not price; N = renews at N%
LANGS = {"en": "English", "de": "German", "fr": "French", "ja": "Japanese"}
TONES = ["formal", "casual"]
OWNERS = ["account manager", "support lead", "VP of customer success", "engineering on-call"]
FOLLOWUPS = ["same day", "next business day", "within the week"]
FEATURES = ["the SSO add-on", "the audit-log module", "priority support", "a sandbox environment"]
PLANS = ["starter", "pro", "enterprise"]
ARR_BAND = {"starter": "$5-20k", "pro": "$20-100k", "enterprise": "$100k+"}

EN_NAMES = ["Marco", "Elena", "Aisha", "Tomas", "Ravi", "Daniel", "Ines", "Omar", "Sofia", "Noah", "Mei", "Farah"]
TWIN_NAMES = ["Priya", "Alex", "Sam", "Jordan"]
LANG_NAMES = {"de": ["Klaus", "Lena", "Jonas"], "fr": ["Amelie", "Chloe", "Luc"], "ja": ["Hiroshi", "Yuki", "Kenji"]}
LANG_TZ = {"de": "CET", "fr": "CET", "ja": "JST"}
COMPANIES = ["Northwind Traders", "Globex", "Initech", "Vandelay Logistics", "Hooli", "Umbrella Health",
             "Wayne Analytics", "Soylent Foods", "Tyrell Robotics", "Wonka Retail", "Cyberdyne Systems",
             "Oscorp Labs", "Pied Piper", "Massive Dynamic"]

# Hidden preference dimensions each request kind exercises, in the order a violation is reported.
KIND_DIMS = {"meeting": ["no_contact_day", "window"], "report": ["language", "format", "cadence"],
             "renewal": ["discount", "term"], "outreach": ["language", "channel", "tone"],
             "escalation": ["channel", "owner", "followup"]}
DIMS = ["window", "channel", "format", "cadence", "discount", "term", "no_contact_day", "language", "tone",
        "owner", "followup"]
HARD_DIMS = {"language", "no_contact_day"}   # never-discount is hard too; it is critical (see step)
POOL: dict[str, list[Any]] = {"channel": CHANNELS, "format": FORMATS, "term": TERMS, "discount": DISCOUNTS,
                              "win_start": WIN_STARTS, "tz": list(TZ_OFFSET)}
CHANNEL_REJECT = {"phone": "I don't take calls", "email": "I don't read vendor email",
                  "slack": "I'm not in the shared Slack", "video": "no video calls"}
ASKS = {"meeting": ["wants to schedule a 30-minute check-in next week; pick one slot",
                    "asked for a quick sync on the rollout plan; pick one slot",
                    "needs a 30-minute call to walk through the renewal paperwork; pick one slot"],
        "report": ["asked for their usage report; pick the deliverable",
                   "wants the adoption numbers for a leadership review; pick the deliverable",
                   "requested the account health report; pick the deliverable"],
        "renewal": ["has a renewal coming up; pick the offer to send",
                    "renewal notice is due this week; pick the offer to send",
                    "asked what the renewal will look like; pick the offer to send"],
        "outreach": ["needs a heads-up about next week's maintenance window; pick how to reach out",
                     "should hear about the new SSO feature before the public announcement; pick how to reach out",
                     "needs to know their API keys rotate on the 1st; pick how to reach out"],
        "escalation": ["filed a complaint about yesterday's outage; pick how to handle it",
                       "is unhappy that a P1 ticket sat unanswered for two days; pick how to handle it",
                       "complained that last month's invoice was wrong; pick how to handle it"]}


def _hm(h: float) -> str:
    h %= 24
    return f"{int(h):02d}:{30 if h % 1 else 0:02d}"


def _a(word: str) -> str:
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


def _slug(name: str, company: str) -> str:
    return f"{name.lower()}-{company.split()[0].lower()}"


class AccountsScenario(Scenario):
    name = "accounts"
    agent_base = "accounts-agent"
    role = ("You are the account-management assistant for Acme's customer success team. For each request "
            "from a customer contact, pick the single option that fits that specific customer.")
    N = 12
    PROFILE_FACTS = 3   # preferences a seeded CRM profile mentions (of 11); the rest must be learned

    tools = [
        ToolSpec("crm_lookup", "Look up a customer contact in the CRM: company, plan, ARR band, last touch, open tickets.",
                 {"type": "object", "properties": {"contact": {"type": "string", "description": "name and/or company"}},
                  "required": ["contact"]}),
        ToolSpec("propose", "Propose exactly one option to the contact.",
                 finish_params({"contact": {"type": "string"},
                                "option": {"type": "string", "description": "the option letter, e.g. B"},
                                "rationale": {"type": "string"}}, ["contact", "option", "rationale"]), terminal=True),
    ]

    # ---- world -----------------------------------------------------------------------
    def build(self) -> None:
        rng = self.rng
        self.contacts = self._make_contacts(rng)
        self.twin_of = {0: 11, 11: 0, 1: 2, 2: 1}   # same name / same company pairs
        singles = rng.sample(range(3, self.N), 5)
        c1, c2 = self.contacts[1], self.contacts[2]
        # Partly wrong CRM profiles: the colleague's channel copied onto the other contact at the same
        # company, "flexible on discounts" for someone who never takes one, the wrong report format.
        self.contacts[singles[0]]["discount"] = "never"
        fmt_wrong = rng.choice([f for f in FORMATS if f != self.contacts[singles[1]]["format"]])
        self.wrong: dict[int, tuple[str, Any]] = {2: ("channel", c1["channel"]), singles[0]: ("discount", 10),
                                                  singles[1]: ("format", fmt_wrong)}
        # Regime change: four contacts change ONE preference each after ``change_at``.
        s_tz, s_disc, s_term = singles[2], singles[3], singles[4]
        self.contacts[s_disc]["discount"] = "never"
        if self.contacts[s_term]["term"] == "annual":
            self.contacts[s_term]["term"] = rng.choice(["monthly", "multi-year"])
        old_tz = self.contacts[s_tz]["tz"]
        changes_a: dict[int, dict[str, Any]] = {
            1: {"field": "channel", "dim": "channel", "why": "new role, moved off the old channel",
                "new": rng.choice([c for c in CHANNELS if c not in (c1["channel"], c2["channel"])])},
            s_tz: {"field": "tz", "dim": "window", "why": "relocated to another office",
                   "new": rng.choice([z for z in TZ_OFFSET if abs(TZ_OFFSET[z] - TZ_OFFSET[old_tz]) >= 3])},
            s_disc: {"field": "discount", "dim": "discount", "why": "budget cut; procurement now needs a discount",
                     "new": rng.choice([5, 10, 15])},
            s_term: {"field": "term", "dim": "term", "why": "finance standardised on annual terms", "new": "annual"},
        }
        # Both sets are built whatever the protocol (B from its own rng, so A's draws are unchanged);
        # ``changes`` (base) is the union of the sets the cell applies, ``active_changes(ep)`` what is in force.
        self._change_sets = {"A": changes_a, "B": self._changes_b(set(changes_a) | set(self.wrong))}
        # Seeded profile facts: PROFILE_FACTS per contact, always including the one that is wrong or
        # will change (so the stale phrase is in the store and ``_mentions_stale`` can see it served).
        self.profile: dict[int, dict[str, Any]] = {}
        for ci, c in enumerate(self.contacts):
            forced = self.wrong[ci][0] if ci in self.wrong else self.changes[ci]["dim"] if ci in self.changes else None
            dims = ([forced] if forced else []) + rng.sample([d for d in DIMS if d != forced], self.PROFILE_FACTS - bool(forced))
            facts = {d: (c["tz"], c["win_start"]) if d == "window" else c[d] for d in dims}
            if ci in self.wrong:
                facts[self.wrong[ci][0]] = self.wrong[ci][1]
            self.profile[ci] = facts
        # Schedule: rounds over all 12 contacts, the kind rotating per contact, so 60 episodes cover every
        # (contact, kind) exactly once; the four confusable contacts all land in the first six episodes.
        # A contact with a wrong profile fact is first met on a request that exercises that fact.
        koff = [rng.randrange(len(KINDS)) for _ in range(self.N)]
        for ci, (dim, _) in self.wrong.items():
            koff[ci] = KINDS.index(rng.choice([k for k in KINDS if dim in KIND_DIMS[k]]))
        self.sched: list[tuple[int, str]] = []
        r = 0
        while len(self.sched) < self.episodes:
            order = list(range(self.N))
            rng.shuffle(order)
            if r == 0:
                conf = [ci for ci in order if ci in self.twin_of]
                rest = [ci for ci in order if ci not in self.twin_of]
                head = conf + rest[:2]
                rng.shuffle(head)
                order = head + rest[2:]
            self.sched += [(ci, KINDS[(r + koff[ci]) % len(KINDS)]) for ci in order]
            r += 1
        self.sched = self.sched[:self.episodes]
        self.instances = [self._instance(ep, ci, kind, 0) for ep, (ci, kind) in enumerate(self.sched)]

    # Change set "B" (see Scenario.change_variant): four contacts "A" leaves alone (and whose CRM
    # profile is right) change one preference each on dimensions "A" never moves — report format,
    # reporting cadence, escalation owner and follow-up timing — so no (contact, dimension) pair
    # here changes under "A". Drawn from a separate rng after every "A" draw, so the contacts,
    # profiles and schedule of the "A" world are consumed identically.
    B_DIMS: dict[str, tuple[str, str]] = {
        "format": ("format", "switched BI tools; wants the report in another format now"),
        "cadence": ("cadence", "leadership review moved; wants a different reporting cadence"),
        "owner": ("owner", "reorg on the customer side; escalations go to a different owner"),
        "followup": ("followup", "after last quarter's outage they expect follow-up on a different clock"),
    }

    def _changes_b(self, taken: set[int]) -> dict[int, dict[str, Any]]:
        rng = random.Random(f"acc-changes-b-{self.seed}")
        pool = [ci for ci in range(self.N) if ci not in taken]
        picks = rng.sample(pool, len(self.B_DIMS))
        out: dict[int, dict[str, Any]] = {}
        for ci, (field, (dim, why)) in zip(picks, self.B_DIMS.items()):
            cur = self.contacts[ci][field]
            options = {"format": FORMATS, "cadence": CADENCES, "owner": OWNERS, "followup": FOLLOWUPS}[field]
            out[ci] = {"field": field, "dim": dim, "why": why, "new": rng.choice([v for v in options if v != cur])}
        return out

    def _make_contacts(self, rng: random.Random) -> list[dict[str, Any]]:
        twin = rng.choice(TWIN_NAMES)
        en = rng.sample(EN_NAMES, 7)
        comps = rng.sample(COMPANIES, 11)
        singles = [(en[k], "en", None) for k in range(2, 7)]
        singles += [(rng.choice(LANG_NAMES[lg]), lg, LANG_TZ[lg]) for lg in ("de", "fr", "ja")]
        rng.shuffle(singles)
        # 0 and 11 share a first name at different companies; 1 and 2 share a company; 3-10 are singletons
        specs = [(twin, comps[0], "en", None), (en[0], comps[2], "en", None), (en[1], comps[2], "en", None)]
        specs += [(n, comps[3 + k], lg, tz) for k, (n, lg, tz) in enumerate(singles)]
        specs.append((twin, comps[1], "en", "CET"))
        out = []
        for name, comp, lang, tz in specs:
            out.append({"slug": _slug(name, comp), "name": name, "company": comp, "plan": rng.choice(PLANS),
                        "tz": tz or rng.choice(list(TZ_OFFSET)), "win_start": rng.choice(WIN_STARTS),
                        "channel": rng.choice(CHANNELS), "format": rng.choice(FORMATS), "cadence": rng.choice(CADENCES),
                        "discount": rng.choice(DISCOUNTS), "term": rng.choice(TERMS),
                        "no_contact_day": rng.choice([None, None, "Mon", "Wed", "Fri"]), "language": lang,
                        "tone": rng.choice(TONES), "owner": rng.choice(OWNERS), "followup": rng.choice(FOLLOWUPS)})
        # the conflating seed note ("<name> prefers CET mornings") is true for contact 11 only
        out[11]["win_start"] = 9
        if out[0]["tz"] == "CET":
            out[0]["tz"] = rng.choice([z for z in TZ_OFFSET if z != "CET"])
        self._differ(rng, out[11], out[0], ["channel", "format", "term"])
        self._differ(rng, out[2], out[1], ["channel", "format", "term", "discount"])
        # twins must have meeting windows that differ in UTC (a decoy slot exists in both directions);
        # contact 11 keeps 09:00-12:00 CET, so its twin is the one redrawn
        for fixed, free, tzs in ((out[11], out[0], [z for z in TZ_OFFSET if z != "CET"]), (out[1], out[2], list(TZ_OFFSET))):
            tries = 0
            while not (self._utc_hours(fixed) - self._utc_hours(free) and self._utc_hours(free) - self._utc_hours(fixed)):
                free["win_start"] = rng.choice(WIN_STARTS)
                tries += 1
                if tries % 5 == 0:
                    free["tz"] = rng.choice(tzs)
        return out

    @staticmethod
    def _utc_hours(c: dict[str, Any]) -> set[int]:
        ws, off = c["win_start"], TZ_OFFSET[c["tz"]]
        return {h for h in range(24) if ws <= (h + off) % 24 < ws + 3}

    @staticmethod
    def _differ(rng: random.Random, c: dict[str, Any], other: dict[str, Any], fields: list[str]) -> None:
        for f in fields:
            if c[f] == other[f]:
                c[f] = rng.choice([v for v in POOL[f] if v != other[f]])

    def _current(self, ci: int, ep: int) -> dict[str, Any]:
        """The contact's preferences as they are at episode ``ep`` (regime change applied)."""
        c = self.contacts[ci]
        ch = self.active_changes(ep).get(ci)
        if ch:
            c = {**c, ch["field"]: ch["new"]}
        return c

    def change_set(self, variant: str) -> dict[int, dict[str, Any]]:
        return self._change_sets[variant]

    # ---- preferences -------------------------------------------------------------------
    # How the previous assistant phrased each preference in the CRM profile (also the stale phrase).
    FACT = {
        "window": lambda c: f"takes meetings {_hm(c['win_start'])}-{_hm(c['win_start'] + 3)} {c['tz']}",
        "channel": lambda c: f"prefers {c['channel']}",
        "format": lambda c: f"wants {c['format']} reports",
        "cadence": lambda c: f"{c['cadence']} reporting cadence",
        "discount": lambda c: ("never takes discounts, wants features instead" if c["discount"] == "never"
                               else f"flexible on discounts, usually renews at {c['discount']}% off"),
        "term": lambda c: f"on {_a(c['term'])} term",
        "no_contact_day": lambda c: (f"no contact on {DAY_NAME[c['no_contact_day']]}s" if c["no_contact_day"]
                                     else "reachable any weekday"),
        "language": lambda c: f"communicates in {LANGS[c['language']]}",
        "tone": lambda c: f"likes a {c['tone']} tone",
        "owner": lambda c: f"escalations go to the {c['owner']}",
        "followup": lambda c: f"expects follow-up {c['followup']}",
    }
    # The contact's objection when an option misses on a plain-equality dimension.
    OBJECT = {
        "language": lambda c, o: f"please write to me in {LANGS[c['language']]}",
        "format": lambda c, o: f"I want the {c['format']} version, not {o['format']}",
        "cadence": lambda c, o: f"we agreed on {c['cadence']} reporting, not {o['cadence']}",
        "term": lambda c, o: f"we're on {_a(c['term'])} term and staying there",
        "channel": lambda c, o: f"{CHANNEL_REJECT[o['channel']]}; {c['channel']} only",
        "tone": lambda c, o: "keep it professional, please" if c["tone"] == "formal" else "drop the formalities, we're past that",
        "owner": lambda c, o: f"loop in the {c['owner']}, not the {o['owner']}",
        "followup": lambda c, o: f"I expect follow-up {c['followup']}, not {o['followup']}",
    }

    @classmethod
    def _fact(cls, dim: str, c: dict[str, Any]) -> str:
        return cls.FACT[dim](c)

    @staticmethod
    def _with_facts(c: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
        p = dict(c)
        for d, v in facts.items():
            if d == "window":
                p["tz"], p["win_start"] = v
            else:
                p[d] = v
        return p

    @classmethod
    def _check(cls, dim: str, c: dict[str, Any], o: dict[str, Any]) -> str | None:
        """The contact's objection to option ``o`` on one dimension, or None if it fits."""
        if dim in cls.OBJECT:
            return cls.OBJECT[dim](c, o) if o[dim] != c[dim] else None
        if dim == "window":
            local, ws = (o["utc_hour"] + TZ_OFFSET[c["tz"]]) % 24, c["win_start"]
            if not ws <= local < ws + 3:
                return f"that's {_hm(local)} my time; I take meetings {_hm(ws)}-{_hm(ws + 3)} {c['tz']}"
        elif dim == "no_contact_day":
            if c["no_contact_day"] and o["day"] == c["no_contact_day"]:
                return f"I don't take meetings on {DAY_NAME[o['day']]}s"
        elif dim == "discount":
            want, d = c["discount"], o["discount"]
            if want == "never":
                return "we don't take discounts; we need the roadmap features instead" if d else None
            if not d:
                return f"features won't get this through procurement; we need our usual {want}%"
            if d < want:
                return f"procurement won't sign below {want}%"
            if d > want:   # over-discounting is caught by Acme finance, not by the contact
                return f"FINANCE:{c['name']} renews at {want}%; the extra {d - want} points are wasted margin"
        return None

    def _violations(self, c: dict[str, Any], kind: str, o: dict[str, Any],
                    dims: list[str] | None = None) -> list[tuple[str, str]]:
        out = []
        for d in (KIND_DIMS[kind] if dims is None else dims):
            msg = self._check(d, c, o)
            if msg:
                out.append((d, msg))
        return out

    def _implied(self, facts: dict[str, Any], kind: str, options: list[dict[str, Any]], base: dict[str, Any]) -> list[str]:
        """Options consistent with a (partial, possibly wrong) profile: what a profile-follower would pick among."""
        dims = [d for d in KIND_DIMS[kind] if d in facts]
        p = self._with_facts(base, facts)
        return [o["id"] for o in options if not self._violations(p, kind, o, dims)]

    # ---- options -----------------------------------------------------------------------
    @staticmethod
    def _draw(kind: str, rng: random.Random) -> dict[str, Any]:
        langs = list(LANGS) + ["en", "en"]
        if kind == "meeting":
            return {"day": rng.choice(DAYS), "utc_hour": rng.randrange(24)}
        if kind == "report":
            return {"format": rng.choice(FORMATS), "cadence": rng.choice(CADENCES), "language": rng.choice(langs)}
        if kind == "renewal":
            d = rng.choice([0, 0, 5, 10, 15, 20])
            return {"term": rng.choice(TERMS), "discount": d, "feature": None if d else rng.choice(FEATURES)}
        if kind == "outreach":
            return {"channel": rng.choice(CHANNELS), "tone": rng.choice(TONES), "language": rng.choice(langs)}
        return {"owner": rng.choice(OWNERS), "channel": rng.choice(CHANNELS), "followup": rng.choice(FOLLOWUPS)}

    @staticmethod
    def _fit(kind: str, c: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        """One option that satisfies contact ``c`` (free dimensions drawn at random)."""
        if kind == "meeting":
            ws, off = c["win_start"], TZ_OFFSET[c["tz"]]
            return {"day": rng.choice([d for d in DAYS if d != c["no_contact_day"]]),
                    "utc_hour": rng.choice([h for h in range(24) if ws <= (h + off) % 24 < ws + 3])}
        if kind == "report":
            return {"format": c["format"], "cadence": c["cadence"], "language": c["language"]}
        if kind == "renewal":
            never = c["discount"] == "never"
            return {"term": c["term"], "discount": 0 if never else c["discount"], "feature": rng.choice(FEATURES) if never else None}
        if kind == "outreach":
            return {"channel": c["channel"], "tone": c["tone"], "language": c["language"]}
        return {"owner": c["owner"], "channel": c["channel"], "followup": c["followup"]}

    @staticmethod
    def _render(kind: str, o: dict[str, Any]) -> str:
        if kind == "meeting":
            return f"{o['day']} {o['utc_hour']:02d}:00 UTC"
        if kind == "report":
            return f"{o['format']} report, {o['cadence']}, in {LANGS[o['language']]}"
        if kind == "renewal":
            return f"{o['term']} term, " + (f"{o['discount']}% discount" if o["discount"] else f"list price, {o['feature']} included")
        if kind == "outreach":
            return f"{o['channel']}, {o['tone']} tone, in {LANGS[o['language']]}"
        return f"bring in the {o['owner']} via {o['channel']}, follow-up {o['followup']}"

    def _gen(self, rng: random.Random, kind: str, cur: dict[str, Any], old: dict[str, Any] | None,
             twin: dict[str, Any] | None) -> tuple[list[dict[str, Any]], str, str | None, str | None]:
        """Four lettered options with exactly one valid for ``cur``. When the contact's preference just
        changed, exactly one option satisfies the OLD preference (the stale pick); when the contact has
        a confusable twin, at least one wrong option is valid for the twin (the decoy). The fitting
        options are drawn to fit and the rest at random, then the set is verified as a whole, so the
        result is the same as rejection sampling without the wait. The extra requirements are dropped
        one at a time if a draw cannot satisfy them."""
        for need_old, need_twin in ((bool(old), bool(twin)), (bool(old), False), (False, False)):
            for _ in range(400):
                opts = [self._fit(kind, cur, rng)]
                opts += [self._fit(kind, old, rng)] if need_old else []
                opts += [self._fit(kind, twin, rng)] if need_twin else []
                opts += [self._draw(kind, rng) for _ in range(4 - len(opts))]
                rng.shuffle(opts)
                opts = [{"id": "ABCD"[i], **o} for i, o in enumerate(opts)]
                ok = [o["id"] for o in opts if not self._violations(cur, kind, o)]
                if len(ok) != 1 or len({self._render(kind, o) for o in opts}) < 4:
                    continue
                good, stale, decoy = ok[0], None, None
                if need_old:
                    ok_old = [o["id"] for o in opts if not self._violations(old, kind, o)]
                    if len(ok_old) != 1 or ok_old[0] == good:
                        continue
                    stale = ok_old[0]
                if need_twin:
                    ok_twin = [o["id"] for o in opts if o["id"] != good and not self._violations(twin, kind, o)]
                    if not ok_twin:
                        continue
                    decoy = ok_twin[0]
                return opts, good, stale, decoy
        raise RuntimeError("accounts: could not draw an option set with exactly one valid option")

    def _instance(self, ep: int, ci: int, kind: str, salt: int) -> dict[str, Any]:
        rng = random.Random(f"acc-{self.seed}-{ep}" + (f"-{salt}" if salt else ""))
        base, cur = self.contacts[ci], self._current(ci, ep)
        ch = self.changes.get(ci)
        # the task is in the changed class only when the request kind exercises the changed preference
        exercises = bool(ch) and ch["dim"] in KIND_DIMS[kind]
        affected = exercises and self.class_changed(ci, ep)
        twin = self.twin_of.get(ci)
        opts, good, stale, decoy = self._gen(rng, kind, cur, base if affected else None,
                                             self._current(twin, ep) if twin is not None else None)
        return {"ci": ci, "contact": base["slug"], "name": base["name"], "company": base["company"], "kind": kind,
                "ask": rng.choice(ASKS[kind]), "options": opts, "good": good, "stale": stale, "twin_decoy": decoy,
                "old_fix": self._fact(ch["dim"], base) if affected else None, "change_class": affected,
                "change_cls": ci if exercises else None,
                "changed_contact": ci in self.changes, "confusable": twin is not None,
                "twin": self.contacts[twin]["slug"] if twin is not None else None,
                "wrong_dim": self.wrong[ci][0] if ci in self.wrong else None}

    def regenerate(self, ep: int, salt: int) -> None:
        kind = self.sched[ep][1]
        cands = [ci for ci, ch in (self.active_changes(ep) or self.changes).items() if ch["dim"] in KIND_DIMS[kind]]
        if not cands:   # no change touches this request kind; leave the episode alone
            return
        ci = random.Random(f"acc-focus-{self.seed}-{ep}-{salt}").choice(cands)
        self.sched[ep] = (ci, kind)
        self.instances[ep] = self._instance(ep, ci, kind, salt)

    # ---- scenario protocol -------------------------------------------------------------
    def seed_entries(self) -> list[tuple[str, str, float]]:
        out = []
        for ci, c in enumerate(self.contacts):
            facts = self.profile[ci]
            p = self._with_facts(c, facts)
            text = (f"CRM profile: {c['name']} ({c['company']}, {c['plan']} plan) — "
                    + "; ".join(self._fact(d, p) for d in facts) + ". Noted by the previous account assistant.")
            out.append((f"crm-profile-{c['slug']}", text, 0.8 if (ci + self.seed) % 2 else 0.7))
        tw = self.contacts[0]["name"]
        out.append((f"crm-note-{tw.lower()}-meetings", f"{tw} prefers CET mornings for calls; book 09:00-12:00 CET.", 0.7))
        return out + self.distractor_entries()

    def oracle_fix_entries(self, variant: str | None = None) -> list[tuple[str, str, float]]:
        out = []
        for ci, ch in self.change_set(variant or self.change_variant).items():
            base = self.contacts[ci]
            new = {**base, ch["field"]: ch["new"]}
            out.append((f"crm-update-{base['slug']}", f"CRM update — {base['name']} ({base['company']}): now "
                                                      f"{self._fact(ch['dim'], new)}; previously {self._fact(ch['dim'], base)}. "
                                                      f"Reason: {ch['why']}.", 1.0))
        return out

    def task(self, ep: int) -> Task:
        i = self.instances[ep]
        lines = "\n".join(f"  {o['id']}: {self._render(i['kind'], o)}" for o in i["options"])
        prompt = (f"Request from {i['name']} at {i['company']}: {i['ask']}.\nOptions:\n{lines}\n"
                  f"Propose one option for this contact.")
        implied = self._implied(self.profile[i["ci"]], i["kind"], i["options"], self.contacts[i["ci"]])
        return Task(ep, prompt, dict(i), self.agent_for(ep),
                    {"contact": i["contact"], "kind": i["kind"], "class": f"{i['contact']}:{i['kind']}",
                     "confusable": i["confusable"], "quirk": implied != [i["good"]],
                     "changed_contact": i["changed_contact"],
                     **change_tags(self, ep, i["change_class"], cls=i["change_cls"])})

    def _crm(self, q: str, ep: int) -> str:
        ql = q.lower()
        def hit(c: dict[str, Any], need_company: bool) -> bool:
            comp = c["company"].lower() in ql or c["company"].split()[0].lower() in ql
            return c["slug"] in ql or (c["name"].lower() in ql and (comp or not need_company)) or (comp and not need_company)
        hits = [c for c in self.contacts if hit(c, True)] or [c for c in self.contacts if hit(c, False)]
        if not hits:
            return f"CRM: no contact matches '{q}'."
        if len(hits) > 1:
            return ("CRM: multiple matches — " + "; ".join(f"{c['name']} ({c['company']}, {c['plan']})" for c in hits)
                    + ". Specify the company.")
        c = hits[0]
        r = random.Random(f"crm-{c['slug']}-{ep}")
        return (f"CRM: {c['name']} — {c['company']}; plan {c['plan']}; ARR band {ARR_BAND[c['plan']]}; "
                f"last touch {r.randint(2, 45)} days ago; open tickets {r.randint(0, 3)}.")

    def step(self, task: Task, name: str, args: dict[str, Any]) -> StepResult:
        t = task.truth
        ci, ep, kind = t["ci"], task.episode, t["kind"]
        if name == "crm_lookup":
            return StepResult(self._crm(str(args.get("contact", "")), ep))
        if name != "propose":
            return StepResult(f"Unknown tool {name}")
        cur = self._current(ci, ep)
        who = cur["name"]
        pick = str(args.get("option", "")).strip().upper()[:1]
        o = next((x for x in t["options"] if x["id"] == pick), None)
        if o is None:
            return StepResult(f"{who}: 'That's not one of the options.'", terminal=True, success=False,
                              severity="failure", answer=pick)
        # stale: the option satisfies the preference this contact had before the regime change
        stale = {"stale_pick": not self._violations(self.contacts[ci], kind, o)} if t.get("old_fix") else {}
        if pick == t["good"]:
            return StepResult(f"{who}: 'Works for me, go ahead.'", terminal=True, success=True, flags=stale, answer=pick)
        viol = self._violations(cur, kind, o)
        dims = [d for d, _ in viol]
        never_hit = kind == "renewal" and cur["discount"] == "never" and o["discount"] > 0
        sev = "critical_failure" if never_hit else "failure" if HARD_DIMS & set(dims) else "minor_failure"
        # cross-user contamination: would this option have been right for somebody else (the twin)?
        valid_for = [self.contacts[cj]["slug"] for cj in range(self.N)
                     if cj != ci and not self._violations(self._current(cj, ep), kind, o)]
        flags: dict[str, Any] = {"violation": viol[0][1], "violated_dims": dims, "cross_user": bool(valid_for),
                                 "cross_user_with": valid_for[:3], "cross_user_twin": t["twin"] in valid_for,
                                 "seed_wrong_followed": False, **stale}
        if t.get("wrong_dim") in dims:
            p = self._with_facts(cur, self.profile[ci])
            flags["seed_wrong_followed"] = self._check(t["wrong_dim"], p, o) is None
        if kind == "renewal":
            want = 0 if cur["discount"] == "never" else cur["discount"]
            if o["discount"] > want:
                flags["wasted_margin_pts"] = o["discount"] - want
        msg = viol[0][1]
        fb = f"Finance: '{msg[8:]}.'" if msg.startswith("FINANCE:") else f"{who}: 'No — {msg}.'"
        return StepResult(fb, terminal=True, success=False, severity=sev, flags=flags, answer=pick)
