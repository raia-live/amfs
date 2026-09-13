"""What a commit is told about the memory it did not draw on.

Storing a memory is only half a loop. The other half is using one, and nothing in
a session ever told an agent it had skipped something: it finished the task, sealed
the outcome, and the four entries that would have answered the question sat
unread. ``POST /api/v1/outcomes`` is the one moment that can still be said out
loud — the request is in the body, so what *would* have matched is computable, and
the trace already knows which entries were linked to the outcome.

Two things about this are easy to get wrong, and both are what these tests are
mostly about.

**A measurement must not move the thing it measures.** The obvious way to find the
matching memories is to retrieve them, and the obvious retrieve is the one the
product already ships — which bumps ``recall_count`` on its top hit and costs a
metered operation. Run that inside a commit and the diagnostic credits reuse that
never happened, to an entry the agent never saw, in the same counter that reports
reuse; ``effective_confidence`` then intends to *rank* on that counter.
``AgentMemory.explain`` refused this trap already, serving from read-time snapshots
rather than re-reading, and said so in a comment. So the gap report gathers
candidates in-process and stops before step 10 of the retrieve pipeline.

**The report must not accuse the innocent.** ``causal_entry_keys`` holds entries
that were explicitly recorded as read — direct reads, and retrieve's top hit only.
A briefing records none. A bare search records none. So "matched but not linked" is
emphatically not "ignored", and a report that phrases it that way would tell the
best-behaved session in the product — one that opened with a briefing and used a
dozen entries well — that it read nothing. It reports linkage, which is provable,
and names both readings so the agent can tell which is its own.
"""

from __future__ import annotations

import asyncio
import types
from datetime import UTC, datetime, timedelta

import pytest
from amfs_core.models import MemoryEntry, Provenance
from amfs_http import server
from amfs_http.models import OutcomeRequest

FLOOR = 0.15


def _entry(entity_path: str, key: str, value: str = "x", *, confidence: float = 0.9) -> MemoryEntry:
    return MemoryEntry(
        entity_path=entity_path,
        key=key,
        value=value,
        confidence=confidence,
        provenance=Provenance(
            agent_id="a",
            session_id="s",
            written_at=datetime.now(UTC) - timedelta(days=1),
        ),
    )


class _Adapter:
    """Semantic and lexical hits on demand, and a ledger of every recall credited.

    The recall ledger is the point: it stays empty for the whole of a commit, and
    a test that only checked the response could not tell.
    """

    _has_is_artifact_col = True

    def __init__(self, semantic=None, lexical=None, *, fail=False):
        self._semantic = semantic or []
        self._lexical = lexical or []
        self._fail = fail
        self.recall_bumps: list[tuple[str, str]] = []
        self.embedded: list[str] = []
        self.lexical_queries: list[str] = []

    async def semantic_search(self, query, embedder, *, branch="main"):
        if self._fail:
            raise RuntimeError("pgvector is having a day")
        self.embedded.append(query.text)
        return list(self._semantic)

    async def search(self, query, *, branch="main"):
        if self._fail:
            raise RuntimeError("full-text index is having a day")
        self.lexical_queries.append(query.query or "")
        return list(self._lexical)

    async def increment_recall_count(self, entity_path, key, *, branch="main"):
        self.recall_bumps.append((entity_path, key))


class _Vis:
    def __init__(self, allowed):
        self._allowed = set(allowed)

    def should_filter(self):
        return True

    def filter_entries(self, entries):
        return [e for e in entries if e.entry_key in self._allowed]


class _Handle:
    """The server's shared memory handle, reduced to what the route touches."""

    def __init__(self) -> None:
        self._tagger = types.SimpleNamespace(agent_id="server")
        self.namespace = "test"
        self._adapter = types.SimpleNamespace(
            ensure_agent=lambda *a, **k: None, save_trace=lambda t: t
        )

    def commit_outcome(self, *a, **k):
        return []


@pytest.fixture(autouse=True)
def _server(monkeypatch):
    monkeypatch.setattr(server, "_get_memory", lambda: _Handle())
    monkeypatch.setattr(server, "_get_server_embedder", lambda: object())
    monkeypatch.setattr(server, "_audit_log", lambda *a, **k: None)
    monkeypatch.setattr(server, "_link_agent_owner_once", lambda *a, **k: None)
    monkeypatch.setattr(server, "_retrieve_min_semantic", lambda: FLOOR)


def _commit(adapter, monkeypatch, *, task_input="roll the api back to v41", read=None, vis=None):
    """Drive a real commit through the route and hand back its response."""
    monkeypatch.setattr(server, "_async_adapter", adapter, raising=False)
    request = types.SimpleNamespace(
        client=types.SimpleNamespace(host="10.0.0.1"),
        state=types.SimpleNamespace(visibility_filter=vis),
    )
    return asyncio.run(
        server.commit_outcome(
            OutcomeRequest(
                outcome_ref="deploy-142",
                outcome_type="success",
                task_input=task_input,
                causal_entry_keys=list(read or []),
                # The caller's own trace follows, so the seal path stays out of
                # this file — it has its own tests and needs the Pro package.
                trace_follows=True,
            ),
            request,
            None,
        )
    )


class TestTheMeasurementDoesNotMoveTheThingItMeasures:
    def test_no_recall_is_credited_while_reporting_the_gap(self, monkeypatch):
        """The whole reason this is computed here and not through retrieve.

        A retrieve would bump ``recall_count`` on its top hit. Doing that from
        inside a commit credits reuse to an entry the agent never opened, in the
        counter that exists to report reuse — and ``effective_confidence`` plans
        to rank on it, so the error would not stay cosmetic.
        """
        adapter = _Adapter(semantic=[(_entry("api/deploy", "rollback-runbook"), 0.81)])
        result = _commit(adapter, monkeypatch)

        assert result["memory_gap"]["matched"] == 1
        assert adapter.recall_bumps == []

    def test_a_novel_is_not_embedded_whole(self, monkeypatch):
        """``task_input`` is capped at 200k, and embedding that to count matches
        would cost more than the commit it rides on."""
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        _commit(adapter, monkeypatch, task_input="rollback " * 60_000)

        assert adapter.embedded, "the semantic channel should still have run"
        assert len(adapter.embedded[0]) <= server._GAP_QUERY_CHARS
        assert len(adapter.lexical_queries[0]) <= server._GAP_QUERY_CHARS


class TestTheCommitOutranksTheReport:
    def test_a_broken_report_does_not_cost_the_commit(self, monkeypatch):
        """Both channels fail. The outcome is still sealed and still answered.

        This is the same order of priorities that put the loose annotation on the
        MCP ``actions`` parameter: the commit is the valuable half, and it must
        never be lost to a defect in the reporting half.
        """
        result = _commit(_Adapter(fail=True), monkeypatch)

        assert result["outcome_ref"] == "deploy-142"
        assert "memory_gap" not in result

    def test_a_report_that_raises_outright_does_not_cost_the_commit(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("gap reporting is broken in a new way")

        monkeypatch.setattr(server, "_memory_gap", _boom)
        result = _commit(_Adapter(), monkeypatch)

        assert result["outcome_ref"] == "deploy-142"
        assert "memory_gap" not in result


class TestWhenThereIsNothingWorthSaying:
    def test_a_task_that_matches_nothing_reports_nothing(self, monkeypatch):
        """Quiet rather than a zero: an empty account would otherwise be told at
        every commit that none of its no memories were used."""
        result = _commit(_Adapter(), monkeypatch)
        assert "memory_gap" not in result

    def test_a_commit_with_no_request_reports_nothing(self, monkeypatch):
        """Nothing to match against, so nothing to say — and notably not a claim
        that the session ignored its memory."""
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        result = _commit(adapter, monkeypatch, task_input="")

        assert "memory_gap" not in result
        assert adapter.embedded == []

    def test_everything_matched_was_used_so_nothing_is_listed(self, monkeypatch):
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        result = _commit(adapter, monkeypatch, read=["api/deploy/runbook"])

        gap = result["memory_gap"]
        assert gap == {"matched": 1, "linked_to_outcome": 1}
        assert "note" not in gap


class TestTheReportDoesNotAccuse:
    """The failure mode that would make this worse than shipping nothing.

    A briefing records no reads, and neither does a bare search. So the session
    most likely to show matched-but-unlinked entries is one that opened with a
    briefing and used them well. Telling that session it ignored its memory would
    be wrong on the one surface whose job is to show the product working.
    """

    @pytest.mark.parametrize(
        "forbidden",
        ["none were read", "you did not read", "ignored", "unread", "failed to"],
    )
    def test_the_note_never_claims_the_memory_went_unread(self, monkeypatch, forbidden):
        adapter = _Adapter(
            semantic=[
                (_entry("api/deploy", "runbook"), 0.9),
                (_entry("api/deploy", "postmortem"), 0.7),
            ]
        )
        note = _commit(adapter, monkeypatch)["memory_gap"]["note"]
        assert forbidden not in note.lower()

    def test_the_note_offers_the_briefing_reading_too(self, monkeypatch):
        """It says what to do in both cases, because from here the two are
        indistinguishable and only the agent knows which happened."""
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        note = _commit(adapter, monkeypatch)["memory_gap"]["note"]

        assert "briefing" in note.lower()
        assert "amfs_retrieve" in note

    def test_it_reports_linkage_rather_than_reading(self, monkeypatch):
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        gap = _commit(adapter, monkeypatch)["memory_gap"]

        assert "linked_to_outcome" in gap
        assert "read" not in gap


class TestWhatCountsAsMatched:
    def test_the_bar_is_retrieves_own_abstain_rule(self, monkeypatch):
        """A number the agent cannot reproduce is worse than no number.

        The agent can check this claim by running retrieve on the same text, so
        the relevance predicate here is retrieve's: a real semantic hit, or a
        lexical one. An entry below the floor with no keyword match is not a
        match, because retrieve would not have shown it.
        """
        adapter = _Adapter(
            semantic=[
                (_entry("api/deploy", "runbook"), FLOOR + 0.01),
                (_entry("api/trivia", "unrelated"), FLOOR - 0.01),
            ]
        )
        gap = _commit(adapter, monkeypatch)["memory_gap"]

        assert gap["matched"] == 1
        assert gap["unlinked_sample"] == ["api/deploy/runbook"]

    def test_a_weak_semantic_hit_still_counts_if_the_words_match(self, monkeypatch):
        """Deliberately generous in the same way retrieve is: the lexical channel
        is what still finds the entries whose embedding the propagation trigger
        stripped, which are precisely the outcome-validated ones."""
        weak = _entry("api/deploy", "runbook")
        adapter = _Adapter(semantic=[(weak, FLOOR - 0.05)], lexical=[weak])
        gap = _commit(adapter, monkeypatch)["memory_gap"]

        assert gap["matched"] == 1

    def test_benchmark_and_system_scratch_stay_out_of_the_count(self, monkeypatch):
        adapter = _Adapter(
            semantic=[
                (_entry("api/deploy", "runbook"), 0.9),
                (_entry("_system/internals", "note"), 0.95),
                (_entry("bench/harness", "case"), 0.95),
            ]
        )
        gap = _commit(adapter, monkeypatch)["memory_gap"]

        assert gap["matched"] == 1
        assert gap["unlinked_sample"] == ["api/deploy/runbook"]

    def test_the_same_entry_from_both_channels_is_counted_once(self, monkeypatch):
        both = _entry("api/deploy", "runbook")
        adapter = _Adapter(semantic=[(both, 0.9)], lexical=[both])
        assert _commit(adapter, monkeypatch)["memory_gap"]["matched"] == 1

    def test_a_date_in_the_request_is_not_searched_for(self, monkeypatch):
        """Found in review, and it broke the reproducibility claim above.

        Retrieve splits temporal intent out and searches only the topical
        remainder, so a request phrased "like we did yesterday" searches for the
        rollback and treats the date as a recency signal. This counted matches
        against the raw text, putting "yesterday" into both the embedding and the
        keyword query as noise — so the number the note invites the agent to
        reproduce with retrieve was one retrieve would not produce.
        """
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        _commit(adapter, monkeypatch, task_input="roll the api back like we did yesterday")

        assert adapter.embedded, "the semantic channel should still have run"
        searched = adapter.embedded[0] + " " + adapter.lexical_queries[0]
        assert "yesterday" not in searched.lower()
        assert "roll the api back" in searched.lower(), "the topical half must survive"

    def test_a_request_that_is_only_a_date_still_searches(self, monkeypatch):
        """The edge of the fix above: stripping "yesterday" from "yesterday"
        leaves nothing. ``normalize_temporal`` falls back to the original text
        rather than returning empty, and the count must not silently become 0."""
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        gap = _commit(adapter, monkeypatch, task_input="yesterday")["memory_gap"]

        assert gap["matched"] == 1


class TestTheCountIsNotALeakPath:
    def test_a_lexical_only_hit_is_filtered_like_a_semantic_one(self, monkeypatch):
        """The bug this shape of code invites, and the reason the filter runs over
        the merged set rather than per channel.

        A count is a disclosure: "4 memories match this task" about entries the
        caller may not read is a fact about someone else's memory.
        """
        mine = _entry("api/deploy", "runbook")
        theirs = _entry("api/secrets", "someone-elses")
        adapter = _Adapter(semantic=[(mine, 0.9)], lexical=[theirs])
        gap = _commit(adapter, monkeypatch, vis=_Vis({mine.entry_key}))["memory_gap"]

        assert gap["matched"] == 1
        assert gap["unlinked_sample"] == ["api/deploy/runbook"]

    def test_a_filtered_semantic_hit_is_not_counted_either(self, monkeypatch):
        theirs = _entry("api/secrets", "someone-elses")
        adapter = _Adapter(semantic=[(theirs, 0.99)])
        result = _commit(adapter, monkeypatch, vis=_Vis(set()))

        assert "memory_gap" not in result


class TestWhatTheBlockSays:
    def test_the_used_ones_are_not_listed_as_unlinked(self, monkeypatch):
        adapter = _Adapter(
            semantic=[
                (_entry("api/deploy", "runbook"), 0.95),
                (_entry("api/deploy", "postmortem"), 0.85),
                (_entry("api/deploy", "oncall"), 0.75),
            ]
        )
        gap = _commit(adapter, monkeypatch, read=["api/deploy/runbook"])["memory_gap"]

        assert gap["matched"] == 3
        assert gap["linked_to_outcome"] == 1
        assert gap["unlinked"] == 2
        assert "api/deploy/runbook" not in gap["unlinked_sample"]
        assert gap["unlinked_sample"] == ["api/deploy/postmortem", "api/deploy/oncall"]

    def test_a_read_of_something_unrelated_does_not_count_as_linkage(self, monkeypatch):
        """Linkage is counted against what matched, not against read volume — an
        agent that read ten unrelated entries did not consult these."""
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        gap = _commit(adapter, monkeypatch, read=["billing/invoices/format"])["memory_gap"]

        assert gap["matched"] == 1
        assert gap["linked_to_outcome"] == 0

    def test_the_list_stays_a_sample(self, monkeypatch):
        adapter = _Adapter(
            semantic=[(_entry("api/deploy", f"entry-{n}"), 0.9 - n / 100) for n in range(12)]
        )
        gap = _commit(adapter, monkeypatch)["memory_gap"]

        assert gap["matched"] == 12
        assert gap["unlinked"] == 12
        assert len(gap["unlinked_sample"]) == server._GAP_SAMPLE

    def test_the_strongest_match_is_named_first(self, monkeypatch):
        adapter = _Adapter(
            semantic=[
                (_entry("api/deploy", "weak"), 0.3),
                (_entry("api/deploy", "strong"), 0.95),
            ]
        )
        gap = _commit(adapter, monkeypatch)["memory_gap"]
        assert gap["unlinked_sample"][0] == "api/deploy/strong"

    def test_one_match_is_described_in_the_singular(self, monkeypatch):
        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        note = _commit(adapter, monkeypatch)["memory_gap"]["note"]
        assert "1 stored memory match" in note

    def test_the_partial_case_says_how_many_were_linked(self, monkeypatch):
        adapter = _Adapter(
            semantic=[
                (_entry("api/deploy", "runbook"), 0.95),
                (_entry("api/deploy", "postmortem"), 0.85),
            ]
        )
        note = _commit(adapter, monkeypatch, read=["api/deploy/runbook"])["memory_gap"]["note"]

        assert "2 stored memories match" in note
        assert "1 are linked" in note
        assert "1 that are not" in note

    def test_the_note_does_not_pass_a_sample_off_as_the_whole_list(self, monkeypatch):
        """Found in review. The note used to say "the rest are listed in unlinked"
        beside a field capped at three, which reads as a complete list on any
        commit with more than three unmatched keys.

        A block whose only job is to be believed cannot round its own numbers, so
        the count and the sample are separate fields and the note says which it is
        showing.
        """
        adapter = _Adapter(
            semantic=[(_entry("api/deploy", f"entry-{n}"), 0.9 - n / 100) for n in range(9)]
        )
        gap = _commit(adapter, monkeypatch, read=["api/deploy/entry-0"])["memory_gap"]

        assert gap["unlinked"] == 8
        assert len(gap["unlinked_sample"]) == server._GAP_SAMPLE
        assert "8 that are not" in gap["note"]
        assert "the strongest are in unlinked_sample" in gap["note"]

    def test_a_short_unlinked_list_is_not_called_a_sample(self, monkeypatch):
        """The other half of the fix: when everything unlinked does fit, the note
        should say so rather than hedge about strength for no reason."""
        adapter = _Adapter(
            semantic=[
                (_entry("api/deploy", "runbook"), 0.95),
                (_entry("api/deploy", "postmortem"), 0.85),
            ]
        )
        note = _commit(adapter, monkeypatch, read=["api/deploy/runbook"])["memory_gap"]["note"]

        assert "they are in unlinked_sample" in note
        assert "strongest" not in note


class TestNoTimeOrMoneyFigureCreepsIn:
    """The value line already cost credibility once by dividing a token estimate
    into minutes, and a test blocks it there. This block is new surface for the
    same mistake, so it gets the same guard."""

    def test_the_block_reports_no_time_and_no_cost(self, monkeypatch):
        import json
        import re

        adapter = _Adapter(semantic=[(_entry("api/deploy", "runbook"), 0.9)])
        payload = json.dumps(_commit(adapter, monkeypatch)["memory_gap"])

        assert not re.search(r"\d+(\.\d+)?\s*(sec|min|hr|hour|minute)", payload, re.I)
        assert "$" not in payload
