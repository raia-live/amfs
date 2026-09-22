"""Structured lessons (``amfs_core.lessons``): a claim on (situation, action,
worked) that the outcome record follows across rewrites of the text."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from amfs_core.evidence import inherit_evidence, is_training_excluded_key, same_claim
from amfs_core.lessons import (
    LESSON_KEY_PREFIX,
    lesson_claim,
    lesson_key,
    lesson_of,
    make_lesson,
    render_lesson,
)
from amfs_core.models import MemoryEntry, Provenance
from amfs_core.render import ContextEntry


def test_make_lesson_folds_whitespace_keeps_extras_and_rejects_empty_claims() -> None:
    lesson = make_lesson("  jest snapshot\n failed ", " fix:fix_code ", 1, " real\tregression ", check="totals")
    assert lesson == {"kind": "lesson", "situation": "jest snapshot failed", "action": "fix:fix_code",
                      "worked": True, "text": "real regression", "check": "totals"}
    assert make_lesson("s", "a", False).get("text") is None
    with pytest.raises(ValueError):
        make_lesson("   ", "fix:a", True)
    with pytest.raises(ValueError):
        make_lesson("s", "", True)


def test_lesson_of_accepts_only_a_complete_claim() -> None:
    assert lesson_of(make_lesson("s", "a", True)) is not None
    assert lesson_of({"kind": "lesson", "situation": "s", "action": "a", "worked": "yes"}) is None
    assert lesson_of({"kind": "lesson", "situation": "", "action": "a", "worked": True}) is None
    assert lesson_of({"situation": "s", "action": "a", "worked": True}) is None
    assert lesson_of("When: s. a worked.") is None
    assert lesson_of(None) is None


def test_same_claim_reads_the_claim_and_not_the_words() -> None:
    a = make_lesson("jest snapshot failed, backend-only PR", "fix:fix_code", True, "one wording")
    b = make_lesson("Jest snapshot   failed, backend-only PR", "fix:fix_code", True, "another wording")
    c = make_lesson("jest snapshot failed, backend-only PR", "fix:fix_code", False, "one wording")
    d = make_lesson("jest snapshot failed, UI PR", "fix:fix_code", True, "one wording")
    assert same_claim(a, b)
    assert not same_claim(a, c), "worked is part of the claim"
    assert not same_claim(a, d), "the situation is part of the claim"
    assert not same_claim(a, "prose that says the same thing")
    assert not same_claim(a, {**a, "kind": "note"})
    assert lesson_claim(a) == ("jest snapshot failed, backend-only pr", "fix:fix_code", True)
    # Plain values are unchanged by the rule.
    assert same_claim({"x": 1, "y": [1, 2]}, {"y": [1, 2], "x": 1})
    assert not same_claim("a", "b")


def test_inherit_evidence_carries_the_record_across_a_reworded_lesson() -> None:
    prov = Provenance(agent_id="a", session_id="s", written_at=datetime.now(UTC))
    current = MemoryEntry(
        entity_path="acme/ci", key="learned-x", value=make_lesson("s", "fix:a", True, "old words"),
        provenance=prov, version=3, confidence=0.7, success_count=8, failure_count=1, outcome_count=9,
        evidence_status="validated",
    )
    new = MemoryEntry(
        entity_path="acme/ci", key="learned-x", value=make_lesson("S", "fix:a", True, "new words"),
        provenance=prov, version=4, confidence=0.9,
    )
    kept = inherit_evidence(new, current)
    assert kept.success_count == 8 and kept.evidence_status == "validated" and kept.confidence == 0.7
    assert kept.value["text"] == "new words"
    flipped = new.model_copy(update={"value": make_lesson("s", "fix:a", False, "stopped working")})
    assert inherit_evidence(flipped, current).success_count == 0


def test_lesson_key_is_one_per_situation_and_never_a_reserved_prefix() -> None:
    k1 = lesson_key("jest snapshot failed, backend-only PR")
    k2 = lesson_key("  Jest snapshot failed,   backend-only PR ")
    assert k1 == k2 and k1.startswith(LESSON_KEY_PREFIX)
    assert k1 != lesson_key("jest snapshot failed, UI PR")
    assert not is_training_excluded_key(lesson_key("contrast between a and b"))
    assert not is_training_excluded_key(lesson_key("risk of losing data"))
    long = lesson_key("x" * 500)
    assert len(long) < 80
    with pytest.raises(ValueError):
        lesson_key("  ")


def test_render_puts_the_claim_first_in_a_fixed_shape() -> None:
    lesson = make_lesson("jest snapshot failed, backend-only PR", "fix:fix_code", True, "real regression")
    assert render_lesson(lesson) == (
        "When: jest snapshot failed, backend-only PR. fix:fix_code worked. real regression"
    )
    assert render_lesson(make_lesson("flaky webhook timeout", "fix:rerun_job", False)) == (
        "When: flaky webhook timeout. fix:rerun_job did not work."
    )
    assert render_lesson("prose") == ""
    entry = ContextEntry("acme/ci", "learned-x", lesson, confidence=0.9, evidence_status="validated",
                         success_count=3)
    assert entry.is_lesson
    assert entry.render() == (
        "- acme/ci/learned-x (high, validated 3/0): When: jest snapshot failed, backend-only PR. "
        "fix:fix_code worked. real regression"
    )
    # Two wordings of one lesson render the same claim prefix.
    other = ContextEntry("acme/ci", "learned-x", {**lesson, "text": "other words"}, confidence=0.9,
                         evidence_status="validated", success_count=3)
    assert entry.render().split(" worked.")[0] == other.render().split(" worked.")[0]
    assert not ContextEntry("acme/ci", "k", "prose").is_lesson
