"""Structured lessons: what an agent learned, as a claim the record can follow.

An agent that reflects after a task writes a note — "on a snapshot failure
with no UI change, fix the code; regenerating snapshots hides a real bug".
Written as prose, the note's *claim* is whatever the text happens to say, and
:func:`amfs_core.evidence.inherit_evidence` can only carry an outcome record
across a rewrite when the text is identical. Agents restate: the same lesson
comes back rephrased, sometimes with a detail added, and every rephrasing
opened a new claim with an empty record. A lesson validated eight times looked
untested; a lesson that had just been discredited came back clean; and the
regime-shift reading, which needs the record to see the second failure in a
row, never fired — the agent had rewritten the note after the first.

A structured lesson names its claim outright::

    {"kind": "lesson", "situation": "jest snapshot failed, backend-only PR",
     "action": "fix:fix_code", "worked": true,
     "text": "the snapshot caught a real regression in the totals"}

The claim is ``(situation, action, worked)``. The text is the agent's words
and may change freely; two lessons with the same claim are the same lesson,
and the outcome record is inherited across the rewrite. ``worked=False`` is a
claim too — "this action does not resolve this situation" — and its record
is kept apart from the positive one, so a lesson that flips keeps neither.

Rendering puts the claim first, in a fixed shape, so a model sees the same
line for the same lesson whatever the agent's phrasing that day.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

#: The ``kind`` a structured lesson carries.
LESSON_KIND = "lesson"
#: Longest situation and text kept on a lesson; longer values are free text
#: that has stopped being a claim.
LESSON_SITUATION_MAX_CHARS = 300
LESSON_TEXT_MAX_CHARS = 1_000


def make_lesson(
    situation: str,
    action: str,
    worked: bool,
    text: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A lesson value. *extra* keys are kept (a ``check``, an ``example``)
    and are not part of the claim."""
    situation = " ".join(str(situation).split())[:LESSON_SITUATION_MAX_CHARS]
    action = str(action).strip()
    if not situation:
        raise ValueError("a lesson needs a situation")
    if not action:
        raise ValueError("a lesson needs an action")
    value: dict[str, Any] = {
        "kind": LESSON_KIND,
        "situation": situation,
        "action": action,
        "worked": bool(worked),
    }
    if text is not None and str(text).strip():
        value["text"] = " ".join(str(text).split())[:LESSON_TEXT_MAX_CHARS]
    for key, val in extra.items():
        if key not in value and val is not None:
            value[key] = val
    return value


#: Prefix of the keys :func:`lesson_key` builds. Not ``lesson-``: the system's
#: own ``lesson-contrast-*`` entries are excluded from training and hidden
#: from prompts, and a situation slug must not be able to land on that prefix.
LESSON_KEY_PREFIX = "learned-"
_SLUG_MAX = 48


def lesson_key(situation: str) -> str:
    """One key per situation: ``learned-<slug>-<8-hex>``. The slug is the
    situation's first words, for a reader; the hash is the whole situation,
    for uniqueness. The same situation text always yields the same key, so
    the lesson that says an action worked and the later one that says it did
    not are versions of one entry, which is what lets the briefing report the
    second as replacing the first."""
    folded = " ".join(str(situation).split()).lower()
    if not folded:
        raise ValueError("a lesson key needs a situation")
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")[:_SLUG_MAX].rstrip("-") or "situation"
    digest = hashlib.sha1(folded.encode("utf-8")).hexdigest()[:8]
    return f"{LESSON_KEY_PREFIX}{slug}-{digest}"


def lesson_of(value: Any) -> dict[str, Any] | None:
    """The lesson a value is, or ``None``. A dict with ``kind == "lesson"``, a
    non-empty ``situation`` and ``action``, and a boolean ``worked``."""
    if not isinstance(value, dict) or value.get("kind") != LESSON_KIND:
        return None
    situation = value.get("situation")
    action = value.get("action")
    worked = value.get("worked")
    if not (isinstance(situation, str) and situation.strip()):
        return None
    if not (isinstance(action, str) and action.strip()):
        return None
    if not isinstance(worked, bool):
        return None
    return value


def lesson_claim(value: Any) -> tuple[str, str, bool] | None:
    """``(situation, action, worked)`` of a lesson, normalised for comparison
    (whitespace folded, case kept on the action key, lowered on the
    situation); ``None`` for anything that is not a lesson."""
    lesson = lesson_of(value)
    if lesson is None:
        return None
    situation = " ".join(str(lesson["situation"]).split()).lower()
    return situation, str(lesson["action"]).strip(), bool(lesson["worked"])


#: Share of a situation's words that must appear in the task for the lesson
#: to be read as about it, when no declared situation is compared exactly.
LESSON_APPLIES_MIN_COVERAGE = 0.85
_WORD = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "on", "in", "to", "for", "with", "is",
    "are", "was", "no", "not", "has", "have", "by", "at", "as", "it", "this",
    "that", "from", "pr", "task", "changed", "files",
})


def _words(text: str) -> set[str]:
    return {w.strip(".,;:()[]\"'") for w in _WORD.findall(str(text).lower())} - _STOP - {""}


def lesson_applies(situation: str, task: str | None, *, declared: str | None = None) -> bool:
    """Whether a lesson about *situation* is about this task.

    With a *declared* situation (the one the run named on ``begin``), the two
    are compared exactly, whitespace folded and case ignored: a run that
    says what its situation is gets the lessons filed under it and no other.
    Without one, the situation's words are looked for in the task text: the
    lesson applies when :data:`LESSON_APPLIES_MIN_COVERAGE` of them are
    there. A situation has few words and they are the task's own (the agent
    wrote them from the task), so "pip-audit: requests has ; fix version ·
    changed requirements.txt" is found in a requests audit failure and not
    in a urllib3 one, while a jest lesson's words are not in either.

    Why this exists (ops-queue demo, 2026-09-22): the plan after a turned
    winner was drawn from priors pooled over a similarity neighbourhood, and
    the neighbourhood held two classes with opposite answers; the lesson for
    the exact class said the plan's second action had failed there. Lessons
    are the situation-exact record; priors are not.
    """
    sit = " ".join(str(situation or "").split()).lower()
    if not sit:
        return False
    if declared is not None and str(declared).strip():
        return sit == " ".join(str(declared).split()).lower()
    if not task:
        return False
    need = _words(sit)
    if not need:
        return False
    have = _words(task)
    covered = sum(1 for w in need if w in have)
    return covered / len(need) >= LESSON_APPLIES_MIN_COVERAGE


def applicable_claims(
    lessons: Any, task: str | None, *, declared: str | None = None,
) -> list[dict[str, Any]]:
    """The claims among *lessons* that are about this task, each as
    ``{"action", "worked", "evidence_status"}``. *lessons* are mappings with
    at least ``situation``, ``action`` and ``worked`` — a lesson value, or
    the SDK's lesson rows; ``evidence_status`` is carried when present so a
    discredited lesson can be left out of an order."""
    out: list[dict[str, Any]] = []
    for lesson in lessons or []:
        if not isinstance(lesson, dict):
            continue
        situation = lesson.get("situation")
        action = lesson.get("action")
        worked = lesson.get("worked")
        if not (isinstance(situation, str) and isinstance(action, str) and isinstance(worked, bool)):
            continue
        if not lesson_applies(situation, task, declared=declared):
            continue
        out.append({
            "action": action.strip(), "worked": worked,
            "evidence_status": lesson.get("evidence_status"),
        })
    return out


def render_lesson(value: Any) -> str:
    """One line: the claim, then the agent's words. ``""`` for a non-lesson."""
    lesson = lesson_of(value)
    if lesson is None:
        return ""
    situation = " ".join(str(lesson["situation"]).split())
    verb = "worked" if lesson["worked"] else "did not work"
    line = f"When: {situation}. {lesson['action']} {verb}."
    text = lesson.get("text")
    if isinstance(text, str) and text.strip():
        line += " " + " ".join(text.split())
    return line


__all__ = [
    "LESSON_KEY_PREFIX",
    "LESSON_KIND",
    "LESSON_SITUATION_MAX_CHARS",
    "LESSON_TEXT_MAX_CHARS",
    "LESSON_APPLIES_MIN_COVERAGE",
    "applicable_claims",
    "lesson_applies",
    "lesson_claim",
    "lesson_key",
    "lesson_of",
    "make_lesson",
    "render_lesson",
]
