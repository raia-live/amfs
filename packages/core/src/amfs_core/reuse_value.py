"""The reuse block a read returns: what memory actually did for this call.

One implementation, computed where the reuse is credited, because the
alternative was tried. The block used to be built client-side in two copied
``value_ledger.py`` files — one in the hosted MCP gateway, one in the Pro stdio
server — kept in step by a test asserting their class bodies were
source-identical. Everything without a copy showed the user nothing: the OSS MCP
server, the SDK, the raw HTTP API, and so every self-hoster and every direct API
caller. Adding a surface meant adding a third copy.

Three things about the old block were wrong beyond where it lived, and this is
the corrected shape.

**It led with an estimate and buried the fact.** ``est_tokens_saved`` came first
and the count of memories second. The token figure is a modelled quantity — it
prices a reuse by the size of the recalled text rather than by the work of
re-deriving it — so it is the weaker of the two claims and now sits behind the
things that are simply true: how many memories were reused, and how many times
this memory has been reused before. It is also labelled ``estimate`` rather than
left to look measured.

**It told the model what to say.** The first block of a session carried "SAY THIS
FIRST, before you start the work", which is an instruction competing with the
user's own request, and it is the same mechanism as the always-applied rule that
agents skip. Nothing here commands. ``note`` states a fact in one sentence and
the caller may relay it or not; the durable record does not depend on the model
choosing to speak, because the reuse is persisted server-side either way.

**It was silent exactly when it mattered.** A note was attached on lookups 1, 3
and 6 and withheld on 2, 4 and 5, so a session doing the right thing repeatedly
was told so intermittently. Counting is not available here anyway — each request
is independent — and that turns out to be the fix rather than a limitation: the
note is attached when there is something worth saying about *this* reuse, not
when a counter comes round. The one claim that always deserves it is a memory
written by one agent being reused by another, which is the thing a local file or
a per-tool memory cannot do.
"""

from __future__ import annotations

from typing import Any

from amfs_core.aggregates import (
    CHARS_PER_TOKEN,
    RECALL_TOKENS_CEIL,
    RECALL_TOKENS_FLOOR,
    recall_tokens_for_chars,
)

__all__ = [
    "BASIS",
    "format_tokens",
    "plain",
    "reuse_value_block",
]

#: How the token figure is arrived at, carried with it so a caller repeating the
#: number can also say where it came from. Stated as a method rather than a
#: promise: what is measured is the size of what was returned, and the claim is
#: that re-deriving it would have cost at least that.
BASIS = (
    f"Estimate: the credited memory's own content at ~{CHARS_PER_TOKEN} chars per "
    f"token, clamped to {RECALL_TOKENS_FLOOR}-{RECALL_TOKENS_CEIL} tokens per reuse "
    "so neither a one-line preference nor a large artifact distorts it. Counts the "
    "memory the agent took, not the candidates ranked below it."
)


def format_tokens(tokens: int) -> str:
    """Approximate token count, e.g. ``~1.2K``."""
    if tokens < 1000:
        return f"~{tokens}"
    if tokens < 1_000_000:
        return f"~{tokens / 1000:.1f}K".replace(".0K", "K")
    return f"~{tokens / 1_000_000:.1f}M"


def plain(approx: str) -> str:
    """Strip the ``~`` for use inside prose.

    Agents relay notes verbatim into chat, where a pair of tildes is
    strikethrough — "~21.4K tokens" renders with the number struck out, which
    reads as a figure someone retracted. The sentence around it already says
    "about", so the tilde adds nothing there.
    """
    return approx.replace("~", "")


def reuse_value_block(
    *,
    hits: int,
    content_chars: int,
    reused_before: int = 0,
    written_by: str | None = None,
    reused_by: str | None = None,
) -> dict[str, Any] | None:
    """Build the reuse block for one lookup that credited *hits* memories.

    Returns ``None`` when nothing was credited, so a lookup that found nothing
    stays silent rather than reporting a reuse of zero — which reads as a failure
    and is not what happened.

    ``reused_before`` is the credited entry's stored ``recall_count`` as it was
    *before* this reuse. Deliberately the prior count and not a total including
    this one: the increment is best-effort and issued separately, so a total
    would be optimistic by one whenever that write failed, and this block's only
    job is to be believed.

    ``written_by`` and ``reused_by`` are agent ids. When both are known and they
    differ, the block carries the cross-surface claim — one agent using what
    another worked out. That is the strongest thing memory can demonstrate and it
    costs nothing extra here: both ids are already in hand at the point the
    credit is applied.
    """
    if hits <= 0:
        return None

    tokens = recall_tokens_for_chars(content_chars, hits=hits)
    approx = format_tokens(tokens)
    plural = "y" if hits == 1 else "ies"

    # Facts first, estimate after. The order is the point: a caller that relays
    # only the opening of this block relays something true.
    block: dict[str, Any] = {
        "memories_used": hits,
        "reused_before": max(reused_before, 0),
        "est_tokens_saved": approx,
        "estimate": True,
        "basis": BASIS,
    }

    cross_surface = bool(written_by and reused_by and written_by != reused_by)
    if cross_surface:
        block["cross_surface"] = {"written_by": written_by, "reused_by": reused_by}
        block["note"] = (
            f"This memory was written by {written_by} and has just been reused by "
            f"{reused_by} — a different agent. Worth telling the user: work done in "
            "one session carried to another one, which is the thing a local file or "
            "a single tool's memory cannot do."
        )
    elif reused_before == 0:
        block["note"] = (
            f"First reuse of this memor{plural}: saved once, now answering a "
            f"question without re-deriving it. About {plain(approx)} tokens of "
            "re-research avoided on this call, by the estimate above."
        )
    else:
        block["note"] = (
            f"Reused {hits} saved memor{plural}, the {_ordinal(reused_before + 1)} "
            f"time for this one. About {plain(approx)} tokens of re-research "
            "avoided on this call, by the estimate above."
        )
    return block


def _ordinal(n: int) -> str:
    """1 -> first, 2 -> second, ... beyond a few, "12th"."""
    words = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    }
    if n in words:
        return words[n]
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
