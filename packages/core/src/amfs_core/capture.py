"""Sanitising captured prompt and response text before it is persisted.

``DecisionTrace.task_input`` and ``DecisionTrace.response_text`` hold the request
an agent was given and the answer it produced. That is the most useful material
in the trace — it is the prompt half of a training example — and also the most
dangerous, because it is raw caller text that routinely contains API keys,
connection strings and customer data.

This lives in core rather than next to any one entry point because a trace can be
built from several: the SDK's ``commit_outcome``, the HTTP server's trace
endpoints, and the MCP tool that wraps the SDK. When the scan lived only in the
HTTP server, the MCP path stored raw text while its own documentation promised it
had been redacted. One implementation, called wherever a trace is constructed, is
the only arrangement where that cannot drift apart again.

The scan itself is provided by the Pro ``amfs_safety`` package. On an OSS install
it is absent, and capture is opt-in, so the text is stored as supplied.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Captured prompts and responses are unbounded caller-supplied text. Truncating
#: before the safety scan keeps a runaway payload from becoming a storage or
#: latency problem, and no training example is allowed near this size anyway.
MAX_CAPTURED_CHARS = 200_000

_GATE_CACHE: dict[str, Any] = {}


def scan_captured_text(
    text: str | None,
    *,
    adapter: Any = None,
    agent_id: str = "",
    session_id: str = "",
) -> str | None:
    """Redact secrets from captured prompt/response text before it is persisted.

    Returns ``None`` when the gate blocks the text outright, and also when the
    gate is present but errors. Losing one training example is always preferable
    to writing a customer credential into a dataset that later leaves our
    infrastructure for a tuning job.
    """
    if not text:
        return None
    if len(text) > MAX_CAPTURED_CHARS:
        text = text[:MAX_CAPTURED_CHARS]
    try:
        from amfs_core.models import MemoryEntry, Provenance

        if "gate" not in _GATE_CACHE:
            from amfs_safety import SafetyGate  # Pro package, optional

            _GATE_CACHE["gate"] = SafetyGate(adapter=adapter)
        gate = _GATE_CACHE["gate"]
        entry = MemoryEntry(
            entity_path="_capture/task_input",
            key="scan",
            value=text,
            provenance=Provenance(
                agent_id=agent_id,
                session_id=session_id,
                written_at=datetime.now(UTC),
            ),
            confidence=0.5,
        )
        decision = gate.check_write(entry)
        if not decision.allowed:
            logger.info("Captured text blocked by SafetyGate; dropping from trace")
            return None
        scanned = decision.entry.value
        if isinstance(scanned, str):
            return scanned
        # The gate allowed the write but handed back something that is not a
        # string. Falling back to the caller's original text here would return the
        # one value known *not* to be what the gate approved — if it redacted into
        # a non-string form, the secret would go straight through. Drop instead.
        logger.warning(
            "SafetyGate returned a non-string value (%s); dropping capture",
            type(scanned).__name__,
        )
        return None
    except ImportError:
        # OSS install without the Pro safety package. Capture is opt-in and the
        # caller asked for it, so store the text rather than silently dropping.
        return text
    except Exception:
        # Fail closed. The gate exists here — it simply failed — so returning the
        # text would store exactly what we could not vet, which is the one thing
        # this function is for. Logged at warning rather than debug because a gate
        # that errors is silently degrading every capture until someone notices.
        logger.warning(
            "SafetyGate scan of captured text failed; dropping capture", exc_info=True
        )
        return None
