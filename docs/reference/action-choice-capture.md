# Capture available action choices

Python `AgentMemory.record_action` and MCP `amfs_record_action` accept optional
`choices` and `decision_type` fields:

```python
memory.record_action(
    "retry_job", {"job_id": "example"},
    choices=["retry_job", "escalate"], decision_type="next_tool",
)
```

Choices are 1–32 unique nonblank strings of at most 128 characters each. The
capture decision type is a caller-defined identifier of at most 128 characters,
using letters, digits, underscore, period, colon or hyphen. It describes the
recorded decision (for example `next_tool` or `workflow.retry`); it is distinct
from the typed prediction API's question types. Neither field is required and
labels need not be tool names. Recording a label does not assert its hypothetical
outcome or prove that it was executable. No training-data pooling permission is
implied by recording choices.

Metadata travels with the outcome's tool calls and the saved trace, including
immutable sealing in the SaaS package. Unset fields are omitted from serialization,
so historical action payloads and hashes retain their original shape. Choices are
copied when recorded so subsequent caller list changes cannot alter the record.

Where the optional safety scanner is installed, metadata is scanned before trace
persistence. If a label is blocked or changed by redaction, the action is dropped
rather than silently changing its choice set. OSS installations without that
scanner retain caller-supplied text, as with existing opt-in action capture.
