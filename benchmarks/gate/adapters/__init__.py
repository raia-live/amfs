"""Loops the harness can drive.

- :mod:`.reference` — two loops that run in-process with a simulated clock, the
  harness's own fixtures: ``accept-all`` (a gate that waves everything through,
  the floor every real loop should clear) and ``replay`` (an offline replay with
  a configurable false-accept probability and, optionally, a memory of what it
  verified). Both are what ``--loop reference:...`` means.
- :mod:`.senselab` — SenseLab's repair loop over its HTTP API
  (``--loop senselab --agent <id>``).

A new loop is a class satisfying :class:`~benchmarks.gate.protocol.GatedLoop`
and a line in :func:`resolve`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..protocol import GatedLoop


def resolve(spec: str, options: Mapping[str, Any]) -> GatedLoop:
    """``reference`` / ``reference:accept-all`` / ``reference:replay`` /
    ``senselab`` → a loop. *options* are the CLI's ``--opt key=value`` pairs plus
    ``agent`` / ``url`` / ``api_key`` / ``mode``."""
    name, _, variant = spec.partition(":")
    if name == "reference":
        from . import reference

        return reference.make(variant or "replay", options)
    if name == "senselab":
        from . import senselab

        return senselab.SenseLabLoop.from_options(options)
    raise ValueError(f"unknown loop {spec!r}; reference[:accept-all|:replay] or senselab")


__all__ = ["resolve"]
