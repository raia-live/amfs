"""The substrate contract: the shapes downstream loops import from this package.

``amfs-core`` is the substrate under more than one consumer — the OSS engine,
the SDK's prompt guidance, and the closed repair and training loops that run
against the hosted service. Those loops do not subclass or wrap this package;
they *import* a handful of shapes and arithmetic from it and depend on them
holding still: the fields a procedure carries, the fields an outcome record
carries, the evidence functions that say what a success is, and the one
renderer that turns memory into a prompt the same way at train and serve time.

This module is the list. It is not a type-checker and not an API freeze for
the whole package — only the symbols named here are promised, and the promise
is:

* a symbol listed here is importable from the named module;
* the field tuples listed here are present on the named models, by name;
* removing or renaming anything listed bumps :data:`SUBSTRATE_VERSION`, in the
  same commit, with the entry's replacement (or ``None``) in
  :data:`RETIRED`.

Adding a field, a function or a constant is not a contract change. A consumer
checks the contract it was written against with :func:`check`, and pins the
package version that introduced the :data:`SUBSTRATE_VERSION` it needs.

The module is deliberately data, not code: a consumer that cannot import the
new ``amfs-core`` yet can still read this file in its source tree.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
from typing import Any

#: Bumped when a listed symbol or field is removed or renamed. Additions do
#: not bump it.
SUBSTRATE_VERSION = 1

#: Symbols a downstream loop may import, by module. Each name must resolve
#: on ``import <module>; getattr(module, name)``.
SYMBOLS: dict[str, tuple[str, ...]] = {
    "amfs_core.models": (
        # Shapes.
        "MemoryEntry",
        "MemoryType",
        "Provenance",
        "ProvenanceTier",
        "ArtifactRef",
        "OutcomeType",
        "OutcomeRecord",
        "AttemptRecord",
        "RecallConfig",
        # What an outcome does to confidence, by outcome type. A calibrator
        # that learns per-domain multipliers falls back to these.
        "OUTCOME_MULTIPLIERS",
        # Procedures: what a value of ``memory_type == "procedure"`` carries.
        "PROCEDURE_REQUIRED_FIELDS",
        "PROCEDURE_OPTIONAL_FIELDS",
        "PROCEDURE_ENVIRONMENT_PRECONDITIONS",
        "procedure_issues",
        "procedure_action_keys",
        # The workflow fields of a procedure, read the one way.
        "procedure_roles",
        "procedure_call_order",
    ),
    "amfs_core.evidence": (
        # What an outcome means.
        "is_success",
        "is_known",
        "outcome_model",
        "regime_shifted",
        "evidence_signal",
        # Which keys are system-written and which stay out of a model's context.
        "is_synthetic_key",
        "is_training_excluded_key",
        "contrast_lesson_key",
    ),
    "amfs_core.actions": (
        "action_key",
        "actions_taken",
        "render_priors",
    ),
    "amfs_core.render": (
        # One renderer for SDK guidance and trained models.
        "RENDER_VERSION",
        "ContextEntry",
        "TriedAction",
        "render_context",
        "render_procedures",
        "render_tried",
        "tried_actions",
        "guidance_id",
        "confidence_bucket",
        "procedure_parts",
        "workflow_parts",
    ),
    "amfs_core.capture": (
        "scan_captured_text",
    ),
    "amfs_core.default_embedder": (
        # The embedder a consumer falls back to when none is configured, and
        # the class it checks for to know it did.
        "HashEmbedder",
        "create_default_embedder",
    ),
}

#: Fields a consumer may read on a model, by ``module:Class``. Pydantic
#: models are checked through ``model_fields``; dataclasses through their
#: fields; anything else through ``inspect.signature`` of the constructor.
FIELDS: dict[str, tuple[str, ...]] = {
    "amfs_core.models:OutcomeRecord": (
        "outcome_ref",
        "outcome_type",
        "causal_confidence",
        "committed_at",
        "causal_entry_keys",
        "causal_entry_versions",
        "agent_id",
        "attempts",
        "final_action_index",
        "actions_taken",
        "entity_paths",
        "situation",
        "task_input",
        "response_text",
    ),
    "amfs_core.models:AttemptRecord": (
        "attempt",
        "outcome_type",
        "causal_entry_keys",
        "causal_entry_versions",
        "action_indices",
        "summary",
    ),
    "amfs_core.render:ContextEntry": (
        "entity_path",
        "key",
        "value",
        "confidence",
        "evidence_status",
        "success_count",
        "failure_count",
        "memory_type",
        "version",
    ),
}

#: Procedure value keys, mirrored from ``amfs_core.models`` so a consumer
#: reading this file without importing the package sees the same list. The
#: test in this repository keeps the two equal.
PROCEDURE_FIELDS: tuple[str, ...] = (
    "goal",
    "steps",
    "preconditions",
    "on_failure",
    "verify",
    "effects",
    "depends_on",
    "evidence",
    "roles",
    "call_order",
    "output_contracts",
)

#: ``"module:name" -> "module:replacement" | None`` for every symbol or field
#: removed since version 1. Empty while the contract is at its first version.
RETIRED: dict[str, str | None] = {}


def _field_names(obj: Any) -> set[str]:
    model_fields = getattr(obj, "model_fields", None)
    if isinstance(model_fields, dict):
        return set(model_fields)
    dataclass_fields = getattr(obj, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, dict):
        return set(dataclass_fields)
    try:
        return set(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return set()


def check(
    symbols: dict[str, Iterable[str]] | None = None,
    fields: dict[str, Iterable[str]] | None = None,
) -> list[str]:
    """Problems with the contract, as ``"module:name"`` strings; empty when
    it holds. With no arguments, checks this module's own lists. A consumer
    passes the subset it imports to check *its* contract against the
    installed package."""
    problems: list[str] = []
    for module_name, names in (symbols if symbols is not None else SYMBOLS).items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            problems.extend(f"{module_name}:{name}" for name in names)
            continue
        problems.extend(f"{module_name}:{name}" for name in names if not hasattr(module, name))
    for ref, names in (fields if fields is not None else FIELDS).items():
        module_name, _, class_name = ref.partition(":")
        try:
            obj = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            problems.extend(f"{ref}.{name}" for name in names)
            continue
        present = _field_names(obj)
        problems.extend(f"{ref}.{name}" for name in names if name not in present)
    return problems


def covers(symbols: dict[str, Iterable[str]], fields: dict[str, Iterable[str]] | None = None
           ) -> list[str]:
    """The ``"module:name"`` entries of a consumer's lists that this contract
    does **not** promise. A consumer that imports an unlisted symbol is
    outside the contract and should either add it here or stop."""
    out: list[str] = []
    for module_name, names in symbols.items():
        promised = set(SYMBOLS.get(module_name, ()))
        out.extend(f"{module_name}:{name}" for name in names if name not in promised)
    for ref, names in (fields or {}).items():
        promised = set(FIELDS.get(ref, ()))
        out.extend(f"{ref}.{name}" for name in names if name not in promised)
    return out


__all__ = [
    "FIELDS",
    "PROCEDURE_FIELDS",
    "RETIRED",
    "SUBSTRATE_VERSION",
    "SYMBOLS",
    "check",
    "covers",
]
