"""The substrate contract (``amfs_core.substrate``) holds on this checkout.

A failure here is a contract change, not a bug to work around: either restore
the symbol, or bump ``SUBSTRATE_VERSION`` and record the replacement in
``RETIRED`` in the same commit. Downstream loops check the same lists against
the package they have installed.
"""

from __future__ import annotations

import dataclasses

from amfs_core import models, substrate
from amfs_core.render import ContextEntry


def test_every_promised_symbol_and_field_is_present():
    assert substrate.check() == []


def test_the_procedure_field_mirror_matches_the_models():
    assert substrate.PROCEDURE_FIELDS == (
        models.PROCEDURE_REQUIRED_FIELDS + models.PROCEDURE_OPTIONAL_FIELDS
    )


def test_version_is_one_while_nothing_has_been_retired():
    assert substrate.SUBSTRATE_VERSION >= 1
    if substrate.SUBSTRATE_VERSION == 1:
        assert substrate.RETIRED == {}
    else:
        assert substrate.RETIRED, "a bumped version names what was retired"


def test_check_reports_a_missing_symbol_and_a_missing_field_by_name():
    assert substrate.check({"amfs_core.models": ("MemoryEntry", "NoSuchThing")}, {}) == [
        "amfs_core.models:NoSuchThing"
    ]
    assert substrate.check({}, {"amfs_core.models:OutcomeRecord": ("agent_id", "nope")}) == [
        "amfs_core.models:OutcomeRecord.nope"
    ]
    assert substrate.check({"amfs_core.not_a_module": ("x",)}, {}) == ["amfs_core.not_a_module:x"]


def test_covers_names_what_a_consumer_imports_outside_the_contract():
    assert substrate.covers({"amfs_core.models": ("MemoryEntry",)}) == []
    assert substrate.covers({"amfs_core.models": ("MemoryEntry", "SearchQuery")}) == [
        "amfs_core.models:SearchQuery"
    ]
    assert substrate.covers({}, {"amfs_core.render:ContextEntry": ("key", "colour")}) == [
        "amfs_core.render:ContextEntry.colour"
    ]


def test_field_lists_are_read_from_pydantic_and_dataclasses_alike():
    assert set(substrate.FIELDS["amfs_core.models:OutcomeRecord"]) <= set(
        models.OutcomeRecord.model_fields
    )
    assert set(substrate.FIELDS["amfs_core.render:ContextEntry"]) <= {
        f.name for f in dataclasses.fields(ContextEntry)
    }
