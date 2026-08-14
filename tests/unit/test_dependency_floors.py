"""A declared floor on a sibling may not be lower than the sibling itself.

These packages are released separately and installed from PyPI, most often by
`uvx`, which resolves against whatever it already has cached. So a floor is not
documentation — it is the only thing standing between a user and a version of a
sibling that predates the feature the caller depends on.

The failure is always quiet. The old sibling imports, the call succeeds, and the
field the caller added is simply not there: `tool_calls` dropped on the way to
`/outcomes` and a trace sealed with no actions. Nothing raises, so nothing is
noticed until someone asks why a model trained on those traces never became
ready. `amfs-mcp-server` shipped exactly that, flooring `amfs-adapter-http` at
0.1.2 while relying on 0.1.10 to copy the field.

Scope, deliberately: this checks floors that exist. Several integration packages
name a sibling with no floor at all, which is the same hazard in a stronger
form, but raising those is a release decision for each of them rather than
something to do on the way past. ``test_every_floor_that_exists_is_checked``
pins how many are currently unfloored, so the number can only go down without a
deliberate edit.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)\s*(?P<op>>=)?\s*(?P<version>[0-9][^,;\s]*)?"
)

# Packages that name a sibling without a floor. Not a target to hold at — a
# ratchet, so the gap cannot widen unnoticed.
UNFLOORED_TODAY = 18


def _parts(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text))


def _packages() -> dict[str, tuple[str, Path]]:
    """Every distribution in this repo, by name, with its version and path."""
    found: dict[str, tuple[str, Path]] = {}
    for path in sorted(REPO.glob("packages/**/pyproject.toml")):
        project = tomllib.loads(path.read_text()).get("project") or {}
        if project.get("name") and project.get("version"):
            found[project["name"]] = (project["version"], path)
    return found


PACKAGES = _packages()


def _edges() -> list[tuple[str, Path, str, str | None]]:
    """(dependent, its pyproject, sibling name, declared floor or None)."""
    edges = []
    for name, (_, path) in PACKAGES.items():
        project = tomllib.loads(path.read_text())["project"]
        for requirement in project.get("dependencies") or []:
            match = REQUIREMENT.match(requirement)
            if match and match["name"] in PACKAGES:
                floor = match["version"] if match["op"] else None
                edges.append((name, path, match["name"], floor))
    return edges


EDGES = _edges()
FLOORED = [edge for edge in EDGES if edge[3] is not None]


def test_the_repo_has_packages_to_check() -> None:
    """A glob that quietly matched nothing would make every test below pass."""
    assert len(PACKAGES) > 3, f"only found {sorted(PACKAGES)}"
    assert FLOORED, "no package floors a sibling; this guard would never fire"


def test_every_floor_that_exists_is_checked() -> None:
    unfloored = [f"{dep} -> {sib}" for dep, _, sib, floor in EDGES if floor is None]
    assert len(unfloored) <= UNFLOORED_TODAY, (
        f"{len(unfloored)} sibling dependencies have no floor, up from "
        f"{UNFLOORED_TODAY}. A new one is a new way for a cached release to "
        f"satisfy a dependency it predates:\n  " + "\n  ".join(sorted(unfloored))
    )


@pytest.mark.parametrize(
    ("dependent", "path", "sibling", "floor"),
    FLOORED,
    ids=[f"{dep}->{sib}" for dep, _, sib, _ in FLOORED],
)
def test_a_declared_floor_is_at_least_the_version_in_this_repo(
    dependent: str, path: Path, sibling: str, floor: str
) -> None:
    current, _ = PACKAGES[sibling]
    assert _parts(floor) >= _parts(current), (
        f"{dependent} floors {sibling} at {floor}, but this repo has {sibling} "
        f"{current}. A resolver may satisfy that floor with a release that "
        f"predates the code {dependent} is written against, and the mismatch is "
        f"silent. Raise it in {path.relative_to(REPO)}."
    )
