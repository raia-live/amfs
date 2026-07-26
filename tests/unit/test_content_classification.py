"""Unit tests for amfs_core.content — artifact classification, descriptors, embedding routing."""

from __future__ import annotations

from datetime import datetime, timezone

from amfs_core.content import (
    ARTIFACT_PENALTY,
    artifact_descriptor,
    classify_artifact,
    embedding_input,
    is_artifact_key,
    is_code_like,
)
from amfs_core.models import MemoryEntry, Provenance, SearchQuery


class TestIsArtifactKey:
    def test_source_file_paths_are_artifacts(self):
        assert is_artifact_key("src/pages/InsightsPage.tsx")
        assert is_artifact_key("packages/core/engine.py")
        assert is_artifact_key("styles/app.css")
        assert is_artifact_key("config.yaml")

    def test_knowledge_keys_are_not_artifacts(self):
        assert not is_artifact_key("food-preference-pizza")
        assert not is_artifact_key("user/preferences/favorite-color")
        assert not is_artifact_key("decision-use-jwt")
        # A dotted key that isn't a known code extension stays a fact.
        assert not is_artifact_key("release-v1.2")

    def test_extension_case_insensitive(self):
        assert is_artifact_key("Component.TSX")


class TestIsCodeLike:
    def test_detects_code_content(self):
        assert is_code_like("import React from 'react';\nexport const App = () => null;")
        assert is_code_like("def compute(x):\n    return x * 2\n")
        assert is_code_like("class FooBarBaz:\n    def __init__(self):\n        self.x = 1\n")

    def test_prose_is_not_code(self):
        assert not is_code_like("The user prefers pizza with extra cheese on Fridays.")
        assert not is_code_like("We decided to use JWT because sessions did not scale.")

    def test_short_strings_are_not_code(self):
        # Below the min-length guard; avoids misreading a terse fact as code.
        assert not is_code_like("return home")


class TestClassifyArtifact:
    def test_key_or_content_triggers(self):
        # Key alone.
        assert classify_artifact("a/b/x.py", "just some prose value here for length")
        # Content alone (non-code key).
        assert classify_artifact("notes", "export default function App() { return null }")
        # Neither.
        assert not classify_artifact("food-pref", "I like pizza and pasta on weekends.")

    def test_non_string_values(self):
        assert not classify_artifact("prefs", {"food": "pizza", "drink": "water"})


class TestProseAboutCodeIsNotCode:
    """Every one of these is a real production memory that got filed as code.

    All seven rows flagged ``is_artifact`` in the production store were technical
    notes, not files, and each was caught by a bare substring appearing mid
    sentence. Left unfixed, an entry's meaning is replaced by "source file
    <key>." in its vector and it stops being findable by meaning.
    """

    def test_path_globs(self):
        assert not is_code_like(
            "Reads keyed by entry_key, keyword signal in blend, exclude "
            "bench-*/_system/* (_is_excluded_entity), UserVisibilityFilter applied ONCE."
        )
        assert not is_code_like(
            "Ran the main checkout's .venv pytest with PYTHONPATH listing worktree "
            "packages/*/src to shadow the installed copies."
        )
        assert not is_code_like(
            "auto-publish.yml triggers on push to main touching "
            "packages/**/pyproject.toml or packages/sdk-typescript/package.json."
        )
        assert not is_code_like(
            "CRITICAL: the /api/v1/pro/tenant/* provisioning router has no auth "
            "dependency and the tenant middleware passes it through."
        )

    def test_quoted_markup_inside_a_sentence(self):
        assert not is_code_like(
            "POPUP LOGO: replaced the text <span class='logo'>SenseLab</span> in "
            "the popup Header with an image tag pointing at the bundled logo."
        )

    def test_arrow_used_as_prose(self):
        assert not is_code_like(
            "Pro forwards include_artifacts to an OSS retrieve that lacks it => "
            "TypeError on every call, even when the caller omits the argument."
        )

    def test_quoted_function_call(self):
        assert not is_code_like(
            "sendTeamInviteEmail has no success log line — only "
            "console.error('[invite] Email failed for %s') on failure, so verify from logs."
        )

    def test_still_catches_real_code(self):
        # The tightening must not cost genuine detection.
        assert is_code_like("import React from 'react'\nexport const App = () => <div/>\n")
        assert is_code_like("def compute(x):\n    return x * 2\n")
        assert is_code_like(
            "<div className='wrapper'>\n  <span>hi</span>\n</div>\n"
        )

    def test_weak_hints_corroborate(self):
        # One line-anchored weak hint is not enough; two are.
        assert not is_code_like("/* a note that opens with a comment marker and then prose */")
        assert is_code_like("const total = 1\nconsole.log(total)\n")


class TestArtifactDescriptor:
    def test_includes_language_filename_and_symbols(self):
        code = (
            "import React from 'react'\n"
            "export const InsightsPage = () => <div/>\n"
            "function helper() { return 1 }\n"
        )
        desc = artifact_descriptor("src/pages/InsightsPage.tsx", code)
        assert "TypeScript React" in desc
        assert "InsightsPage.tsx" in desc
        assert "InsightsPage" in desc
        # Descriptor is compact, not the whole blob.
        assert len(desc) < len(code) + 200

    def test_python_symbols(self):
        code = "def alpha():\n    pass\n\nclass Beta:\n    pass\n"
        desc = artifact_descriptor("mod/thing.py", code)
        assert "Python" in desc
        assert "alpha" in desc and "Beta" in desc


class TestEmbeddingInput:
    def test_artifact_returns_descriptor(self):
        is_art, text = embedding_input("src/x.ts", "export const y = 1\n" * 5)
        assert is_art is True
        assert "TypeScript" in text
        # Not the raw blob.
        assert "export const y" not in text

    def test_fact_returns_value_text(self):
        is_art, text = embedding_input("food-pref", "I love pizza")
        assert is_art is False
        assert text == "I love pizza"

    def test_code_under_a_prose_key_embeds_its_content(self):
        # A descriptor built from a prose key collapses to "source file <key>."
        # and erases the entry's meaning, so it is reserved for file keys. The
        # entry is still reported as an artifact, so ranking still demotes it.
        value = "export default function App() { return null }"
        is_art, text = embedding_input("notes", value)

        assert is_art is True
        assert text == value

    def test_descriptor_requires_a_file_key(self):
        _, text = embedding_input("decision-use-jwt", "const token = sign(payload)\nconsole.log(token)\n")

        assert "source file" not in text

    def test_penalty_in_range(self):
        assert 0.0 < ARTIFACT_PENALTY < 1.0


class TestModelField:
    def test_memory_entry_defaults_not_artifact(self):
        e = MemoryEntry(
            entity_path="svc/a", key="k", value="v",
            provenance=Provenance(
                agent_id="a", session_id="s",
                written_at=datetime.now(timezone.utc),
            ),
        )
        assert e.is_artifact is False

    def test_search_query_include_artifacts_default(self):
        assert SearchQuery().include_artifacts is True
        assert SearchQuery(include_artifacts=False).include_artifacts is False
