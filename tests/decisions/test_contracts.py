import pytest
from pydantic import ValidationError

from amfs_core.decisions import Candidate, DecisionRequest, Question


def question(**kwargs):
    return Question(instructions="Choose", candidates=[Candidate(id="a", description="First"), Candidate(id="b", description="Second")], **kwargs)


def test_dependency_validation_and_spec_identity():
    with pytest.raises(ValidationError, match="cycle"):
        DecisionRequest(decision="x", spec_version="v1", state={}, questions={"q": question(depends_on=["q"])})
    one = DecisionRequest(decision="x", spec_version="v1", state={}, questions={"q": question()})
    two = DecisionRequest(decision="x", spec_version="v1", state={"new": True}, questions={"q": question()})
    assert one.spec_hash == two.spec_hash
    restricted = DecisionRequest(decision="x", spec_version="v1", state={}, questions={"q": question()}, allowed_candidates={"q": ["a"]})
    assert restricted.spec_hash != one.spec_hash


def test_invalid_candidate_tuple_and_score():
    with pytest.raises(ValidationError):
        question(type="boolean")
    with pytest.raises(ValidationError):
        question(type="score", levels=[1, 1])
    with pytest.raises(ValidationError):
        DecisionRequest(decision="x", spec_version="v1", state={}, questions={"q": question()}, valid_tuples=[{"q": "unknown"}])
