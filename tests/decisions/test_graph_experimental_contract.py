"""Portable graph metadata stays observe-only without importing private algorithms."""
import pytest
from amfs_core.decisions import DecisionAnswer,GraphDecisionDiagnostics,canonical_json


def diagnostic():
    return GraphDecisionDiagnostics(architecture='experimental-relation-graph.v1',artifact_sha256='a'*64,
        compiler_sha256='b'*64,runtime_source_sha256='c'*64,assignment={'check':'skip','action':'retry'},
        independent_assignment={'check':'skip','action':'review'},joint_score=-3.,independent_score=-4.,
        feasible_assignments=5,solver_status='feasible',remaining_cost=1.,selected_cost=.1,policy_reason='diagnostic')


def test_graph_roundtrip_and_legacy_omission():
    baseline=DecisionAnswer(reason='test',served_by='d1',model_version='v1')
    assert 'experimental' not in baseline.model_dump() and 'experimental' not in canonical_json(baseline)
    answer=baseline.model_copy(update={'experimental':diagnostic()})
    assert DecisionAnswer.model_validate_json(answer.model_dump_json())==answer
    assert answer.experimental.joint_score==-3.  # raw potential, not probability


@pytest.mark.parametrize('change',[{'automate':True},{'risk':{}},{'distribution':{'retry':1.}},
    {'decision_probability':.9},{'estimated_success':.9},{'candidate_distribution':{'retry':1.}}, {'disposition':'act'}])
def test_graph_cannot_claim_probability_risk_or_automation(change):
    with pytest.raises(ValueError):DecisionAnswer.model_validate(dict(reason='test',served_by='d1',model_version='v1',experimental=diagnostic().model_dump())|change)


def test_review_cannot_retain_partial_assignment():
    value=diagnostic().model_dump();value['solver_status']='review'
    with pytest.raises(ValueError):GraphDecisionDiagnostics.model_validate(value)
    for key in ('assignment','independent_assignment','joint_score','independent_score','selected_cost'):value[key]=None
    value['feasible_assignments']=0
    assert GraphDecisionDiagnostics.model_validate(value).assignment is None
