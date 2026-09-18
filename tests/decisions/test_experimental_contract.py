"""Additive experimental diagnostics cannot silently alter historical seals."""
import pytest
from pydantic import ValidationError
from amfs_core.decisions import DecisionAnswer, ExperimentalDecisionDiagnostics, fingerprint


def diagnostics():
    return dict(architecture='experimental-recovery-vector.v1',artifact_sha256='a'*64,
        feature_contract_sha256='b'*64,continuation_sha256='c'*64,
        branches={'skip':{'completion':.4,'harm':.02},'acquire':{'completion':.6,'harm':.01}},
        acquisition_cost=.1,remaining_cost=1.,policy_reason='experimental_observe_only')


def test_legacy_answer_wire_and_digest_survive_roundtrip():
    original={'value':'retry','candidate_id':'retry','distribution':{'retry':.8,'review':.2},
        'decision_probability':.8,'estimated_success':None,'automate':False,'disposition':'defer',
        'reason':'observe_mode','served_by':'d1','model_version':'v1','risk':None,
        'verification':[],'candidate_distribution':None}
    restored=DecisionAnswer.model_validate(original).model_dump(mode='json')
    assert restored==original
    assert fingerprint(restored)==fingerprint(original)


@pytest.mark.parametrize('change',[
    {'automate':True},{'distribution':{'skip':.5,'acquire':.5}},
    {'decision_probability':.5},{'estimated_success':.6},
    {'candidate_distribution':{'skip':.5,'acquire':.5}},{'disposition':'gather'}])
def test_experimental_output_cannot_claim_probability_or_authorize_action(change):
    with pytest.raises(ValidationError):
        DecisionAnswer(value='acquire',candidate_id='acquire',reason='experimental_observe_only',
            served_by='d1',model_version='exp-v1',experimental=diagnostics(),**change)


def test_typed_estimates_roundtrip_and_reject_incomplete_or_certified_claims():
    answer=DecisionAnswer(value='acquire',candidate_id='acquire',reason='experimental_observe_only',
        served_by='d1',model_version='exp-v1',experimental=diagnostics())
    assert answer.model_dump(mode='json')['experimental']['research_gate_status']=='failed'
    assert answer.distribution=={} and answer.decision_probability is None
    assert DecisionAnswer.model_validate_json(answer.model_dump_json())==answer
    for patch in ({'branches':{'skip':{'completion':.4,'harm':.02}}},
                  {'observe_only':False},{'research_gate_status':'passed'},
                  {'acquisition_cost':float('nan')}):
        with pytest.raises(ValidationError):ExperimentalDecisionDiagnostics(**(diagnostics()|patch))
