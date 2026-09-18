from types import SimpleNamespace
import string
import torch
from torch import nn
import pytest
from amfs_core.decisions import Question,Candidate
from amfs_decision_runtime.decider import DeciderChoiceScorer,MODEL_REVISION,TEMPERATURE

class Tokenizer:
    pad_token_id=0;eos_token_id=1;all_special_ids=[0,1]
    def encode(self,text,**kwargs):
        if text and len(text)<=2 and all(c in string.ascii_uppercase for c in text):
            return [300+string.ascii_uppercase.index(text)] if len(text)==1 else [326+26*string.ascii_uppercase.index(text[0])+string.ascii_uppercase.index(text[1])]
        return [v+2 for v in text.encode()]
class Backbone(nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,input_ids,attention_mask,use_cache):
        assert use_cache is False;assert input_ids.shape==attention_mask.shape
        self.calls.append((input_ids.clone(),attention_mask.clone()))
        h=(input_ids*attention_mask).cumsum(1).float()%13
        return SimpleNamespace(last_hidden_state=torch.stack((h,torch.ones_like(h)),-1))
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.model=Backbone();self.lm_head=nn.Linear(2,1100,bias=False)
        with torch.no_grad():self.lm_head.weight.copy_(torch.arange(2200).reshape(1100,2)/10000)

def questions():
    return {'x':Question(instructions='Choose',candidates=[Candidate(id='shared_A',description='alpha'),Candidate(id='shared_B',description='beta')]),'y':Question(instructions='Choose longer question',candidates=[Candidate(id='yes',description='yes'),Candidate(id='no',description='no')])}
def test_native_prompt_full_state_ids_metadata_and_no_teacher_labels():
    s=DeciderChoiceScorer(Model(),Tokenizer());rows=s._compile('observable state',questions())
    # Native prompt uses no chat wrapper and independently repeats context.
    text=bytes(i-2 for i in rows[0]['tokens']).decode()
    assert text.startswith('Context:\nobservable state\n\nQuestion: Choose\nOptions:\n(A) shared_A: ')
    assert '"description":"alpha"' in text and text.endswith('\nAnswer: (')
    assert 'gold' not in text and 'Question 2' not in text
    invalid={'x':{'instructions':'Choose','candidates':[{'id':'a','gold':True},{'id':'b'}]}}
    with pytest.raises(ValueError):s.score('state',invalid)
def test_native_head_temperature_and_variable_padding_matches_serial():
    model=Model();s=DeciderChoiceScorer(model,Tokenizer());p=s.score('observable',questions());serial=s.score('observable',questions(),parallel=False)
    assert p.distributions==serial.distributions
    assert p.output_tokens==0 and p.diagnostics['calibrated'] is False and p.diagnostics['temperature']==1.3
    rows=s._compile('observable',questions());h=sum(rows[0]['tokens'])%13
    expected=torch.softmax(torch.nn.functional.linear(torch.tensor([h,1.]),model.lm_head.weight[s.labels[:2]])/TEMPERATURE,-1)
    assert list(p.distributions['x'].values())==pytest.approx(expected.detach().tolist())
    ids,mask=model.model.calls[0];assert ids.shape[1]%64==0
    assert all(mask[i,len(r['tokens']):].sum()==0 for i,r in enumerate(rows))
def test_limits_collision_identity_and_closed_fail_closed():
    with pytest.raises(ValueError,match='revision'):DeciderChoiceScorer(Model(),Tokenizer(),revision='main')
    with pytest.raises(ValueError,match='truncation'):DeciderChoiceScorer(Model(),Tokenizer(),max_context_tokens=2).score('state',questions())
    with pytest.raises(ValueError,match='batch'):DeciderChoiceScorer(Model(),Tokenizer(),max_batch_tokens=2).score('state',questions())
    class Collision(Tokenizer):
        def encode(self,text,**kw):return [5]
    with pytest.raises(ValueError,match='collision'):DeciderChoiceScorer(Model(),Collision())
    s=DeciderChoiceScorer(Model(),Tokenizer());s.close()
    with pytest.raises(RuntimeError,match='closed'):s.score('state',questions())

def test_native_tiny_hybrid_qwen35_eager_independent_row_parity():
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(151)
        config=Qwen3_5TextConfig(vocab_size=1100,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
            layer_types=['linear_attention','full_attention'],max_position_embeddings=4096,pad_token_id=0,eos_token_id=1)
        model=Qwen3_5ForCausalLM(config)
    scorer=DeciderChoiceScorer(model,Tokenizer())
    parallel=scorer.score('exposed fixture',questions())
    sequential=scorer.score('exposed fixture',questions(),parallel=False)
    for key,dist in parallel.distributions.items():assert dist==pytest.approx(sequential.distributions[key],abs=1e-6)
