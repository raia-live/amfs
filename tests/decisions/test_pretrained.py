from types import SimpleNamespace
import torch
from torch import nn
import pytest
from amfs_core.decisions import Question, Candidate
from amfs_decision_runtime.pretrained import PretrainedChoiceScorer, _cache_bytes


class Tokenizer:
    pad_token_id=0
    eos_token_id=1
    all_special_ids=[0,1]
    def encode(self,text,add_special_tokens=False):return [v+2 for v in text.encode()]
    def apply_chat_template(self,messages,**kwargs):
        return ''.join(f"<{m['role']}>\n{m['content']}\n" for m in messages)+'<assistant>\n'


class Cache:
    def __init__(self,ids):self.layers=[SimpleNamespace(keys=ids.float()[:,None,:,None].clone(),values=ids.float()[:,None,:,None].clone())]
    def batch_repeat_interleave(self,count):
        for layer in self.layers:
            layer.keys=layer.keys.repeat_interleave(count,0);layer.values=layer.values.repeat_interleave(count,0)


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.tensor(0.));self.calls=[]
    def forward(self,input_ids,attention_mask,position_ids,cache_position,use_cache,logits_to_keep,past_key_values=None):
        prefix=0 if past_key_values is None else past_key_values.layers[0].keys.shape[2]
        assert attention_mask.shape==(len(input_ids),prefix+input_ids.shape[1])
        assert torch.equal(cache_position,torch.arange(prefix,prefix+input_ids.shape[1]))
        assert torch.equal(position_ids,cache_position[None].expand(len(input_ids),-1))
        self.calls.append({'ids':input_ids.clone(),'mask':attention_mask.clone(),'positions':position_ids.clone(),'prefix':prefix})
        previous=torch.zeros((len(input_ids),0)) if past_key_values is None else past_key_values.layers[0].keys[:,0,:,0]
        all_ids=torch.cat((previous,input_ids),1)
        cumulative=(all_ids*attention_mask).cumsum(1)[:,prefix:]
        logits=(cumulative[:,:,None]%17-8)*torch.arange(258)[None,None,:]/1000
        positions=slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) and logits_to_keep else logits_to_keep if isinstance(logits_to_keep,torch.Tensor) else slice(None)
        if past_key_values is not None:
            # Deliberately mutate like DynamicCache: caller must isolate copies.
            past_key_values.layers[0].keys=all_ids[:,None,:,None]
            past_key_values.layers[0].values=all_ids[:,None,:,None]
        return SimpleNamespace(logits=logits[:,positions,:],past_key_values=Cache(all_ids))


def questions(count=4):
    return {f'field-{i}':Question(instructions='Choose next step'+'.'*i,candidates=[Candidate(id='retry_shared_prefix_A',description='retry '+'.'*i),Candidate(id='retry_shared_prefix_B',description='ask reviewer',kind='defer')]) for i in range(count)}


def scorer(model=None,tokenizer=None,**kwargs):
    return PretrainedChoiceScorer(model or Model(),tokenizer or Tokenizer(),model_id='fixture',revision='a'*40,**kwargs)


@pytest.mark.parametrize('fields',[1,4,16])
def test_parallel_sequential_equivalence_independent_cache_masks_and_positions(fields):
    model=Model();engine=scorer(model)
    batch=engine.score('observable state',questions(fields))
    assert len(model.calls)==2
    assert batch.diagnostics['calibrated'] is False
    assert batch.output_tokens==0
    assert batch.diagnostics['estimated_peak_cache_bytes']>batch.diagnostics['prefix_cache_bytes']
    suffix=model.calls[1];prefix=suffix['prefix']
    assert suffix['mask'].shape[0]==fields
    for i,length in enumerate(batch.diagnostics['suffix_tokens']):
        assert suffix['mask'][i,:prefix+length].all()
        assert not suffix['mask'][i,prefix+length:].any()
    model.calls=[]
    serial=engine.score('observable state',questions(fields),parallel=False)
    assert len(model.calls)==fields+1
    assert len({call['prefix'] for call in model.calls[1:]})==1
    for name,distribution in batch.distributions.items():
        assert distribution==pytest.approx(serial.distributions[name],abs=1e-7)
        assert sum(distribution.values())==pytest.approx(1.,abs=1e-7)
    assert list(batch.distributions)==list(questions(fields))


def test_candidate_ids_do_not_supply_first_token_logits_and_unknown_labels_rejected():
    model=Model();engine=scorer(model)
    result=engine.score('observable',questions(1))
    assert set(result.distributions['field-0'])=={'retry_shared_prefix_A','retry_shared_prefix_B'}
    assert len(set(result.distributions['field-0'].values()))==2
    invalid={'field':{'instructions':'choice','candidates':[{'id':'a','description':'A','gold':True},{'id':'b','description':'B'}]}}
    with pytest.raises(ValueError):engine.score('observable',invalid)


def test_token_collisions_and_context_boundary_are_fail_closed():
    class Collision(Tokenizer):
        def encode(self,text,**kwargs):return [9]
    with pytest.raises(ValueError,match='distinct'):scorer(tokenizer=Collision())
    class Boundary(Tokenizer):
        def encode(self,text,**kwargs):
            ids=super().encode(text,**kwargs)
            if text.endswith('<assistant>\nA'):ids[-2]=99
            return ids
    with pytest.raises(ValueError,match='boundary'):scorer(tokenizer=Boundary()).score('state',questions(1))


def test_cache_budget_token_bound_and_closed_state():
    engine=scorer(max_cache_bytes=1)
    with pytest.raises(ValueError,match='cache budget'):engine.score('state',questions())
    assert len(engine.model.calls)==1
    with pytest.raises(ValueError,match='token bound'):scorer(max_context_tokens=10).score('state',questions())
    engine=scorer();engine.close()
    with pytest.raises(RuntimeError,match='closed'):engine.score('state',questions())
    with pytest.raises(ValueError,match='revision'):PretrainedChoiceScorer(Model(),Tokenizer(),model_id='fixture',revision='main')


def test_real_tiny_qwen_dynamic_cache_matches_batched_and_sequential():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(91)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=258,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=4096,pad_token_id=0,eos_token_id=1,attention_dropout=0.))
    engine=scorer(model)
    parallel=engine.score('owned state',questions(4))
    sequential=engine.score('owned state',questions(4),parallel=False)
    for key in parallel.distributions:
        assert parallel.distributions[key]==pytest.approx(sequential.distributions[key],abs=1e-6)


def test_explicit_fp32_recipe_scopes_and_restores_precision_on_success_and_failure():
    class PrecisionModel(Model):
        def forward(self, *args, **kwargs):
            assert torch.get_float32_matmul_precision() == 'highest'
            return super().forward(*args, **kwargs)
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('medium')
        engine = scorer(PrecisionModel(), execution_mode='fp32_math')
        result = engine.score('exposed fixture', questions(4))
        assert result.diagnostics['execution_mode'] == 'fp32_math'
        assert result.diagnostics['weight_dtype'] == 'torch.float32'
        assert torch.get_float32_matmul_precision() == 'medium'
        with pytest.raises(ValueError, match='token bound'):
            scorer(PrecisionModel(), execution_mode='fp32_math', max_context_tokens=2).score('exposed', questions(1))
        assert torch.get_float32_matmul_precision() == 'medium'
        with pytest.raises(ValueError, match='float32'):
            scorer(PrecisionModel().to(dtype=torch.bfloat16), execution_mode='fp32_math')
    finally:
        torch.set_float32_matmul_precision(previous)


def test_loader_fp32_is_explicit_and_does_not_change_default_recipe(monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    loaded = []
    def load_model(*args, **kwargs):
        loaded.append(kwargs)
        return Model().to(dtype=kwargs['torch_dtype'])
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', load_model)
    default = PretrainedChoiceScorer.from_pretrained('fixture', revision='a'*40, device='cpu')
    explicit = PretrainedChoiceScorer.from_pretrained('fixture', revision='a'*40, device='cpu', dtype='fp32')
    assert default.execution_mode == 'default'
    assert explicit.execution_mode == 'fp32_math'
    assert all(k['trust_remote_code'] is False and k['revision'] == 'a'*40 for k in loaded)
    assert all(k['attn_implementation'] == 'sdpa' for k in loaded)
    with pytest.raises(ValueError, match='determined'):
        PretrainedChoiceScorer.from_pretrained('fixture', revision='a'*40, device='cpu', dtype='bf16', execution_mode='fp32_math')


def test_real_tiny_fp32_math_matches_uncached_full_prompt():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(116)
        config = Qwen2Config(vocab_size=258, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                            max_position_embeddings=4096, pad_token_id=0, eos_token_id=1)
        config._attn_implementation = 'sdpa'
        model = Qwen2ForCausalLM(config)
    engine = scorer(model, execution_mode='fp32_math')
    parallel = engine.score('exposed fixture', questions(4))
    sequential = engine.score('exposed fixture', questions(4), parallel=False)
    from amfs_decision_runtime.pretrained import _execution_context
    _, fields = engine._compile('exposed fixture', questions(4))
    with _execution_context('fp32_math'), torch.inference_mode():
        for field in fields:
            ids = torch.tensor([field['tokens']])
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            reference = torch.softmax(output.logits[0,-1,field['labels']].float(), -1).tolist()
            assert list(parallel.distributions[field['name']].values()) == pytest.approx(reference, abs=1e-6)
            assert parallel.distributions[field['name']] == pytest.approx(sequential.distributions[field['name']], abs=1e-6)
