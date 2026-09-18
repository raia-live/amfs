from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from amfs_decision_runtime.decider_artifact import validate_artifact,load_artifact,SCHEMA
from amfs_decision_runtime.decider import MODEL_ID,MODEL_REVISION,DeciderChoiceScorer

_spec=importlib.util.spec_from_file_location('artifact_fixture',Path(__file__).with_name('test_decider.py'))
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
Model,Tokenizer,questions=_fixture.Model,_fixture.Tokenizer,_fixture.questions


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@pytest.fixture
def artifact(tmp_path):
    (tmp_path/'base').mkdir();(tmp_path/'adapter').mkdir()
    docs={'base/config.json':{'model_type':'qwen3_5_text'},'base/tokenizer.json':{},'base/tokenizer_config.json':{},
          'adapter/adapter_config.json':{'peft_type':'LORA','task_type':'CAUSAL_LM','base_model_name_or_path':MODEL_ID,
              'r':8,'lora_alpha':16,'target_modules':['model.layers.0.self_attn.q_proj']}}
    for name,doc in docs.items():(tmp_path/name).write_text(json.dumps(doc))
    for name in ('base/model.safetensors','adapter/adapter_model.safetensors'):(tmp_path/name).write_bytes(b'mocked weights')
    manifest={'schema':SCHEMA,'account_id':'00000000-0000-0000-0000-000000000001','model_id':'00000000-0000-0000-0000-000000000002',
              'version':'trained-v1','spec_hash':'a'*64,'base_id':MODEL_ID,'base_revision':MODEL_REVISION,
              'files':{p.relative_to(tmp_path).as_posix():sha(p) for p in tmp_path.rglob('*') if p.is_file()}}
    def bind():
        (tmp_path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True))
        return {'expected_digest':sha(tmp_path/'manifest.json'),**{k:manifest[k] for k in ('account_id','model_id','version','spec_hash')}}
    return tmp_path,manifest,bind


def test_strict_identity_and_tampered_files(artifact):
    root,manifest,bind=artifact;kwargs=bind();identity=validate_artifact(root,**kwargs)
    assert identity.version=='trained-v1' and identity.adapter_sha256==manifest['files']['adapter/adapter_model.safetensors']
    with pytest.raises(FrozenInstanceError):identity.version='other'
    for key,value in [('account_id','00000000-0000-0000-0000-000000000003'),('model_id','00000000-0000-0000-0000-000000000003'),('version','other'),('spec_hash','b'*64)]:
        with pytest.raises(ValueError):validate_artifact(root,**(kwargs|{key:value}))
    (root/'adapter/adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='digest'):validate_artifact(root,**kwargs)


def test_remote_code_extra_files_traversal_and_base_pin_rejected(artifact):
    root,m,bind=artifact
    m['base_revision']='0'*40
    with pytest.raises(ValueError,match='base identity'):validate_artifact(root,**bind())
    m['base_revision']=MODEL_REVISION;m['files']['../external.safetensors']='a'*64
    with pytest.raises(ValueError,match='unsafe'):validate_artifact(root,**bind())
    del m['files']['../external.safetensors'];(root/'run.py').write_text('raise RuntimeError')
    with pytest.raises(ValueError,match='unmanifested'):validate_artifact(root,**bind())
    (root/'run.py').unlink();p=root/'base/tokenizer_config.json';p.write_text(json.dumps({'auto_map':{'AutoTokenizer':'evil'}}));m['files']['base/tokenizer_config.json']=sha(p)
    with pytest.raises(ValueError,match='remote'):validate_artifact(root,**bind())


def test_local_loader_attaches_weights_and_exposes_adapter_identity(artifact,monkeypatch):
    import transformers,peft,torch
    root,m,bind=artifact;kwargs=bind();calls=[]
    def tokenizer(path,**kw):
        assert path==root/'base' and kw=={'local_files_only':True,'trust_remote_code':False};return Tokenizer()
    def base(path,**kw):
        assert kw['local_files_only'] is True and kw['trust_remote_code'] is False and kw['use_safetensors'] is True
        return Model()
    def adapter(model,path,**kw):
        calls.append(path)
        assert path==root/'adapter' and kw=={'local_files_only':True,'is_trainable':False}
        with torch.no_grad():model.lm_head.weight[300]+=5
        return SimpleNamespace(get_base_model=lambda:model)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',base)
    monkeypatch.setattr(peft.PeftModel,'from_pretrained',adapter)
    scorer=load_artifact(root,**kwargs);result=scorer.score('same',questions())
    plain=DeciderChoiceScorer(Model(),Tokenizer()).score('same',questions())
    assert result.distributions!=plain.distributions and calls==[root/'adapter']
    assert result.model_version=='trained-v1' and result.engine=='mapika_peft'
    assert result.diagnostics['artifact_sha256']==kwargs['expected_digest']
    assert result.diagnostics['adapter_sha256']==m['files']['adapter/adapter_model.safetensors']
    assert all(not p.requires_grad for p in scorer.model.parameters())


def test_symlink_and_duplicate_metadata_rejected(artifact):
    root,m,bind=artifact;kwargs=bind()
    p=root/'base/tokenizer.json';p.unlink();p.symlink_to(root/'base/config.json')
    with pytest.raises(ValueError):validate_artifact(root,**kwargs)
    p.unlink();p.write_text('{}');m['files']['base/tokenizer.json']=sha(p)
    bind();p=root/'manifest.json';p.write_text(p.read_text()[:-1]+',"schema":"'+SCHEMA+'"}')
    with pytest.raises(ValueError,match='duplicate'):validate_artifact(root,**(kwargs|{'expected_digest':sha(p)}))
