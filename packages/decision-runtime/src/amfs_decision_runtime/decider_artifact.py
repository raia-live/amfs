"""Local-only immutable Mapika PEFT serving artifact; not runtime dispatch.

An operator must curate this serving bundle and supply its trusted digest and
ownership bindings. Development training manifests are deliberately incompatible.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from uuid import UUID

from .decider import DeciderChoiceScorer, MODEL_ID, MODEL_REVISION

SCHEMA = 'amfs.mapika-peft.v1'
MAX_BYTES = 12 * 1024**3
SHA = re.compile(r'[a-f0-9]{64}')
FIELDS = {'schema','account_id','model_id','version','spec_hash','base_id','base_revision','files'}


def _json(path):
    if path.stat().st_size > 1024**2:
        raise ValueError('bounded JSON metadata required')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate metadata key')
            result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=pairs)


def _sha(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        while block:=handle.read(1024**2):value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class ArtifactIdentity:
    account_id: str
    model_id: str
    version: str
    spec_hash: str
    artifact_sha256: str
    base_revision: str
    adapter_sha256: str
    tokenizer_sha256: str


def validate_artifact(directory, *, expected_digest, account_id, model_id, version, spec_hash):
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():raise ValueError('real artifact directory required')
    if not isinstance(expected_digest,str) or not SHA.fullmatch(expected_digest):raise ValueError('trusted artifact digest required')
    manifest_path=root/'manifest.json'
    if manifest_path.is_symlink() or manifest_path.stat().st_size>1024**2 or _sha(manifest_path)!=expected_digest:raise ValueError('artifact manifest digest mismatch')
    manifest=_json(manifest_path)
    if not isinstance(manifest,dict) or set(manifest)!=FIELDS or manifest['schema']!=SCHEMA:raise ValueError('unsupported serving manifest')
    expected={'account_id':str(UUID(str(account_id))),'model_id':str(UUID(str(model_id))), 'version':version,'spec_hash':spec_hash}
    if (not isinstance(version,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',version)
        or not isinstance(spec_hash,str) or not SHA.fullmatch(spec_hash)
        or any(manifest[k]!=v for k,v in expected.items())):raise ValueError('artifact ownership/spec/version mismatch')
    if (manifest['base_id'],manifest['base_revision'])!=(MODEL_ID,MODEL_REVISION):raise ValueError('unsupported base identity')
    files=manifest['files']
    required={'base/config.json','base/tokenizer.json','base/tokenizer_config.json',
              'adapter/adapter_config.json','adapter/adapter_model.safetensors'}
    if not isinstance(files,dict) or not required.issubset(files) or not 5<=len(files)<=128:raise ValueError('bounded complete file manifest required')
    total=0
    for name,sha in files.items():
        if not isinstance(name,str):raise ValueError('unsafe artifact file declaration')
        path=PurePosixPath(name)
        if (not isinstance(name,str) or '\\' in name or path.is_absolute() or '..' in path.parts
            or str(path)!=name or len(path.parts)!=2 or path.parts[0] not in ('base','adapter')
            or path.suffix not in ('.json','.safetensors','.txt','.jinja','.model')
            or not isinstance(sha,str) or not SHA.fullmatch(sha)):
            raise ValueError('unsafe artifact file declaration')
        target=root/name
        if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe artifact path')
        total+=target.stat().st_size
        if total>MAX_BYTES:raise ValueError('artifact size bound exceeded')
        if _sha(target)!=sha:raise ValueError('artifact file digest mismatch')
    actual=set()
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('symlinks prohibited')
        if path.is_file():actual.add(path.relative_to(root).as_posix())
        if len(actual)>129:raise ValueError('artifact file count exceeded')
    if actual != set(files)|{'manifest.json'}:raise ValueError('unmanifested artifact files')
    for name in (n for n in files if n.endswith('.json')):
        metadata=_json(root/name)
        if isinstance(metadata,dict) and metadata.get('auto_map'):raise ValueError('remote model/tokenizer code prohibited')
    config=_json(root/'base/config.json')
    if config.get('model_type')!='qwen3_5_text':raise ValueError('native text architecture required')
    weights={n for n in files if n.startswith('base/') and n.endswith('.safetensors')}
    if not weights:raise ValueError('base safetensor weights required')
    if 'base/model.safetensors.index.json' in files:
        index=_json(root/'base/model.safetensors.index.json')
        mapped=set(index.get('weight_map',{}).values())
        if not mapped or {'base/'+n for n in mapped}!=weights:raise ValueError('weight shard manifest mismatch')
    elif weights!={'base/model.safetensors'}:raise ValueError('single or indexed base weights required')
    adapter=_json(root/'adapter/adapter_config.json')
    if (adapter.get('peft_type')!='LORA' or adapter.get('task_type')!='CAUSAL_LM'
        or adapter.get('base_model_name_or_path')!=MODEL_ID or adapter.get('revision') not in (None,MODEL_REVISION)
        or adapter.get('bias','none')!='none' or adapter.get('modules_to_save')
        or adapter.get('auto_mapping') or adapter.get('use_dora',False)
        or adapter.get('trainable_token_indices') or adapter.get('target_parameters') or adapter.get('layer_replication')
        or adapter.get('lora_bias',False) or adapter.get('use_qalora',False)
        or type(adapter.get('r')) is not int or not 1<=adapter['r']<=128
        or type(adapter.get('lora_alpha')) not in (int,float) or not math.isfinite(adapter['lora_alpha'])
        or not 0<adapter['lora_alpha']<=1024):
        raise ValueError('unsupported adapter configuration')
    targets=adapter.get('target_modules')
    if (not isinstance(targets,list) or not targets or len(targets)>512 or len(set(targets))!=len(targets)
        or any(not isinstance(n,str) or not re.fullmatch(r'model\.layers\.\d+\.(self_attn|linear_attn|mlp)\.[a-z_]+',n) for n in targets)):
        raise ValueError('explicit language-layer LoRA targets required')
    return ArtifactIdentity(**expected,artifact_sha256=expected_digest,base_revision=MODEL_REVISION,
        adapter_sha256=files['adapter/adapter_model.safetensors'],tokenizer_sha256=files['base/tokenizer.json'])


class AdapterChoiceScorer(DeciderChoiceScorer):
    """Each instance owns its adapted model; no shared set_adapter mutation."""
    def __init__(self,model,tokenizer,identity,**kwargs):
        self._artifact_identity=identity
        super().__init__(model,tokenizer,**kwargs)
    @property
    def artifact_identity(self):return self._artifact_identity
    def score(self,state,questions,*,parallel=True):
        result=super().score(state,questions,parallel=parallel)
        identity=self.artifact_identity
        return replace(result,model_version=identity.version,engine='mapika_peft',diagnostics={**result.diagnostics,
            'artifact_sha256':identity.artifact_sha256,'adapter_sha256':identity.adapter_sha256,
            'base_revision':identity.base_revision,'tokenizer_sha256':identity.tokenizer_sha256,'spec_hash':identity.spec_hash})


def load_artifact(directory, *, expected_digest, account_id, model_id, version, spec_hash, device='cpu', **scorer_options):
    identity=validate_artifact(directory,expected_digest=expected_digest,account_id=account_id,model_id=model_id,version=version,spec_hash=spec_hash)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    if str(device).startswith('cuda') and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):raise ValueError('CUDA BF16 required; no fallback')
    root=Path(directory)
    tokenizer=AutoTokenizer.from_pretrained(root/'base',local_files_only=True,trust_remote_code=False)
    base=AutoModelForCausalLM.from_pretrained(root/'base',local_files_only=True,trust_remote_code=False,
        use_safetensors=True,dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32).to(device)
    adapted=PeftModel.from_pretrained(base,root/'adapter',local_files_only=True,is_trainable=False)
    # PEFT's wrapper.model is not necessarily the causal backbone expected by
    # DeciderChoiceScorer. Its base retains the injected LoRA layers themselves.
    model=adapted.get_base_model().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return AdapterChoiceScorer(model,tokenizer,identity,**scorer_options)
