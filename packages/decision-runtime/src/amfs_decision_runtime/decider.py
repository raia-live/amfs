"""Pinned Mapika v8 decision baseline, using its native state-first choice format.

Clean inference implementation based on the documented native format; no remote
helper code, CUDA graphs or schema cache. Every question has an independent row.
The checkpoint temperature is preserved, but probabilities are not calibrated on
our workflows. This eager baseline is not the author's optimized latency claim.
"""
from __future__ import annotations
import gc
import itertools
import string
import threading
from collections.abc import Mapping
import torch
from amfs_core.decisions import Question
from .pretrained import ChoiceScores, canonical

MODEL_ID = 'Mapika/decider-2b'
MODEL_REVISION = '1d96be0093133e194fe18105a521b3e69be931d2'
CODE_REVISION = '1c15a48f199adc0eb1952b17441516b5358d79a0'
TEMPERATURE = 1.3


class DeciderChoiceScorer:
    def __init__(self, model, tokenizer, *, revision=MODEL_REVISION,
                 max_fields=16, max_context_tokens=4096, max_batch_tokens=32768):
        if revision != MODEL_REVISION:
            raise ValueError('audited immutable Mapika revision required')
        if type(max_fields) is not int or not 1 <= max_fields <= 16 or max_context_tokens < 1 or max_batch_tokens < 1:
            raise ValueError('bounded inference configuration required')
        self.model, self.tokenizer = model.eval(), tokenizer
        self.revision = revision
        self.max_fields, self.max_context_tokens, self.max_batch_tokens = max_fields, max_context_tokens, max_batch_tokens
        self.lock = threading.Lock()
        # Native head computes the full255 label projection before option masking.
        names = itertools.chain(string.ascii_uppercase, (''.join(v) for v in itertools.product(string.ascii_uppercase, repeat=2)))
        self.labels = []
        specials = set(getattr(tokenizer, 'all_special_ids', []))
        for name in names:
            ids = tokenizer.encode(name, add_special_tokens=False)
            if len(ids) == 1:
                if ids[0] in specials or ids[0] in self.labels:
                    raise ValueError('native label collision or special token')
                self.labels.append(ids[0])
                if len(self.labels) == 255: break
        if len(self.labels) != 255 or any(tokenizer.encode(c, add_special_tokens=False) != [self.labels[i]] for i,c in enumerate('ABCDEFGHIJ')):
            raise ValueError('native255 label table unavailable')

    @classmethod
    def from_pretrained(cls, model_id=MODEL_ID, *, revision=MODEL_REVISION, device='cuda', local_files_only=False, **kwargs):
        if model_id != MODEL_ID or revision != MODEL_REVISION:
            raise ValueError('audited model identity required')
        if str(device).startswith('cuda') and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise ValueError('CUDA BF16 required; no precision fallback')
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False, local_files_only=local_files_only)
        if config.model_type != 'qwen3_5_text' or getattr(config, 'auto_map', None):
            raise ValueError('native audited model architecture required')
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=False, local_files_only=local_files_only,
                    dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32).to(device)
        return cls(model, tokenizer, revision=revision, **kwargs)

    def _compile(self, state, questions):
        if not isinstance(state, str) or len(state.encode()) > 1024**2 or not isinstance(questions, Mapping) or not 1 <= len(questions) <= self.max_fields:
            raise ValueError('bounded state and questions required')
        context = self.tokenizer.encode('Context:\n'+state, add_special_tokens=False)
        rows = []
        for name, raw in questions.items():
            if not isinstance(name, str) or not 1 <= len(name) <= 128:
                raise ValueError('bounded question ID required')
            q = Question.model_validate(raw.model_dump(mode='json') if hasattr(raw, 'model_dump') else raw)
            if q.type != 'choice' or q.depends_on or not 2 <= len(q.candidates) <= 10:
                raise ValueError('independent choice questions with2..10 candidates required')
            if not q.instructions or len({c.id for c in q.candidates}) != len(q.candidates):
                raise ValueError('instructions and unique candidate IDs required')
            options = [f'{c.id}: '+canonical({'description':c.description,'kind':c.kind,'cost':c.cost}) for c in q.candidates]
            text = '\n\nQuestion: '+q.instructions+'\nOptions:'+''.join(f'\n({string.ascii_uppercase[i]}) {o}' for i,o in enumerate(options))+'\nAnswer: ('
            ids = context+self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids)>self.max_context_tokens:
                raise ValueError('context bound exceeded; native silent truncation disabled')
            rows.append({'name':name,'ids':[c.id for c in q.candidates],'tokens':ids})
        return rows

    def score(self, state, questions, *, parallel=True):
        with self.lock, torch.inference_mode():
            if self.model is None: raise RuntimeError('scorer is closed')
            rows = self._compile(state, questions)
            width = ((max(len(r['tokens']) for r in rows)+63)//64)*64
            if width*(len(rows) if parallel else 1)>self.max_batch_tokens:
                raise ValueError('batch token budget exceeded')
            pad = self.tokenizer.pad_token_id
            if pad is None: pad = self.tokenizer.eos_token_id
            if pad is None: raise ValueError('padding or EOS token required')
            device = next(self.model.parameters()).device
            distributions = {}; groups=[rows] if parallel else [[r] for r in rows]
            for group in groups:
                length=((max(len(r['tokens']) for r in group)+63)//64)*64
                ids=torch.full((len(group),length),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
                for i,r in enumerate(group):
                    ids[i,:len(r['tokens'])]=torch.tensor(r['tokens'],device=device);mask[i,:len(r['tokens'])]=1
                hidden=self.model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                slots=hidden[torch.arange(len(group),device=device),torch.tensor([len(r['tokens'])-1 for r in group],device=device)]
                logits=torch.nn.functional.linear(slots,self.model.lm_head.weight[self.labels]).float()
                for i,r in enumerate(group):
                    selected=logits[i,:len(r['ids'])]
                    if not torch.isfinite(selected).all():raise ValueError('nonfinite native option logits')
                    probs=torch.softmax(selected/TEMPERATURE,-1).cpu().tolist()
                    distributions[r['name']]=dict(zip(r['ids'],probs))
            return ChoiceScores(distributions,MODEL_ID+'@'+self.revision,sum(len(r['tokens']) for r in rows),0,
                {'calibrated':False,'temperature':TEMPERATURE,'native_layout':'state_first_independent_rows','parallel':parallel,
                 'fields':len(rows),'weight_dtype':str(next(self.model.parameters()).dtype),'source_code_revision':CODE_REVISION,
                 'context_truncated':False,'cuda_graphs':False,'shared_prefix_cache':False},engine='mapika_decider')

    def close(self):
        with self.lock:
            self.model=None;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
