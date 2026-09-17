"""State encoded once, candidate queries cross-attend to token-level state.

The tiny byte encoder is a deterministic offline test configuration, not a
pretrained model. Production artifacts must name a pinned backbone revision.
"""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn


def artifact_digest(directory: str | Path) -> str:
    """Bind weights, configuration and tokenizer to the same serving identity."""
    root = Path(directory).resolve()
    files = [root / "config.json", root / "model.safetensors"]
    if (root / "backbone").exists():
        files.extend(p for p in (root / "backbone").rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.resolve().is_relative_to(root):
            raise ValueError("artifact files must remain within the artifact directory")
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(str(path.relative_to(root)).encode() + b"\0" + file_digest.digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class ScorerConfig:
    backbone: str | None = None
    revision: str | None = None
    hidden_size: int = 128
    heads: int = 4
    max_state_tokens: int = 2048
    max_candidate_tokens: int = 256
    version: str = "experimental-v1"
    architecture: str = "senselab.shared-state-cross-attention.v1"
    lora_rank: int | None = None
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("query", "value")
    lora_merged: bool = False

    def __post_init__(self):
        if self.lora_rank is not None:
            if not self.backbone or self.lora_rank < 1 or self.lora_alpha < 1:
                raise ValueError("LoRA requires a pretrained backbone and positive rank/alpha")
            if not 0 <= self.lora_dropout < 1 or not self.lora_targets:
                raise ValueError("invalid LoRA dropout or target modules")
        elif self.lora_merged:
            raise ValueError("merged LoRA metadata requires a rank")


class DecisionScorer(nn.Module):
    def __init__(self, config: ScorerConfig, *, backbone_directory: Path | None = None):
        super().__init__()
        self.config = config
        self.tokenizer = None
        if config.backbone:
            if not config.revision or not re.fullmatch(r"[0-9a-f]{40}", config.revision):
                raise ValueError("a pretrained backbone requires a pinned revision")
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            if backbone_directory is not None:
                encoder_config = AutoConfig.from_pretrained(backbone_directory, local_files_only=True,
                                                            trust_remote_code=False)
                self.encoder = AutoModel.from_config(encoder_config, trust_remote_code=False)
                self.tokenizer = AutoTokenizer.from_pretrained(backbone_directory, local_files_only=True,
                                                               trust_remote_code=False)
            else:
                self.encoder = AutoModel.from_pretrained(
                    config.backbone, revision=config.revision, trust_remote_code=False
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.backbone, revision=config.revision, trust_remote_code=False
                )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                raise ValueError("backbone tokenizer requires a padding or EOS token")
            width = self.encoder.config.hidden_size
            self.projection = nn.Linear(width, config.hidden_size)
        else:
            self.embedding = nn.Embedding(257, config.hidden_size, padding_idx=0)
            self.positions = nn.Embedding(max(config.max_state_tokens, config.max_candidate_tokens), config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                config.hidden_size, config.heads, config.hidden_size * 4,
                dropout=0, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        if config.lora_rank is not None and not config.lora_merged:
            from peft import LoraConfig, get_peft_model
            # Each scorer owns its adapter. Never mutate a shared backbone to
            # switch customers during concurrent inference.
            adapter = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
                                 lora_dropout=config.lora_dropout, bias="none",
                                 target_modules=list(config.lora_targets))
            self.encoder = get_peft_model(self.encoder, adapter)
        self.attention = nn.MultiheadAttention(config.hidden_size, config.heads, batch_first=True)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.decision_head = nn.Linear(config.hidden_size, 1)
        # Separate heads: observational success and acquisition utility are not
        # decision probabilities. Do not expose them unless separately trained/evaluated.
        self.outcome_head = nn.Linear(config.hidden_size, 1)
        self.utility_head = nn.Linear(config.hidden_size, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, texts: list[str], limit: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer:
            # Artifact limits cannot increase the pretrained encoder's context.
            # Reject explicitly instead of truncating state or failing inside
            # positional embeddings (e.g. BERT's 512-position capacity).
            for maximum in (getattr(self.encoder.config, "max_position_embeddings", None),
                            getattr(self.tokenizer, "model_max_length", None)):
                if isinstance(maximum, int) and maximum > 0:
                    limit = min(limit, maximum)
            encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            if encoded.input_ids.shape[1] > limit:
                raise ValueError("text exceeds model token limit; explicit state compilation required")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            hidden = self.encoder(**encoded).last_hidden_state
            return self.projection(hidden), encoded["attention_mask"].bool()
        values = [[b + 1 for b in text.encode()] or [1] for text in texts]
        if any(len(row) > limit for row in values):
            raise ValueError("text exceeds experimental byte encoder limit")
        size = max(map(len, values))
        ids = torch.zeros((len(values), size), dtype=torch.long, device=self.device)
        for i, row in enumerate(values):
            ids[i, :len(row)] = torch.tensor(row, device=self.device)
        mask = ids != 0
        positions = torch.arange(size, device=self.device)
        hidden = self.embedding(ids) + self.positions(positions)
        return self.encoder(hidden, src_key_padding_mask=~mask), mask

    def forward(self, state: str, candidate_queries: list[str]) -> dict[str, torch.Tensor]:
        if not candidate_queries:
            raise ValueError("at least one candidate is required")
        memory, memory_mask = self.encode([state], self.config.max_state_tokens)
        queries, query_mask = self.encode(candidate_queries, self.config.max_candidate_tokens)
        pooled = (queries * query_mask.unsqueeze(-1)).sum(1) / query_mask.sum(1, keepdim=True)
        # One shared state encoding, batched candidate query attention.
        attended, _ = self.attention(
            pooled.unsqueeze(0), memory, memory, key_padding_mask=~memory_mask, need_weights=False
        )
        representation = self.norm(pooled + attended.squeeze(0))
        return {
            "logits": self.decision_head(representation).squeeze(-1),
            "outcome_logits": self.outcome_head(representation).squeeze(-1),
            "utility": self.utility_head(representation).squeeze(-1),
        }

    def merge_adapter(self) -> None:
        """Finalize a training model into a standalone immutable serving model.

        Call only after training/checkpoint selection, never on a live server.
        The merged export loads without the optional PEFT dependency.
        """
        if self.config.lora_rank is not None and not self.config.lora_merged:
            self.encoder = self.encoder.merge_and_unload(safe_merge=True)
            self.config = replace(self.config, lora_merged=True)

    def save(self, directory: str | Path) -> None:
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        if self.tokenizer is not None:
            self.encoder.config.save_pretrained(directory / "backbone")
            self.tokenizer.save_pretrained(directory / "backbone")
        save_file({k: v.detach().cpu().contiguous().clone() for k, v in self.state_dict().items()},
                  str(directory / "model.safetensors"))

    @classmethod
    def load(cls, directory: str | Path) -> DecisionScorer:
        from safetensors.torch import load_file
        directory = Path(directory)
        config = ScorerConfig(**json.loads((directory / "config.json").read_text()))
        if config.architecture != "senselab.shared-state-cross-attention.v1":
            raise ValueError("unsupported artifact architecture")
        model = cls(config, backbone_directory=directory / "backbone" if config.backbone else None)
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model.eval()
