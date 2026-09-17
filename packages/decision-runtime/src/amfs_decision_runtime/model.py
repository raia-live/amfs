"""State encoded once, candidate queries cross-attend to token-level state.

The tiny byte encoder is a deterministic offline test configuration, not a
pretrained model. Production artifacts must name a pinned backbone revision.
"""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass
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
