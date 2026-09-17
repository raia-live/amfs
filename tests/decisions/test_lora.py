"""Optional offline PEFT tests; no hub downloads or third-party credentials."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import BertConfig, BertModel, BertTokenizer
from amfs_decision_runtime import DecisionScorer, ScorerConfig
from amfs_decision_runtime.model import artifact_digest


def test_lora_reduces_parameters_and_merged_export_loads_offline(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nstate\nretry\nwait\n")
    BertTokenizer(vocab_file=str(source / "vocab.txt")).save_pretrained(source)
    BertModel(BertConfig(vocab_size=8, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64)).save_pretrained(source)
    model = DecisionScorer(ScorerConfig(backbone=str(source), revision="a" * 40,
                           hidden_size=16, heads=2, lora_rank=2, lora_alpha=4))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable < total / 4
    assert all(not p.requires_grad for name, p in model.encoder.named_parameters() if "lora_" not in name)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        logits = model("state", ["retry", "wait"])["logits"]
        (-logits.log_softmax(-1)[0]).backward()
        optimizer.step()
    model.eval()
    before = model("state", ["retry", "wait"])["logits"].detach()
    checkpoint = tmp_path / "checkpoint"
    model.save(checkpoint)
    restored = DecisionScorer.load(checkpoint)
    assert torch.allclose(before, restored("state", ["retry", "wait"])["logits"], atol=1e-6)
    restored.merge_adapter()
    exported = tmp_path / "exported"
    restored.save(exported)
    assert json.loads((exported / "config.json").read_text())["lora_merged"] is True
    # Merged serving must not depend on adapter selection or PEFT import.
    import peft
    monkeypatch.setattr(peft, "get_peft_model", lambda *a, **k: pytest.fail("merged artifact imported adapter"))
    loaded = DecisionScorer.load(exported)
    assert torch.allclose(before, loaded("state", ["retry", "wait"])["logits"], atol=1e-5)
    digest = artifact_digest(exported)
    config = json.loads((exported / "config.json").read_text())
    config["lora_alpha"] = 8
    (exported / "config.json").write_text(json.dumps(config))
    assert artifact_digest(exported) != digest


def test_lora_requires_explicit_pretrained_configuration():
    with pytest.raises(ValueError, match="pretrained"):
        ScorerConfig(lora_rank=4)
    with pytest.raises(ValueError, match="rank"):
        ScorerConfig(lora_merged=True)
