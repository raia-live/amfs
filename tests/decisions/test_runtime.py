import pytest
torch = pytest.importorskip("torch")
from amfs_decision_runtime import DecisionScorer, ScorerConfig
from amfs_decision_runtime.server import ArtifactPool
from amfs_decision_runtime.model import artifact_digest


def test_candidate_permutation_and_artifact_roundtrip(tmp_path):
    torch.manual_seed(1)
    model = DecisionScorer(ScorerConfig(hidden_size=16, max_state_tokens=64, max_candidate_tokens=32)).eval()
    with torch.inference_mode():
        first = model("failure", ["retry", "inspect"])["logits"]
        reversed_scores = model("failure", ["inspect", "retry"])["logits"]
        assert torch.allclose(first, reversed_scores.flip(0), atol=1e-6)
        model.save(tmp_path)
        restored = DecisionScorer.load(tmp_path)
        assert torch.allclose(first, restored("failure", ["retry", "inspect"])["logits"], atol=1e-6)
    digest = artifact_digest(tmp_path)
    pool = ArtifactPool({"account-a/model-a/v1": {"directory": str(tmp_path), "version": model.config.version, "sha256": digest}}, 1)
    assert pool.get("account-a/model-a/v1").config.version == model.config.version
    with pytest.raises(Exception) as failure:
        pool.get("account-b/model-a/v1")
    assert failure.value.status_code == 404


def test_explicit_limits_and_pinned_revision():
    model = DecisionScorer(ScorerConfig(hidden_size=16, max_state_tokens=8))
    with pytest.raises(ValueError, match="limit"):
        model("more than eight bytes", ["retry"])
    with pytest.raises(ValueError, match="pinned revision"):
        DecisionScorer(ScorerConfig(backbone="example/model", revision="main"))


def test_pretrained_architecture_export_loads_offline(tmp_path, monkeypatch):
    from transformers import BertConfig, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    backbone = tmp_path / "seed-backbone"
    BertConfig(vocab_size=6, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
               intermediate_size=32).save_pretrained(backbone)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "failure": 2, "retry": 3, "inspect": 4, "[EOS]": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]").save_pretrained(backbone)
    model = DecisionScorer(ScorerConfig(backbone="local-test-only", revision="a" * 40, hidden_size=16), backbone_directory=backbone).eval()
    destination = tmp_path / "artifact"
    model.save(destination)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    restored = DecisionScorer.load(destination)
    with torch.inference_mode():
        assert torch.allclose(model("failure", ["retry", "inspect"])["logits"],
                              restored("failure", ["retry", "inspect"])["logits"], atol=1e-6)


def test_multiple_model_eviction_and_configuration_integrity(tmp_path):
    registry = {}
    for key in ("account-a:recovery:v1", "account-a:triage:v2"):
        directory = tmp_path / key.replace(":", "-")
        version = key.rsplit(":", 1)[-1]
        DecisionScorer(ScorerConfig(hidden_size=16, version=version)).save(directory)
        registry[key] = {"directory": str(directory), "version": version, "sha256": artifact_digest(directory)}
    pool = ArtifactPool(registry, 1)
    keys = list(registry)
    assert pool.get(keys[0]).config.version == "v1"
    assert pool.get(keys[1]).config.version == "v2"
    assert list(pool.cache) == [keys[1]]
    assert pool.get(keys[0]).config.version == "v1"
    config = tmp_path / keys[1].replace(":", "-") / "config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(Exception) as failure:
        pool.get(keys[1])
    assert failure.value.status_code == 503
