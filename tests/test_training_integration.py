"""Optional CPU integration with a tiny random Qwen; exclusively synthetic tokens."""

import importlib

import pytest


def test_real_peft_loss_mask_frozen_base_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    peft = importlib.import_module("peft")
    tf = importlib.import_module("transformers")
    torch.manual_seed(42)
    config = tf.Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    base = tf.Qwen2ForCausalLM(config)
    initial = {n: p.detach().clone() for n, p in base.named_parameters()}
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, -100, 4, 5, -100]]),
    }
    result = model(**batch)
    manual = torch.nn.functional.cross_entropy(
        result.logits[:, :-1].reshape(-1, 128),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(result.loss, manual)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    result.loss.backward()
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    optimizer.step()
    assert any(p.detach().abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)
    for name, parameter in model.base_model.model.named_parameters():
        if "lora_" in name:
            continue
        assert torch.equal(parameter.detach(), initial[name.replace(".base_layer", "")])
    model.save_pretrained(tmp_path / "adapter")
    fresh = tf.Qwen2ForCausalLM(config)
    fresh.load_state_dict(initial)
    restored = peft.PeftModel.from_pretrained(fresh, tmp_path / "adapter").eval()
    with torch.no_grad():
        assert torch.allclose(model(**batch).logits, restored(**batch).logits, atol=1e-6)
