"""Synthetic-only supervision and reproducibility checks; no model downloads."""

from pathlib import Path

import pytest

from model_adaptation_lab.io import LabError
from model_adaptation_lab.models.loader import yaml_config
from model_adaptation_lab.models.lora import validate_training
from model_adaptation_lab.models.training_data import AnswerCollator, encode_records, nested_subset


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        tokens = [1, 2, 3]
        if not add_generation_prompt:
            tokens += [ord(c) for c in messages[-1]["content"]] + [99]
        return tokens


def rows():
    return [
        {
            "question_id": str(i),
            "question": "Synthetic question",
            "answer": "OK",
            "answerable": i % 2 == 0,
        }
        for i in range(12)
    ]


def test_assistant_boundary_and_eos():
    data, audit = encode_records(rows()[:2], Tokenizer(), {"system": "S", "abstention": "No"}, 12)
    assert data[0]["labels"] == [-100, -100, -100, 79, 75, 99]
    assert data[1]["labels"] == [-100, -100, -100, 78, 111, 99]
    assert audit["supervised_tokens"] == 6


def test_excludes_whole_overflow_not_partial_answer():
    sample = rows()[:2]
    sample[0]["answer"] = "A" * 20
    data, audit = encode_records(sample, Tokenizer(), {"system": "S", "abstention": "No"}, 12)
    assert len(data) == 1
    assert audit["excluded"] == [{"question_id": "0", "tokens": 24, "reason": "sequence_overflow"}]
    with pytest.raises(LabError, match="No examples"):
        encode_records(sample, Tokenizer(), {"system": "S", "abstention": "No"}, 2)


def test_nested_subsets_are_order_independent():
    quarter = nested_subset(rows(), 0.25, 42)
    half = nested_subset(list(reversed(rows())), 0.5, 42)
    full = nested_subset(rows(), 1.0, 42)
    assert quarter == half[:3] == full[:3]
    assert half == full[:6]
    with pytest.raises(LabError):
        nested_subset(rows(), 0.8, 42)


def test_padding_loss_mask():
    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(values, dtype):
            return values

    batch = AnswerCollator(Tokenizer(), FakeTorch)(
        [
            {"input_ids": [1, 5, 99], "attention_mask": [1, 1, 1], "labels": [-100, 5, 99]},
            {"input_ids": [1, 99], "attention_mask": [1, 1], "labels": [-100, 99]},
        ]
    )
    assert batch["labels"][1] == [-100, 99, -100]
    assert batch["attention_mask"][1] == [1, 1, 0]
    assert batch["input_ids"][1] == [1, 99, 0]


@pytest.mark.parametrize("key,value", [("batch_size", 0), ("seed", -1), ("quantization", "4bit")])
def test_invalid_training_config(key, value):
    config = yaml_config(Path("configs/training.yaml"))
    validate_training(config)
    config["training"][key] = value
    with pytest.raises(LabError):
        validate_training(config)


def test_adapter_provenance_and_smoke_guard(tmp_path, synthetic_protocol):
    from model_adaptation_lab.io import atomic_write, canonical, file_hash
    from model_adaptation_lab.models.adapter import verify_adapter

    manifests, _ = synthetic_protocol
    model = yaml_config(Path("configs/model.yaml"))
    adapter = tmp_path / "fit"
    (adapter / "adapter").mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        (adapter / "adapter" / name).write_text("synthetic artifact, not real weights")
    state = {
        "status": "completed",
        "scope": "smoke",
        "model_config": model,
        "freeze_sha256": file_hash(manifests / "freeze.json"),
        "adapter_files": {p.name: file_hash(p) for p in (adapter / "adapter").iterdir()},
    }
    atomic_write(adapter / "training.json", canonical(state))
    assert verify_adapter(adapter, model, manifests, smoke=True)["status"] == "completed"
    with pytest.raises(LabError, match="Smoke"):
        verify_adapter(adapter, model, manifests, smoke=False)
    with pytest.raises(LabError, match="base mismatch"):
        verify_adapter(adapter, model | {"revision": "b" * 40}, manifests, smoke=True)
    (adapter / "adapter" / "adapter_model.safetensors").write_text("corrupted")
    with pytest.raises(LabError, match="hash mismatch"):
        verify_adapter(adapter, model, manifests, smoke=True)


def test_rejects_inexact_chat_boundary():
    class BrokenTokenizer(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1, 2] if add_generation_prompt else [7, 8, 9]

    with pytest.raises(LabError, match="boundary"):
        encode_records(rows(), BrokenTokenizer(), {"system": "S", "abstention": "No"}, 30)


def test_adapted_strategy_contracts():
    from types import SimpleNamespace

    from model_adaptation_lab.models.loader import Generation
    from model_adaptation_lab.retrieval.retriever import RetrievalResult
    from model_adaptation_lab.strategies.base import AnsweringStrategy
    from model_adaptation_lab.strategies.fine_tuned import FineTunedStrategy
    from model_adaptation_lab.strategies.fine_tuned_rag import FineTunedRAGStrategy

    backend = SimpleNamespace(
        model_config={"revision": "a" * 40, "tokenizer_revision": "a" * 40},
        generation_parameters={},
        generate=lambda messages: Generation("Synthetic [doc-1]", 8, 3, 4.0, None, None, "eos"),
    )
    retriever = SimpleNamespace(metadata={}, retrieve=lambda q: RetrievalResult((), 1.0))
    prompts = yaml_config(Path("configs/prompts.yaml"))
    strategies = [
        FineTunedStrategy(backend, prompts),
        FineTunedRAGStrategy(
            backend, prompts, retriever, yaml_config(Path("configs/retrieval.yaml"))
        ),
    ]
    for strategy in strategies:
        assert isinstance(strategy, AnsweringStrategy)
        result = strategy.answer("Synthetic question")
        assert result.strategy == strategy.name
        assert result.input_tokens == 8
    assert strategies[1].answer("Synthetic question").citation_document_ids == ("doc-1",)


def test_failed_training_persists_state_and_marks_mlflow_failed(
    tmp_path, synthetic_protocol, monkeypatch
):
    import json
    from types import SimpleNamespace

    from model_adaptation_lab.models import lora

    calls = []

    class MLflow:
        def set_tracking_uri(self, uri):
            calls.append(("uri", uri))

        def set_experiment(self, name):
            pass

        def start_run(self, run_name):
            return SimpleNamespace(info=SimpleNamespace(run_id="synthetic-run"))

        def log_params(self, params):
            json.dumps(params)

        def log_metrics(self, metrics, **kwargs):
            json.dumps(metrics)

        def log_artifact(self, path):
            assert Path(path).exists()

        def log_artifacts(self, path, artifact_path):
            assert Path(path).is_dir()
            calls.append(("artifact_directory", artifact_path))

        def end_run(self, status):
            calls.append(("end", status))

    class SyntheticTokenizer(Tokenizer):
        chat_template = "synthetic template"

    class FailModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("synthetic out of memory")

    tf = SimpleNamespace(
        set_seed=lambda seed: None,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: SyntheticTokenizer()),
        AutoModelForCausalLM=FailModel,
    )
    torch = SimpleNamespace(float32="float32")
    modules = {
        "torch": torch,
        "transformers": tf,
        "peft": SimpleNamespace(),
        "datasets": SimpleNamespace(),
        "mlflow": MLflow(),
    }
    # Patch only this module's imported namespace; don't replace global importlib behavior.
    monkeypatch.setattr(
        lora, "importlib", SimpleNamespace(import_module=lambda name: modules[name])
    )
    monkeypatch.setattr(lora, "capture_environment", lambda: {"fixture": "synthetic"})

    def synthetic_snapshot(output):
        (output / "code_snapshot").mkdir()
        (output / "code_snapshot" / "fixture.txt").write_text("synthetic")
        return {}

    monkeypatch.setattr(lora, "snapshot_code", synthetic_snapshot)
    manifests, processed = synthetic_protocol
    output = tmp_path / "failed-training"
    with pytest.raises(LabError, match="memory"):
        lora.train_lora(
            config=yaml_config(Path("configs/training.yaml")),
            model_config=yaml_config(Path("configs/model.yaml")) | {"device": "cpu"},
            generation=yaml_config(Path("configs/generation.yaml")),
            prompts=yaml_config(Path("configs/prompts.yaml")),
            manifests=manifests,
            processed=processed,
            output=output,
            cache=tmp_path / "cache",
            local_files_only=True,
        )
    state = json.loads((output / "training.json").read_text())
    assert state["status"] == "failed"
    assert state["peak_allocated_bytes"] is None
    assert state["training_gpu_hours"] is None
    assert state["mlflow_run_id"] == "synthetic-run"
    assert calls[-1] == ("end", "FAILED")
    assert ("artifact_directory", "code_snapshot") in calls
    assert "out of memory" in state["error"]
