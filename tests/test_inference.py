"""Synthetic fixtures and CPU-only fakes; no torch import, network, or model download."""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_adaptation_lab.cli import main
from model_adaptation_lab.evaluation.metrics import diagnostics, percentile, summarize
from model_adaptation_lab.evaluation.runner import (
    evaluate_questions,
    run_evaluation,
    validation_sample,
)
from model_adaptation_lab.io import LabError, file_hash
from model_adaptation_lab.models.loader import (
    Generation,
    TransformersBackend,
    require_device,
    validate_configs,
    yaml_config,
)
from model_adaptation_lab.models.prompting import choose_demonstrations, render_messages
from model_adaptation_lab.strategies.base import AnsweringStrategy, BaseStrategy
from model_adaptation_lab.strategies.few_shot import FewShotStrategy


@pytest.fixture
def settings():
    model = yaml_config(Path("configs/model.yaml")) | {"device": "cpu", "dtype": "float32"}
    generation = yaml_config(Path("configs/generation.yaml"))
    prompts = yaml_config(Path("configs/prompts.yaml"))
    return model, generation, prompts


class FakeBackend:
    def __init__(self, model):
        self.model_config = model
        self.generation_parameters = {"do_sample": False, "max_new_tokens": 4}
        self.metadata = {"backend": "synthetic_fixture"}
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return Generation("Synthetic response", 12, 3, 10.0, None, None, "eos")


def test_strategy_conformance_ids_token_counts_and_no_retrieval(settings, synthetic_protocol):
    model, _, prompts = settings
    backend = FakeBackend(model)
    for strategy in (
        BaseStrategy(backend, prompts),
        FewShotStrategy(backend, prompts, *synthetic_protocol),
    ):
        assert isinstance(strategy, AnsweringStrategy)
        result = strategy.answer("Synthetic question")
        assert result.input_tokens == 12 and result.output_tokens == 3
        assert result.output_tokens_per_second == 300
        assert result.retrieved_document_ids == () and result.retrieval_latency_ms == 0
        assert result.peak_allocated_bytes is None
        assert result.model_revision == model["revision"]
        assert backend.calls[-1][-1] == {"role": "user", "content": "Synthetic question"}
    assert len(backend.calls[0]) == 2
    assert len(backend.calls[1]) == 8


def test_demonstrations_are_stable_training_only(settings, synthetic_protocol):
    _, _, prompts = settings
    demos = choose_demonstrations(*synthetic_protocol, prompts)
    assert demos == choose_demonstrations(*synthetic_protocol, prompts)
    assert len(demos) == 3 and all(r["split"] == "train" for r in demos)
    assert sum(r["answerable"] for r in demos) == 2
    bad = [dict(demos[0], split="validation")]
    with pytest.raises(LabError, match="training"):
        render_messages("Synthetic", prompts, bad)
    with pytest.raises(LabError, match="empty"):
        render_messages(" ", prompts, [])


def test_missing_demonstrations_never_fall_back(settings, synthetic_protocol):
    prompts = settings[2] | {"demonstrations": dict(settings[2]["demonstrations"], answerable=999)}
    with pytest.raises(LabError, match="Insufficient"):
        choose_demonstrations(*synthetic_protocol, prompts)


def test_diagnostics_not_correctness_or_groundedness():
    metrics = diagnostics("alpha beta", "alpha gamma", True, "ABSTAIN")
    assert metrics["token_f1_diagnostic"] == 0.5
    assert metrics["normalized_exact_match"] == 0
    assert metrics["reference_evidence_support"] is None
    assert metrics["correctness_rubric_score"] is None
    assert diagnostics("ABSTAIN", None, False, "ABSTAIN")[
        "appropriate_abstention_annotation_scoped"
    ]
    assert diagnostics("ABSTAIN", "answer", True, "ABSTAIN")["unnecessary_abstention"]
    assert percentile([], 0.5) is None and percentile([10, 20], 0.95) == 19.5
    assert summarize([])["latency_p50_ms"] is None


def test_failure_persisted_without_fake_metrics(tmp_path, settings):
    class Failed(BaseStrategy):
        def answer(self, question):
            raise LabError("synthetic OOM")

    rows = [{"question_id": "SYNTH", "question": "failure", "answer": None, "answerable": False}]
    path = tmp_path / "attempt.jsonl"
    result = evaluate_questions(Failed(FakeBackend(settings[0]), settings[2]), rows, path, "NO")
    assert result[0]["status"] == "failed"
    assert result[0]["input_tokens"] is None
    assert summarize(result)["failed"] == 1
    assert json.loads(path.read_text())["error"] == "LabError: synthetic OOM"
    with pytest.raises(FileExistsError):
        evaluate_questions(Failed(FakeBackend(settings[0]), settings[2]), rows, path, "NO")


def test_common_runner_same_rows_traceability_and_immutable_output(
    tmp_path,
    settings,
    synthetic_protocol,
):
    model, generation, prompts = settings
    manifests, processed = synthetic_protocol
    args = dict(
        strategy_names=["base", "prompt"],
        model=model,
        generation=generation,
        prompts=prompts,
        manifests=manifests,
        processed=processed,
        output=tmp_path / "run",
        cache=tmp_path / "cache",
        smoke=True,
        limit=None,
        local_files_only=True,
        backend=FakeBackend(model),
    )
    run = run_evaluation(**args)
    assert run["status"] == "completed" and run["backend_type"] == "injected_test"
    assert run["results"]["base"]["attempted"] == 2
    assert run["results"]["prompt"]["attempted"] == 2
    for name in ("base", "prompt"):
        rows = [
            json.loads(line) for line in (args["output"] / f"{name}.jsonl").read_text().splitlines()
        ]
        assert [r["question_id"] for r in rows] == run["validation_ids"]
        assert file_hash(args["output"] / f"{name}.jsonl") == run["results"][name]["file_sha256"]
    assert json.loads((args["output"] / "run.json").read_text())["status"] == "completed"
    with pytest.raises(LabError, match="directory exists"):
        run_evaluation(**args)


def test_validation_sampler_balanced_and_refuses_other_splits():
    rows = [
        {"question_id": f"SYNTH{i}", "split": "validation", "answerable": i % 2 == 0}
        for i in range(6)
    ]
    picked = validation_sample(rows, 2, 42)
    assert picked == validation_sample(list(reversed(rows)), 2, 42)
    assert {r["answerable"] for r in picked} == {True, False}
    with pytest.raises(LabError, match="validation only"):
        validation_sample([dict(rows[0], split="test")], 2, 42)


def test_unavailable_devices_are_explicit():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    for device in ("cuda", "mps"):
        with pytest.raises(LabError, match="unavailable"):
            require_device(torch, device)
    require_device(torch, "cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "main"),
        ("tokenizer_revision", "main"),
        ("trust_remote_code", True),
        ("device", "auto"),
    ],
)
def test_model_config_requires_pins_and_explicit_settings(settings, field, value):
    with pytest.raises(LabError):
        validate_configs(settings[0] | {field: value}, settings[1])


def test_sampling_disallowed_and_cli_has_no_test_switch(settings, tmp_path):
    with pytest.raises(LabError, match="greedy"):
        validate_configs(settings[0], settings[1] | {"do_sample": True})
    assert (
        main(
            ["evaluate", "--strategy", "base", "--limit", "0", "--output-dir", str(tmp_path / "x")]
        )
        == 2
    )
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--strategy", "base", "--split", "test", "--output-dir", "unused"])
    assert exc.value.code == 2


class FakeTokens(dict):
    def __init__(self, count=4):
        super().__init__(input_ids=SimpleNamespace(shape=(1, count)), attention_mask="mask")

    def to(self, device):
        return self


class FakeOutput:
    def __getitem__(self, key):
        assert key[1].start == 4  # Strip prompt before decoding/counting output.
        return SimpleNamespace(tolist=lambda: [99, 2])


def transformer_fake(device="cpu", count=4):
    calls = []
    backend = TransformersBackend.__new__(TransformersBackend)
    backend.device = device
    backend.settings = {"max_input_tokens": 8, "max_new_tokens": 4}
    backend.generation_config = object()
    backend.logits_processors = []
    backend.eos_ids = {2}

    def template(messages, **kwargs):
        calls.append(kwargs)
        return FakeTokens(count)

    backend.tokenizer = SimpleNamespace(
        apply_chat_template=template,
        decode=lambda ids, **kwargs: "Answer" if ids == [99, 2] else "unexpected",
    )
    backend.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=16), generate=lambda **kwargs: FakeOutput()
    )
    backend.torch = SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 123,
            max_memory_reserved=lambda: 456,
        ),
    )
    return backend, calls


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_chat_template_token_accounting_eos_and_memory(device):
    backend, calls = transformer_fake(device)
    result = backend.generate([{"role": "user", "content": "Synthetic"}])
    assert result.text == "Answer" and result.input_tokens == 4 and result.output_tokens == 2
    assert result.finish_reason == "eos"
    assert result.peak_allocated_bytes == (123 if device == "cuda" else None)
    assert calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
    ]


def test_context_overflow_fails_without_truncation():
    backend, _ = transformer_fake(count=9)
    with pytest.raises(LabError, match="silently truncated"):
        backend.generate([{"role": "user", "content": "Synthetic"}])


def test_nonfinite_logits_fail_instead_of_decoding_garbage():
    from model_adaptation_lab.models.loader import FiniteScoreGuard

    class Flag:
        def __init__(self, value):
            self.value = value

        def any(self, **kwargs):
            return self

        def all(self):
            return self

        def item(self):
            return self.value

    torch = SimpleNamespace(
        isnan=lambda s: Flag(True), isposinf=lambda s: Flag(False), isfinite=lambda s: Flag(False)
    )
    with pytest.raises(LabError, match="Non-finite"):
        FiniteScoreGuard(torch)(None, "synthetic logits")
    torch.isnan = lambda s: Flag(False)
    torch.isfinite = lambda s: Flag(True)
    assert FiniteScoreGuard(torch)(None, "finite logits") == "finite logits"


def test_missing_inference_dependency_reports_actionable_error(settings, monkeypatch, tmp_path):
    def missing(name):
        raise ImportError("synthetic missing dependency")

    monkeypatch.setattr("model_adaptation_lab.models.loader.importlib.import_module", missing)
    with pytest.raises(LabError, match="uv sync --extra inference"):
        TransformersBackend(settings[0], settings[1], tmp_path)


def test_startup_failure_is_saved(tmp_path, settings, synthetic_protocol, monkeypatch):
    def fail(*args, **kwargs):
        raise LabError("synthetic unavailable device")

    monkeypatch.setattr("model_adaptation_lab.evaluation.runner.TransformersBackend", fail)
    output = tmp_path / "failed-run"
    with pytest.raises(LabError, match="unavailable device"):
        run_evaluation(
            strategy_names=["base"],
            model=settings[0],
            generation=settings[1],
            prompts=settings[2],
            manifests=synthetic_protocol[0],
            processed=synthetic_protocol[1],
            output=output,
            cache=tmp_path / "cache",
            smoke=True,
            limit=None,
            local_files_only=True,
        )
    state = json.loads((output / "run.json").read_text())
    assert state["status"] == "failed" and state["results"] == {}
    assert state["error"] == "LabError: synthetic unavailable device"
    assert state["finished_unix"] >= state["started_unix"]
    assert state["code_snapshot_sha256"]
