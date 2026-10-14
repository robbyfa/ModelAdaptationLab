"""Real PEFT training with guarded data, immutable outputs and local MLflow tracking."""

import gc
import importlib
import math
import time
from pathlib import Path
from typing import Any

from model_adaptation_lab.data.splits import load_annotations
from model_adaptation_lab.evaluation.runner import capture_environment, snapshot_code
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, file_hash
from model_adaptation_lab.models.loader import require_device, validate_configs, validate_prompts
from model_adaptation_lab.models.training_data import AnswerCollator, encode_records, nested_subset


def validate_training(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("format") != "question_only_v1":
        raise LabError("Expected question_only_v1 training schema")
    try:
        t, lora = config["training"], config["lora"]
        for key in (
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "max_seq_length",
            "early_stopping_patience",
        ):
            if type(t[key]) is not int or t[key] <= 0:
                raise LabError(f"training.{key} must be a positive integer")
        if not 0 < t["learning_rate"] < 1 or not 0 <= lora["dropout"] < 1:
            raise LabError("Invalid learning rate or LoRA dropout")
        if any(type(lora[k]) is not int or lora[k] <= 0 for k in ("rank", "alpha")):
            raise LabError("LoRA rank/alpha must be positive integers")
        if lora["target_modules"] != "auto" and not (
            isinstance(lora["target_modules"], list)
            and lora["target_modules"]
            and all(isinstance(v, str) and v for v in lora["target_modules"])
        ):
            raise LabError("target_modules must be auto or a nonempty module-name list")
        if t["quantization"] != "none":
            raise LabError("Quantized training is not yet supported; use quantization: none")
        if t["overflow_policy"] != "exclude_and_report":
            raise LabError("Expected explicit exclude_and_report overflow policy")
        if type(t["seed"]) is not int or t["seed"] < 0:
            raise LabError("Training seed must be nonnegative")
        if type(t["gradient_checkpointing"]) is not bool:
            raise LabError("gradient_checkpointing must be boolean")
        if not config["tracking"]["uri"].startswith("sqlite:///"):
            raise LabError("Phase 4 uses explicit local SQLite MLflow tracking")
    except (KeyError, TypeError) as exc:
        raise LabError(f"Invalid training configuration: {exc}") from exc


def train_lora(
    *,
    config: dict[str, Any],
    model_config: dict[str, Any],
    generation: dict[str, Any],
    prompts: dict[str, Any],
    manifests: Path,
    processed: Path,
    output: Path,
    cache: Path,
    local_files_only: bool,
    smoke: bool = False,
    fraction: float = 1.0,
) -> dict[str, Any]:
    validate_training(config)
    validate_configs(model_config, generation)
    validate_prompts(prompts)
    if model_config["dtype"] != "float32":
        raise LabError("Initial controlled training requires float32 weights")
    try:
        torch, tf, peft, datasets, mlflow = [
            importlib.import_module(n)
            for n in ("torch", "transformers", "peft", "datasets", "mlflow")
        ]
    except ImportError as exc:
        raise LabError("Install training dependencies: uv sync --locked --extra training") from exc
    require_device(torch, model_config["device"])
    rows = load_annotations(manifests, processed, "train", purpose="training")
    validation = load_annotations(manifests, processed, "validation", purpose="selection")
    t, lora = config["training"], config["lora"]
    rows = nested_subset(rows, fraction, t["seed"])
    if smoke:
        rows, validation = rows[:4], sorted(validation, key=lambda r: r["question_id"])[:4]
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise LabError(
            "Training output already exists; preserve it and choose a new directory"
        ) from exc
    state: dict[str, Any] = {
        "status": "initializing",
        "scope": "smoke" if smoke else "training",
        "config": config,
        "model_config": model_config,
        "prompts": prompts,
        "fraction": fraction,
        "freeze_sha256": file_hash(manifests / "freeze.json"),
        "training_records_sha256": digest(canonical(rows)),
        "validation_records_sha256": digest(canonical(validation)),
        "environment": capture_environment(),
        "code_snapshot_sha256": snapshot_code(output),
        "memory_scope": "CUDA process peaks; unavailable on MPS/CPU",
    }
    state_path = output / "training.json"
    atomic_write(state_path, canonical(state))
    started = time.perf_counter()
    trainer = model = None
    active_run = False
    try:
        tf.set_seed(t["seed"])
        tokenizer = tf.AutoTokenizer.from_pretrained(
            model_config["model_id"],
            revision=model_config["tokenizer_revision"],
            cache_dir=str(cache),
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        train_data, state["train_encoding"] = encode_records(
            rows, tokenizer, prompts, t["max_seq_length"]
        )
        eval_data, state["validation_encoding"] = encode_records(
            validation, tokenizer, prompts, t["max_seq_length"]
        )
        state["chat_template_sha256"] = digest(str(tokenizer.chat_template).encode())
        targets = (
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if lora["target_modules"] == "auto"
            else lora["target_modules"]
        )
        state["resolved_target_modules"] = targets
        mlflow.set_tracking_uri(config["tracking"]["uri"])
        mlflow.set_experiment(config["tracking"]["experiment"])
        run = mlflow.start_run(run_name=output.name)
        active_run = True
        state["mlflow_run_id"] = run.info.run_id
        mlflow.log_params(
            {
                "model_revision": model_config["revision"],
                "rank": lora["rank"],
                "alpha": lora["alpha"],
                "seed": t["seed"],
                "fraction": fraction,
                "scope": state["scope"],
                "format": config["format"],
                "freeze_sha256": state["freeze_sha256"],
            }
        )
        atomic_write(state_path, canonical(state))
        if model_config["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = tf.AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            revision=model_config["revision"],
            cache_dir=str(cache),
            local_files_only=local_files_only,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model.config.use_cache = False
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=lora["rank"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                target_modules=targets,
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        state["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        state["total_parameters"] = sum(p.numel() for p in model.parameters())
        if any(p.requires_grad for n, p in model.named_parameters() if "lora_" not in n):
            raise LabError("Unexpected trainable base-model parameters")
        args = tf.TrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=t["epochs"],
            max_steps=2 if smoke else -1,
            learning_rate=t["learning_rate"],
            per_device_train_batch_size=1 if smoke else t["batch_size"],
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1 if smoke else t["gradient_accumulation_steps"],
            gradient_checkpointing=t["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps" if smoke else "epoch",
            save_strategy="steps" if smoke else "epoch",
            eval_steps=1 if smoke else None,
            save_steps=1 if smoke else 500,
            logging_steps=1,
            logging_nan_inf_filter=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            prediction_loss_only=True,
            report_to=[],
            optim="adamw_torch",
            seed=t["seed"],
            data_seed=t["seed"],
            use_cpu=model_config["device"] == "cpu",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            label_names=["labels"],
            max_grad_norm=1.0,
        )

        class TrackingCallback(tf.TrainerCallback):  # type: ignore[misc, name-defined]
            def on_log(
                self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
            ) -> None:
                metrics = {
                    k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))
                }
                if any(not math.isfinite(v) for v in metrics.values()):
                    raise LabError("Non-finite training metric; failed run retained")
                mlflow.log_metrics(metrics, step=state.global_step)
                atomic_write(output / "loss_history.json", canonical(state.log_history))

        trainer = tf.Trainer(
            model=model,
            args=args,
            train_dataset=datasets.Dataset.from_list(train_data),
            eval_dataset=datasets.Dataset.from_list(eval_data),
            data_collator=AnswerCollator(tokenizer, torch),
            processing_class=tokenizer,
            callbacks=[
                TrackingCallback(),
                tf.EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"]),
            ],
        )
        if trainer.args.device.type != model_config["device"]:
            raise LabError("Trainer selected a different device than explicitly requested")
        state["effective_training_arguments"] = args.to_dict()
        state["status"] = "running"
        atomic_write(state_path, canonical(state))
        baseline = trainer.evaluate()
        state["initial_validation_loss"] = baseline["eval_loss"]
        trainer.train()
        state["best_checkpoint"] = trainer.state.best_model_checkpoint
        state["best_validation_loss"] = trainer.state.best_metric
        state["global_step"] = trainer.state.global_step
        trainer.save_model(str(output / "adapter"))
        state["adapter_files"] = {
            p.name: file_hash(p) for p in (output / "adapter").iterdir() if p.is_file()
        }
        atomic_write(output / "loss_history.json", canonical(trainer.state.log_history))
        plot_losses(trainer.state.log_history, output / "loss_curves.png")
        state["status"] = "completed"
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, RuntimeError):
            raise LabError(
                f"Training failed (check device memory); artifacts retained: {exc}"
            ) from exc
        raise
    finally:
        state["wall_seconds"] = time.perf_counter() - started
        cuda = model_config["device"] == "cuda"
        state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if cuda else None
        state["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved()) if cuda else None
        state["training_gpu_hours"] = state["wall_seconds"] / 3600 if cuda else None
        state["gpu_name"] = torch.cuda.get_device_name(0) if cuda else None
        atomic_write(state_path, canonical(state))
        if active_run:
            mlflow.log_metrics(
                {
                    k: float(state[k])
                    for k in (
                        "wall_seconds",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                        "training_gpu_hours",
                        "best_validation_loss",
                        "initial_validation_loss",
                    )
                    if state.get(k) is not None
                }
            )
            mlflow.log_artifact(str(state_path))
            mlflow.log_artifacts(str(output / "code_snapshot"), artifact_path="code_snapshot") if (
                output / "code"
            ).exists() else None
            for name in ("loss_history.json", "loss_curves.png"):
                if (output / name).exists():
                    mlflow.log_artifact(str(output / name))
            if state["status"] == "completed":
                mlflow.log_artifacts(str(output / "adapter"), artifact_path="adapter")
            mlflow.end_run(status="FINISHED" if state["status"] == "completed" else "FAILED")
        del trainer, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        elif model_config["device"] == "mps":
            torch.mps.empty_cache()


def plot_losses(history: list[dict[str, Any]], destination: Path) -> None:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    fig, ax = plt.subplots()
    for key, label in (
        ("loss", "Training (logged step mean)"),
        ("eval_loss", "Validation (Trainer mean)"),
    ):
        points = [(r["step"], r[key]) for r in history if key in r]
        if points:
            ax.plot(*zip(*points, strict=True), marker="o", label=label)
    ax.set(
        xlabel="Optimizer step",
        ylabel="Assistant-token cross-entropy",
        title="LoRA loss diagnostics",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
