"""Pinned Transformers inference with explicit device checks and honest measurements."""

import gc
import importlib
import importlib.metadata
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from model_adaptation_lab.io import LabError, digest
from model_adaptation_lab.models.prompting import Message


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    generation_latency_ms: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    finish_reason: str


class GenerationBackend(Protocol):
    model_config: dict[str, Any]
    generation_parameters: dict[str, Any]
    metadata: dict[str, Any]

    def generate(self, messages: list[Message]) -> Generation: ...


def yaml_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise LabError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LabError(f"Expected configuration object: {path}")
    return value


def validate_configs(model: dict[str, Any], generation: dict[str, Any]) -> None:
    for key in ("revision", "tokenizer_revision"):
        if not isinstance(model.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", model[key]):
            raise LabError(f"Immutable 40-character {key} required")
    if model.get("trust_remote_code") is not False:
        raise LabError("trust_remote_code must be false")
    if not isinstance(model.get("model_id"), str) or not model["model_id"]:
        raise LabError("model_id is required")
    if model.get("device") not in {"cpu", "cuda", "mps"}:
        raise LabError("Explicit device cpu, cuda, or mps required")
    if model.get("dtype") not in {"float32", "float16", "bfloat16"}:
        raise LabError("Unsupported dtype")
    if model["device"] == "cpu" and model["dtype"] != "float32":
        raise LabError("Phase 2 CPU inference requires float32")
    required = {
        "schema_version",
        "max_new_tokens",
        "do_sample",
        "num_beams",
        "use_cache",
        "seed",
        "max_input_tokens",
    }
    if set(generation) != required or generation["schema_version"] != 1:
        raise LabError("Unexpected generation configuration schema")
    if generation["do_sample"] is not False or generation["num_beams"] != 1:
        raise LabError("Phase 2 requires shared greedy decoding (do_sample=false, num_beams=1)")
    for key in ("max_new_tokens", "max_input_tokens"):
        if type(generation[key]) is not int or generation[key] <= 0:
            raise LabError(f"{key} must be a positive integer")
    if type(generation["seed"]) is not int or generation["seed"] < 0:
        raise LabError("seed must be a nonnegative integer")
    if type(generation["use_cache"]) is not bool:
        raise LabError("use_cache must be boolean")


def validate_prompts(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise LabError("Prompt schema_version must be 1")
    for key in ("system", "abstention", "base_version", "few_shot_version", "selection_status"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise LabError(f"Prompt {key} must be nonempty")
    settings = config.get("demonstrations")
    if not isinstance(settings, dict):
        raise LabError("demonstrations settings required")
    for key in ("answerable", "unanswerable", "max_characters", "seed"):
        if type(settings.get(key)) is not int or settings[key] < 0:
            raise LabError(f"demonstrations.{key} must be a nonnegative integer")
    if settings["max_characters"] == 0 or settings["answerable"] + settings["unanswerable"] == 0:
        raise LabError("Few-shot prompting requires demonstrations and a positive character budget")


def require_device(torch: Any, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise LabError(
            "CUDA requested but unavailable. Use a CUDA host or explicitly select cpu/mps"
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise LabError(
            "MPS requested but unavailable. Select CPU explicitly or use supported hardware"
        )


class FiniteScoreGuard:
    """Fail explicitly on invalid logits instead of returning argmax-of-NaN token garbage."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if (
            self.torch.isnan(scores).any().item()
            or self.torch.isposinf(scores).any().item()
            or not self.torch.isfinite(scores).any(dim=-1).all().item()
        ):
            raise LabError(
                "Non-finite generation scores; this device/dtype is numerically "
                "invalid for this run. Try float32 and preserve the failed attempt"
            )
        return scores


class TransformersBackend:
    def __init__(
        self,
        model_config: dict[str, Any],
        generation: dict[str, Any],
        cache_dir: Path,
        *,
        local_files_only: bool = False,
    ) -> None:
        validate_configs(model_config, generation)
        self.model_config = dict(model_config)
        self.settings = dict(generation)
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise LabError(
                "Install inference dependencies: uv sync --extra inference --locked"
            ) from exc
        self.device = model_config["device"]
        require_device(self.torch, self.device)
        if (
            self.device == "cuda"
            and model_config["dtype"] == "bfloat16"
            and not self.torch.cuda.is_bf16_supported()
        ):
            raise LabError("This CUDA device does not support bfloat16")
        transformers.set_seed(generation["seed"])
        started = time.perf_counter()
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_config["model_id"],
                revision=model_config["tokenizer_revision"],
                trust_remote_code=False,
                cache_dir=str(cache_dir),
                local_files_only=local_files_only,
            )
            if not self.tokenizer.chat_template:
                raise LabError("Pinned tokenizer has no supported chat template")
            self.model = (
                transformers.AutoModelForCausalLM.from_pretrained(
                    model_config["model_id"],
                    revision=model_config["revision"],
                    trust_remote_code=False,
                    use_safetensors=True,
                    cache_dir=str(cache_dir),
                    local_files_only=local_files_only,
                    dtype=getattr(self.torch, model_config["dtype"]),
                    attn_implementation="eager",
                )
                .to(self.device)
                .eval()
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise LabError(f"Pinned model loading failed: {exc}") from exc
        self.sync()
        eos = self.model.generation_config.eos_token_id
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = eos[0] if isinstance(eos, list) else eos
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        # Fresh config avoids inheriting sampling defaults from the model repository.
        self.generation_config = transformers.GenerationConfig(
            max_new_tokens=generation["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=generation["use_cache"],
            eos_token_id=eos,
            pad_token_id=pad,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        self.logits_processors = [FiniteScoreGuard(self.torch)]
        self.generation_parameters = self.generation_config.to_dict()
        self.generation_parameters["seed"] = generation["seed"]
        self.metadata = {
            "backend": "transformers",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": self.device,
            "dtype": model_config["dtype"],
            "attention_implementation": "eager",
            "finite_score_guard": "every decoding step; synchronization overhead included",
            "model_load_ms": (time.perf_counter() - started) * 1000,
            "gpu_name": self.torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "cuda_version": self.torch.version.cuda,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "safetensors")
            },
            "chat_template_sha256": digest(str(self.tokenizer.chat_template).encode()),
            "memory_scope": "CUDA per-question peak including resident model; null on CPU/MPS",
            "timing_scope": "synchronized generate() includes prefill; model loading separate",
        }

    def sync(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def generate(self, messages: list[Message]) -> Generation:
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        input_count = int(tokens["input_ids"].shape[-1])
        max_context = int(self.model.config.max_position_embeddings)
        if (
            input_count > self.settings["max_input_tokens"]
            or input_count + self.settings["max_new_tokens"] > max_context
        ):
            raise LabError(
                f"Prompt has {input_count} tokens; context budget exceeded. "
                "No question or demonstration was silently truncated"
            )
        self.sync()
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **tokens,
                    generation_config=self.generation_config,
                    logits_processor=self.logits_processors,
                )
            self.sync()
        except (RuntimeError, LabError) as exc:
            del tokens
            gc.collect()
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device == "mps":
                self.torch.mps.empty_cache()
            raise LabError(
                f"Generation failed on {self.device}; check memory/context limits: {exc}"
            ) from exc
        elapsed = (time.perf_counter() - started) * 1000
        generated = output[0, input_count:].tolist()
        reason = "eos" if generated and generated[-1] in self.eos_ids else "length"
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return Generation(
            text=text,
            input_tokens=input_count,
            output_tokens=len(generated),
            generation_latency_ms=elapsed,
            peak_allocated_bytes=int(self.torch.cuda.max_memory_allocated())
            if self.device == "cuda"
            else None,
            peak_reserved_bytes=int(self.torch.cuda.max_memory_reserved())
            if self.device == "cuda"
            else None,
            finish_reason=reason,
        )
