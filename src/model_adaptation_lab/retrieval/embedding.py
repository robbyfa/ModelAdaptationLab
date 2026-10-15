"""Pinned MiniLM mean pooling following its model card; no silent truncation."""

import importlib
import importlib.metadata
import re
from pathlib import Path
from typing import Any, Protocol

from model_adaptation_lab.io import LabError
from model_adaptation_lab.models.loader import require_device


class Embedder(Protocol):
    tokenizer: Any
    metadata: dict[str, Any]

    def encode(self, texts: list[str]) -> Any: ...


def validate_retrieval(config: dict[str, Any]) -> None:
    try:
        e, c, i, r = (config[k] for k in ("embedding", "chunking", "index", "retrieval"))
        valid = (
            config["schema_version"] == 1
            and all(
                isinstance(e[k], str) and re.fullmatch(r"[0-9a-f]{40}", e[k])
                for k in ("revision", "tokenizer_revision")
            )
            and e["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"
            and e["pooling"] == "attention_mask_mean_l2_normalized"
            and e["dimensions"] == 384
            and e["max_tokens"] == 256
            and e["dtype"] == "float32"
            and e["device"] in {"cpu", "mps", "cuda"}
            and type(e["batch_size"]) is int
            and e["batch_size"] > 0
            and c["version"] == "body-wordpiece-v1"
            and type(c["tokens"]) is int
            and type(c["overlap"]) is int
            and 0 <= c["overlap"] < c["tokens"] <= 224
            and i == {"type": "IndexFlatIP", "threads": 1}
            and type(r["top_k"]) is int
            and r["top_k"] > 0
            and r["query_overflow"] == "error"
            and r["unit"] == "chunks"
            and isinstance(config["rag_prompt"]["instruction"], str)
            and bool(config["rag_prompt"]["instruction"].strip())
            and isinstance(config["rag_prompt"]["version"], str)
        )
    except (KeyError, TypeError) as exc:
        raise LabError(f"Invalid retrieval configuration: {exc}") from exc
    if not valid:
        raise LabError("Unsupported retrieval configuration, revision, or chunk bounds")


class MiniLMEmbedder:
    def __init__(self, config: dict[str, Any], cache: Path, local_files_only: bool) -> None:
        validate_retrieval(config)
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise LabError(
                "Install retrieval dependencies: uv sync --locked --extra retrieval"
            ) from exc
        self.config = config["embedding"]
        self.device = self.config["device"]
        require_device(self.torch, self.device)
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config["model_id"],
                revision=self.config["tokenizer_revision"],
                cache_dir=str(cache),
                trust_remote_code=False,
                use_fast=True,
                local_files_only=local_files_only,
            )
            self.model = (
                transformers.AutoModel.from_pretrained(
                    self.config["model_id"],
                    revision=self.config["revision"],
                    cache_dir=str(cache),
                    trust_remote_code=False,
                    use_safetensors=True,
                    dtype=self.torch.float32,
                    attn_implementation="eager",
                    local_files_only=local_files_only,
                )
                .to(self.device)
                .eval()
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise LabError(f"Pinned embedding model loading failed: {exc}") from exc
        self.metadata = dict(self.config) | {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "embedded_content": "body_only_no_title",
            "normalization": "L2",
        }

    def encode(self, texts: list[str]) -> Any:
        if not texts:
            raise LabError("Cannot embed an empty batch")
        batch = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if batch["input_ids"].shape[-1] > self.config["max_tokens"]:
            raise LabError(
                "Embedding input exceeds 256 wordpieces; no silent query/chunk truncation"
            )
        batch = batch.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        if not self.torch.isfinite(vectors).all().item():
            raise LabError("Non-finite embedding vectors")
        return vectors.cpu().numpy().astype("float32")
