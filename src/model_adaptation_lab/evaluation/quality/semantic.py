"""Optional pinned semantic similarity: a diagnostic, never an entailment verdict."""

import math
from pathlib import Path
from typing import Any

from model_adaptation_lab.evaluation.quality.packet import verify_packet
from model_adaptation_lab.io import LabError, atomic_write, canonical, file_hash, read_json
from model_adaptation_lab.models.loader import yaml_config
from model_adaptation_lab.retrieval.embedding import MiniLMEmbedder


def similarity(answer: str, reference: str, encoder: Any, max_tokens: int) -> dict[str, Any]:
    lengths = [
        len(encoder.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in (answer, reference)
    ]
    if max(lengths) > max_tokens:
        return {"cosine": None, "reason": "input_overflow_no_truncation", "token_lengths": lengths}
    vectors = encoder.encode([answer, reference])
    value = sum(float(x) * float(y) for x, y in zip(vectors[0], vectors[1], strict=True))
    if not math.isfinite(value):
        raise LabError("Non-finite semantic similarity")
    return {"cosine": max(-1.0, min(1.0, value)), "reason": None, "token_lengths": lengths}


def semantic_diagnostics(
    *,
    packet: Path,
    output: Path,
    cache: Path,
    device: str,
    retrieval_config: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    _, cases = verify_packet(packet)
    config = read_json(packet / "config.json")
    embedding = yaml_config(retrieval_config)
    if (
        embedding["embedding"]["model_id"] != config["semantic_model"]
        or embedding["embedding"]["revision"] != config["semantic_revision"]
        or embedding["embedding"]["tokenizer_revision"] != config["semantic_revision"]
    ):
        raise LabError("Semantic encoder differs from the versioned evaluator")
    if output.exists():
        raise LabError("Semantic output exists; preserve it")
    embedding["embedding"]["device"] = device
    encoder = MiniLMEmbedder(embedding, cache, local_files_only)
    rows = []
    result: dict[str, Any]
    for case in cases:
        if case["status"] != "ok" or not case["answer"].strip():
            result = {"cosine": None, "reason": "generation_failed_or_empty"}
        elif not case["answerable"] or not case["reference_answer"]:
            result = {"cosine": None, "reason": "reference_unavailable"}
        else:
            result = similarity(
                case["answer"], case["reference_answer"], encoder, config["semantic_max_tokens"]
            )
        rows.append({"item_id": case["item_id"], **result})
    result = {
        "packet_sha256": file_hash(packet / "manifest.json"),
        "encoder": encoder.metadata,
        "interpretation": (
            "Uncalibrated reference similarity; not correctness or evidence entailment"
        ),
        "rows": rows,
    }
    atomic_write(output, canonical(result))
    return result
