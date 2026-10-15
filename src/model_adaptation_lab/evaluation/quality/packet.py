"""Integrity-checked, partially blinded development review packets."""

import json
import re
from pathlib import Path
from typing import Any

from model_adaptation_lab.data.splits import load_annotations
from model_adaptation_lab.evaluation.metrics import diagnostics
from model_adaptation_lab.evaluation.runner import capture_environment, snapshot_code
from model_adaptation_lab.io import (
    LabError,
    atomic_write,
    canonical,
    digest,
    file_hash,
    jsonl,
    read_json,
)
from model_adaptation_lab.models.loader import yaml_config
from model_adaptation_lab.retrieval.chunking import load_corpus

STRATEGIES = {"base", "prompt", "rag", "lora", "lora-rag"}
RAG = {"rag", "lora-rag"}


def validate_quality_config(config: dict[str, Any]) -> None:
    try:
        valid = (
            config["schema_version"] == 1
            and config["scope"] == "validation_only"
            and config["rubric_version"] == "answer-quality-v1"
            and set(config["required_strategies"]) == STRATEGIES
            and len(config["required_strategies"]) == 5
            and config["minimum_reviewed_questions_per_answerability"] == 6
            and config["minimum_independent_human_reviewers"] == 2
            and config["correctness_scale"] == [0, 1, 2, 3]
            and config["semantic_model"] == "sentence-transformers/all-MiniLM-L6-v2"
            and config["semantic_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
            and config["semantic_max_tokens"] == 256
            and config["semantic_overflow"] == "unavailable"
            and config["semantic_interpretation"] == "diagnostic_not_correctness"
            and type(config["seed"]) is int
            and config["seed"] >= 0
        )
    except (KeyError, TypeError) as exc:
        raise LabError(f"Invalid quality-v1 configuration: {exc}") from exc
    if not valid:
        raise LabError("Quality configuration differs from the supported v1 rubric")


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Expected JSON objects")
        return rows
    except (ValueError, OSError) as exc:
        raise LabError(f"Invalid JSONL {path}: {exc}") from exc


def spans(answer: str) -> list[dict[str, Any]]:
    # Suggestions only: reviewers must split compound claims and preserve full text coverage.
    return [
        {
            "start": m.start(),
            "end": m.end(),
            "text": m.group(),
            "kind": None,
            "reference_support": None,
            "retrieved_support": None,
            "reference_evidence": [],
            "retrieved_evidence": [],
            "cited_document_ids": [],
            "citation_support": None,
            "rationale": "",
        }
        for m in re.finditer(r"[^\n]+(?:\n|$)", answer)
    ]


def citation_ids(answer: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([\w.:-]+)\]", answer)))


def verify_packet(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(path / "manifest.json")
    if manifest.get("scope") != "validation_only" or manifest.get("schema_version") != 1:
        raise LabError("Only validation quality packets are supported")
    if set(manifest.get("files", {})) != {
        "cases.jsonl",
        "mapping.json",
        "rubric.md",
        "config.json",
        "diagnostics.jsonl",
    }:
        raise LabError("Invalid quality packet artifact set")
    for name, expected in manifest["files"].items():
        if file_hash(path / name) != expected:
            raise LabError(f"Quality packet changed: {name}")
    validate_quality_config(read_json(path / "config.json"))
    cases = read_rows(path / "cases.jsonl")
    if len({c["item_id"] for c in cases}) != len(cases):
        raise LabError("Duplicate review item IDs")
    return dict(manifest), cases


def prepare_packet(
    *,
    runs: list[Path],
    manifests: Path,
    processed: Path,
    output: Path,
    config_path: Path,
    rubric_path: Path,
) -> dict[str, Any]:
    config = yaml_config(config_path)
    validate_quality_config(config)
    annotations = {
        r["question_id"]: r
        for r in load_annotations(manifests, processed, "validation", purpose="selection")
    }
    documents, corpus = load_corpus(manifests, processed)
    docs = {d["document_id"]: d for d in documents}
    results: dict[str, list[dict[str, Any]]] = {}
    sources, identity = [], None
    for path in runs:
        run = read_json(path / "run.json")
        if (
            run.get("scope") != "validation_only"
            or run.get("status") not in {"completed", "partial_failure"}
            or run.get("backend_type") != "real_transformers"
            or run.get("freeze_sha256") != file_hash(manifests / "freeze.json")
        ):
            raise LabError("Source must be an integrity-matched real validation run")
        current = {
            k: run[k] for k in ("model_config", "generation_config", "validation_ids", "smoke")
        }
        if identity is not None and current != identity:
            raise LabError("Source runs differ in questions, model or generation conditions")
        identity = current
        if not set(run["validation_ids"]) <= set(annotations):
            raise LabError("Source contains non-validation IDs")
        if len(set(run["validation_ids"])) != len(run["validation_ids"]):
            raise LabError("Repeated validation IDs")
        rows = [annotations[q] for q in run["validation_ids"]]
        if digest(canonical(rows)) != run["validation_records_sha256"]:
            raise LabError("Source validation record hash mismatch")
        for strategy, summary in run["results"].items():
            if strategy not in STRATEGIES or strategy in results:
                raise LabError("Unknown or repeated source strategy")
            source = path / f"{strategy}.jsonl"
            if file_hash(source) != summary["file_sha256"]:
                raise LabError("Source output hash mismatch")
            records = read_rows(source)
            if (
                len(records) != len(rows)
                or {r["question_id"] for r in records} != set(run["validation_ids"])
                or any(r.get("strategy") != strategy for r in records)
            ):
                raise LabError("Source result coverage/strategy mismatch")
            for record in records:
                if record.get("status") not in {"ok", "failed"}:
                    raise LabError("Invalid generation status")
                if record["status"] == "ok" and not isinstance(record.get("answer"), str):
                    raise LabError("Successful output must have an answer string")
                if strategy not in RAG and record.get("retrieved_chunks"):
                    raise LabError("Non-RAG output unexpectedly includes retrieval")
                for chunk in record.get("retrieved_chunks", []):
                    doc = docs.get(chunk["document_id"])
                    if (
                        doc is None
                        or type(chunk.get("start")) is not int
                        or type(chunk.get("end")) is not int
                        or not 0 <= chunk["start"] < chunk["end"] <= len(doc["text"])
                        or doc["text"][chunk["start"] : chunk["end"]] != chunk["text"]
                    ):
                        raise LabError("Retrieved passage differs from frozen source evidence")
            results[strategy] = records
        sources.append(
            {
                "path": str(path),
                "run_sha256": file_hash(path / "run.json"),
                "results": {s: v["file_sha256"] for s, v in run["results"].items()},
            }
        )
    if set(results) != STRATEGIES:
        raise LabError("Provide each of the five strategies exactly once")
    common: dict[str, set[str]] = {}
    for records in results.values():
        for r in records:
            q = r["question_id"]
            common.setdefault(q, set()).update(
                c["document_id"] for c in r.get("retrieved_chunks", [])
            )
            gold = annotations[q].get("document_id")
            if gold:
                common[q].add(gold)
    cases, mapping, automatic = [], {}, []
    for strategy, records in results.items():
        for record in records:
            row = annotations[record["question_id"]]
            answer = record.get("answer") or ""
            item = digest(canonical({"seed": config["seed"], "record": record}))[:24]
            case = {
                "item_id": item,
                "question_id": row["question_id"],
                "question": row["question"],
                "answer": answer,
                "reference_answer": row["answer"],
                "answerable": row["answerable"],
                "status": record["status"],
                "error": record.get("error"),
                "finish_reason": record.get("finish_reason"),
                "retrieval_applicable": strategy in RAG,
                "reference_documents": [docs[k] for k in sorted(common[row["question_id"]])],
                "retrieved_passages": record.get("retrieved_chunks", []),
                "reference_scope": "reviewed frozen documents; candidate pool is not exhaustive",
            }
            cases.append(case)
            mapping[item] = {"strategy": strategy, "question_id": row["question_id"]}
            ids = citation_ids(answer)
            retrieved_ids = {c["document_id"] for c in case["retrieved_passages"]}
            automatic.append(
                {
                    "item_id": item,
                    "method": "deterministic_diagnostics_not_quality_scores",
                    "lexical": diagnostics(
                        answer,
                        row["answer"],
                        row["answerable"],
                        "I don't have enough information to answer this question.",
                    )
                    if record["status"] == "ok"
                    else None,
                    "truncated": record.get("finish_reason") == "length",
                    "candidate_citation_ids": ids,
                    "citation_id_membership": sum(i in retrieved_ids for i in ids) / len(ids)
                    if ids and strategy in RAG
                    else None,
                    "citation_support": None,
                    "correctness": None,
                }
            )
    cases.sort(key=lambda c: c["item_id"])
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise LabError("Packet output exists; use a new version/directory") from exc
    atomic_write(output / "cases.jsonl", jsonl(cases))
    atomic_write(output / "mapping.json", canonical(mapping))
    atomic_write(output / "config.json", canonical(config))
    atomic_write(output / "rubric.md", rubric_path.read_bytes())
    atomic_write(output / "diagnostics.jsonl", jsonl(sorted(automatic, key=lambda r: r["item_id"])))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scope": "validation_only",
        "status": "awaiting_independent_review",
        "freeze_sha256": file_hash(manifests / "freeze.json"),
        "corpus": corpus,
        "source_runs": sources,
        "environment": capture_environment(),
        "code_snapshot_sha256": snapshot_code(output),
        "files": {
            n: file_hash(output / n)
            for n in (
                "cases.jsonl",
                "mapping.json",
                "rubric.md",
                "config.json",
                "diagnostics.jsonl",
            )
        },
    }
    atomic_write(output / "manifest.json", canonical(manifest))
    template = review_template(output, manifest, cases)
    atomic_write(output / "review_template.json", canonical(template))
    return manifest


def review_template(
    packet: Path, manifest: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a fresh unrated form from verified cases, never from editable ratings."""
    return {
        "schema_version": 1,
        "packet_sha256": file_hash(packet / "manifest.json"),
        "rubric_sha256": manifest["files"]["rubric.md"],
        "reviewer_id": "",
        "kind": "human",
        "independent": True,
        "model_revision": None,
        "judge_prompt_sha256": None,
        "items": [
            {
                "item_id": c["item_id"],
                "correctness": None,
                "correctness_unavailable_reason": "",
                "behavior": None,
                "reference_sufficiency": None,
                "retrieved_sufficiency": None,
                "reviewed_document_ids": [],
                "rationale": "",
                "claims": spans(c["answer"]),
            }
            for c in cases
        ],
    }
