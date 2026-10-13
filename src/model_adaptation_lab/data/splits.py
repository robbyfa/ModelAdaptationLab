"""Immutable official assignments; an empty pending test manifest is not a test set."""

import json
from pathlib import Path
from typing import Any

from model_adaptation_lab.data.ingest import Question
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, file_hash, jsonl


def freeze(rows: list[Question], audit: dict[str, Any], directory: Path) -> dict[str, Any]:
    if not rows or not all(audit["splits"][s]["rows"] for s in ("train", "validation")):
        raise LabError(
            "Cannot prepare manifests without both official training and development data"
        )
    if any(q.source != "original" or q.split not in {"train", "validation"} for q in rows):
        raise LabError("Only official original train/development records may be frozen in Phase 1")
    blockers = []
    if audit["cross_split_exact_question_groups"]:
        blockers.append("cross_split_exact_questions")
    if audit["evidence"].get("span_mismatch_question_ids"):
        blockers.append("reference_span_mismatches")
    if audit["evidence"].get("status") != "available":
        blockers.append("source_evidence_unavailable")
    if not audit.get("source_files_verified", False):
        blockers.append("source_lineage_unverified")
    artifacts: dict[str, bytes] = {}
    for split in ("train", "validation", "test"):
        artifacts[f"{split}.jsonl"] = jsonl(
            [
                {
                    "question_id": q.question_id,
                    "split": split,
                    "source": q.source,
                    "record_sha256": digest(canonical(q.record())),
                    "source_revision": audit["dataset_config"]["original"]["revision"],
                    "role": "model_training" if split == "train" else "development_selection",
                }
                for q in sorted(rows, key=lambda q: q.question_id)
                if q.split == split
            ]
        )
    lock = {
        "schema_version": 1,
        "policy": "official train/dev; no inferred final test",
        "experimental_use_ready": not blockers,
        "experimental_use_blockers": blockers,
        "final_test_ready": False,
        "test_status": "pending_protocol_not_an_evaluation_set",
        "dataset_config_sha256": audit["dataset_config_sha256"],
        "sources": audit["sources"],
        "files": {name: digest(content) for name, content in artifacts.items()},
    }
    artifacts["freeze.json"] = canonical(lock)
    # Preflight every artifact before any write: never silently overwrite a frozen set.
    for name, content in artifacts.items():
        path = directory / name
        if path.exists() and path.read_bytes() != content:
            raise LabError(f"Frozen artifact differs: {path}. Use a new versioned output directory")
    for name, content in artifacts.items():
        atomic_write(directory / name, content)
    return lock


def verify(directory: Path) -> dict[str, Any]:
    try:
        lock = json.loads((directory / "freeze.json").read_text())
        if set(lock["files"]) != {"train.jsonl", "validation.jsonl", "test.jsonl"}:
            raise LabError("Unexpected manifest file set")
        for name, expected in lock["files"].items():
            if file_hash(directory / name) != expected:
                raise LabError(f"Frozen manifest hash mismatch: {name}")
        return dict(lock)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LabError(f"Cannot verify frozen manifests: {exc}") from exc


def load_split(directory: Path, split: str, *, purpose: str) -> list[dict[str, Any]]:
    """Enforce role boundaries before future training or evaluation touches records."""
    allowed = {
        "train": {"training", "demonstrations"},
        "validation": {"selection"},
        "test": {"final_evaluation"},
    }
    if split not in allowed or purpose not in allowed[split]:
        raise LabError(f"Split {split!r} cannot be accessed for {purpose!r}")
    lock = verify(directory)
    if split == "test" and not lock["final_test_ready"]:
        raise LabError("Final test protocol is pending; an empty manifest is not an evaluation set")
    if not lock.get("experimental_use_ready", False):
        raise LabError(f"Experimental use blocked: {lock.get('experimental_use_blockers')}")
    return [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()]


def load_annotations(
    directory: Path, processed: Path, split: str, *, purpose: str
) -> list[dict[str, Any]]:
    """Load eligible records only after role, manifest, and per-record integrity checks."""
    assignments = load_split(directory, split, purpose=purpose)
    try:
        rows = [
            json.loads(line)
            for line in (processed / f"{split}_annotations.jsonl").read_text().splitlines()
        ]
        by_id = {r["question_id"]: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != {r["question_id"] for r in assignments}:
            raise LabError("Processed question IDs differ from frozen assignments")
        for item in assignments:
            if digest(canonical(by_id[item["question_id"]])) != item["record_sha256"]:
                raise LabError(f"Processed record hash mismatch: {item['question_id']}")
        return [by_id[item["question_id"]] for item in assignments]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LabError(f"Cannot load verified annotations: {exc}") from exc
