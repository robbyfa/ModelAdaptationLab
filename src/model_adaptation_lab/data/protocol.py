"""Versioned, pre-model protocol. Keep the Phase 1 source snapshot immutable."""

import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from model_adaptation_lab.data.audit import normalize, overlaps
from model_adaptation_lab.data.ingest import Document, Question, original_documents, unique_ids
from model_adaptation_lab.data.splits import verify
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, file_hash, jsonl

POLICY = {
    "schema_version": 1,
    "protocol_id": "techqa-internal-v1",
    "seed": 42,
    "near_duplicate_threshold": 0.85,
    "test_group_fraction": 0.20,
    "validation_source": "official_development",
    "final_test_source": "eligible_official_training_groups",
    "reference_policy": "whitespace_only_correction_otherwise_quarantine",
    "overlap_policy": "connected_components_questions_and_identical_answers",
    "conflict_policy": "quarantine_conflicting_exact_question_groups",
    "deduplication": "lowest_question_id_per_exact_question",
    "final_test_access": "sealed_until_phase5",
}


def load_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if value != POLICY:
        raise LabError("Unsupported protocol changes; implement and version a new policy")
    return dict(POLICY)


def components(rows: list[Question], threshold: float) -> dict[str, str]:
    """Transitive components; include invalid/quarantined rows as contamination bridges."""
    unique_ids(rows)
    parent = {q.question_id: q.question_id for q in rows}

    def root(qid: str) -> str:
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    def join(a: str, b: str) -> None:
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    texts = {q.question_id: normalize(q.question) for q in rows}
    tokens = {qid: set(text.split()) for qid, text in texts.items()}
    answers = {q.question_id: normalize(q.answer) if q.answer else "" for q in rows}
    for a, b in combinations(sorted(parent), 2):
        same_answer = bool(answers[a]) and answers[a] == answers[b]
        ta, tb = tokens[a], tokens[b]
        similar = bool(ta and tb) and len(ta & tb) / len(ta | tb) >= threshold
        if texts[a] == texts[b] or same_answer or similar:
            join(a, b)
    return {qid: root(qid) for qid in sorted(parent)}


def correct_reference(q: Question, docs: dict[str, Document]) -> tuple[Question, str]:
    """Never generate reference text or infer an answer from offsets alone."""
    if not q.answerable:
        return q, "unchanged"
    if not q.answer or not q.answer.strip():
        return q, "quarantine_empty_reference"
    doc = docs.get(q.document_id or "")
    if doc is None:
        return q, "quarantine_missing_evidence"
    if q.start is None or q.end is None or not 0 <= q.start < q.end <= len(doc.text):
        return q, "quarantine_invalid_offsets"
    span = doc.text[q.start : q.end]
    if span == q.answer:
        return q, "unchanged"
    if " ".join(span.split()) == " ".join(q.answer.split()):
        return replace(q, answer=span), "corrected_whitespace_only"
    return q, "quarantine_reference_span_disagreement"


def build_protocol(
    rows: list[Question], docs: dict[str, Document], policy: dict[str, Any]
) -> tuple[list[Question], dict[str, Any], dict[str, Any]]:
    if policy != POLICY:
        raise LabError("Only the implemented v1 policy is supported")
    if not rows or any(
        q.source != "original" or q.split not in {"train", "validation"} for q in rows
    ):
        raise LabError("Protocol requires original train/development rows only")
    rows = sorted(rows, key=lambda q: q.question_id)
    groups = components(rows, policy["near_duplicate_threshold"])
    corrected: dict[str, Question] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for q in rows:
        fixed, action = correct_reference(q, docs)
        corrected[q.question_id] = fixed
        decisions[q.question_id] = {
            "original_split": q.split,
            "component_id": groups[q.question_id],
            "reference_action": action,
            "exclusion_reasons": [],
            "original_record_sha256": digest(canonical(q.record())),
            "corrected_record_sha256": digest(canonical(fixed.record())),
            "original_answer_sha256": digest(canonical(q.answer)),
            "corrected_answer_sha256": digest(canonical(fixed.answer)),
        }
        if action.startswith("quarantine"):
            decisions[q.question_id]["exclusion_reasons"].append(action)
    # Normalization is intentionally conservative: conflicting labels are not adjudicated.
    exact: dict[str, list[Question]] = defaultdict(list)
    for q in rows:
        exact[normalize(q.question)].append(q)
    for cluster in exact.values():
        labels = {(q.answerable, normalize(q.answer or "")) for q in cluster}
        if len(labels) > 1:
            for q in cluster:
                decisions[q.question_id]["exclusion_reasons"].append("conflicting_exact_question")
    dev_groups = {groups[q.question_id] for q in rows if q.split == "validation"}
    for q in rows:
        if q.split == "train" and groups[q.question_id] in dev_groups:
            decisions[q.question_id]["exclusion_reasons"].append("component_touches_development")
    # Deduplicate within the remaining roles; quarantine first so bad records cannot win.
    seen: dict[str, str] = {}
    eligible = []
    for q in rows:
        item = decisions[q.question_id]
        if item["exclusion_reasons"]:
            continue
        text = normalize(q.question)
        if text in seen:
            item["exclusion_reasons"].append("duplicate_exact_question")
            item["retained_question_id"] = seen[text]
            continue
        seen[text] = q.question_id
        eligible.append(corrected[q.question_id])
    training_groups: dict[str, list[Question]] = defaultdict(list)
    for q in eligible:
        if q.split == "train":
            training_groups[groups[q.question_id]].append(q)
    strata: dict[str, list[str]] = defaultdict(list)
    for group, members in training_groups.items():
        flags = {q.answerable for q in members}
        stratum = "mixed" if len(flags) > 1 else ("answerable" if True in flags else "unanswerable")
        strata[stratum].append(group)
    heldout: set[str] = set()
    allocation = {}
    for stratum, ids in sorted(strata.items()):
        ordered = sorted(
            ids,
            key=lambda gid: digest(
                f"{policy['protocol_id']}:{policy['seed']}:{stratum}:{gid}".encode()
            ),
        )
        # A singleton stratum stays in training; never split a group to hit a row quota.
        count = min(
            len(ids) - 1, max(1, math.floor(len(ids) * policy["test_group_fraction"] + 0.5))
        )
        count = max(0, count)
        heldout.update(ordered[:count])
        allocation[stratum] = {"groups": len(ids), "test_groups": count}
    assigned = [
        replace(q, split="test") if q.split == "train" and groups[q.question_id] in heldout else q
        for q in eligible
    ]
    split_by_id = {q.question_id: q.split for q in assigned}
    for qid, item in decisions.items():
        item["assigned_split"] = split_by_id.get(qid)
    membership: dict[str, set[str]] = defaultdict(set)
    for q in assigned:
        membership[groups[q.question_id]].add(q.split)
    if any(len(splits) > 1 for splits in membership.values()):
        raise LabError("Internal error: contamination component crosses experiment splits")
    if {q.split for q in assigned} != {"train", "validation", "test"}:
        raise LabError("Insufficient eligible components to create nonempty train/validation/test")
    qa_overlap = overlaps(assigned, policy["near_duplicate_threshold"])
    summary = {
        "protocol_id": policy["protocol_id"],
        "source_rows": len(rows),
        "eligible_rows": len(assigned),
        "excluded_rows": len(rows) - len(assigned),
        "splits": {
            s: {
                "rows": sum(q.split == s for q in assigned),
                "answerable": sum(q.split == s and q.answerable for q in assigned),
                "unanswerable": sum(q.split == s and not q.answerable for q in assigned),
                "components": len({groups[q.question_id] for q in assigned if q.split == s}),
            }
            for s in ("train", "validation", "test")
        },
        "reference_actions": dict(
            sorted(Counter(d["reference_action"] for d in decisions.values()).items())
        ),
        "exclusion_reason_counts_nonexclusive": dict(
            sorted(
                Counter(
                    reason for d in decisions.values() for reason in d["exclusion_reasons"]
                ).items()
            )
        ),
        "group_strata": allocation,
        "cross_split_components": 0,
        "remaining_overlap": qa_overlap,
        "final_test_description": "audit-exposed internal holdout from official training; "
        "not official test, not independently collected, not untouched",
    }
    return assigned, decisions, summary


def prepare_protocol(
    rows: list[Question],
    corpus_path: Path,
    policy: dict[str, Any],
    source_manifests: Path,
    manifests: Path,
    processed: Path,
    config_sha256: str,
) -> dict[str, Any]:
    """Verify lineage, preflight all outputs, and write a separately versioned protocol."""
    source_lock = verify(source_manifests)
    if source_lock["dataset_config_sha256"] != config_sha256:
        raise LabError("Dataset config differs from the frozen source snapshot")
    expected = {}
    if file_hash(corpus_path) != source_lock["sources"]["training_dev_technotes.json"]["sha256"]:
        raise LabError("Corpus differs from the frozen original source")
    docs = original_documents(corpus_path)

    for split in ("train", "validation"):
        for line in (source_manifests / f"{split}.jsonl").read_text().splitlines():
            entry = json.loads(line)
            expected[entry["question_id"]] = entry["record_sha256"]
    if {q.question_id: digest(canonical(q.record())) for q in rows} != expected:
        raise LabError("Source records differ from frozen Phase 1 assignments")
    corpus_records = [vars(docs[key]) for key in sorted(docs)]
    corpus_hash = digest(jsonl(corpus_records))
    assigned, decisions, summary = build_protocol(rows, docs, policy)
    artifacts = {
        "protocol.json": canonical(policy),
        "decisions.json": canonical(decisions),
        "summary.json": canonical(summary),
        "corpus.json": canonical(
            {
                "documents": len(docs),
                "processed_corpus_sha256": corpus_hash,
                "original_corpus_sha256": source_lock["sources"]["training_dev_technotes.json"][
                    "sha256"
                ],
                "setting": "open_book_official_train_dev_document_subset",
                "annotation_content_allowed": False,
            }
        ),
    }
    manifest_bytes = {}
    roles = {
        "train": "model_training",
        "validation": "development_selection",
        "test": "sealed_final_evaluation",
    }
    source_revision = json.loads((source_manifests / "train.jsonl").read_text().splitlines()[0])[
        "source_revision"
    ]
    for split in roles:
        manifest_bytes[f"{split}.jsonl"] = jsonl(
            [
                {
                    "question_id": q.question_id,
                    "source": q.source,
                    "split": q.split,
                    "original_split": decisions[q.question_id]["original_split"],
                    "component_id": decisions[q.question_id]["component_id"],
                    "record_sha256": digest(canonical(q.record())),
                    "source_revision": source_revision,
                    "role": roles[split],
                }
                for q in assigned
                if q.split == split
            ]
        )
    lock = {
        "schema_version": 2,
        "protocol_id": policy["protocol_id"],
        "experimental_use_ready": True,
        "experimental_use_blockers": [],
        "final_test_ready": True,
        "final_test_access": policy["final_test_access"],
        "test_status": "reserved_audit_exposed_internal_holdout",
        "dataset_config_sha256": config_sha256,
        "source_freeze_sha256": file_hash(source_manifests / "freeze.json"),
        "files": {name: digest(value) for name, value in manifest_bytes.items()},
        "artifacts": {name: digest(value) for name, value in artifacts.items()},
    }
    outputs = {manifests / name: value for name, value in (artifacts | manifest_bytes).items()}
    outputs[manifests / "freeze.json"] = canonical(lock)
    for split in ("train", "validation"):
        outputs[processed / f"{split}_annotations.jsonl"] = jsonl(
            [q.record() for q in assigned if q.split == split]
        )
    # Do not materialize test annotation text in the Phase 2 working directory.
    outputs[processed / "source_documents.jsonl"] = jsonl(corpus_records)
    if (processed / "test_annotations.jsonl").exists():
        raise LabError("Sealed test annotations must not exist in the Phase 2 processed directory")
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise LabError(f"Frozen protocol output differs: {path}; create a new version")
    for path, content in outputs.items():
        atomic_write(path, content)
    return summary
