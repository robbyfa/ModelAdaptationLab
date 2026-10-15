"""Deterministic, text-redacted audits; overlap is evidence, not a leakage-free claim."""

import platform
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from model_adaptation_lab.data.ingest import (
    Document,
    Question,
    nvidia_questions,
    original_documents,
    original_questions,
    unique_ids,
)
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, file_hash


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def identity(q: Question) -> str:
    return f"{q.source}:{q.split}:{q.question_id}"


def overlaps(rows: list[Question], threshold: float) -> dict[str, Any]:
    exact: dict[str, list[str]] = defaultdict(list)
    answers: dict[str, list[str]] = defaultdict(list)
    texts, tokens = [], []
    for q in rows:
        text = normalize(q.question)
        texts.append(text)
        tokens.append(set(text.split()))
        exact[digest(text.encode())].append(identity(q))
        # '-' / empty answers must never create artificial unanswerable overlap.
        if q.answer:
            answers[digest(normalize(q.answer).encode())].append(identity(q))
    near = []
    for i, j in combinations(range(len(rows)), 2):
        a, b = tokens[i], tokens[j]
        if texts[i] == texts[j] or not a or not b:
            continue
        if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
            continue
        score = len(a & b) / len(a | b)
        if score >= threshold:
            near.append(
                {
                    "left": identity(rows[i]),
                    "right": identity(rows[j]),
                    "score": round(score, 6),
                    "cross_split": rows[i].split != rows[j].split,
                }
            )
    return {
        "exact_question_groups": [v for _, v in sorted(exact.items()) if len(v) > 1],
        "identical_normalized_answer_groups": [v for _, v in sorted(answers.items()) if len(v) > 1],
        "near_question_pairs": near,
        "near_duplicate_method": "NFKC + casefold + word-set Jaccard, exhaustive pairs",
        "near_duplicate_threshold": threshold,
        "limitations": "Heuristic only: misses semantic paraphrases; flags require review.",
    }


def evidence_audit(rows: list[Question], docs: dict[str, Document] | None) -> dict[str, Any]:
    if docs is None:
        return {
            "status": "unavailable",
            "gold_document_coverage": None,
            "span_matches": None,
            "missing_gold_ids": None,
        }
    answerable = [q for q in rows if q.answerable]
    missing = sorted(
        {
            q.document_id
            for q in answerable
            if q.document_id is not None and q.document_id not in docs
        }
    )
    mismatch, matches = [], 0
    for q in answerable:
        if q.document_id in docs:
            doc = docs[q.document_id]
            if doc.text[q.start : q.end] == q.answer:
                matches += 1
            else:
                mismatch.append(q.question_id)
    candidates = {d for q in rows for d in q.candidate_document_ids}
    doc_hashes: dict[str, list[str]] = defaultdict(list)
    for doc in docs.values():
        doc_hashes[digest(normalize(doc.text).encode())].append(doc.document_id)
    split_docs = {
        s: {d for q in rows if q.split == s for d in q.candidate_document_ids}
        for s in sorted({q.split for q in rows})
    }
    return {
        "status": "available",
        "documents": len(docs),
        "gold_document_coverage": (
            sum(q.document_id in docs for q in answerable) / len(answerable) if answerable else None
        ),
        "span_matches": matches,
        "span_mismatch_question_ids": mismatch,
        "missing_gold_ids": missing,
        "missing_candidate_ids": sorted(candidates - docs.keys()),
        "documents_with_source_urls": sum(bool(d.source_url) for d in docs.values()),
        "empty_document_ids": sorted(d.document_id for d in docs.values() if not d.text.strip()),
        "duplicate_normalized_document_groups": [
            sorted(v) for _, v in sorted(doc_hashes.items()) if len(v) > 1
        ],
        "candidate_document_overlap": {
            f"{a}/{b}": sorted(split_docs[a] & split_docs[b])
            for a, b in combinations(split_docs, 2)
        },
        "interpretation": "Shared source documents are permitted in open-book RAG; annotations "
        "and per-question candidate lists must not become retrieval content.",
    }


def derivative_comparison(
    original: list[Question],
    derivative: list[Question],
    contexts: list[Document],
    docs: dict[str, Document] | None,
) -> dict[str, Any]:
    by_id = {q.question_id: q for q in original}
    matched = [q for q in derivative if q.question_id in by_id]
    context_hashes: dict[str, set[str]] = defaultdict(set)
    for ctx in contexts:
        context_hashes[ctx.document_id].add(digest(ctx.text.encode()))
    return {
        "rows": len(derivative),
        "declared_card_rows": 908,
        "card_count_matches": len(derivative) == 908,
        "role": "inspection_only_never_training",
        "published_split_label": "train",
        "answerable": sum(q.answerable for q in derivative),
        "unanswerable": sum(not q.answerable for q in derivative),
        "id_prefix_counts": dict(
            sorted(Counter(q.question_id.split("_")[0] for q in derivative).items())
        ),
        "original_id_matches": len(matched),
        "original_split_matches": dict(
            sorted(Counter(by_id[q.question_id].split for q in matched).items())
        ),
        "matching_id_question_mismatches": [
            q.question_id
            for q in matched
            if normalize(q.question) != normalize(by_id[q.question_id].question)
        ],
        "matching_id_answer_mismatches": [
            q.question_id for q in matched if q.answer != by_id[q.question_id].answer
        ],
        "matching_id_answerability_mismatches": [
            q.question_id for q in matched if q.answerable != by_id[q.question_id].answerable
        ],
        "unique_context_ids": len(context_hashes),
        "context_ids_with_multiple_text_versions": sorted(
            k for k, v in context_hashes.items() if len(v) > 1
        ),
        "context_ids_present_in_original_corpus": (
            sum(k in docs for k in context_hashes) if docs is not None else None
        ),
        "contexts_containing_original_body": (
            sum(c.document_id in docs and docs[c.document_id].text in c.text for c in contexts)
            if docs is not None
            else None
        ),
        "context_instances": len(contexts),
        "answerable_without_context": sum(
            q.answerable and not q.candidate_document_ids for q in derivative
        ),
    }


def run_audit(raw: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[Question]]:
    rows: list[Question] = []
    sources: dict[str, Any] = {}
    issues: list[str] = []
    for filename, split in (("training_Q_A.json", "train"), ("dev_Q_A.json", "validation")):
        path = raw / "original" / filename
        if path.exists():
            loaded = original_questions(path, split)
            rows.extend(loaded)
            sources[filename] = {"sha256": file_hash(path), "rows": len(loaded)}
        else:
            issues.append(f"Missing official {filename}; run model-lab data download")
    unique_ids(rows)
    corpus = raw / "original" / "training_dev_technotes.json"
    docs = original_documents(corpus) if corpus.exists() else None
    if docs is not None:
        sources[corpus.name] = {"sha256": file_hash(corpus), "documents": len(docs)}
    else:
        issues.append("Original source documents unavailable; evidence coverage cannot be measured")
    derivative = raw / "nvidia" / "train.json"
    derivative_rows, contexts = nvidia_questions(derivative) if derivative.exists() else ([], [])
    if derivative.exists():
        sources["nvidia/train.json"] = {"sha256": file_hash(derivative)}
    else:
        issues.append("NVIDIA derivative unavailable; separate inspection is incomplete")
    archive = raw / "original" / "TechQA.tar.gz"
    archive_verified = False
    if archive.exists():
        archive_hash = file_hash(archive)
        archive_verified = archive_hash == config["original"]["sha256"]
        if not archive_verified:
            raise LabError("Official archive checksum does not match pinned LFS SHA-256")
        sources[archive.name] = {"sha256": archive_hash}
    else:
        issues.append("Archive absent: local file hashes recorded, upstream lineage unverified")
    for name in ("README.txt", "CDLA-Permissive-v1.0.pdf"):
        path = raw / "original" / name
        if path.exists():
            sources[name] = {"sha256": file_hash(path)}
        else:
            issues.append(f"Archive licensing evidence missing: {name}")
    expected_files = dict(config["original"].get("extracted_sha256", {}))
    expected_files["nvidia/train.json"] = config["nvidia"].get("sha256")
    verified_files = bool(expected_files) and all(
        name in sources and expected and sources[name]["sha256"] == expected
        for name, expected in expected_files.items()
    )
    for name, expected in expected_files.items():
        if name in sources and expected and sources[name]["sha256"] != expected:
            raise LabError(f"Source file differs from pinned bytes: {name}")
    if not verified_files:
        issues.append("Extracted source lineage is unverified; experimental use blocked")
    overlap = overlaps(rows, config["policy"]["near_duplicate_threshold"])
    evidence = evidence_audit(rows, docs)
    empty_answers = [q.question_id for q in rows if q.answerable and not q.answer]
    evidence["empty_answerable_reference_ids"] = empty_answers
    if evidence.get("span_mismatch_question_ids"):
        issues.append(
            "Reference answers disagree with source offsets; experimental use blocked "
            "pending annotation policy (no automatic repairs)"
        )

    cross_exact = [
        g for g in overlap["exact_question_groups"] if len({x.split(":")[1] for x in g}) > 1
    ]
    if cross_exact:
        issues.append("Exact questions cross official splits; training use blocked pending review")
    issues.extend(
        [
            "No official final test set imported; "
            "final evaluation is blocked pending a written protocol",
            "Archive declares CDLA-Permissive-1.0; hub card declares Apache-2.0. "
            "Preserve archive terms; "
            "resolve the metadata discrepancy before redistribution",
            "Full 801,998-technote collection is not indexed or exhaustively audited in Phase 1",
        ]
    )
    report = {
        "schema_version": 1,
        "scope": "Phase 1 only; no training or performance evaluation",
        "dataset_config": config,
        "dataset_config_sha256": digest(canonical(config)),
        "sources": sources,
        "archive_checksum_verified": archive_verified,
        "source_files_verified": verified_files,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "splits": {
            s: {
                "rows": sum(q.split == s for q in rows),
                "answerable": sum(q.split == s and q.answerable for q in rows),
                "unanswerable": sum(q.split == s and not q.answerable for q in rows),
            }
            for s in ("train", "validation", "test")
        },
        "overlap": overlap,
        "cross_split_exact_question_groups": cross_exact,
        "evidence": evidence,
        "derivative": derivative_comparison(rows, derivative_rows, contexts, docs)
        if derivative.exists()
        else {"status": "unavailable"},
        "issues": issues,
        "final_test_ready": False,
    }
    return report, rows


def write_report(report: dict[str, Any], directory: Path) -> None:
    atomic_write(directory / "dataset_audit.json", canonical(report))
    lines = [
        "# TechQA dataset audit",
        "",
        report["scope"],
        "",
        "Generated from pinned source files; no question or answer text is printed.",
        "",
        "| Official role | Rows | Answerable | Unanswerable |",
        "|---|---:|---:|---:|",
    ]
    for split, stats in report["splits"].items():
        lines.append(
            f"| {split} | {stats['rows']} | {stats['answerable']} | {stats['unanswerable']} |"
        )
    ev, ov, der = report["evidence"], report["overlap"], report["derivative"]
    lines.extend(
        [
            "",
            "## Provenance and evidence",
            "",
            f"- Official archive checksum verified: {report['archive_checksum_verified']}.",
            f"- Original document subset: {ev.get('documents', 'unavailable')} documents.",
            f"- Gold-document coverage: {ev.get('gold_document_coverage')}.",
            f"- Exact answer-span matches: {ev.get('span_matches')}.",
            "- NVIDIA derivative remains inspection-only, regardless of its `train` label.",
            f"- Derivative rows: {der.get('rows', 'unavailable')}; card declares 908.",
            f"- Derivative ID matches by official split: {der.get('original_split_matches')}.",
            "",
            "## Overlap checks",
            "",
            f"- Exact normalized question groups: {len(ov['exact_question_groups'])}.",
            f"- Cross-split exact groups: {len(report['cross_split_exact_question_groups'])}.",
            f"- Near-question pairs: {len(ov['near_question_pairs'])} "
            f"(word-set Jaccard ≥ {ov['near_duplicate_threshold']}).",
            f"- Identical normalized answer groups: "
            f"{len(ov['identical_normalized_answer_groups'])}.",
            "- Shared documents are allowed only under the documented open-book protocol.",
            "- Near duplicates are review candidates, not proven contamination. "
            "No claim of leakage-free splits is made.",
            "",
            "## Limitations and decisions",
            "",
        ]
    )
    lines.extend(f"- {issue}" for issue in report["issues"])
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "- [IBM baseline repository](https://github.com/IBM/techqa)",
            "- [Original archive](https://huggingface.co/datasets/PrimeQA/TechQA)",
            "- [NVIDIA derivative](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)",
            "",
            "Exact revisions, input hashes, record IDs, span mismatches, source coverage, "
            "and document-overlap lists are in `dataset_audit.json`.",
            "",
        ]
    )
    atomic_write(directory / "dataset_audit.md", "\n".join(lines).encode())
