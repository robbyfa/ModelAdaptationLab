"""Synthetic-only schema and evidence tests."""

from dataclasses import replace

import pytest

from model_adaptation_lab.data.audit import derivative_comparison, evidence_audit, overlaps
from model_adaptation_lab.data.ingest import (
    Document,
    nvidia_questions,
    original_documents,
    original_questions,
)
from model_adaptation_lab.io import LabError


def test_original_schema_and_decimal_offsets(row, write_json):
    q = original_questions(write_json([row]), "train")[0]
    assert q.answerable and q.start == 0 and q.end == 9
    assert q.question.startswith("Synthetic")


@pytest.mark.parametrize(
    "key,value",
    [
        ("ANSWERABLE", True),
        ("DOC_IDS", "doc"),
        ("DOC_IDS", [2]),
        ("START_OFFSET", True),
        ("START_OFFSET", 0),
        ("START_OFFSET", "-1"),
        ("END_OFFSET", "0"),
        ("ANSWER", None),
        ("DOCUMENT", "missing"),
        ("QUESTION_ID", ""),
        ("QUESTION_TEXT", 7),
    ],
)
def test_invalid_original_schema(row, write_json, key, value):
    row[key] = value
    with pytest.raises(LabError):
        original_questions(write_json([row]), "train")


def test_unanswerable_not_false_duplicate(row, write_json):
    row.update(ANSWERABLE="N", ANSWER="-", DOCUMENT="-", START_OFFSET="-", END_OFFSET="-")
    q = original_questions(write_json([row]), "train")[0]
    assert q.answer is None and not q.answerable
    report = overlaps(
        [q, replace(q, question_id="other", question="Unrelated synthetic query")], 0.85
    )
    assert report["identical_normalized_answer_groups"] == []


def test_duplicate_id_rejected(row, write_json):
    with pytest.raises(LabError, match="Duplicate"):
        original_questions(write_json([row, row]), "train")


def test_real_source_empty_reference_is_preserved_for_audit(row, write_json):
    row["ANSWER"] = ""
    q = original_questions(write_json([row]), "validation")[0]
    assert q.answer == "" and q.answerable
    report = evidence_audit(
        [q], {"synthetic-doc": Document("synthetic-doc", "", "Reset it.", None)}
    )
    assert report["span_mismatch_question_ids"] == [q.question_id]


def test_exact_near_and_answer_overlap(row, write_json):
    q = original_questions(write_json([row]), "train")[0]
    near = replace(q, question_id="near", question=q.question + " please", split="validation")
    exact = replace(q, question_id="same", question=q.question.upper(), split="validation")
    report = overlaps([q, near, exact], 0.8)
    assert len(report["exact_question_groups"]) == 1
    assert len(report["near_question_pairs"]) == 2
    assert len(report["identical_normalized_answer_groups"][0]) == 3


def test_corpus_shape_and_id_validation(write_json):
    with pytest.raises(LabError, match="key/id"):
        original_documents(write_json({"x": {"id": "y"}}))
    with pytest.raises(LabError, match="keyed"):
        original_documents(write_json([]))


def test_evidence_coverage_missing_and_duplicate_documents(row, write_json):
    q = original_questions(write_json([row]), "train")[0]
    docs = {
        "synthetic-doc": Document("synthetic-doc", "", "Reset it.", "https://example.org"),
        "copy": Document("copy", "", "RESET IT.", None),
    }
    report = evidence_audit([q], docs)
    assert report["span_matches"] == 1
    assert report["gold_document_coverage"] == 1
    assert len(report["duplicate_normalized_document_groups"]) == 1
    assert evidence_audit([q], {})["gold_document_coverage"] == 0
    assert evidence_audit([q], None)["gold_document_coverage"] is None


def test_nvidia_is_inspection_only(row, write_json):
    original = original_questions(write_json([row]), "train")
    data = [
        {
            "id": row["QUESTION_ID"],
            "question": original[0].question,
            "answer": "Reset it.",
            "is_impossible": False,
            "contexts": [{"filename": "synthetic-doc.txt", "text": "Reset it."}],
        }
    ]
    questions, contexts = nvidia_questions(write_json(data))
    assert questions[0].split == "inspection_only"
    assert contexts[0].source == "nvidia_annotation_context"
    report = derivative_comparison(original, questions, contexts, None)
    assert report["original_split_matches"] == {"train": 1}
    data[0]["is_impossible"] = "false"
    with pytest.raises(LabError, match="boolean"):
        nvidia_questions(write_json(data))
