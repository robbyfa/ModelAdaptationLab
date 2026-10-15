"""Synthetic-only protocol and sealed-holdout tests; never load real holdout text."""

import json
from dataclasses import replace

import pytest

from model_adaptation_lab.data.ingest import Document, Question
from model_adaptation_lab.data.protocol import (
    POLICY,
    build_protocol,
    components,
    correct_reference,
    load_protocol,
    prepare_protocol,
)
from model_adaptation_lab.data.splits import freeze, load_annotations, load_split, verify
from model_adaptation_lab.io import LabError, file_hash


def question(qid, text, split="train", answer="Answer", doc="doc"):
    return Question(
        qid,
        text,
        answer,
        answer is not None,
        doc if answer else None,
        (doc,),
        "original",
        split,
        0 if answer else None,
        len(answer) if answer else None,
    )


def synthetic_data():
    rows, docs = [], {}
    for i in range(30):
        qid = f"SYNTH_T{i:02}"
        answer = f"Unique answer {i}"
        docs[qid] = Document(qid, "Synthetic", answer, None)
        rows.append(question(qid, f"query{i}", answer=answer, doc=qid))
    docs["dev"] = Document("dev", "Synthetic", "Development response", None)
    rows.append(
        question("SYNTH_DEV", "validationrequest", "validation", "Development response", "dev")
    )
    return rows, docs


def frozen_source(tmp_path, rows, docs):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps({key: {"id": key, "title": d.title, "text": d.text} for key, d in docs.items()})
    )
    audit = {
        "splits": {"train": {"rows": 30}, "validation": {"rows": 1}},
        "cross_split_exact_question_groups": [],
        "evidence": {"status": "available"},
        "source_files_verified": True,
        "dataset_config_sha256": "c" * 64,
        "dataset_config": {"original": {"revision": "a" * 40}},
        "sources": {"training_dev_technotes.json": {"sha256": file_hash(corpus)}},
    }
    source = tmp_path / "source"
    freeze(rows, audit, source)
    return corpus, source


def test_whitespace_correction_preserves_content():
    docs = {"doc": Document("doc", "", "Reset\n  it.", None)}
    q = replace(question("synthetic", "query", answer="Reset it."), end=11)
    fixed, action = correct_reference(q, docs)
    assert action == "corrected_whitespace_only"
    assert fixed.answer == "Reset\n  it."
    assert q.answer == "Reset it."


@pytest.mark.parametrize("answer", ["", "reset it.", "Reset it!", "Other content"])
def test_non_whitespace_changes_are_not_invented(answer):
    docs = {"doc": Document("doc", "", "Reset it.", None)}
    q = replace(question("synthetic", "query", answer=answer), end=9)
    fixed, action = correct_reference(q, docs)
    assert action.startswith("quarantine")
    assert fixed == q


def test_missing_evidence_invalid_offsets_and_unanswerable():
    q = question("synthetic", "query")
    assert correct_reference(q, {})[1] == "quarantine_missing_evidence"
    assert (
        correct_reference(q, {"doc": Document("doc", "", "x", None)})[1]
        == "quarantine_invalid_offsets"
    )
    unanswerable = question("n", "q", answer=None)
    assert correct_reference(unanswerable, {}) == (unanswerable, "unchanged")


def test_transitive_question_and_answer_grouping():
    a = question("a", "one two three four", answer="first")
    b = question("b", "one two three four five", answer="second")
    c = question("c", "one two three four five six", answer="third")
    d = question("d", "completely different", answer="third")
    groups = components([d, c, b, a], 0.8)
    assert len(set(groups.values())) == 1
    assert (
        len(
            set(
                components(
                    [question("n1", "alpha", answer=None), question("n2", "beta", answer=None)], 0.8
                ).values()
            )
        )
        == 2
    )


def test_assignment_stability_and_no_cross_split_components():
    rows, docs = synthetic_data()
    result = build_protocol(rows, docs, POLICY)
    assert build_protocol(list(reversed(rows)), docs, POLICY) == result
    assigned, decisions, summary = result
    assert summary["splits"]["test"]["rows"] == 6
    assert summary["splits"]["train"]["rows"] == 24
    assert all(
        decisions[q.question_id]["original_split"] == "train" for q in assigned if q.split == "test"
    )
    group_roles = {}
    for q in assigned:
        group_roles.setdefault(decisions[q.question_id]["component_id"], set()).add(q.split)
    assert all(len(roles) == 1 for roles in group_roles.values())


def test_dev_overlap_excludes_train_even_when_dev_quarantined():
    rows, docs = synthetic_data()
    rows += [question("SYNTH_BAD_DEV", rows[0].question, "validation", "", rows[0].document_id)]
    assigned, decisions, _ = build_protocol(rows, docs, POLICY)
    assert rows[0].question_id not in {q.question_id for q in assigned}
    assert "component_touches_development" in decisions[rows[0].question_id]["exclusion_reasons"]
    assert "conflicting_exact_question" in decisions["SYNTH_BAD_DEV"]["exclusion_reasons"]


def test_exact_duplicates_do_not_overweight_results():
    rows, docs = synthetic_data()
    rows += [replace(rows[0], question_id="ZZ_COPY")]
    assigned, decisions, _ = build_protocol(rows, docs, POLICY)
    assert "ZZ_COPY" not in {q.question_id for q in assigned}
    assert decisions["ZZ_COPY"]["retained_question_id"] == rows[0].question_id


def test_versioned_prepare_seals_test_and_preserves_source(tmp_path):
    rows, docs = synthetic_data()
    corpus, source = frozen_source(tmp_path, rows, docs)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    dest, processed = tmp_path / "v1", tmp_path / "processed"
    args = (rows, corpus, POLICY, source, dest, processed, "c" * 64)
    summary = prepare_protocol(*args)
    assert prepare_protocol(*args) == summary
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert verify(dest)["experimental_use_ready"]
    assert load_annotations(dest, processed, "train", purpose="training")
    assert load_annotations(dest, processed, "validation", purpose="selection")
    assert not (processed / "test_annotations.jsonl").exists()
    with pytest.raises(LabError, match="sealed"):
        load_split(dest, "test", purpose="final_evaluation")
    with pytest.raises(LabError, match="cannot be accessed"):
        load_split(dest, "test", purpose="selection")
    (dest / "decisions.json").write_text("{}")
    with pytest.raises(LabError, match="hash mismatch"):
        verify(dest)
    with pytest.raises(LabError, match="Frozen protocol output differs"):
        prepare_protocol(*args)


def test_source_and_corpus_tampering_refused(tmp_path):
    rows, docs = synthetic_data()
    corpus, source = frozen_source(tmp_path, rows, docs)
    args = (corpus, POLICY, source, tmp_path / "v1", tmp_path / "processed", "c" * 64)
    with pytest.raises(LabError, match="Source records differ"):
        prepare_protocol([replace(rows[0], answer="tampered"), *rows[1:]], *args)
    corpus.write_text("{}")
    with pytest.raises(LabError, match="Corpus differs"):
        prepare_protocol(rows, *args)


def test_no_mutation_of_v1_policy_or_derivative_fallback(tmp_path):
    rows, docs = synthetic_data()
    with pytest.raises(LabError, match="implemented v1"):
        build_protocol(rows, docs, dict(POLICY, seed=13))
    with pytest.raises(LabError, match="original train"):
        build_protocol([replace(rows[0], source="nvidia"), *rows[1:]], docs, POLICY)
    p = tmp_path / "policy.yaml"
    p.write_text("protocol_id: changed")
    with pytest.raises(LabError, match="version a new policy"):
        load_protocol(p)


def test_seal_metadata_removal_fails_closed(tmp_path):
    rows, docs = synthetic_data()
    corpus, source = frozen_source(tmp_path, rows, docs)
    dest = tmp_path / "v1"
    prepare_protocol(rows, corpus, POLICY, source, dest, tmp_path / "processed", "c" * 64)
    path = dest / "freeze.json"
    lock = json.loads(path.read_text())
    del lock["final_test_access"]
    path.write_text(json.dumps(lock))
    with pytest.raises(LabError, match="seal is missing"):
        load_split(dest, "test", purpose="final_evaluation")
