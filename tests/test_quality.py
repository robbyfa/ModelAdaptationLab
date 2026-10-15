"""Synthetic-only quality fixtures: simulated reviews are never real human validation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_adaptation_lab.data.splits import load_annotations
from model_adaptation_lab.evaluation.metrics import diagnostics
from model_adaptation_lab.evaluation.quality.packet import prepare_packet, verify_packet
from model_adaptation_lab.evaluation.quality.render import render_packet
from model_adaptation_lab.evaluation.quality.review import (
    agreement,
    load_review,
    rate_summary,
    score_reviews,
    validate_rating,
)
from model_adaptation_lab.evaluation.quality.semantic import similarity
from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, file_hash, jsonl
from model_adaptation_lab.retrieval.chunking import load_corpus


@pytest.fixture
def quality_packet(tmp_path, synthetic_protocol):
    manifests, processed = synthetic_protocol
    rows = load_annotations(manifests, processed, "validation", purpose="selection")
    documents, _ = load_corpus(manifests, processed)
    docs = {d["document_id"]: d for d in documents}
    runs = []
    for index, names in enumerate((["base", "prompt", "rag"], ["lora", "lora-rag"])):
        run = tmp_path / f"synthetic-run-{index}"
        run.mkdir()
        summaries = {}
        for name in names:
            records = []
            for row in rows:
                rag = name in {"rag", "lora-rag"}
                answer = (
                    row["answer"]
                    if row["answerable"]
                    else "I don't have enough information to answer this question."
                )
                chunks = []
                if rag and row["answerable"]:
                    d = docs[row["document_id"]]
                    chunks = [
                        d | {"start": 0, "end": len(d["text"]), "chunk_id": "synthetic-chunk"}
                    ]
                    answer += f" [{d['document_id']}]"
                records.append(
                    {
                        "question_id": row["question_id"],
                        "strategy": name,
                        "status": "ok",
                        "answer": answer,
                        "finish_reason": "eos",
                        "retrieved_chunks": chunks,
                    }
                )
            atomic_write(run / f"{name}.jsonl", jsonl(records))
            summaries[name] = {"file_sha256": file_hash(run / f"{name}.jsonl")}
        state = {
            "scope": "validation_only",
            "status": "completed",
            "backend_type": "real_transformers",
            "fixture": "synthetic simulated producer",
            "freeze_sha256": file_hash(manifests / "freeze.json"),
            "validation_ids": [r["question_id"] for r in rows],
            "validation_records_sha256": digest(canonical(rows)),
            "model_config": {"revision": "a" * 40},
            "generation_config": {"max_new_tokens": 16},
            "smoke": True,
            "results": summaries,
        }
        atomic_write(run / "run.json", canonical(state))
        runs.append(run)
    packet = tmp_path / "packet"
    prepare_packet(
        runs=runs,
        manifests=manifests,
        processed=processed,
        output=packet,
        config_path=Path("configs/quality-v1.yaml"),
        rubric_path=Path("reports/quality-rubric-v1.md"),
    )
    return packet, manifests, processed, docs, runs


def completed_review(packet, docs, reviewer="synthetic-reviewer-A"):
    template = json.loads((packet / "review_template.json").read_text())
    _, cases = verify_packet(packet)
    by_id = {c["item_id"]: c for c in cases}
    template["reviewer_id"] = reviewer
    for r in template["items"]:
        c = by_id[r["item_id"]]
        r.update(
            correctness=3 if c["answerable"] else None,
            correctness_unavailable_reason="" if c["answerable"] else "No annotated reference",
            behavior="answer" if c["answerable"] else "abstain",
            reference_sufficiency="sufficient" if c["answerable"] else "uncertain",
            retrieved_sufficiency=("sufficient" if c["answerable"] else "uncertain")
            if c["retrieval_applicable"]
            else None,
            reviewed_document_ids=[d["document_id"] for d in c["reference_documents"]],
            rationale="Synthetic fixture judgment, not real human review",
        )
        for claim in r["claims"]:
            claim["kind"] = "factual" if c["answerable"] else "nonfactual"
            claim["rationale"] = "Synthetic explicit source matches"
            if c["answerable"]:
                doc = c["reference_documents"][0]
                evidence = {
                    "document_id": doc["document_id"],
                    "start": 0,
                    "end": len(doc["text"]),
                    "quote": doc["text"],
                }
                claim["reference_support"] = "supported"
                claim["reference_evidence"] = [evidence]
                if c["retrieval_applicable"]:
                    claim["retrieved_support"] = "supported"
                    claim["retrieved_evidence"] = [evidence]
                    claim["cited_document_ids"] = [doc["document_id"]]
                    claim["citation_support"] = "supported"
    return template, by_id


def test_packet_common_evidence_blinding_and_integrity(quality_packet):
    packet, _, _, _, _ = quality_packet
    _, cases = verify_packet(packet)
    assert len(cases) == 10
    assert all("strategy" not in c for c in cases)
    for q in {c["question_id"] for c in cases}:
        pools = [c["reference_documents"] for c in cases if c["question_id"] == q]
        assert all(p == pools[0] for p in pools)
    with (packet / "cases.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(LabError, match="changed"):
        verify_packet(packet)


def test_noncited_nonrag_is_supported_not_penalized(quality_packet):
    packet, _, _, docs, _ = quality_packet
    review, cases = completed_review(packet, docs)
    rates = []
    for r in review["items"]:
        validate_rating(r, cases[r["item_id"]], docs)
        value = rate_summary(r, cases[r["item_id"]])
        rates.append(value)
        if not cases[r["item_id"]]["retrieval_applicable"]:
            assert value["retrieved_support_rate"] is None
            assert value["citation_coverage"] is None
    assert {r["reference_support_rate"] for r in rates if r["correctness"] == 3} == {1.0}


@pytest.mark.parametrize(
    "change,expected",
    [
        ("bad_quote", "quote"),
        ("outside_passage", "outside"),
        ("missing_quote", "require source"),
        ("omitted_text", "cover exact"),
        ("bad_correctness", "0..3"),
        ("unknown_doc", "unknown"),
        ("nonrag_scope", "scope"),
        ("fake_citation", "not present"),
    ],
)
def test_review_rejects_invalid_claim_evidence(quality_packet, change, expected):
    packet, _, _, docs, _ = quality_packet
    review, cases = completed_review(packet, docs)
    r = next(
        r
        for r in review["items"]
        if cases[r["item_id"]]["answerable"] and cases[r["item_id"]]["retrieval_applicable"]
    )
    c = copy.deepcopy(cases[r["item_id"]])
    r = copy.deepcopy(r)
    if change == "bad_quote":
        r["claims"][0]["reference_evidence"][0]["quote"] = "fabricated"
    if change == "outside_passage":
        c["retrieved_passages"][0]["end"] = 1
    if change == "missing_quote":
        r["claims"][0]["reference_evidence"] = []
    if change == "omitted_text":
        r["claims"][0]["start"] = 2
    if change == "bad_correctness":
        r["correctness"] = True
    if change == "unknown_doc":
        r["reviewed_document_ids"] = ["not-in-corpus"]
    if change == "nonrag_scope":
        c["retrieval_applicable"] = False
        r["retrieved_sufficiency"] = None
    if change == "fake_citation":
        r["claims"][0]["cited_document_ids"] = ["invented"]
    with pytest.raises(LabError, match=expected):
        validate_rating(r, c, docs)


def test_unknowns_not_unsupported_and_empty_claims_not_perfect(quality_packet):
    packet, _, _, docs, _ = quality_packet
    review, cases = completed_review(packet, docs)
    r = next(
        r
        for r in review["items"]
        if cases[r["item_id"]]["answerable"] and not cases[r["item_id"]]["retrieval_applicable"]
    )
    r["claims"][0]["reference_support"] = "unverifiable"
    r["claims"][0]["reference_evidence"] = []
    r["reviewed_document_ids"] = []
    r["reference_sufficiency"] = "uncertain"
    validate_rating(r, cases[r["item_id"]], docs)
    score = rate_summary(r, cases[r["item_id"]])
    assert score["reference_unsupported_claim_rate"] is None
    assert score["reference_unverifiable_claims"] == 1
    r["claims"][0]["reference_support"] = "not_supported"
    with pytest.raises(LabError, match="No reviewed evidence"):
        validate_rating(r, cases[r["item_id"]], docs)
    abstain = next(r for r in review["items"] if r["behavior"] == "abstain")
    assert rate_summary(abstain, cases[abstain["item_id"]])["reference_support_rate"] is None


def test_agreement_edge_cases():
    assert agreement([], [])["linear_weighted_kappa"] is None
    assert agreement([3, 3], [3, 3])["linear_weighted_kappa"] is None
    assert agreement([0, 1, 3], [0, 1, 3])["linear_weighted_kappa"] == 1
    assert agreement([0, 3], [3, 0])["linear_weighted_kappa"] == -1
    assert agreement([0, 1], [1, 1])["mean_absolute_error"] == 0.5


def test_scoring_does_not_fabricate_human_validation(quality_packet, tmp_path):
    packet, manifests, processed, docs, _ = quality_packet
    out = tmp_path / "pending.json"
    result = score_reviews(
        packet=packet, reviews=[], manifests=manifests, processed=processed, output=out
    )
    assert result["independent_human_reviewers"] == 0
    assert result["human_review_coverage"] == "incomplete"
    review, _ = completed_review(packet, docs)
    path = tmp_path / "review.json"
    atomic_write(path, canonical(review))
    result = score_reviews(
        packet=packet,
        reviews=[path],
        manifests=manifests,
        processed=processed,
        output=tmp_path / "scored.json",
    )
    assert len(result["per_rating_scores"]) == 10
    assert result["human_review_coverage"] == "incomplete"
    assert result["measurement_validity"].startswith("not_yet_established")
    with pytest.raises(LabError, match="unique"):
        score_reviews(
            packet=packet,
            reviews=[path, path],
            manifests=manifests,
            processed=processed,
            output=tmp_path / "duplicate.json",
        )


def test_llm_reviews_require_provenance_and_never_count_as_human(quality_packet, tmp_path):
    packet, manifests, processed, docs, _ = quality_packet
    review, _ = completed_review(packet, docs)
    review["kind"] = "llm_estimate"
    p = tmp_path / "judge.json"
    atomic_write(p, canonical(review))
    with pytest.raises(LabError, match="model revision"):
        load_review(p, packet, verify_packet(packet)[1], docs)
    review.update(model_revision="synthetic-frozen-judge-v1", judge_prompt_sha256="a" * 64)
    atomic_write(p, canonical(review))
    result = score_reviews(
        packet=packet,
        reviews=[p],
        manifests=manifests,
        processed=processed,
        output=tmp_path / "judge-scores.json",
    )
    assert result["independent_human_reviewers"] == 0


def test_semantic_overflow_and_reference_similarity_not_truth():
    encoder = SimpleNamespace(
        tokenizer=lambda s, **kw: {"input_ids": list(s)},
        encode=lambda texts: [[1.0, 0.0], [1.0, 0.0]],
    )
    assert similarity("long input", "x", encoder, 4)["cosine"] is None
    assert similarity("yes", "no", encoder, 10)["cosine"] == 1.0
    d = diagnostics("Do not restart the server.", "Do restart the server.", True, "abstain")
    assert d["token_f1_diagnostic"] > 0.8
    assert d["correctness_rubric_score"] is None


def test_html_escapes_answer_and_never_inserts_strategy_names(quality_packet, tmp_path):
    packet, _, _, _, _ = quality_packet
    output = tmp_path / "review.html"
    render_packet(packet, output)
    html = output.read_text()
    assert "lora-rag" not in html
    assert "Generated answer" in html
    with pytest.raises(LabError):
        render_packet(packet, output)


def test_failed_empty_and_truncated_are_distinct(quality_packet):
    packet, _, _, docs, _ = quality_packet
    review, cases = completed_review(packet, docs)
    original = review["items"][0]
    case = copy.deepcopy(cases[original["item_id"]])
    rating = copy.deepcopy(original)
    case.update(status="failed", answer="")
    rating.update(
        correctness=None,
        correctness_unavailable_reason="Generation failed",
        behavior="failed",
        reference_sufficiency=None,
        retrieved_sufficiency=None,
        claims=[],
    )
    validate_rating(rating, case, docs)
    score = rate_summary(rating, case)
    assert score["correctness"] is None
    assert score["annotation_scoped_appropriate_abstention"] is None
    rating["behavior"] = "abstain"
    with pytest.raises(LabError, match="Failed generation"):
        validate_rating(rating, case, docs)
    case["status"] = "ok"
    rating["behavior"] = "empty"
    validate_rating(rating, case, docs)


def test_disagreement_and_semantic_pairing_are_explicit(quality_packet, tmp_path):
    from model_adaptation_lab.evaluation.quality.review import compare_reviews, rank_correlation

    packet, manifests, processed, docs, _ = quality_packet
    a, cases = completed_review(packet, docs, "synthetic-A")
    b = copy.deepcopy(a)
    b["reviewer_id"] = "synthetic-B"
    item = next(r for r in b["items"] if r["correctness"] is not None)
    item["correctness"] = 1
    comparison = compare_reviews(a, b)
    assert item["item_id"] in comparison["correctness_disagreements"]
    assert comparison["correctness"]["mean_absolute_error"] == pytest.approx(2 / 5)
    assert rank_correlation([(0.0, 0.0), (0.5, 1.0), (1.0, 3.0)]) == pytest.approx(1.0)
    assert rank_correlation([(1.0, 0.0), (1.0, 1.0), (1.0, 3.0)]) is None
    paths = []
    for review in (a, b):
        path = tmp_path / (review["reviewer_id"] + ".json")
        atomic_write(path, canonical(review))
        paths.append(path)
    semantic = tmp_path / "semantic.json"
    atomic_write(
        semantic,
        canonical(
            {
                "packet_sha256": file_hash(packet / "manifest.json"),
                "rows": [
                    {"item_id": i, "cosine": 0.8 if c["answerable"] else None}
                    for i, c in cases.items()
                ],
            }
        ),
    )
    result = score_reviews(
        packet=packet,
        reviews=paths,
        manifests=manifests,
        processed=processed,
        output=tmp_path / "agreement.json",
        semantic_path=semantic,
    )
    assert result["independent_human_reviewers"] == 2
    assert (
        result["human_review_coverage"] == "incomplete"
    )  # Only two synthetic questions, not twelve.
    assert len(result["summaries"]) == 30
    assert result["semantic_validation"][0]["n"] == 5
    assert result["semantic_validation"][0]["spearman"] is None
    assert result["scorer_code_snapshot_sha256"]


def test_source_hash_and_scope_guard(quality_packet, tmp_path):
    packet, manifests, processed, _, runs = quality_packet
    path = runs[0] / "run.json"
    state = json.loads(path.read_text())
    state["scope"] = "final_test"
    atomic_write(path, canonical(state))
    with pytest.raises(LabError, match="validation run"):
        prepare_packet(
            runs=runs,
            manifests=manifests,
            processed=processed,
            output=tmp_path / "bad",
            config_path=Path("configs/quality-v1.yaml"),
            rubric_path=Path("reports/quality-rubric-v1.md"),
        )
    state["scope"] = "validation_only"
    atomic_write(path, canonical(state))
    (runs[0] / "base.jsonl").write_text("{}\n")
    with pytest.raises(LabError, match="hash mismatch"):
        prepare_packet(
            runs=runs,
            manifests=manifests,
            processed=processed,
            output=tmp_path / "bad",
            config_path=Path("configs/quality-v1.yaml"),
            rubric_path=Path("reports/quality-rubric-v1.md"),
        )


def test_tracking_incomplete_review_remains_explicit(quality_packet, tmp_path, monkeypatch):
    from model_adaptation_lab.evaluation.quality import tracking

    packet, manifests, processed, _, _ = quality_packet
    score_path = tmp_path / "scores.json"
    score_reviews(
        packet=packet, reviews=[], manifests=manifests, processed=processed, output=score_path
    )
    calls = []

    class Run:
        info = SimpleNamespace(run_id="synthetic-mlflow-run")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    fake = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: None,
        start_run=lambda **kwargs: Run(),
        log_params=lambda params: calls.append(params),
        set_tag=lambda key, value: calls.append((key, value)),
        log_metrics=lambda metrics: calls.append(metrics),
        log_artifact=lambda path, **kw: calls.append(Path(path)),
        log_artifacts=lambda path, **kw: calls.append(Path(path)),
    )
    monkeypatch.setattr(tracking, "importlib", SimpleNamespace(import_module=lambda name: fake))
    result = tracking.track_quality(
        packet=packet, scores=score_path, output=tmp_path / "receipt.json"
    )
    assert result["mlflow_run_id"] == "synthetic-mlflow-run"
    assert result["measurement_validity"].startswith("not_yet_established")
    assert any(
        isinstance(c, dict) and c.get("human_review_coverage") == "incomplete" for c in calls
    )
    assert any(isinstance(c, Path) and c.name == "code_snapshot" for c in calls)
    with pytest.raises(LabError, match="local SQLite"):
        tracking.track_quality(
            packet=packet,
            scores=score_path,
            output=tmp_path / "remote.json",
            tracking_uri="https://example.invalid",
        )


def test_reviewer_bundle_excludes_analysis_and_resets_edited_template(quality_packet, tmp_path):
    from model_adaptation_lab.evaluation.quality.handoff import BUNDLE_FILES, export_bundle

    packet, _, _, docs, _ = quality_packet
    edited, _ = completed_review(packet, docs)
    atomic_write(packet / "review_template.json", canonical(edited))
    output = tmp_path / "reviewer"
    export_bundle(packet, output)
    assert {p.name for p in output.iterdir()} == BUNDLE_FILES | {"bundle.json"}
    template = json.loads((output / "review_template.json").read_text())
    assert template["reviewer_id"] == ""
    assert template["independent"] is False
    assert template["packet_sha256"] == file_hash(packet / "manifest.json")
    assert all(item["behavior"] is None for item in template["items"])
    with pytest.raises(LabError, match="already exists"):
        export_bundle(packet, output)


def test_quote_lookup_preserves_unicode_offsets_and_scope(quality_packet, tmp_path):
    from model_adaptation_lab.evaluation.quality.handoff import export_bundle, quote_matches

    packet, *_ = quality_packet
    # Synthetic mutation with recomputed packet hash to exercise repeated Unicode text.
    manifest, cases = verify_packet(packet)
    case = cases[0]
    case["reference_documents"] = [
        {"document_id": "doc", "title": "Synthetic", "text": "éx aba aba"}
    ]
    case["retrieved_passages"] = [{"document_id": "doc", "start": 7, "end": 10, "text": "aba"}]
    atomic_write(packet / "cases.jsonl", jsonl(cases))
    manifest["files"]["cases.jsonl"] = file_hash(packet / "cases.jsonl")
    atomic_write(packet / "manifest.json", canonical(manifest))
    output = tmp_path / "reviewer"
    export_bundle(packet, output)
    for source in (packet, output):
        matches = quote_matches(source, case["item_id"], "doc", "aba", "reference")
        assert [m["start"] for m in matches] == [3, 7]
        assert quote_matches(source, case["item_id"], "doc", "aba", "retrieved") == matches[1:]
        assert quote_matches(source, case["item_id"], "doc", "absent", "reference") == []
        with pytest.raises(LabError, match="unavailable"):
            quote_matches(source, case["item_id"], "missing", "aba", "reference")
        with pytest.raises(LabError, match="nonempty"):
            quote_matches(source, case["item_id"], "doc", "", "reference")
    atomic_write(output / "cases.jsonl", b"{}\n")
    with pytest.raises(LabError, match="changed"):
        quote_matches(output, case["item_id"], "doc", "aba", "reference")
