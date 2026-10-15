"""Strict review import, scope-aware scores and explicit unknowns."""

import re
from pathlib import Path
from typing import Any

from model_adaptation_lab.evaluation.quality.packet import citation_ids, verify_packet
from model_adaptation_lab.evaluation.runner import capture_environment, snapshot_code
from model_adaptation_lab.io import LabError, atomic_write, canonical, file_hash, read_json
from model_adaptation_lab.retrieval.chunking import load_corpus

SUPPORT = {"supported", "contradicted", "not_supported", "unverifiable"}
SUFFICIENCY = {"sufficient", "insufficient", "uncertain"}
BEHAVIOR = {"answer", "abstain", "mixed", "empty", "failed"}


def check_evidence(
    evidence: Any, case: dict[str, Any], docs: dict[str, Any], reviewed: set[str], retrieved: bool
) -> None:
    if not isinstance(evidence, list):
        raise LabError("Evidence must be a list of source quotes")
    for e in evidence:
        if not isinstance(e, dict) or set(e) != {"document_id", "start", "end", "quote"}:
            raise LabError("Evidence requires document_id, start, end and exact quote")
        doc = docs.get(e["document_id"])
        if (
            doc is None
            or type(e["start"]) is not int
            or type(e["end"]) is not int
            or not 0 <= e["start"] < e["end"] <= len(doc["text"])
            or not isinstance(e["quote"], str)
            or not e["quote"].strip()
            or doc["text"][e["start"] : e["end"]] != e["quote"]
        ):
            raise LabError("Evidence quote does not match the frozen source")
        if retrieved:
            if not any(
                c["document_id"] == e["document_id"]
                and c["start"] <= e["start"]
                and c["end"] >= e["end"]
                for c in case["retrieved_passages"]
            ):
                raise LabError("Quote is outside the passages actually supplied to the model")
        elif e["document_id"] not in reviewed:
            raise LabError("Reference quote must belong to an explicitly reviewed document")


def validate_rating(r: dict[str, Any], case: dict[str, Any], docs: dict[str, Any]) -> None:
    required = {
        "item_id",
        "correctness",
        "correctness_unavailable_reason",
        "behavior",
        "reference_sufficiency",
        "retrieved_sufficiency",
        "reviewed_document_ids",
        "rationale",
        "claims",
    }
    if not isinstance(r, dict) or not required <= set(r):
        raise LabError("Rating is missing required review-template fields")
    if (
        r.get("behavior") not in BEHAVIOR
        or not isinstance(r.get("rationale"), str)
        or not r["rationale"].strip()
    ):
        raise LabError("Each rating needs a behavior and substantive rationale")
    failed = case["status"] == "failed"
    empty = not case["answer"].strip()
    if failed and r["behavior"] != "failed":
        raise LabError("Failed generation cannot be relabeled as an answer/abstention")
    if not failed and (
        (empty and r["behavior"] != "empty") or (not empty and r["behavior"] in {"empty", "failed"})
    ):
        raise LabError("Behavior is inconsistent with output availability")
    correctness = r.get("correctness")
    if correctness is not None and (type(correctness) is not int or correctness not in range(4)):
        raise LabError("Correctness must be null or integer 0..3")
    if (failed or empty or not case["answerable"]) and correctness is not None:
        raise LabError("Correctness unavailable for failed/empty/unanswerable-reference cases")
    if correctness is None and not str(r.get("correctness_unavailable_reason", "")).strip():
        raise LabError("Null correctness requires an explicit reason")
    if r["behavior"] == "abstain" and case["answerable"] and correctness not in {0, None}:
        raise LabError("Abstention on an answerable reference cannot receive positive correctness")
    reviewed_list = r.get("reviewed_document_ids")
    if not isinstance(reviewed_list, list) or not all(isinstance(x, str) for x in reviewed_list):
        raise LabError("List the frozen documents actually reviewed")
    reviewed = set(reviewed_list)
    if len(reviewed) != len(reviewed_list) or not reviewed <= set(docs):
        raise LabError("Duplicate or unknown reviewed documents")
    for scope in ("reference", "retrieved"):
        applicable = not (failed or empty) and (
            scope == "reference" or case["retrieval_applicable"]
        )
        value = r.get(scope + "_sufficiency")
        if (applicable and value not in SUFFICIENCY) or (not applicable and value is not None):
            raise LabError("Evidence sufficiency must respect availability and strategy scope")
        available = reviewed if scope == "reference" else case["retrieved_passages"]
        if applicable and not available and value != "uncertain":
            raise LabError("Absent evidence requires uncertain sufficiency")
    claims = r.get("claims")
    if not isinstance(claims, list):
        raise LabError("Claims must be a list")
    cursor = 0
    for claim in claims:
        required_claim = {
            "start",
            "end",
            "text",
            "kind",
            "reference_support",
            "retrieved_support",
            "reference_evidence",
            "retrieved_evidence",
            "cited_document_ids",
            "citation_support",
            "rationale",
        }
        if not isinstance(claim, dict) or not required_claim <= set(claim):
            raise LabError("Claim is missing required review-template fields")
        start, end = claim.get("start"), claim.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or not cursor <= start < end <= len(case["answer"])
            or case["answer"][cursor:start].strip()
            or claim.get("text") != case["answer"][start:end]
        ):
            raise LabError("Claims must cover exact ordered answer spans without omitted text")
        cursor = end
        if claim.get("kind") not in {"factual", "nonfactual"}:
            raise LabError("Every span needs a factual/nonfactual label")
        if not isinstance(claim.get("rationale"), str) or not claim["rationale"].strip():
            raise LabError("Every claim needs a rationale")
        factual = claim["kind"] == "factual"
        for scope in ("reference", "retrieved"):
            applicable = factual and (scope == "reference" or case["retrieval_applicable"])
            status = claim.get(scope + "_support")
            evidence = claim.get(scope + "_evidence")
            if (applicable and status not in SUPPORT) or (not applicable and status is not None):
                raise LabError("Claim support label violates evidence scope")
            check_evidence(evidence, case, docs, reviewed, scope == "retrieved")
            if not applicable and evidence:
                raise LabError("Nonapplicable evidence scope must be empty")
            if status in {"supported", "contradicted"} and not evidence:
                raise LabError("Supported/contradicted claims require source quotations")
            available = reviewed if scope == "reference" else case["retrieved_passages"]
            if applicable and not available and status != "unverifiable":
                raise LabError("No reviewed evidence: use unverifiable, not unsupported")
        cited = claim.get("cited_document_ids")
        if not isinstance(cited, list) or not all(isinstance(v, str) for v in cited):
            raise LabError("Claim citation links must be document IDs")
        if len(set(cited)) != len(cited) or not set(cited) <= set(citation_ids(case["answer"])):
            raise LabError("Claim links to a citation not present in the answer")
        label = claim.get("citation_support")
        if not factual or not case["retrieval_applicable"]:
            if label is not None or cited:
                raise LabError("Citation requirements apply only to factual RAG claims")
        else:
            if label not in SUPPORT | {"missing"}:
                raise LabError("Factual RAG claims need a citation assessment")
            if (not cited) != (label == "missing"):
                raise LabError("Missing citation status and linked IDs disagree")
            if label in {"supported", "contradicted"}:
                quotes = claim["retrieved_evidence"]
                if not quotes or not set(cited) <= {e["document_id"] for e in quotes}:
                    raise LabError("Citation support/contradiction needs quoted cited passages")
    if case["answer"][cursor:].strip():
        raise LabError("Claims omit trailing answer text")
    if failed and claims:
        raise LabError("Failed output cannot have claim judgments")
    if r["behavior"] == "abstain" and any(c["kind"] == "factual" for c in claims):
        raise LabError("Abstention plus factual advice must be labeled mixed")


def rate_summary(r: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    factual = [c for c in r["claims"] if c["kind"] == "factual"]
    present = case["status"] == "ok" and bool(case["answer"].strip())
    abstain = r["behavior"] == "abstain"
    result: dict[str, Any] = {
        "correctness": r["correctness"],
        "factual_claims": len(factual),
        "annotation_scoped_appropriate_abstention": float(abstain)
        if present and not case["answerable"]
        else None,
        "annotation_scoped_unnecessary_abstention": float(abstain)
        if present and case["answerable"]
        else None,
        "mixed_response": float(r["behavior"] == "mixed") if present else None,
        "label_scope_disagreement": bool(
            not case["answerable"] and r["reference_sufficiency"] == "sufficient"
        ),
    }
    for scope in ("reference", "retrieved"):
        labels = [c[scope + "_support"] for c in factual if c[scope + "_support"] is not None]
        assessed = [v for v in labels if v != "unverifiable"]
        result[scope + "_support_rate"] = (
            assessed.count("supported") / len(assessed) if assessed else None
        )
        result[scope + "_unsupported_claim_rate"] = (
            sum(v in {"contradicted", "not_supported"} for v in assessed) / len(assessed)
            if assessed
            else None
        )
        result[scope + "_assessed_claims"] = len(assessed)
        result[scope + "_unverifiable_claims"] = labels.count("unverifiable")
        sufficiency = r[scope + "_sufficiency"]
        result[scope + "_appropriate_abstention"] = (
            float(abstain) if present and sufficiency == "insufficient" else None
        )
        result[scope + "_unnecessary_abstention"] = (
            float(abstain) if present and sufficiency == "sufficient" else None
        )
    labels = [c["citation_support"] for c in factual if c["citation_support"] is not None]
    assessed = [v for v in labels if v in {"supported", "contradicted", "not_supported"}]
    result["citation_coverage"] = (
        sum(v != "missing" for v in labels) / len(labels) if labels else None
    )
    result["citation_support_rate"] = (
        assessed.count("supported") / len(assessed) if assessed else None
    )
    result["citation_assessed_claims"] = len(assessed)
    result["citation_unverifiable_claims"] = labels.count("unverifiable")
    return result


def load_review(
    path: Path, packet: Path, cases: list[dict[str, Any]], docs: dict[str, Any]
) -> dict[str, Any]:
    review = read_json(path)
    manifest = read_json(packet / "manifest.json")
    if (
        review.get("schema_version") != 1
        or review.get("packet_sha256") != file_hash(packet / "manifest.json")
        or review.get("rubric_sha256") != manifest["files"]["rubric.md"]
    ):
        raise LabError("Review packet/rubric identity mismatch")
    if not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip():
        raise LabError("Reviewer ID is required")
    if (
        review.get("kind") not in {"human", "llm_estimate"}
        or type(review.get("independent")) is not bool
    ):
        raise LabError("Declare human/LLM origin and independence explicitly")
    if review["kind"] == "llm_estimate" and (
        not str(review.get("model_revision") or "").strip()
        or not re.fullmatch(r"[0-9a-f]{64}", str(review.get("judge_prompt_sha256")))
    ):
        raise LabError("LLM estimates require model revision and judge prompt SHA-256")
    if review["kind"] == "human" and (
        review.get("model_revision") or review.get("judge_prompt_sha256")
    ):
        raise LabError("Model judgments cannot be imported as human ratings")
    by_id = {c["item_id"]: c for c in cases}
    if not isinstance(review.get("items"), list):
        raise LabError("Review items must be a list")
    seen = set()
    for rating in review["items"]:
        if not isinstance(rating, dict):
            raise LabError("Review items must be objects")
        item = rating.get("item_id")
        if item not in by_id or item in seen:
            raise LabError("Unknown or duplicate review item")
        seen.add(item)
        validate_rating(rating, by_id[item], docs)
    return dict(review)


def agreement(first: list[float], second: list[float]) -> dict[str, Any]:
    n = len(first)
    if n != len(second):
        raise LabError("Agreement pairs differ in length")
    if not n:
        return {
            "n": 0,
            "exact_agreement": None,
            "mean_absolute_error": None,
            "linear_weighted_kappa": None,
        }
    observed = sum(abs(a - b) for a, b in zip(first, second, strict=True)) / n
    expected = sum(abs(a - b) for a in first for b in second) / (n * n)
    return {
        "n": n,
        "exact_agreement": sum(a == b for a, b in zip(first, second, strict=True)) / n,
        "mean_absolute_error": observed,
        "linear_weighted_kappa": 1 - observed / expected if expected else None,
    }


def compare_reviews(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    left, right = ({r["item_id"]: r for r in v["items"]} for v in (a, b))
    common = sorted(set(left) & set(right))
    pairs = [
        (left[i]["correctness"], right[i]["correctness"])
        for i in common
        if left[i]["correctness"] is not None and right[i]["correctness"] is not None
    ]
    result: dict[str, Any] = {
        "reviewers": [a["reviewer_id"], b["reviewer_id"]],
        "kinds": [a["kind"], b["kind"]],
        "shared_items": len(common),
        "correctness": agreement([p[0] for p in pairs], [p[1] for p in pairs]),
        "correctness_disagreements": [
            i for i in common if left[i]["correctness"] != right[i]["correctness"]
        ],
    }
    for key in ("behavior", "reference_sufficiency", "retrieved_sufficiency"):
        values = [
            (left[i][key], right[i][key])
            for i in common
            if left[i][key] is not None and right[i][key] is not None
        ]
        result[key] = {
            "n": len(values),
            "agreement": sum(x == y for x, y in values) / len(values) if values else None,
        }
    aligned, unequal = 0, 0
    dimensions = ("kind", "reference_support", "retrieved_support", "citation_support")
    labels: dict[str, list[tuple[Any, Any]]] = {key: [] for key in dimensions}
    disagreement_items = set()
    for item in common:
        left_claims, right_claims = (
            {(c["start"], c["end"]): c for c in side[item]["claims"]} for side in (left, right)
        )
        unpaired = len(set(left_claims) ^ set(right_claims))
        unequal += unpaired
        if unpaired:
            disagreement_items.add(item)
        for span in set(left_claims) & set(right_claims):
            aligned += 1
            for key in dimensions:
                pair = (left_claims[span][key], right_claims[span][key])
                labels[key].append(pair)
                if pair[0] != pair[1]:
                    disagreement_items.add(item)
    result["claim_segmentation"] = {"aligned_spans": aligned, "nonaligned_spans": unequal}
    result["claim_disagreement_items"] = sorted(disagreement_items)
    for key, values in labels.items():
        assessed = [(x, y) for x, y in values if x is not None and y is not None]
        result[key + "_claim_agreement"] = {
            "n": len(assessed),
            "unpaired_applicability": sum((x is None) != (y is None) for x, y in values),
            "agreement": sum(x == y for x, y in assessed) / len(assessed) if assessed else None,
        }
    return result


def rank_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        return [
            sum(v < value for v in values) + (sum(v == value for v in values) + 1) / 2
            for value in values
        ]

    x, y = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(x) / len(x), sum(y) / len(y)
    denominator = (sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y)) ** 0.5
    return (
        sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / denominator
        if denominator
        else None
    )


def summarize_ratings(
    scored: list[dict[str, Any]], cases: dict[str, Any], mapping: dict[str, Any]
) -> list[dict[str, Any]]:
    summaries = []
    for reviewer in sorted({r["reviewer_id"] for r in scored}):
        for strategy in sorted({v["strategy"] for v in mapping.values()}):
            for stratum in ("all", "answerable", "unanswerable"):
                items = {
                    i
                    for i, c in cases.items()
                    if mapping[i]["strategy"] == strategy
                    and (stratum == "all" or c["answerable"] == (stratum == "answerable"))
                }
                ratings = [
                    r for r in scored if r["reviewer_id"] == reviewer and r["item_id"] in items
                ]
                metrics = {}
                keys = {
                    k
                    for r in ratings
                    for k in r["scores"]
                    if k == "correctness"
                    or k.endswith("_rate")
                    or "abstention" in k
                    or k in {"citation_coverage", "mixed_response"}
                }
                for key in sorted(keys):
                    values = [r["scores"][key] for r in ratings if r["scores"][key] is not None]
                    metrics[key] = {
                        "mean": sum(values) / len(values) if values else None,
                        "denominator": len(values),
                        "unavailable_ratings": len(ratings) - len(values),
                    }
                summaries.append(
                    {
                        "reviewer_id": reviewer,
                        "strategy": strategy,
                        "stratum": stratum,
                        "available_outputs": len(items),
                        "rated_outputs": len(ratings),
                        "aggregation": "per-answer macro means; claim counts remain per-rating",
                        "metrics": metrics,
                    }
                )
    return summaries


def score_reviews(
    *,
    packet: Path,
    reviews: list[Path],
    manifests: Path,
    processed: Path,
    output: Path,
    semantic_path: Path | None = None,
) -> dict[str, Any]:
    manifest, cases = verify_packet(packet)
    if manifest["freeze_sha256"] != file_hash(manifests / "freeze.json"):
        raise LabError("Packet protocol differs from active frozen manifests")
    documents, corpus = load_corpus(manifests, processed)
    if corpus != manifest["corpus"]:
        raise LabError("Packet corpus mismatch")
    docs = {d["document_id"]: d for d in documents}
    loaded = [load_review(p, packet, cases, docs) for p in reviews]
    if len({r["reviewer_id"] for r in loaded}) != len(loaded):
        raise LabError("Reviewer IDs must be unique; preserve revisions as separate score runs")
    by_id = {c["item_id"]: c for c in cases}
    mapping = read_json(packet / "mapping.json")
    scored = [
        {
            "reviewer_id": review["reviewer_id"],
            "kind": review["kind"],
            "item_id": r["item_id"],
            **mapping[r["item_id"]],
            "scores": rate_summary(r, by_id[r["item_id"]]),
        }
        for review in loaded
        for r in review["items"]
    ]
    human = [r for r in loaded if r["kind"] == "human" and r["independent"]]
    required = read_json(packet / "config.json")
    counts = {
        i: sum(any(r["item_id"] == i for r in review["items"]) for review in human) for i in by_id
    }
    covered = {i for i, n in counts.items() if n >= required["minimum_independent_human_reviewers"]}
    eligible = {c["item_id"] for c in cases if c["status"] == "ok" and c["answer"].strip()}
    strata = {
        str(flag): len(
            {
                c["question_id"]
                for c in cases
                if c["answerable"] == flag
                and c["item_id"] in eligible
                and all(
                    x["item_id"] in covered
                    for x in cases
                    if x["question_id"] == c["question_id"] and x["item_id"] in eligible
                )
            }
        )
        for flag in (True, False)
    }
    enough = (
        bool(eligible)
        and eligible <= covered
        and all(
            n >= required["minimum_reviewed_questions_per_answerability"] for n in strata.values()
        )
    )
    result = {
        "schema_version": 1,
        "scope": "validation_only",
        "packet_sha256": file_hash(packet / "manifest.json"),
        "review_files": {str(p): file_hash(p) for p in reviews},
        "human_review_coverage": "met_pending_adjudication" if enough else "incomplete",
        "measurement_validity": (
            "not_yet_established; requires independent adjudication and documented validation"
        ),
        "automated_metric_validity": "not_validated; similarity and syntax remain diagnostics",
        "successful_outputs": len(eligible),
        "failed_outputs": sum(c["status"] == "failed" for c in cases),
        "empty_outputs": sum(c["status"] == "ok" and not c["answer"].strip() for c in cases),
        "truncated_outputs": sum(c["finish_reason"] == "length" for c in cases),
        "independent_human_reviewers": len(human),
        "jointly_reviewed_items": len(covered),
        "jointly_reviewed_questions_by_answerability": strata,
        "per_rating_scores": scored,
        "agreement": [
            compare_reviews(a, b) for index, a in enumerate(loaded) for b in loaded[index + 1 :]
        ],
    }
    result["summaries"] = summarize_ratings(scored, by_id, mapping)
    result["semantic_validation"] = []
    if semantic_path is not None:
        semantic = read_json(semantic_path)
        if semantic.get("packet_sha256") != file_hash(packet / "manifest.json"):
            raise LabError("Semantic diagnostics belong to another packet")
        rows = semantic.get("rows", [])
        if len(rows) != len(cases) or {r["item_id"] for r in rows} != set(by_id):
            raise LabError("Semantic diagnostic coverage differs from packet")
        values = {r["item_id"]: r["cosine"] for r in rows}
        if any(
            v is not None and (type(v) not in {int, float} or not -1 <= v <= 1)
            for v in values.values()
        ):
            raise LabError("Invalid cosine diagnostic")
        result["semantic_sha256"] = file_hash(semantic_path)
        result["semantic_available"] = sum(v is not None for v in values.values())
        for review in human:
            pairs = [
                (float(values[r["item_id"]]), float(r["correctness"]))
                for r in review["items"]
                if r["correctness"] is not None and values[r["item_id"]] is not None
            ]
            result["semantic_validation"].append(
                {
                    "reviewer_id": review["reviewer_id"],
                    "n": len(pairs),
                    "spearman": rank_correlation(pairs),
                    "interpretation": (
                        "Exploratory association only; no validated correctness threshold"
                    ),
                }
            )
    if output.exists():
        raise LabError("Score output exists; preserve previous assessments")
    reproduction = output.parent / (output.stem + "_reproduction")
    try:
        reproduction.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise LabError("Scorer reproduction directory exists; use a new output") from exc
    result["scorer_code_snapshot_sha256"] = snapshot_code(reproduction)
    result["environment"] = capture_environment()
    result["scorer_reproduction_directory"] = str(reproduction)
    atomic_write(output, canonical(result))
    return result
