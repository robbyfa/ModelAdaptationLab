"""Phase 2 diagnostics, not a substitute for the final semantic/evidence rubric."""

import math
from collections import Counter
from typing import Any

from model_adaptation_lab.data.audit import normalize


def diagnostics(
    answer: str, reference: str | None, answerable: bool, abstention: str
) -> dict[str, Any]:
    abstained = answer.strip() == abstention
    tokens, gold = normalize(answer).split(), normalize(reference or "").split()
    common = sum((Counter(tokens) & Counter(gold)).values())
    f1 = 2 * common / (len(tokens) + len(gold)) if tokens and gold else 0.0
    return {
        "normalized_exact_match": float(normalize(answer) == normalize(reference or ""))
        if answerable and reference
        else None,
        "token_f1_diagnostic": f1 if answerable and reference else None,
        "exact_marker_abstention": abstained,
        "appropriate_abstention_annotation_scoped": abstained if not answerable else None,
        "unnecessary_abstention": abstained if answerable else None,
        "correctness_rubric_score": None,
        "reference_evidence_support": None,
        "retrieved_evidence_support": None,
        "unsupported_claim_rate": None,
        "unavailable_reason": (
            "Semantic correctness and evidence verification not implemented in Phase 2"
        ),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    good = [r for r in results if r["status"] == "ok"]
    summary: dict[str, Any] = {
        "attempted": len(results),
        "succeeded": len(good),
        "failed": len(results) - len(good),
        "latency_p50_ms": percentile([r["latency_ms"] for r in good], 0.5),
        "latency_p95_ms": percentile([r["latency_ms"] for r in good], 0.95),
        "input_tokens": sum(r["input_tokens"] for r in good),
        "output_tokens": sum(r["output_tokens"] for r in good),
        "training_gpu_hours": None,
        "inference_cost_per_1000": None,
        "quality_scope": "lexical/abstention diagnostics only; unavailable metrics are null",
    }
    for key in (
        "normalized_exact_match",
        "token_f1_diagnostic",
        "appropriate_abstention_annotation_scoped",
        "unnecessary_abstention",
    ):
        values = [r["metrics"][key] for r in good if r["metrics"][key] is not None]
        summary[key] = sum(values) / len(values) if values else None
        summary[key + "_denominator"] = len(values)
    return summary
