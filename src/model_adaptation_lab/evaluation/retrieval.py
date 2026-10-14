"""Annotation-limited retrieval metrics, computed only after question-only retrieval."""

from typing import Any


def retrieval_metrics(
    chunks: tuple[dict[str, Any], ...],
    citations: tuple[str, ...],
    row: dict[str, Any],
) -> dict[str, Any]:
    gold = row["document_id"]
    annotated = bool(
        row["answerable"] and gold and row["start"] is not None and row["end"] is not None
    )
    spans = sorted((c["start"], c["end"]) for c in chunks if c["document_id"] == gold)
    cursor = row["start"] if annotated else 0
    if annotated:
        for start, end in spans:
            if start <= cursor:
                cursor = max(cursor, end)
    ids = {c["document_id"] for c in chunks}
    return {
        "returned_chunks": len(chunks),
        "returned_unique_documents": len(ids),
        "annotated_document_recall_at_k": float(gold in ids) if annotated else None,
        "annotated_span_recall_at_k": float(cursor >= row["end"]) if annotated else None,
        "annotated_gold_chunk_fraction_at_k": sum(c["document_id"] == gold for c in chunks)
        / len(chunks)
        if annotated and chunks
        else None,
        "precision_at_k": None,
        "relevance_score": None,
        "precision_unavailable_reason": "One gold document, no exhaustive relevance judgments; "
        "unjudged documents are not assumed irrelevant",
        "citation_ids_in_retrieved_documents_fraction": sum(c in ids for c in citations)
        / len(citations)
        if citations
        else None,
        "citation_scope": "ID membership only; not claim support or groundedness",
    }


def summarize_retrieval(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in results if r["status"] == "ok" and r.get("retrieval_metrics") is not None]
    summary: dict[str, Any] = {}
    for metric in ("annotated_document_recall_at_k", "annotated_span_recall_at_k"):
        values = [
            r["retrieval_metrics"][metric]
            for r in valid
            if r["retrieval_metrics"][metric] is not None
        ]
        summary[metric] = sum(values) / len(values) if values else None
        summary[metric + "_denominator"] = len(values)
    summary["retrieval_latency_ms_mean"] = (
        sum(r["retrieval_latency_ms"] for r in valid) / len(valid) if valid else None
    )
    summary["precision_at_k"] = None
    return summary
