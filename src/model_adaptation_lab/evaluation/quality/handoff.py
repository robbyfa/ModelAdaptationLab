"""Reviewer-only exports and literal quote lookup; neither produces quality judgments."""

from pathlib import Path
from typing import Any

from model_adaptation_lab.evaluation.quality.packet import (
    read_rows,
    review_template,
    verify_packet,
)
from model_adaptation_lab.evaluation.quality.render import render_packet
from model_adaptation_lab.io import LabError, atomic_write, canonical, file_hash, read_json

GUIDE = """# Independent development review

Read rubric.md and review.html. Copy review_template.json to ratings.json before editing.
Set your reviewer ID and explicitly declare whether you worked independently. The initial
independence value is false; change it only if true. No ratings have been supplied for you.
Review all cases without seeing strategy labels, similarity scores or another person's ratings.
Evidence can reveal retrieval, so blinding is partial. Save revisions as new files.

Offsets are Python Unicode character offsets in the original source body, with end exclusive.
Use `model-lab quality quote --packet BUNDLE --item-id ITEM --document-id DOC
--quote-file quote.txt --scope reference` to find every exact occurrence. Use scope retrieved
to restrict matches to passages actually supplied. The UTF-8 quote file is read verbatim,
including newlines. Check surrounding context and choose the intended occurrence; matching
text establishes provenance, never entailment. Split compound factual claims in the form.

Return only your completed ratings file to the coordinator. This bundle sends nothing.
The coordinator imports ratings against the original packet, not this reviewer bundle.
Missing evidence, uncertain judgments and reference defects must remain explicit.
Do not share the underlying documentation beyond its applicable usage terms.
"""

BUNDLE_FILES = {
    "cases.jsonl",
    "rubric.md",
    "config.json",
    "review.html",
    "review_template.json",
    "README.md",
}


def export_bundle(packet: Path, output: Path) -> None:
    manifest, cases = verify_packet(packet)
    if output.exists():
        raise LabError("Reviewer bundle already exists; choose a new directory")
    output.mkdir(parents=True)
    for name in ("cases.jsonl", "rubric.md", "config.json"):
        atomic_write(output / name, (packet / name).read_bytes())
    template = review_template(packet, manifest, cases)
    template["independent"] = False
    atomic_write(output / "review_template.json", canonical(template))
    atomic_write(output / "README.md", GUIDE.encode())
    render_packet(packet, output / "review.html")
    atomic_write(
        output / "bundle.json",
        canonical(
            {
                "schema_version": 1,
                "scope": "validation_only",
                "packet_sha256": file_hash(packet / "manifest.json"),
                "files": {name: file_hash(output / name) for name in sorted(BUNDLE_FILES)},
            }
        ),
    )


def quote_matches(
    packet: Path, item_id: str, document_id: str, quote: str, scope: str
) -> list[dict[str, Any]]:
    if not quote.strip() or scope not in {"reference", "retrieved"}:
        raise LabError("A nonempty literal quote and reference/retrieved scope are required")
    if (packet / "bundle.json").exists():
        bundle = read_json(packet / "bundle.json")
        if (
            bundle.get("schema_version") != 1
            or bundle.get("scope") != "validation_only"
            or set(bundle.get("files", {})) != BUNDLE_FILES
        ):
            raise LabError("Invalid reviewer bundle")
        for name, expected in bundle["files"].items():
            if file_hash(packet / name) != expected:
                raise LabError(f"Reviewer bundle changed: {name}")
        cases = read_rows(packet / "cases.jsonl")
    else:
        _, cases = verify_packet(packet)
    selected = [case for case in cases if case["item_id"] == item_id]
    if len(selected) != 1:
        raise LabError("Unknown or duplicate review item ID")
    key = "reference_documents" if scope == "reference" else "retrieved_passages"
    docs = [doc for doc in selected[0][key] if doc["document_id"] == document_id]
    if not docs:
        raise LabError("Document is unavailable in the selected evidence scope")
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for doc in docs:
        offset = doc["start"] if scope == "retrieved" else 0
        position = doc["text"].find(quote)
        while position >= 0:
            start, end = offset + position, offset + position + len(quote)
            found[start, end] = {
                "document_id": document_id,
                "start": start,
                "end": end,
                "quote": quote,
            }
            position = doc["text"].find(quote, position + 1)
    return [found[key] for key in sorted(found)]
