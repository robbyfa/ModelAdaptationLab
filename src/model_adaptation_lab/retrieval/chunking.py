"""Deterministic body chunks with character offsets and stable content-derived IDs."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_adaptation_lab.data.splits import verify
from model_adaptation_lab.io import LabError, canonical, digest, file_hash, read_json


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    start: int
    end: int
    source_url: str | None
    source: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def load_corpus(manifests: Path, processed: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock = verify(manifests)
    if lock.get("protocol_id") != "techqa-internal-v1":
        raise LabError("Retrieval requires the frozen eligible protocol")
    provenance = read_json(manifests / "corpus.json")
    path = processed / "source_documents.jsonl"
    if file_hash(path) != provenance["processed_corpus_sha256"]:
        raise LabError(
            "Corpus differs from frozen source-document bytes; annotations are not allowed"
        )
    documents = [json.loads(line) for line in path.read_text().splitlines()]
    keys = {"document_id", "title", "text", "source_url", "source"}
    ids = []
    for row in documents:
        if set(row) != keys or row["source"] != "original_technotes":
            raise LabError(
                "Corpus must contain original source documents only, without annotations"
            )
        for key in ("document_id", "title", "text"):
            if not isinstance(row[key], str):
                raise LabError(f"Invalid document {key}")
        if not row["document_id"] or not row["text"].strip():
            raise LabError("Empty document ID or body")
        if row["source_url"] is not None and not isinstance(row["source_url"], str):
            raise LabError("Invalid document source URL")
        ids.append(row["document_id"])
    if len(ids) != len(set(ids)) or len(ids) != provenance["documents"]:
        raise LabError("Corpus ID uniqueness/count mismatch")
    return sorted(documents, key=lambda row: row["document_id"]), dict(provenance)


def chunk_document(doc: dict[str, Any], tokenizer: Any, config: dict[str, Any]) -> list[Chunk]:
    size, overlap = config["tokens"], config["overlap"]
    if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
        raise LabError("Chunk overlap must be nonnegative and smaller than token count")
    offsets = tokenizer(
        doc["text"],
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )["offset_mapping"]
    chunks = []
    for position in range(0, len(offsets), size - overlap):
        window = offsets[position : position + size]
        start, end = window[0][0], window[-1][1]
        text = doc["text"][start:end]
        identity = {
            "version": config,
            "document_id": doc["document_id"],
            "start": start,
            "end": end,
            "text": text,
        }
        chunks.append(
            Chunk(
                digest(canonical(identity)),
                doc["document_id"],
                doc["title"],
                text,
                start,
                end,
                doc["source_url"],
                doc["source"],
            )
        )
        if position + size >= len(offsets):
            break
    return chunks
