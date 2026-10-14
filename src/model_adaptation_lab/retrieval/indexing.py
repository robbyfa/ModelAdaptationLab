"""Build and validate immutable FAISS artifacts from the frozen source corpus only."""

import importlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from model_adaptation_lab.io import LabError, atomic_write, canonical, digest, file_hash, read_json
from model_adaptation_lab.retrieval.chunking import chunk_document, load_corpus
from model_adaptation_lab.retrieval.embedding import Embedder, MiniLMEmbedder, validate_retrieval


def numpy_module() -> Any:
    try:
        importlib.metadata.version("faiss-cpu")
        return importlib.import_module("numpy")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise LabError(
            "Install retrieval dependencies: uv sync --locked --extra retrieval"
        ) from exc


def checked_vectors(vectors: Any, count: int, dimensions: int, np: Any) -> Any:
    values = np.ascontiguousarray(vectors, dtype="float32")
    if values.shape != (count, dimensions) or not np.isfinite(values).all():
        raise LabError("Embedding shape or finite-value check failed")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise LabError("Embedding vectors must be nonzero and L2-normalized")
    return values


def build_index(
    *,
    manifests: Path,
    processed: Path,
    config: dict[str, Any],
    output: Path,
    cache: Path,
    local_files_only: bool,
    max_documents: int | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    validate_retrieval(config)
    if max_documents is not None and max_documents <= 0:
        raise LabError("max_documents must be positive")
    documents, corpus = load_corpus(manifests, processed)
    if max_documents is not None:
        # Explicit smoke subsets are independent of QA records and gold document identifiers.
        documents = sorted(documents, key=lambda d: digest(d["document_id"].encode()))[
            :max_documents
        ]
        documents.sort(key=lambda d: d["document_id"])
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise LabError(
            "Index directory exists; use a new directory to preserve prior builds"
        ) from exc
    from model_adaptation_lab.evaluation.runner import capture_environment, snapshot_code

    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "environment": capture_environment(),
        "code_snapshot_sha256": snapshot_code(output),
        "status": "building",
        "config": config,
        "corpus": corpus,
        "protocol_freeze_sha256": file_hash(manifests / "freeze.json"),
        "corpus_scope": "full_frozen_subset" if max_documents is None else "explicit_smoke_subset",
        "documents_indexed": len(documents),
        "max_documents": max_documents,
        "backend_type": "injected_test" if embedder else "real_minilm",
        "chunks_indexed": 0,
        "files": {},
        "embedding_metadata": None,
    }
    meta_path = output / "index.json"
    atomic_write(meta_path, canonical(metadata))
    try:
        encoder = embedder or MiniLMEmbedder(config, cache, local_files_only)
        np = numpy_module()
        metadata["embedding_metadata"] = encoder.metadata
        metadata["packages"] = {
            name: importlib.metadata.version(name) for name in ("faiss-cpu", "numpy")
        }
        dimensions = config["embedding"]["dimensions"]
        pending: list[dict[str, Any]] = []
        batch_size = config["embedding"]["batch_size"]
        with (
            (output / "chunks.jsonl").open("x", encoding="utf-8") as stream,
            (output / "vectors.f32.partial").open("xb") as vector_stream,
        ):

            def add_batch() -> None:
                if not pending:
                    return
                vectors = checked_vectors(
                    encoder.encode([c["text"] for c in pending]), len(pending), dimensions, np
                )
                vectors.tofile(vector_stream)
                for chunk in pending:
                    stream.write(json.dumps(chunk, sort_keys=True, ensure_ascii=False) + "\n")
                metadata["chunks_indexed"] += len(pending)
                pending.clear()

            for number, doc in enumerate(documents, 1):
                chunks = chunk_document(doc, encoder.tokenizer, config["chunking"])
                if not chunks:
                    raise LabError(f"Document has no tokenizable content: {doc['document_id']}")
                for chunk in chunks:
                    pending.append(chunk.record())
                    if len(pending) == batch_size:
                        add_batch()
                if number % 500 == 0:
                    stream.flush()
                    atomic_write(meta_path, canonical(metadata))
                    print(
                        f"Indexed {number}/{len(documents)} documents; "
                        f"{metadata['chunks_indexed']} chunks",
                        flush=True,
                    )
            add_batch()
        command = [
            sys.executable,
            "-m",
            "model_adaptation_lab.retrieval.faiss_worker",
            "build",
            str(output / "vectors.faiss"),
            str(output / "vectors.f32.partial"),
            str(metadata["chunks_indexed"]),
            str(dimensions),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise LabError(f"FAISS build worker failed: {completed.stderr[-2000:]}")
        (output / "vectors.f32.partial").unlink()
        metadata["faiss_execution"] = "isolated_cpu_process_no_torch"
        metadata["files"] = {
            name: file_hash(output / name) for name in ("chunks.jsonl", "vectors.faiss")
        }
        metadata["status"] = "completed"
        return metadata
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["build_wall_seconds"] = time.perf_counter() - started
        atomic_write(meta_path, canonical(metadata))


def verify_index(path: Path, manifests: Path) -> dict[str, Any]:
    from model_adaptation_lab.data.splits import verify

    verify(manifests)
    metadata = read_json(path / "index.json")
    if metadata.get("status") != "completed":
        raise LabError("Index is incomplete or failed; cannot use partial artifacts")
    if metadata["protocol_freeze_sha256"] != file_hash(manifests / "freeze.json"):
        raise LabError("Index protocol provenance mismatch")
    if metadata["corpus"] != read_json(manifests / "corpus.json"):
        raise LabError("Index corpus provenance mismatch")
    if set(metadata["files"]) != {"chunks.jsonl", "vectors.faiss"}:
        raise LabError("Index artifact set is invalid")
    for name, expected in metadata["files"].items():
        if file_hash(path / name) != expected:
            raise LabError(f"Index artifact hash mismatch: {name}")
    validate_retrieval(metadata["config"])
    return dict(metadata)
