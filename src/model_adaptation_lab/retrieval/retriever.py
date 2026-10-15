"""Question-only retrieval; never accept gold answers or per-question candidate IDs."""

import contextlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from model_adaptation_lab.io import LabError, file_hash
from model_adaptation_lab.retrieval.embedding import Embedder, MiniLMEmbedder, validate_retrieval
from model_adaptation_lab.retrieval.indexing import checked_vectors, numpy_module, verify_index


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[dict[str, Any], ...]
    latency_ms: float


class Retriever(Protocol):
    metadata: dict[str, Any]

    def retrieve(self, question: str) -> RetrievalResult: ...


class FaissRetriever:
    def __init__(
        self,
        path: Path,
        manifests: Path,
        config: dict[str, Any],
        cache: Path,
        local_files_only: bool,
        *,
        allow_smoke_index: bool = False,
        embedder: Embedder | None = None,
    ) -> None:
        load_started = time.perf_counter()
        validate_retrieval(config)
        self.metadata = verify_index(path, manifests)
        if self.metadata["backend_type"] != "real_minilm" and embedder is None:
            raise LabError("Synthetic embedding indexes cannot be used with a real encoder")
        if self.metadata["corpus_scope"] != "full_frozen_subset" and not allow_smoke_index:
            raise LabError("A smoke corpus index is not valid for normal evaluation")
        original = self.metadata["config"]
        # Device/batch changes are recorded, but representation and chunking must match.
        for key, value in original["embedding"].items():
            if key not in {"device", "batch_size"} and config["embedding"].get(key) != value:
                raise LabError(f"Query/index embedding mismatch: {key}")
        if config["chunking"] != original["chunking"] or config["index"] != original["index"]:
            raise LabError("Query/index chunking or index settings mismatch")
        self.encoder = embedder or MiniLMEmbedder(config, cache, local_files_only)
        self.np = numpy_module()
        self.worker: subprocess.Popen[str] | None = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "model_adaptation_lab.retrieval.faiss_worker",
                "serve",
                str(path / "vectors.faiss"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self.worker is not None and self.worker.stdout is not None
        ready = self.worker.stdout.readline()
        if not ready:
            self.close()
            raise LabError("FAISS search worker failed to initialize")
        try:
            ready_info = json.loads(ready)
            self.count, self.dimensions = ready_info["count"], ready_info["dimensions"]
        except (ValueError, KeyError) as exc:
            self.close()
            raise LabError("Invalid FAISS worker response") from exc
        self.chunks = [
            json.loads(line) for line in (path / "chunks.jsonl").read_text().splitlines()
        ]
        if len(self.chunks) != self.count or self.dimensions != config["embedding"]["dimensions"]:
            raise LabError("Index vector count/dimension mismatch")
        if len({c["chunk_id"] for c in self.chunks}) != len(self.chunks) or not self.chunks:
            raise LabError("Duplicate/empty chunk identifiers")
        self.top_k = config["retrieval"]["top_k"]
        self.metadata = self.metadata | {
            "index_metadata_sha256": file_hash(path / "index.json"),
            "query_config": config,
            "query_embedding_metadata": self.encoder.metadata,
            "index_load_ms": (time.perf_counter() - load_started) * 1000,
            "retrieval_timing_scope": (
                "query embedding, CPU FAISS IPC/search, and result formatting"
            ),
        }

    def retrieve(self, question: str) -> RetrievalResult:
        if not question.strip():
            raise LabError("Cannot retrieve for an empty question")
        started = time.perf_counter()
        query = checked_vectors(self.encoder.encode([question]), 1, self.dimensions, self.np)
        if self.worker is None:
            raise LabError("Retriever is closed")
        assert self.worker.stdin is not None and self.worker.stdout is not None
        try:
            self.worker.stdin.write(
                json.dumps({"query": query[0].tolist(), "top_k": self.top_k}) + "\n"
            )
            self.worker.stdin.flush()
            response = json.loads(self.worker.stdout.readline())
        except (OSError, ValueError) as exc:
            raise LabError(f"FAISS search worker failed: {exc}") from exc
        if "error" in response:
            raise LabError(f"FAISS search failed: {response['error']}")
        hits = [
            self.chunks[int(i)] | {"score": float(score)}
            for score, i in zip(response["scores"], response["positions"], strict=True)
            if i >= 0
        ]
        count = min(self.top_k, self.count)
        hits.sort(key=lambda c: (-c["score"], c["chunk_id"]))
        return RetrievalResult(tuple(hits[:count]), (time.perf_counter() - started) * 1000)

    def close(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None:
            if worker.stdin:
                with contextlib.suppress(OSError):
                    worker.stdin.close()
            if worker.poll() is None:
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=5)
            for stream in (worker.stdout, worker.stderr):
                if stream:
                    stream.close()
            self.worker = None

    def __del__(self) -> None:
        self.close()
