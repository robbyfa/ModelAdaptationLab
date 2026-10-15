"""Actual CPU FAISS over synthetic vectors/text; no model downloads or QA dataset access."""

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from model_adaptation_lab.evaluation.retrieval import retrieval_metrics, summarize_retrieval
from model_adaptation_lab.io import LabError, file_hash
from model_adaptation_lab.models.loader import Generation, yaml_config
from model_adaptation_lab.retrieval.chunking import chunk_document, load_corpus
from model_adaptation_lab.retrieval.embedding import validate_retrieval
from model_adaptation_lab.retrieval.indexing import build_index, checked_vectors, verify_index
from model_adaptation_lab.retrieval.retriever import FaissRetriever, RetrievalResult
from model_adaptation_lab.strategies.base import AnsweringStrategy
from model_adaptation_lab.strategies.rag import RAGStrategy


class SyntheticEncoder:
    metadata = {"backend": "synthetic_vectors_only"}

    def __init__(self):
        self.calls = []

    def tokenizer(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    def encode(self, texts):
        self.calls.extend(texts)
        vectors = np.zeros((len(texts), 384), dtype="float32")
        vectors[:, 0] = 1  # Ties intentionally exercise deterministic FAISS ordering.
        return vectors


@pytest.fixture
def config():
    return yaml_config(Path("configs/retrieval.yaml"))


@pytest.fixture
def built_index(tmp_path, synthetic_protocol, config):
    manifests, processed = synthetic_protocol
    output = tmp_path / "index"
    encoder = SyntheticEncoder()
    info = build_index(
        manifests=manifests,
        processed=processed,
        config=config,
        output=output,
        cache=tmp_path / "cache",
        local_files_only=True,
        embedder=encoder,
    )
    return output, encoder, info


def test_chunk_offsets_overlap_stability_and_provenance():
    text = "alpha  beta\ngamma delta epsilon zeta"
    doc = {
        "document_id": "synthetic-doc",
        "title": "Synthetic",
        "text": text,
        "source_url": "https://example.invalid",
        "source": "original_technotes",
    }
    config = {"version": "body-wordpiece-v1", "tokens": 3, "overlap": 1}
    chunks = chunk_document(doc, SyntheticEncoder().tokenizer, config)
    assert [c.text for c in chunks] == ["alpha  beta\ngamma", "gamma delta epsilon", "epsilon zeta"]
    assert chunks == chunk_document(doc, SyntheticEncoder().tokenizer, config)
    assert all(
        c.text == text[c.start : c.end] and c.document_id == doc["document_id"] for c in chunks
    )
    assert len({c.chunk_id for c in chunks}) == 3
    assert all(c.source_url == doc["source_url"] for c in chunks)
    with pytest.raises(LabError, match="overlap"):
        chunk_document(doc, SyntheticEncoder().tokenizer, config | {"overlap": 3})


def test_corpus_rejects_modified_or_annotation_content(synthetic_protocol):
    manifests, processed = synthetic_protocol
    docs, _ = load_corpus(manifests, processed)
    assert len(docs) == 14 and all("answer" not in d for d in docs)
    path = processed / "source_documents.jsonl"
    path.write_text(json.dumps({"question": "must not index", "answer": "forbidden"}) + "\n")
    with pytest.raises(LabError, match="frozen source-document"):
        load_corpus(manifests, processed)


def test_faiss_real_roundtrip_ids_and_boundary_ties(
    built_index, synthetic_protocol, config, tmp_path
):
    path, encoder, info = built_index
    assert info["documents_indexed"] == 14 and info["chunks_indexed"] == 14
    retriever = FaissRetriever(
        path, synthetic_protocol[0], config, tmp_path, True, embedder=encoder
    )
    result = retriever.retrieve("synthetic question")
    assert len(result.chunks) == 5
    all_chunks = [json.loads(line) for line in (path / "chunks.jsonl").read_text().splitlines()]
    assert [c["chunk_id"] for c in result.chunks] == sorted(c["chunk_id"] for c in all_chunks)[:5]
    assert all(c["score"] == pytest.approx(1.0) for c in result.chunks)
    assert result.latency_ms >= 0
    assert encoder.calls[-1] == "synthetic question"
    assert verify_index(path, synthetic_protocol[0])["files"]["vectors.faiss"] == file_hash(
        path / "vectors.faiss"
    )


def test_index_immutable_and_corruption_refused(built_index, synthetic_protocol, config, tmp_path):
    path, encoder, _ = built_index
    with pytest.raises(LabError, match="directory exists"):
        build_index(
            manifests=synthetic_protocol[0],
            processed=synthetic_protocol[1],
            config=config,
            output=path,
            cache=tmp_path,
            local_files_only=True,
            embedder=encoder,
        )
    with (path / "vectors.faiss").open("ab") as out:
        out.write(b"changed")
    with pytest.raises(LabError, match="hash mismatch"):
        verify_index(path, synthetic_protocol[0])


def test_smoke_index_and_embedding_mismatch_refused(synthetic_protocol, config, tmp_path):
    path, encoder = tmp_path / "small", SyntheticEncoder()
    build_index(
        manifests=synthetic_protocol[0],
        processed=synthetic_protocol[1],
        config=config,
        output=path,
        cache=tmp_path,
        local_files_only=True,
        max_documents=2,
        embedder=encoder,
    )
    with pytest.raises(LabError, match="smoke corpus"):
        FaissRetriever(path, synthetic_protocol[0], config, tmp_path, True, embedder=encoder)
    changed = copy.deepcopy(config)
    changed["embedding"]["revision"] = "0" * 40
    with pytest.raises(LabError, match="embedding mismatch"):
        FaissRetriever(
            path,
            synthetic_protocol[0],
            changed,
            tmp_path,
            True,
            allow_smoke_index=True,
            embedder=encoder,
        )
    retriever = FaissRetriever(
        path,
        synthetic_protocol[0],
        config,
        tmp_path,
        True,
        allow_smoke_index=True,
        embedder=encoder,
    )
    assert len(retriever.retrieve("query").chunks) == 2  # No FAISS -1 padding artifacts.


def test_embedding_validation():
    with pytest.raises(LabError, match="shape"):
        checked_vectors(np.zeros((1, 2)), 2, 2, np)
    with pytest.raises(LabError, match="normalized"):
        checked_vectors(np.zeros((1, 2)), 1, 2, np)
    with pytest.raises(LabError, match="finite-value"):
        checked_vectors(np.array([[np.nan, 1.0]]), 1, 2, np)


def test_failed_build_retained_and_not_loadable(tmp_path, synthetic_protocol, config):
    class Broken(SyntheticEncoder):
        def encode(self, texts):
            raise LabError("synthetic encoder failure")

    path = tmp_path / "failed"
    with pytest.raises(LabError, match="encoder failure"):
        build_index(
            manifests=synthetic_protocol[0],
            processed=synthetic_protocol[1],
            config=config,
            output=path,
            cache=tmp_path,
            local_files_only=True,
            embedder=Broken(),
        )
    assert json.loads((path / "index.json").read_text())["status"] == "failed"
    with pytest.raises(LabError, match="incomplete or failed"):
        verify_index(path, synthetic_protocol[0])


def test_retrieval_metrics_distinguish_document_and_span_and_missing_judgments():
    row = {"document_id": "gold", "answerable": True, "start": 5, "end": 20}
    partial = ({"document_id": "gold", "start": 0, "end": 10},)
    metrics = retrieval_metrics(partial, ("gold", "invented"), row)
    assert metrics["annotated_document_recall_at_k"] == 1
    assert metrics["annotated_span_recall_at_k"] == 0
    assert metrics["precision_at_k"] is None
    assert metrics["citation_ids_in_retrieved_documents_fraction"] == 0.5
    covered = partial + ({"document_id": "gold", "start": 8, "end": 25},)
    assert retrieval_metrics(covered, (), row)["annotated_span_recall_at_k"] == 1
    gap = partial + ({"document_id": "gold", "start": 11, "end": 25},)
    assert retrieval_metrics(gap, (), row)["annotated_span_recall_at_k"] == 0
    unanswerable = dict(row, answerable=False, document_id=None, start=None, end=None)
    assert retrieval_metrics(covered, (), unanswerable)["annotated_document_recall_at_k"] is None
    assert summarize_retrieval([])["annotated_document_recall_at_k"] is None


def test_rag_interface_passes_question_only_and_exposes_citable_sources(config):
    class FakeRetriever:
        metadata = {"index_metadata_sha256": "test"}

        def retrieve(self, question):
            assert question == "Synthetic question only"
            return RetrievalResult(
                (
                    {
                        "document_id": "doc",
                        "chunk_id": "chunk",
                        "title": "Title",
                        "text": "Synthetic evidence",
                        "start": 0,
                        "end": 18,
                        "source_url": None,
                    },
                ),
                3.0,
            )

    class FakeBackend:
        model_config = {"revision": "a" * 40, "tokenizer_revision": "a" * 40}
        generation_parameters = {"max_new_tokens": 10}

        def generate(self, messages):
            assert "Synthetic evidence" in messages[1]["content"]
            assert "untrusted reference" in messages[0]["content"]
            assert "[doc]" in messages[1]["content"]
            return Generation("Supported [doc]", 40, 5, 2.0, None, None, "eos")

    strategy = RAGStrategy(FakeBackend(), {"base_version": "unused"}, FakeRetriever(), config)
    assert isinstance(strategy, AnsweringStrategy)
    result = strategy.answer("Synthetic question only")
    assert result.retrieved_document_ids == ("doc",) and result.citation_document_ids == ("doc",)
    assert result.retrieval_latency_ms == 3 and result.generation_latency_ms == 2
    assert result.retrieved_chunks[0]["start"] == 0


def test_mutable_embedding_revision_and_invalid_chunking_rejected(config):
    bad = copy.deepcopy(config)
    bad["embedding"]["revision"] = "main"
    with pytest.raises(LabError, match="Unsupported"):
        validate_retrieval(bad)
    bad = copy.deepcopy(config)
    bad["chunking"]["overlap"] = 224
    with pytest.raises(LabError, match="Unsupported"):
        validate_retrieval(bad)


def test_rag_runs_through_common_harness(config, synthetic_protocol, tmp_path):
    from model_adaptation_lab.evaluation.runner import run_evaluation

    class Backend:
        model_config = yaml_config(Path("configs/model.yaml")) | {"device": "cpu"}
        generation_parameters = {"do_sample": False}
        metadata = {"backend": "synthetic"}

        def generate(self, messages):
            return Generation("Synthetic response [SYNTH_12]", 20, 5, 3.0, None, None, "eos")

    class Retriever:
        metadata = {"index_metadata_sha256": "synthetic"}

        def retrieve(self, question):
            assert question in {"syntheticrequest12", "syntheticrequest13"}
            return RetrievalResult(
                (
                    {
                        "document_id": "SYNTH_12",
                        "chunk_id": "c",
                        "title": "Title",
                        "text": "response12",
                        "start": 0,
                        "end": 10,
                    },
                ),
                1.0,
            )

    state = run_evaluation(
        strategy_names=["base", "rag"],
        model=Backend.model_config,
        generation=yaml_config(Path("configs/generation.yaml")),
        prompts=yaml_config(Path("configs/prompts.yaml")),
        manifests=synthetic_protocol[0],
        processed=synthetic_protocol[1],
        output=tmp_path / "rag-run",
        cache=tmp_path,
        smoke=True,
        limit=None,
        local_files_only=True,
        backend=Backend(),
        retrieval_config=config,
        retriever=Retriever(),
    )
    assert state["status"] == "completed"
    assert state["results"]["rag"]["retrieval"]["annotated_span_recall_at_k"] == 1
    assert state["results"]["rag"]["retrieval"]["annotated_span_recall_at_k_denominator"] == 1
    assert state["results"]["rag"]["retrieval"]["precision_at_k"] is None


def test_search_worker_closes_cleanly(built_index, synthetic_protocol, config, tmp_path):
    retriever = FaissRetriever(
        built_index[0], synthetic_protocol[0], config, tmp_path, True, embedder=built_index[1]
    )
    worker = retriever.worker
    retriever.close()
    assert worker.poll() is not None
    retriever.close()  # Idempotent cleanup.
    with pytest.raises(LabError, match="closed"):
        retriever.retrieve("synthetic question")
