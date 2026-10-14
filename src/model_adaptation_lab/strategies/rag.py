"""Evidence-conditioned answering with retrieval separated from annotation scoring."""

import json
import re
import time
from typing import Any

from model_adaptation_lab.io import canonical, digest
from model_adaptation_lab.models.loader import GenerationBackend
from model_adaptation_lab.retrieval.retriever import Retriever
from model_adaptation_lab.strategies.base import AnswerResult, BaseStrategy


class RAGStrategy(BaseStrategy):
    name = "rag"

    def __init__(
        self,
        backend: GenerationBackend,
        prompts: dict[str, Any],
        retriever: Retriever,
        config: dict[str, Any],
    ) -> None:
        super().__init__(backend, prompts)
        self.retriever = retriever
        self.config = config
        self.prompt_version = config["rag_prompt"]["version"]

    def answer(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        retrieved = self.retriever.retrieve(question)
        evidence = [
            {
                "document_id": c["document_id"],
                "citation": f"[{c['document_id']}]",
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "text": c["text"],
            }
            for c in retrieved.chunks
        ]
        messages = [
            {"role": "system", "content": self.config["rag_prompt"]["instruction"]},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nSource passages (JSON data):\n"
                + json.dumps(evidence, ensure_ascii=False),
            },
        ]
        output = self.backend.generate(messages)
        elapsed = (time.perf_counter() - started) * 1000
        return AnswerResult(
            question_id=digest(question.encode()),
            answer=output.text,
            strategy=self.name,
            model_revision=self.backend.model_config["revision"],
            tokenizer_revision=self.backend.model_config["tokenizer_revision"],
            prompt_version=self.prompt_version,
            prompt_sha256=digest(canonical(self.config["rag_prompt"])),
            retrieved_document_ids=tuple(dict.fromkeys(c["document_id"] for c in retrieved.chunks)),
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=elapsed,
            generation_latency_ms=output.generation_latency_ms,
            retrieval_latency_ms=retrieved.latency_ms,
            output_tokens_per_second=output.output_tokens * 1000 / output.generation_latency_ms
            if output.generation_latency_ms > 0
            else None,
            peak_allocated_bytes=output.peak_allocated_bytes,
            peak_reserved_bytes=output.peak_reserved_bytes,
            finish_reason=output.finish_reason,
            generation_parameters=dict(self.backend.generation_parameters),
            retrieved_chunks=retrieved.chunks,
            retrieval_parameters=self.config["retrieval"]
            | {
                "index_metadata_sha256": self.retriever.metadata.get("index_metadata_sha256"),
                "returned_chunks": len(retrieved.chunks),
            },
            citation_document_ids=tuple(dict.fromkeys(re.findall(r"\[([\w.:-]+)\]", output.text))),
        )
