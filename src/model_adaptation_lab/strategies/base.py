"""Zero-shot strategy and the shared, serializable answering contract."""

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from model_adaptation_lab.io import digest
from model_adaptation_lab.models.loader import GenerationBackend
from model_adaptation_lab.models.prompting import prompt_hash, render_messages


@dataclass(frozen=True)
class AnswerResult:
    question_id: str
    answer: str
    strategy: str
    model_revision: str
    tokenizer_revision: str
    prompt_version: str
    prompt_sha256: str
    retrieved_document_ids: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    generation_latency_ms: float
    retrieval_latency_ms: float
    output_tokens_per_second: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    finish_reason: str
    generation_parameters: dict[str, Any]
    retrieved_chunks: tuple[dict[str, Any], ...] = ()
    retrieval_parameters: dict[str, Any] | None = None
    citation_document_ids: tuple[str, ...] = ()


@runtime_checkable
class AnsweringStrategy(Protocol):
    name: str

    def answer(self, question: str) -> AnswerResult: ...


class BaseStrategy:
    name = "base"

    def __init__(self, backend: GenerationBackend, prompts: dict[str, Any]) -> None:
        self.backend = backend
        self.prompts = prompts
        self.demonstrations: list[dict[str, Any]] = []
        self.prompt_version = str(prompts["base_version"])

    def answer(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        messages = render_messages(question, self.prompts, self.demonstrations)
        output = self.backend.generate(messages)
        elapsed = (time.perf_counter() - started) * 1000
        return AnswerResult(
            question_id=digest(question.encode()),
            answer=output.text,
            strategy=self.name,
            model_revision=self.backend.model_config["revision"],
            tokenizer_revision=self.backend.model_config["tokenizer_revision"],
            prompt_version=self.prompt_version,
            prompt_sha256=prompt_hash(self.prompts, self.demonstrations),
            retrieved_document_ids=(),
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            latency_ms=elapsed,
            generation_latency_ms=output.generation_latency_ms,
            retrieval_latency_ms=0.0,
            output_tokens_per_second=output.output_tokens * 1000 / output.generation_latency_ms
            if output.generation_latency_ms > 0
            else None,
            peak_allocated_bytes=output.peak_allocated_bytes,
            peak_reserved_bytes=output.peak_reserved_bytes,
            finish_reason=output.finish_reason,
            generation_parameters=dict(self.backend.generation_parameters),
        )
