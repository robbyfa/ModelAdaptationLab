# Phase 2 — base and few-shot inference

Both strategies now run through one validation-only evaluator with the pinned Qwen2.5-1.5B
base/tokenizer, shared greedy decoding, native chat formatting, and saved per-question outputs.
A real two-question smoke run completed on an Apple M3 Pro (36 GiB unified system RAM)
using MPS with float32. The chip/RAM were verified separately with `sysctl` after execution. This verifies the
execution path; it does **not** establish a quality or performance improvement.

## Implemented

- `AnsweringStrategy.answer(question)` and structured `AnswerResult` with model/tokenizer
  revisions, prompt version/hash, generation settings, token counts, timing, finish reason,
  empty retrieval IDs, and nullable CUDA memory measurements.
- Optional locked PyTorch/Transformers dependencies, explicit CUDA/CPU/MPS device selection,
  safetensors loading with `trust_remote_code=False`, and no implicit hardware fallback.
- A versioned zero-shot instruction and predefined few-shot candidate. They use the same
  system instruction; the latter adds two answerable and one unanswerable training demonstrations.
- Demonstrations loaded only through guarded eligible training access, hash-ranked using seed
  42 and filtered to at most 1,800 question-plus-answer characters. Their IDs and record hashes
  are saved. This short-example filter is a practical prompt-budget choice and a possible bias.
- A single validation runner, balanced deterministic smoke sampling, code/config snapshots,
  lockfile/package/environment metadata, exclusive output directories, and persistent errors.
- Every decoding step checks for non-finite logits. Invalid scores fail explicitly; they are
  never silently replaced with valid-looking output. Synchronization overhead is part of timings.

The prompt is named `few-shot-v1-candidate`, **not an optimized or validation-selected prompt**.
Prompt comparison/selection over eligible validation data remains necessary before the primary
benchmark; do not choose a winner using the final holdout or this two-example smoke run.

## Real execution and failed attempts

Three local attempts are preserved under ignored `outputs/`:

1. `phase2-smoke-mps-v1`: float16, before the score guard. Generation calls completed, but all
   four responses were repeated exclamation marks. This attempt is **invalid**, despite its
   original execution status of `completed`. No quality claim uses these outputs.
2. `phase2-smoke-mps-fp16-guarded`: float16, with the guard. Warmup failed on non-finite scores;
   the error and run metadata were saved, and the CLI returned exit 2.
3. `phase2-smoke-mps-fp32`: float32 with the guard. Both strategies completed on the same two
   eligible validation questions; all four responses ended with EOS, not the output limit.

The finite-score guard establishes numerical failure in this float16/device/software combination;
it does not diagnose a universal issue with Qwen, MPS, or float16 on other versions/hardware.
Use float32 for reproduction here. Failed attempts were neither removed nor rewritten.

| Float32 smoke measurement | Base | Few-shot candidate |
|---|---:|---:|
| Questions completed | 2/2 | 2/2 |
| Total input tokens | 264 | 1,104 |
| Total generated tokens, including EOS | 24 | 56 |
| Median strategy latency (ms) | 1,186.49 | 2,679.15 |

The baseline emitted the exact abstention marker for both questions. The candidate produced
an answer for the answerable question and abstained on the unanswerable question. That answer
has not been independently assessed for semantic correctness or evidence support. With only
two observations, different output lengths, one fixed strategy order, and no repeated trials,
these timings cannot support an inference-cost or speed comparison.

All per-question outputs remain locally saved. `phase2_smoke.json` records sanitized measured
summaries, file hashes, runtime versions, and the failures without redistributing QA text.
The original raw outputs are required to inspect responses; report summaries are not a substitute.

## Measurement definitions and limitations

- Input tokens include the complete native chat template, system message, demonstrations,
  question, and assistant-generation prefix. No duplicate special tokens are added.
- Output token counts exclude the prompt and include generated EOS. Decoded text omits special
  tokens. Oversized prompts fail; no questions or demonstrations are silently truncated.
- Strategy latency includes prompt construction, tokenization, transfer, synchronized generation,
  and decoding. Model loading and a fixed synthetic warmup are recorded separately. Per-result
  metadata hashing, file persistence, dataset loading, and cold startup are outside this latency.
- Generation latency includes prefill and all decoding, plus per-step score checks. Tokens/second
  uses that elapsed time; it is not a pure decode-only throughput measurement.
- p50/p95 use linear interpolation of successful observations. Counts, failed attempts, and
  denominators are always saved. A two-question p95 is not a reliable tail-latency estimate.
- CUDA peak allocated/reserved memory includes the resident model. CPU/MPS CUDA-memory fields
  are null; CUDA behavior has synthetic coverage but was not exercised on physical CUDA hardware.
- Exact match and token F1 are **lexical diagnostics**, not correctness scores. F1 can reward
  superficial words even in abstentions. Only exact marker matches count as abstentions here;
  paraphrased refusals are not comprehensively classified. Unanswerable labels remain scoped
  to the original candidate documents, as specified in protocol v1.
- Semantic correctness, groundedness, unsupported-claim rate, costs, and training GPU-hours
  remain unavailable/null. Non-RAG groundedness is not assigned a failing score for lacking citations.
- MLflow integration is not implemented yet; these are local reconstructable JSON runs. Phase 4
  will add tracking integration before training and final experimental execution.

## Reproduce

```bash
uv sync --locked --extra inference

# CUDA host (default float32); downloads only the pinned model if not already cached.
uv run --extra inference model-lab evaluate --strategy both --smoke \
  --output-dir outputs/phase2-cuda-smoke

# Apple Silicon host with MPS available.
uv run --extra inference model-lab evaluate --strategy both --smoke \
  --device mps --dtype float32 --output-dir outputs/phase2-mps-smoke

# Explicit CPU fallback; may be slow and memory-intensive.
uv run --extra inference model-lab evaluate --strategy base --smoke \
  --device cpu --output-dir outputs/phase2-cpu-smoke

# Larger development run; still validation only.
uv run --extra inference model-lab evaluate --strategy both --limit 20 \
  --output-dir outputs/phase2-validation-20
```

Use `--local-files-only` after caching weights, and a **new directory** for each attempt.
`--smoke` uses two validation rows unless `--limit` is supplied, and caps generation at 64 new
tokens; normal development runs use the configured 256-token limit. Effective settings are
saved for both strategies. CPU tests need only `uv sync --locked`; no inference extra is required.

There is no `--split test`, mock CLI mode, or fallback to synthetic data. Missing prerequisites
return exit 2; per-question partial failures return exit 1; interruption returns 130 with partial
artifacts retained. Successful completion returns 0 and does not imply correct answers.

The final holdout stayed sealed. Source protocol/manifests were not changed. The next implementation
phase is frozen-corpus chunking, embedding/indexing, and RAG through this shared inference interface.
Before five-way evaluation, complete validation prompt selection and the frozen quality rubric.

## Validation

70 synthetic tests passed, including strategy conformance, role-safe demonstrations, token slicing,
chat-template options, missing devices/dependencies, startup/per-question failures, non-finite score
rejection, null memory handling, persistence, and immutable outputs. Ruff lint/format and strict mypy
passed. The full suite was also run in a separate environment without PyTorch. Hosted CI has not
been observed running for this change.

Implementation follows the pinned-major Transformers [chat-template documentation](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)
and [generation configuration reference](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation).
