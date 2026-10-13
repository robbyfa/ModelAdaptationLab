# Implementation plan and phase boundary

## Phase 1 — implemented

- Inspected the initial repository (one README, clean Git state, no AGENTS.md instructions).
- Verified immutable source/model revisions, actual data schemas, and archive terms before adapters.
- Added an installable Python 3.11+ package, CLI, source configuration, uv lockfile, and CI.
- Added strict original/NVIDIA adapters, original corpus validation, deterministic overlap and
  evidence auditing, source hashes, separated processed records, and immutable manifests.
- Executed the audit against real downloaded data; preserved source defects and limitations.
- Added synthetic-only tests for schema errors, duplicates, missing evidence, source IDs,
  annotation spans, manifest determinism, role boundaries, integrity, persistence, and reports.

**Next concrete step:** write the overlap/reference/final-test protocol described in
`source_review.md`, review the identified IDs, and create a new versioned eligible manifest
without rewriting this source snapshot. Then start Phase 2.

## Remaining phases

2. Base/few-shot: common AnsweringStrategy and AnswerResult, pinned model loader, chat template,
   frozen generation config, training-only demonstrations, latency/token accounting, CPU mocks,
   explicit hardware prerequisites, and validation-only smoke runs.
3. Retrieval: independently frozen corpus, deterministic chunks, pinned embeddings, FAISS,
   source IDs, retrieval timings, evidence recall, and no annotation-based oracle retrieval.
4. LoRA: assistant-only loss, padding masks, validation checkpoint selection, adapter-only
   checkpoints, CUDA/quantization checks, memory/time capture, MLflow reconstruction metadata,
   and one shared adapter for LoRA and LoRA + RAG.
5. Evaluation: finalized untouched test protocol, five-way benchmark, missing/failure-aware
   quality/resource metrics, frozen judging rubric plus human review, nested training subsets,
   ranks 8/16, and clearly separate ablations.
6. Report: measured results, plots, error taxonomy, validity threats, explicit cost assumptions,
   and a concise standalone summary.

Model execution, training, MLflow runs, retrieval indexing, ablations, and research conclusions
are **not implemented or claimed** in Phase 1. The user’s full definition of done remains unmet.

## Verification performed

- `uv run --locked --offline pytest -q`: **37 passed** (synthetic fixtures only).
- `uv run --locked --offline ruff check .`: passed.
- `uv run --locked --offline ruff format --check .`: passed.
- `uv run --locked --offline mypy`: passed, strict checking of seven package source files.
- `model-lab data download`: verified the cached official archive, re-extracted its allowlisted
  subset, and verified the derivative. A separate uncached NVIDIA download exercised the
  implemented network downloader and checksum validation with TLS verification enabled.
- Real-data `audit`, `prepare`, repeat `prepare`, and `verify`: passed. Source snapshots remain
  byte-stable across repeat preparation. Experimental-use flags remain blocked as intended.
- Missing-source CLI audit: saves an explicit report and returns exit code 2, covered by tests.
- GitHub Actions is configured with the same CPU-only checks; a hosted CI run has not been observed.

Local verification used Python 3.12.12 on macOS arm64. Package versions are fixed in `uv.lock`.
