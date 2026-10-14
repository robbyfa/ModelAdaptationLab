# Model Adaptation Lab

**When are prompting and retrieval sufficient for technical support QA, and when is LoRA worth its cost?**

A reproducible experiment comparing Qwen2.5-1.5B-Instruct across zero-shot prompting,
few-shot prompting, RAG, LoRA, and LoRA + RAG. This is a research benchmark, not a chatbot.

**Status:** all five strategies are implemented. Real rank-8 LoRA training completed locally
on Apple M3 Pro MPS: 417 training records, 286 validation records, three epochs. Validation
loss fell from **1.6336 to 1.3014** at the selected second-epoch checkpoint. This does **not**
establish answer-quality improvement. The final benchmark remains sealed.

Read the [training results and loss curves](reports/phase4.md),
[retrieval results](reports/phase3.md), and [research milestones](reports/milestones.md).

## Scientific boundaries

The TechQA audit found 23 exact normalized question overlaps across official splits and
11 development reference/span mismatches. NVIDIA's `train` derivative contains all 910
original train/development records and is inspection-only. Archive and hub licensing labels
differ; see the [dataset audit](reports/dataset_audit.md).

[Protocol v1](reports/protocol-v1.md) freezes **417 training / 286 validation / 106 internal
holdout records**, excluding 101 records. The internal holdout is audit-exposed, not an official
or wholly untouched test set. Development uses guarded loaders; final access stays sealed.

RAG indexes **154,630 chunks from 28,482 source documents**, with pinned MiniLM embeddings
and exact FAISS search. QA annotations and per-question candidate lists cannot become retrieval
content or filters. Both adapted strategies use the same adapter and frozen base revision.

## Reproduce

Use Python 3.12 and [uv](https://docs.astral.sh/uv/) from the repository root. Data preparation
needs roughly 5 GB; models, indexes, and training artifacts need additional space.

```bash
uv sync --locked
uv run model-lab data download
uv run model-lab data audit
uv run model-lab data prepare
uv run model-lab data protocol
uv run model-lab data verify --manifests-dir data/manifests/techqa-internal-v1

uv sync --locked --extra training --extra retrieval
uv run --extra retrieval model-lab retrieval build \
  --index-dir outputs/my-index --embedding-device mps
uv run --extra training model-lab train --device mps \
  --output-dir outputs/my-lora-run
uv run --extra training --extra retrieval model-lab evaluate --strategy adapted --smoke \
  --device mps --dtype float32 --embedding-device mps \
  --adapter-dir outputs/my-lora-run --index-dir outputs/my-index \
  --output-dir outputs/my-adapted-smoke
```

Use `--strategy available` without `--adapter-dir` for base/few-shot/RAG. Use new output
directories for each run. MPS float32 has real execution evidence; CUDA remains unverified
locally. Explicit CPU execution may be slow. Quantized training and interruption resume are
not yet supported. See [Phase 4](reports/phase4.md) for MLflow and checkpoint details.

Raw text, model weights, indexes, and MLflow data are Git-ignored. Committed summaries contain
metrics and hashes. CPU CI uses synthetic fixtures, without model downloads or paid APIs:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

**Next:** validate the quality evaluator on development data, then freeze and execute the
five-way benchmark and predeclared ablations. Portfolio publication waits for measured
findings and an evidence-based conclusion.
