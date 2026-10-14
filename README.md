# Model Adaptation Lab

**When are prompting and retrieval sufficient for technical support QA, and when is LoRA worth its cost?**

This research framework will compare Qwen2.5-1.5B-Instruct across zero-shot, few-shot,
RAG, LoRA, and LoRA + RAG under a shared evaluation protocol. It is not a chatbot.

**Status: Phase 3 implemented; base/few-shot/RAG real-model smoke verified.** No LoRA training
or final benchmark has run. A two-question validation smoke is execution evidence only;
see [Phase 3 results and limitations](reports/phase3.md).

## What the audit found

- Official training/development: **600 / 310 questions**, **610 answerable** overall.
- **23 exact normalized question overlaps** cross those splits.
- **11 development reference/span mismatches**, including two empty reference answers.
- **28,482 source documents** cover all annotated gold documents.
- NVIDIA's `train` file contains **all 910 train + development records**; it is inspection-only.
- Archive terms say **CDLA-Permissive-1.0**, while the hub card says Apache-2.0.
- Protocol v1 resolves eligibility: **417 training / 286 validation / 106 sealed internal
  holdout questions**. It excludes 101 records and preserves the original source snapshot.
- The holdout is audit-exposed and internally derived, not an official or wholly untouched test set.

See the [generated audit](reports/dataset_audit.md), [full findings](reports/dataset_audit.json),
and [protocol v1](reports/protocol-v1.md) for the active decisions.

## Reproduce Phase 1

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended. Run from the repository root:

```bash
uv sync --locked
uv run model-lab data download   # ~2.96 GB; streams extraction of selected files
uv run model-lab data audit
uv run model-lab data prepare
uv run model-lab data verify
uv run model-lab data protocol  # Versioned eligible split; final holdout stays sealed
uv run model-lab data verify --manifests-dir data/manifests/techqa-internal-v1
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Allow about 5 GB of disk space for the archive, extracted subset, and processed data.
No GPU, model weights, API keys, or paid services are needed. CI uses synthetic fixtures only
and never downloads datasets. The lockfile pins data and optional inference dependencies.

`data audit` saves a limitations report and exits 2 when sources are missing. `prepare`
preserves official assignments and creates an immutable manifest lock; a changed input
requires a new versioned output directory. `verify` checks integrity, **not readiness**.
`evaluate` supports base/few-shot/RAG validation inference; training and the five-way benchmark
are not implemented yet. Install model dependencies separately with `uv sync --locked --extra inference`.

```bash
uv run --extra inference model-lab evaluate --strategy both --smoke \
  --device mps --dtype float32 --output-dir outputs/my-smoke
```

Use `--device cuda` on a CUDA host, or explicitly select `cpu` (potentially slow). All final-test
access remains sealed. The successful local smoke used float32; float16 MPS failed numerical checks.

## Data boundaries

```mermaid
flowchart LR
  A[Original pinned archive] --> B[Schema and provenance checks]
  N[NVIDIA pinned derivative] --> D[Separate comparison audit]
  B --> C[Question annotations]
  B --> S[Source documents only]
  C --> F[Hashed official split assignments]
  D --> R[Dataset audit]
  F --> R
  S --> R
```

Raw data and processed text are ignored by Git. Committed manifests contain IDs and hashes,
not questions or answers. No annotation-derived NVIDIA context is exported as a corpus.
RAG uses held-out source documents under the frozen open-book protocol; it does not use
held-out annotations or per-question candidate lists to select retrieval evidence.

## Reproduce retrieval

```bash
uv sync --locked --extra retrieval
uv run --extra retrieval model-lab retrieval build --index-dir outputs/my-index \
  --embedding-device mps
uv run model-lab retrieval verify --index-dir outputs/my-index
uv run --extra retrieval model-lab evaluate --strategy available --smoke \
  --device mps --dtype float32 --embedding-device mps \
  --index-dir outputs/my-index --output-dir outputs/my-rag-smoke
```

The verified index contains **154,630 chunks from 28,482 source documents**. Use a new
output directory for every build/run. Embeddings default to CPU; choose CUDA or MPS explicitly.
Index files, model weights, and per-question outputs remain outside Git.

**Next:** Phase 4 LoRA training with assistant-only loss, validation checkpoint selection,
adapter reuse, and MLflow tracking. CUDA training requires a suitable host; this local machine
has Apple MPS only. Prompt selection and the final evaluation rubric remain pending.
See [development plan](reports/implementation_plan.md).
