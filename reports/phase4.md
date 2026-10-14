# Phase 4 — LoRA fitting and adapter reuse

Status: implementation and real training verification in progress. This document will be
updated from saved run artifacts; no final-benchmark or answer-quality conclusion is claimed.
The sequence of remaining research milestones is recorded in [milestones.md](milestones.md).

## Controlled training design

`configs/training.yaml` uses rank 8, alpha 16, dropout 0.05, learning rate 0.0002, three
maximum epochs, batch size 2 and eight accumulation steps (nominal effective batch size 16;
the final group in an epoch can be smaller). Maximum sequence length is 1,024. Seed is 42.
Auto targets resolve explicitly to Qwen attention q/k/v/o and MLP gate/up/down projections.
Only LoRA weights are trainable; base weights remain frozen and adapters are never merged.

Training and validation use `load_annotations` with their required roles. The final holdout
stays sealed. Training uses the baseline system prompt and question with a reference answer,
or the fixed abstention marker for annotation-unanswerable records. No documents, candidate
lists, validation answers, or held-out annotations enter model fitting. Validation answers
are used solely to measure checkpoint-selection loss.

The exact chat template must preserve the generation-prefix token boundary. Only tokens
following that boundary, including assistant end-of-turn tokens, receive loss. Prompt and
padding labels are -100. Padding is right-sided, with an attention mask. Whole over-length
records are excluded and their IDs/counts recorded; neither questions nor reference answers
are silently truncated. The primary run's inclusion counts will be reported below.

The initial implementation uses Transformers Trainer, PEFT, Datasets, and Accelerate.
TRL is unnecessary for this explicit collator and standard supervised Trainer path. Weights
and optimizer use float32, AdamW, a linear scheduler, no warmup, gradient norm clipping at 1,
and non-reentrant gradient checkpointing. Quantization is explicitly rejected in this first
implementation rather than silently ignored. CUDA and explicitly selected MPS/CPU devices
are supported in code; actual execution support is stated separately below.

Checkpoint selection minimizes validation loss measured at each epoch. Early stopping
patience is two evaluations. The initial pre-training validation loss is retained as a
reference, not a competing untrained adapter checkpoint. Two checkpoints are retained under
Trainer's best/latest policy; the selected adapter is exported separately. Epoch metrics
remain available even when a checkpoint is pruned.

Loss curves describe Trainer's supervised cross-entropy and logged optimizer-step loss;
validation loss is Trainer's sample-aggregated batch loss with evaluation batch size 1.
These are training diagnostics, not a validated correctness or groundedness score.

## Provenance, tracking and reproducibility

Each new output directory preserves training config, effective Trainer arguments, dataset
record hashes, the frozen protocol hash, exact model/tokenizer revisions, chat-template hash,
resolved LoRA targets, parameter counts, environment/package versions, and a code snapshot.
MLflow uses a local SQLite database and records loss history, checkpoint-selection metrics,
resource metrics, adapter artifacts and run metadata. Model weights and tracking files remain
Git-ignored. No external tracking service or paid resource is created.

Native MPS/CUDA timings differ; device and dtype are explicit. CUDA allocated/reserved peaks
are process measurements, while MPS/CPU fields are null. Wall time includes setup and
validation as well as fitting; CUDA hours describe that allocated interval on one device,
not accelerator-active utilization. Pricing and inference cost are not inferred from it.
Failed and interrupted runs retain their metadata. Existing output directories cannot be
reused. Resume-from-interruption and quantized training remain future capabilities.

Adapter loading checks completion, base/tokenizer revisions, protocol hash, and artifact
hashes; smoke adapters are rejected for normal evaluation. Base and adapted strategies run
in separate evaluator invocations to avoid accidental adapter contamination of the baseline.
LoRA + RAG uses the identical adapter and retrieval settings. Its evidence-conditioned prompt
is a format mismatch with question-only training; that is a limitation of this controlled
comparison, not a reason to train a different adapter silently.

## Reproduction

```bash
uv sync --locked --extra training --extra retrieval
uv run --extra training model-lab train --config configs/training.yaml \
  --device cuda --output-dir outputs/lora-r8-full
uv run --extra training --extra retrieval model-lab evaluate --strategy adapted \
  --adapter-dir outputs/lora-r8-full --index-dir outputs/my-index \
  --device cuda --dtype float32 --output-dir outputs/adapted-validation
```

Select `--device mps` explicitly on supported Apple hardware. `--smoke` performs two real
optimizer steps on four records per split and marks its artifacts as smoke-only. The
`--fraction` option accepts deterministic nested 0.25/0.5/1.0 subsets; those are infrastructure
for later predeclared ablations, not ablation results. MLflow records can be inspected with
`mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db` from the repository root.

Implementation references: [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora),
[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), and
[MLflow tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api/).
