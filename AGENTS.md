# Model Adaptation Lab experimental boundaries

The active experiment protocol is `configs/protocol-v1.yaml`, documented in
`reports/protocol-v1.md`. The original Phase 1 snapshot remains immutable.

During Phase 2 and subsequent model/prompt/training development:

- Use `data/manifests/techqa-internal-v1` and `data/processed/techqa-internal-v1`.
- Load training/demonstration and validation/selection records through
  `model_adaptation_lab.data.splits.load_annotations`; do not bypass its integrity or role checks.
- Do not inspect raw original QA or NVIDIA annotations for model development. The original
  training file now includes sealed final-holdout questions and references.
- Do not read final-holdout question/answer text, reconstruct it from raw IDs, or unseal it.
  Phase 5 requires the predeclared evaluation plan and tested release procedure in the protocol.
- Use synthetic fixtures exclusively in tests. Never include real heldout text in fixtures or logs.
- Source-document text is permitted under the frozen open-book protocol; QA annotations and
  candidate-document lists must not become retrieval content or oracle retrieval filters.
- Do not alter frozen manifests or policy constants in place. Changes require a new version
  and a migration/exposure record; never relabel this internal holdout as an official test set.

Dataset-maintenance tasks explicitly requested by the user may run the protocol builder
and integrity audits. Keep such preparation distinct from model selection, and preserve an
honest record of any holdout exposure. Phase 1's reports are historical, not current readiness.
