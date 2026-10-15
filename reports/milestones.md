# Milestones after retrieval

The next milestones follow the user's research sequence. The project is not ready for
portfolio claims or a final scientific conclusion until all three have real execution evidence.

1. **Real LoRA fitting:** execute rank-8 supervised training, retain assistant-token training
   and validation losses, select the best validation checkpoint, save a separate adapter,
   plot loss curves, verify reloading, and track provenance/resources in MLflow. Use the same
   adapter for LoRA and LoRA + RAG. Do not interpret lower cross-entropy as answer correctness.
2. **Development evaluator:** version a correctness and evidence-support rubric; distinguish
   retrieved-passage support from full-reference evidence; include unsupported claims,
   appropriate abstention and unnecessary refusal. Validate deterministic and semantic
   checks against a stratified, independently human-reviewed development sample. Treat any
   LLM judge as an estimate, measure reviewer/judge agreement, and preserve missing evidence,
   failures, truncations, and disagreements. Select prompts and retrieval settings only on
   development data. Freeze the evaluator, review sampling procedure, cost assumptions,
   comparison settings, and statistical analysis plan before final-test release.
3. **Frozen comparison:** implement and test the protocol's final-release procedure, freeze
   hashes of all five configurations, then run the same held-out questions once under the
   declared settings. Run rank 8/16 and nested 25/50/100% training-size ablations separately
   (record whether alpha scales with rank). Save per-question results, MLflow artifacts,
   failure rates, resource measurements, and uncertainty estimates. Write conclusions that
   distinguish measured effects from assumptions and acknowledge the audit-exposed internal
   holdout. No generated or inferred measurements may fill unavailable metrics.

Only after those results and limitations can be reviewed independently should a fourth
flagship portfolio entry be considered. Website publication is outside the current phase.
