# TechQA internal experiment protocol v1

**Status:** fixed before Phase 2; training and prompt selection may begin using the
versioned access layer. Final holdout access remains sealed. No model run, prompt selection,
or hyperparameter search preceded this protocol in this repository.

Protocol ID: `techqa-internal-v1`. Executable policy: `configs/protocol-v1.yaml`.
Frozen artifacts: `data/manifests/techqa-internal-v1/`.
The original Phase 1 manifests and audit remain unchanged as the source snapshot.

## Decision and rationale

Keep eligible official development examples as validation. Reserve a deterministic,
component-disjoint subset of eligible original **training** examples for final evaluation.
This preserves the development role and creates a holdout before model development without
pretending an official test set is available. No development example moves into training.
The NVIDIA derivative and leaderboard validation files are never used as new examples.

The final set is an **audit-exposed internal holdout**, not an official TechQA test set,
not independently collected data, and not wholly untouched. The raw original training
file was processed by Phase 1 audits and the protocol builder, and public upstream data
may have appeared in base-model pretraining. Some upstream training examples also appeared
in source-schema inspection before holdout assignment. No model outputs informed assignment.
These limitations must accompany every result; an external replication remains desirable.

## Overlap policy

Compute an undirected graph on all 910 original records **before exclusions**, connecting:

1. Questions equal after NFKC, casefolding, and word-token normalization.
2. Questions with normalized word-set Jaccard similarity at least 0.85.
3. Nonempty reference answers equal after the same normalization.

Use transitive connected components; quarantined rows remain graph bridges. Do not
connect all unanswerable rows through their empty-answer sentinel. Record all membership.
The normalization and threshold are conservative heuristics and do not prove the absence
of paraphrases or semantic contamination. Identical short answers can over-group unrelated
questions; this trades sample size for a stricter separation rule.

Exclude original training rows in **any** component touching official development,
even if the development member is later quarantined. Keep remaining components together
when assigning training and holdout. Source-document reuse is allowed under the explicit
open-book setting and is not an edge by itself.

Exact normalized question clusters with different answerability or different normalized
reference answers are quarantined in full. No majority vote or preferred source split is
used to resolve contradictory labels. After quarantine, retain the lexicographically
smallest question ID for each remaining exact question; preserve near duplicates within
their component. Future uncertainty estimates should resample components, not assume all
question observations are independent.

## Reference-correction policy

Validate each answerable record against the exact original document character span.

- A matching answer is unchanged.
- If the nonempty answer and valid source span differ **only in whitespace**, use the
  exact source span as the corrected answer. Preserve original and corrected hashes.
- Empty answers, absent evidence, invalid offsets, and all other disagreements are
  quarantined. Do not infer labels from offsets, perform fuzzy alignment, lower-case
  reference text, remove punctuation, or generate a new answer.
- Correction does not override the independent exact-question conflict rule.

Actual dispositions:

| Action | Records |
|---|---:|
| References unchanged | 899 |
| Whitespace-only corrections | 2 |
| Empty reference quarantines | 2 |
| Other reference/span disagreement quarantines | 7 |

Whitespace corrections are `DEV_Q038` and `DEV_Q303`. Both are additionally quarantined
because their exact-question groups have conflicting references. Thus **none of the 11
original span-mismatch rows enters validation**. Corrections are recorded, not silently
used to resolve a different label conflict. This is deterministic mechanical adjudication,
not a claimed independent human expert review.

`decisions.json` records every original question ID, component, source role, reference
action, original/corrected record and answer hashes, final assignment, and all exclusion
reasons. Raw originals remain local; corrections can be reconstructed from pinned source
spans. No original QA file is overwritten. Hashes avoid redistributing reference text.

## Holdout allocation and frozen counts

Among remaining original-training components, stratify by all-answerable, all-unanswerable,
or mixed answerability. Within each stratum, order components by SHA-256 of
`protocol_id:seed:stratum:component_id`, with seed 42 and the component's smallest question ID.
Reserve round-half-up 20% of components, bounded to leave at least one component for training
and one for holdout when at least two exist. A singleton stratum stays in training.
Never split a component to hit an example quota. Do not sweep seeds or thresholds.

| Role | Questions | Answerable | Unanswerable | Components |
|---|---:|---:|---:|---:|
| Training | 417 | 309 | 108 | 392 |
| Validation (official development subset) | 286 | 142 | 144 | 276 |
| Sealed internal final holdout | 106 | 78 | 28 | 98 |
| Total eligible | 809 | 529 | 280 | 766 |

101 original records are excluded. Reasons are nonexclusive: 67 training rows touch a
development component; 48 records belong to conflicting exact-question clusters; four
additional exact duplicates are removed; nine records have unresolved reference defects.
Do not sum these reason counts. The final set is approximately 20% of eligible original
training rows, not 20% of the entire dataset. Its answerability mix differs from validation;
report results separately for answerable/unanswerable strata as well as in aggregate.
Exclusions alter the benchmark population and may introduce selection bias. With only 106
holdout questions, uncertainty may be substantial; report component-aware intervals and avoid
generalizing small differences to all technical support questions.
There are zero cross-split components under these declared rules. Residual within-split
near-question and answer overlaps are retained in the frozen `summary.json`.

## Corpus and answerability scope

Freeze the original 28,482-document train/development subset as an **open-book** corpus,
using both its original file SHA-256 and the SHA-256 of the deterministic processed corpus
in `corpus.json`. All strategies use this same reference corpus for evidence analysis;
both RAG methods use the same index and retrieval settings. Full-corpus retrieval is a
separate experiment. The subset was chosen upstream from candidate-document annotations,
which can make retrieval easier than full-corpus search.

Only document IDs, titles, bodies, source URLs, and provenance enter the corpus. No QA
annotations, question-specific candidate lists, NVIDIA gold contexts, or corrected answers
are indexed. Shared documents and repeated document bodies across roles remain possible;
this measures question generalization in an open-book setting, not unseen-document transfer.

Original unanswerable labels mean no answer was found in the supplied candidate documents;
they do not establish global absence in all 28,482 documents. Score abstention against this
annotation scope, label it accordingly, and report any corpus-supported answer to an
unanswerable-labelled question as a disagreement needing review rather than automatically
calling it hallucination. Do not revise labels in response to favorable model outputs.

## Access and release gate

Phase 2 consumers must use `load_annotations` with these paths:

- Manifests: `data/manifests/techqa-internal-v1`
- Processed data: `data/processed/techqa-internal-v1`
- Training purpose: `training` or `demonstrations`; validation purpose: `selection`.

The versioned freeze sets `experimental_use_ready: true` and `final_test_ready: true`:
the latter means a valid nonempty holdout has been assigned, **not permission to read it**.
`final_test_access: sealed_until_phase5` blocks even `final_evaluation` calls. The builder
exports training/validation annotation files only; no test annotation file is created.
It also verifies the Phase 1 lock, every source record hash, and original corpus hash.
All versioned sidecars are hash-checked, and conflicting re-preparation fails before writes.
Raw data remain on disk; these are workflow safeguards, not OS-level secret storage.

There is deliberately no release command in Phase 1/2. Phase 5 must add and test an explicit
release operation requiring an immutable evaluation-plan artifact with:

- This protocol/manifest/corpus hash and exact model/tokenizer revisions.
- All five selected strategy configurations, prompt/demonstration hashes, generation
  settings, index/embedding/chunk settings, and trained adapter checkpoint hashes.
- Completed validation selection evidence, fixed seeds, planned rank/data-size ablations,
  and their checkpoint hashes. Primary and ablation results remain separate.
- Frozen scoring/groundedness rubric, abstention rules, failure/missing-value handling,
  confidence-interval method, and stratified manual-review sampling plan.
- Hardware, timing methodology, and explicit infrastructure-pricing assumptions.

Register the plan in version control and MLflow before unsealing. Record release time and
plan hash, then run only the predeclared battery. Log failures and retries; exact-config
resumption is permitted, but changes driven by final outputs invalidate confirmatory status.
After exposure, any tuning requires a new external holdout or an explicitly exploratory label.
Do not toggle flags or edit this frozen lock manually. New protocol decisions need a new
version, a migration report, and an honest exposure history.

## Reproduction and remaining limitations

```bash
uv run model-lab data protocol
uv run model-lab data verify --manifests-dir data/manifests/techqa-internal-v1
uv run pytest -q
```

Requires the prepared Phase 1 sources and manifests. Repeat preparation is byte-stable.
The synthetic test suite covers corrections, quarantine, transitive grouping, contradictory
labels, stable allocation, source/corpus tampering, sealed access, and immutable outputs.
No final question or answer text is printed by the protocol CLI or checked into Git.

The source archive's CDLA-Permissive-1.0 terms are retained. The inconsistent hub-card label
remains documented; this protocol does not purport to resolve upstream license metadata or
authorize redistributing corpus text. No raw or processed QA/document text is committed.

## Verification recorded

On 2026-10-13, **51 synthetic tests passed**; strict mypy, Ruff lint, and formatting checks
passed. Real-data preparation was repeated with identical frozen outputs, versioned hashes
verified, and the original Phase 1 manifest/audit files had no Git diff. Both eligible
training and validation loaders were exercised; final-evaluation access was rejected as sealed.
The hosted CI workflow has not been observed running for this change.
