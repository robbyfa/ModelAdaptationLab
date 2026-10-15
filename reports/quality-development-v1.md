# Development answer-quality measurement v1

**Status: implemented and exercised on real development outputs; independent human validation
is pending.** Software tests and execution checks do not establish measurement validity.
No correctness, groundedness, or unsupported-claim results are claimed without ratings.

This milestone changed no model weights, training configuration, answering prompts, retrieval
settings, or frozen manifests. The internal final holdout remains sealed. No additional
training experiment, prompt selection, retrieval tuning, or external LLM judging was performed.

## Measurement design

The versioned [rubric](quality-rubric-v1.md) and `configs/quality-v1.yaml` define:

| Dimension | Measurement | Availability boundary |
|---|---|---|
| Correctness | Human 0–3 reference-relative rubric | Null for absent references, failed/empty output, or unjudgeable cases |
| Reference support | Per factual claim: supported, contradicted, not supported, unverifiable | Same frozen corpus available to every strategy; actual reviewed documents declared |
| Retrieved support | Same claim labels against the exact supplied passages | Applicable only to RAG and LoRA + RAG |
| Unsupported claims | Contradicted/not-supported claims divided by assessed factual claims | Unknown claims excluded with explicit coverage counts; not a global falsehood rate |
| Abstention | Human answer/abstain/mixed/empty/failed plus evidence sufficiency | Annotation-scoped and evidence-scoped results remain separate |
| Citations | Claim-linked citation coverage and cited-passage support | Separate from correctness and reference support; non-RAG requirements are null |
| Similarity | Pinned MiniLM cosine and deterministic lexical diagnostics | No correctness threshold; not entailment or a substitute for review |

The importer checks exact quote text and offsets against frozen documents. Retrieved-evidence
quotes must fall within passages actually supplied to the model. This validates evidence
provenance, **not semantic entailment**: reviewers still need to check negation, versions,
quantities, prerequisites, product identity, and whether the evidence supports the claim.

Claim spans must cover the entire response without overlapping or omitting text. Reviewers
must split compound claims and label nonfactual spans; automatic initial paragraph spans
are editing suggestions. Supported/contradicted labels require quotations. Missing evidence
is unverifiable, not automatically unsupported. A response with no factual claims has null
support rates rather than a perfect score.

Common reference candidates are the annotated source document plus the union of documents
retrieved for that question, shared identically across all five answers. This is a review aid,
not exhaustive corpus verification. Additional frozen documentation may be reviewed and listed.
No question-specific candidate lists or annotations become retrieval content. An answer to an
annotation-unanswerable question may be supported outside the original annotation scope;
sufficiency disagreements are preserved for review rather than automatically called hallucinations.

Review packets use shuffled opaque item IDs and omit strategy labels. Their separate
`mapping.json` is for analysis only. Retrieval passages can reveal retrieval use, so blinding
is partial. Human identities and independence are declarations, not independently authenticated.

## Actual development execution

Twelve validation questions were selected using the existing deterministic balanced sampler
(seed 42): **six answerable, six annotation-unanswerable, twelve distinct overlap components**.
All five unchanged strategies produced outputs, for **60 outputs and zero generation failures**.
The development IDs overlap earlier smoke questions; this is a calibration pilot, not an unseen
performance set. The 60 answers are clustered within 12 questions and are not 60 independent
question observations.

Generation used the existing 256-token configuration, greedy decoding, Qwen MPS float32,
and the pinned base/tokenizer revision. The adapted strategies reused the selected rank-8
adapter from Phase 4, and both RAG strategies reused the existing full frozen index and top-k.
This was new inference solely to obtain evaluation material; no settings were optimized.

**Ten answers reached the token cap.** They remain in the packet and must be scored as delivered.
Reference cosine diagnostics were available for **25 of 30 answerable outputs**. Five inputs
exceeded MiniLM's 256-token limit and remain null without truncation. The other 30 outputs
have no annotated reference, so their reference similarity remains null. These counts describe
measurement coverage and execution; they are not correctness or groundedness findings.

Local artifacts (all source/question/answer text remains Git-ignored):

- `outputs/quality-dev-v1-base`: base, prompt, and RAG inference.
- `outputs/quality-dev-v1-adapted`: LoRA and LoRA + RAG inference.
- `outputs/quality-review-v1/cases.jsonl`: common-evidence, partially blinded review cases.
- `outputs/quality-review-v1/review.html`: escaped, offline readable cases and evidence.
- `outputs/quality-review-v1/review_template.json`: editable rating template.
- `outputs/quality-review-v1/semantic.json`: pinned diagnostic values and explicit missing reasons.
- `outputs/quality-review-v1/assessment-v1.json`: current coverage and validity status, with zero human reviewers.

MLflow run **`295a05bbaf1b4ddcb3291ebd8a776311`**, experiment `model-adaptation-lab-quality`,
uses local `outputs/mlflow.db`. Its FINISHED status means measurement execution finished,
not that the evaluator is scientifically validated. The validity tag explicitly remains
`not_yet_established`. Assessment, semantic diagnostics, packet/rubric/config files, original
inference outputs/code snapshots, and scorer code snapshots are preserved. Their stored hashes
were checked against local artifacts after upload. The [committed summary](quality-development-v1.json)
contains counts and provenance without source text, generated answers, or reference answers.

## Review and reproduce

Run from the repository root. The saved packet is immutable; make one editable copy of
`review_template.json` per reviewer in `outputs/`, and preserve originals when revising ratings.
Read `rubric.md` and `review.html` without opening `mapping.json`, semantic diagnostics, another
reviewer's ratings, or strategy output directories. Review all cases independently first.

Set `reviewer_id`, `kind: human`, and the independence declaration honestly. For each item,
complete behavior, correctness or its unavailable reason, evidence sufficiency, reviewed document
IDs, a rationale, and claim assessments. Split the initial spans where needed. Each evidence
quote is an object with `document_id`, `start`, `end`, and `quote`; offsets refer to the original
source body, not the rendered HTML. Use exact text and inspect surrounding context. If a quote
occurs multiple times, choose the occurrence actually reviewed rather than blindly using the
first match. The corpus is available through the hash-checked `load_corpus` loader; never open
raw QA files to find evidence.

```bash
# Optional reconstruction from the saved source runs into a NEW directory:
uv run model-lab quality prepare \
  --runs outputs/quality-dev-v1-base outputs/quality-dev-v1-adapted \
  --output-dir outputs/my-quality-packet
uv run model-lab quality render --packet outputs/my-quality-packet \
  --output outputs/my-quality-packet/review.html
uv run --extra retrieval model-lab quality semantic --packet outputs/my-quality-packet \
  --output outputs/my-quality-packet/semantic.json --device cpu --local-files-only

# After independent reviews, use a new score output for every assessment:
uv run model-lab quality score --packet outputs/quality-review-v1 \
  --reviews outputs/reviewer-a.json outputs/reviewer-b.json \
  --semantic outputs/quality-review-v1/semantic.json \
  --output outputs/quality-review-v1/reviewed-assessment.json
uv run --extra training model-lab quality track --packet outputs/quality-review-v1 \
  --scores outputs/quality-review-v1/reviewed-assessment.json \
  --output outputs/quality-review-v1/reviewed-tracking.json
```

Incomplete templates are rejected. Partial submissions can contain only completed items;
missing items remain explicit coverage gaps. Failed and empty generations retain separate
counts; null judgments are not quietly treated as zeros or successes. Scoring reports per-rater,
per-strategy and answerability-stratified macro means with denominators. It does not silently
average conflicting reviewers into a gold answer.

Agreement outputs include correctness exact agreement, absolute disagreement, linear-weighted
Cohen kappa, behavior/sufficiency agreement, and aligned claim/citation-label agreement.
Different claim segmentation is counted explicitly. Undefined coefficients are null. Semantic
Spearman correlation with human correctness is exploratory, with paired counts; no classifier
threshold or validity claim follows automatically. External LLM estimates require declared
model revision and prompt hash and remain separate from human ratings. No LLM judge is called
by these commands; an external judge runner and its full execution provenance are future work.

## Verification and the remaining validation gate

**113 CPU-only tests passed; one optional training-dependency test skipped.** With all local
dependencies, **114 tests passed**, with one existing expected warning from the tiny synthetic
PEFT fixture. Ruff formatting/lint and strict mypy passed. All fixtures are synthetic; none
contains real development or held-out content. New tests cover corpus/output tampering,
non-validation inputs, citation neutrality, fabricated/out-of-passage quotes, missing evidence,
claim coverage, failed/empty outputs, reviewer provenance, disagreement, undefined statistics,
semantic overflow, missing human review, and local tracking. Hosted CI has not been observed.

The software is ready for review, but **this milestone's human validation is not complete**.
Rubric v1 calls for two independently declared human reviewers on every successful output,
with at least six jointly reviewed questions per answerability stratum. These are pilot coverage
requirements, not statistically established guarantees. Even meeting them yields
`met_pending_adjudication`, not an automatic validated status.

Next, collect the independent ratings, inspect and adjudicate disagreements and critical-error
cases, and document whether the rubric separates correct answers from plausible unsupported
ones. Preserve reference defects and scope disagreements. If the rubric changes, version it
and validate on a separate development sample. A later validity report must justify readiness;
no final-release command or holdout access was added here. Model tuning, final benchmarking,
ablations, and portfolio claims remain outside this measurement milestone.
