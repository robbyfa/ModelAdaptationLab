# Answer quality rubric v1

This rubric is for measurement on development data. It changes no model, prompt, retrieval
setting, dataset label, or final-holdout access. Scores must cite their method: human review,
LLM estimate, or deterministic diagnostic. A model-generated assessment is never a human label.

## Correctness (same rules for every strategy)

Judge whether the response answers the question relative to the supplied reference. Ignore
whether citations are present. Preserve critical product/version names, preconditions,
negation, quantities, commands, and causal direction. Paraphrases can receive full credit;
lexical overlap alone cannot establish correctness.

- **3:** Correct and sufficiently complete; all necessary actions, conditions, and qualifications
  are preserved, with no material contradiction or incorrect additional advice.
- **2:** Mostly correct and useful, but omits a secondary qualification or has a minor error.
- **1:** Relevant fragments, but missing a necessary step/condition or containing a material error.
- **0:** Wrong, irrelevant, or abstaining when a reference answer is available.
- **null:** Failed/empty response, absent reference, or genuinely unjudgeable. Supply a reason.

Truncated answers are scored as delivered and flagged; do not invent the missing completion.
Record suspected reference defects as disagreements, never silently rewrite a reference.
Annotation-unanswerable cases have null reference correctness, not an automatic zero.

## Evidence support and unsupported claims

Split the response into atomic factual claims (including procedural advice) and nonfactual
text. The initial sentence spans are suggestions; edit/split them where necessary. Cover all
non-whitespace response text, preserve exact offsets/text, and do not omit inconvenient claims.
Classify each factual claim independently against two evidence scopes:

1. **Reference evidence:** the same frozen source corpus is available for every strategy.
   The packet presents the annotated source document plus the union of retrieved documents
   for that question, identically for all five answers. These are review candidates, not an
   exhaustive search of all 28,482 documents. List every document actually reviewed; additional
   frozen documents may be inspected. No citation is required for a non-RAG answer to be supported.
2. **Retrieved evidence:** only the exact passages actually supplied to that RAG generation.
   This scope is not applicable to non-RAG answers and must remain null, not zero.

Statuses are **supported**, **contradicted**, **not_supported**, or **unverifiable**.
Supported and contradicted decisions require exact source quotations with document IDs and
character offsets. Interpret quotations with their surrounding context: lexical overlap,
negation changes, different versions, or a similarly named product do not prove entailment.
`not_supported` means no support in the explicitly reviewed evidence, not a claim of global
absence or falsehood. Use `unverifiable` for ambiguous claims, insufficient review, or missing
evidence. An annotation-unanswerable question can still have a corpus-supported answer;
record this as a label-scope disagreement needing review, not automatic hallucination.

Report reference and retrieved support separately. Unsupported-claim rate is the fraction
of assessed factual claims marked contradicted or not_supported; unverifiable claims are
excluded from its denominator and counted explicitly. Report assessed/total coverage. An
answer without factual claims has null claim-support rates, not a perfect score. Absence of
citations never lowers the reference-support score.

## Abstention and evidence sufficiency

Classify behavior as answer, abstain, mixed, empty, or failed. A mixed response includes both
uncertainty/refusal and substantive advice. Exact-marker detection is only a diagnostic.
Separately judge whether reference evidence and (for RAG) retrieved evidence are sufficient,
insufficient, or uncertain to answer the question. Sufficiency is question-relative, not
whether the generated answer sounds plausible.

Report annotation-scoped abstention separately from evidence-scoped abstention. Report
unnecessary abstention on answerable annotations and on sufficient reviewed evidence. Preserve
mixed responses and uncertain sufficiency rather than forcing them into successful abstentions.
Do not equate annotation-unanswerable with globally impossible to answer.

## Citations (separate from correctness and groundedness)

For each factual RAG claim, link cited document IDs to the claim and verify the cited passage
supports it. A known ID alone is insufficient. Cite-supported claims require quotations from
those exact retrieved passages. Use missing, supported, contradicted, not_supported, or
unverifiable. Citation coverage and assessed citation-support rate have separate denominators.
Non-RAG citation requirements are not applicable. Citation syntax/ID checks are diagnostics;
ordinary bracketed code can be misidentified and must be reviewed.

## Review procedure and validation boundary

Use blinded item IDs; do not consult strategy mapping or another reviewer's answers while
rating. Retrieval passages can reveal the presence of RAG, so blinding is partial. Review
independently before adjudicating disagreements. Keep all original ratings and rationales.

The initial development sample has twelve questions (six per annotation-answerability stratum),
with outputs from all five unchanged strategies. This is a pilot for rubric/evaluator validity,
not a representative performance benchmark. Human-review coverage requires two independently
declared human reviewers on every successful output, with at least six jointly reviewed
questions per stratum. This is an operational coverage check, not proof of validity.

Report exact agreement, ordinal mean absolute disagreement and linear-weighted Cohen kappa
for correctness, and agreement for behavior, sufficiency and aligned claim statuses. Unequal
claim segmentation is reported as nonaligned; it is not discarded without a count. Zero
variance or missing pairs yield null coefficients. Agreement does not establish truth.
LLM-vs-human comparisons are labeled estimates and remain separate from human-vs-human.

Semantic cosine similarity and normalized string/token overlap are auxiliary diagnostics.
They can miss negation, wrong numbers and unsupported additions; there is no correctness
threshold in v1. Long inputs are marked unavailable rather than silently truncated. No LLM
judge is required or silently invoked; external estimates must declare model revision and
prompt hash. No automated semantic metric is considered validated by passing software tests.

Before calling the evaluator validated, inspect human disagreements, reference defects,
claim coverage and critical-error cases, document adjudication, and validate any revised rubric
on a separate development sample. Version any rubric changes; do not overwrite this packet or
reuse final-holdout outputs for calibration. Final release remains blocked in this milestone.
