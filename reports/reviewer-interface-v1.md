# Offline reviewer interface v1

The reviewer workflow is now an interactive, self-contained `review.html`. It renders one
response at a time, with the unchanged rubric available inside the page. No application server,
model, API credentials, JavaScript package installation, or internet connection is required by
the reviewer. The interface does not assign ratings or suggest support judgments.

## Start reviewing

Open your own bundle's `review.html` in a modern browser:

- Reviewer A: `outputs/quality-reviewer-a-ui-v1/review.html`
- Reviewer B: `outputs/quality-reviewer-b-ui-v1/review.html`

These fresh bundles contain the original 12 questions / 60 responses. Earlier static bundles
remain intact. The cases, rubric and configuration files are byte-identical to the original
packet. Its identity hash remains the identity used by `quality score`.

1. Enter your reviewer ID. Work independently; do not inspect another reviewer's ratings.
2. Read the question, generated response and available reference. Score correctness and
   behavior, evidence sufficiency, and explain your judgment. Missing judgments stay missing.
3. Split compound claims by placing the caret inside a claim's text and clicking **Split at
   caret**. **Merge with next** combines adjacent spans. These actions preserve all source text
   and clear affected judgments; no claim can be deleted to hide an inconvenient assertion.
4. Select a factual claim for evidence. Choose reference documents or exact retrieved passages.
   Search literal text within the selected document, or highlight it yourself. Declare reference
   documents actually reviewed, then attach selected text as a quotation. Exact Unicode character
   offsets are recorded automatically. Decide support yourself after reading surrounding context.
5. For applicable responses, link citations present in the answer and assess them separately.
   Non-RAG responses have no citation or retrieved-support rating requirement.
6. Click **Validate & mark complete**. Fix any listed issues. Subsequent edits clear completion.
7. Download draft backups regularly. **Export completed ratings** downloads only completed,
   locally validated responses; it does not upload anything. Partial exports are allowed and
   retain the existing pipeline's explicit coverage gaps.

Draft autosave is isolated by bundle location identity, packet hash, and reviewer ID. On reload,
enter the same ID to resume. Separate reviewers should use separate bundles/browser profiles.
This is a convenience boundary, not access control or verification of human independence.
Browser storage may be unavailable or cleared. **Download draft** and **Restore my draft** provide
portable backups, checked against reviewer and packet identity. Restore explicitly confirms
replacement of the current workspace. Keep prior exported files when revising ratings.

Only mark the independence checkbox when accurate. No independence declaration or correctness,
claim, evidence, or citation label is preselected. The JSON template remains available as a
manual fallback. Do not modify the hashed bundle files to store your work.

## Measurement boundaries

Document bodies are stored once in the HTML, with each response retaining its exact candidate
set and retrieval passages. Source selection and search aid navigation; they do not assert
relevance or entailment. Reviewed-document declarations and all judgments remain per response.
The interface presents the packet's shared reference candidates, not an exhaustive corpus search.
Additional frozen-corpus review still uses the existing guarded tooling/manual JSON workflow.

The generated page embeds neither strategy mappings nor similarity diagnostics. Model/document
text is escaped for script embedding and rendered through text nodes. A content security policy
blocks network connections and external resources. No remote analytics or libraries are loaded.
The rubric, annotation protocol, dataset, generation settings, trained adapter, retrieval settings,
and final-holdout access are unchanged. No inference or training experiment was run for this UI.

Client checks mirror the existing importer but do not replace it. Run the original command on
exported files:

```bash
uv run model-lab quality score --packet outputs/quality-review-v1 \
  --reviews outputs/reviewer-a-ratings.json outputs/reviewer-b-ratings.json \
  --semantic outputs/quality-review-v1/semantic.json \
  --output outputs/quality-review-v1/independent-assessment.json
```

The interface changes do not establish human measurement validity. Independent reviews,
agreement analysis, disagreement adjudication and a documented validity decision remain pending.

## Verification

- **119 tests passed** with local dependencies; **118 passed / 1 optional skip** in the CPU-only
  environment. Ruff formatting/lint and strict mypy passed. Hosted CI has not been observed.
- Synthetic contract tests compare browser validation to the Python validator across all five
  strategies and invalid ratings, and check Unicode claim edits, judgment resets, script-safe
  model text, and bundle provenance. Node is included in CI for these dependency-free JS checks.
- Chromium interactions checked exact quote selection/attachment, a completed rating export
  accepted by `load_review`, claim split/merge, autosave reload, draft download/restore, mobile
  width, and navigation/export after networking was disabled. All test ratings are synthetic,
  explicitly named as such, and excluded from research tracking.
- Browser automation used a localhost synthetic fixture because the automation tool disallows
  `file:` navigation. The delivered page is self-contained; direct-file browser storage behavior
  may vary, which is why downloadable drafts are provided.
- The wheel build contains the HTML/JavaScript assets; future code snapshots include them.
- A Unicode line-separator bug in the packet JSONL reader was fixed: records are now separated
  by actual file lines instead of splitting valid embedded Unicode separators.

Regenerate into a new directory without overwriting earlier bundles:

```bash
uv run model-lab quality bundle --packet outputs/quality-review-v1 \
  --output-dir outputs/new-reviewer-ui-bundle
```
