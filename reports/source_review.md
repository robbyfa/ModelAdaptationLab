# Source review and methodological decisions

**Historical Phase 1 source review.** Eligibility decisions below are now resolved by
[protocol v1](protocol-v1.md); the original snapshot remains blocked and unchanged.

Reviewed on 2026-10-13. These are observations from downloaded source files, not model results.

## Verified sources

| Source | Pinned revision | Role |
|---|---|---|
| [IBM/techqa](https://github.com/IBM/techqa/tree/f0cf8ce11c6ef778c6bc064ee6c1d9b3eca76faf) | `f0cf8ce11c6ef778c6bc064ee6c1d9b3eca76faf` | Baseline schema implementation and archive pointer |
| [PrimeQA/TechQA](https://huggingface.co/datasets/PrimeQA/TechQA/tree/60437bc79ab217679682217598a3693cab78365b) | `60437bc79ab217679682217598a3693cab78365b` | Original archive, annotations, source documents, terms |
| [NVIDIA derivative](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/tree/0b5bbc84b7f07d6d09d063130e90b716d8d4a32a) | `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a` | Separate inspection; never training |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Future shared base and tokenizer; Apache-2.0 |

The official archive is 2,959,973,525 bytes. Its SHA-256 matches the upstream LFS
object hash in `configs/dataset.yaml`. That config also pins each extracted file's
hash and NVIDIA's JSON hash, so a local replacement cannot silently inherit upstream provenance.
Model metadata was inspected; model weights were not downloaded.

## Actual schemas

The official QA files are JSON arrays. Fields `QUESTION_ID`, `QUESTION_TITLE`,
`QUESTION_TEXT`, `ANSWERABLE`, `DOCUMENT`, `ANSWER`, `START_OFFSET`, and `END_OFFSET`
are strings; `DOC_IDS` is a list of strings. Answerable offsets are decimal **strings**,
converted explicitly to integer character offsets. Unanswerable records use `N` and
`-` for answer/document/offset fields. Original answerability means an annotator found
an answer within the supplied candidate documents, not necessarily anywhere in the full corpus.

The archive README says `QUESTION_BODY` in one section; actual files and IBM's baseline
use `QUESTION_TEXT`. The adapter validates either spelling, preferring TEXT. Empty answers
are structurally strings but are reported as annotation defects, not silently imputed.

The original document subset is an object keyed by document ID. Each record includes
`id`, `title`, `text`, `content` (HTML), and `metadata`, including `canonicalUrl`.
Processed corpus records contain only document ID, title, body, source URL, and provenance.
They contain no question, answer, gold span, or question-specific candidate list.

NVIDIA's JSON is an array of objects with `id`, `question`, `answer` (strings),
`is_impossible` (boolean), and `contexts` (list of `{filename, text}`). Its card says
908 examples; the pinned file actually contains 910. Matching IDs, normalized questions,
answers, and answerability establish that it includes the complete official train/dev data.
All 610 provided context instances contain the matching original document body; there are
496 unique context IDs. This is a gold-associated subset, not an independently selected
retrieval collection. The derivative is not a source for training or index construction.

## Licensing and document availability

IBM's code repository uses Apache-2.0. The PrimeQA hub card declares Apache-2.0 but its
archive README explicitly assigns **CDLA-Permissive-1.0** to the QA dataset and separately
to the technote corpus. The archive includes the license PDF. Local copies of the README
and PDF remain alongside the data; their checksums are recorded. NVIDIA's card declares
Apache-2.0 for its derivative. These source statements are recorded separately; the code
repository's license is not assumed to cover the original corpus.

The original documents are available in the verified archive: no generated evidence or
third-party substitute was needed. Phase 1 audits only `training_dev_technotes.json`, not
all 801,998 technotes described in the README. The archive's full-corpus files were streamed
past during selective extraction. Resolve the license metadata discrepancy before
redistribution; no license is newly assigned to upstream data by this project.

## Freeze and access policy

- Preserve official `training_Q_A.json` as train and `dev_Q_A.json` as validation.
- Store IDs and normalized-record SHA-256 hashes in sorted assignment manifests.
- Store file hashes and readiness flags in `freeze.json`; refuse conflicting re-preparation.
- Preserve every record, including overlaps and malformed references, in this source snapshot.
- Block training, demonstration selection, and development selection through `load_split`
  while exact cross-split overlaps or span mismatches remain unresolved.
- Near duplicates use exhaustive NFKC/casefold word-set Jaccard with threshold 0.85.
  They are candidates for manual review, not proof of identical intent. Answer overlap
  excludes unanswerable sentinels. Document duplication uses normalized body hashes.
- Shared corpus documents are permitted under a future defined open-book protocol.
  Candidate IDs and gold annotations must never be used as oracle retrieval inputs.
- `test.jsonl` is intentionally empty and `final_test_ready` is false. Loading it for
  final evaluation fails. A frozen assignment snapshot is not a finalized benchmark.

The integrity mechanism detects accidental drift relative to the committed lock/config;
it is not a security boundary against someone deliberately modifying both data and hashes.
`load_annotations` checks processed record hashes against assignment manifests before returning records.
Future consumers must use this access layer.
No final test questions were designated or inspected. The archive's `validation_*` files
were listed but not ingested; the README describes them as leaderboard plumbing checks,
not a scientifically independent final test set.

## Decisions required before experiments

1. **Overlap:** choose and version a decontamination policy without quietly changing official
   results. A conservative candidate is to exclude training examples overlapping development
   (including reviewed near duplicates), report both original counts and eligible counts,
   and never move development examples into training.
2. **References:** adjudicate the 11 development span discrepancies, including `DEV_Q014`
   and `DEV_Q094` with empty answers. Do not automatically reconstruct labels from offsets;
   preserve original and corrected versions with an explicit reason if corrections are chosen.
3. **Final test:** obtain an authorized independent test set, or pre-register a new held-out
   protocol before model development. Do not call a reused development set an untouched test.
   Any repurposing must acknowledge this Phase 1 audit exposure and invalidate prior tuning.
4. **Corpus:** decide whether the 28,482-document subset or full corpus defines open-book
   retrieval. Freeze the corpus independently; report that the subset was selected upstream
   using train/dev candidate-document annotations and is not representative of full-corpus difficulty.
5. **Terms:** retain CDLA terms and document resolution of the inconsistent dataset-card metadata.

No automatic decontamination or semantic label repair is performed in Phase 1. These choices
change the experiment itself and must be explicit, versioned, and decided before selection.
