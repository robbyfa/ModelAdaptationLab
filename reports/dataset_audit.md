# TechQA dataset audit

Phase 1 only; no training or performance evaluation

Generated from pinned source files; no question or answer text is printed.

| Official role | Rows | Answerable | Unanswerable |
|---|---:|---:|---:|
| train | 600 | 450 | 150 |
| validation | 310 | 160 | 150 |
| test | 0 | 0 | 0 |

## Provenance and evidence

- Official archive checksum verified: True.
- Original document subset: 28482 documents.
- Gold-document coverage: 1.0.
- Exact answer-span matches: 599.
- NVIDIA derivative remains inspection-only, regardless of its `train` label.
- Derivative rows: 910; card declares 908.
- Derivative ID matches by official split: {'train': 600, 'validation': 310}.

## Overlap checks

- Exact normalized question groups: 32.
- Cross-split exact groups: 23.
- Near-question pairs: 86 (word-set Jaccard ≥ 0.85).
- Identical normalized answer groups: 34.
- Shared documents are allowed only under the documented open-book protocol.
- Near duplicates are review candidates, not proven contamination. No claim of leakage-free splits is made.

## Limitations and decisions

- Reference answers disagree with source offsets; experimental use blocked pending annotation policy (no automatic repairs)
- Exact questions cross official splits; training use blocked pending review
- No official final test set imported; final evaluation is blocked pending a written protocol
- Archive declares CDLA-Permissive-1.0; hub card declares Apache-2.0. Preserve archive terms; resolve the metadata discrepancy before redistribution
- Full 801,998-technote collection is not indexed or exhaustively audited in Phase 1

## Sources

- [IBM baseline repository](https://github.com/IBM/techqa)
- [Original archive](https://huggingface.co/datasets/PrimeQA/TechQA)
- [NVIDIA derivative](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)

Exact revisions, input hashes, record IDs, span mismatches, source coverage, and document-overlap lists are in `dataset_audit.json`.
