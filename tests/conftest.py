"""Exclusively synthetic fixtures: no IBM/NVIDIA question or document text."""

import json
from pathlib import Path
from typing import Any

import pytest

from model_adaptation_lab.data.ingest import Document, Question
from model_adaptation_lab.data.protocol import POLICY, prepare_protocol
from model_adaptation_lab.data.splits import freeze
from model_adaptation_lab.io import file_hash


@pytest.fixture
def row() -> dict[str, Any]:
    return {
        "QUESTION_ID": "SYNTH_TRAIN_1",
        "QUESTION_TITLE": "Synthetic connection error",
        "QUESTION_TEXT": "How do I reset the demo connection?",
        "ANSWERABLE": "Y",
        "ANSWER": "Reset it.",
        "DOCUMENT": "synthetic-doc",
        "DOC_IDS": ["synthetic-doc"],
        "START_OFFSET": "0",
        "END_OFFSET": "9",
    }


@pytest.fixture
def write_json(tmp_path: Path):
    def write(value: Any, name: str = "input.json") -> Path:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    return write


@pytest.fixture
def synthetic_protocol(tmp_path):
    rows, docs = [], {}
    for i in range(14):
        qid = f"SYNTH_{i:02}"
        answerable = i < 8 or i == 12
        answer = f"response{i}" if answerable else None
        split = "train" if i < 12 else "validation"
        docs[qid] = Document(qid, "Synthetic only", answer or "irrelevant", None)
        rows.append(
            Question(
                qid,
                f"syntheticrequest{i}",
                answer,
                answerable,
                qid if answerable else None,
                (qid,),
                "original",
                split,
                0 if answer else None,
                len(answer) if answer else None,
            )
        )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps({k: {"id": k, "title": d.title, "text": d.text} for k, d in docs.items()})
    )
    audit = {
        "splits": {"train": {"rows": 12}, "validation": {"rows": 2}},
        "cross_split_exact_question_groups": [],
        "evidence": {"status": "available"},
        "source_files_verified": True,
        "dataset_config_sha256": "c" * 64,
        "dataset_config": {"original": {"revision": "a" * 40}},
        "sources": {"training_dev_technotes.json": {"sha256": file_hash(corpus)}},
    }
    source, manifests, processed = tmp_path / "source", tmp_path / "v1", tmp_path / "processed"
    freeze(rows, audit, source)
    prepare_protocol(rows, corpus, POLICY, source, manifests, processed, "c" * 64)
    return manifests, processed
