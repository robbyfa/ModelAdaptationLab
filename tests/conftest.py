"""Exclusively synthetic fixtures: no IBM/NVIDIA question or document text."""

import json
from pathlib import Path
from typing import Any

import pytest


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
