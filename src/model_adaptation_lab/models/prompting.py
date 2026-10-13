"""Versioned prompts with demonstrations loaded exclusively from eligible training."""

from pathlib import Path
from typing import Any

from model_adaptation_lab.data.splits import load_annotations
from model_adaptation_lab.io import LabError, canonical, digest

Message = dict[str, str]


def choose_demonstrations(
    manifests: Path, processed: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = load_annotations(manifests, processed, "train", purpose="demonstrations")
    chosen = []
    settings = config["demonstrations"]
    for answerable, count in ((True, settings["answerable"]), (False, settings["unanswerable"])):
        eligible = [
            r
            for r in rows
            if r["answerable"] == answerable
            and len(r["question"]) + len(r["answer"] or config["abstention"])
            <= settings["max_characters"]
        ]
        eligible.sort(key=lambda r: digest(f"{settings['seed']}:{r['question_id']}".encode()))
        if len(eligible) < count:
            raise LabError("Insufficient eligible training demonstrations; no fallback permitted")
        chosen.extend(eligible[:count])
    return chosen


def render_messages(
    question: str, config: dict[str, Any], demonstrations: list[dict[str, Any]]
) -> list[Message]:
    if not question.strip():
        raise LabError("Question must not be empty")
    messages = [{"role": "system", "content": config["system"]}]
    for row in demonstrations:
        if row.get("split") != "train" or row.get("source") != "original":
            raise LabError("Only eligible original training records may be demonstrations")
        messages.extend(
            [
                {"role": "user", "content": row["question"]},
                {
                    "role": "assistant",
                    "content": row["answer"] if row["answerable"] else config["abstention"],
                },
            ]
        )
    messages.append({"role": "user", "content": question})
    return messages


def prompt_hash(config: dict[str, Any], demonstrations: list[dict[str, Any]]) -> str:
    return digest(canonical({"config": config, "demonstrations": demonstrations}))
