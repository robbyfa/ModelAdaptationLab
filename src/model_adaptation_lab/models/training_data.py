"""Assistant-only supervision using the exact pinned chat template; no silent truncation."""

from typing import Any

from model_adaptation_lab.io import LabError, digest
from model_adaptation_lab.models.prompting import render_messages


def nested_subset(rows: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    if fraction not in {0.25, 0.5, 1.0}:
        raise LabError("Training fraction must be 0.25, 0.5, or 1.0")
    ordered = sorted(rows, key=lambda r: digest(f"{seed}:{r['question_id']}".encode()))
    return ordered[: max(1, int(len(ordered) * fraction))]


def encode_records(
    rows: list[dict[str, Any]], tokenizer: Any, prompts: dict[str, Any], max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded, included, excluded = [], [], []
    for row in rows:
        messages = render_messages(row["question"], prompts, [])
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        answer = row["answer"] if row["answerable"] else prompts["abstention"]
        if not isinstance(answer, str) or not answer.strip():
            raise LabError("Empty supervised answer")
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if full[: len(prompt)] != prompt or len(full) <= len(prompt):
            raise LabError("Chat template does not preserve an exact assistant-token boundary")
        if len(full) > max_length:
            excluded.append(
                {
                    "question_id": row["question_id"],
                    "tokens": len(full),
                    "reason": "sequence_overflow",
                }
            )
            continue
        encoded.append(
            {
                "input_ids": full,
                "attention_mask": [1] * len(full),
                "labels": [-100] * len(prompt) + full[len(prompt) :],
            }
        )
        included.append(row["question_id"])
    if not encoded:
        raise LabError("No examples remain within the sequence budget")
    return encoded, {
        "included_ids": included,
        "excluded": excluded,
        "supervised_tokens": sum(sum(t != -100 for t in e["labels"]) for e in encoded),
    }


class AnswerCollator:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.pad_id = tokenizer.pad_token_id
        self.torch = torch
        if self.pad_id is None:
            raise LabError("Tokenizer needs an explicit padding token")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(f["input_ids"]) for f in features)
        return {
            key: self.torch.tensor(
                [list(f[key]) + [pad] * (width - len(f[key])) for f in features],
                dtype=self.torch.long,
            )
            for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100))
        }
