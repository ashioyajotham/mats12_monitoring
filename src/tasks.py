"""Question schemas, JSONL I/O, and leakage-safe grouped splits."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Question(BaseModel):
    """Normalized multiple-choice question and its provenance metadata."""
    question_id: str
    task_family: str
    prompt: str
    options: dict[str, str] = Field(min_length=2)
    gold_answer: str
    difficulty: str | None = None
    template_group: str | None = None
    source: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def answer_is_an_option(self) -> Question:
        """Require the gold answer to name one of the available options."""
        if self.gold_answer not in self.options:
            raise ValueError("gold_answer must be present in options")
        return self


class MathProblem(BaseModel):
    """Normalized free-response mathematics problem."""
    question_id: str
    task_family: str = "math"
    prompt: str
    gold_answer: str
    difficulty: str | None = None
    template_group: str | None = None
    source: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def answer_is_nonempty(self) -> MathProblem:
        """Require a canonical, non-empty reference answer."""
        if not self.gold_answer.strip():
            raise ValueError("gold_answer must be non-empty")
        return self


def read_jsonl(path: str | Path, model: type[BaseModel] = Question) -> list[BaseModel]:
    """Read non-empty JSONL records and validate each against a Pydantic model.

    Raises:
        ValueError: If a record is invalid. The error includes its source line number.
    """
    records: list[BaseModel] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(model.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> None:
    """Create a JSONL artifact without overwriting an existing file.

    Raises:
        FileExistsError: If ``path`` already exists.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def grouped_split(
    question_ids: Iterable[str],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 17,
) -> dict[str, str]:
    """Assign stable question groups to train/validation/test without rollout leakage."""
    if len(fractions) != 3 or any(value < 0 for value in fractions):
        raise ValueError("fractions must contain three non-negative values")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to one")

    unique_ids = sorted(set(question_ids))
    random.Random(seed).shuffle(unique_ids)
    n_total = len(unique_ids)
    train_end = round(n_total * fractions[0])
    val_end = train_end + round(n_total * fractions[1])
    names = (["train"] * train_end) + (["validation"] * (val_end - train_end))
    names += ["test"] * (n_total - val_end)
    return dict(zip(unique_ids, names, strict=True))


def assert_no_group_leakage(rows: Iterable[dict]) -> None:
    """Raise when one question identifier appears in more than one split."""
    seen: dict[str, str] = {}
    for row in rows:
        question_id, split = str(row["question_id"]), str(row["split"])
        if question_id in seen and seen[question_id] != split:
            raise ValueError(f"question_id {question_id!r} appears in multiple splits")
        seen[question_id] = split
