"""Normalize and deterministically sample the pinned AI2 ARC dataset."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.tasks import Question

OPTION_LABELS = ("A", "B", "C", "D")


def source_collection(source_id: str) -> str:
    """Extract the upstream assessment collection from an ARC question identifier."""
    collection = source_id.split("_", maxsplit=1)[0].strip()
    if not collection:
        raise ValueError("ARC source id must contain a collection prefix")
    return collection


def normalize_arc_row(
    row: Mapping[str, object],
    *,
    revision: str,
    source_split: str,
) -> Question:
    """Normalize one four-choice ARC record into the authoritative question schema.

    Original choice labels may be letters or numbers. They are remapped by display order to
    ``A`` through ``D`` while the original labels and identifier remain in metadata.

    Raises:
        ValueError: If required fields are absent, malformed, or do not define four choices.
    """
    source_id = row.get("id")
    prompt = row.get("question")
    choices = row.get("choices")
    answer_key = row.get("answerKey")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("ARC row requires a non-empty string id")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"ARC row {source_id!r} requires a non-empty question")
    if not isinstance(choices, Mapping):
        raise ValueError(f"ARC row {source_id!r} requires a choices mapping")

    texts = choices.get("text")
    labels = choices.get("label")
    if not isinstance(texts, list) or not isinstance(labels, list):
        raise ValueError(f"ARC row {source_id!r} choices require text and label lists")
    if len(texts) != 4 or len(labels) != 4 or len(set(labels)) != 4:
        raise ValueError(f"ARC row {source_id!r} must contain four uniquely labelled choices")
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError(f"ARC row {source_id!r} contains an empty choice")
    if answer_key not in labels:
        raise ValueError(f"ARC row {source_id!r} answerKey is not a choice label")

    label_map = dict(zip(labels, OPTION_LABELS, strict=True))
    options = {
        normalized: text.strip()
        for normalized, text in zip(OPTION_LABELS, texts, strict=True)
    }
    collection = source_collection(source_id)
    return Question(
        question_id=f"arc_challenge:{source_split}:{source_id}",
        task_family="science_mcq",
        prompt=re.sub(r"\s+", " ", prompt).strip(),
        options=options,
        gold_answer=label_map[answer_key],
        difficulty="challenge",
        template_group=collection,
        source="allenai/ai2_arc",
        metadata={
            "source_id": source_id,
            "source_revision": revision,
            "source_config": "ARC-Challenge",
            "source_split": source_split,
            "source_collection": collection,
            "source_license": "CC-BY-SA-4.0",
            "original_choice_labels": labels,
        },
    )


def read_arc_parquet(
    path: str | Path,
    *,
    revision: str,
    source_split: str,
) -> tuple[list[Question], int]:
    """Read a pinned ARC parquet and return eligible questions plus rejected-row count.

    Rows outside the four-choice contract are counted and excluded. Other schema errors are
    surfaced rather than silently skipped.

    Raises:
        RuntimeError: If the optional PyArrow dependency is unavailable.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised without the data extra
        raise RuntimeError('ARC parquet loading requires `pip install -e ".[data]"`') from exc

    rows = pq.read_table(Path(path)).to_pylist()
    questions: list[Question] = []
    rejected = 0
    for row in rows:
        choices = row.get("choices")
        texts = choices.get("text") if isinstance(choices, Mapping) else None
        labels = choices.get("label") if isinstance(choices, Mapping) else None
        if not isinstance(texts, list) or not isinstance(labels, list):
            raise ValueError(f"ARC row {row.get('id')!r} has malformed choices")
        if len(texts) != 4 or len(labels) != 4 or row.get("answerKey") not in labels:
            rejected += 1
            continue
        questions.append(normalize_arc_row(row, revision=revision, source_split=source_split))
    return questions, rejected


def select_arc_questions(
    questions: Iterable[Question],
    *,
    n_questions: int,
    seed: int,
    min_source_group_size: int = 5,
) -> list[Question]:
    """Select a deterministic, collection-balanced ARC pilot sample.

    Eligible source collections are sorted, their rows are shuffled deterministically, and the
    sample is filled round-robin. This prevents the dominant Mercury collection from crowding out
    the diversity needed by the pilot gate.
    """
    if n_questions <= 0:
        raise ValueError("n_questions must be positive")
    if min_source_group_size <= 0:
        raise ValueError("min_source_group_size must be positive")

    grouped: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        if question.template_group is None:
            raise ValueError(f"question {question.question_id!r} has no source collection")
        grouped[question.template_group].append(question)
    eligible = {
        group: sorted(rows, key=lambda item: item.question_id)
        for group, rows in grouped.items()
        if len(rows) >= min_source_group_size
    }
    if sum(map(len, eligible.values())) < n_questions:
        raise ValueError("not enough eligible ARC questions for the requested freeze")

    rng = random.Random(seed)
    for group in sorted(eligible):
        rng.shuffle(eligible[group])
    selected: list[Question] = []
    offsets = Counter[str]()
    groups = sorted(eligible)
    while len(selected) < n_questions:
        progressed = False
        for group in groups:
            offset = offsets[group]
            if offset < len(eligible[group]):
                selected.append(eligible[group][offset])
                offsets[group] += 1
                progressed = True
                if len(selected) == n_questions:
                    break
        if not progressed:
            raise ValueError("not enough eligible ARC questions for the requested freeze")
    return sorted(selected, key=lambda item: item.question_id)


def validate_pilot_questions(questions: Iterable[Question]) -> dict[str, int]:
    """Validate freeze-level uniqueness and return counts by source collection."""
    rows = list(questions)
    if not rows:
        raise ValueError("pilot freeze must contain at least one question")
    ids = [row.question_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("pilot freeze contains duplicate question ids")
    prompts = [re.sub(r"\s+", " ", row.prompt).strip().casefold() for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("pilot freeze contains duplicate question text")
    expected_options = set(OPTION_LABELS)
    for row in rows:
        if set(row.options) != expected_options:
            raise ValueError(f"question {row.question_id!r} must use options A through D")
        if row.template_group is None:
            raise ValueError(f"question {row.question_id!r} has no source collection")
    counts = Counter(row.template_group for row in rows)
    if len(counts) < 2:
        raise ValueError("pilot freeze must span at least two source collections")
    return dict(sorted(counts.items()))
