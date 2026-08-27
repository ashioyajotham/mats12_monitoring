import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.datasets.arc import (
    normalize_arc_row,
    read_arc_parquet,
    select_arc_questions,
    validate_pilot_questions,
)
from src.tasks import Question

REVISION = "210d026faf9955653af8916fad021475a3f00453"


def arc_row(source_id="Mercury_1", labels=None, answer="2"):
    labels = labels or ["1", "2", "3", "4"]
    return {
        "id": source_id,
        "question": "  Which option is correct?  ",
        "choices": {"text": ["one", "two", "three", "four"], "label": labels},
        "answerKey": answer,
    }


def question(group: str, index: int) -> Question:
    return Question(
        question_id=f"arc:{group}:{index}",
        task_family="science_mcq",
        prompt=f"Question {group} {index}?",
        options={"A": "one", "B": "two", "C": "three", "D": "four"},
        gold_answer="A",
        difficulty="challenge",
        template_group=group,
        source="allenai/ai2_arc",
    )


def test_arc_normalization_remaps_numeric_labels_and_records_provenance():
    result = normalize_arc_row(arc_row(), revision=REVISION, source_split="validation")
    assert result.gold_answer == "B"
    assert result.options == {"A": "one", "B": "two", "C": "three", "D": "four"}
    assert result.prompt == "Which option is correct?"
    assert result.template_group == "Mercury"
    assert result.metadata["original_choice_labels"] == ["1", "2", "3", "4"]


def test_arc_normalization_rejects_non_four_choice_rows():
    row = arc_row(labels=["A", "B", "C"], answer="A")
    row["choices"]["text"] = ["one", "two", "three"]
    with pytest.raises(ValueError, match="four uniquely labelled"):
        normalize_arc_row(row, revision=REVISION, source_split="validation")


def test_arc_parquet_reader_counts_ineligible_rows(tmp_path):
    rows = [arc_row("Mercury_1"), arc_row("Mercury_2", labels=["A", "B", "C"], answer="A")]
    rows[1]["choices"]["text"] = ["one", "two", "three"]
    path = tmp_path / "arc.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    questions, rejected = read_arc_parquet(
        path, revision=REVISION, source_split="validation"
    )
    assert len(questions) == 1
    assert rejected == 1


def test_selection_is_deterministic_and_balanced_across_eligible_groups():
    pool = [question(group, index) for group in ("Alpha", "Beta", "Gamma") for index in range(5)]
    first = select_arc_questions(pool, n_questions=6, seed=17, min_source_group_size=5)
    second = select_arc_questions(reversed(pool), n_questions=6, seed=17, min_source_group_size=5)
    assert first == second
    assert validate_pilot_questions(first) == {"Alpha": 2, "Beta": 2, "Gamma": 2}


def test_freeze_validation_rejects_duplicate_prompts():
    first = question("Alpha", 1)
    duplicate = question("Beta", 2).model_copy(update={"prompt": first.prompt})
    with pytest.raises(ValueError, match="duplicate question text"):
        validate_pilot_questions([first, duplicate])
