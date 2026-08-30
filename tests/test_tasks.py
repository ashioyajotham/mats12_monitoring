import pytest

from src.tasks import (
    Question,
    assert_no_group_leakage,
    grouped_split,
    read_jsonl,
    read_jsonl_objects,
    write_jsonl,
)


def test_question_requires_valid_gold_answer():
    with pytest.raises(ValueError):
        Question(
            question_id="q1",
            task_family="test",
            prompt="Question?",
            options={"A": "one", "B": "two"},
            gold_answer="C",
        )


def test_grouped_split_is_deterministic_and_complete():
    ids = [f"q{i}" for i in range(20)]
    first = grouped_split(ids, seed=3)
    second = grouped_split(reversed(ids), seed=3)
    assert first == second
    assert set(first) == set(ids)
    assert set(first.values()) == {"train", "validation", "test"}


def test_group_leakage_is_rejected():
    with pytest.raises(ValueError, match="multiple splits"):
        assert_no_group_leakage(
            [{"question_id": "q1", "split": "train"}, {"question_id": "q1", "split": "test"}]
        )


def test_jsonl_round_trip_is_validated_and_immutable(tmp_path):
    path = tmp_path / "questions.jsonl"
    question = Question(
        question_id="q1",
        task_family="test",
        prompt="Question?",
        options={"A": "one", "B": "two"},
        gold_answer="A",
    )
    write_jsonl(path, [question])
    assert read_jsonl(path) == [question]
    with pytest.raises(FileExistsError):
        write_jsonl(path, [question])


def test_jsonl_reader_preserves_unicode_line_separator_inside_record(tmp_path):
    """Provider text may contain U+2028 without ending the physical JSONL record."""
    path = tmp_path / "unicode-lines.jsonl"
    question = Question(
        question_id="q-unicode",
        task_family="test",
        prompt="first\u2028second",
        options={"A": "one", "B": "two"},
        gold_answer="A",
    )
    path.write_text(question.model_dump_json() + "\n", encoding="utf-8")
    assert read_jsonl(path) == [question]


def test_dictionary_jsonl_reader_preserves_unicode_line_separator(tmp_path):
    """Unmodelled JSONL artifacts obey the same physical-line rule."""
    path = tmp_path / "objects.jsonl"
    path.write_text('{"text":"first\u2028second"}\n', encoding="utf-8")
    assert read_jsonl_objects(path) == [{"text": "first\u2028second"}]
