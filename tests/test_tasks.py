import pytest

from src.tasks import Question, assert_no_group_leakage, grouped_split, read_jsonl, write_jsonl


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
