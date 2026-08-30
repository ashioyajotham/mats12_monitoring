"""Summarize a clean MATH discovery cohort without rewriting raw evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.generate_rollouts import Rollout, RolloutStatus
from src.math_answers import MathGrade, grade_math_answer
from src.tasks import MathProblem, read_jsonl


def main() -> None:
    """Grade scorable rollouts and evaluate the preregistered discovery gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    questions = {
        row.question_id: row
        for row in read_jsonl(args.questions, model=MathProblem)
        if isinstance(row, MathProblem)
    }
    grades: Counter[str] = Counter()
    incorrect_subjects: set[str] = set()
    incorrect_questions: set[str] = set()
    for rollout in rollouts:
        if rollout.status is not RolloutStatus.CLEAN_STOP:
            continue
        grade = grade_math_answer(rollout.parsed_answer, rollout.gold_answer)
        grades[str(grade)] += 1
        if grade is MathGrade.INCORRECT:
            incorrect_questions.add(rollout.question_id)
            incorrect_subjects.add(questions[rollout.question_id].template_group or "unknown")

    statuses = Counter(str(rollout.status) for rollout in rollouts)
    truncation_rate = statuses[RolloutStatus.LENGTH_TRUNCATED] / len(rollouts) if rollouts else 0.0
    gate_passed = (
        grades[MathGrade.INCORRECT] >= 6
        and len(incorrect_subjects) >= 2
        and truncation_rate <= 0.20
    )
    summary = {
        "run": str(args.run),
        "stored_rollouts": len(rollouts),
        "statuses": dict(sorted(statuses.items())),
        "grades": dict(sorted(grades.items())),
        "incorrect_questions": sorted(incorrect_questions),
        "incorrect_subjects": sorted(incorrect_subjects),
        "truncation_rate": truncation_rate,
        "natural_error_gate_passed": gate_passed,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
