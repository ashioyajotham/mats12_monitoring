"""Freeze difficult calibration questions from clean rollout telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import git_revision, sha256_file, utc_now
from src.calibration import rank_by_clean_deliberation
from src.generate_rollouts import Rollout, write_manifest
from src.tasks import Question, read_jsonl, write_jsonl


def main() -> None:
    """Select and write the highest-deliberation questions without overwriting artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    questions = {
        row.question_id: row
        for row in read_jsonl(args.questions, model=Question)
        if isinstance(row, Question)
    }
    rollouts = [
        Rollout.model_validate_json(line)
        for line in Path(args.rollouts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ranking = rank_by_clean_deliberation(rollouts)
    if len(ranking) < args.count:
        raise SystemExit(f"only {len(ranking)} eligible questions; need {args.count}")
    selected_ranking = ranking[: args.count]
    missing = [
        question_id
        for question_id, _, _ in selected_ranking
        if question_id not in questions
    ]
    if missing:
        raise SystemExit(f"selected question missing from source freeze: {missing[0]}")

    output = Path(args.output)
    selected = [questions[question_id] for question_id, _, _ in selected_ranking]
    write_jsonl(output, selected)
    manifest_path = output.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        {
            "purpose": "intervention_calibration_question_freeze",
            "created_at": utc_now(),
            "code_revision": git_revision(),
            "selection_rule": "top mean clean completion tokens; correct stopped samples only",
            "requested_count": args.count,
            "source_questions": {"path": args.questions, "sha256": sha256_file(args.questions)},
            "source_rollouts": {"path": args.rollouts, "sha256": sha256_file(args.rollouts)},
            "ranking": [
                {"question_id": question_id, "mean_completion_tokens": mean, "samples": count}
                for question_id, mean, count in selected_ranking
            ],
            "output": {"path": str(output), "sha256": sha256_file(output)},
        },
    )
    print(json.dumps({"output": str(output), "selected": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
