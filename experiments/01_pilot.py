"""Validate the full pipeline with deterministic mock data (not research evidence)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src import __version__
from src.audit import (
    canonical_config_hash,
    git_revision,
    load_config,
    runtime_environment,
    sha256_file,
    utc_now,
)
from src.causal_labels import CausalLabel, derive_label
from src.generate_rollouts import MockBackend, collect_rollout, write_manifest
from src.hints import Condition, build_variant
from src.resampling import estimate_answer_shifts
from src.tasks import Question, write_jsonl


def mock_questions(n: int) -> list[Question]:
    """Create deterministic synthetic questions for pipeline validation only."""
    return [
        Question(
            question_id=f"mock_{index:04d}",
            task_family="pipeline_test",
            prompt=f"Which option is designated correct for mock item {index}?",
            options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            gold_answer=("A", "B", "C", "D")[index % 4],
            template_group=f"mock_group_{index % 3}",
            source="synthetic_plumbing_only",
        )
        for index in range(n)
    ]


def main() -> None:
    """Run the complete data and causal-label pipeline with a mock backend."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["generation"]["backend"] != "mock":
        raise SystemExit("01_pilot currently validates plumbing only; set generation.backend=mock")

    started_at = datetime.now(UTC)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config["paths"]["generated_dir"]) / f"mock_pilot_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    questions = mock_questions(config["data"]["n_questions"])
    rollouts = []
    generation = config["generation"]

    for question_index, question in enumerate(questions):
        hinted_option = next(key for key in sorted(question.options) if key != question.gold_answer)
        for condition_name in config["conditions"]:
            condition = Condition(condition_name)
            variant = build_variant(question, condition, hinted_option=hinted_option)
            backend = MockBackend(
                list(question.options), question.gold_answer, variant.hinted_option
            )
            for sample_index in range(generation["samples_per_condition"]):
                seed = generation["base_seed"] + question_index * 100 + sample_index
                rollouts.append(
                    collect_rollout(
                        question,
                        variant,
                        backend,
                        model=generation["model"],
                        seed=seed,
                        temperature=generation["temperature"],
                        top_p=generation["top_p"],
                        max_new_tokens=generation["max_new_tokens"],
                    )
                )

    shifts = estimate_answer_shifts(rollouts)
    shift_by_question = {item.question_id: item for item in shifts}
    labelled = [
        derive_label(
            rollout,
            shift_by_question.get(rollout.question_id),
            min_hint_effect=config["labels"]["min_hint_effect"],
            patterns=config["labels"]["acknowledgment_patterns"],
            require_positive_lower_bound=config["labels"]["require_positive_lower_bound"],
        )
        for rollout in rollouts
    ]
    write_jsonl(output_dir / "rollouts.jsonl", rollouts)
    write_jsonl(output_dir / "answer_shifts.jsonl", shifts)
    write_jsonl(output_dir / "candidate_labels.jsonl", labelled)
    artifact_paths = {
        "rollouts": output_dir / "rollouts.jsonl",
        "answer_shifts": output_dir / "answer_shifts.jsonl",
        "candidate_labels": output_dir / "candidate_labels.jsonl",
    }
    invalid_count = sum(rollout.parsed_answer is None for rollout in rollouts)
    write_manifest(
        output_dir / "manifest.json",
        {
            "purpose": "pipeline_test_only",
            "config_path": str(Path(args.config)),
            "config_sha256": canonical_config_hash(config),
            "model": {
                "identifier": generation["model"],
                "revision": generation.get("model_revision"),
            },
            "backend": {"name": generation["backend"], "version": f"mock_v{__version__}"},
            "generation_parameters": {
                key: generation[key]
                for key in ("samples_per_condition", "temperature", "top_p", "max_new_tokens")
            },
            "source": {
                "name": config["data"]["source"],
                "revision": config["data"].get("source_revision"),
                "license": config["data"].get("source_license"),
                "selection_rule": "deterministic synthetic plumbing questions",
            },
            "code_revision": git_revision(),
            "started_at": started_at.isoformat(),
            "completed_at": utc_now(),
            "runtime_environment": runtime_environment(),
            "counts": {
                "questions": len(questions),
                "requested": len(questions)
                * len(config["conditions"])
                * generation["samples_per_condition"],
                "completed": len(rollouts),
                "invalid": invalid_count,
                "excluded": 0,
            },
            "output_sha256": {
                name: sha256_file(path) for name, path in artifact_paths.items()
            },
            "warning": "Mock outputs are not research evidence.",
        },
    )
    positives = sum(item.label is CausalLabel.SILENT_HINT_USE for item in labelled)
    summary = {
        "output_dir": str(output_dir),
        "rollouts": len(rollouts),
        "candidate_positives": positives,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
