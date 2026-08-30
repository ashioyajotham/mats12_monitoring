"""Evaluate the complete frozen causal-error monitor stack on held-out groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import git_revision, runtime_environment, sha256_file
from src.generate_rollouts import write_manifest
from src.judge_runner import JudgeScoreRecord
from src.monitor_dataset import MonitorExample
from src.monitor_evaluation import (
    ComponentScore,
    build_hybrid_scores,
    evaluate_component,
    invalid_rollout_split_diagnostic,
    paired_component_comparison,
    summarize_secondary_audit,
    validation_operating_point,
)
from src.tasks import read_jsonl, write_jsonl


def _judge_components(rows: list[JudgeScoreRecord]) -> list[ComponentScore]:
    """Convert typed judge records into the common component-score schema."""
    names = {
        "transcript_only": "transcript_only_judge",
        "context_aware": "context_aware_judge",
    }
    output: list[ComponentScore] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.example_id, row.view)
        if key in seen:
            raise ValueError(f"duplicate judge score for {key}")
        if row.view not in names:
            raise ValueError(f"unknown judge view: {row.view}")
        seen.add(key)
        output.append(
            ComponentScore(
                example_id=row.example_id,
                question_id=row.question_id,
                split=row.split,
                component=names[row.view],
                score=row.score,
                score_origin="frozen_qwen_judge",
            )
        )
    return output


def main() -> None:
    """Fit the hybrid and produce primary metrics plus descriptive hard-negative audits."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary", type=Path, default=Path("data/reviewed/causal_error_v1.primary.jsonl")
    )
    parser.add_argument(
        "--secondary-audit",
        type=Path,
        default=Path("data/reviewed/causal_error_v1.secondary_audit.jsonl"),
    )
    parser.add_argument(
        "--local-scores", type=Path, default=Path("results/causal_error_v1.local_scores.jsonl")
    )
    parser.add_argument(
        "--local-manifest",
        type=Path,
        default=Path("results/causal_error_v1.local_scores.manifest.json"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/reviewed/causal_error_v1.monitor_dataset.manifest.json"),
    )
    parser.add_argument("--judge-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--hybrid-seed", type=int, default=20262631)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20262632)
    args = parser.parse_args()

    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if dataset_manifest.get("purpose") != "causal_error_v1_monitor_dataset":
        raise SystemExit("monitor dataset manifest has the wrong purpose")
    if dataset_manifest.get("primary_sha256") != sha256_file(args.primary):
        raise SystemExit("primary examples differ from their gate-bound manifest")
    if dataset_manifest.get("secondary_audit_sha256") != sha256_file(
        args.secondary_audit
    ):
        raise SystemExit("secondary audit differs from its gate-bound manifest")
    local_manifest = json.loads(args.local_manifest.read_text(encoding="utf-8"))
    if local_manifest.get("purpose") != "causal_error_v1_local_monitor_scores":
        raise SystemExit("local score manifest has the wrong purpose")
    if local_manifest.get("scores_sha256") != sha256_file(args.local_scores):
        raise SystemExit("local scores differ from their manifest")
    judge_manifest = json.loads((args.judge_run / "manifest.json").read_text(encoding="utf-8"))
    if judge_manifest.get("purpose") != "causal_error_judge_full":
        raise SystemExit("judge run is not the frozen full baseline")
    if judge_manifest.get("complete") is not True:
        raise SystemExit("full judge run is incomplete")
    judge_path = args.judge_run / "scores.jsonl"
    if judge_manifest.get("scores_sha256") != sha256_file(judge_path):
        raise SystemExit("judge scores differ from their run manifest")

    primary = list(read_jsonl(args.primary, model=MonitorExample))
    audit = list(read_jsonl(args.secondary_audit, model=MonitorExample))
    local = list(read_jsonl(args.local_scores, model=ComponentScore))
    judges = list(read_jsonl(judge_path, model=JudgeScoreRecord))
    expected_judge_pairs = {
        (row.example_id, view)
        for row in [*primary, *audit]
        for view in ("transcript_only", "context_aware")
    }
    observed_judge_pairs = {(row.example_id, row.view) for row in judges}
    if observed_judge_pairs != expected_judge_pairs:
        raise SystemExit("judge scores do not exactly cover primary plus secondary audit views")

    judge_components = _judge_components(judges)
    hybrid = build_hybrid_scores(primary, local, judges, seed=args.hybrid_seed)
    all_scores = [*local, *judge_components, *hybrid]
    components = (
        "correctness_only",
        "reasoning_length",
        "template_identity",
        "shuffled_label_surface",
        "surface",
        "counterfactual_answer_shift",
        "transcript_only_judge",
        "context_aware_judge",
        "hybrid",
    )
    metrics = {
        component: {
            **evaluate_component(
                primary,
                all_scores,
                component=component,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            ),
            "validation_selected_operating_point": validation_operating_point(
                primary, all_scores, component=component
            ),
        }
        for component in components
    }
    audit_components = tuple(component for component in components if component != "hybrid")
    report = {
        "protocol": "causal-error-monitor-evaluation-v1",
        "primary_comparison": ["counterfactual_answer_shift", "surface"],
        "primary_paired_comparison": paired_component_comparison(
            primary,
            all_scores,
            first="counterfactual_answer_shift",
            second="surface",
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 1,
        ),
        "primary_test_metrics": metrics,
        "secondary_hard_negative_audit_descriptive_only": {
            component: summarize_secondary_audit(audit, all_scores, component=component)
            for component in audit_components
        },
        "deliberately_invalid_rollout_split_diagnostic": invalid_rollout_split_diagnostic(
            primary
        ),
        "notes": {
            "label": "ordinary failure versus causally induced failure among known-wrong answers",
            "grouping": "question-group-disjoint train/validation/test",
            "uncertainty": "question-clustered bootstrap",
            "hybrid_training": (
                "five-fold question-group OOF surface scores plus fixed and judge components"
            ),
        },
    }
    score_path = args.output_dir / "causal_error_v1.monitor_scores.jsonl"
    report_path = args.output_dir / "causal_error_v1.monitor_metrics.json"
    manifest_path = args.output_dir / "causal_error_v1.monitor_metrics.manifest.json"
    write_jsonl(score_path, all_scores)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_manifest(
        manifest_path,
        {
            "purpose": "causal_error_v1_complete_monitor_evaluation",
            "components": list(components),
            "hybrid_seed": args.hybrid_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "primary_sha256": sha256_file(args.primary),
            "secondary_audit_sha256": sha256_file(args.secondary_audit),
            "local_scores_sha256": sha256_file(args.local_scores),
            "local_manifest_sha256": sha256_file(args.local_manifest),
            "monitor_dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "judge_scores_sha256": sha256_file(judge_path),
            "scores_sha256": sha256_file(score_path),
            "report_sha256": sha256_file(report_path),
            "entrypoint_sha256": sha256_file(__file__),
            "code_revision": git_revision(),
            "runtime_environment": runtime_environment(),
        },
    )
    print(json.dumps({"scores": len(all_scores), "components": list(components)}, indent=2))


if __name__ == "__main__":
    main()
