"""Fit and score the frozen local causal-error monitor controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import git_revision, runtime_environment, sha256_file
from src.generate_rollouts import Rollout, write_manifest
from src.monitor_dataset import MonitorExample
from src.monitor_evaluation import build_local_component_scores, evaluate_component
from src.tasks import read_jsonl, write_jsonl


def main() -> None:
    """Create local component scores and grouped test reports immutably."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("data/reviewed/causal_error_v1.primary.jsonl"),
    )
    parser.add_argument(
        "--secondary-audit",
        type=Path,
        default=Path("data/reviewed/causal_error_v1.secondary_audit.jsonl"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/reviewed/causal_error_v1.monitor_dataset.manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20262630)
    args = parser.parse_args()

    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if dataset_manifest.get("purpose") != "causal_error_v1_monitor_dataset":
        raise SystemExit("monitor dataset manifest has the wrong purpose")
    if dataset_manifest.get("primary_sha256") != sha256_file(args.primary):
        raise SystemExit("primary examples differ from the gate-bound dataset manifest")
    if dataset_manifest.get("secondary_audit_sha256") != sha256_file(
        args.secondary_audit
    ):
        raise SystemExit("secondary audit differs from the gate-bound dataset manifest")
    primary = list(read_jsonl(args.primary, model=MonitorExample))
    audit = list(read_jsonl(args.secondary_audit, model=MonitorExample))
    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    scores = build_local_component_scores(primary, audit, rollouts, seed=args.seed)
    score_path = args.output_dir / "causal_error_v1.local_scores.jsonl"
    report_path = args.output_dir / "causal_error_v1.local_metrics.json"
    manifest_path = args.output_dir / "causal_error_v1.local_scores.manifest.json"
    write_jsonl(score_path, scores)
    components = (
        "correctness_only",
        "reasoning_length",
        "template_identity",
        "shuffled_label_surface",
        "surface",
        "counterfactual_answer_shift",
    )
    report = {
        "protocol": "causal-error-local-monitor-evaluation-v1",
        "components": {
            component: evaluate_component(primary, scores, component=component)
            for component in components
        },
        "primary_comparison": ["counterfactual_answer_shift", "surface"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_manifest(
        manifest_path,
        {
            "purpose": "causal_error_v1_local_monitor_scores",
            "seed": args.seed,
            "components": list(components),
            "primary_sha256": sha256_file(args.primary),
            "secondary_audit_sha256": sha256_file(args.secondary_audit),
            "monitor_dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "confirmatory_rollouts_sha256": sha256_file(args.run / "rollouts.jsonl"),
            "scores_sha256": sha256_file(score_path),
            "report_sha256": sha256_file(report_path),
            "entrypoint_sha256": sha256_file(__file__),
            "code_revision": git_revision(),
            "runtime_environment": runtime_environment(),
        },
    )
    print(json.dumps({"scores": len(scores), "components": list(components)}, indent=2))


if __name__ == "__main__":
    main()
