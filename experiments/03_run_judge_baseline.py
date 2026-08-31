"""Run the frozen transcript-only and context-aware Qwen judge baselines."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.audit import canonical_config_hash, git_revision, runtime_environment, sha256_file
from src.backends.tinker import TinkerBackend, TinkerBackendError
from src.generate_rollouts import Rollout, write_manifest
from src.judge_runner import (
    JudgeScoreRecord,
    judge_plan,
    score_judge_plan,
    summarize_judge_scores,
)
from src.monitor_dataset import MonitorExample, qualification_smoke_examples
from src.monitors.llm_judge import TinkerQwenJudge
from src.tasks import MathProblem, read_jsonl


def _load_mapping(path: Path) -> dict[str, object]:
    """Load one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_config(path: Path) -> dict[str, object]:
    """Load and minimally validate the dedicated judge configuration."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("judge"), dict):
        raise ValueError("judge config must contain a judge mapping")
    return value


def _append_jsonl(path: Path, value: object) -> None:
    """Append and flush one validated record so interrupted jobs remain resumable."""
    payload = value.model_dump(mode="json") if isinstance(value, JudgeScoreRecord) else value
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _validate_qualification(run: Path, report_path: Path, questions: Path) -> None:
    """Bind smoke inputs to the passed, permanently excluded qualification cohort."""
    report = _load_mapping(report_path)
    manifest = _load_mapping(run / "manifest.json")
    if report.get("qualification_gate_passed") is not True:
        raise ValueError("qualification gate did not pass")
    if report.get("stored_rollouts") != manifest.get("counts", {}).get("completed"):  # type: ignore[union-attr]
        raise ValueError("qualification report and run counts differ")
    if manifest.get("output_sha256") != sha256_file(run / "rollouts.jsonl"):
        raise ValueError("qualification rollout hash differs from its manifest")
    source = manifest.get("source_questions")
    if not isinstance(source, dict) or source.get("sha256") != sha256_file(questions):
        raise ValueError("qualification questions differ from the run manifest")


def _validate_full_gate(report_path: Path, smoke_manifest_path: Path, config_hash: str) -> None:
    """Require both the causal gate and exact judge smoke gate before full scoring."""
    report = _load_mapping(report_path)
    smoke = _load_mapping(smoke_manifest_path)
    if report.get("confirmatory_causal_gate_passed") is not True:
        raise ValueError("confirmatory causal gate did not pass")
    if report.get("monitor_training_authorized") is not True:
        raise ValueError("monitor training is not authorized")
    if smoke.get("judge_smoke_gate_passed") is not True:
        raise ValueError("judge smoke gate did not pass")
    if smoke.get("config_sha256") != config_hash:
        raise ValueError("judge smoke used a different frozen configuration")


def _create_or_resume_output(
    *,
    output_root: Path,
    purpose: str,
    resume_from: Path | None,
    plan_record: dict[str, object],
    allow_parser_amendment: bool,
) -> tuple[Path, list[JudgeScoreRecord], list[dict[str, object]]]:
    """Create an append-only output directory or verify a compatible partial run."""
    if resume_from:
        output_dir = resume_from
        observed = _load_mapping(output_dir / "plan.json")
        if observed != plan_record:
            observed_comparable = json.loads(json.dumps(observed))
            current_comparable = json.loads(json.dumps(plan_record))
            observed_inputs = observed_comparable.get("input_sha256", {})
            current_inputs = current_comparable.get("input_sha256", {})
            if isinstance(observed_inputs, dict):
                observed_inputs.pop("judge_implementation", None)
            if isinstance(current_inputs, dict):
                current_inputs.pop("judge_implementation", None)
            if not allow_parser_amendment or observed_comparable != current_comparable:
                raise ValueError("resume judge plan differs from the current frozen plan")
            amendment_path = output_dir / "plan_parser_amendment.json"
            if amendment_path.is_file():
                if _load_mapping(amendment_path) != plan_record:
                    raise ValueError("resume parser amendment differs from the current plan")
            else:
                with amendment_path.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(plan_record, indent=2, sort_keys=True) + "\n")
        scores = list(read_jsonl(output_dir / "scores.jsonl", model=JudgeScoreRecord))
        failures_path = output_dir / "request_errors.jsonl"
        failures = (
            [
                json.loads(line)
                for line in failures_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if failures_path.is_file()
            else []
        )
        return output_dir, scores, failures
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"tinker_{purpose}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "plan.json").write_text(
        json.dumps(plan_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "scores.jsonl").touch(exist_ok=False)
    (output_dir / "request_errors.jsonl").touch(exist_ok=False)
    return output_dir, [], []


def main() -> None:
    """Execute a smoke or gate-authorized full judge plan with exact resumption."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tinker_qwen_causal_error_judge.yaml")
    )
    parser.add_argument("--run", type=Path, help="Qualification run for smoke mode")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/causal_error_detection_v1.qualification.jsonl"),
    )
    parser.add_argument(
        "--primary", type=Path, default=Path("data/reviewed/causal_error_v1.primary.jsonl")
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
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--allow-parser-amendment",
        action="store_true",
        help=(
            "Resume when only the hashed judge implementation changed under a documented "
            "parser-validity amendment"
        ),
    )
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-errors", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.request_delay_seconds < 0 or args.max_errors <= 0:
        raise SystemExit("delay must be non-negative and max-errors must be positive")

    config = _load_config(args.config)
    judge_config = config["judge"]
    assert isinstance(judge_config, dict)
    config_hash = canonical_config_hash(config)
    input_hashes: dict[str, str] = {
        "config": sha256_file(args.config),
        "judge_implementation": sha256_file("src/monitors/llm_judge.py"),
    }
    if args.mode == "smoke":
        if args.run is None:
            raise SystemExit("--run is required in smoke mode")
        _validate_qualification(args.run, args.report, args.questions)
        questions = list(read_jsonl(args.questions, model=MathProblem))
        rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
        examples = qualification_smoke_examples(
            questions,
            rollouts,
            count=int(judge_config["smoke_examples"]),
            seed=int(judge_config["smoke_selection_seed"]),
        )
        base_seed = int(judge_config["smoke_base_seed"])
        purpose = "causal_error_judge_smoke"
        input_hashes.update(
            {
                "report": sha256_file(args.report),
                "questions": sha256_file(args.questions),
                "rollouts": sha256_file(args.run / "rollouts.jsonl"),
            }
        )
    else:
        if args.smoke_manifest is None:
            raise SystemExit("--smoke-manifest is required in full mode")
        _validate_full_gate(args.report, args.smoke_manifest, config_hash)
        dataset_manifest = _load_mapping(args.dataset_manifest)
        if dataset_manifest.get("purpose") != "causal_error_v1_monitor_dataset":
            raise SystemExit("monitor dataset manifest has the wrong purpose")
        if dataset_manifest.get("primary_sha256") != sha256_file(args.primary):
            raise SystemExit("primary examples differ from their gate-bound manifest")
        if dataset_manifest.get("secondary_audit_sha256") != sha256_file(
            args.secondary_audit
        ):
            raise SystemExit("secondary audit differs from its gate-bound manifest")
        if dataset_manifest.get("confirmatory_report_sha256") != sha256_file(args.report):
            raise SystemExit("confirmatory report differs from the monitor dataset manifest")
        primary = list(read_jsonl(args.primary, model=MonitorExample))
        audit = list(read_jsonl(args.secondary_audit, model=MonitorExample))
        if len(audit) > int(judge_config["secondary_audit_cap"]):
            raise SystemExit("secondary audit exceeds the frozen judge cap")
        examples = [*primary, *audit]
        base_seed = int(judge_config["full_base_seed"])
        purpose = "causal_error_judge_full"
        input_hashes.update(
            {
                "report": sha256_file(args.report),
                "primary": sha256_file(args.primary),
                "secondary_audit": sha256_file(args.secondary_audit),
                "monitor_dataset_manifest": sha256_file(args.dataset_manifest),
                "smoke_manifest": sha256_file(args.smoke_manifest),
            }
        )
    plan = judge_plan(
        examples,
        model=str(judge_config["model"]),
        prompt_version=str(judge_config["prompt_version"]),
        base_seed=base_seed,
    )
    plan_record = {
        "protocol": "causal-error-qwen-judge-plan-v1",
        "purpose": purpose,
        "config_sha256": config_hash,
        "input_sha256": input_hashes,
        "model": judge_config["model"],
        "open_weights_revision": judge_config["open_weights_revision"],
        "renderer": judge_config["renderer"],
        "prompt_version": judge_config["prompt_version"],
        "base_seed": base_seed,
        "examples": len(examples),
        "requests": len(plan),
        "score_ids": [item[3] for item in plan],
    }
    if args.dry_run:
        summary = {key: value for key, value in plan_record.items() if key != "score_ids"}
        print(json.dumps(summary, indent=2))
        return

    try:
        backend = TinkerBackend(
            str(judge_config["model"]), renderer_name=str(judge_config["renderer"])
        )
    except TinkerBackendError as exc:
        raise SystemExit(str(exc)) from exc
    judge = TinkerQwenJudge(
        backend,
        model=str(judge_config["model"]),
        max_new_tokens=int(judge_config["max_new_tokens"]),
        temperature=float(judge_config["temperature"]),
        top_p=float(judge_config["top_p"]),
        max_retries=int(judge_config["max_retries"]),
    )
    output_root = Path(config["paths"]["generated_dir"])  # type: ignore[index]
    output_dir, records, previous_failures = _create_or_resume_output(
        output_root=output_root,
        purpose=purpose,
        resume_from=args.resume_from,
        plan_record=plan_record,
        allow_parser_amendment=args.allow_parser_amendment,
    )
    current_failures: list[dict[str, str]] = []
    completed_ids = {row.score_id for row in records}
    for item in plan:
        if item[3] in completed_ids:
            continue
        if records and args.request_delay_seconds:
            time.sleep(args.request_delay_seconds)
        new_records, failures = score_judge_plan([item], judge)
        for record in new_records:
            _append_jsonl(output_dir / "scores.jsonl", record)
            records.append(record)
            completed_ids.add(record.score_id)
        for failure in failures:
            _append_jsonl(output_dir / "request_errors.jsonl", failure)
            current_failures.append(failure)
        if len(current_failures) >= args.max_errors:
            break

    summary = summarize_judge_scores(plan, records, current_failures)
    complete = summary["completed"] == summary["requested"] and not current_failures
    smoke_gate = args.mode != "smoke" or (
        complete
        and summary["completed"] == 4
        and summary["view_counts"] == {"context_aware": 2, "transcript_only": 2}
        and summary["unique_provider_request_ids"] == 4
    )
    manifest = {
        **plan_record,
        **summary,
        "complete": complete,
        "judge_smoke_gate_passed": smoke_gate,
        "failures_in_previous_attempts": len(previous_failures),
        "parser_amendment_used": args.allow_parser_amendment,
        "original_plan_sha256": sha256_file(output_dir / "plan.json"),
        "parser_amendment_plan_sha256": (
            sha256_file(output_dir / "plan_parser_amendment.json")
            if (output_dir / "plan_parser_amendment.json").is_file()
            else None
        ),
        "scores_sha256": sha256_file(output_dir / "scores.jsonl"),
        "request_errors_sha256": sha256_file(output_dir / "request_errors.jsonl"),
        "entrypoint_sha256": sha256_file(__file__),
        "code_revision": git_revision(),
        "runtime_environment": runtime_environment(),
    }
    print(
        json.dumps(
            {"output_dir": str(output_dir), **summary, "gate_passed": smoke_gate},
            indent=2,
        )
    )
    if not complete or not smoke_gate:
        (output_dir / "status.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise SystemExit(
            "judge run incomplete or failed its integrity gate; resume from the output directory"
        )
    write_manifest(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
