"""Collect append-only reasoning-model rollouts from frozen pilot questions."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from src.audit import (
    canonical_config_hash,
    git_revision,
    load_config,
    runtime_environment,
    sha256_file,
    utc_now,
)
from src.backends.tinker import TinkerBackend, TinkerBackendError
from src.backends.zai import ZAIBackend, ZAIBackendError
from src.generate_rollouts import (
    Rollout,
    RolloutStatus,
    collect_rollout,
    rollout_id,
    summarize_rollouts,
    write_manifest,
)
from src.hints import Condition, build_variant, select_incorrect_option
from src.tasks import MathProblem, Question, read_jsonl


def collection_plan(
    questions: list[Question | MathProblem],
    conditions: list[Condition],
    *,
    samples_per_condition: int,
    base_seed: int,
) -> Iterator[tuple[Question | MathProblem, Condition, str | None, int]]:
    """Yield a stable question, condition, hint, and logical sample identifier plan."""
    if samples_per_condition <= 0:
        raise ValueError("samples_per_condition must be positive")
    indexed_questions = list(enumerate(questions))
    if questions and all(isinstance(question, MathProblem) for question in questions):
        groups: dict[str, deque[tuple[int, Question | MathProblem]]] = defaultdict(deque)
        for question_index, question in indexed_questions:
            groups[question.template_group or "unknown"].append((question_index, question))
        indexed_questions = []
        while any(groups.values()):
            for group in sorted(groups):
                if groups[group]:
                    indexed_questions.append(groups[group].popleft())

    for sample_index in range(samples_per_condition):
        for question_index, question in indexed_questions:
            hinted_option = (
                select_incorrect_option(question, variant_index=question_index)
                if isinstance(question, Question)
                else None
            )
            for condition_index, condition in enumerate(conditions):
                logical_seed = (
                    base_seed + question_index * 1000 + condition_index * 100 + sample_index
                )
                yield question, condition, hinted_option, logical_seed


def load_resume_rollouts(path: Path, *, config_sha256: str) -> list[Rollout]:
    """Load completed rollouts from a compatible prior run without modifying it."""
    manifest_path = path / "manifest.json"
    rollouts_path = path / "rollouts.jsonl"
    if not manifest_path.is_file() or not rollouts_path.is_file():
        raise ValueError("resume directory must contain manifest.json and rollouts.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("resume run uses a different configuration hash")
    return list(read_jsonl(rollouts_path, model=Rollout))


def main() -> None:
    """Run a bounded smoke collection or the configured full model pilot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/glm_smoke.yaml")
    parser.add_argument("--limit", type=int, help="Use only the first N frozen questions")
    parser.add_argument(
        "--samples-per-condition",
        type=int,
        help="Override the configured sample count for a smoke run",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the request plan"
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Import compatible completed rollouts from a prior run and skip their identities",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Wait between provider requests to reduce free-tier throttling",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record bounded request failures and continue collecting the remaining plan",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Abort after this many recorded request failures",
    )
    parser.add_argument(
        "--retry-truncated",
        action="store_true",
        help="Retry compatible resumed responses that ended at the token limit",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    generation = config["generation"]
    if generation["backend"] not in {"tinker", "zai"}:
        raise SystemExit("02_generate_dataset requires generation.backend=tinker or zai")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.request_delay_seconds < 0:
        raise SystemExit("--request-delay-seconds must be non-negative")
    if args.max_errors <= 0:
        raise SystemExit("--max-errors must be positive")

    model = MathProblem if config.get("data", {}).get("problem_type") == "math" else Question
    questions = list(read_jsonl(config["paths"]["raw_questions"], model=model))
    requested_question_ids = config.get("data", {}).get("question_ids")
    if requested_question_ids:
        by_id = {question.question_id: question for question in questions}
        missing = sorted(set(requested_question_ids) - set(by_id))
        if missing:
            raise SystemExit(f"configured question_ids are missing: {missing}")
        questions = [by_id[question_id] for question_id in requested_question_ids]
    if args.limit is not None:
        questions = questions[: args.limit]
    samples = args.samples_per_condition or generation["samples_per_condition"]
    conditions = [Condition(name) for name in config["conditions"]]
    plan = list(
        collection_plan(
            questions,
            conditions,
            samples_per_condition=samples,
            base_seed=generation["base_seed"],
        )
    )
    purpose = generation.get("purpose") or (
        "pilot"
        if args.limit is None and args.samples_per_condition is None
        else "smoke_test_only"
    )
    plan_summary = {
        "purpose": purpose,
        "questions": len(questions),
        "conditions": len(conditions),
        "samples_per_condition": samples,
        "requests": len(plan),
        "model": generation["model"],
        "provider_seed_supported": generation["provider_seed_supported"],
    }
    if args.dry_run:
        print(json.dumps(plan_summary, indent=2))
        return

    config_sha256 = canonical_config_hash(config)
    loaded_resume: list[Rollout] = []
    if args.resume_from:
        try:
            loaded_resume = load_resume_rollouts(
                args.resume_from, config_sha256=config_sha256
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot resume: {exc}") from exc
    planned_ids = {
        (question.question_id, condition, logical_seed)
        for question, condition, _, logical_seed in plan
    }
    compatible = [
        rollout
        for rollout in loaded_resume
        if (rollout.question_id, rollout.condition, rollout.seed) in planned_ids
        and rollout.model == generation["model"]
        and bool(rollout.provider_request_id)
    ]
    imported = [
        rollout
        for rollout in compatible
        if not args.retry_truncated or rollout.status is not RolloutStatus.LENGTH_TRUNCATED
    ]
    completed_ids = {rollout.rollout_id for rollout in imported}

    try:
        if generation["backend"] == "zai":
            backend = ZAIBackend.from_env(
                base_url=generation["base_url"],
                timeout_seconds=generation["timeout_seconds"],
                max_retries=generation["max_retries"],
            )
        else:
            backend = TinkerBackend(
                generation["model"], renderer_name=generation["renderer"]
            )
    except (TinkerBackendError, ZAIBackendError) as exc:
        raise SystemExit(str(exc)) from exc

    started_at = datetime.now(UTC)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(config["paths"]["generated_dir"])
        / f"{generation['backend']}_{purpose}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    rollouts_path = output_dir / "rollouts.jsonl"
    errors_path = output_dir / "request_errors.jsonl"
    completed: list[Rollout] = list(imported)
    request_errors: list[dict[str, object]] = []
    failure: str | None = None
    with (
        rollouts_path.open("x", encoding="utf-8") as handle,
        errors_path.open("x", encoding="utf-8") as errors_handle,
    ):
        for rollout in imported:
            handle.write(rollout.model_dump_json() + "\n")
        handle.flush()
        try:
            pending = [
                item
                for item in plan
                if rollout_id(item[0].question_id, item[1], item[3], generation["model"])
                not in completed_ids
            ]
            for request_index, (question, condition, hinted_option, logical_seed) in enumerate(
                pending
            ):
                if request_index and args.request_delay_seconds:
                    time.sleep(args.request_delay_seconds)
                try:
                    variant = build_variant(
                        question,
                        condition,
                        hinted_option=hinted_option,
                        math_prompt_style=generation.get(
                            "prompt_style", "concise_final_last_v3"
                        ),
                    )
                    rollout = collect_rollout(
                        question,
                        variant,
                        backend,
                        model=generation["model"],
                        seed=logical_seed,
                        temperature=generation["temperature"],
                        top_p=generation["top_p"],
                        max_new_tokens=generation["max_new_tokens"],
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    error = {
                        "question_id": question.question_id,
                        "condition": str(condition),
                        "seed": logical_seed,
                        "model": generation["model"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recorded_at": utc_now(),
                    }
                    request_errors.append(error)
                    errors_handle.write(json.dumps(error, sort_keys=True) + "\n")
                    errors_handle.flush()
                    if len(request_errors) >= args.max_errors:
                        raise RuntimeError(
                            f"reached --max-errors={args.max_errors}"
                        ) from exc
                    continue
                handle.write(rollout.model_dump_json() + "\n")
                handle.flush()
                completed.append(rollout)
        except Exception as exc:  # preserve partial evidence before exiting
            failure = f"{type(exc).__name__}: {exc}"

    quality = summarize_rollouts(completed)
    smoke_min_scorable = config.get("gates", {}).get("smoke_min_scorable", len(plan))
    smoke_max_truncation_rate = config.get("gates", {}).get(
        "smoke_max_truncation_rate", 0.0
    )
    truncation_rate = quality["truncated"] / quality["completed"] if completed else 0.0
    smoke_gate_passed = (
        purpose != "smoke_test_only"
        or (
            not failure
            and quality["completed"] == len(plan)
            and quality["scorable"] >= smoke_min_scorable
            and truncation_rate <= smoke_max_truncation_rate
            and quality["parse_invalid"] == 0
            and quality["malformed"] == 0
            and (
                generation.get("thinking") != "enabled"
                or quality["reasoning_present"] == len(completed)
            )
            and quality["unique_provider_request_ids"] == len(completed)
            and not request_errors
        )
    )
    status = (
        "failed"
        if failure
        else "complete_with_errors"
        if request_errors
        else "complete_with_invalid"
        if quality["invalid"]
        else "complete"
    )
    if purpose == "smoke_test_only" and not failure and not smoke_gate_passed:
        status = "quality_failed"
    manifest_payload = {
        **plan_summary,
        "status": status,
        "failure": failure,
        "smoke_gate_passed": smoke_gate_passed if purpose == "smoke_test_only" else None,
        "smoke_gate": (
            {
                "min_scorable": smoke_min_scorable,
                "max_truncation_rate": smoke_max_truncation_rate,
                "observed_truncation_rate": truncation_rate,
            }
            if purpose == "smoke_test_only"
            else None
        ),
        "config_path": str(Path(args.config)),
        "config_sha256": config_sha256,
        "code_revision": git_revision(),
        "started_at": started_at.isoformat(),
        "completed_at": utc_now(),
        "runtime_environment": runtime_environment(),
        "backend": {
            "name": generation["backend"],
            "base_url": generation.get("base_url"),
            "renderer": generation.get("renderer"),
            "thinking": generation.get("thinking", "enabled"),
            "seed_supported": generation["provider_seed_supported"],
            "open_weights": generation["open_weights"],
            "open_weights_revision": generation["open_weights_revision"],
            "assistant_prefill": generation.get("assistant_prefill", "disabled"),
        },
        "source_questions": {
            "path": config["paths"]["raw_questions"],
            "sha256": sha256_file(config["paths"]["raw_questions"]),
        },
        "generation_parameters": {
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "max_new_tokens": generation["max_new_tokens"],
            "logical_base_seed": generation["base_seed"],
            "request_delay_seconds": args.request_delay_seconds,
            "continue_on_error": args.continue_on_error,
            "max_errors": args.max_errors,
            "retry_truncated": args.retry_truncated,
            "assistant_prefill": generation.get("assistant_prefill", "disabled"),
        },
        "resume": {
            "source": str(args.resume_from) if args.resume_from else None,
            "loaded_rollouts": len(loaded_resume),
            "imported_rollouts": len(imported),
            "rejected_rollouts": len(loaded_resume) - len(imported),
        },
        "counts": {
            "requested": len(plan),
            "completed": quality["completed"],
            "scorable": quality["scorable"],
            "truncated": quality["truncated"],
            "malformed": quality["malformed"],
            "parse_invalid": quality["parse_invalid"],
            "invalid": quality["invalid"] + len(request_errors),
            "request_errors": len(request_errors),
            "reasoning_present": quality["reasoning_present"],
            "unique_provider_request_ids": quality["unique_provider_request_ids"],
        },
        "finish_reasons": quality["finish_reasons"],
        "provider_models": quality["provider_models"],
        "latency_seconds": quality["latency_seconds"],
        "usage": quality["usage"],
        "output_sha256": sha256_file(rollouts_path),
        "request_errors_sha256": sha256_file(errors_path),
    }
    write_manifest(output_dir / "manifest.json", manifest_payload)
    print(json.dumps({"output_dir": str(output_dir), **manifest_payload["counts"]}, indent=2))
    if failure:
        raise SystemExit(f"collection failed after {len(completed)} requests; see run manifest")
    if purpose == "smoke_test_only" and not smoke_gate_passed:
        raise SystemExit(
            "smoke collection completed but failed its integrity gate; see run manifest"
        )


if __name__ == "__main__":
    main()
