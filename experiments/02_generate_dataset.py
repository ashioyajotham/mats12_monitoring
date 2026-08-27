"""Collect append-only GLM rollouts from the frozen pilot questions."""

from __future__ import annotations

import argparse
import json
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
from src.backends.zai import ZAIBackend, ZAIBackendError
from src.generate_rollouts import Rollout, collect_rollout, summarize_rollouts, write_manifest
from src.hints import Condition, build_variant, select_incorrect_option
from src.tasks import Question, read_jsonl


def collection_plan(
    questions: list[Question],
    conditions: list[Condition],
    *,
    samples_per_condition: int,
    base_seed: int,
) -> Iterator[tuple[Question, Condition, str, int]]:
    """Yield a stable question, condition, hint, and logical sample identifier plan."""
    if samples_per_condition <= 0:
        raise ValueError("samples_per_condition must be positive")
    for question_index, question in enumerate(questions):
        hinted_option = select_incorrect_option(question, variant_index=question_index)
        for condition_index, condition in enumerate(conditions):
            for sample_index in range(samples_per_condition):
                logical_seed = (
                    base_seed + question_index * 1000 + condition_index * 100 + sample_index
                )
                yield question, condition, hinted_option, logical_seed


def main() -> None:
    """Run a bounded smoke collection or the configured full GLM pilot collection."""
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
    args = parser.parse_args()

    config = load_config(args.config)
    generation = config["generation"]
    if generation["backend"] != "zai":
        raise SystemExit("02_generate_dataset requires generation.backend=zai")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    questions = [
        record
        for record in read_jsonl(config["paths"]["raw_questions"], model=Question)
        if isinstance(record, Question)
    ]
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
    purpose = (
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
        "provider_seed_supported": False,
    }
    if args.dry_run:
        print(json.dumps(plan_summary, indent=2))
        return

    try:
        backend = ZAIBackend.from_env(
            base_url=generation["base_url"],
            timeout_seconds=generation["timeout_seconds"],
            max_retries=generation["max_retries"],
        )
    except ZAIBackendError as exc:
        raise SystemExit(str(exc)) from exc

    started_at = datetime.now(UTC)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config["paths"]["generated_dir"]) / f"glm_{purpose}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    rollouts_path = output_dir / "rollouts.jsonl"
    completed: list[Rollout] = []
    failure: str | None = None
    with rollouts_path.open("x", encoding="utf-8") as handle:
        try:
            for question, condition, hinted_option, logical_seed in plan:
                variant = build_variant(question, condition, hinted_option=hinted_option)
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
                handle.write(rollout.model_dump_json() + "\n")
                handle.flush()
                completed.append(rollout)
        except Exception as exc:  # preserve partial evidence before exiting
            failure = f"{type(exc).__name__}: {exc}"

    quality = summarize_rollouts(completed)
    smoke_gate_passed = (
        purpose != "smoke_test_only"
        or (
            not failure
            and quality["completed"] == len(plan)
            and quality["invalid"] == 0
            and quality["reasoning_present"] == len(completed)
            and quality["unique_provider_request_ids"] == len(completed)
            and set(quality["finish_reasons"]) == {"stop"}
        )
    )
    status = "failed" if failure else "complete"
    if purpose == "smoke_test_only" and not failure and not smoke_gate_passed:
        status = "quality_failed"
    manifest_payload = {
        **plan_summary,
        "status": status,
        "failure": failure,
        "smoke_gate_passed": smoke_gate_passed if purpose == "smoke_test_only" else None,
        "config_path": str(Path(args.config)),
        "config_sha256": canonical_config_hash(config),
        "code_revision": git_revision(),
        "started_at": started_at.isoformat(),
        "completed_at": utc_now(),
        "runtime_environment": runtime_environment(),
        "backend": {
            "name": "zai",
            "base_url": generation["base_url"],
            "thinking": "enabled",
            "seed_supported": False,
            "open_weights": generation["open_weights"],
            "open_weights_revision": generation["open_weights_revision"],
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
        },
        "counts": {
            "requested": len(plan),
            "completed": quality["completed"],
            "invalid": quality["invalid"],
            "reasoning_present": quality["reasoning_present"],
            "unique_provider_request_ids": quality["unique_provider_request_ids"],
        },
        "finish_reasons": quality["finish_reasons"],
        "provider_models": quality["provider_models"],
        "latency_seconds": quality["latency_seconds"],
        "usage": quality["usage"],
        "output_sha256": sha256_file(rollouts_path),
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
