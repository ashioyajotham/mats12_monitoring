"""Evaluate the fresh clean procedural-v2 mixed-outcome gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.generate_rollouts import Rollout
from src.procedural_v2_pilot import analyze_mixed_outcome_v2
from src.tasks import MathProblem, read_jsonl


def _validated_manifest(run: Path, questions: Path) -> dict:
    """Require the exact frozen v2 clean discovery design."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = {"purpose": "procedural_v2_clean_mixed_outcome", "questions": 40,
                "conditions": 1, "samples_per_condition": 3,
                "model": "openai/gpt-oss-20b"}
    mismatches = {key: {"expected": value, "observed": manifest.get(key)}
                  for key, value in expected.items() if manifest.get(key) != value}
    source_hash = manifest.get("source_questions", {}).get("sha256")
    if source_hash != sha256_file(questions):
        mismatches["source_questions_sha256"] = {
            "expected": sha256_file(questions), "observed": source_hash}
    expected_hash = canonical_config_hash(
        load_config("configs/tinker_procedural_v2_discovery.yaml"))
    if manifest.get("config_sha256") != expected_hash:
        mismatches["config_sha256"] = {"expected": expected_hash,
                                         "observed": manifest.get("config_sha256")}
    if mismatches:
        raise SystemExit(f"v2 discovery manifest does not match the freeze: {mismatches}")
    return manifest


def main() -> None:
    """Summarize immutable fresh rollouts without modifying evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v2.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    manifest = _validated_manifest(args.run, args.questions)
    report = analyze_mixed_outcome_v2(
        questions, rollouts,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
