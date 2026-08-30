"""Analyze the bounded low-reasoning attribution diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.generate_rollouts import Rollout
from src.procedural_pilot import analyze_reasoning_effort_diagnostic
from src.tasks import MathProblem, read_jsonl


def _validate_manifest(run: Path, questions: Path) -> dict:
    """Require the frozen low-reasoning diagnostic collection contract."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected_hash = canonical_config_hash(
        load_config("configs/tinker_procedural_low_reasoning_diagnostic.yaml")
    )
    checks = {
        "purpose": manifest.get("purpose") == "procedural_reasoning_effort_diagnostic",
        "questions": manifest.get("questions") == 24,
        "samples": manifest.get("samples_per_condition") == 1,
        "model": manifest.get("model") == "openai/gpt-oss-20b",
        "renderer": manifest.get("backend", {}).get("renderer")
        == "gpt_oss_low_reasoning",
        "config": manifest.get("config_sha256") == expected_hash,
        "source": manifest.get("source_questions", {}).get("sha256")
        == sha256_file(questions),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"diagnostic run manifest failed checks: {failed}")
    return manifest


def main() -> None:
    """Grade diagnostic rollouts and evaluate the preregistered attribution gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/procedural_low_reasoning_diagnostic_v1.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = _validate_manifest(args.run, args.questions)
    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = [
        Rollout.model_validate_json(line)
        for line in (args.run / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = analyze_reasoning_effort_diagnostic(
        questions,
        rollouts,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
