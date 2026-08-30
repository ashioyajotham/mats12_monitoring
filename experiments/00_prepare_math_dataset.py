"""Prepare a deterministic, audited MATH level-4/5 free-response freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from src.math_answers import extract_math_answer
from src.tasks import MathProblem, write_jsonl


def novelize(prompt: str) -> str:
    """Apply structure-preserving formatting novelization without changing semantics."""
    normalized = " ".join(prompt.split())
    return f"Solve this competition mathematics problem. {normalized}"


def load_rows(paths: list[Path]) -> list[dict]:
    """Read parquet source shards using the project's optional data dependency."""
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def main() -> None:
    """Select balanced level-4/5 records and write an append-only JSONL freeze."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/math_questions.jsonl"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=8700)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    rows = []
    for row in load_rows(args.source):
        level = str(row.get("level", "")).lower().replace("level", "").strip()
        answer = extract_math_answer(str(row.get("solution", "")))
        if level in {"4", "5"} and answer:
            row["_answer"] = answer
            rows.append(row)
    if len(rows) < args.count:
        raise SystemExit(f"only {len(rows)} eligible rows; need {args.count}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: args.count]
    records = []
    for index, row in enumerate(selected):
        prompt = novelize(str(row["problem"]))
        records.append(MathProblem(
            question_id=f"math_l45_{index:03d}", prompt=prompt,
            gold_answer=row["_answer"], difficulty=str(row.get("level")),
            template_group=str(row.get("type", "unknown")),
            source="EleutherAI/hendrycks_math",
            metadata={"novelization": "format_novelization_v1", "source_problem": row["problem"]},
        ))
    write_jsonl(args.output, records)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "output": str(args.output),
        "sha256": digest,
        "count": len(records),
        "seed": args.seed,
        "source_shards": [str(path) for path in args.source],
        "source_shard_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.source
        },
        "source_revision": args.source_revision,
        "source_license": "MIT",
        "novelization": "format_novelization_v1",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
