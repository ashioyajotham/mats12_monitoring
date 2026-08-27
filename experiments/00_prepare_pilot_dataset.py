"""Create a licensed, deterministic ARC-Challenge pilot question freeze."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from src.audit import canonical_config_hash, git_revision, load_config, sha256_file, utc_now
from src.datasets.arc import read_arc_parquet, select_arc_questions, validate_pilot_questions
from src.generate_rollouts import write_manifest
from src.tasks import write_jsonl


def main() -> None:
    """Normalize a pinned ARC parquet and create the question and manifest artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--source", required=True, help="Pinned ARC-Challenge parquet file")
    parser.add_argument("--output", help="Override config paths.raw_questions")
    parser.add_argument("--manifest", help="Override the derived manifest path")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    source_path = Path(args.source)
    actual_source_hash = sha256_file(source_path)
    expected_source_hash = data["source_file_sha256"]
    if actual_source_hash != expected_source_hash:
        raise SystemExit(
            f"source SHA-256 mismatch: expected {expected_source_hash}, got {actual_source_hash}"
        )

    pool, rejected = read_arc_parquet(
        source_path,
        revision=data["source_revision"],
        source_split=data["source_split"],
    )
    selected = select_arc_questions(
        pool,
        n_questions=data["n_questions"],
        seed=data["split_seed"],
        min_source_group_size=data["min_source_group_size"],
    )
    group_counts = validate_pilot_questions(selected)

    output = Path(args.output or config["paths"]["raw_questions"])
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    write_jsonl(output, selected)
    write_manifest(
        manifest_path,
        {
            "purpose": "pilot_input_freeze",
            "created_at": utc_now(),
            "code_revision": git_revision(),
            "config_path": str(Path(args.config)),
            "config_sha256": canonical_config_hash(config),
            "source": {
                "name": data["source"],
                "config": data["source_config"],
                "split": data["source_split"],
                "revision": data["source_revision"],
                "license": data["source_license"],
                "url": data["source_url"],
                "file_sha256": actual_source_hash,
            },
            "loader": {"format": "parquet", "pyarrow_version": version("pyarrow")},
            "selection": {
                "rule": "four-choice rows; source groups with at least five eligible rows; "
                "seeded shuffle within sorted groups; round-robin allocation",
                "seed": data["split_seed"],
                "requested": data["n_questions"],
                "min_source_group_size": data["min_source_group_size"],
            },
            "counts": {
                "source_rows": len(pool) + rejected,
                "eligible_rows": len(pool),
                "rejected_rows": rejected,
                "selected_rows": len(selected),
                "selected_by_source_collection": group_counts,
            },
            "output": {
                "path": str(output),
                "sha256": sha256_file(output),
                "license": data["source_license"],
            },
        },
    )
    print(f"wrote {len(selected)} questions to {output}")
    print(f"wrote freeze manifest to {manifest_path}")


if __name__ == "__main__":
    main()
