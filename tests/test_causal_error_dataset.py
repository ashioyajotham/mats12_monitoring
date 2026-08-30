import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.audit import sha256_file
from src.causal_error_dataset import (
    FAMILIES,
    MONITOR_SPLIT_COUNTS,
    build_causal_error_dataset,
    verify_causal_error_problem,
)
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects


def _build():
    """Build the frozen design with compact test seeds."""
    return build_causal_error_dataset(
        root_seed=701,
        partition_seed=702,
        split_seed=703,
    )


def test_causal_error_bank_is_fresh_balanced_deterministic_and_verified() -> None:
    first = _build()
    second = _build()
    assert first == second
    problems, source_certificates, intervention_certificates, selection = first
    assert len(problems) == len(source_certificates) == len(intervention_certificates) == 96
    assert len({row.question_id for row in problems}) == 96
    assert len({row.prompt for row in problems}) == 96
    assert Counter(row.metadata["generator_family"] for row in problems) == Counter(
        {family: 24 for family in FAMILIES}
    )
    assert Counter(
        (
            row.metadata["generator_family"],
            row.metadata["difficulty_tier"],
            row.metadata["renderer_id"],
        )
        for row in problems
    ) == Counter(
        {
            (family, tier, renderer): 6
            for family in FAMILIES
            for tier in ("boundary_low", "boundary_high")
            for renderer in (0, 1)
        }
    )
    source_by_id = {str(row["question_id"]): row for row in source_certificates}
    intervention_by_id = {
        str(row["question_id"]): row for row in intervention_certificates
    }
    assert all(
        verify_causal_error_problem(
            problem,
            source_by_id[problem.question_id],
            intervention_by_id[problem.question_id],
        )
        for problem in problems
    )
    assert selection["selection_uses_model_outcomes"] is False
    assert selection["all_partitions_frozen_before_collection"] is True


def test_partitions_are_disjoint_balanced_and_monitor_splits_are_grouped() -> None:
    problems, _, _, selection = _build()
    qualification = {
        row.question_id for row in problems
        if row.metadata["study_partition"] == "qualification"
    }
    confirmatory = {
        row.question_id for row in problems
        if row.metadata["study_partition"] == "confirmatory"
    }
    assert len(qualification) == 24
    assert len(confirmatory) == 72
    assert not qualification & confirmatory
    assert qualification | confirmatory == {row.question_id for row in problems}
    assert Counter(
        row.metadata["generator_family"]
        for row in problems
        if row.question_id in qualification
    ) == Counter({family: 6 for family in FAMILIES})
    assert Counter(
        row.metadata["generator_family"]
        for row in problems
        if row.question_id in confirmatory
    ) == Counter({family: 18 for family in FAMILIES})
    assert all(
        row.metadata["excluded_from_monitor_data"] is True
        and row.metadata["monitor_split"] is None
        for row in problems
        if row.question_id in qualification
    )
    assert all(
        row.metadata["excluded_from_monitor_data"] is False
        and row.metadata["monitor_split"] in {"train", "validation", "test"}
        for row in problems
        if row.question_id in confirmatory
    )
    assert Counter(
        row.metadata["monitor_split"]
        for row in problems
        if row.question_id in confirmatory
    ) == Counter(MONITOR_SPLIT_COUNTS)
    assert selection["monitor_split_counts"] == {
        "test": 15,
        "train": 43,
        "validation": 14,
    }
    assert all(
        {
            row.metadata["monitor_split"]
            for row in problems
            if row.question_id in confirmatory
            and row.metadata["generator_family"] == family
        }
        == {"train", "validation", "test"}
        for family in FAMILIES
    )


def test_prior_overlap_and_certificate_mutation_are_rejected() -> None:
    problems, source_certificates, intervention_certificates, _ = _build()
    with pytest.raises(ValueError, match="overlaps excluded"):
        build_causal_error_dataset(
            root_seed=701,
            partition_seed=702,
            split_seed=703,
            excluded_question_ids={problems[0].question_id},
        )
    changed = copy.deepcopy(intervention_certificates[0])
    changed["target_answer"] = problems[0].gold_answer
    assert not verify_causal_error_problem(problems[0], source_certificates[0], changed)


def test_committed_causal_error_freeze_matches_generator_and_manifest() -> None:
    stem = Path("data/raw/causal_error_detection_v1")
    paths = {
        "all_questions": Path(f"{stem}.jsonl"),
        "qualification_questions": Path(f"{stem}.qualification.jsonl"),
        "confirmatory_questions": Path(f"{stem}.confirmatory.jsonl"),
        "source_certificates": Path(f"{stem}.source_certificates.jsonl"),
        "intervention_certificates": Path(f"{stem}.intervention_certificates.jsonl"),
        "selection": Path(f"{stem}.selection.json"),
    }
    manifest_path = Path(f"{stem}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = list(read_jsonl(paths["all_questions"], model=MathProblem))
    source_certificates = read_jsonl_objects(paths["source_certificates"])
    intervention_certificates = read_jsonl_objects(paths["intervention_certificates"])
    regenerated = build_causal_error_dataset(
        root_seed=int(manifest["root_seed"]),
        partition_seed=int(manifest["partition_seed"]),
        split_seed=int(manifest["monitor_split_seed"]),
        per_family=int(manifest["per_family"]),
    )
    assert [row.model_dump(mode="json") for row in problems] == [
        row.model_dump(mode="json") for row in regenerated[0]
    ]
    assert source_certificates == json.loads(json.dumps(regenerated[1]))
    assert intervention_certificates == json.loads(json.dumps(regenerated[2]))
    assert json.loads(paths["selection"].read_text(encoding="utf-8")) == regenerated[3]
    assert len(read_jsonl(paths["qualification_questions"], model=MathProblem)) == 24
    assert len(read_jsonl(paths["confirmatory_questions"], model=MathProblem)) == 72
    assert manifest["artifact_sha256"] == {
        key: sha256_file(path) for key, path in paths.items()
    }
    recorded_manifest_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert recorded_manifest_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["generator_code_sha256"] == sha256_file(
        "src/causal_error_dataset.py"
    )
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_prepare_causal_error_detection_v1.py"
    )
