"""Tests for the qualification-informed causal-audit-v2.1 freeze."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.audit import sha256_file
from src.causal_audit_v2 import MECHANISMS
from src.causal_audit_v21 import (
    FAMILIES_V21,
    build_causal_audit_v21,
    verify_causal_audit_v21_problem,
)
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects


def test_v21_bank_is_fresh_balanced_deterministic_and_verified() -> None:
    first = build_causal_audit_v21(root_seed=901)
    assert first == build_causal_audit_v21(root_seed=901)
    problems, sources, interventions, selection = first
    assert len(problems) == len(sources) == len(interventions) == 72
    assert Counter(
        (row.metadata["generator_family"], row.metadata["corruption_mechanism"]) for row in problems
    ) == Counter(
        {(family, str(mechanism)): 12 for family in FAMILIES_V21 for mechanism in MECHANISMS}
    )
    assert selection["selection_uses_prior_qualification_outcomes"] is True
    assert selection["selection_uses_v21_model_outcomes"] is False
    assert selection["excluded_family"] == "subset_counting"
    source_by_id = {str(row["question_id"]): row for row in sources}
    intervention_by_id = {str(row["question_id"]): row for row in interventions}
    assert all(
        verify_causal_audit_v21_problem(
            row, source_by_id[row.question_id], intervention_by_id[row.question_id]
        )
        for row in problems
    )


def test_v21_rejects_tampering_and_prior_overlap() -> None:
    problems, sources, interventions, _ = build_causal_audit_v21(root_seed=902)
    changed = copy.deepcopy(interventions[0])
    changed["mechanism"] = (
        "drop_component" if changed["mechanism"] != "drop_component" else "duplicate_component"
    )
    assert not verify_causal_audit_v21_problem(problems[0], sources[0], changed)
    try:
        build_causal_audit_v21(root_seed=902, excluded_question_ids={problems[0].question_id})
    except ValueError as error:
        assert "overlaps excluded" in str(error)
    else:
        raise AssertionError("prior overlap was not rejected")


def test_committed_v21_freeze_matches_generator_and_manifest() -> None:
    stem = Path("data/raw/causal_audit_v21")
    paths = {
        "confirmatory_questions": Path(f"{stem}.confirmatory.jsonl"),
        "source_certificates": Path(f"{stem}.source_certificates.jsonl"),
        "intervention_certificates": Path(f"{stem}.intervention_certificates.jsonl"),
        "selection": Path(f"{stem}.selection.json"),
    }
    manifest = json.loads(Path(f"{stem}.manifest.json").read_text(encoding="utf-8"))
    regenerated = build_causal_audit_v21(root_seed=int(manifest["root_seed"]))
    committed = list(read_jsonl(paths["confirmatory_questions"], model=MathProblem))
    assert [row.model_dump(mode="json") for row in committed] == [
        row.model_dump(mode="json") for row in regenerated[0]
    ]
    assert read_jsonl_objects(paths["source_certificates"]) == json.loads(
        json.dumps(regenerated[1])
    )
    assert read_jsonl_objects(paths["intervention_certificates"]) == json.loads(
        json.dumps(regenerated[2])
    )
    assert json.loads(paths["selection"].read_text(encoding="utf-8")) == regenerated[3]
    assert manifest["artifact_sha256"] == {key: sha256_file(path) for key, path in paths.items()}
    assert manifest["generator_code_sha256"] == sha256_file("src/causal_audit_v21.py")
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_prepare_causal_audit_v21.py"
    )
    assert manifest["preregistration_sha256"] == sha256_file(
        "docs/PREREGISTRATION_CAUSAL_AUDIT_V21.md"
    )
    assert manifest["gate_code_sha256"] == sha256_file("src/causal_audit_gates.py")
    assert manifest["gate_entrypoint_sha256"] == sha256_file(
        "experiments/02_analyze_causal_audit_v21.py"
    )
    assert manifest["evaluation_entrypoint_sha256"] == sha256_file(
        "experiments/03_evaluate_causal_audit_v2.py"
    )
    recorded = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert recorded == hashlib.sha256(canonical.encode()).hexdigest()
