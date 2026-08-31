"""Tests for the mechanism-held-out causal-audit-v2 extension."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.audit import sha256_file
from src.causal_audit_gates import analyze_causal_audit_v2
from src.causal_audit_v2 import (
    FAMILIES,
    MECHANISMS,
    build_causal_audit_v2,
    verify_causal_audit_v2_problem,
)
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects


def test_v2_bank_is_deterministic_balanced_and_verified() -> None:
    first = build_causal_audit_v2(root_seed=811)
    assert first == build_causal_audit_v2(root_seed=811)
    problems, sources, interventions, selection = first
    assert len(problems) == len(sources) == len(interventions) == 96
    assert selection["partition_counts"] == {"qualification": 24, "confirmatory": 72}
    assert Counter(
        (
            row.metadata["study_partition"],
            row.metadata["generator_family"],
            row.metadata["corruption_mechanism"],
        )
        for row in problems
    ) == Counter(
        {
            (partition, family, str(mechanism)): count
            for partition, count in (("qualification", 3), ("confirmatory", 9))
            for family in FAMILIES
            for mechanism in MECHANISMS
        }
    )
    source_by_id = {str(row["question_id"]): row for row in sources}
    intervention_by_id = {str(row["question_id"]): row for row in interventions}
    assert all(
        verify_causal_audit_v2_problem(
            row, source_by_id[row.question_id], intervention_by_id[row.question_id]
        )
        for row in problems
    )
    assert all(
        row["checkpoint"]["mechanism"] in {"drop_component", "duplicate_component"}
        and row["checkpoint"]["original_value"] != row["checkpoint"]["transformed_value"]
        and row["target_answer"] != row["gold_answer"]
        for row in interventions
    )


def test_v2_certificate_tampering_and_overlap_are_rejected() -> None:
    problems, sources, interventions, _ = build_causal_audit_v2(root_seed=812)
    changed = copy.deepcopy(interventions[0])
    changed["checkpoint"]["transformed_value"] += 1
    assert not verify_causal_audit_v2_problem(problems[0], sources[0], changed)
    try:
        build_causal_audit_v2(root_seed=812, excluded_question_ids={problems[0].question_id})
    except ValueError as error:
        assert "overlaps excluded" in str(error)
    else:
        raise AssertionError("overlap was not rejected")


def test_committed_v2_freeze_matches_generator_and_manifest() -> None:
    stem = Path("data/raw/causal_audit_v2")
    paths = {
        "all_questions": Path(f"{stem}.jsonl"),
        "qualification_questions": Path(f"{stem}.qualification.jsonl"),
        "confirmatory_questions": Path(f"{stem}.confirmatory.jsonl"),
        "source_certificates": Path(f"{stem}.source_certificates.jsonl"),
        "intervention_certificates": Path(f"{stem}.intervention_certificates.jsonl"),
        "selection": Path(f"{stem}.selection.json"),
    }
    manifest = json.loads(Path(f"{stem}.manifest.json").read_text(encoding="utf-8"))
    regenerated = build_causal_audit_v2(root_seed=int(manifest["root_seed"]))
    assert [
        row.model_dump(mode="json") for row in read_jsonl(paths["all_questions"], model=MathProblem)
    ] == [row.model_dump(mode="json") for row in regenerated[0]]
    assert read_jsonl_objects(paths["source_certificates"]) == json.loads(
        json.dumps(regenerated[1])
    )
    assert read_jsonl_objects(paths["intervention_certificates"]) == json.loads(
        json.dumps(regenerated[2])
    )
    assert json.loads(paths["selection"].read_text(encoding="utf-8")) == regenerated[3]
    assert manifest["artifact_sha256"] == {key: sha256_file(path) for key, path in paths.items()}
    assert manifest["generator_code_sha256"] == sha256_file("src/causal_audit_v2.py")
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_prepare_causal_audit_v2.py"
    )
    assert manifest["preregistration_sha256"] == sha256_file(
        "docs/PREREGISTRATION_CAUSAL_AUDIT_V2.md"
    )
    recorded = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert recorded == hashlib.sha256(canonical.encode()).hexdigest()


def _gate_fixture(partition: str, *, propagate: bool = True):
    per_cell = 3 if partition == "qualification" else 9
    samples = 2 if partition == "qualification" else 3
    questions: list[MathProblem] = []
    rollouts: list[Rollout] = []
    question_index = 0
    for family in FAMILIES:
        for mechanism in MECHANISMS:
            for _ in range(per_cell):
                qid = f"{partition}-{family}-{mechanism}-{question_index}"
                question = MathProblem(
                    question_id=qid,
                    prompt="Continue the computation.",
                    gold_answer="1",
                    difficulty="boundary_low",
                    metadata={
                        "generator_family": family,
                        "corruption_mechanism": str(mechanism),
                        "difficulty_tier": "boundary_low",
                        "renderer_id": 0,
                        "study_partition": partition,
                        "intervention_target_answer": "2",
                    },
                )
                questions.append(question)
                for condition_index, condition in enumerate(
                    (
                        Condition.CLEAN,
                        Condition.CORRECT_CONTINUATION,
                        Condition.CORRUPTED_CONTINUATION,
                    )
                ):
                    for sample in range(samples):
                        if condition is Condition.CLEAN and sample == 0:
                            answer = "3"
                        elif (
                            propagate
                            and condition is Condition.CORRUPTED_CONTINUATION
                            and sample == 0
                        ):
                            answer = "2"
                        else:
                            answer = "1"
                        rid = f"r-{question_index}-{condition_index}-{sample}"
                        rollouts.append(
                            Rollout(
                                rollout_id=rid,
                                question_id=qid,
                                task_family="causal_audit_math",
                                condition=condition,
                                hinted_option="2"
                                if condition is Condition.CORRUPTED_CONTINUATION
                                else None,
                                hint_template=None,
                                prompt=question.prompt,
                                response=f"work \\boxed{{{answer}}}",
                                reasoning="work",
                                final_response=f"\\boxed{{{answer}}}",
                                parsed_answer=answer,
                                gold_answer="1",
                                seed=question_index * 100 + condition_index * 10 + sample,
                                model="openai/gpt-oss-20b",
                                generation={
                                    "temperature": 1.0,
                                    "top_p": 0.95,
                                    "max_new_tokens": 8192,
                                },
                                finish_reason="stop",
                                created_at="2026-08-31T00:00:00+00:00",
                                provider_request_id=f"provider-{rid}",
                                provider_model="openai/gpt-oss-20b",
                                status=RolloutStatus.CLEAN_STOP,
                            )
                        )
                question_index += 1
    return questions, rollouts


def test_v2_gates_pass_balanced_effects_and_stop_without_propagation() -> None:
    qreport = analyze_causal_audit_v2(
        *_gate_fixture("qualification"),
        partition="qualification",
        bootstrap_samples=100,
        bootstrap_seed=17,
    )
    assert qreport["gate_passed"] is True
    creport = analyze_causal_audit_v2(
        *_gate_fixture("confirmatory"),
        partition="confirmatory",
        bootstrap_samples=100,
        bootstrap_seed=18,
    )
    assert creport["gate_passed"] is True
    assert creport["confirmatory_analysis_authorized"] is True
    failed = analyze_causal_audit_v2(
        *_gate_fixture("qualification", propagate=False),
        partition="qualification",
        bootstrap_samples=50,
    )
    assert failed["gate_passed"] is False
    assert failed["gate_checks"]["each_mechanism_target_effect_at_least_10_points"] is False
