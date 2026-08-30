import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.audit import sha256_file
from src.partial_solution_interventions import (
    build_causal_yield_freeze,
    verify_intervention,
)
from src.tasks import MathProblem, read_jsonl


def _source() -> tuple[list[MathProblem], list[dict[str, object]]]:
    questions = list(read_jsonl("data/raw/procedural_math_pilot_v21.jsonl", model=MathProblem))
    certificates = [
        json.loads(line)
        for line in Path(
            "data/raw/procedural_math_pilot_v21.certificates.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    return questions, certificates


def test_causal_yield_freeze_is_balanced_deterministic_and_outcome_blind() -> None:
    questions, certificates = _source()
    first, first_certificates, first_report = build_causal_yield_freeze(
        questions, certificates
    )
    second, second_certificates, second_report = build_causal_yield_freeze(
        questions, certificates
    )
    assert first == second
    assert first_certificates == second_certificates
    assert first_report == second_report
    assert len(first) == 12
    assert Counter(row.metadata["generator_family"] for row in first) == Counter({
        "affine_modular": 3,
        "conditional_dag": 3,
        "finite_state": 3,
        "subset_counting": 3,
    })
    assert not first_report["selection_uses_clean_outcomes"]
    assert all(row.metadata["excluded_from_monitor_data"] for row in first)
    assert all(
        verify_intervention(problem, certificate)
        for problem, certificate in zip(first, first_certificates, strict=True)
    )


def test_intervention_certificate_mutation_is_rejected() -> None:
    questions, certificates = _source()
    frozen, intervention_certificates, _ = build_causal_yield_freeze(
        questions, certificates
    )
    changed = copy.deepcopy(intervention_certificates[0])
    changed["target_answer"] = frozen[0].gold_answer
    assert not verify_intervention(frozen[0], changed)


def test_committed_causal_yield_freeze_is_bound_and_reproducible() -> None:
    questions_path = Path("data/raw/procedural_causal_yield_v1.jsonl")
    certificates_path = Path("data/raw/procedural_causal_yield_v1.certificates.jsonl")
    manifest_path = Path("data/raw/procedural_causal_yield_v1.manifest.json")
    committed = list(read_jsonl(questions_path, model=MathProblem))
    committed_certificates = [
        json.loads(line) for line in certificates_path.read_text().splitlines() if line.strip()
    ]
    source_questions, source_certificates = _source()
    regenerated, regenerated_certificates, _ = build_causal_yield_freeze(
        source_questions, source_certificates
    )
    assert [row.model_dump(mode="json") for row in committed] == [
        row.model_dump(mode="json") for row in regenerated
    ]
    assert committed_certificates == regenerated_certificates
    manifest = json.loads(manifest_path.read_text())
    recorded_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert recorded_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["questions_sha256"] == sha256_file(questions_path)
    assert manifest["certificates_sha256"] == sha256_file(certificates_path)
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_prepare_causal_yield_pilot.py"
    )
    assert manifest["intervention_code_sha256"] == sha256_file(
        "src/partial_solution_interventions.py"
    )
