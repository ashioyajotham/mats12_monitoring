import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.audit import sha256_file
from src.datasets.procedural_math_v2 import (
    FAMILIES_V2,
    TIERS_V2,
    generate_candidate_bank_v2,
    verify_problem_v2,
)
from src.tasks import MathProblem, read_jsonl


def test_v2_bank_is_deterministic_balanced_and_dual_verified() -> None:
    first, first_certificates = generate_candidate_bank_v2(root_seed=91, per_cell=2)
    second, second_certificates = generate_candidate_bank_v2(root_seed=91, per_cell=2)
    assert first == second
    assert first_certificates == second_certificates
    assert Counter(
        (row.metadata["generator_family"], row.metadata["difficulty_tier"])
        for row in first
    ) == Counter({(family, tier): 2 for family in FAMILIES_V2 for tier in TIERS_V2})
    assert all(
        verify_problem_v2(problem, certificate)
        for problem, certificate in zip(first, first_certificates, strict=True)
    )
    different, _ = generate_candidate_bank_v2(root_seed=92, per_cell=2)
    assert [row.question_id for row in first] != [row.question_id for row in different]


def test_v2_certificate_mutation_is_rejected() -> None:
    problems, certificates = generate_candidate_bank_v2(root_seed=93, per_cell=1)
    changed = copy.deepcopy(certificates[0])
    changed["oracle_answer"] = str(int(str(changed["oracle_answer"])) + 1)
    assert not verify_problem_v2(problems[0], changed)


def test_committed_v2_freeze_matches_generator_and_manifest() -> None:
    questions_path = Path("data/raw/procedural_math_candidates_v2.jsonl")
    certificates_path = Path("data/raw/procedural_math_certificates_v2.jsonl")
    manifest = json.loads(
        Path("data/raw/procedural_math_candidates_v2.manifest.json").read_text()
    )
    questions = list(read_jsonl(questions_path, model=MathProblem))
    certificates = [json.loads(line) for line in certificates_path.read_text().splitlines()]
    regenerated, regenerated_certificates = generate_candidate_bank_v2(
        root_seed=manifest["root_seed"], per_cell=manifest["per_cell"]
    )
    assert [row.model_dump(mode="json") for row in questions] == [
        row.model_dump(mode="json") for row in regenerated
    ]
    assert json.loads(json.dumps(certificates)) == json.loads(
        json.dumps(regenerated_certificates)
    )
    assert len(questions) == 80
    assert manifest["questions_sha256"] == sha256_file(questions_path)
    assert manifest["certificates_sha256"] == sha256_file(certificates_path)
    recorded_manifest_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert recorded_manifest_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["generator_code_sha256"] == sha256_file(
        "src/datasets/procedural_math_v2.py"
    )
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_generate_procedural_math_v2.py"
    )
