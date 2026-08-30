import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.audit import sha256_file
from src.datasets.procedural_math import (
    FAMILIES,
    TIERS,
    generate_candidate_bank,
    verify_problem,
)
from src.tasks import MathProblem, read_jsonl


def test_candidate_bank_is_deterministic_balanced_and_verified():
    first_problems, first_certificates = generate_candidate_bank(root_seed=91, per_cell=2)
    second_problems, second_certificates = generate_candidate_bank(root_seed=91, per_cell=2)

    assert [row.model_dump_json() for row in first_problems] == [
        row.model_dump_json() for row in second_problems
    ]
    assert first_certificates == second_certificates
    assert len(first_problems) == len(FAMILIES) * len(TIERS) * 2
    assert len({row.question_id for row in first_problems}) == len(first_problems)
    assert len({row.prompt for row in first_problems}) == len(first_problems)
    counts = Counter(
        (row.metadata["generator_family"], row.metadata["difficulty_tier"])
        for row in first_problems
    )
    assert set(counts.values()) == {2}
    assert all(
        verify_problem(problem, certificate)
        for problem, certificate in zip(first_problems, first_certificates, strict=True)
    )


def test_different_root_seed_changes_generated_bank():
    first, _ = generate_candidate_bank(root_seed=1, per_cell=1)
    second, _ = generate_candidate_bank(root_seed=2, per_cell=1)
    assert [row.question_id for row in first] != [row.question_id for row in second]
    assert [row.prompt for row in first] != [row.prompt for row in second]


def test_certificate_rejects_mutated_answer_prompt_and_parameters():
    problems, certificates = generate_candidate_bank(root_seed=3, per_cell=1)
    problem = problems[0]
    certificate = certificates[0]

    wrong_answer = problem.model_copy(update={"gold_answer": str(int(problem.gold_answer) + 1)})
    wrong_prompt = problem.model_copy(update={"prompt": problem.prompt + " Changed."})
    wrong_certificate = copy.deepcopy(certificate)
    wrong_certificate["parameters"]["residues"][0] += 1

    assert not verify_problem(wrong_answer, certificate)
    assert not verify_problem(wrong_prompt, certificate)
    assert not verify_problem(problem, wrong_certificate)


def test_generation_rejects_nonpositive_cell_size():
    try:
        generate_candidate_bank(per_cell=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected a nonpositive cell size to fail")


def test_committed_candidate_freeze_matches_generator_and_manifest():
    questions_path = Path("data/raw/procedural_math_candidates_v1.jsonl")
    certificates_path = Path("data/raw/procedural_math_certificates_v1.jsonl")
    manifest_path = Path("data/raw/procedural_math_candidates_v1.manifest.json")
    committed_questions = list(read_jsonl(questions_path, model=MathProblem))
    committed_certificates = [
        json.loads(line)
        for line in certificates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated_questions, generated_certificates = generate_candidate_bank()

    assert [row.model_dump(mode="json") for row in committed_questions] == [
        row.model_dump(mode="json") for row in generated_questions
    ]
    assert json.loads(json.dumps(committed_certificates)) == json.loads(
        json.dumps(generated_certificates)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert expected_manifest_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["questions_sha256"] == sha256_file(questions_path)
    assert manifest["certificates_sha256"] == sha256_file(certificates_path)
    assert manifest["generator_code_sha256"] == sha256_file(
        "src/datasets/procedural_math.py"
    )
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_generate_procedural_math.py"
    )


def test_committed_low_reasoning_diagnostic_is_bound_and_monitor_excluded():
    questions_path = Path("data/raw/procedural_low_reasoning_diagnostic_v1.jsonl")
    certificates_path = Path(
        "data/raw/procedural_low_reasoning_diagnostic_v1.certificates.jsonl"
    )
    manifest_path = Path("data/raw/procedural_low_reasoning_diagnostic_v1.manifest.json")
    questions = list(read_jsonl(questions_path, model=MathProblem))
    certificates = [
        json.loads(line)
        for line in certificates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(questions) == len(certificates) == 24
    assert all(question.metadata["excluded_from_monitor_data"] for question in questions)
    assert Counter(
        question.metadata["diagnostic_stratum"] for question in questions
    ) == Counter({"previously_truncated": 12, "matched_clean_control": 12})
    assert all(
        verify_problem(question, certificate)
        for question, certificate in zip(questions, certificates, strict=True)
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert expected_manifest_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["questions_sha256"] == sha256_file(questions_path)
    assert manifest["certificates_sha256"] == sha256_file(certificates_path)
    assert manifest["selection_code_sha256"] == sha256_file("src/procedural_pilot.py")
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/02_prepare_low_reasoning_diagnostic.py"
    )
