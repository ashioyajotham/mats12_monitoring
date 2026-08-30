import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.audit import sha256_file
from src.datasets.procedural_math_v21 import (
    TIERS_V21,
    generate_subset_replacement_bank,
    verify_subset_replacement,
)
from src.tasks import MathProblem, read_jsonl


def test_v21_replacements_are_deterministic_balanced_and_verified() -> None:
    first, first_certificates = generate_subset_replacement_bank(root_seed=201, per_tier=2)
    second, second_certificates = generate_subset_replacement_bank(root_seed=201, per_tier=2)
    assert first == second
    assert first_certificates == second_certificates
    assert Counter(row.difficulty for row in first) == Counter({tier: 2 for tier in TIERS_V21})
    assert all(
        verify_subset_replacement(problem, certificate)
        for problem, certificate in zip(first, first_certificates, strict=True)
    )
    different, _ = generate_subset_replacement_bank(root_seed=202, per_tier=2)
    assert [row.question_id for row in first] != [row.question_id for row in different]


def test_v21_replacement_certificate_mutation_is_rejected() -> None:
    problems, certificates = generate_subset_replacement_bank(root_seed=203, per_tier=1)
    changed = copy.deepcopy(certificates[0])
    changed["oracle_answer"] = str(int(str(changed["oracle_answer"])) + 1)
    assert not verify_subset_replacement(problems[0], changed)


def test_committed_v21_replacement_freeze_matches_generator() -> None:
    questions_path = Path("data/raw/procedural_math_subset_replacements_v21.jsonl")
    certificates_path = Path(
        "data/raw/procedural_math_subset_replacements_v21.certificates.jsonl"
    )
    manifest = json.loads(
        Path("data/raw/procedural_math_subset_replacements_v21.manifest.json").read_text()
    )
    questions = list(read_jsonl(questions_path, model=MathProblem))
    certificates = [json.loads(line) for line in certificates_path.read_text().splitlines()]
    regenerated, regenerated_certificates = generate_subset_replacement_bank(
        root_seed=manifest["root_seed"], per_tier=manifest["per_tier"]
    )
    assert [row.model_dump(mode="json") for row in questions] == [
        row.model_dump(mode="json") for row in regenerated
    ]
    assert certificates == regenerated_certificates
    assert len(questions) == 20
    assert manifest["questions_sha256"] == sha256_file(questions_path)
    assert manifest["certificates_sha256"] == sha256_file(certificates_path)
    recorded_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert recorded_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["generator_code_sha256"] == sha256_file(
        "src/datasets/procedural_math_v21.py"
    )
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_generate_procedural_math_v21.py"
    )
