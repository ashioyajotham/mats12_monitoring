import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.audit import canonical_config_hash


def test_mock_pilot_writes_complete_manifest(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / "configs/pilot.yaml").read_text(encoding="utf-8"))
    config["paths"]["generated_dir"] = str(tmp_path / "generated")
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "experiments/01_pilot.py", "--config", str(config_path)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    output_dir = Path(summary["output_dir"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["rollouts"] == 300
    assert manifest["counts"] == {
        "completed": 300,
        "excluded": 0,
        "invalid": 0,
        "questions": 20,
        "requested": 300,
    }
    assert set(manifest["output_sha256"]) == {
        "answer_shifts",
        "candidate_labels",
        "rollouts",
    }


def test_glm_smoke_dry_run_requires_no_api_key():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/02_generate_dataset.py",
            "--config",
            "configs/glm_smoke.yaml",
            "--limit",
            "3",
            "--samples-per-condition",
            "1",
            "--dry-run",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary == {
        "purpose": "smoke_test_only",
        "questions": 3,
        "conditions": 3,
        "samples_per_condition": 1,
        "requests": 9,
        "model": "glm-4.7-flash",
        "provider_seed_supported": False,
    }


def test_tinker_smoke_dry_run_requires_no_api_key():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/02_generate_dataset.py",
            "--config",
            "configs/tinker_smoke.yaml",
            "--limit",
            "1",
            "--samples-per-condition",
            "1",
            "--dry-run",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "purpose": "smoke_test_only",
        "questions": 1,
        "conditions": 3,
        "samples_per_condition": 1,
        "requests": 3,
        "model": "Qwen/Qwen3.6-35B-A3B",
        "provider_seed_supported": True,
    }


def test_resume_loader_accepts_only_matching_config(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    module = runpy.run_path(repo / "experiments/02_generate_dataset.py")
    load_resume_rollouts = module["load_resume_rollouts"]
    config = yaml.safe_load((repo / "configs/glm_smoke.yaml").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    run.mkdir()
    (run / "rollouts.jsonl").write_text("", encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps({"config_sha256": canonical_config_hash(config)}), encoding="utf-8"
    )

    assert load_resume_rollouts(run, config_sha256=canonical_config_hash(config)) == []
    with pytest.raises(ValueError, match="different configuration hash"):
        load_resume_rollouts(run, config_sha256="different")
