import json
import subprocess
import sys
from pathlib import Path

import yaml


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
