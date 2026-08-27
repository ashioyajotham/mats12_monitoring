import hashlib
import json

import pytest

from src.audit import canonical_config_hash, load_config, runtime_environment, sha256_file
from src.generate_rollouts import write_manifest


def test_load_config_requires_mapping_root(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)


def test_load_config_reports_missing_sections(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("project: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required sections"):
        load_config(path)


def test_hash_helpers_are_stable(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("evidence\n", encoding="utf-8")
    assert sha256_file(path) == hashlib.sha256(b"evidence\n").hexdigest()
    assert canonical_config_hash({"b": 2, "a": 1}) == canonical_config_hash({"a": 1, "b": 2})


def test_manifest_is_content_addressed_and_immutable(tmp_path):
    path = tmp_path / "manifest.json"
    payload = {"purpose": "pipeline_test_only", "counts": {"completed": 3}}
    write_manifest(path, payload)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert manifest["manifest_sha256"] == hashlib.sha256(canonical).hexdigest()
    with pytest.raises(FileExistsError):
        write_manifest(path, payload)


def test_runtime_environment_records_python_and_platform():
    assert set(runtime_environment()) == {"python", "platform"}
