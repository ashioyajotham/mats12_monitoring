"""Audit helpers for configuration, files, runs, and claims."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

REQUIRED_CONFIG_SECTIONS = {
    "project",
    "data",
    "conditions",
    "generation",
    "labels",
    "gates",
    "evaluation",
    "paths",
}


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration and validate its required top-level sections.

    Args:
        path: Configuration file to read.

    Returns:
        Parsed configuration mapping.

    Raises:
        ValueError: If the YAML root is not a mapping or required sections are missing.
    """
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    missing = REQUIRED_CONFIG_SECTIONS - set(config)
    if missing:
        raise ValueError(f"config missing required sections: {sorted(missing)}")
    return config


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    """Return the current Git commit, or ``None`` outside a committed repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def canonical_config_hash(config: dict) -> str:
    """Hash a configuration using a stable JSON representation."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def runtime_environment() -> dict[str, str]:
    """Describe the Python and operating-system environment for a run manifest."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()
