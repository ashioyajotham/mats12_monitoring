"""Contracts for publication figures and their committed renderings."""

from __future__ import annotations

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _module():
    """Load the numbered figure entrypoint as a module."""
    path = Path("experiments/07_make_paper_figures.py")
    spec = importlib.util.spec_from_file_location("paper_figures", path)
    if spec is None or spec.loader is None:
        raise AssertionError("paper figure module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_figures_are_deterministic_valid_svgs(tmp_path: Path) -> None:
    module = _module()
    v1 = json.loads(Path("results/causal_error_v1.monitor_metrics.json").read_text())
    v2 = json.loads(Path("results/causal_audit_v2_qualification.json").read_text())
    v21_gate = json.loads(Path("results/causal_audit_v21_confirmatory.json").read_text())
    v21 = json.loads(Path("results/causal_audit_v21.monitor_metrics.json").read_text())
    generated = {
        "paper_study_flow.svg": lambda path: module._study_flow(path),
        "paper_monitor_performance.svg": lambda path: module._performance_figure(v1, v21, path),
        "paper_causal_cells.svg": lambda path: module._cell_figure(v2, v21_gate, path),
    }
    for name, render in generated.items():
        output = tmp_path / name
        render(output)
        ET.parse(output)
        committed = Path("results/figures") / name
        assert output.read_bytes() == committed.read_bytes()


def test_paper_references_every_committed_figure() -> None:
    paper = Path("docs/PAPER_DRAFT.md").read_text(encoding="utf-8")
    for figure in Path("results/figures").glob("paper_*.svg"):
        assert f"../results/figures/{figure.name}" in paper
