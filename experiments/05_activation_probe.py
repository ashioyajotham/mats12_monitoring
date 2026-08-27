"""Activation-probe experiment gate and explicit scope boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Enforce pilot gates before architecture-specific probe work begins."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.gate_report).read_text(encoding="utf-8"))
    required = ("phenomenon", "validity", "diversity", "feasibility")
    failed = [gate for gate in required if not report.get(gate, {}).get("passed", False)]
    if failed:
        raise SystemExit(f"Activation probe is out of scope until all pilot gates pass: {failed}")
    raise SystemExit(
        "Gates passed. Freeze layers, token position, grouped split, and controls in the "
        "preregistration before adding model-specific extraction code."
    )


if __name__ == "__main__":
    main()
