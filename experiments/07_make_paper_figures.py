"""Generate dependency-free SVG figures from committed result artifacts."""

# ruff: noqa: E501 -- SVG element strings are intentionally kept as single literals.

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _text(
    x: float, y: float, value: str, *, size: int = 14, anchor: str = "start", weight: int = 400
) -> str:
    """Render one escaped SVG text element."""
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#172033">'
        f"{html.escape(value)}</text>"
    )


def _write(path: Path, width: int, height: int, elements: list[str]) -> None:
    """Write one deterministic standalone SVG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n  ".join(elements)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <rect width="100%" height="100%" fill="#ffffff"/>\n  {body}\n</svg>\n',
        encoding="utf-8",
    )


def _performance_figure(v1: dict[str, object], v21: dict[str, object], path: Path) -> None:
    """Compare answer-shift and surface discrimination across both studies."""
    width, height = 980, 570
    left, right, top, bottom = 105, 35, 80, 100
    plot_width, plot_height = width - left - right, height - top - bottom
    elements = [
        _text(
            width / 2,
            35,
            "Monitor performance across frozen evaluations",
            size=22,
            anchor="middle",
            weight=700,
        )
    ]
    elements.append(
        _text(
            width / 2,
            59,
            "Bars are point estimates; whiskers are question-clustered 95% intervals",
            size=13,
            anchor="middle",
        )
    )
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + plot_height * (1 - value)
        elements.append(
            f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="#d9dfeb" stroke-width="1"/>'
        )
        elements.append(_text(left - 12, y + 5, f"{value:.1f}", anchor="end"))
    groups = (
        ("v1 AUROC", "auroc"),
        ("v1 AUPRC", "auprc"),
        ("v2.1 AUROC", "auroc"),
        ("v2.1 AUPRC", "auprc"),
    )
    colors = {"answer": "#d1495b", "surface": "#4f78a8"}
    group_width = plot_width / len(groups)
    bar_width = 55
    for group_index, (label, metric) in enumerate(groups):
        source = v1 if label.startswith("v1 ") else v21
        center = left + group_width * (group_index + 0.5)
        for offset, (monitor, key) in zip(
            (-36, 36),
            (("answer", "counterfactual_answer_shift"), ("surface", "surface")),
            strict=True,
        ):
            if label.startswith("v1 "):
                report = source["primary_test_metrics"][key]
                point = float(report["point"][metric])
                interval = report["cluster_bootstrap"][metric]
            else:
                point = float(source["point_metrics"][key][metric])
                interval = source["question_clustered_intervals_95"][key][metric]
            x = center + offset - bar_width / 2
            y = top + plot_height * (1 - point)
            elements.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{top + plot_height - y}" rx="3" fill="{colors[monitor]}"/>'
            )
            low_y = top + plot_height * (1 - float(interval["low"]))
            high_y = top + plot_height * (1 - float(interval["high"]))
            whisker_x = center + offset
            elements.extend(
                [
                    f'<line x1="{whisker_x}" y1="{high_y}" x2="{whisker_x}" y2="{low_y}" stroke="#172033" stroke-width="2"/>',
                    f'<line x1="{whisker_x - 8}" y1="{high_y}" x2="{whisker_x + 8}" y2="{high_y}" stroke="#172033" stroke-width="2"/>',
                    f'<line x1="{whisker_x - 8}" y1="{low_y}" x2="{whisker_x + 8}" y2="{low_y}" stroke="#172033" stroke-width="2"/>',
                    _text(whisker_x, y - 9, f"{point:.3f}", size=12, anchor="middle", weight=700),
                ]
            )
        elements.append(_text(center, top + plot_height + 29, label, size=13, anchor="middle"))
    elements.extend(
        [
            f'<rect x="{width / 2 - 130}" y="{height - 42}" width="16" height="16" fill="{colors["answer"]}"/>',
            _text(width / 2 - 106, height - 29, "Answer shift", size=13),
            f'<rect x="{width / 2 + 20}" y="{height - 42}" width="16" height="16" fill="{colors["surface"]}"/>',
            _text(width / 2 + 44, height - 29, "Surface", size=13),
            _text(25, top + plot_height / 2, "Score", size=15, anchor="middle", weight=700),
        ]
    )
    _write(path, width, height, elements)


def _cell_figure(v2: dict[str, object], v21: dict[str, object], path: Path) -> None:
    """Show qualification failure and supported-domain replication by causal cell."""
    width, height = 1040, 600
    elements = [
        _text(
            width / 2,
            35,
            "Exact-target uptake under corrupted continuation",
            size=22,
            anchor="middle",
            weight=700,
        )
    ]
    elements.append(
        _text(
            width / 2,
            59,
            "V2 qualification exposed the subset boundary; v2.1 used fresh supported-family questions",
            size=13,
            anchor="middle",
        )
    )
    families = ("affine_modular", "conditional_dag", "finite_state", "subset_counting")
    labels = {
        "affine_modular": "Affine",
        "conditional_dag": "DAG",
        "finite_state": "Finite state",
        "subset_counting": "Subset",
    }
    mechanisms = ("drop_component", "duplicate_component")
    cell_w, cell_h = 105, 62
    start_x, start_y = 270, 115
    for column, family in enumerate(families):
        elements.append(
            _text(
                start_x + column * cell_w + cell_w / 2,
                96,
                labels[family],
                size=13,
                anchor="middle",
                weight=700,
            )
        )
    panels = (("v2 qualification", v2, 0), ("v2.1 external", v21, 3))
    for title, report, row_offset in panels:
        panel_y = start_y + row_offset * cell_h
        elements.append(_text(18, panel_y + cell_h, title, size=15, weight=700))
        for mechanism_index, mechanism in enumerate(mechanisms):
            y = panel_y + mechanism_index * cell_h
            elements.append(
                _text(
                    start_x - 15,
                    y + 38,
                    "Drop" if mechanism_index == 0 else "Duplicate",
                    size=13,
                    anchor="end",
                )
            )
            for column, family in enumerate(families):
                key = f"{family}:{mechanism}"
                cell = report["family_mechanism_cells"].get(key)
                x = start_x + column * cell_w
                if cell is None:
                    fill, value, foreground = "#eceff4", "excluded", "#687386"
                else:
                    condition = cell["conditions"]["corrupted_continuation"]
                    rate = float(condition["target_rate"])
                    red = int(244 - 100 * rate)
                    green = int(248 - 45 * rate)
                    blue = int(252 - 155 * rate)
                    fill = f"rgb({red},{green},{blue})"
                    value = f"{condition['target_selections']}/{condition['scorable']}"
                    foreground = "#172033"
                elements.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" rx="5" fill="{fill}" stroke="#c5ccda"/>'
                )
                elements.append(
                    f'<text x="{x + (cell_w - 8) / 2}" y="{y + 33}" font-family="Arial, sans-serif" font-size="16" font-weight="700" text-anchor="middle" fill="{foreground}">{value}</text>'
                )
        if row_offset == 0:
            elements.append(
                f'<line x1="{start_x}" y1="{panel_y + 2 * cell_h + 22}" x2="{start_x + 4 * cell_w - 8}" y2="{panel_y + 2 * cell_h + 22}" stroke="#9da8ba" stroke-width="1"/>'
            )
    elements.append(
        _text(
            width / 2,
            height - 35,
            "Cell values are exact target selections / scorable corrupted-condition rollouts",
            size=13,
            anchor="middle",
        )
    )
    _write(path, width, height, elements)


def _study_flow(path: Path) -> None:
    """Render the study's decision sequence and stopping-rule outcomes."""
    width, height = 1180, 300
    steps = (
        ("Silent-use attempts", "Weak, acknowledged, or null", "#a7b0bf"),
        ("Mixed-outcome tasks", "Ordinary failures established", "#4f78a8"),
        ("Causal-error v1", "+1 intervention passed", "#d89032"),
        ("Monitor evaluation", "Answer shift > surface", "#d1495b"),
        ("Four-family v2", "Stopped: subset failed", "#8b5e83"),
        ("Three-family v2.1", "External transfer passed", "#2b8a6e"),
    )
    elements = [
        _text(
            width / 2,
            34,
            "Research progression under preregistered stopping rules",
            size=22,
            anchor="middle",
            weight=700,
        )
    ]
    box_w, box_h, gap = 165, 118, 23
    start_x = (width - (len(steps) * box_w + (len(steps) - 1) * gap)) / 2
    y = 86
    for index, (title, subtitle, color) in enumerate(steps):
        x = start_x + index * (box_w + gap)
        elements.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="9" fill="#ffffff" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(_text(x + box_w / 2, y + 35, title, size=14, anchor="middle", weight=700))
        words = subtitle.split()
        split = max(1, len(words) // 2)
        lines = (" ".join(words[:split]), " ".join(words[split:]))
        elements.append(_text(x + box_w / 2, y + 69, lines[0], size=12, anchor="middle"))
        elements.append(_text(x + box_w / 2, y + 88, lines[1], size=12, anchor="middle"))
        if index < len(steps) - 1:
            x1, x2, mid_y = x + box_w + 4, x + box_w + gap - 5, y + box_h / 2
            elements.append(
                f'<line x1="{x1}" y1="{mid_y}" x2="{x2}" y2="{mid_y}" stroke="#687386" stroke-width="2"/>'
            )
            elements.append(
                f'<polygon points="{x2},{mid_y} {x2 - 8},{mid_y - 5} {x2 - 8},{mid_y + 5}" fill="#687386"/>'
            )
    elements.append(
        _text(
            width / 2,
            254,
            "Failures constrain the claim; they are not discarded pilot noise.",
            size=14,
            anchor="middle",
            weight=700,
        )
    )
    _write(path, width, height, elements)


def main() -> None:
    """Read committed results and write all paper SVGs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    v1 = json.loads(Path("results/causal_error_v1.monitor_metrics.json").read_text())
    v2 = json.loads(Path("results/causal_audit_v2_qualification.json").read_text())
    v21_gate = json.loads(Path("results/causal_audit_v21_confirmatory.json").read_text())
    v21 = json.loads(Path("results/causal_audit_v21.monitor_metrics.json").read_text())
    _study_flow(args.output_dir / "paper_study_flow.svg")
    _performance_figure(v1, v21, args.output_dir / "paper_monitor_performance.svg")
    _cell_figure(v2, v21_gate, args.output_dir / "paper_causal_cells.svg")
    print(json.dumps({"figures": 3, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
