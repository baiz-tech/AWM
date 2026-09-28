#!/usr/bin/env python3
"""Render the paper figure ``fig:relation-component`` (interaction_prediction.pdf).

Input is the ``metrics.json`` written by
``src.studies.clevrer_relation_controls.run`` (protocol
``clevrer_relation_controls_v1``).  That JSON stores one entry per input
combination under ``metrics[variant]``; each entry holds
``contact.auroc`` (higher is better) and ``ttc.mae`` (lower is better), which
are drawn here as grouped bars per combination.

Nothing is imputed: a missing file, a missing combination or a missing /
non-numeric score raises immediately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Mirrors run.py's ``VARIANTS`` mapping (names and their order):
#   VARIANTS = {"Entity": ("entity",), "Dynamic": ("dynamic",),
#               "Relation-only": ("relation",), "Entity+Dynamic": (...),
#               "Entity+Relation": (...), "Dynamic+Relation": (...),
#               "Full": FEATURES}
VARIANTS = (
    "Entity",
    "Dynamic",
    "Relation-only",
    "Entity+Dynamic",
    "Entity+Relation",
    "Dynamic+Relation",
    "Full",
)
# (metrics section, score key, legend label) in figure order.
SCORE_SPECS = (
    ("contact", "auroc", "Contact AUROC (higher better)"),
    ("ttc", "mae", "TTC MAE (lower better)"),
)
DEFAULT_OUTPUT_NAME = "interaction_prediction.pdf"


def _load_pyplot():
    """Import matplotlib lazily and force the headless Agg backend."""
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to render the paper figures but is not importable "
            "from the active interpreter. Install it with `python -m pip install matplotlib`."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_variant_scores(metrics_path: str | Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Read ``metrics[variant][section][score]`` for every known variant present."""
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"interaction-prediction metrics JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "metrics" not in payload:
        raise KeyError(f"metrics JSON has no top-level 'metrics' object: {path}")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError(f"top-level 'metrics' must be an object, got {type(metrics).__name__}: {path}")
    variants = [name for name in VARIANTS if name in metrics]
    if not variants:
        raise KeyError(f"metrics JSON contains none of the expected combinations {VARIANTS}: {path}")
    scores: dict[str, dict[str, float]] = {}
    for variant in variants:
        entry = metrics[variant]
        if not isinstance(entry, dict):
            raise TypeError(f"metrics['{variant}'] must be an object, got {type(entry).__name__}: {path}")
        row: dict[str, float] = {}
        for section, key, _ in SCORE_SPECS:
            if section not in entry:
                raise KeyError(f"metrics['{variant}'] is missing required section '{section}': {path}")
            block = entry[section]
            if not isinstance(block, dict):
                raise TypeError(f"metrics['{variant}']['{section}'] must be an object: {path}")
            if key not in block:
                raise KeyError(
                    f"metrics['{variant}']['{section}'] is missing required key '{key}': {path}"
                )
            value = block[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"metrics['{variant}']['{section}']['{key}'] must be numeric, got {value!r}: {path}"
                )
            row[f"{section}.{key}"] = float(value)
        scores[variant] = row
    return variants, scores


def render(metrics_path: str | Path, output_path: str | Path, title: str | None = None) -> Path:
    """Write the grouped bar chart to ``output_path`` and return the PDF path."""
    # Validate the inputs before importing the plotting backend so that schema
    # errors are reported even where matplotlib is unavailable.
    variants, scores = load_variant_scores(metrics_path)
    plt = _load_pyplot()

    figure, auroc_axes = plt.subplots(figsize=(4.8, 3.0), dpi=200)
    mae_axes = auroc_axes.twinx()
    positions = list(range(len(variants)))
    width = 0.36
    colors = ("#3b6ea5", "#c1553b")

    auroc_values = [scores[name]["contact.auroc"] for name in variants]
    auroc_bars = auroc_axes.bar(
        [position - width / 2.0 for position in positions],
        auroc_values,
        width=width,
        color=colors[0],
        label=SCORE_SPECS[0][2],
    )
    for bar, value in zip(auroc_bars, auroc_values):
        auroc_axes.text(
            bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6
        )

    mae_values = [scores[name]["ttc.mae"] for name in variants]
    mae_bars = mae_axes.bar(
        [position + width / 2.0 for position in positions],
        mae_values,
        width=width,
        color=colors[1],
        label=SCORE_SPECS[1][2],
    )
    for bar, value in zip(mae_bars, mae_values):
        mae_axes.text(
            bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6
        )

    auroc_axes.set_xticks(positions)
    auroc_axes.set_xticklabels(variants, rotation=30, ha="right")
    auroc_axes.set_ylabel(SCORE_SPECS[0][2], color=colors[0])
    auroc_axes.set_ylim(0.0, 1.0)
    auroc_axes.tick_params(axis="y", labelcolor=colors[0])
    mae_axes.set_ylabel(SCORE_SPECS[1][2], color=colors[1])
    mae_axes.set_ylim(0.0, max(mae_values) * 1.25 if max(mae_values) > 0 else 0.1)
    mae_axes.tick_params(axis="y", labelcolor=colors[1])
    auroc_axes.grid(axis="y", linewidth=0.4, alpha=0.4)
    auroc_axes.set_axisbelow(True)
    handles = [auroc_bars, mae_bars]
    auroc_axes.legend(handles, [spec[2] for spec in SCORE_SPECS], loc="upper center", fontsize=6, frameon=False)
    if title:
        auroc_axes.set_title(title)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, required=True, help="clevrer_relation_controls metrics.json")
    parser.add_argument("--output", type=Path, required=True, help="destination PDF")
    parser.add_argument("--title", default=None, help="optional figure title")
    args = parser.parse_args()
    output = render(args.metrics, args.output, args.title)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
