#!/usr/bin/env python3
"""Render the paper figure ``fig:entity-intervention`` (entity_intervention.pdf).

Input is the OCP intervention metrics JSON written by
``src.experiments.vjepa2_physionpp_dynamics_decoder_seed239.scripts.physionpp.evaluate_ocp_interventions``
(protocol ``physionpp3_ocp_intervention_v1``).  That JSON stores one entry per
intervention condition under ``metrics[mode]`` with the binary classification
scores ``accuracy``, ``balanced_accuracy`` and ``auroc``.

Nothing is imputed: a missing file, a missing condition container or a missing
/ non-numeric score raises immediately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Conditions evaluated by evaluate_ocp_interventions.MODES, in figure order.
MODES = ("visual_only", "visual_probe", "target_mask", "irrelevant_mask", "random_slot")
# The paper caption calls the unmasked baseline "Full"; keep the JSON key as the
# identity and only translate the tick label.
DISPLAY_LABELS = {
    "visual_only": "Visual only",
    "visual_probe": "Full (visual+probe)",
    "target_mask": "Target mask",
    "irrelevant_mask": "Irrelevant mask",
    "random_slot": "Random slot",
}
SCORE_KEYS = ("accuracy", "balanced_accuracy", "auroc")
SCORE_LABELS = ("Accuracy", "Balanced accuracy", "AUROC")
DEFAULT_OUTPUT_NAME = "entity_intervention.pdf"


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


def load_condition_scores(metrics_path: str | Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Read ``metrics[mode][score]`` for every requested condition present."""
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"entity-intervention metrics JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "metrics" not in payload:
        raise KeyError(f"metrics JSON has no top-level 'metrics' object: {path}")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError(f"top-level 'metrics' must be an object, got {type(metrics).__name__}: {path}")
    conditions = [mode for mode in MODES if mode in metrics]
    if not conditions:
        raise KeyError(f"metrics JSON contains none of the expected conditions {MODES}: {path}")
    scores: dict[str, dict[str, float]] = {}
    for mode in conditions:
        entry = metrics[mode]
        if not isinstance(entry, dict):
            raise TypeError(f"metrics['{mode}'] must be an object, got {type(entry).__name__}: {path}")
        row: dict[str, float] = {}
        for key in SCORE_KEYS:
            if key not in entry:
                raise KeyError(f"metrics['{mode}'] is missing required key '{key}': {path}")
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"metrics['{mode}']['{key}'] must be numeric, got {value!r}: {path}")
            row[key] = float(value)
        scores[mode] = row
    return conditions, scores


def render(metrics_path: str | Path, output_path: str | Path, title: str | None = None) -> Path:
    """Write the grouped bar chart to ``output_path`` and return the PDF path."""
    # Validate the inputs before importing the plotting backend so that schema
    # errors are reported even where matplotlib is unavailable.
    conditions, scores = load_condition_scores(metrics_path)
    plt = _load_pyplot()

    figure, axes = plt.subplots(figsize=(7.0, 3.0), dpi=200)
    positions = list(range(len(conditions)))
    width = 0.8 / len(SCORE_KEYS)
    for index, (key, label) in enumerate(zip(SCORE_KEYS, SCORE_LABELS)):
        offsets = [position - 0.4 + width * (index + 0.5) for position in positions]
        heights = [scores[mode][key] for mode in conditions]
        bars = axes.bar(offsets, heights, width=width, label=label)
        for bar, height in zip(bars, heights):
            axes.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    axes.set_xticks(positions)
    axes.set_xticklabels([DISPLAY_LABELS.get(mode, mode) for mode in conditions])
    axes.set_ylabel("Score")
    axes.set_ylim(0.0, 1.0)
    axes.grid(axis="y", linewidth=0.4, alpha=0.4)
    axes.set_axisbelow(True)
    axes.legend(loc="lower right", fontsize=7, frameon=False, ncols=len(SCORE_KEYS))
    if title:
        axes.set_title(title)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, required=True, help="evaluate_ocp_interventions JSON")
    parser.add_argument("--output", type=Path, required=True, help="destination PDF")
    parser.add_argument("--title", default=None, help="optional figure title")
    args = parser.parse_args()
    output = render(args.metrics, args.output, args.title)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
