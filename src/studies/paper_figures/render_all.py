#!/usr/bin/env python3
"""Render every data-driven AWM/HASP paper figure in one call.

Thin CLI wrapper around ``render_entity_intervention`` and
``render_interaction_prediction``.  Both upstream metric files must already
exist; nothing is imputed and no figure is skipped silently.

Usage (from the ``makeup`` root):

    python -m src.studies.paper_figures.render_all \
        --entity-intervention-metrics /path/to/ocp_interventions.json \
        --interaction-metrics /path/to/clevrer_relation_controls/metrics.json \
        --output-dir /path/to/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import render_entity_intervention, render_interaction_prediction
from src.core.run_context import apply_cli_defaults, task_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--entity-intervention-metrics",
        type=Path,
        required=True,
        help="evaluate_ocp_interventions JSON for fig:entity-intervention",
    )
    parser.add_argument(
        "--interaction-metrics",
        type=Path,
        required=True,
        help="clevrer_relation_controls metrics.json for fig:relation-component",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for the rendered PDFs")
    apply_cli_defaults(parser, task_context())
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entity_pdf = render_entity_intervention.render(
        args.entity_intervention_metrics, args.output_dir / render_entity_intervention.DEFAULT_OUTPUT_NAME
    )
    interaction_pdf = render_interaction_prediction.render(
        args.interaction_metrics, args.output_dir / render_interaction_prediction.DEFAULT_OUTPUT_NAME
    )
    print(f"wrote {entity_pdf}")
    print(f"wrote {interaction_pdf}")


if __name__ == "__main__":
    main()
