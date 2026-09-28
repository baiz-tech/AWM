#!/usr/bin/env python3
"""Predict one Physion++ Future clip and classify contact over all Future."""

from src.data.physionpp.evaluate.eval_single_future_ocp import main


if __name__ == "__main__":
    main(
        task_name="single_future_all_future_ocp",
        default_label_scope="all_future",
    )
